# Deploying LS-YOLO (Decoupled_Detect head) onto KV260 — Plan A

> This script suite **replaces** the quantize/demo section in the tutorial PDF. The PDF was written
> for the standard `Detect` head (`detect.m[i]`); your actual model uses
> **`Decoupled_Detect`** (m_stem + m_cls + **CAM dilation 1/3/5** + m_reg + m_conf),
> so running the PDF script will be incorrect/crash.

## Already modified in repo (mandatory, done once — BEFORE training)
`models/common.py`:
- `Conv.default_act`: `nn.SiLU()` -> `nn.Hardswish()` (DPU lacks SiLU)
- `BottleneckCSP.act`: `nn.SiLU()` -> `nn.Hardswish()`
- `class ECA`: `Conv1d`+transpose+Sigmoid -> SE-block (`Conv2d` 1×1 + `Hardsigmoid`),
  kept name + `k_size` so `MSFE` doesn't need to change.

Upsample is already `nearest` -> no need to change.

## Files in this directory
| File | Run Where | Task |
|---|---|---|
| `ls_yolo_dpu.py` | (core, import) | `DecoupledDPU` wrapper + `decode_decoupled` (CPU) |
| `verify_decode.py` | laptop | Proves decode == original forward (PASS, diff=0) |
| `dryrun_export.py` | Docker Vitis-AI | **De-risk**: export xmodel from random weights to check subgraphs |
| `quantize_calib.py` | Docker Vitis-AI | Calib INT8 (needs best.pt + calib_images) |
| `quantize_export.py` | Docker Vitis-AI | Export `DecoupledDPU_int.xmodel` + `head_constants.json` |
| `demo_dpu_live.py` | KV260 | DPU Inference + decode + NMS + display |

## Workflow

### 0. Verify decode (laptop, already run)
```bash
python kv260_export/verify_decode.py     # PASS = decode logic is correct
```

### 1. DRY-RUN before training (Docker Vitis-AI) — most crucial step of Plan A
Purpose: know if the architecture maps entirely to the DPU **before** spending 4–8h training.
```bash
# inside container, activated vitis-ai-pytorch, copied LS-YOLO -> /workspace/LS-YOLO
cd /workspace
LSYOLO_SRC=/workspace/LS-YOLO python LS-YOLO/kv260_export/dryrun_export.py

ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
vai_c_xir -x /workspace/compiled_dryrun/DecoupledDPU_int.xmodel -a $ARCH \
          -o /workspace/compiled_dryrun -n dryrun
```
**READ LOG:**
- `1 DPU subgraph` -> maps entirely (OK) -> proceed to step 2 with peace of mind.
- Multiple subgraphs -> check which op drops. Suspect #1 is **CAM dilation 3/5**. Fix:
  in `models/common.py` class `CAM`, change `dilation` of `conv2`/`conv3` from
  `3`/`5` -> `2`/`4` (values DPU definitely supports), keep the decoupled head, then
  re-run dry-run. (Will require training with the new config.)

### 2. Train (laptop GPU) — use domain-gap robust hyp
```bash
python train.py --cfg models/landslide/Improve.yaml --data data/landslide.yaml \
  --hyp data/hyps/hyp.scratch-landslide-dpu.yaml --weights "" \
  --epochs 300 --batch-size 8 --imgsz 512 --optimizer SGD --device 0 --name ls_yolo_dpu_A
```

### 3. Real Quantize + compile (Docker Vitis-AI)
```bash
cd /workspace            # has best.pt, calib_images/, LS-YOLO/
LSYOLO_SRC=/workspace/LS-YOLO python LS-YOLO/kv260_export/quantize_calib.py
LSYOLO_SRC=/workspace/LS-YOLO python LS-YOLO/kv260_export/quantize_export.py

ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
vai_c_xir -x /workspace/compiled/DecoupledDPU_int.xmodel -a $ARCH \
          -o /workspace/compiled -n ls_yolo_landslide
# -> /workspace/compiled/ls_yolo_landslide.xmodel  (+ head_constants.json)
```

### 4. Run on KV260
Copy to board: `ls_yolo_landslide.xmodel`, `head_constants.json`, the entire
`LS-YOLO/` directory (contains `kv260_export/` + `utils/`).
```bash
# inside smartcam container on KV260:
cd ~/project/LS-YOLO/kv260_export
MODEL=~/project/LS-YOLO/ls_yolo_landslide.xmodel SOURCE=0 python3 demo_dpu_live.py
# Test matching domain (recommended): point to aerial image instead of room webcam
MODEL=.../ls_yolo_landslide.xmodel SOURCE=/path/to/aerial.jpg python3 demo_dpu_live.py
```

## Accuracy Notes (read carefully)
- **Domain gap**: model trains on top-down images (UAV/satellite). Ground-level webcam -> bounding boxes will be
  meaningless. Use `SOURCE=<aerial image/video>` to demo correct domain. `hyp.scratch-landslide-dpu.yaml`
  only increases robustness *within* the top-down domain, it cannot compensate for major differences.
- **INT8 Quantize**: usually drops 1–5% mAP, small/thin landslides are easily missed. Compare mAP
  of float version (val.py) vs feel on board to gauge the drop.
