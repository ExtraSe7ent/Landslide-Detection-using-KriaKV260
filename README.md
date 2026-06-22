# LS-YOLO on Kria KV260 — Real-time Landslide Detection on FPGA

Landslide detection from aerial/UAV imagery using the **LS-YOLO** architecture
(MSFE + improved Decoupled Head), trained on a UAV landslide dataset and deployed
to a **Xilinx Kria KV260** FPGA with INT8 quantization for real-time inference on
the edge.

The detection backbone follows the LS-YOLO paper (Zhang et al., IEEE JSTARS 2024,
built on Ultralytics YOLOv5). This repository adds:

- A full training pipeline for a single-class (`landslide`) UAV dataset;
- DPU-friendly architecture changes so the model maps onto the Vitis-AI DPU;
- An end-to-end **KV260 deployment** flow (quantization → compile → on-board demo/eval).

## Results

Best model (UAV-only, image size 512), evaluated on the held-out test set:

| Metric       | Value |
| ------------ | ----- |
| mAP@0.5      | 0.927 |
| mAP@0.5:0.95 | 0.645 |
| Precision    | 0.933 |
| Recall       | 0.860 |

After INT8 quantization the model runs on the KV260 DPU (DPUCZDX8G B4096). On-board
precision/recall/mAP and latency are reported by `eval_display_kv260.py`.

## Architecture

- **Backbone/Neck:** YOLOv5s (v6.0).
- **MSFE** (Multi-Scale Feature Extraction) module at the top of the neck.
- **Decoupled_Detect** head: separate stem / classification / regression / objectness
  branches with a context-aggregation module (CAM, dilated convolutions).

Model definition: `Training/models/landslide/Improve.yaml`. Custom modules live in
`Training/models/common.py` and `Training/models/yolo.py`.

For FPGA compatibility, activations were switched from SiLU to Hardswish and the
ECA attention (depthwise `Conv1d`) was replaced by an SE-style block (`Conv2d` 1×1 +
Hardsigmoid). These ops are supported by the DPU; the model is retrained from scratch
after the change.

## Repository layout

```
Training/              Training pipeline and evaluation on PC
  ├── models/          YOLOv5 model code + custom modules
  ├── data/            Dataset configs and hyperparameters
  ├── train.py         Training entry point
  ├── val.py           Validation entry point
  ├── evaluate.py      Evaluation + visualization helpers
  └── convert_labels.py / data_split.py
Quantization/          Scripts for INT8 quantization (Vitis-AI DPU)
  ├── prepare_calib.py Calibration data prep
  ├── quantize_calib.py
  └── quantize_export.py
KV260_Deployment/      Deployment scripts for Kria KV260
  ├── demo_dpu_live.py Live inference via camera
  └── eval_display_kv260.py P/R/mAP eval + realtime display
```

## Setup

```bash
conda create -n ls-yolo python=3.9
conda activate ls-yolo
pip install -r requirements.txt
```

## Dataset

All paths are relative to a `datasets/` directory at the repository root:

```
datasets/
├── images/              landslide events, each with an img/ (and mask/) subfolder
├── labels_yolo_clean/   YOLO .txt labels (per event)
└── splits/              train.txt / val.txt / test.txt (lists of image paths)
```

Point this at your data by either creating `datasets/` (or a symlink/junction to it),
or overriding the location with environment variables — `LS_IMAGES`, `LS_SPLITS`,
`LS_LABELS_OUT` — used by the data-prep scripts.

```bash
cd Training
python convert_labels.py     # segmentation masks -> YOLO .txt labels
python data_split.py         # write train/val/test split lists
```

## Training

```bash
cd Training
python train.py \
  --cfg models/landslide/Improve.yaml \
  --data data/landslide.yaml \
  --hyp data/hyps/hyp.scratch-landslide-dpu.yaml \
  --weights yolov5s.pt \
  --epochs 100 --batch-size 16 --imgsz 512 \
  --optimizer SGD --device 0 --workers 2
```

## Evaluation

```bash
cd Training
python val.py --weights runs/train/<run>/weights/best.pt \
  --data data/landslide.yaml --imgsz 512 --task test

python evaluate.py           # metrics + side-by-side prediction vs ground-truth
```

## KV260 deployment

End-to-end flow (quantize on a Vitis-AI Docker host, run on the board).

```bash
# 1. prepare calibration images (On PC)
cd Quantization
python prepare_calib.py

# 2. quantize + compile (Inside Vitis-AI container)
python quantize_calib.py
python quantize_export.py
vai_c_xir -x compiled/FullDPU_int.xmodel \
          -a $ARCH -o compiled -n ls_yolo_landslide

# 3. run on the KV260
cd KV260_Deployment
python3 demo_dpu_live.py        # live camera/image demo
python3 eval_display_kv260.py   # P/R/mAP + latency on test set
```

The DPU runs the backbone, neck and head convolutions; the ARM CPU handles the
lightweight sigmoid/grid decode and NMS.

## Acknowledgements

- **LS-YOLO**: Zhang W., Liu Z., Zhou S., Qi W., Wu X., Zhang T., Han L.,
  _"LS-YOLO: A Novel Model for Detecting Multi-Scale Landslides with Remote Sensing
  Images"_, IEEE J-STARS, 17, 4952–4965, 2024.
  Implementation based on [wenjieo/LS-YOLO](https://github.com/wenjieo/LS-YOLO).
- Built on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (GPL-3.0).
