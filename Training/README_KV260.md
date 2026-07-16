# Triển khai LS-YOLO (head Decoupled_Detect) lên KV260 — Hướng A

> Bộ script này **thay thế** phần quantize/demo trong file PDF hướng dẫn. PDF viết
> cho head `Detect` chuẩn (`detect.m[i]`); model thật của bạn dùng
> **`Decoupled_Detect`** (m_stem + m_cls + **CAM dilation 1/3/5** + m_reg + m_conf),
> chạy script PDF sẽ sai/crash.

## Đã sửa trong repo (bắt buộc, làm 1 lần — TRƯỚC khi train)
`models/common.py`:
- `Conv.default_act`: `nn.SiLU()` -> `nn.Hardswish()` (DPU không có SiLU)
- `BottleneckCSP.act`: `nn.SiLU()` -> `nn.Hardswish()`
- `class ECA`: `Conv1d`+transpose+Sigmoid -> SE-block (`Conv2d` 1×1 + `Hardsigmoid`),
  giữ tên + `k_size` nên `MSFE` không phải đổi.

Upsample đã là `nearest` sẵn -> không cần sửa.

## File trong thư mục này
| File | Chạy ở đâu | Việc |
|---|---|---|
| `ls_yolo_dpu.py` | (lõi, import) | `DecoupledDPU` wrapper + `decode_decoupled` (CPU) |
| `verify_decode.py` | laptop | Chứng minh decode == forward gốc (đã PASS, diff=0) |
| `dryrun_export.py` | Docker Vitis-AI | **De-risk**: xuất xmodel từ weight random để check subgraph |
| `quantize_calib.py` | Docker Vitis-AI | Calib INT8 (cần best.pt + calib_images) |
| `quantize_export.py` | Docker Vitis-AI | Xuất `DecoupledDPU_int.xmodel` + `head_constants.json` |
| `demo_dpu_live.py` | KV260 | Inference DPU + decode + NMS + hiển thị |

## Quy trình

### 0. Verify decode (laptop, đã chạy)
```bash
python kv260_export/verify_decode.py     # PASS = logic decode đúng
```

### 1. DRY-RUN trước khi train (Docker Vitis-AI) — bước quan trọng nhất của hướng A
Mục đích: biết kiến trúc có map trọn vẹn lên DPU **trước khi** tốn 4–8h train.
```bash
# trong container, đã activate vitis-ai-pytorch, đã copy LS-YOLO -> /workspace/LS-YOLO
cd /workspace
LSYOLO_SRC=/workspace/LS-YOLO python LS-YOLO/kv260_export/dryrun_export.py

ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
vai_c_xir -x /workspace/compiled_dryrun/DecoupledDPU_int.xmodel -a $ARCH \
          -o /workspace/compiled_dryrun -n dryrun
```
**ĐỌC LOG:**
- `1 DPU subgraph` -> map trọn vẹn (OK) -> sang bước 2 train yên tâm.
- Nhiều subgraph -> xem op nào rớt. Nghi ngờ số 1 là **CAM dilation 3/5**. Cách xử lý:
  trong `models/common.py` class `CAM`, đổi `dilation` của `conv2`/`conv3` từ
  `3`/`5` -> `2`/`4` (giá trị DPU chắc chắn hỗ trợ), giữ nguyên decoupled head, rồi
  chạy lại dry-run. (Sẽ phải train với cấu hình mới.)

### 2. Train (laptop GPU) — dùng hyp robust domain-gap
```bash
python train.py --cfg models/landslide/Improve.yaml --data data/landslide.yaml \
  --hyp data/hyps/hyp.scratch-landslide-dpu.yaml --weights "" \
  --epochs 300 --batch-size 8 --imgsz 512 --optimizer SGD --device 0 --name ls_yolo_dpu_A
```

### 3. Quantize + compile thật (Docker Vitis-AI)
```bash
cd /workspace            # có best.pt, calib_images/, LS-YOLO/
LSYOLO_SRC=/workspace/LS-YOLO python LS-YOLO/kv260_export/quantize_calib.py
LSYOLO_SRC=/workspace/LS-YOLO python LS-YOLO/kv260_export/quantize_export.py

ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
vai_c_xir -x /workspace/compiled/DecoupledDPU_int.xmodel -a $ARCH \
          -o /workspace/compiled -n ls_yolo_landslide
# -> /workspace/compiled/ls_yolo_landslide.xmodel  (+ head_constants.json)
```

### 4. Chạy trên KV260
Copy sang board: `ls_yolo_landslide.xmodel`, `head_constants.json`, cả thư mục
`LS-YOLO/` (chứa `kv260_export/` + `utils/`).
```bash
# trong container smartcam trên KV260:
cd ~/project/LS-YOLO/kv260_export
MODEL=~/project/LS-YOLO/ls_yolo_landslide.xmodel SOURCE=0 python3 demo_dpu_live.py
# Test domain khớp (khuyến nghị): trỏ vào ảnh aerial thay vì webcam phòng
MODEL=.../ls_yolo_landslide.xmodel SOURCE=/path/to/aerial.jpg python3 demo_dpu_live.py
```

## Lưu ý accuracy (đọc kỹ)
- **Domain gap**: model train ảnh top-down (UAV/vệ tinh). Webcam mặt đất -> box vô
  nghĩa. Dùng `SOURCE=<ảnh/video aerial>` để demo đúng domain. `hyp.scratch-landslide-dpu.yaml`
  chỉ tăng độ bền *trong* domain top-down, không bù được khác biệt lớn.
- **Quantize INT8**: thường tụt 1–5% mAP, landslide nhỏ/mảnh dễ miss. So mAP
  bản float (val.py) vs cảm nhận trên board để biết mức rớt.
