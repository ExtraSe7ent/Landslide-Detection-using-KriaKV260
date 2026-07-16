"""
quantize_calib.py — Bước CALIB của quantize INT8 cho LS-YOLO (head Decoupled_Detect).
Chạy TRONG Docker Vitis-AI (conda env vitis-ai-pytorch). Thay cho script PDF
(vốn viết cho head Detect chuẩn).

Layout giả định trên máy/board:
  /workspace/best.pt           <- model đã train
  /workspace/calib_images/     <- ~200 ảnh calib
  /workspace/compiled/         <- output
  /workspace/LS-YOLO/          <- source repo (chứa kv260_export/ này)
Đổi qua biến môi trường nếu khác (MODEL, CALIB, OUT, LSYOLO_SRC).
"""
import os, sys, glob
import numpy as np
import torch

LSYOLO_SRC = os.environ.get("LSYOLO_SRC", "/workspace/LS-YOLO")
sys.path.insert(0, LSYOLO_SRC)
sys.path.insert(0, os.path.join(LSYOLO_SRC, "kv260_export"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from pytorch_nndct.apis import torch_quantizer
from ls_yolo_dpu import DecoupledDPU
from utils.augmentations import letterbox

MODEL = os.environ.get("MODEL", "/workspace/best.pt")
CALIB = os.environ.get("CALIB", "/workspace/calib_images")
OUT   = os.environ.get("OUT",   "/workspace/compiled")
IMG   = 512   # khớp imgsz lúc train UAV (best.pt mAP 0.927); chia hết 32
os.makedirs(OUT, exist_ok=True)


def load_net():
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = (ck["model"] if isinstance(ck, dict) else ck).float().eval()
    net = DecoupledDPU(model).eval()
    return net


def calib_data(n=200):
    files = [f for f in sorted(glob.glob(os.path.join(CALIB, "*")))
             if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff"))][:n]
    imgs = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        im = letterbox(im, (IMG, IMG), auto=False)[0][:, :, ::-1].transpose(2, 0, 1)  # letterbox(giữ tỉ lệ), BGR->RGB, HWC->CHW
        imgs.append(im.astype(np.float32) / 255.0)
    if not imgs:
        raise RuntimeError(f"Không đọc được ảnh calib nào trong {CALIB}")
    return torch.tensor(np.array(imgs))


def main():
    net = load_net()
    inp = torch.randn(1, 3, IMG, IMG)

    with torch.no_grad():
        out = net(inp)
    print(f"[CHECK] {len(out)} output (mỗi level 1): {[tuple(o.shape) for o in out]}")

    q = torch_quantizer("calib", net, (inp,), output_dir=OUT)
    qm = q.quant_model
    data = calib_data()
    print(f"[INFO] Calibrating {len(data)} ảnh...")
    with torch.no_grad():
        for i in range(len(data)):
            _ = qm(data[i].unsqueeze(0))
    q.export_quant_config()
    print("[DONE] Calib xong:", OUT)


if __name__ == "__main__":
    main()
