"""
quantize_calib.py — CALIB step of INT8 quantize for LS-YOLO (Decoupled_Detect head).
Run INSIDE Vitis-AI Docker (conda env vitis-ai-pytorch). Replaces the PDF script
(which was written for the standard Detect head).

Assumed layout on machine/board:
  /workspace/best.pt           <- trained model
  /workspace/calib_images/     <- ~200 calib images
  /workspace/compiled/         <- output
  /workspace/LS-YOLO/          <- source repo (contains this kv260_export/ folder)
Change via environment variables if different (MODEL, CALIB, OUT, LSYOLO_SRC).
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
IMG   = 512   # matches imgsz during UAV train (best.pt mAP 0.927); divisible by 32
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
        im = letterbox(im, (IMG, IMG), auto=False)[0][:, :, ::-1].transpose(2, 0, 1)  # letterbox(keep ratio), BGR->RGB, HWC->CHW
        imgs.append(im.astype(np.float32) / 255.0)
    if not imgs:
        raise RuntimeError(f"Could not read any calib images in {CALIB}")
    return torch.tensor(np.array(imgs))


def main():
    net = load_net()
    inp = torch.randn(1, 3, IMG, IMG)

    with torch.no_grad():
        out = net(inp)
    print(f"[CHECK] {len(out)} outputs (1 per level): {[tuple(o.shape) for o in out]}")

    q = torch_quantizer("calib", net, (inp,), output_dir=OUT)
    qm = q.quant_model
    data = calib_data()
    print(f"[INFO] Calibrating {len(data)} images...")
    with torch.no_grad():
        for i in range(len(data)):
            _ = qm(data[i].unsqueeze(0))
    q.export_quant_config()
    print("[DONE] Calib complete:", OUT)


if __name__ == "__main__":
    main()
