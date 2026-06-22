"""
quantize_calib.py — Calibration for LS-YOLO with Decoupled_Detect.
Fix: Read one image at a time instead of loading all into RAM to avoid OOM.
Run INSIDE Docker Vitis-AI: python quantize_calib.py
"""
import sys
sys.path.append('../Training')
import os, cv2, numpy as np
import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer

try:
    import seaborn
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "opencv-python-headless", "seaborn", "pandas",
        "tqdm", "matplotlib", "pyyaml", "requests", "--quiet"])

MODEL = "XXXXXX/best.pt"
OUT   = "XXXXXX/compiled"
CALIB = "XXXXXX/calib_images"
os.makedirs(OUT, exist_ok=True)

# ── FullDPU wrapper ──────────────────────────────────────────────
class FullDPU(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.model
        self.save   = model.save
        self.detect = model.model[-1]

    def forward(self, x):
        y = [None] * len(self.layers)
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i in self.save:
                y[m.i] = x

        feats = [y[j] for j in self.detect.f]
        outs  = []
        for i in range(self.detect.nl):
            xi     = self.detect.m_stem[i](feats[i])
            x_cls  = self.detect.m_cls[i](xi)
            x_cam  = self.detect.cam[i](xi)
            x_reg  = self.detect.m_reg[i](x_cam)
            x_conf = self.detect.m_conf[i](x_cam)
            outs.append(torch.cat([x_reg, x_conf, x_cls], dim=1))
        return tuple(outs)

# ── Get file list (do not load into RAM) ─────────────────────────
def get_calib_files():
    if not os.path.exists(CALIB):
        raise RuntimeError(f"[ERROR] Not found: {CALIB}")

    files = sorted([f for f in os.listdir(CALIB)
                    if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))
                    and not f.startswith('.')])[:1000]

    if not files:
        raise RuntimeError(f"[ERROR] No valid images in {CALIB}")

    print(f"[INFO] Found {len(files)} calibration images.")
    return files

# ── Read single image ───────────────────────────────────────────
def load_one(path):
    im = cv2.imread(path)
    if im is None:
        return None
    im = cv2.resize(im, (512, 512))
    im = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(im).unsqueeze(0)   # [1, 3, 512, 512]

# ── Main ────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading best.pt...")
    m = torch.load(MODEL, map_location='cpu', weights_only=False)
    m = (m['model'] if isinstance(m, dict) else m).float().eval()

    net = FullDPU(m).eval()
    inp = torch.randn(1, 3, 512, 512)

    # Check output shape
    with torch.no_grad():
        outs = net(inp)
    print(f"[CHECK] Output tensor count: {len(outs)}")
    for k, o in enumerate(outs):
        print(f"  Scale {k}: {tuple(o.shape)}")
    # Expected: (1,18,64,64) / (1,18,32,32) / (1,18,16,16)

    # Init quantizer
    q  = torch_quantizer('calib', net, (inp,), output_dir=OUT)
    qm = q.quant_model

    # Calibrate — read one by one, do not load all into RAM
    files = get_calib_files()
    print(f"[INFO] Starting calibration on {len(files)} images...")
    done = 0
    with torch.no_grad():
        for f in files:
            tensor = load_one(os.path.join(CALIB, f))
            if tensor is None:
                continue
            _ = qm(tensor)
            done += 1
            if done % 100 == 0:
                print(f"  [{done}/{len(files)}]...")

    print(f"[INFO] Calibrated {done} images.")
    q.export_quant_config()
    print(f"[DONE] Calibration complete. Data at: {OUT}")

if __name__ == "__main__":
    main()