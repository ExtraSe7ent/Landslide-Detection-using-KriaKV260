"""
quantize_calib.py — Calibration for LS-YOLO with Decoupled_Detect.
Supports running on both CPU (Mac) and GPU (Windows). Optimized speed with DataLoader (Batching).
"""
import sys
sys.path.append('/workspace/LS-YOLO')
import os, cv2, numpy as np
import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer
from torch.utils.data import Dataset, DataLoader

try:
    import seaborn
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "opencv-python-headless", "seaborn", "pandas",
        "tqdm", "matplotlib", "pyyaml", "requests", "--quiet"])

MODEL = "/workspace/best_qat.pt"
OUT   = "/workspace/compiled"
CALIB = "/workspace/eval_kv260_dataset/test_images"  # Pointed directly to the directory containing 1017 test images
os.makedirs(OUT, exist_ok=True)

# Get optimal thread count (Allow full 100% resource utilization to reach 800%)
if not torch.cuda.is_available():
    torch.set_num_threads(os.cpu_count())

# ── DecoupledDPU wrapper ──────────────────────────────────────────────
class DecoupledDPU(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.model
        self.save = model.save
        self.detect = model.model[-1]
        assert hasattr(self.detect, 'm_stem'), "Need to use model with Decoupled_Detect head"

    def forward(self, x):
        y = [None] * len(self.layers)
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i in self.save:
                y[m.i] = x
        feats = [y[j] for j in self.detect.f]
        d = self.detect
        outs = []
        for i in range(d.nl):
            stem = d.m_stem[i](feats[i])
            cls_raw = d.m_cls[i](stem)
            cam = d.cam[i](stem)
            reg_raw = d.m_reg[i](cam)
            conf_raw = d.m_conf[i](cam)
            outs.append(torch.cat([reg_raw, conf_raw, cls_raw], dim=1))
        return tuple(outs)

# ── Dataset for reading images ────────────────────────────────────────────────
class CalibDataset(Dataset):
    def __init__(self, img_dir, limit=None):
        if not os.path.exists(img_dir):
            raise RuntimeError(f"[ERROR] Not found: {img_dir}")
        self.img_dir = img_dir
        self.files = sorted([f for f in os.listdir(img_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
                             and not f.startswith('.')])
        if limit is not None:
            self.files = self.files[:limit]
        if not self.files:
            raise RuntimeError(f"[ERROR] No valid images in {img_dir}")
        
        # Import local letterbox (fallback between dataloaders and augmentations)
        sys.path.append('/workspace/LS-YOLO')
        try:
            from utils.augmentations import letterbox
            self.letterbox = letterbox
        except ImportError:
            try:
                from utils.dataloaders import letterbox
                self.letterbox = letterbox
            except ImportError:
                raise RuntimeError("[ERROR] Cannot find letterbox function in LS-YOLO/utils/")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.files[idx])
        im = cv2.imread(path)
        if im is None:
            # Fallback black tensor if image error
            return torch.zeros((3, 512, 512), dtype=torch.float32)
        im = self.letterbox(im, (512, 512), auto=False)[0]
        im = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(im)

# ── Legacy ECA (For old best_qat.pt) ─────────────────────────
class LegacyECA(nn.Module):
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.gate(y)
        # Remove .expand_as(x) to eliminate nndct_expand_as node not supported by DPU hardware
        # Since fc1 and fc2 are Conv2d, y is already in [B, C, 1, 1] shape, PyTorch will broadcast smoothly.
        return x * y

# ── Main ────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Running on device: {device}")
    print("[INFO] Loading best_qat.pt...")
    
    m = torch.load(MODEL, map_location=device, weights_only=False)
    m = (m['model'] if isinstance(m, dict) else m).float().eval()

    # Sync ECA (Fix YOLO version error by swapping class)
    for module in m.modules():
        if module.__class__.__name__ == 'ECA':
            if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                # This is actually the old SE network. Assign it to LegacyECA
                module.__class__ = LegacyECA
            else:
                # Other fixes (if it's a real ECA)
                if not hasattr(module, 'conv'):
                    for name, child in module.named_children():
                        if isinstance(child, nn.Conv1d):
                            module.conv = child
                            break
                if hasattr(module, 'gate') and not hasattr(module, 'sigmoid'):
                    module.sigmoid = module.gate
                elif hasattr(module, 'sigmoid') and not hasattr(module, 'gate'):
                    module.gate = module.sigmoid
                if hasattr(module, 'gate') and isinstance(module.gate, nn.Hardsigmoid):
                    module.gate.once = False
                    module.gate.inplace = False
                if hasattr(module, 'act') and not hasattr(module, 'relu'):
                    module.relu = module.act
                elif hasattr(module, 'relu') and not hasattr(module, 'act'):
                    module.act = module.relu

    net = DecoupledDPU(m).to(device).eval()
    inp = torch.randn(1, 3, 512, 512).to(device)

    # Check output shape
    with torch.no_grad():
        outs = net(inp)
    print(f"[CHECK] Number of output tensors: {len(outs)}")
    for k, o in enumerate(outs):
        print(f"  Scale {k}: {tuple(o.shape)}")

    # Init quantizer
    q  = torch_quantizer('calib', net, (inp,), output_dir=OUT)
    qm = q.quant_model.to(device)

    # Optimization: Use batch_size=1 to completely match dummy input when tracing Vitis-AI graph
    # (Avoid BATCH_SIZE shape mismatch error in NNDCT)
    dataset = CalibDataset(CALIB, limit=None)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"[INFO] Start calibrating {len(dataset)} images (Batch size: {batch_size})...")
    done = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _ = qm(batch)
            done += len(batch)
            print(f"  [{min(done, len(dataset))}/{len(dataset)}]...")

    print(f"[INFO] Finished calibrating {done} images.")
    q.export_quant_config()
    print(f"[DONE] Calibration done. Data at: {OUT}")

if __name__ == "__main__":
    main()