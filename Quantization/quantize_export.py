"""
quantize_export.py — Export xmodel for LS-YOLO.
Run INSIDE Vitis-AI Docker AFTER quantize_calib.py:
    python3 /workspace/quantize_export.py
"""
import sys
sys.path.append('/workspace/LS-YOLO')
import os
import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer

MODEL = "/workspace/best_qat.pt"
OUT   = "/workspace/compiled"

# (Do not force CPU threads here as underlying DPU compiler may not be thread-safe)

# ── DecoupledDPU wrapper ──────────────────────────────────────────────
class DecoupledDPU(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = model.model
        self.save = model.save
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
    print(f"[INFO] Running export on device: {device}")
    print("[INFO] Loading best_qat.pt for export...")
    
    m = torch.load(MODEL, map_location=device, weights_only=False)
    m = (m['model'] if isinstance(m, dict) else m).float().eval()

    # Sync ECA (Fix YOLO version error by swapping class)
    for module in m.modules():
        if module.__class__.__name__ == 'ECA':
            if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                # This is actually the old SE network. Assign it to LegacyECA
                module.__class__ = LegacyECA
            else:
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

    q = torch_quantizer('test', net, (inp,), output_dir=OUT)
    with torch.no_grad():
        _ = q.quant_model(inp)

    q.export_xmodel(deploy_check=True, output_dir=OUT)
    print(f"[DONE] Export complete: {OUT}/DecoupledDPU_int.xmodel")

if __name__ == "__main__":
    main()
