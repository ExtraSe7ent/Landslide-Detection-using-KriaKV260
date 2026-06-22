"""
quantize_export.py — Export xmodel for LS-YOLO.
Run INSIDE Docker Vitis-AI AFTER quantize_calib.py:
    python3 quantize_export.py
"""
import sys
sys.path.append('../Training')
import os
import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer

MODEL = "XXXXXX/best.pt"
OUT   = "XXXXXX/compiled"

# ── FullDPU wrapper (same as quantize_calib.py) ───────────────
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

# ── Main ────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading best.pt for export...")
    m = torch.load(MODEL, map_location='cpu', weights_only=False)
    m = (m['model'] if isinstance(m, dict) else m).float().eval()

    # Sync ECA (harmless if already correct)
    for module in m.modules():
        if module.__class__.__name__ == 'ECA':
            if hasattr(module, 'act') and not hasattr(module, 'relu'):
                module.relu = module.act
            elif hasattr(module, 'relu') and not hasattr(module, 'act'):
                module.act = module.relu
            if hasattr(module, 'gate') and not hasattr(module, 'sigmoid'):
                module.sigmoid = module.gate
            elif hasattr(module, 'sigmoid') and not hasattr(module, 'gate'):
                module.gate = module.sigmoid
            if hasattr(module, 'gate') and isinstance(module.gate, nn.Hardsigmoid):
                module.gate.inplace = False

    net = FullDPU(m).eval()
    inp = torch.randn(1, 3, 512, 512)

    q = torch_quantizer('test', net, (inp,), output_dir=OUT)
    with torch.no_grad():
        _ = q.quant_model(inp)

    q.export_xmodel(deploy_check=False, output_dir=OUT)
    print(f"[DONE] Export complete: {OUT}/FullDPU_int.xmodel")

if __name__ == "__main__":
    main()
