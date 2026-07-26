"""
verify_decode.py — Proves that DecoupledDPU + decode_decoupled gives EXACTLY
the SAME RESULT as the original model forward (model.model[-1] is Decoupled_Detect).

Run:  python kv260_export/verify_decode.py
No GPU needed, no training needed. PASS => decode logic on KV260 will be correct.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.yolo import Model
from ls_yolo_dpu import DecoupledDPU, decode_decoupled, head_constants

CFG = "models/landslide/Improve.yaml"


def main():
    torch.manual_seed(0)
    model = Model(CFG)
    model.eval()
    det = model.model[-1]

    x = torch.randn(1, 3, 512, 512)   # matches imgsz deploy UAV (512)
    with torch.no_grad():
        ref = model(x)[0]                       # original forward (inference) -> [bs, N, 5+nc]
        wrap = DecoupledDPU(model)
        raw = wrap(x)                           # raw conv maps (DPU output)
        dec = decode_decoupled(raw, det.anchors, det.stride, int(det.nc), int(det.na))

    print("ref :", tuple(ref.shape))
    print("dec :", tuple(dec.shape))
    print("raw outputs:", len(raw), "tensors ->", [tuple(t.shape) for t in raw])
    same_shape = ref.shape == dec.shape
    maxdiff = (ref - dec).abs().max().item() if same_shape else float('nan')
    print(f"shape match: {same_shape} | max abs diff: {maxdiff:.3e}")
    ok = same_shape and torch.allclose(ref, dec, atol=1e-4, rtol=1e-4)
    print("HEAD CONSTANTS for demo:", head_constants(model))
    print("\n==>", "PASS [OK] decode matches original forward" if ok else "FAIL [X] decode DOES NOT match")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
