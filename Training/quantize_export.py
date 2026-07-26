"""
quantize_export.py — TEST step + export .xmodel for LS-YOLO (Decoupled_Detect head).
Run AFTER quantize_calib.py, in the same Vitis-AI Docker.
Also writes head_constants.json (anchors/stride/nc/na) next to xmodel for demo on
KV260 to decode correctly — DO NOT hardcode wrong anchors.
"""
import os, sys, json
import torch

LSYOLO_SRC = os.environ.get("LSYOLO_SRC", "/workspace/LS-YOLO")
sys.path.insert(0, LSYOLO_SRC)
sys.path.insert(0, os.path.join(LSYOLO_SRC, "kv260_export"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_nndct.apis import torch_quantizer
from ls_yolo_dpu import DecoupledDPU, head_constants

MODEL = os.environ.get("MODEL", "/workspace/best.pt")
OUT   = os.environ.get("OUT",   "/workspace/compiled")
IMG   = 512   # matches imgsz during UAV train (best.pt mAP 0.927); divisible by 32
os.makedirs(OUT, exist_ok=True)


def main():
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = (ck["model"] if isinstance(ck, dict) else ck).float().eval()

    # save head constants for demo usage (correct anchors/stride of the model)
    consts = head_constants(model)
    with open(os.path.join(OUT, "head_constants.json"), "w") as f:
        json.dump(consts, f, indent=2)
    print("[INFO] head_constants:", consts)

    net = DecoupledDPU(model).eval()
    inp = torch.randn(1, 3, IMG, IMG)

    q = torch_quantizer("test", net, (inp,), output_dir=OUT)
    with torch.no_grad():
        _ = q.quant_model(inp)
    q.export_xmodel(deploy_check=False, output_dir=OUT)
    print("[DONE] Export complete:", os.path.join(OUT, "DecoupledDPU_int.xmodel"))


if __name__ == "__main__":
    main()
