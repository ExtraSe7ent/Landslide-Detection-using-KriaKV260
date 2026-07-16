"""
quantize_export.py — Bước TEST + xuất .xmodel cho LS-YOLO (head Decoupled_Detect).
Chạy SAU quantize_calib.py, trong cùng Docker Vitis-AI.
Cũng ghi head_constants.json (anchors/stride/nc/na) cạnh xmodel để demo trên
KV260 decode đúng — KHÔNG hardcode nhầm anchors.
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
IMG   = 512   # khớp imgsz lúc train UAV (best.pt mAP 0.927); chia hết 32
os.makedirs(OUT, exist_ok=True)


def main():
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = (ck["model"] if isinstance(ck, dict) else ck).float().eval()

    # lưu hằng số head để demo dùng (đúng anchors/stride của model)
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
    print("[DONE] Xuất xong:", os.path.join(OUT, "DecoupledDPU_int.xmodel"))


if __name__ == "__main__":
    main()
