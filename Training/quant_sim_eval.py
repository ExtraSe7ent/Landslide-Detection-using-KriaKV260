"""
quant_sim_eval.py — MÔ PHỎNG INT8 kiểu Vitis-AI DPU (per-tensor, scale = 2^n,
weight + activation) ngay trên GPU, rồi đo mAP bằng val.py.

Mục đích: ước lượng INT8 làm mất bao nhiêu điểm (so float 92.56% test).
KHÔNG phải Vitis-AI thật, nhưng bắt đúng 2 đặc trưng gây mất chính:
  - 8-bit, đối xứng, PER-TENSOR (không per-channel)
  - scale luỹ thừa 2 (power-of-two fix_point) — thô hơn scale float
Đây thường là CẬN TRÊN lạc quan (Vitis-AI có thêm fast_finetune); nếu sim này
đã ~80% thì INT8 đúng là thủ phạm. Nếu sim ~90%+ thì 80% trên board phần lớn do đo sai.

Chạy:  python kv260_export/quant_sim_eval.py
"""
import os, sys, math, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, cv2, torch, torch.nn as nn

DATA = "data/landslide_uav_finetune.yaml"
WEIGHTS = "D:/Training/runs_dpu/ls_yolo_dpu_uav/weights/best.pt"
IMGSZ, BW = 512, 8
QMIN, QMAX = -(2**(BW-1)), 2**(BW-1) - 1   # -128..127
DEVICE = "cuda:0"
N_CALIB = 200


def calc_fp(max_abs):
    if max_abs is None or max_abs < 1e-12:
        return 0
    return int(math.floor(math.log2(QMAX / max_abs)))   # 2^fp * max_abs <= 127


def fq(x, fp):
    s = float(2.0 ** fp)
    return torch.clamp(torch.round(x * s), QMIN, QMAX) / s


class QuantSim:
    """Fake-quant weight (1 lần) + activation (qua hook, calib rồi freeze)."""
    def __init__(self, model):
        self.model = model
        self.act_max, self.act_fp = {}, {}
        self.calibrating = True
        self.handles = []
        self._wrap()

    def _wrap(self):
        # id các conv2d nằm trong Conv-wrapper (đã được hook ở mức Conv, tránh trùng)
        inside = set()
        for m in self.model.modules():
            if m.__class__.__name__ == "Conv" and hasattr(m, "conv"):
                inside.add(id(m.conv))
        # 1) quant weight tất cả nn.Conv2d (per-tensor, po2) — tĩnh, làm ngay
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                w = m.weight.data
                fp = calc_fp(w.abs().max().item())
                m.weight.data = fq(w, fp)
        # 2) hook activation: Conv-wrapper output + nn.Conv2d đứng riêng
        targets = []
        for name, m in self.model.named_modules():
            if m.__class__.__name__ == "Conv":
                targets.append((name, m))
            elif isinstance(m, nn.Conv2d) and id(m) not in inside:
                targets.append((name, m))
        for name, m in targets:
            self.handles.append(m.register_forward_hook(self._mk_hook(name)))
        # 3) hook input ảnh
        self.handles.append(self.model.register_forward_pre_hook(self._input_hook))

    def _mk_hook(self, key):
        def hook(mod, inp, out):
            if not torch.is_tensor(out):
                return out
            if self.calibrating:
                mx = out.detach().abs().max().item()
                self.act_max[key] = max(self.act_max.get(key, 0.0), mx)
                return out
            return fq(out, self.act_fp.get(key, 0))
        return hook

    def _input_hook(self, mod, args):
        x = args[0]
        if not torch.is_tensor(x):
            return None
        if self.calibrating:
            self.act_max["__input__"] = max(self.act_max.get("__input__", 0.0),
                                            x.detach().abs().max().item())
            return None
        return (fq(x, self.act_fp.get("__input__", 0)),) + args[1:]

    def freeze(self):
        self.act_fp = {k: calc_fp(v) for k, v in self.act_max.items()}
        self.calibrating = False


def letterbox_load(path):
    from utils.dataloaders import letterbox
    data = np.fromfile(path, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = letterbox(im, (IMGSZ, IMGSZ), auto=False, scaleup=False)[0][:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(im)).float().div(255)


CFG = None  # dict data, set trong main


def run_val(model, dataloader, tag):
    from val import run
    out = run(data=CFG, model=model, dataloader=dataloader, task="test",
              half=False, plots=False, compute_loss=None, verbose=False)
    mp, mr, map50, map5095 = out[0][0], out[0][1], out[0][2], out[0][3]  # out[0]=tuple metrics
    print(f"\n[{tag}] P={mp:.4f} R={mr:.4f} "
          f"mAP@0.5={map50:.4f} ({map50*100:.2f}%) mAP@0.5:0.95={map5095:.4f}", flush=True)
    return map50


def main():
    global CFG
    import yaml
    import utils.dataloaders as _dl
    from utils.dataloaders import create_dataloader
    from models.experimental import attempt_load
    with open(DATA) as f:
        cfg = yaml.safe_load(f)
    CFG = cfg
    _dl._LABEL_DIR = cfg["label_dir"]
    _dl._IMG_SUBDIR = cfg.get("img_subdir", "img")

    model = attempt_load(WEIGHTS, device=DEVICE, fuse=True).float().eval()

    # dataloader test (rect, letterbox — giống val.py)
    dl = create_dataloader(cfg["test"], IMGSZ, 16, 32, False,
                           pad=0.5, rect=True, workers=0, prefix="test: ")[0]

    # (1) FLOAT qua cùng path — kỳ vọng ~92.5 (xác nhận harness không thiên lệch)
    run_val(model, dl, "FLOAT (sanity)")

    # (2) Calib INT8 trên N_CALIB ảnh train
    qs = QuantSim(model)
    train_files = [l.strip() for l in open(cfg["train"], encoding="utf-8") if l.strip()]
    random.seed(0); random.shuffle(train_files)
    done = 0
    with torch.no_grad():
        for fpath in train_files:
            if done >= N_CALIB:
                break
            t = letterbox_load(fpath)
            if t is None:
                continue
            _ = model(t.unsqueeze(0).to(DEVICE))
            done += 1
    qs.freeze()
    print(f"[INFO] Calib INT8 xong trên {done} ảnh. "
          f"#activation tensors quant = {len(qs.act_fp)}", flush=True)

    # (3) INT8-SIM
    run_val(model, dl, "INT8-SIM")


if __name__ == "__main__":
    main()
