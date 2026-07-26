"""
qat_finetune.py — Real QAT (simulation) on local machine: fine-tune model injected with fake-quant
INT8 (STE) using the repo's original ComputeLoss, then measure INT8 before/after QAT.

Purpose: prove that "quantize-aware training" recovers the INT8 accuracy drop (~4 points).
The final deploy version should run QAT using Vitis-AI QatProcessor; this is a local/fallback version.

Run:  python kv260_export/qat_finetune.py [n_steps]
"""
import os, sys, math, random
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DATA = "data/landslide_uav_finetune.yaml"
WEIGHTS = "D:/Training/runs_dpu/ls_yolo_dpu_uav/weights/best.pt"
HYP = "D:/Training/runs_dpu/ls_yolo_dpu_uav/hyp.yaml"
IMGSZ, BW = 512, 8
QMIN, QMAX = -(2**(BW-1)), 2**(BW-1) - 1
DEVICE = "cuda:0"
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
BATCH = 8
SAVE = os.environ.get("QAT_SAVE", "D:/Training/runs_dpu/ls_yolo_dpu_uav/weights/best_qat.pt")
CFG = None


def calc_fp(m):
    if m is None or m < 1e-12:
        return 0
    return int(math.floor(math.log2(QMAX / m)))


def fq(x, fp):
    s = float(2.0 ** fp)
    return torch.clamp(torch.round(x * s), QMIN, QMAX) / s


def ste(x, fp):
    """fake-quant with straight-through estimator (gradient passes directly)."""
    return x + (fq(x, fp) - x).detach()


class QATConv2d(nn.Module):
    """Wraps an nn.Conv2d: fake-quant weight (STE) on each forward pass. Original weight remains trainable."""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.w_fp = calc_fp(conv.weight.detach().abs().max().item())

    def forward(self, x):
        w = ste(self.conv.weight, self.w_fp)
        return F.conv2d(x, w, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


class QAT:
    def __init__(self, model):
        self.model = model
        self.act_max, self.act_fp = {}, {}
        self.calibrating = True
        self.handles = []
        self._replace_convs()
        self._hook_acts()
        self.handles.append(self.model.register_forward_pre_hook(self._input_hook))

    def _replace_convs(self):
        for name, m in list(self.model.named_modules()):
            for cn, c in list(m.named_children()):
                if isinstance(c, nn.Conv2d):
                    setattr(m, cn, QATConv2d(c))

    def _hook_acts(self):
        for name, m in self.model.named_modules():
            if m.__class__.__name__ == "Conv":      # post-act output (after BN+act)
                self.handles.append(m.register_forward_hook(self._mk(name)))

    def unwrap(self):
        """Remove QATConv2d + remove hooks -> CLEAN float model (quant-robust weights)."""
        for h in self.handles:
            h.remove()
        for name, m in list(self.model.named_modules()):
            for cn, c in list(m.named_children()):
                if isinstance(c, QATConv2d):
                    setattr(m, cn, c.conv)

    def _mk(self, key):
        def hook(mod, inp, out):
            if not torch.is_tensor(out):
                return out
            if self.calibrating:
                self.act_max[key] = max(self.act_max.get(key, 0.0),
                                        out.detach().abs().max().item())
                return out
            return ste(out, self.act_fp.get(key, 0))
        return hook

    def _input_hook(self, mod, args):
        x = args[0]
        if not torch.is_tensor(x):
            return None
        if self.calibrating:
            self.act_max["__in__"] = max(self.act_max.get("__in__", 0.0),
                                         x.detach().abs().max().item())
            return None
        return (ste(x, self.act_fp.get("__in__", 0)),) + args[1:]

    def freeze(self):
        self.act_fp = {k: calc_fp(v) for k, v in self.act_max.items()}
        self.calibrating = False


def eval_int8(model, dl, tag):
    from val import run
    out = run(data=CFG, model=model, dataloader=dl, task="test",
              half=False, plots=False, compute_loss=None, verbose=False)
    mp, mr, ap50, ap = out[0][0], out[0][1], out[0][2], out[0][3]
    print(f"\n[{tag}] P={mp:.4f} R={mr:.4f} mAP@0.5={ap50:.4f} "
          f"({ap50*100:.2f}%) mAP@0.5:0.95={ap:.4f}", flush=True)
    return ap50


def main():
    global CFG
    import yaml
    import utils.dataloaders as _dl
    from utils.dataloaders import create_dataloader
    from models.experimental import attempt_load
    from utils.loss import ComputeLoss
    with open(DATA) as f:
        CFG = yaml.safe_load(f)
    with open(HYP) as f:
        hyp = yaml.safe_load(f)
    _dl._LABEL_DIR = CFG["label_dir"]
    _dl._IMG_SUBDIR = CFG.get("img_subdir", "img")

    model = attempt_load(WEIGHTS, device=DEVICE, fuse=False).float()
    model.hyp = hyp
    model.nc = 1
    model.gr = 1.0
    for p in model.parameters():     # attempt_load freezes grad for inference -> turn it back on
        p.requires_grad_(True)

    test_dl = create_dataloader(CFG["test"], IMGSZ, 16, 32, False,
                                pad=0.5, rect=True, workers=0, prefix="test: ")[0]
    train_dl = create_dataloader(CFG["train"], IMGSZ, BATCH, 32, False, hyp=hyp,
                                 augment=False, workers=0, shuffle=True, prefix="train: ")[0]

    qat = QAT(model)

    # calib fix_point on ~100 small batches
    model.eval()
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(train_dl):
            model(imgs.to(DEVICE).float() / 255)
            if i >= 12:
                break
    qat.freeze()
    print(f"[INFO] calib done, #act tensors = {len(qat.act_fp)}", flush=True)

    # measure INT8 BEFORE QAT
    eval_int8(model, test_dl, "INT8 before QAT")

    # fine-tune QAT
    compute_loss = ComputeLoss(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=1e-5)
    model.train()
    step = 0
    while step < N_STEPS:
        for imgs, targets, _, _ in train_dl:
            imgs = imgs.to(DEVICE).float() / 255
            pred = model(imgs)
            loss, items = compute_loss(pred, targets.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{N_STEPS}  loss={loss.item():.4f}", flush=True)
            if step >= N_STEPS:
                break

    # measure INT8 AFTER QAT (model still has fake-quant)
    eval_int8(model, test_dl, "INT8 after QAT")

    # Unwrap wrapper -> save CLEAN float checkpoint (quant-robust) as FALLBACK
    qat.unwrap()
    model.eval()
    ckpt = {"model": model.half(), "epoch": -1, "best_fitness": None,
            "qat_note": "QAT fine-tuned (fake-quant STE local). Load into Vitis-AI for real PTQ/QAT."}
    torch.save(ckpt, SAVE)
    print(f"[SAVE] Fallback QAT checkpoint -> {SAVE}", flush=True)
    model.float()
    eval_int8(model, test_dl, "FLOAT after QAT (saved checkpoint)")


if __name__ == "__main__":
    main()
