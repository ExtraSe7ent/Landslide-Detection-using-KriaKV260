"""
quantize_qat.py — REAL QAT for LS-YOLO using Vitis-AI QatProcessor.
Run INSIDE Vitis-AI Docker (conda env vitis-ai-pytorch), WITH GPU.

This is the QAT version for DEPLOYMENT (unlike qat_finetune.py which is local STE fake-quant).
QatProcessor flow: trainable_model -> fine-tune -> to_deployable -> export_xmodel.

NOTE on workflow:
  - QAT train needs loss on HEAD OUTPUT in training format -> we train on FULL model
    (models.yolo.Model, Decoupled_Detect head in train-mode returns [bs,na,h,w,no]).
  - Deployment needs forward pass of ONLY convs -> use DecoupledDPU when exporting.
  Simple & robust way: QAT on FULL model, then load QAT weights into DecoupledDPU
  then run quantize_calib.py + quantize_export.py as usual (PTQ on quant-robust weights).
  This script illustrates the full QatProcessor flow; if env fails on head decode, use the simple way above.

Env (change if different):
  MODEL=/workspace/best_qat.pt   (or best.pt)  OUT=/workspace/compiled
"""
import os, sys, math, yaml
sys.path.insert(0, os.environ.get("LSYOLO_SRC", "/workspace/LS-YOLO"))
import torch
from pytorch_nndct import QatProcessor          # Vitis-AI QAT API
from utils.dataloaders import create_dataloader
import utils.dataloaders as _dl
from utils.loss import ComputeLoss
from models.experimental import attempt_load

MODEL = os.environ.get("MODEL", "/workspace/best_qat.pt")
DATA  = os.environ.get("DATA",  "data/landslide_uav_finetune.yaml")
HYP   = os.environ.get("HYP",   "data/hyps/hyp.scratch-landslide-dpu.yaml")
OUT   = os.environ.get("OUT",   "/workspace/compiled")
IMG, BATCH, EPOCH_STEPS, LR = 512, 8, 1500, 1e-4
DEVICE = "cuda"
os.makedirs(OUT, exist_ok=True)


def main():
    cfg = yaml.safe_load(open(DATA))
    hyp = yaml.safe_load(open(HYP))
    _dl._LABEL_DIR = cfg["label_dir"]; _dl._IMG_SUBDIR = cfg.get("img_subdir", "img")

    model = attempt_load(MODEL, device=DEVICE, fuse=False).float()
    model.hyp, model.nc, model.gr = hyp, int(cfg["nc"]), 1.0
    for p in model.parameters():
        p.requires_grad_(True)

    # 1) Initialize QatProcessor on FULL model (train-mode head returns raw [bs,na,h,w,no])
    inp = torch.randn(1, 3, IMG, IMG, device=DEVICE)
    qat = QatProcessor(model, (inp,), bitwidth=8, device=torch.device(DEVICE))
    qmodel = qat.trainable_model()              # model with fake-quant, trainable
    qmodel.hyp, qmodel.nc, qmodel.gr = hyp, int(cfg["nc"]), 1.0

    # 2) Fine-tune QAT
    train_dl = create_dataloader(cfg["train"], IMG, BATCH, 32, False, hyp=hyp,
                                 augment=True, workers=4, shuffle=True, prefix="qat: ")[0]
    compute_loss = ComputeLoss(qmodel if hasattr(qmodel, "model") else model)
    opt = torch.optim.SGD(qmodel.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCH_STEPS, eta_min=1e-6)
    qmodel.train(); step = 0
    while step < EPOCH_STEPS:
        for imgs, targets, _, _ in train_dl:
            imgs = imgs.to(DEVICE).float() / 255
            loss, _ = compute_loss(qmodel(imgs), targets.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % 100 == 0:
                print(f"[QAT] step {step}/{EPOCH_STEPS} loss={loss.item():.4f}", flush=True)
            if step >= EPOCH_STEPS:
                break

    # 3) Convert to deployable model + export xmodel (INT8)
    qmodel.eval()
    deployable = qat.to_deployable(qmodel, OUT)
    qat.export_xmodel(OUT, deploy_check=True)   # ENABLE deploy_check to catch float/int8 mismatch
    print(f"[DONE] QAT done. xmodel + deploy checkpoint at: {OUT}")


if __name__ == "__main__":
    main()
