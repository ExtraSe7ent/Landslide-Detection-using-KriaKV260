"""Run val directly via its run() API and print final metrics only."""
import os, sys, io, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch
import utils.dataloaders as _dl

# Suppress tqdm noise
import tqdm as _tqdm_mod
_tqdm_mod.tqdm = _tqdm_mod.tqdm  # keep reference

from val import run

# Need to pre-inject label_dir before run() calls check_dataset
import yaml
with open('data/landslide.yaml') as f:
    data_cfg = yaml.safe_load(f)
_dl._LABEL_DIR = data_cfg['label_dir']
_dl._IMG_SUBDIR = data_cfg.get('img_subdir', 'img')

print("Running evaluation on test split...")
results = run(
    data='data/landslide.yaml',
    weights='runs/train/ls_yolo_v1/weights/best.pt',
    imgsz=512,
    batch_size=16,
    conf_thres=0.001,   # low conf for valid mAP computation
    iou_thres=0.5,
    task='test',
    device='0',
    plots=False,
    verbose=True,
)

# results = (mp, mr, map50, map, (loss_box, loss_obj, loss_cls), maps, t)
mp, mr, map50, map5095 = results[0], results[1], results[2], results[3]
speed = results[6]  # (preprocess, inference, nms) ms

print("\n" + "="*55)
print(f"  Test Set Results — LS-YOLO (50 epochs)")
print("="*55)
print(f"  Precision   (P):  {mp:.4f}  ({mp*100:.1f}%)")
print(f"  Recall      (R):  {mr:.4f}  ({mr*100:.1f}%)")
print(f"  mAP@0.5      :   {map50:.4f}  ({map50*100:.2f}%)")
print(f"  mAP@0.5:0.95 :   {map5095:.4f}  ({map5095*100:.2f}%)")
print(f"  Speed        : preprocess={speed[0]:.1f}ms  infer={speed[1]:.1f}ms  NMS={speed[2]:.1f}ms")
print("="*55)
