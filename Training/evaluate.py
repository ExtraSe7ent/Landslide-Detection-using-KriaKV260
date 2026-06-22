"""
Evaluate LS-YOLO on test set + visualize predictions vs ground truth.

Usage:
    python evaluate.py                    # eval + visualize 16 random test images
    python evaluate.py --no-vis          # eval only (metrics)
    python evaluate.py --n 32            # visualize 32 images
    python evaluate.py --conf 0.3        # adjust confidence threshold
    python evaluate.py --split val       # run on val split instead of test
"""

import os, sys, argparse, random
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from pathlib import Path
import numpy as np
import torch
import cv2
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Config
WEIGHTS   = "runs/train/ls_yolo_v22/weights/best.pt"
DATA_YAML = "data/landslide.yaml"
IMGSZ     = 512
DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR   = Path("runs/evaluate")

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--weights', default=WEIGHTS)
parser.add_argument('--conf',    type=float, default=0.25)
parser.add_argument('--iou',     type=float, default=0.45)
parser.add_argument('--n',       type=int,   default=16)
parser.add_argument('--no-vis',  action='store_true')
parser.add_argument('--split',   default='test')
opt = parser.parse_args()


# Helpers
def read_tif(path):
    try:
        import tifffile
        data = tifffile.imread(str(path))
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        elif data.ndim == 3 and data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
            data = np.transpose(data, (1, 2, 0))
        data = data[:, :, :3]
    except Exception:
        from PIL import Image
        data = np.array(Image.open(str(path)))
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        elif data.ndim == 3 and data.shape[2] > 3:
            data = data[:, :, :3]
    if data.dtype != np.uint8:
        p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
        data = np.clip((data - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    return data

def read_image(path):
    path = Path(path)
    if path.suffix.lower() in ('.tif', '.tiff'):
        return read_tif(path)
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_gt(label_path, img_shape):
    H, W = img_shape[:2]
    boxes = []
    if not Path(label_path).exists():
        return boxes
    for line in Path(label_path).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        x1 = int((cx - bw/2) * W); y1 = int((cy - bh/2) * H)
        x2 = int((cx + bw/2) * W); y2 = int((cy + bh/2) * H)
        boxes.append({'bbox': [x1, y1, x2, y2], 'cls': cls})
    return boxes


# MAIN — guard prevents workers from re-running this
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    LABEL_DIR = Path(data_cfg['label_dir'])
    IMG_SUBDIR = data_cfg.get('img_subdir', 'img')

    import utils.dataloaders as _dl
    _dl._LABEL_DIR  = str(LABEL_DIR)
    _dl._IMG_SUBDIR = IMG_SUBDIR

    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    from val import run as val_run

    device = select_device(DEVICE)
    model  = DetectMultiBackend(opt.weights, device=device, fp16=False)
    model.eval()
    stride, names = model.stride, model.names
    print(f"Loaded : {opt.weights}")
    print(f"Device : {device}  |  Classes: {names}\n")

    def infer(img_rgb):
        img_lb, _, _ = letterbox(img_rgb, IMGSZ, stride=int(stride), auto=True)
        t = torch.from_numpy(img_lb).permute(2, 0, 1).float().to(device) / 255.0
        t = t.unsqueeze(0)
        with torch.no_grad():
            preds = model(t)
        preds = non_max_suppression(preds, opt.conf, opt.iou, max_det=300)
        det = preds[0]
        results = []
        if len(det):
            det[:, :4] = scale_boxes(t.shape[2:], det[:, :4], img_rgb.shape).round()
            for *xyxy, conf, cls in det:
                results.append({'bbox': [int(v) for v in xyxy],
                                'conf': float(conf), 'cls': int(cls)})
        return results

    def get_label_path(img_path):
        p = Path(img_path)
        parts = p.parts
        try:
            img_idx = parts.index(IMG_SUBDIR)
            loc = parts[img_idx - 1]
        except ValueError:
            loc = p.parent.name
        return LABEL_DIR / loc / (p.stem + '.txt')

    # 1. Quantitative evaluation
    print("=" * 60)
    print(f"  Evaluating [{opt.split}] split  |  conf={opt.conf}  iou={opt.iou}")
    print("=" * 60)

    results = val_run(
        data=DATA_YAML,
        weights=opt.weights,
        imgsz=IMGSZ,
        batch_size=8,
        conf_thres=0.001,
        iou_thres=0.5,
        task=opt.split,
        device=DEVICE,
        plots=False,
        verbose=False,
        workers=0,          # avoid multiprocessing conflicts
    )
    mp, mr, map50, map5095 = results[0], results[1], results[2], results[3]
    speed = results[6]

    print(f"\n{'─'*60}")
    print(f"  {'Precision':<20} {mp:.4f}   ({mp*100:.1f}%)")
    print(f"  {'Recall':<20} {mr:.4f}   ({mr*100:.1f}%)")
    print(f"  {'mAP@0.5':<20} {map50:.4f}   ({map50*100:.2f}%)")
    print(f"  {'mAP@0.5:0.95':<20} {map5095:.4f}   ({map5095*100:.2f}%)")
    print(f"  {'Speed (infer)':<20} {speed[1]:.1f} ms/img")
    print(f"{'─'*60}\n")

    if opt.no_vis:
        sys.exit(0)

    # 2. Load image list
    split_file = data_cfg.get(opt.split)
    if not split_file or not Path(split_file).exists():
        print(f"Split file not found: {split_file}")
        sys.exit(1)

    img_paths = [p.strip() for p in Path(split_file).read_text().splitlines() if p.strip()]
    random.seed(42)
    sample = random.sample(img_paths, min(opt.n, len(img_paths)))
    print(f"Visualizing {len(sample)} images from [{opt.split}] split...")

    # 3. Grid: prediction (red) vs ground truth (green)
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')

    for i, img_path in enumerate(sample):
        img_rgb = read_image(img_path)
        H, W = img_rgb.shape[:2]
        preds = infer(img_rgb)
        gt    = load_gt(get_label_path(img_path), img_rgb.shape)

        ax = axes[i]
        ax.imshow(img_rgb)
        for g in gt:
            x1, y1, x2, y2 = g['bbox']
            ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                         linewidth=1.5, edgecolor='lime', facecolor='none'))
        for d in preds:
            x1, y1, x2, y2 = d['bbox']
            ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                         linewidth=1.5, edgecolor='red', facecolor='none'))
            ax.text(x1, max(y1-3, 8), f"{d['conf']:.2f}", color='red',
                    fontsize=6, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.4, pad=1, linewidth=0))

        stem = Path(img_path).stem
        ax.set_title(f"{stem}\nGT:{len(gt)}  Pred:{len(preds)}",
                     fontsize=7, color='white' if preds else 'yellow', pad=3)
        ax.set_facecolor('#111')

    fig.legend(handles=[
        mpatches.Patch(edgecolor='lime', facecolor='none', label='Ground Truth'),
        mpatches.Patch(edgecolor='red',  facecolor='none', label='Prediction'),
    ], loc='lower center', ncol=2, fontsize=11, framealpha=0.8)
    fig.patch.set_facecolor('#1a1a1a')
    plt.suptitle(
        f"LS-YOLO | {opt.split} | mAP@0.5={map50:.3f}  P={mp:.3f}  R={mr:.3f}",
        color='white', fontsize=13, y=1.001)
    plt.tight_layout()

    out_path = OUT_DIR / f"eval_{opt.split}_n{len(sample)}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Grid saved -> {out_path.resolve()}")

    # 4. Per-image summary
    tp = fp = fn = 0
    for img_path in sample:
        img_rgb = read_image(img_path)
        preds   = infer(img_rgb)
        gt      = load_gt(get_label_path(img_path), img_rgb.shape)
        tp += min(len(preds), len(gt))
        fp += max(0, len(preds) - len(gt))
        fn += max(0, len(gt) - len(preds))

    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    print(f"\n{'─'*60}")
    print(f"  Per-image (approx): TP={tp}  FP={fp}  FN={fn}")
    print(f"  Box-level P={prec:.3f}  R={rec:.3f}")
    print(f"{'─'*60}")
    print(f"\nDone. Results in: {OUT_DIR.resolve()}")
