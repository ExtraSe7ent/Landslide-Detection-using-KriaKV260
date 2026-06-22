"""
convert_labels.py
Convert all .tif labels to YOLO .txt (tight bounding box)
"""

import os
import glob
import random
import numpy as np
import rasterio
import cv2
from tqdm import tqdm

# CONFIG
DATASET_ROOT  = os.environ.get("LS_IMAGES", "datasets/images")
OUTPUT_ROOT   = os.environ.get("LS_LABELS_OUT", "datasets/labels_yolo")
PREVIEW_DIR   = os.path.join(OUTPUT_ROOT, "_preview")
MIN_AREA      = 100     # pixel² — ignore contours smaller than this (noise)
N_PREVIEW     = 5       # number of random images to preview per folder
IMG_SIZE      = 512     # image size (used for normalization)

os.makedirs(PREVIEW_DIR, exist_ok=True)


def mask_to_yolo_boxes(mask_path):
    """
    Read .tif label -> return list of (cx, cy, w, h) normalized [0,1].
    Use THRESH_BINARY_INV because mask is inverted: 255=background, 0=landslide.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)
        H, W = mask.shape

    # INV: pixel=0 (landslide) -> 255 after threshold
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw = w / W
        nh = h / H
        # Clamp to [0, 1] to avoid floating point edge cases
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        boxes.append((cx, cy, nw, nh))
    return boxes


def make_preview(img_path, mask_path, boxes, out_path):
    """
    Create 2-panel preview image: original | original + tight bboxes.
    """
    with rasterio.open(img_path) as src:
        img = np.stack([src.read(i) for i in range(1, src.count + 1)],
                       axis=-1)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    panel_bbox = img_bgr.copy()
    for (cx, cy, nw, nh) in boxes:
        x1 = int((cx - nw / 2) * W)
        y1 = int((cy - nh / 2) * H)
        x2 = int((cx + nw / 2) * W)
        y2 = int((cy + nh / 2) * H)
        cv2.rectangle(panel_bbox, (x1, y1), (x2, y2), (0, 255, 80), 2)
        cv2.rectangle(panel_bbox, (x1, y1 - 14), (x1 + 70, y1),
                      (0, 255, 80), -1)
        cv2.putText(panel_bbox, "slide",
                    (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    def title(im, txt):
        out = im.copy()
        cv2.rectangle(out, (0, 0), (W, 26), (20, 20, 20), -1)
        cv2.putText(out, txt, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out

    p1 = title(img_bgr,  "Original")
    p2 = title(panel_bbox, f"BBoxes ({len(boxes)})")
    combined = np.hstack([p1, p2])
    cv2.imwrite(out_path, combined)


# Main loop
total_imgs   = 0
total_boxes  = 0
total_empty  = 0   # images with no landslides (empty .txt file)
folder_stats = []

locations = sorted(os.listdir(DATASET_ROOT))

for location in locations:
    img_dir   = os.path.join(DATASET_ROOT, location, "img")
    label_dir = os.path.join(DATASET_ROOT, location, "label")
    out_dir   = os.path.join(OUTPUT_ROOT, location)

    if not os.path.isdir(label_dir) or not os.path.isdir(img_dir):
        continue

    os.makedirs(out_dir, exist_ok=True)

    # Find all label files (tif / TIF)
    label_files = (glob.glob(os.path.join(label_dir, "*.tif")) +
                   glob.glob(os.path.join(label_dir, "*.TIF")))

    loc_boxes = 0
    loc_empty = 0
    processed = 0

    for lf in tqdm(label_files, desc=f"{location:<35}", ncols=80):
        stem     = os.path.splitext(os.path.basename(lf))[0]
        out_txt  = os.path.join(out_dir, stem + ".txt")

        boxes = mask_to_yolo_boxes(lf)

        with open(out_txt, "w") as f:
            for (cx, cy, nw, nh) in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        loc_boxes += len(boxes)
        if len(boxes) == 0:
            loc_empty += 1
        processed += 1

    total_imgs  += processed
    total_boxes += loc_boxes
    total_empty += loc_empty
    folder_stats.append((location, processed, loc_boxes, loc_empty))

    # Preview: take N_PREVIEW images with at least 1 bbox
    label_with_boxes = [
        lf for lf in label_files
        if len(mask_to_yolo_boxes(lf)) > 0
    ]
    preview_samples = random.sample(
        label_with_boxes, min(N_PREVIEW, len(label_with_boxes))
    )

    preview_panels = []
    for lf in preview_samples:
        stem    = os.path.splitext(os.path.basename(lf))[0]
        # Find corresponding original image (tif / TIF)
        img_candidates = (
            glob.glob(os.path.join(img_dir, stem + ".tif")) +
            glob.glob(os.path.join(img_dir, stem + ".TIF"))
        )
        if not img_candidates:
            continue
        boxes = mask_to_yolo_boxes(lf)
        tmp   = os.path.join(PREVIEW_DIR, f"_tmp_{stem}.jpg")
        make_preview(img_candidates[0], lf, boxes, tmp)
        preview_panels.append(cv2.imread(tmp))
        os.remove(tmp)

    if preview_panels:
        # Vertically stack all preview samples
        combined_preview = np.vstack(preview_panels)
        loc_clean = location.replace(" ", "_").replace("（", "_").replace("）", "_")
        preview_out = os.path.join(PREVIEW_DIR, f"{loc_clean}.jpg")
        cv2.imwrite(preview_out, combined_preview)


# Report
print()
print("=" * 65)
print("  CONVERT COMPLETE")
print("=" * 65)
print(f"{'Location':<35} {'Imgs':>6} {'Boxes':>7} {'Empty':>6}")
print("─" * 65)
for loc, imgs, boxes, empty in folder_stats:
    print(f"{loc:<35} {imgs:>6,} {boxes:>7,} {empty:>6,}")
print("─" * 65)
print(f"{'TOTAL':<35} {total_imgs:>6,} {total_boxes:>7,} {total_empty:>6,}")
print("=" * 65)
print(f"\nOutput labels : {OUTPUT_ROOT}")
print(f"Preview images: {PREVIEW_DIR}")
print(f"\n-> Check preview before training!")