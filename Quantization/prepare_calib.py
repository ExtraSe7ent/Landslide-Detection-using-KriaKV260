import os, shutil, random, cv2, numpy as np
from pathlib import Path

# ══ CONFIGURATION ═══════════════════════════════════════════════════
DATASET_ROOT = "XXXXXX/Datasets"
OUTPUT       = "XXXXXX/calib_images"
TOTAL        = 1000
FILTER = "UAV"

MIN_WHITE = 5   # Minimum white pixels in mask
# ════════════════════════════════════════════════════════════════════

Path(OUTPUT).mkdir(parents=True, exist_ok=True)
random.seed(42)

# ── Find event folders ───────────────────────────────────────────
event_folders = []
for d in sorted(Path(DATASET_ROOT).iterdir()):
    if not d.is_dir(): continue
    if not (d / "img").exists(): continue
    if not (d / "mask").exists(): continue
    if FILTER and FILTER not in d.name: continue
    event_folders.append(d)

if not event_folders:
    print(f"[ERROR] No folders found (FILTER='{FILTER}')")
    exit(1)

print(f"[INFO] Found {len(event_folders)} event folders (FILTER='{FILTER}'):")
for e in event_folders:
    print(f"  • {e.name}")

# ── Collect images WITH landslide ────────────────────────────────
print("\n[INFO] Scanning for images with landslides...")
all_valid = []

for event in event_folders:
    img_dir  = event / "img"
    mask_dir = event / "mask"
    count = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in ('.tif', '.tiff', '.jpg', '.png'):
            continue
        if img_path.name.startswith('.'):
            continue

        # Find corresponding mask
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            for ext in ('.tif', '.tiff', '.png'):
                alt = mask_dir / (img_path.stem + ext)
                if alt.exists():
                    mask_path = alt
                    break

        if not mask_path.exists():
            continue

        # Check for landslide (mask uses 0/1)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if np.sum(mask > 0) < MIN_WHITE:
            continue

        all_valid.append(img_path)
        count += 1

    print(f"  {event.name}: {count} images with landslide")

print(f"\n[INFO] Total images with landslide: {len(all_valid)}")

if not all_valid:
    print("[ERROR] No images found!")
    exit(1)

# ── Sample TOTAL images randomly ─────────────────────────────────
random.shuffle(all_valid)
sampled = all_valid[:TOTAL]

print(f"[INFO] Copying {len(sampled)} images to:\n  {OUTPUT}")
copied = 0
for img_path in sampled:
    event_name = img_path.parent.parent.name
    out_name = (f"{event_name}_{img_path.name}"
                .replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("（", "").replace("）", ""))
    shutil.copy(img_path, Path(OUTPUT) / out_name)
    copied += 1
    if copied % 100 == 0:
        print(f"  [{copied}/{len(sampled)}]...")

print(f"\n[OK] Copied {copied} images WITH landslide")
print(f"[INFO] Next step: copy calib_images folder to the KV260 directory and run quantize_calib.py")
