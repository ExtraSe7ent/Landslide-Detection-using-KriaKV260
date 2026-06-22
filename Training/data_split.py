import os
import random
from pathlib import Path

DATASET_ROOT = os.environ.get("LS_IMAGES", "datasets/images")
OUTPUT_DIR = os.environ.get("LS_SPLITS", "datasets/splits")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

uav_imgs, sat_imgs = [], []

for location in Path(DATASET_ROOT).iterdir():
    if not location.is_dir():
        continue
    img_dir = location / "img"
    if not img_dir.exists():
        continue
    images = list(img_dir.glob("*.tif")) + list(img_dir.glob("*.TIF"))
    name = location.name.upper()
    if "SAT" in name or "SENTINEL" in name or "PLANET" in name:
        sat_imgs.extend(images)
    else:
        uav_imgs.extend(images)

print(f"UAV images: {len(uav_imgs)}")
print(f"SAT images: {len(sat_imgs)}")
print(f"Total: {len(uav_imgs) + len(sat_imgs)}")

def split(items):
    random.shuffle(items)
    n = len(items)
    t = int(n * TRAIN_RATIO)
    v = int(n * VAL_RATIO)
    return items[:t], items[t:t+v], items[t+v:]

uav_train, uav_val, uav_test = split(uav_imgs)
sat_train, sat_val, sat_test = split(sat_imgs)

def write_split(name, items):
    path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for p in items:
            f.write(str(p) + "\n")
    print(f"{name}: {len(items)} images")

write_split("train",    uav_train + sat_train)
write_split("val",      uav_val   + sat_val)
write_split("test",     uav_test  + sat_test)
write_split("test_uav", uav_test)
write_split("test_sat", sat_test)

print("\nDone!")