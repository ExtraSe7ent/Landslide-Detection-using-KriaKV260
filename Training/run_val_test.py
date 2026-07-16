"""Chạy val.py CHỈ task=test, bắt full traceback."""
import os, sys, traceback
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

DATA = "data/landslide_uav_finetune.yaml"
WEIGHTS = "D:/Training/runs_dpu/ls_yolo_dpu_uav/weights/best.pt"


def main():
    import yaml
    import utils.dataloaders as _dl
    from val import run
    with open(DATA) as f:
        cfg = yaml.safe_load(f)
    _dl._LABEL_DIR = cfg["label_dir"]
    _dl._IMG_SUBDIR = cfg.get("img_subdir", "img")
    try:
        res = run(data=DATA, weights=WEIGHTS, imgsz=512, batch_size=16,
                  conf_thres=0.001, iou_thres=0.6, task="test", device="0",
                  workers=0, plots=False, verbose=False)
        print(f"\n[RESULT test] P={res[0]:.4f} R={res[1]:.4f} "
              f"mAP@0.5={res[2]:.4f} ({res[2]*100:.2f}%) mAP@0.5:0.95={res[3]:.4f}", flush=True)
    except Exception:
        print("\n=== FULL TRACEBACK ===", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
