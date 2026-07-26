"""Run the repo's ORIGINAL val.py for DPU model on both val and test splits.
   workers=0 + main-guard to avoid Windows spawn errors."""
import os, sys
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

    for task in ("val", "test"):
        print(f"\n########## TASK = {task} ##########", flush=True)
        res = run(data=DATA, weights=WEIGHTS, imgsz=512, batch_size=16,
                  conf_thres=0.001, iou_thres=0.6, task=task, device="0",
                  workers=0, plots=False, verbose=False)
        mp, mr, map50, map5095 = res[0], res[1], res[2], res[3]
        print(f"[RESULT {task}] P={mp:.4f} R={mr:.4f} mAP@0.5={map50:.4f} "
              f"({map50*100:.2f}%) mAP@0.5:0.95={map5095:.4f}", flush=True)


if __name__ == "__main__":
    main()
