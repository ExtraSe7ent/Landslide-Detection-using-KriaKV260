"""
eval_display_kv260.py
Realtime display + P/R/mAP evaluation on KV260.
"""
import os, sys, time
import cv2, numpy as np, torch
import vart, xir
from pathlib import Path

# ══ CONFIGURATION ════════════════════════════════════════════════════════
MODEL      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ls_yolo_landslide.xmodel")
IMG_DIR    = "XXXXXX/test_images"
LBL_DIR    = "XXXXXX/test_labels"
DELAY_MS   = 1000

# Fix 1: Separate conf into 2 thresholds
CONF_THRES_DISPLAY = 0.25    
CONF_THRES_MAP     = 0.001   

IOU_THRES  = 0.45
NC, NA     = 1, 3
STRIDES    = [8, 16, 32]
ANCHORS    = [
    [10,13,  16,30,  33,23],
    [30,61,  62,45,  59,119],
    [116,90, 156,198, 373,326],
]
# ════════════════════════════════════════════════════════════════════


# ── Decode ──────────────────────────────────────────────────────────
def make_grid(nx, ny):
    yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
    return torch.stack((xv, yv), 2).view(1,1,ny,nx,2).float()

def decode(raws):
    z = []
    for i, raw in enumerate(raws):
        bs, _, H, W = raw.shape
        r = raw[:, :4*NA    ].view(bs,NA,4, H,W).permute(0,1,3,4,2)
        c = raw[:, 4*NA:5*NA].view(bs,NA,1, H,W).permute(0,1,3,4,2)
        l = raw[:, 5*NA:    ].view(bs,NA,NC,H,W).permute(0,1,3,4,2)
        y = torch.cat([r,c,l], -1).sigmoid()
        g = make_grid(W, H)
        a = torch.tensor(ANCHORS[i]).float().view(1,NA,1,1,2)
        y[...,0:2] = (y[...,0:2]*2 - 0.5 + g) * STRIDES[i]
        y[...,2:4] = (y[...,2:4]*2)**2 * a
        z.append(y.view(bs,-1,5+NC))
    return torch.cat(z, 1)

def xywh2xyxy(x):
    y = x.clone()
    y[...,0] = x[...,0] - x[...,2]/2
    y[...,1] = x[...,1] - x[...,3]/2
    y[...,2] = x[...,0] + x[...,2]/2
    y[...,3] = x[...,1] + x[...,3]/2
    return y

def box_iou(b1, b2):
    def area(b): return (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    inter = (torch.min(b1[:,None,2:], b2[:,2:]) -
             torch.max(b1[:,None,:2], b2[:,:2])).clamp(0).prod(2)
    return inter / (area(b1)[:,None] + area(b2) - inter + 1e-8)

def nms(pred, conf_thres):
    """NMS with variable conf_thres — different for display and mAP."""
    out = []
    for x in pred:
        x = x[x[:,4] > conf_thres]
        if not len(x): out.append(torch.zeros((0,6))); continue
        x[:,5:] *= x[:,4:5]
        box = xywh2xyxy(x[:,:4])
        conf, cls = x[:,5:].max(1, keepdim=True)
        x = torch.cat((box, conf, cls.float()), 1)
        x = x[x[:,4] > conf_thres]
        if not len(x): out.append(torch.zeros((0,6))); continue
        keep, idxs = [], x[:,4].argsort(descending=True)
        while idxs.numel():
            i = idxs[0]; keep.append(i)
            if idxs.numel() == 1: break
            iou = box_iou(x[i:i+1,:4], x[idxs[1:],:4])[0]
            idxs = idxs[1:][iou <= IOU_THRES]
        out.append(x[torch.stack(keep)])
    return out

def scale_boxes(src, boxes, dst):
    g = min(src[0]/dst[0], src[1]/dst[1])
    px = (src[1] - dst[1]*g) / 2
    py = (src[0] - dst[0]*g) / 2
    boxes[:,[0,2]] -= px; boxes[:,[1,3]] -= py
    boxes[:,:4] /= g
    boxes[:,0].clamp_(0, dst[1]); boxes[:,1].clamp_(0, dst[0])
    boxes[:,2].clamp_(0, dst[1]); boxes[:,3].clamp_(0, dst[0])
    return boxes


# ── mAP ─────────────────────────────────────────────────────────────
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

def compute_metrics(all_preds, all_labels, iou_thresh=0.5):
    """
    Compute P/R/AP using COCO standard:
    Fix 1: use preds filtered with CONF_THRES_MAP (0.001) — not 0.25
    Fix 2: sort by confidence descending before matching
    Fix 3: match unused GT (not global best GT)
    """
    tp_list, conf_list, n_gt = [], [], 0

    for preds, labels in zip(all_preds, all_labels):
        n_gt += len(labels)

        if not len(preds):
            continue

        if not len(labels):
            tp_list.extend([0] * len(preds))
            conf_list.extend(preds[:,4].tolist())
            continue

        # Sort predictions by confidence descending
        order   = preds[:,4].argsort(descending=True)
        preds_s = preds[order]

        gt   = labels[:,1:].float()
        det  = preds_s[:,:4].float()
        iou  = box_iou(det, gt)        # [N, M]
        used = torch.zeros(len(labels), dtype=torch.bool)

        for di in range(len(preds_s)):
            # Fix 3: Only consider unused GT, find best GT among them
            iou_row = iou[di].clone()
            iou_row[used] = -1.0       # Ignore used GT
            best_iou, best_j = iou_row.max(0)

            if best_iou >= iou_thresh:
                tp_list.append(1)
                used[best_j] = True
            else:
                tp_list.append(0)
            conf_list.append(preds_s[di, 4].item())

    if not conf_list:
        return 0.0, 0.0, 0.0

    # Sort all by confidence to plot P/R curve
    order  = np.argsort(conf_list)[::-1]
    tp_arr = np.array(tp_list)[order]
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(1 - tp_arr)
    prec   = cum_tp / (cum_tp + cum_fp + 1e-8)
    rec    = cum_tp / (n_gt + 1e-8)
    ap     = compute_ap(rec, prec)

    # P/R at F1-max (comparable to val.py)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = f1.argmax()
    return float(prec[best]), float(rec[best]), ap


# ── Main ─────────────────────────────────────────────────────────────
def main():
    # Init DPU
    graph = xir.Graph.deserialize(MODEL)
    sg    = [c for c in graph.get_root_subgraph().toposort_child_subgraph()
             if c.has_attr("device") and c.get_attr("device").upper() == "DPU"]
    if not sg:
        print("[ERROR] No DPU subgraph!"); sys.exit(1)

    runner  = vart.Runner.create_runner(sg[0], "run")
    it, ot  = runner.get_input_tensors(), runner.get_output_tensors()
    _, h, w, _ = it[0].dims
    in_sc   = 2 ** it[0].get_attr("fix_point")
    out_scs = [2 ** -t.get_attr("fix_point") for t in ot]
    idata   = [np.empty(tuple(it[0].dims), dtype=np.int8, order="C")]
    odata   = [np.empty(tuple(t.dims),     dtype=np.int8, order="C") for t in ot]

    print(f"[INFO] DPU input : {tuple(it[0].dims)}")
    for k, t in enumerate(ot):
        print(f"[INFO] DPU output[{k}]: {tuple(t.dims)}")

    # Get list of images with labels
    exts  = ('.jpg','.jpeg','.png')
    imgs  = sorted([f for f in os.listdir(IMG_DIR)
                    if f.lower().endswith(exts) and not f.startswith(".")])
    valid = [(f, Path(f).stem+".txt") for f in imgs
             if os.path.exists(os.path.join(LBL_DIR, Path(f).stem+".txt"))]

    print(f"[INFO] {len(valid)} images with labels → starting eval")
    print(f"[INFO] CONF display={CONF_THRES_DISPLAY} | CONF mAP={CONF_THRES_MAP}")
    print("[INFO] Press Q to stop early\n")

    cv2.namedWindow("LS-YOLO Eval", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("LS-YOLO Eval", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    all_preds_map  = []   # preds with CONF_THRES_MAP  — for mAP calculation
    all_labels     = []
    times_pre      = []   # CPU preprocessing
    times_dpu      = []   # Pure DPU
    times_cpu      = []   # CPU decode + NMS
    stopped_at     = len(valid)

    for idx, (img_f, lbl_f) in enumerate(valid):
        frame = cv2.imread(os.path.join(IMG_DIR, img_f))
        if frame is None: continue
        oh, ow = frame.shape[:2]

        # Read label → pixel xyxy
        gt_boxes = []
        lpath = os.path.join(LBL_DIR, lbl_f)
        for line in open(lpath, errors="ignore").read().strip().splitlines():
            parts = line.split()
            if len(parts) < 5: continue
            cls, cx, cy, bw, bh = map(float, parts[:5])
            x1=(cx-bw/2)*ow; y1=(cy-bh/2)*oh
            x2=(cx+bw/2)*ow; y2=(cy+bh/2)*oh
            gt_boxes.append([cls, x1, y1, x2, y2])
        gt_t = torch.tensor(gt_boxes) if gt_boxes else torch.zeros((0,5))

        # Fix 2: CPU preprocessing — measured separately, not included in DPU time
        t_pre_start = time.time()
        img = cv2.cvtColor(cv2.resize(frame,(w,h)), cv2.COLOR_BGR2RGB)
        idata[0][0,...] = (img.astype(np.float32)/255.0*in_sc).astype(np.int8)
        t_pre = (time.time() - t_pre_start) * 1000

        # Pure DPU — start measuring from execute_async
        t_dpu_start = time.time()
        jid = runner.execute_async(idata, odata)
        runner.wait(jid)
        t_dpu = (time.time() - t_dpu_start) * 1000

        # Decode + NMS — measured separately
        t1 = time.time()
        raws = sorted(
            [torch.from_numpy((o.astype(np.float32)*np.float32(out_scs[k]))
                              .transpose(0,3,1,2).copy())
             for k,o in enumerate(odata)],
            key=lambda t: t.shape[2], reverse=True)
        decoded = decode(raws)

        # Preds for mAP (low conf, integral of full P/R curve)
        preds_map = nms(decoded, CONF_THRES_MAP)[0]
        if len(preds_map):
            preds_map[:,:4] = scale_boxes((h,w), preds_map[:,:4], (oh,ow,3)).round()

        # Preds for display (high conf, only show confident boxes)
        preds_disp = nms(decoded, CONF_THRES_DISPLAY)[0]
        if len(preds_disp):
            preds_disp[:,:4] = scale_boxes((h,w), preds_disp[:,:4], (oh,ow,3)).round()

        t_cpu = (time.time() - t1) * 1000

        times_pre.append(t_pre)
        times_dpu.append(t_dpu)
        times_cpu.append(t_cpu)
        all_preds_map.append(preds_map)
        all_labels.append(gt_t)

        # Display
        disp     = frame.copy()
        n_det    = 0
        n_gt_img = len(gt_t)

        # Ground truth — green
        for row in gt_t:
            x1,y1,x2,y2 = map(int, row[1:])
            cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)

        # Prediction — red (uses CONF_THRES_DISPLAY)
        if len(preds_disp):
            n_det = len(preds_disp)
            for *xy, conf, cls in preds_disp:
                x1,y1,x2,y2 = map(int,xy)
                cv2.rectangle(disp,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(disp, f"landslide {conf:.2f}",
                            (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,(0,0,255),2)

        fps = 1000/(t_pre+t_dpu+t_cpu+1e-6)
        cv2.rectangle(disp,(0,0),(ow,90),(0,0,0),-1)
        cv2.putText(disp, f"[{idx+1}/{len(valid)}] {img_f}",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(255,255,255),2)
        cv2.putText(disp,
                    f"FPS:{fps:.1f}  Pre:{t_pre:.0f}ms  DPU:{t_dpu:.0f}ms  CPU:{t_cpu:.0f}ms",
                    (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0,255,255),2)
        cv2.putText(disp,
                    f"GT:{n_gt_img}  Det:{n_det}  (display conf≥{CONF_THRES_DISPLAY})",
                    (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0,255,255),2)
        cv2.putText(disp,"GT",   (ow-120,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(disp,"Pred", (ow-70, 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        print(f"[{idx+1:4d}/{len(valid)}] {img_f:40s} "
              f"Pre:{t_pre:4.1f}ms DPU:{t_dpu:5.1f}ms CPU:{t_cpu:4.1f}ms "
              f"GT:{n_gt_img} Det:{n_det}")

        cv2.imshow("LS-YOLO Eval", disp)
        key = cv2.waitKey(DELAY_MS) & 0xFF
        if key == ord('q'):
            stopped_at = idx+1
            print(f"\n[INFO] Stopped early at image {stopped_at}")
            break

    cv2.destroyAllWindows()

    # Calculate metrics
    print("\n[INFO] Calculating metrics...")
    p50, r50, ap50 = compute_metrics(all_preds_map, all_labels, iou_thresh=0.5)

    ap_list = []
    for thr in np.arange(0.5, 1.0, 0.05):
        _, _, ap = compute_metrics(all_preds_map, all_labels, iou_thresh=float(thr))
        ap_list.append(ap)
    map5095 = float(np.mean(ap_list))

    t_pre_mean = float(np.mean(times_pre))
    t_dpu_mean = float(np.mean(times_dpu))
    t_cpu_mean = float(np.mean(times_cpu))
    fps_mean   = 1000 / (t_pre_mean + t_dpu_mean + t_cpu_mean)

    print("\n" + "="*60)
    print("  KV260 EVALUATION RESULTS  (DPU INT8)")
    print("="*60)
    print(f"  Images tested        : {stopped_at}")
    print(f"  Conf mAP threshold   : {CONF_THRES_MAP}  (YOLO standard)")
    print(f"  Conf display thresh  : {CONF_THRES_DISPLAY}")
    print(f"  IoU threshold        : {IOU_THRES}")
    print("-"*60)
    print(f"  Precision (P@F1max)  : {p50:.4f}   ({p50*100:.2f}%)")
    print(f"  Recall    (R@F1max)  : {r50:.4f}   ({r50*100:.2f}%)")
    print(f"  mAP@0.5              : {ap50:.4f}   ({ap50*100:.2f}%)")
    print(f"  mAP@0.5:0.95         : {map5095:.4f}   ({map5095*100:.2f}%)")
    print("-"*60)
    print(f"  Preprocess CPU       : {t_pre_mean:.1f} ms")
    print(f"  Pure DPU             : {t_dpu_mean:.1f} ms")
    print(f"  Decode+NMS CPU       : {t_cpu_mean:.1f} ms")
    print(f"  Average FPS          : {fps_mean:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
