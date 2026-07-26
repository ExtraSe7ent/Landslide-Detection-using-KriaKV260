import os, sys, time
import cv2, numpy as np, torch
import torchvision
import vart, xir
from pathlib import Path

# Need to add LS-YOLO to path to use letterbox
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "LS-YOLO"))
try:
    from utils.dataloaders import letterbox
except ImportError:
    # If import fails, create local letterbox
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)


# ══ CONFIGURATION ════════════════════════════════════════════════════════
MODEL      = os.path.join(CURRENT_DIR, "ls_yolo_landslide.xmodel")
IMG_DIR    = "/workspace/test_images"
LBL_DIR    = "/workspace/test_labels"
DELAY_MS   = 100
SHOW_DISPLAY = os.environ.get("DISPLAY") is not None

# Fix 1: Split conf into 2 thresholds
CONF_THRES_DISPLAY = 0.25    # draw box on screen
CONF_THRES_MAP     = 0.001   # calculate mAP — standard YOLO val.py

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

_GRID_CACHE = {}
_ANCHOR_CACHE = {}
def get_grid_and_anchor(W, H, i):
    key = (W, H, i)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = make_grid(W, H)
        _ANCHOR_CACHE[key] = torch.tensor(ANCHORS[i]).float().view(1, NA, 1, 1, 2)
    return _GRID_CACHE[key], _ANCHOR_CACHE[key]

def decode_decoupled(raws):
    z = []
    for i, raw in enumerate(raws):
        bs, _, H, W = raw.shape
        reg  = raw[:, :4 * NA,          :, :]
        conf = raw[:, 4*NA:(4+1)*NA,    :, :]
        cls  = raw[:, (4+1)*NA:,        :, :]

        reg  = reg.view( bs, NA, 4,  H, W).permute(0, 1, 3, 4, 2)
        conf = conf.view(bs, NA, 1,  H, W).permute(0, 1, 3, 4, 2)
        cls  = cls.view( bs, NA, NC, H, W).permute(0, 1, 3, 4, 2)

        y = torch.cat([reg, conf, cls], dim=-1).sigmoid()
        g, a = get_grid_and_anchor(W, H, i)
        y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + g) * STRIDES[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * a
        z.append(y.view(bs, -1, 5 + NC))
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
        
        # USE FAST TORCHVISION NMS
        keep = torchvision.ops.nms(x[:,:4], x[:,4], IOU_THRES)
        out.append(x[keep])
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
            iou_row = iou[di].clone()
            iou_row[used] = -1.0       # exclude used GT
            best_iou, best_j = iou_row.max(0)

            if best_iou >= iou_thresh:
                tp_list.append(1)
                used[best_j] = True
            else:
                tp_list.append(0)
            conf_list.append(preds_s[di, 4].item())

    if not conf_list:
        return 0.0, 0.0, 0.0

    order  = np.argsort(conf_list)[::-1]
    tp_arr = np.array(tp_list)[order]
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(1 - tp_arr)
    prec   = cum_tp / (cum_tp + cum_fp + 1e-8)
    rec    = cum_tp / (n_gt + 1e-8)
    ap     = compute_ap(rec, prec)

    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best = f1.argmax()
    return float(prec[best]), float(rec[best]), ap


# ── Main ─────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL):
        print(f"[ERROR] Not found: {MODEL}")
        sys.exit(1)

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

    exts  = ('.jpg','.jpeg','.png')
    if not os.path.exists(IMG_DIR):
        print(f"[ERROR] Directory {IMG_DIR} does not exist!")
        sys.exit(1)

    imgs  = sorted([f for f in os.listdir(IMG_DIR)
                    if f.lower().endswith(exts) and not f.startswith(".")])
    valid = [(f, Path(f).stem+".txt") for f in imgs
             if os.path.exists(os.path.join(LBL_DIR, Path(f).stem+".txt"))]

    print(f"[INFO] {len(valid)} images with labels → start eval")
    print(f"[INFO] CONF display={CONF_THRES_DISPLAY} | CONF mAP={CONF_THRES_MAP}")
    print("[INFO] Press Q to stop early\n")

    if SHOW_DISPLAY:
        cv2.namedWindow("LS-YOLO Eval", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("LS-YOLO Eval", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    all_preds_map  = []   
    all_labels     = []
    
    # Table 1: Pipeline Stage Breakdown
    times_read     = []
    times_resize   = []
    times_norm     = []
    times_dpu      = []
    times_decode   = []
    times_nms      = []
    times_draw     = []
    stopped_at     = len(valid)

    for idx, (img_f, lbl_f) in enumerate(valid):
        # 1. Image acquisition
        t_start = time.time()
        frame = cv2.imread(os.path.join(IMG_DIR, img_f))
        if frame is None: continue
        oh, ow = frame.shape[:2]
        t_read = (time.time() - t_start) * 1000

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

        # 2. Resize + Letterbox
        t_start = time.time()
        img_pad, _, _ = letterbox(frame, (h, w), auto=False)
        img = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
        t_resize = (time.time() - t_start) * 1000

        # 3. Tensor normalization
        t_start = time.time()
        idata[0][0,...] = (img.astype(np.float32)/255.0*in_sc).astype(np.int8)
        t_norm = (time.time() - t_start) * 1000

        # 4. DMA transfer + DPU inference
        t_start = time.time()
        jid = runner.execute_async(idata, odata)
        runner.wait(jid)
        t_dpu_dma = (time.time() - t_start) * 1000

        # 5. Decode output
        t_start = time.time()
        raws = sorted(
            [torch.from_numpy((o.astype(np.float32)*np.float32(out_scs[k]))
                              .transpose(0,3,1,2).copy())
             for k,o in enumerate(odata)],
            key=lambda t: t.shape[2], reverse=True)
        decoded = decode_decoupled(raws)
        t_decode = (time.time() - t_start) * 1000

        # 6. NMS
        t_start = time.time()
        preds_map = nms(decoded, CONF_THRES_MAP)[0]
        if len(preds_map):
            preds_map[:,:4] = scale_boxes((h,w), preds_map[:,:4], (oh,ow,3)).round()

        preds_disp = preds_map[preds_map[:, 4] >= CONF_THRES_DISPLAY] if len(preds_map) else preds_map
        t_nms = (time.time() - t_start) * 1000

        times_read.append(t_read)
        times_resize.append(t_resize)
        times_norm.append(t_norm)
        times_dpu.append(t_dpu_dma)
        times_decode.append(t_decode)
        times_nms.append(t_nms)
        all_preds_map.append(preds_map)
        all_labels.append(gt_t)

        # 7. Bounding Box drawing
        t_start = time.time()
        disp     = frame.copy()
        n_det = 0
        n_gt_img = len(gt_t)
        for row in gt_t:
            cls, x1, y1, x2, y2 = row
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)

        if len(preds_disp):
            n_det = len(preds_disp)
            for *xy, conf, cls in preds_disp:
                x1,y1,x2,y2 = map(int,xy)
                cv2.rectangle(disp,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(disp, f"landslide {conf:.2f}",
                            (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,(0,0,255),2)
        t_draw = (time.time() - t_start) * 1000
        times_draw.append(t_draw)

        core_t = t_resize + t_norm + t_dpu_dma + t_decode + t_nms
        fps_core = 1000 / (core_t + 1e-6)
        fps_e2e  = 1000 / (t_read + core_t + t_draw + 1e-6)
        
        cv2.rectangle(disp,(0,0),(ow,90),(0,0,0),-1)
        cv2.putText(disp, f"[{idx+1}/{len(valid)}] {img_f}",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(255,255,255),2)
        cv2.putText(disp,
                    f"Core FPS:{fps_core:.1f}  E2E FPS:{fps_e2e:.1f}  DPU:{t_dpu_dma:.0f}ms",
                    (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0,255,255),2)
        cv2.putText(disp,
                    f"GT:{n_gt_img}  Det:{n_det}  (display conf>={CONF_THRES_DISPLAY})",
                    (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0,255,255),2)
        cv2.putText(disp,"GT", (ow-120,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(disp,"Pred", (ow-70, 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        print(f"[{idx+1:4d}/{len(valid)}] {img_f:40s} "
              f"Read:{t_read:4.1f} Rsz:{t_resize:4.1f} Nrm:{t_norm:4.1f} "
              f"DPU:{t_dpu_dma:5.1f} Dcd:{t_decode:4.1f} NMS:{t_nms:4.1f} Drw:{t_draw:4.1f} "
              f"GT:{n_gt_img} Det:{n_det}")

        if SHOW_DISPLAY:
            cv2.imshow("LS-YOLO Eval", disp)
            key = cv2.waitKey(DELAY_MS) & 0xFF
            if key == ord('q'):
                stopped_at = idx+1
                print(f"\n[INFO] Stopped early at image {stopped_at}")
                break

    if SHOW_DISPLAY:
        cv2.destroyAllWindows()

    if not all_preds_map:
        return

    # Calculate metrics
    print("\n[INFO] Calculating metrics...")
    p50, r50, ap50 = compute_metrics(all_preds_map, all_labels, iou_thresh=0.5)

    ap_list = []
    for thr in np.arange(0.5, 1.0, 0.05):
        _, _, ap = compute_metrics(all_preds_map, all_labels, iou_thresh=float(thr))
        ap_list.append(ap)
    map5095 = float(np.mean(ap_list))

    avg_read   = float(np.mean(times_read))
    avg_resize = float(np.mean(times_resize))
    avg_norm   = float(np.mean(times_norm))
    avg_dpu    = float(np.mean(times_dpu))
    avg_decode = float(np.mean(times_decode))
    avg_nms    = float(np.mean(times_nms))
    avg_draw   = float(np.mean(times_draw))
    
    core_time  = avg_resize + avg_norm + avg_dpu + avg_decode + avg_nms
    total_time = avg_read + core_time + avg_draw
    fps_core   = 1000 / core_time
    fps_e2e    = 1000 / total_time

    print("\n" + "="*60)
    print(" EVALUATION RESULTS ON KV260 (DPU INT8)")
    print("="*60)
    print(f" Number of tested images         : {stopped_at}")
    print(f" Conf mAP threshold     : {CONF_THRES_MAP}")
    print(f" Conf display thresh    : {CONF_THRES_DISPLAY}")
    print(f" IoU threshold          : {IOU_THRES}")
    print("-" * 60)
    print(f" Precision              : {p50:.4f}  ({p50*100:.2f}%)")
    print(f" Recall                 : {r50:.4f}  ({r50*100:.2f}%)")
    print(f" mAP@0.5                : {ap50:.4f}  ({ap50*100:.2f}%)")
    print(f" mAP@0.5:0.95           : {map5095:.4f}  ({map5095*100:.2f}%)")
    print("-" * 60)
    print(f" Image acquisition      : {avg_read:5.2f} ms ({avg_read/total_time*100:5.1f}%)")
    print(f" Resize + Letterbox     : {avg_resize:5.2f} ms ({avg_resize/total_time*100:5.1f}%)")
    print(f" Tensor normalization   : {avg_norm:5.2f} ms ({avg_norm/total_time*100:5.1f}%)")
    print(f" DPU + DMA              : {avg_dpu:5.2f} ms ({avg_dpu/total_time*100:5.1f}%)")
    print(f" Output Decode          : {avg_decode:5.2f} ms ({avg_decode/total_time*100:5.1f}%)")
    print(f" NMS                    : {avg_nms:5.2f} ms ({avg_nms/total_time*100:5.1f}%)")
    print(f" Draw BBox + Disp       : {avg_draw:5.2f} ms ({avg_draw/total_time*100:5.1f}%)")
    print(f" Total Latency          : {total_time:5.2f} ms (100.0%)")
    print("-" * 60)
    print(f" Core FPS               : {fps_core:.1f} FPS")
    print(f" End-to-End FPS         : {fps_e2e:.1f} FPS")
    print("="*60)

if __name__ == "__main__":
    main()
