import os, sys, time
import cv2, numpy as np, torch
import vart, xir

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "LS-YOLO"))
from utils.general import non_max_suppression, scale_boxes

# Need to add LS-YOLO to path to use letterbox
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

MODEL = os.path.join(CURRENT_DIR, "ls_yolo_landslide.xmodel")

# ══ Parameters from models/landslide/Improve.yaml ══
NC      = 1    # nc: 1
NA      = 3    # 3 anchors per scale
STRIDES = [8, 16, 32]
ANCHORS = [
    [10, 13,  16, 30,  33, 23],        # P3 stride 8
    [30, 61,  62, 45,  59, 119],       # P4 stride 16
    [116, 90, 156, 198, 373, 326],     # P5 stride 32
]


def make_grid(nx, ny):
    yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
    return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()

_GRID_CACHE = {}
_ANCHOR_CACHE = {}
def get_grid_and_anchor(W, H, i):
    key = (W, H, i)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = make_grid(W, H)
        _ANCHOR_CACHE[key] = torch.tensor(ANCHORS[i]).float().view(1, NA, 1, 1, 2)
    return _GRID_CACHE[key], _ANCHOR_CACHE[key]


def decode_decoupled(raw_list):
    """
    Decode output of FullDPU (from Decoupled_Detect).

    Channel layout of each tensor: [4*NA reg | 1*NA conf | NC*NA cls]
    Example nc=1, na=3: [12 reg | 3 conf | 3 cls] = 18 channels

    Returns: tensor [1, N_total, 5+NC] — standard for non_max_suppression
    """
    z = []
    for i, raw in enumerate(raw_list):
        bs, _, H, W = raw.shape

        # Split channels by semantic
        reg  = raw[:, :4 * NA,          :, :]   # [bs, 12, H, W]
        conf = raw[:, 4*NA:(4+1)*NA,    :, :]   # [bs,  3, H, W]
        cls  = raw[:, (4+1)*NA:,        :, :]   # [bs,  3, H, W]

        # Reshape to per-anchor [bs, NA, H, W, dim]
        reg  = reg.view( bs, NA, 4,  H, W).permute(0, 1, 3, 4, 2)
        conf = conf.view(bs, NA, 1,  H, W).permute(0, 1, 3, 4, 2)
        cls  = cls.view( bs, NA, NC, H, W).permute(0, 1, 3, 4, 2)

        # Concatenate → [bs, NA, H, W, 5+NC] then sigmoid
        y = torch.cat([reg, conf, cls], dim=-1).sigmoid()

        # Grid + anchor decode (cached)
        g, a = get_grid_and_anchor(W, H, i)
        y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + g) * STRIDES[i]
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * a

        z.append(y.view(bs, -1, 5 + NC))

    return torch.cat(z, 1)


def main():
    # ── Init DPU ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL):
        print(f"[ERROR] Not found: {MODEL}")
        sys.exit(1)

    graph = xir.Graph.deserialize(MODEL)
    sg = [c for c in graph.get_root_subgraph().toposort_child_subgraph()
          if c.has_attr("device") and c.get_attr("device").upper() == "DPU"]

    if not sg:
        print("[ERROR] No DPU subgraph in xmodel!")
        sys.exit(1)

    print(f"[INFO] Found {len(sg)} DPU subgraph(s).")
    if len(sg) > 1:
        print("[WARN] More than 1 subgraph → some ops are not on DPU → lower FPS.")

    runner    = vart.Runner.create_runner(sg[0], "run")
    it        = runner.get_input_tensors()
    ot        = runner.get_output_tensors()
    _, h, w, _ = it[0].dims
    in_scale   = 2 ** it[0].get_attr("fix_point")
    out_scales = [2 ** -t.get_attr("fix_point") for t in ot]

    idata = [np.empty(tuple(it[0].dims), dtype=np.int8, order="C")]
    odata = [np.empty(tuple(t.dims),     dtype=np.int8, order="C") for t in ot]

    # Print output shapes for debugging
    print(f"[INFO] DPU input : {tuple(it[0].dims)}")
    for k, t in enumerate(ot):
        print(f"[INFO] DPU output[{k}]: {tuple(t.dims)}")

    # ── Camera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    torch.set_num_threads(4)   # use all 4 ARM A53 cores

    SHOW_DISPLAY = os.environ.get("DISPLAY") is not None
    if SHOW_DISPLAY:
        cv2.namedWindow("LS-YOLO KV260", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("LS-YOLO KV260", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        sys.exit(1)

    print(f"[INFO] Stream started ({w}×{h}). Press Q to exit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── 1. DPU inference ──────────────────────────────────────────
        t0  = time.time()
        # Fix 1: Use letterbox for correct ratio with scale_boxes
        img_pad, _, _ = letterbox(frame, (h, w), auto=False)
        img = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)
        
        idata[0][0, ...] = (img.astype(np.float32) / 255.0 * in_scale).astype(np.int8)
        jid = runner.execute_async(idata, odata)
        runner.wait(jid)
        t_dpu = (time.time() - t0) * 1000

        # ── 2. CPU decode (lightweight) ───────────────────────────────────────
        t1 = time.time()

        # NHWC int8 → NCHW float32, sort P3→P4→P5 (spatial large → small)
        raws = []
        for k, o in enumerate(odata):
            f = o.astype(np.float32) * np.float32(out_scales[k])
            raws.append(torch.from_numpy(f.transpose(0, 3, 1, 2).copy()))
        raws.sort(key=lambda t: t.shape[2], reverse=True)   # P3(64)>P4(32)>P5(16)

        preds = decode_decoupled(raws)
        boxes = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)[0]
        t_cpu = (time.time() - t1) * 1000

        # ── 3. Draw results ─────────────────────────────────────────────
        if boxes is not None and len(boxes):
            boxes[:, :4] = scale_boxes((h, w), boxes[:, :4], frame.shape).round()
            for *xy, conf, cls in boxes:
                x1, y1, x2, y2 = map(int, xy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"landslide {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        fps = 1000 / (t_dpu + t_cpu + 1e-6)
        cv2.putText(frame,
                    f"FPS:{fps:.1f}  DPU:{t_dpu:.1f}ms  CPU:{t_cpu:.1f}ms",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if SHOW_DISPLAY:
            cv2.imshow("LS-YOLO KV260", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Run in background without display, print FPS to terminal
            print(f"[INFO] FPS: {fps:.1f} | DPU: {t_dpu:.1f}ms | CPU: {t_cpu:.1f}ms", end="\r")

    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    else:
        print("\n[INFO] Stream ended.")


if __name__ == "__main__":
    main()
