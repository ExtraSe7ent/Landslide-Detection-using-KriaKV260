"""
ls_yolo_dpu.py — Lõi triển khai LS-YOLO (head Decoupled_Detect) lên KV260 DPU.
QUAN TRỌNG: Script quantize/demo trong file PDF hướng dẫn được viết cho head
`Detect` CHUẨN (truy cập `detect.m[i]`). Model thực tế của bạn dùng
`Decoupled_Detect` (m_stem + m_cls + CAM + m_reg + m_conf) — KHÔNG có `.m`,
nên chạy script PDF sẽ sai/crash. File này thay thế cho đúng kiến trúc đó.

Phân chia DPU / CPU:
  - DPU  : toàn bộ backbone + neck + TẤT CẢ conv trong head (m_stem, m_cls,
           cam, m_reg, m_conf). `DecoupledDPU.forward` trả về CHỈ các feature
           map conv thô (chưa sigmoid/chưa decode).
  - CPU  : `decode_decoupled` làm sigmoid + grid decode (vài ms), rồi NMS.

Cách dùng:
  from ls_yolo_dpu import DecoupledDPU, decode_decoupled, head_constants
"""
import torch
import torch.nn as nn


class DecoupledDPU(nn.Module):
    """Bọc một LS-YOLO đã train (head Decoupled_Detect) thành một forward thuần
    conv/pool/concat để Vitis-AI map lên DPU. Trả về tuple gồm `nl` tensor (1 cho
    mỗi level P3/P4/P5), mỗi tensor NCHW có na*(5+nc) kênh, layout theo BLOCK:
        [ reg(na*4) | conf(na*1) | cls(na*nc) ]

    Gộp 3 nhánh thành 1 output/level (thay vì 3) để khi đọc output từ VART trên
    KV260 KHÔNG bị lẫn conf/cls (khi nc=1 cả hai đều 3 kênh). Phân biệt level bằng
    grid size (64/32/16). Toàn bộ sigmoid/grid/NMS để CPU làm qua decode_decoupled.
    """

    def __init__(self, model):
        super().__init__()
        # model: DetectionModel — có .model (ModuleList các layer), .save, layer cuối là Decoupled_Detect
        self.layers = model.model
        self.save = model.save
        self.detect = model.model[-1]
        assert hasattr(self.detect, 'm_stem'), \
            "Layer cuối không phải Decoupled_Detect. Dùng wrapper FullDPU của PDF cho head Detect chuẩn."

    def forward(self, x):
        y = [None] * len(self.layers)
        # chạy backbone + neck (mọi layer trừ head)
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i in self.save:
                y[m.i] = x
        feats = [y[j] for j in self.detect.f]  # 3 feature map vào head

        # chạy phần CONV của head (không sigmoid, không grid), gộp 1 output/level
        d = self.detect
        outs = []
        for i in range(d.nl):
            stem = d.m_stem[i](feats[i])
            cls_raw = d.m_cls[i](stem)
            cam = d.cam[i](stem)
            reg_raw = d.m_reg[i](cam)
            conf_raw = d.m_conf[i](cam)
            outs.append(torch.cat([reg_raw, conf_raw, cls_raw], dim=1))
        return tuple(outs)


def head_constants(model):
    """Trích các hằng cần cho decode từ model PyTorch, để demo trên KV260 (chỉ có
    .xmodel, không có model PyTorch) dùng đúng giá trị. Trả về dict."""
    d = model.model[-1]
    return {
        'nc': int(d.nc),
        'na': int(d.na),
        'nl': int(d.nl),
        'stride': [float(s) for s in d.stride],
        # anchors buffer (đơn vị "stride units" như trong _make_grid gốc), shape [nl, na, 2]
        'anchors': d.anchors.detach().cpu().tolist(),
    }


def _make_grid(nx, ny, na, anchors_i, stride_i, device, dtype):
    """Tái tạo y hệt Decoupled_Detect._make_grid: grid có sẵn offset -0.5,
    anchor_grid = anchors(stride units) * stride = anchors(pixel)."""
    shape = (1, na, ny, nx, 2)
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=device, dtype=dtype),
        torch.arange(nx, device=device, dtype=dtype),
        indexing='ij')
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
    anchor_grid = (anchors_i * stride_i).view(1, na, 1, 1, 2).expand(shape)
    return grid, anchor_grid


def decode_decoupled(outs, anchors, stride, nc, na):
    """Decode các feature map conv thô (đầu ra DPU) -> tensor [bs, N, 5+nc].
    Khớp CHÍNH XÁC nhánh inference của Decoupled_Detect.forward.

    outs    : list/tuple `nl` tensor torch, mỗi cái [bs, na*(5+nc), H, W] NCHW,
              layout block [reg(na*4) | conf(na*1) | cls(na*nc)] (đầu ra DecoupledDPU).
    anchors : tensor [nl, na, 2] (= model.model[-1].anchors, đơn vị stride units).
    stride  : list/tensor [nl].
    """
    nl = len(stride)
    z = []
    for i in range(nl):
        o = outs[i]
        bs, _, ny, nx = o.shape
        reg_raw = o[:, :na * 4]
        conf_raw = o[:, na * 4: na * 4 + na * 1]
        cls_raw = o[:, na * 4 + na * 1:]
        x_reg = reg_raw.reshape(bs, na, 4, ny, nx).permute(0, 1, 3, 4, 2)
        x_conf = conf_raw.reshape(bs, na, 1, ny, nx).permute(0, 1, 3, 4, 2)
        x_cls = cls_raw.reshape(bs, na, nc, ny, nx).permute(0, 1, 3, 4, 2)
        x = torch.cat([x_reg, x_conf, x_cls], 4).contiguous()

        grid, anchor_grid = _make_grid(nx, ny, na, anchors[i], float(stride[i]), x.device, x.dtype)
        xy, wh, conf = x.sigmoid().split((2, 2, nc + 1), 4)
        xy = (xy * 2 + grid) * float(stride[i])
        wh = (wh * 2) ** 2 * anchor_grid
        y = torch.cat((xy, wh, conf), 4)
        z.append(y.view(bs, na * nx * ny, nc + 5))
    return torch.cat(z, 1)
