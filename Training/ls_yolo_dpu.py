"""
ls_yolo_dpu.py — Core deployment of LS-YOLO (Decoupled_Detect head) onto KV260 DPU.
IMPORTANT: The quantize/demo script in the tutorial PDF is written for the standard
Detect head (accesses detect.m[i]). Your actual model uses
Decoupled_Detect (m_stem + m_cls + CAM + m_reg + m_conf) — which does NOT have .m,
so running the PDF script will fail/crash. This file replaces it for the correct architecture.

DPU / CPU partition:
  - DPU  : entire backbone + neck + ALL convs in the head (m_stem, m_cls,
           cam, m_reg, m_conf). DecoupledDPU.forward returns ONLY the raw conv feature
           maps (no sigmoid/no decode yet).
  - CPU  : decode_decoupled handles sigmoid + grid decode (a few ms), then NMS.

Usage:
  from ls_yolo_dpu import DecoupledDPU, decode_decoupled, head_constants
"""
import torch
import torch.nn as nn


class DecoupledDPU(nn.Module):
    """Wraps a trained LS-YOLO (Decoupled_Detect head) into a pure conv/pool/concat
    forward pass so Vitis-AI can map it to the DPU. Returns a tuple of nl tensors (1 for
    each level P3/P4/P5), each NCHW tensor has na*(5+nc) channels, layout by BLOCK:
        [ reg(na*4) | conf(na*1) | cls(na*nc) ]

    Merges 3 branches into 1 output/level (instead of 3) so that when reading VART output on
    KV260, conf/cls are NOT mixed up (when nc=1 both have 3 channels). Levels are distinguished by
    grid size (64/32/16). All sigmoid/grid/NMS operations are left to the CPU via decode_decoupled.
    """

    def __init__(self, model):
        super().__init__()
        # model: DetectionModel — has .model (ModuleList of layers), .save, the last layer is Decoupled_Detect
        self.layers = model.model
        self.save = model.save
        self.detect = model.model[-1]
        assert hasattr(self.detect, 'm_stem'), \
            "Last layer is not Decoupled_Detect. Use the FullDPU wrapper from the PDF for the standard Detect head."

    def forward(self, x):
        y = [None] * len(self.layers)
        # run backbone + neck (all layers except the head)
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i in self.save:
                y[m.i] = x
        feats = [y[j] for j in self.detect.f]  # 3 feature maps into the head

        # run the CONV part of the head (no sigmoid, no grid), merge into 1 output/level
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
    """Extract constants needed for decoding from the PyTorch model, so the KV260 demo (which only has
    the .xmodel, not the PyTorch model) uses the correct values. Returns a dict."""
    d = model.model[-1]
    return {
        'nc': int(d.nc),
        'na': int(d.na),
        'nl': int(d.nl),
        'stride': [float(s) for s in d.stride],
        # anchors buffer ("stride units" like in the original _make_grid), shape [nl, na, 2]
        'anchors': d.anchors.detach().cpu().tolist(),
    }


def _make_grid(nx, ny, na, anchors_i, stride_i, device, dtype):
    """Recreates Decoupled_Detect._make_grid exactly: grid already has a -0.5 offset,
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
    """Decode raw conv feature maps (DPU output) -> tensor [bs, N, 5+nc].
    EXACTLY matches the inference branch of Decoupled_Detect.forward.

    outs    : list/tuple of nl torch tensors, each [bs, na*(5+nc), H, W] NCHW,
              layout block [reg(na*4) | conf(na*1) | cls(na*nc)] (DecoupledDPU output).
    anchors : tensor [nl, na, 2] (= model.model[-1].anchors, unit is stride units).
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
