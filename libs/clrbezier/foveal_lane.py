"""FovealLaneNet: a lane detector that reads the far field twice (clrbezierlane/foveal_lane_network_design.md).

Lanes are small where they converge, so the network gets more pixels there without changing the frame the lanes
live in. The pipeline augments the crop at 2x (1600 x 640, dataset_curvelanes_clrbezier_2x.py) and keeps the GT in
the 800 x 320 context frame (ToContextFrame, libs/datasets/foveation.py). The detector makes two views of that image:
- context view C: 2 x 2 average pooling, 800 x 320, the whole crop;
- fovea view F: a crop of the 2x image, a window_w x window_h window of the context frame magnified 2x with no
  resampling. Rows: window_top .. window_top + window_h (the far band). Columns: centred on the predicted vanishing
  column, clamped inside the crop, the origin on a half context pixel so the crop is exact.
A context point p (px, edge convention: pixel j covers [j, j + 1)) sits at 2 (p - o) in F, o the window origin.
GT, predictions, losses and the metric stay in the context frame; F is only a source of features.

    FovealLaneNet   detector: backbone on C, window placement (the head's vanishing-column head), backbone on F
    FovealLaneHead  head (an HBLRHead: its 2D geometry, matching, losses and output layout are reused)

Head, in order:
1. tokens: the backbone levels of both views (C3-C5) through a shared 1 x 1 conv + GroupNorm to dim, plus a level and
   a view embedding and a sine embedding of each token's centre in the context frame. No FPN.
2. cross-view deformable encoder: every token samples all maps (3 levels x 2 views) around its centre, mapped into
   each view; F's maps are masked (weight 0) for tokens whose centre lies outside the window, so the softmax falls
   back to C there.
3. vanishing column: a light head on raw C5 (soft-argmax over columns); target = median pairwise intersection of
   lines fitted to the upper halves of the GT lanes (as foveation_gate.py). The window is placed on the prediction at
   test and, in training, on the GT + uniform jitter with probability p_gt, else on the (detached) prediction.
4. proposals (the dynamic anchors): a dense head per view on its stride-8 encoder map predicts a lane-centre heatmap,
   the bottom and top ends and the two inner control points in the chord frame. A GT lane whose centre (arc-length
   midpoint) lies in the window is a target of F's heatmap only, every other lane of C's. The 3 x 3 peaks of both
   views, C's peaks inside the window removed, give the top num_queries. A query = a shared token + the curve
   encoding of its proposal + its own encoder token. The proposals (made monotone, detached) are the step-0 curves.
5. decoder, num_iters lane steps (+ hier_cfg segment steps): self-attention over the queries with a curve position
   encoding (dec_cfg.self_attn "full"; "relation" = HBLRHead's relation attention), dual-view curve attention at the
   current curve's points (C's maps at p, F's at 2 (p - o), F masked where the point is outside the window, one
   softmax over all points, maps and samples), FFN, update -> monotone step (curve_cfg frame "image", y_order
   "monotone"). No GRU, no motion encoding; features are sampled at the new curve every step.
6. scoring as HBLRHead: o2m per query; o2o after one more self-attention over the converged lanes (keys and values
   carry the o2m score).
Losses: HBLRHead's o2o / o2m matching and per-step LaneIoU + endpoint losses (no control-point loss unless
dec_cfg.cp_loss), the proposal heatmap focal loss + endpoint smooth-L1 + LaneIoU of the decoded proposal at the cells
around each centre, the vanishing-column smooth-L1, the seg auxiliary on C's stride-8 map.
fovea_cfg.mode: "window" (the method), "off" (C only; the same network otherwise), "full" (the window is the whole
crop: F = the 2x image; the upper bound).
Logs: vp_err_px, vp_gt_frac, fovea_share (F's share of the last lane step's curve-attention weight), prop_recall (GT
lanes with a selected peak within radius_cells of their centre cell, in their view; prop_recall_fov: of the lanes
centred in the window), prop_fovea_frac (queries from F), win_lane_frac (GT lanes whose centre is in the window),
cov0 / iou0 (HBLRHead)."""
import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS
from torch.utils.checkpoint import checkpoint

from .batched_loss import GTLayout, per_image_sum
from .blrnet import _MLP, _deform_grid_bias, curve_points
from .head import StagePredictions, _smooth_l1
from .hblrnet import (HBLRHead, LaneRelationAttention2D, bottom_first, curve_code, lane_iou_2d, monotone_fix,
                      pieces_of, piecewise_points, subdivide_cubic)

CTX, FOV = 0, 1


# ------------------------------------------------------------------ geometry
def sine_embed(xy, dim, temperature=10000.0):
    """Sine embedding of points xy [..., 2] (normalized, 0..1) -> [..., dim] (dim divisible by 4); on a grid it is
    blrnet.sine_position_2d."""
    n = dim // 4
    freq = temperature ** (torch.arange(n, device=xy.device, dtype=torch.float32) / n)
    p = xy.float() * (2.0 * math.pi)
    px, py = p[..., :1] / freq, p[..., 1:] / freq
    return torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], dim=-1)


def vanishing_point(lanes_px, img_h):
    """Median pairwise intersection of lines x = a y + b fitted to the upper half of each lane (lanes_px: list of
    [M, 2] arrays, px) -> (x, y), or None without two non-parallel lanes (|a1 - a2| <= 0.05) meeting within
    [-img_h, img_h]. foveation_gate.py's vanishing_row, with x."""
    lines = []
    for p in lanes_px:
        p = np.asarray(p, dtype=np.float64)
        if len(p) < 2:
            continue
        top = p[p[:, 1] <= np.median(p[:, 1])] if len(p) >= 4 else p
        if len(top) < 2 or np.ptp(top[:, 1]) < 4:
            continue
        lines.append(np.polyfit(top[:, 1], top[:, 0], 1))
    xs, ys = [], []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (a1, b1), (a2, b2) = lines[i], lines[j]
            if abs(a1 - a2) > 0.05:
                y = (b2 - b1) / (a1 - a2)
                if -img_h <= y <= img_h:
                    xs.append(a1 * y + b1)
                    ys.append(y)
    if not xs:
        return None
    return float(np.median(xs)), float(np.median(ys))


class _Map:
    """One feature map of the token set: its view, size, first token, and cell size in context px."""

    def __init__(self, view, level, h, w, start, cell):
        self.view, self.level, self.h, self.w, self.start, self.cell = view, level, h, w, start, cell
        self.n = h * w


class ViewFrame:
    """Where the views sit in the context frame (ctx_w x ctx_h px) for one batch: the window origin [B, 2] (px) and
    size (px). Feature maps are read with grid_sample(align_corners=False): a map of view C spans [0, ctx_w] x
    [0, ctx_h], one of view F spans the window."""

    def __init__(self, ctx_w, ctx_h, win_w, win_h, origin):
        self.ctx = (float(ctx_w), float(ctx_h))
        self.win = (float(win_w), float(win_h))
        self.origin = origin                                                   # [B, 2] px, or None (no F)

    def _o(self, loc):
        return self.origin.to(loc.dtype).view(-1, *([1] * (loc.dim() - 2)), 2)

    def grid(self, loc, view):
        """Context px loc [B, ..., 2] -> grid_sample coordinates in the maps of `view`."""
        if view == CTX:
            return loc * loc.new_tensor([2.0 / self.ctx[0], 2.0 / self.ctx[1]]) - 1.0
        return (loc - self._o(loc)) * loc.new_tensor([2.0 / self.win[0], 2.0 / self.win[1]]) - 1.0

    def inside(self, loc):
        """[B, ...] bool: context px loc [B, ..., 2] inside the window."""
        if self.origin is None:
            return loc.new_zeros(loc.shape[:-1], dtype=torch.bool)
        r = loc - self._o(loc)
        return (r[..., 0] >= 0) & (r[..., 0] < self.win[0]) & (r[..., 1] >= 0) & (r[..., 1] < self.win[1])


def _sample_maps(values, maps, frame, loc, weights, heads):
    """Deformable sampling over several maps. values [B, T, H, c] (the token values); loc: per map [B, Q, H, K, 2]
    (context px); weights [B, Q, H, M, K] -> [B, Q, H * c]."""
    batch_size, _, _, c_h = values.shape
    num_q, k = weights.shape[1], weights.shape[-1]
    out = 0.0
    for m, mp in enumerate(maps):
        v = values[:, mp.start:mp.start + mp.n].permute(0, 2, 3, 1).reshape(batch_size * heads, c_h, mp.h, mp.w)
        g = frame.grid(loc[m].float(), mp.view).permute(0, 2, 1, 3, 4).reshape(batch_size * heads, num_q, k, 2)
        s = F.grid_sample(v, g.to(v.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
        w = weights[:, :, :, m].permute(0, 2, 1, 3).reshape(batch_size * heads, num_q, k)
        out = out + torch.einsum("bcqk,bqk->bqc", s, w.to(s.dtype))                    # [BH, Q, c]
    return out.view(batch_size, heads, num_q, c_h).permute(0, 2, 1, 3).reshape(batch_size, num_q, heads * c_h)


# ------------------------------------------------------------------ modules
class CrossViewEncoderLayer(nn.Module):
    """A Deformable-DETR encoder layer over the tokens of both views: each token samples every map at num_points
    learned offsets (in the map's cells) around its centre; F's maps are masked for tokens outside the window."""

    def __init__(self, dim, num_maps, num_heads, num_points, ffn_dim, dropout):
        super().__init__()
        if dim % num_heads:
            raise ValueError("FovealLaneHead enc_cfg.num_heads must divide dim")
        self.heads, self.maps, self.points = int(num_heads), int(num_maps), int(num_points)
        self.c_h = dim // self.heads
        self.value_proj = nn.Linear(dim, dim)
        self.offsets = nn.Linear(dim, self.heads * self.maps * self.points * 2)
        self.weights = nn.Linear(dim, self.heads * self.maps * self.points)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.zeros_(self.offsets.weight)
        with torch.no_grad():
            self.offsets.bias.copy_(_deform_grid_bias(self.heads, self.maps, self.points))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        for lin in (self.value_proj, self.out, self.ffn[0], self.ffn[3]):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for n in self.norms:
            nn.init.ones_(n.weight)
            nn.init.zeros_(n.bias)

    def forward(self, x, pos, ref, maps, frame, fov_mask):
        """x [B, T, D]; pos [B, T, D]; ref [B, T, 2] token centres (context px); fov_mask [B, T] bool (centre
        outside the window: F's maps get no weight) or None."""
        batch_size, num_t, dim = x.shape
        H, M, K = self.heads, self.maps, self.points
        q = x + pos
        off = self.offsets(q).float().view(batch_size, num_t, H, M, K, 2)
        logit = self.weights(q).float().view(batch_size, num_t, H, M, K)
        if fov_mask is not None:
            is_f = torch.tensor([mp.view == FOV for mp in maps], device=x.device)
            logit = logit.masked_fill(fov_mask[:, :, None, None, None] & is_f.view(1, 1, 1, M, 1), float("-inf"))
        w = torch.softmax(logit.view(batch_size, num_t, H, M * K), dim=-1).view(batch_size, num_t, H, M, K)
        loc = [ref[:, :, None, None, :].float() + off[:, :, :, m] * mp.cell for m, mp in enumerate(maps)]
        a = _sample_maps(self.value_proj(x).view(batch_size, num_t, H, self.c_h), maps, frame, loc, w, H)
        x = self.norms[0](x + self.drop(self.out(a.to(x.dtype))))
        return self.norms[1](x + self.drop(self.ffn(x)))


class DualCurveAttention(nn.Module):
    """Curve attention over both views: a query samples every map at num_points learned offsets (in the map's cells)
    around each of the S reference points of its current curve, with one softmax per head over all S x maps x points.
    F's samples get no weight where the reference point is outside the window. The values are the encoder tokens."""

    def __init__(self, dim, num_maps, num_heads, num_points):
        super().__init__()
        if dim % num_heads:
            raise ValueError("FovealLaneHead dec_cfg.deform_heads must divide dim")
        self.heads, self.maps, self.points = int(num_heads), int(num_maps), int(num_points)
        self.c_h = dim // self.heads
        self.value_proj = nn.Linear(dim, dim)
        self.offsets = nn.Linear(dim, self.heads * self.maps * self.points * 2)
        self.weights = nn.Linear(dim, self.heads * self.maps * self.points)
        self.out = nn.Linear(dim, dim)
        self.last_offset_px = None
        self.last_fovea_share = None

    def reset_parameters(self):
        nn.init.zeros_(self.offsets.weight)
        with torch.no_grad():
            self.offsets.bias.copy_(_deform_grid_bias(self.heads, self.maps, self.points))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        for lin in (self.value_proj, self.out):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, q, pos, ref, tokens, maps, frame):
        """q [B, N, D]; pos [S, D]; ref [B, N, S, 2] context px; tokens [B, T, D] -> [B, N, D]."""
        batch_size, num_q, dim = q.shape
        S, H, M, K = ref.shape[2], self.heads, self.maps, self.points
        qs = q[:, :, None, :] + pos[None, None]                                       # [B, N, S, D]
        off = self.offsets(qs).float().view(batch_size, num_q, S, H, M, K, 2)
        logit = self.weights(qs).float().view(batch_size, num_q, S, H, M, K)
        is_f = torch.tensor([mp.view == FOV for mp in maps], device=q.device)
        if bool(is_f.any()):
            outside = ~frame.inside(ref.float())                                       # [B, N, S]
            logit = logit.masked_fill(outside[..., None, None, None] & is_f.view(1, 1, 1, 1, M, 1), float("-inf"))
        w = torch.softmax(logit.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, num_q, H, S * M * K), dim=-1)
        w = w.view(batch_size, num_q, H, S, M, K)
        # per map: [B, N * S, H, K] as the query axis
        w_m = w.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, num_q * S, H, M, K)
        loc = [(ref[:, :, :, None, None, :].float() + off[:, :, :, :, m] * mp.cell).reshape(batch_size, num_q * S, H, K, 2)
               for m, mp in enumerate(maps)]
        a = _sample_maps(self.value_proj(tokens).view(batch_size, tokens.shape[1], H, self.c_h), maps, frame, loc, w_m,
                         H)
        a = a.view(batch_size, num_q, S, dim).sum(2)                                  # the softmax spans S
        with torch.no_grad():
            self.last_fovea_share = w[..., is_f, :].sum(dim=(-1, -2, -3)).mean() if bool(is_f.any()) else None
            self.last_offset_px = (off[..., 0, :, :].abs().mean() * maps[0].cell)
        return self.out(a.to(q.dtype))


class PlainLaneSA(nn.Module):
    """Self-attention over all lanes (or segments) of an image with a curve position encoding: an MLP of the lane's
    points (normalized). With use_scores, the keys and values also carry the o2m score (a zero-initialized
    projection). Called as LaneRelationAttention2D: (h [B, N, D], lanes [B, N, P, 2] px, scores [B, N] or None) ->
    the attention output [B, N, D] (no residual)."""

    def __init__(self, dim, num_heads, num_points, img_w, img_h, dropout=0.0, use_scores=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.pe = _MLP(2 * int(num_points), dim)
        self.score = nn.Linear(1, dim) if use_scores else None
        self.register_buffer("scale", torch.tensor([float(img_w - 1), float(img_h - 1)]), persistent=False)
        self.last_self_weight = None

    def reset_parameters(self):
        self.attn._reset_parameters()
        for m in self.pe.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if self.score is not None:
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)

    def forward(self, h, lanes, scores=None):
        pe = self.pe((lanes.detach().float() / self.scale).flatten(2)).to(h.dtype)
        q = h + pe
        k, v = q, h
        if self.score is not None and scores is not None:
            s = self.score(scores.detach()[..., None].to(h.dtype))
            k, v = q + s, h + s
        out, w = self.attn(q, k, v, need_weights=True, average_attn_weights=True)
        self.last_self_weight = w.detach().diagonal(dim1=-2, dim2=-1).mean()
        return out


class FovealStep(nn.Module):
    """One decoder step: self-attention (plain or relation), dual-view curve attention, FFN."""

    def __init__(self, dim, sa, num_maps, deform_heads, deform_points, ffn_dim, dropout):
        super().__init__()
        self.sa = sa
        self.curve = DualCurveAttention(dim, num_maps, deform_heads, deform_points)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        if isinstance(self.sa, PlainLaneSA):
            self.sa.reset_parameters()
        else:
            for m in self.sa.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        self.curve.reset_parameters()
        for lin in (self.ffn[0], self.ffn[3]):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for n in self.norms:
            nn.init.ones_(n.weight)
            nn.init.zeros_(n.bias)

    def forward(self, h, lanes, pos, ref, tokens, maps, frame):
        n = self.norms
        h = n[0](h + self.drop(self.sa(h, lanes)))
        h = n[1](h + self.drop(self.curve(h, pos, ref, tokens, maps, frame)))
        return n[2](h + self.drop(self.ffn(h)))


class ProposalHead(nn.Module):
    """Dense lane proposals on one view's stride-8 map: a centre heatmap logit and 8 regression channels (bottom x,
    bottom y gap, top x, top y gap, then per inner control point: along the chord, across it)."""

    def __init__(self, dim, channels):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(dim, channels, 3, padding=1), nn.ReLU(inplace=True),
                                   nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.hm = nn.Conv2d(channels, 1, 1)
        self.reg = nn.Conv2d(channels, 8, 1)

    def reset_parameters(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        nn.init.normal_(self.hm.weight, std=1.0e-2)
        nn.init.constant_(self.hm.bias, -math.log(9.0))              # CenterNet: every cell starts at 0.1
        nn.init.normal_(self.reg.weight, std=1.0e-2)
        nn.init.zeros_(self.reg.bias)

    def forward(self, m):
        t = self.trunk(m)
        return self.hm(t).float(), self.reg(t).float()


class VanishingColumnHead(nn.Module):
    """The vanishing column from C5: conv 3 x 3 + ReLU + 1 x 1 -> a logit per cell, averaged over rows, soft-argmax
    over the column centres (context px). The last conv starts at zero: the image centre."""

    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.out = nn.Conv2d(channels, 1, 1)

    def reset_parameters(self):
        self.conv.reset_parameters()
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, c5, ctx_w):
        logit = self.out(F.relu(self.conv(c5))).float().mean(2)[:, 0]                 # [B, W5]
        w5 = logit.shape[-1]
        centres = (torch.arange(w5, device=logit.device, dtype=torch.float32) + 0.5) * (float(ctx_w) / w5)
        return (torch.softmax(logit, dim=-1) * centres).sum(-1)


@contextlib.contextmanager
def _running_bn(module):
    """BatchNorm layers of `module` in eval mode inside the block (normalize with the running statistics, do not
    update them); restored after."""
    bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
    for m in bns:
        m.train(False)
    try:
        yield
    finally:
        for m in bns:
            m.train(True)


# ------------------------------------------------------------------ detector
@MODELS.register_module()
class FovealLaneNet(SingleStageDetector):
    """Two views of the 2x input through one backbone (module docstring); no neck. The head places the window."""

    def __init__(self, backbone, bbox_head, neck=None, train_cfg=None, test_cfg=None, data_preprocessor=None,
                 init_cfg=None):
        if neck is not None:
            raise ValueError("FovealLaneNet has no neck: the head's cross-view encoder reads the backbone levels")
        super().__init__(backbone=backbone, neck=None, bbox_head=bbox_head, train_cfg=train_cfg, test_cfg=test_cfg,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    def extract_feat(self, batch_inputs, batch_data_samples=None):
        head = self.bbox_head
        want = (2 * head.img_h, 2 * head.img_w)
        if tuple(batch_inputs.shape[-2:]) != want:
            raise ValueError(f"FovealLaneNet expects the 2x input {want[1]} x {want[0]} (dataset_curvelanes_clrbezier_2x"
                             f".py), got {batch_inputs.shape[-1]} x {batch_inputs.shape[-2]}")
        ctx = F.avg_pool2d(batch_inputs, 2)
        c = tuple(self.backbone(ctx))
        view = head.place_window(c, batch_data_samples if self.training else None)
        f = None
        if head.fovea_mode != "off":
            fov = head.crop_fovea(batch_inputs, view["origin"])
            bn = _running_bn(self.backbone) if head.fovea_bn == "context" else contextlib.nullcontext()
            with bn:
                f = tuple(self.backbone(fov))
        return dict(ctx=c, fov=f, **view)

    def loss(self, batch_inputs, batch_data_samples):
        return self.bbox_head.loss(self.extract_feat(batch_inputs, batch_data_samples), batch_data_samples)

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        for sample in batch_data_samples:
            sample.metainfo["batch_input_shape"] = tuple(batch_inputs.size()[-2:])
        return self.bbox_head.predict(self.extract_feat(batch_inputs), batch_data_samples)

    def _forward(self, batch_inputs, batch_data_samples=None):
        return self.bbox_head.forward(self.extract_feat(batch_inputs))


# ------------------------------------------------------------------ head
@MODELS.register_module()
class FovealLaneHead(HBLRHead):
    """The foveal head (module docstring). Config, all keys optional (defaults in brackets):
    in_channels ([128, 256, 512], backbone C3-C5), num_queries (16), dim (256), num_points / img_w / img_h /
    seg_num_classes (as HBLRHead), curve_cfg (HBLRHead's; frame "image" and y_order "monotone" required),
    hier_cfg (HBLRHead's, schedule "sequential"; seg_sample_points unused, seg_neighbors only with relation),
    assign_cfg / loss_cfg / target_adapter / test_score / test_cfg (as BLRHead; loss_cfg.cp_loss_weight only with
    dec_cfg.cp_loss).
    fovea_cfg: mode ("window" | "off" | "full"), window_w (400) / window_h (80) / window_top (0) in context px (2x
    each must be a multiple of 32), p_gt (0.5, training windows placed on the GT vanishing column), jitter_px (40,
    uniform jitter of that GT column), bn ("context": F is normalized with the backbone's running statistics, which
    only C updates, so training and test treat F alike; "batch": plain training-mode BatchNorm on F too).
    vp_cfg: channels (128), loss_weight (1.0), beta (0.05; smooth-L1 on the error / (img_w / 2)), in every mode.
    enc_cfg: num_layers (3), num_heads (8), num_points (4), ffn_dim (1024), dropout (0.0), checkpoint (False:
    recompute each layer in the backward pass to save memory).
    prop_cfg: channels (64), sigma_cells (2.0), radius_cells (2), scale_px (100; F's offsets at half, its own px),
    hm_weight (1.0), end_weight (0.05, px smooth-L1 of the two ends), iou_weight (1.0, 1 - 2D LaneIoU of the decoded
    proposal).
    dec_cfg: num_iters (6; lane steps), self_attn ("full" | "relation"), num_heads (8), ffn_dim (1024), dropout (0.0),
    curve_points (36, reference points per lane), deform_heads (8), deform_points (4), step_scale (1.0), cp_loss
    (False), num_neighbors (16) / dist_scale (15) / max_dist (10) (relation only)."""

    def __init__(self, in_channels=(128, 256, 512), num_queries=16, dim=256, num_points=72, img_w=800, img_h=320,
                 seg_num_classes=2, curve_cfg=None, hier_cfg=None, fovea_cfg=None, vp_cfg=None, enc_cfg=None,
                 prop_cfg=None, dec_cfg=None, assign_cfg=None, loss_cfg=None, target_adapter=None, test_score=None,
                 train_cfg=None, test_cfg=None, init_cfg=None):
        fc, vc, ec, pc, dc = (dict(c or {}) for c in (fovea_cfg, vp_cfg, enc_cfg, prop_cfg, dec_cfg))
        for name, cfg, known in (
                ("fovea_cfg", fc, {"mode", "window_w", "window_h", "window_top", "p_gt", "jitter_px", "bn"}),
                ("vp_cfg", vc, {"channels", "loss_weight", "beta"}),
                ("enc_cfg", ec, {"num_layers", "num_heads", "num_points", "ffn_dim", "dropout", "checkpoint"}),
                ("prop_cfg", pc, {"channels", "sigma_cells", "radius_cells", "scale_px", "hm_weight", "end_weight",
                                  "iou_weight"}),
                ("dec_cfg", dc, {"num_iters", "self_attn", "num_heads", "ffn_dim", "dropout", "curve_points",
                                 "deform_heads", "deform_points", "step_scale", "cp_loss", "num_neighbors",
                                 "dist_scale", "max_dist"})):
            if set(cfg) - known:
                raise ValueError(f"FovealLaneHead {name}: unknown keys {sorted(set(cfg) - known)}")
        cc = dict(curve_cfg or {})
        if cc.get("frame") != "image" or cc.get("y_order") != "monotone":
            raise ValueError("FovealLaneHead needs curve_cfg frame='image' and y_order='monotone'")
        if hier_cfg is not None and dict(hier_cfg).get("schedule", "sequential") != "sequential":
            raise ValueError("FovealLaneHead supports hier_cfg schedule 'sequential' only")
        if dc.get("self_attn", "full") not in ("full", "relation"):
            raise ValueError("FovealLaneHead dec_cfg.self_attn must be 'full' or 'relation'")
        heads = int(dc.get("num_heads", 8))
        ffn = int(dc.get("ffn_dim", 1024))
        drop = float(dc.get("dropout", 0.0))
        curve_pts = int(dc.get("curve_points", 36))
        # HBLRHead / BLRHead arguments: their anchor, interaction-block and GRU modules are built and removed below
        super().__init__(
            curve_cfg=cc, hier_cfg=hier_cfg, num_points=num_points, sample_points=curve_pts, prior_feat_channels=dim,
            num_levels=3, img_w=img_w, img_h=img_h, seg_num_classes=seg_num_classes,
            anchor_cfg=dict(num_anchors=int(num_queries), dim=int(dim), num_blocks=0, num_heads=heads,
                            num_neighbors=int(dc.get("num_neighbors", 16)), ffn_dim=ffn, dropout=drop,
                            dist_scale=float(dc.get("dist_scale", 15.0)), max_dist=float(dc.get("max_dist", 10.0))),
            refine_cfg=dict(num_iters=int(dc.get("num_iters", 6)), resample="fine",
                            step_scale=float(dc.get("step_scale", 1.0)), point_offsets=False, step_blocks=True,
                            step_ffn_dim=ffn, deform_heads=int(dc.get("deform_heads", 8)),
                            deform_points=int(dc.get("deform_points", 4)), deform_levels=3,
                            deform_curve_points=curve_pts),
            assign_cfg=assign_cfg, loss_cfg=loss_cfg, target_adapter=target_adapter, test_score=test_score,
            train_cfg=train_cfg, test_cfg=test_cfg, init_cfg=init_cfg)
        for name in ("anchor_cps", "anchor_embed", "img_proj", "sem_pre", "sem_pre_norm", "point_proj", "point_pos",
                     "point_norm", "geo_point", "geo_vec", "fuse", "delta_enc", "resample_enc", "gru", "step_blocks",
                     "step_pos", "seg_decoder", "blocks"):
            delattr(self, name)
        if self.hier is not None:
            for name in ("seg_blocks", "seg_step_pos", "seg_delta_enc", "seg_resample_enc", "seg_gru"):
                delattr(self, name)
        self.step_blocks = self.seg_decoder = None
        self.blocks = []
        self.relation = dc.get("self_attn", "full") == "relation"
        self.cp_loss = bool(dc.get("cp_loss", False))

        # views
        self.fovea_mode = str(fc.get("mode", "window"))
        if self.fovea_mode not in ("window", "off", "full"):
            raise ValueError("FovealLaneHead fovea_cfg.mode must be 'window', 'off' or 'full'")
        self.fovea_bn = str(fc.get("bn", "context"))
        if self.fovea_bn not in ("context", "batch"):
            raise ValueError("FovealLaneHead fovea_cfg.bn must be 'context' or 'batch'")
        if self.fovea_mode == "full":
            self.win_w, self.win_h, self.win_top = float(self.img_w), float(self.img_h), 0.0
        else:
            self.win_w, self.win_h = float(fc.get("window_w", 400)), float(fc.get("window_h", 80))
            self.win_top = float(fc.get("window_top", 0))
        if (2 * self.win_w) % 32 or (2 * self.win_h) % 32 or self.win_w > self.img_w or \
                self.win_top + self.win_h > self.img_h or self.win_top % 0.5:
            raise ValueError("FovealLaneHead fovea_cfg: 2 x window_w and 2 x window_h must be multiples of 32 and the "
                             "window must fit in the crop")
        self.p_gt = float(fc.get("p_gt", 0.5))
        self.jitter_px = float(fc.get("jitter_px", 40.0))
        views = 1 if self.fovea_mode == "off" else 2
        num_maps = 3 * views

        # tokens and encoder
        in_channels = [int(c) for c in in_channels]
        if len(in_channels) != 3:
            raise ValueError("FovealLaneHead in_channels: the 3 backbone levels C3-C5")
        if dim % 32:
            raise ValueError("FovealLaneHead dim must be a multiple of 32 (GroupNorm)")
        self.in_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, dim, 1), nn.GroupNorm(32, dim)) for c in in_channels])
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        self.view_embed = nn.Parameter(torch.zeros(views, dim))
        self.enc = nn.ModuleList([CrossViewEncoderLayer(dim, num_maps, int(ec.get("num_heads", 8)),
                                                        int(ec.get("num_points", 4)), int(ec.get("ffn_dim", 1024)),
                                                        float(ec.get("dropout", 0.0)))
                                  for _ in range(int(ec.get("num_layers", 3)))])
        self.enc_checkpoint = bool(ec.get("checkpoint", False))

        # vanishing column
        self.vp_head = VanishingColumnHead(in_channels[-1], int(vc.get("channels", 128)))
        self.vp_weight, self.vp_beta = float(vc.get("loss_weight", 1.0)), float(vc.get("beta", 0.05))

        # proposals
        self.prop = dict(sigma=float(pc.get("sigma_cells", 2.0)), radius=int(pc.get("radius_cells", 2)),
                         scale_px=float(pc.get("scale_px", 100.0)), hm_weight=float(pc.get("hm_weight", 1.0)),
                         end_weight=float(pc.get("end_weight", 0.05)), iou_weight=float(pc.get("iou_weight", 1.0)))
        self.prop_heads = nn.ModuleList([ProposalHead(dim, int(pc.get("channels", 64))) for _ in range(views)])
        self.query_token = nn.Parameter(torch.zeros(dim))
        self.tok_proj = nn.Linear(dim, dim)
        self.query_norm = nn.LayerNorm(dim)

        # decoder
        rel = dict(num_heads=heads, n_strips=self.n_strips, x_scale=float(self.img_w - 1),
                   dist_scale=float(dc.get("dist_scale", 15.0)), max_dist=float(dc.get("max_dist", 10.0)),
                   dropout=drop)

        def sa(points, neighbours):
            if self.relation:
                m = LaneRelationAttention2D(dim=dim, num_neighbors=neighbours, **rel)
                m.orientation_free = self.orientation_free
                return m
            return PlainLaneSA(dim, heads, points, self.img_w, self.img_h, drop)

        deform = (num_maps, int(dc.get("deform_heads", 8)), int(dc.get("deform_points", 4)), ffn, drop)
        self.dec = nn.ModuleList([FovealStep(dim, sa(self.rel_points, int(dc.get("num_neighbors", 16))), *deform)
                                  for _ in range(self.num_iters)])
        self.dec_pos = nn.Parameter(torch.zeros(curve_pts, dim))
        if self.hier is not None:
            nb = int(dict(hier_cfg).get("seg_neighbors", 16))
            self.seg_dec = nn.ModuleList([FovealStep(dim, sa(self.hier["rel_points"], nb), *deform)
                                          for _ in range(self.hier["iters"])])
            self.seg_dec_pos = nn.Parameter(torch.zeros(self.hier["deform_points"], dim))
        if not self.relation:
            # scoring: plain self-attention whose keys / values carry the o2m score
            self.final_sa = PlainLaneSA(dim, heads, self.rel_points, self.img_w, self.img_h, drop, use_scores=True)
        self.seg_head = nn.Sequential(nn.Conv2d(dim, 64, 3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, seg_num_classes, 1))
        self._reset_foveal()

    def _reset_foveal(self):
        for m in self.in_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for p in (self.level_embed, self.view_embed, self.query_token, self.dec_pos):
            nn.init.trunc_normal_(p, std=0.02)
        for m in list(self.enc) + list(self.dec) + list(self.prop_heads) + [self.vp_head]:
            m.reset_parameters()
        if self.hier is not None:
            nn.init.trunc_normal_(self.seg_dec_pos, std=0.02)
            for m in self.seg_dec:
                m.reset_parameters()
        if isinstance(self.final_sa, PlainLaneSA):
            self.final_sa.reset_parameters()
        nn.init.xavier_uniform_(self.tok_proj.weight)
        nn.init.zeros_(self.tok_proj.bias)
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    # ------------------------------------------------------------------ views
    def place_window(self, c_feats, batch_data_samples=None):
        """The vanishing column from C5 and the window origin [B, 2] (context px; zeros for "off" / "full"). With
        data samples (training), also the GT lanes and the GT vanishing column (NaN where undefined)."""
        vp = self.vp_head(c_feats[-1], self.img_w)
        batch_size, dev = vp.shape[0], vp.device
        out = dict(vp=vp, vp_gt=None, gts=None)
        if batch_data_samples is not None:
            gts = self.extract_lanes_2d(batch_data_samples, dev)
            out["gts"] = gts
            v = [vanishing_point(self._px(g.double()).cpu().numpy(), self.img_h) for g in gts]
            out["vp_gt"] = torch.tensor([math.nan if p is None else p[0] for p in v], device=dev)
        origin = torch.zeros(batch_size, 2, device=dev)
        if self.fovea_mode == "window":
            centre = vp.detach().float()
            if self.training and out["vp_gt"] is not None and self.p_gt > 0:
                use = (torch.rand(batch_size, device=dev) < self.p_gt) & torch.isfinite(out["vp_gt"])
                jit = (torch.rand(batch_size, device=dev) * 2.0 - 1.0) * self.jitter_px
                centre = torch.where(use, torch.nan_to_num(out["vp_gt"]) + jit, centre)
            ox = (centre - 0.5 * self.win_w).clamp(0.0, self.img_w - self.win_w)
            origin[:, 0] = torch.round(ox * 2.0) / 2.0                         # half context px: whole 2x px
            origin[:, 1] = self.win_top
        out["origin"] = origin
        return out

    def crop_fovea(self, x2, origin):
        """The F view: per image, the 2x pixels of the window at origin (context px) -> [B, C, 2 win_h, 2 win_w]."""
        if self.fovea_mode == "full":
            return x2
        w2, h2 = int(2 * self.win_w), int(2 * self.win_h)
        o2 = torch.round(origin.detach() * 2.0).long().tolist()
        return torch.stack([x2[b, :, oy:oy + h2, ox:ox + w2] for b, (ox, oy) in enumerate(o2)])

    def _view_layout(self, x):
        """The frame of x (window origin) and its maps (C's levels fine -> coarse, then F's), with each map's token
        centres [B, h * w, 2] (context px, row-major)."""
        origin = x["origin"]
        frame = ViewFrame(self.img_w, self.img_h, self.win_w, self.win_h,
                          origin.float() if x.get("fov") is not None else None)
        views = [(CTX, x["ctx"])] + ([(FOV, x["fov"])] if x.get("fov") is not None else [])
        maps, refs, start = [], [], 0
        batch_size, dev = origin.shape[0], origin.device
        for v, feats in views:
            span_w, span_h = (self.img_w, self.img_h) if v == CTX else (self.win_w, self.win_h)
            for lvl, f in enumerate(feats):
                h, w = f.shape[-2:]
                cell = (span_w / w, span_h / h)
                if abs(cell[0] - cell[1]) > 1e-6:
                    raise ValueError(f"FovealLaneHead: non-square cells {cell} on view {v} level {lvl}")
                ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                        torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
                ref = torch.stack([(xs.reshape(-1) + 0.5) * cell[0], (ys.reshape(-1) + 0.5) * cell[1]], dim=-1)
                ref = ref[None].expand(batch_size, -1, -1)
                refs.append(ref + origin[:, None, :].float() if v == FOV else ref)
                maps.append(_Map(v, lvl, h, w, start, cell[0]))
                start += h * w
        return frame, maps, refs

    def _tokens(self, x):
        """Encoded tokens of both views [B, T, D], the maps, the frame, and each view's stride-8 map [B, D, h, w]."""
        frame, maps, refs = self._view_layout(x)
        feats = list(x["ctx"]) + (list(x["fov"]) if x.get("fov") is not None else [])
        scale = torch.tensor([1.0 / self.img_w, 1.0 / self.img_h], device=refs[0].device)
        toks = [self.in_proj[mp.level](f).flatten(2).transpose(1, 2) for mp, f in zip(maps, feats)]
        poss = [sine_embed(r * scale, self.dim) + self.level_embed[mp.level] + self.view_embed[mp.view]
                for mp, r in zip(maps, refs)]
        x_tok = torch.cat(toks, dim=1).float()
        pos, ref = torch.cat(poss, dim=1), torch.cat(refs, dim=1)
        fov_mask = None if frame.origin is None else ~frame.inside(ref)
        for layer in self.enc:
            if self.enc_checkpoint and self.training:
                x_tok = checkpoint(layer, x_tok, pos, ref, maps, frame, fov_mask, use_reentrant=False)
            else:
                x_tok = layer(x_tok, pos, ref, maps, frame, fov_mask)
        batch_size = x_tok.shape[0]
        fine = [x_tok[:, mp.start:mp.start + mp.n].transpose(1, 2).reshape(batch_size, self.dim, mp.h, mp.w)
                for mp in maps if mp.level == 0]
        return x_tok, maps, frame, fine

    # ------------------------------------------------------------------ proposals
    def _cell_centres(self, view, h, w, frame, dev):
        """Cell centres of a view's stride-8 grid -> [B or 1, h * w, 2] (context px)."""
        span_w, span_h = (self.img_w, self.img_h) if view == CTX else (self.win_w, self.win_h)
        ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
        c = torch.stack([(xs.reshape(-1) + 0.5) * (span_w / w), (ys.reshape(-1) + 0.5) * (span_h / h)], dim=-1)[None]
        return c + frame.origin[:, None, :].float() if view == FOV else c

    def _prop_decode(self, reg, cell, scale):
        """Regression channels reg [..., 8] at cell centres cell [..., 2] (context px), offsets in units of scale
        (px) -> bottom, top [..., 2] and the monotone cubic [..., 4, 2] (context px)."""
        bottom = torch.stack([cell[..., 0] + scale * reg[..., 0], cell[..., 1] + scale * F.softplus(reg[..., 1])], -1)
        top = torch.stack([cell[..., 0] + scale * reg[..., 2], cell[..., 1] - scale * F.softplus(reg[..., 3])], -1)
        d = top - bottom
        nrm = torch.stack([-d[..., 1], d[..., 0]], dim=-1)                        # |nrm| = |d|
        p1 = bottom + (1.0 / 3.0 + reg[..., 4:5]) * d + reg[..., 5:6] * nrm
        p2 = bottom + (2.0 / 3.0 + reg[..., 6:7]) * d + reg[..., 7:8] * nrm
        return bottom, top, monotone_fix(torch.stack([bottom, p1, p2, top], dim=-2))

    def _view_scale(self, view):
        return self.prop["scale_px"] * (1.0 if view == CTX else 0.5)

    def _proposals(self, fine, frame):
        """Dense maps of each view and the num_queries strongest 3 x 3 peaks of their union (C's peaks inside the
        window last) -> dict(hm, reg: per view [B, 1 | 8, h, w]; view [B, N]; cell [B, N] (index in its view);
        cps [B, N, 4, 2] normalized, detached; tok [B, N, D])."""
        hms, regs, ranks, regs_f, cells, toks, scales = [], [], [], [], [], [], []
        for v, (m, head) in enumerate(zip(fine, self.prop_heads)):
            hm, reg = head(m)
            hms.append(hm)
            regs.append(reg)
            h, w = hm.shape[-2:]
            with torch.no_grad():
                s = hm.sigmoid()
                peak = (s == F.max_pool2d(s, 3, stride=1, padding=1)).flatten(1)
                rank = torch.where(peak, s.flatten(1), s.flatten(1) - 2.0)
                c = self._cell_centres(v, h, w, frame, hm.device)
                if v == CTX and frame.origin is not None:
                    rank = rank - 4.0 * frame.inside(c.expand(rank.shape[0], -1, -1)).float()
            ranks.append(rank)
            regs_f.append(reg.flatten(2).transpose(1, 2))
            cells.append(c.expand(rank.shape[0], -1, -1))
            toks.append(m.flatten(2).transpose(1, 2))
            scales.append(torch.full((h * w,), self._view_scale(v), device=hm.device))
        sizes = [r.shape[1] for r in ranks]
        idx = torch.cat(ranks, dim=1).topk(self.num_anchors, dim=1).indices                 # [B, N]
        view = (idx >= sizes[0]).long() if len(sizes) > 1 else torch.zeros_like(idx)
        cell = idx - view * sizes[0]

        def take(t):
            return torch.cat(t, dim=1).gather(1, idx[..., None].expand(-1, -1, t[0].shape[-1]))

        scale = torch.cat(scales)[idx]
        _, _, c_px = self._prop_decode(take(regs_f), take(cells), scale)
        return dict(hm=hms, reg=regs, view=view, cell=cell, cps=self._norm(c_px).detach(), tok=take(toks))

    # ------------------------------------------------------------------ forward
    def _forward(self, x):
        tokens, maps, frame, fine = self._tokens(x)
        prop = self._proposals(fine, frame)
        batch_size, num_q, dim = prop["tok"].shape
        cps0 = prop["cps"]
        h = self.query_norm(self.query_token + self.geo_enc(curve_code(self._px(cps0), self.img_h - 1))
                            + self.tok_proj(prop["tok"]))
        cps = cps0
        curves, cps_list, ends, step_off, step_w = [], [], [], [], []
        for step in self.dec:
            ref = self._px(curve_points(cps, self.t_deform)).detach()
            rel = self._px(curve_points(cps, self.t_rel)).detach()
            h = step(h, rel, self.dec_pos, ref, tokens, maps, frame)
            step_off.append(step.curve.last_offset_px)
            step_w.append(step.sa.last_self_weight)
            upd = self.update(h).view(batch_size, num_q, 4, 2).float() * self.step_scale
            cps = self._norm(self._lane_step(self._px(cps), upd))
            curves.append(curve_points(cps, self.t_out_2d))
            cps_list.append(cps)
            ends.append(cps)
        fovea_share = self.dec[-1].curve.last_fovea_share
        num_lane_steps = len(curves)
        seg_move = None
        if self.hier is not None:
            segs = self.hier["segments"]
            q = subdivide_cubic(cps, segs)                                             # [B, N, 3S+1, 2]
            hs = self.seg_norm(h[:, :, None] + self.seg_embed
                               + self.seg_geo(curve_code(self._px(pieces_of(q)), self.img_h - 1)))
            hs = hs.reshape(batch_size, num_q * segs, dim)
            index, count = self._joint_index(segs, q.device)
            moves = []
            for step in self.seg_dec:
                pc = pieces_of(q)                                                     # [B, N, S, 4, 2]
                rel = self._px(curve_points(pc, self.t_seg_rel)).reshape(batch_size, num_q * segs, -1, 2).detach()
                ref = self._px(curve_points(pc, self.t_seg_deform)).reshape(batch_size, num_q * segs, -1, 2).detach()
                hs = step(hs, rel, self.seg_dec_pos, ref, tokens, maps, frame)
                step_off.append(step.curve.last_offset_px)
                step_w.append(step.sa.last_self_weight)
                upd = self.seg_update(hs).view(batch_size, num_q, segs, 4, 2).float() * self.step_scale
                q_px, d_q = self._chain_step(self._px(q), upd, index, count)
                q = self._norm(q_px)
                moves.append(d_q.detach().norm(dim=-1).mean())
                curves.append(piecewise_points(q, self.out_per_segment))
                cps_list.append(q)
                ends.append(torch.cat([q[..., :2, :], q[..., -2:, :]], dim=-2))
            h = h + self.seg_pool(hs.view(batch_size, num_q, segs, dim).mean(2))
            seg_move = torch.stack(moves).mean()
        lane_cps = ((cps_list[:num_lane_steps] if self.cp_loss else [None] * num_lane_steps)
                    + [None] * (len(curves) - num_lane_steps))
        out = self._finish(h, None, None, curves, cps_list, ends, lane_cps, num_lane_steps, step_off, step_w,
                           seg_move=seg_move)
        out["curve0"] = curve_points(cps0, self.t_out_2d).detach()
        out["prop"] = prop
        out["fovea_share"] = fovea_share
        out["seg_map"] = fine[CTX]
        return out

    def forward_train(self, x):
        out = self._forward(x)
        out["seg"] = self.seg_head(out.pop("seg_map"))
        return out

    def forward_test(self, x):
        out = self._forward(x)
        cfg = self.test_cfg or {}
        use_nms = bool(cfg.get("use_nms", True))
        source = self.test_score or ("o2m" if use_nms else "o2o")
        z = out["cls_o2o"] if source == "o2o" or out["cls_o2m"] is None else out["cls_o2m"]
        if not use_nms:
            topk = cfg.get("nms_topk")
            if topk is not None and 0 < int(topk) < z.shape[1]:
                fg = z[..., 1] - z[..., 0]
                keep = torch.zeros_like(fg, dtype=torch.bool)
                keep.scatter_(1, fg.topk(int(topk), dim=1).indices, True)
                z = torch.stack([z[..., 0], torch.where(keep, z[..., 1], z[..., 1] - 1e4)], dim=-1)
        preds = list(out["preds"])
        step = cfg.get("output_step")
        if step is not None:                             # evaluate an earlier output with the final scores
            if not 0 <= int(step) < len(preds):
                raise ValueError(f"test_cfg.output_step {step}: this head has outputs 0..{len(preds) - 1}")
            preds = preds[:int(step) + 1]
        last = preds[-1].clone()
        last[..., :2] = z.to(last.dtype)
        preds[-1] = last
        return StagePredictions([self.to_official_pred_dict(p) for p in preds])

    def forward(self, x, *args, **kwargs):
        return self.forward_train(x) if self.training else self.forward_test(x)

    # ------------------------------------------------------------------ losses
    def loss(self, x, batch_data_samples, *args, **kwargs):
        if self.training:
            self._train_step += 1
        outs = self.forward_train(x)
        dev = outs["cls_o2o"].device
        gts = x.get("gts")
        if gts is None:
            gts = self.extract_lanes_2d(batch_data_samples, dev)
        seg_gt = None
        if self.seg_loss_weight > 0:
            seg_gt = self.target_adapter.extract_seg(batch_data_samples, dev, outs["seg"].shape[-2:])
        losses = self.loss_by_outputs_2d(outs, gts, seg_gt)
        losses.update(self.foveal_losses(outs, x, gts))
        return losses

    def foveal_losses(self, outs, x, gts):
        """Proposal and vanishing-column losses and the foveal logs."""
        dev = outs["cls_o2o"].device
        layout = GTLayout([g.reshape(g.shape[0], 2 * self.gt_points) for g in gts], dev)
        gt_flat = layout.flat.view(layout.num, self.gt_points, 2)
        n_img = max(1.0, float(sum(1 for c in layout.counts if c > 0)))
        out = self._prop_loss(outs["prop"], layout, gt_flat, x["origin"], n_img)
        vp, vp_gt = x["vp"], x.get("vp_gt")
        zero = vp.sum() * 0.0
        out["loss_vp"] = zero
        if vp_gt is not None:
            ok = torch.isfinite(vp_gt)
            out["vp_gt_frac"] = ok.float().mean()
            if bool(ok.any()):
                err = (vp[ok] - vp_gt[ok]) / (0.5 * self.img_w)
                out["loss_vp"] = _smooth_l1(err, 0.0, self.vp_beta).mean() * self.vp_weight
                out["vp_err_px"] = (vp[ok] - vp_gt[ok]).detach().abs().mean()
        if outs.get("fovea_share") is not None:
            out["fovea_share"] = outs["fovea_share"]
        return out

    def _prop_owner(self, layout, gt_flat, frame, views):
        """GT lanes in px, bottom first [G, M, 2], their centres (arc-length midpoints) [G, 2] and the view whose
        heatmap owns each [G] (F where the centre is in the window)."""
        g_px = self._px(bottom_first(gt_flat.float()))
        m = g_px.shape[1]
        centre = 0.5 * (g_px[:, (m - 1) // 2] + g_px[:, m // 2])
        if views == 1:
            return g_px, centre, torch.zeros(layout.num, dtype=torch.long, device=centre.device)
        lane = torch.arange(layout.num, device=centre.device)
        in_win = frame.inside(centre[None].expand(layout.batch_size, -1, -1))[layout.img, lane]
        return g_px, centre, in_win.long()

    def _centre_cells(self, view, centre, img, frame, hc, wc):
        """Cells (column ci, row cj) of the view's hc x wc grid holding the centres [g, 2] (px) of images img [g],
        and the cell size (px)."""
        span_w, span_h = (self.img_w, self.img_h) if view == CTX else (self.win_w, self.win_h)
        sx, sy = span_w / wc, span_h / hc
        o = frame.origin[img].float() if view == FOV else centre.new_zeros(img.numel(), 2)
        ci = ((centre[:, 0] - o[:, 0]) / sx).floor().clamp(0, wc - 1).long()
        cj = ((centre[:, 1] - o[:, 1]) / sy).floor().clamp(0, hc - 1).long()
        return ci, cj, sx, sy

    def _prop_loss(self, p, layout, gt_flat, origin, n_img):
        """Proposal losses: per view, the CenterNet focal loss of its heatmap (Gaussian targets of the GT lanes it
        owns); at the cells within radius_cells of each owned centre (each cell to its nearest centre), the endpoint
        smooth-L1 and 1 - LaneIoU of the decoded proposal, weighted by the Gaussian, averaged per image."""
        cc = self.prop
        frame = ViewFrame(self.img_w, self.img_h, self.win_w, self.win_h,
                          origin.float() if len(p["hm"]) > 1 else None)
        batch_size = p["hm"][0].shape[0]
        out = {}
        zero = sum(r.sum() for r in p["reg"]) * 0.0
        if layout.num:
            g_px, centre, owner = self._prop_owner(layout, gt_flat, frame, len(p["hm"]))
            out["win_lane_frac"] = owner.float().mean() if len(p["hm"]) > 1 else zero.detach()
        num_pos = 0
        l_hm, cells = [], []
        for v, (hm, reg) in enumerate(zip(p["hm"], p["reg"])):
            hm = hm[:, 0]
            hc, wc = hm.shape[-2:]
            target = hm.new_zeros(batch_size, hc * wc)
            if layout.num:
                sel = torch.nonzero(owner == v)[:, 0]
                ci, cj, sx, sy = self._centre_cells(v, centre[sel], layout.img[sel], frame, hc, wc)
                if sel.numel():
                    gx = torch.arange(wc, device=hm.device)
                    gy = torch.arange(hc, device=hm.device)
                    d2 = ((gx[None, None, :] - ci[:, None, None]) ** 2
                          + (gy[None, :, None] - cj[:, None, None]) ** 2).float()
                    gauss = torch.exp(-d2 / (2.0 * cc["sigma"] ** 2)).flatten(1)
                    target = target.scatter_reduce(0, layout.img[sel][:, None].expand_as(gauss), gauss, "amax")
                    cells.append((v, sel, ci, cj, reg, hc, wc, sx, sy))
            pos = target == 1.0
            num_pos += int(pos.sum())
            xl = hm.flatten(1)
            l_pos = -F.logsigmoid(xl) * (1.0 - torch.sigmoid(xl)) ** 2
            l_neg = -F.logsigmoid(-xl) * torch.sigmoid(xl) ** 2 * (1.0 - target) ** 4
            l_hm.append(torch.where(pos, l_pos, l_neg).sum())
        out["loss_prop_hm"] = sum(l_hm) / float(max(1, num_pos)) * cc["hm_weight"]
        out["loss_prop_end"] = zero
        out["loss_prop_iou"] = zero
        if not layout.num:
            return out
        # regression cells of both views, pooled per image
        rad = cc["radius"]
        off = torch.arange(-rad, rad + 1, device=centre.device)
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        oy, ox = oy.flatten(), ox.flatten()
        dd0 = (oy ** 2 + ox ** 2).float()
        reg_rows, cell_px, scales, g_idx_all, w_all = [], [], [], [], []
        recall = torch.zeros(layout.num, dtype=torch.bool, device=centre.device)
        for v, sel, ci, cj, reg, hc, wc, sx, sy in cells:
            yy, xx = cj[:, None] + oy[None], ci[:, None] + ox[None]                         # [g, (2R + 1)^2]
            dd = dd0[None].expand_as(yy)
            gi = torch.arange(sel.numel(), device=yy.device)[:, None].expand_as(yy)
            inside = (yy >= 0) & (yy < hc) & (xx >= 0) & (xx < wc)
            yy, xx, dd, gi = yy[inside], xx[inside], dd[inside], gi[inside]
            img = layout.img[sel[gi]]
            key = img * (hc * wc) + yy * wc + xx
            order = torch.sort(dd, stable=True).indices
            order = order[torch.sort(key[order], stable=True).indices]
            key_s = key[order]
            first = torch.ones_like(key_s, dtype=torch.bool)
            first[1:] = key_s[1:] != key_s[:-1]
            pick = order[first]
            gi, yy, xx, dd, img = gi[pick], yy[pick], xx[pick], dd[pick], img[pick]
            o = frame.origin[img].float() if v == FOV else centre.new_zeros(img.numel(), 2)
            cell_px.append(torch.stack([(xx.float() + 0.5) * sx, (yy.float() + 0.5) * sy], dim=-1) + o)
            reg_rows.append(reg[img, :, yy, xx])
            scales.append(torch.full_like(dd, self._view_scale(v)))
            g_idx_all.append(sel[gi])
            w_all.append(torch.exp(-dd / (2.0 * cc["sigma"] ** 2)))
            with torch.no_grad():                    # a selected peak of this view within radius of the centre cell
                b_sel = layout.img[sel]
                q_cell = p["cell"][b_sel]                                                      # [g, N]
                q_ok = p["view"][b_sel] == v
                qy, qx = torch.div(q_cell, wc, rounding_mode="floor"), q_cell % wc
                near = (torch.maximum((qy - cj[:, None]).abs(), (qx - ci[:, None]).abs()) <= rad) & q_ok
                recall[sel] = near.any(dim=1)
        g_idx, w = torch.cat(g_idx_all), torch.cat(w_all)
        bottom, top, c_px = self._prop_decode(torch.cat(reg_rows), torch.cat(cell_px), torch.cat(scales))
        g_sel = g_px[g_idx]
        beta = self.smooth_l1_beta
        l_end = _smooth_l1(torch.stack([bottom - g_sel[:, 0], top - g_sel[:, -1]], dim=1), 0.0, beta).mean(dim=(-1, -2))
        curve = curve_points(self._norm(c_px), self.t_out_2d)
        l_iou = 1.0 - lane_iou_2d(curve * self.eval_scale, gt_flat[g_idx].float() * self.eval_scale,
                                  self.lane_width * self.iou_w)
        b = layout.img[g_idx]
        den = per_image_sum(w, b, batch_size).clamp_min(1e-6)

        def red(val):
            return (per_image_sum(w * val, b, batch_size) / den).sum() / n_img

        out["loss_prop_end"] = red(l_end) * cc["end_weight"]
        out["loss_prop_iou"] = red(l_iou) * cc["iou_weight"]
        out["prop_recall"] = recall.float().mean()
        if len(p["hm"]) > 1 and bool((owner == FOV).any()):
            out["prop_recall_fov"] = recall[owner == FOV].float().mean()        # the lanes centred in the window
        out["prop_fovea_frac"] = p["view"].float().mean()
        return out
