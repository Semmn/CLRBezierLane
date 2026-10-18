"""HBLRNet: BLRNet with 2D Bezier lanes and a lane -> segment anchor hierarchy
(clrbezierlane/hier_bezier_design.md).

No row representation anywhere in training:
- a lane is a cubic Bezier with 4 free 2D control points (no y ordering);
- its updates are in the lane's own frame: each control point moves along the curve's tangent and
  normal at its parameter (k / 3), in px;
- geometry losses and the matching cost use a symmetric 2D LaneIoU (lane_iou_2d), the start/length
  loss is a 2D endpoint loss, the relation attention reads 2D distances (LaneRelationAttention2D);
- the GT is the augmented annotation polyline (``gt_points`` in the train meta keys), clipped to the
  image and resampled by arc length.
The final curve is rendered to the official rows only for the test output (the CurveLanes metric
reads every lane as x(y)).

hier_cfg (None: lane level only, run A) adds the segment level (run B): after the lane steps each lane
curve is split exactly (de Casteljau, t = 1/3, 2/3) into 3 cubic segments with shared joints. Each
segment gets its own query (the lane state + a segment embedding + its geometry), its own decoder
block (relation attention over all segments of the image, curve attention along its own segment), its
own GRU state and update, in the lane frame. The joints are shared (G0) and their two handles are
projected onto one line after every update (C1). The lane is scored once, on its state plus the
pooled segment states. At the split the piecewise curve equals the lane curve exactly.

hier_cfg.schedule "chain" (run C) alternates the levels instead: lane layer, segment layer, lane layer, ...
(refine_cfg.num_iters lane layers, as many segment layers). The geometry is one lane cubic plus a segment
residual kept in the lane's own frame (per piece control point: along the tangent, along the normal, px);
the curve is the exact subdivision of the cubic plus the residual (shared joints averaged, C1 projected).
A lane layer moves the cubic, and the residual moves and turns with it; a segment layer changes only the
residual. The levels talk in a chain: before each lane layer (but the first) and before scoring, the
lane token reads its segment tokens (low -> high; their keys carry the segment's place and residual);
before each segment layer, each segment token reads its lane token and its siblings (high -> low). Both
exchanges start as the identity (zero-initialized output projections). Every lane layer's cubic gets the
control-point loss; every layer's curve gets the LaneIoU and endpoint losses.

curve_cfg.orientation_free (False): a lane has no direction. The GT runs bottom -> top, so the start of a
near-horizontal lane is set by a tiny slope (and flips under the rotation augmentation), while the IoU
loss and the matching ignore direction. With the switch on, the endpoint and control-point losses take
the better of the two GT directions per pair and step (MapTR's permutation-equivalent polylines), the
relation features read every lane bottom end first, and the test output turns each lane bottom -> top
before the rows. Logs end_rev_frac (o2o pairs scored against the reversed GT, last output).

Not supported here (raise): the row residual (point_offsets), the proposal, the alignment offsets,
strip / merge, the o2m branch, o2o_stop_grad, per-step matching (step_cls). step_blocks and
resample "fine" are required.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

from .batched_loss import GTLayout, batched_assign, per_image_sum
from .blrnet import (BLRHead, CurveDeformAttention, StepBlock, _MLP, anchor_control_points, bezier_end_slopes,
                     curve_points, focal_2cls, sample_points)
from .data_adapters import _lookup
from .geometry import bernstein_basis
from .head import _smooth_l1
from .lane_decoder import REL_FEATURES, LaneRelationAttention


# ------------------------------------------------------------------ 2D curve geometry
def split_cubic(c, u):
    """de Casteljau split of cubic Beziers c [..., 4, 2] at parameter u -> (left, right), each [..., 4, 2]."""
    p0, p1, p2, p3 = c.unbind(-2)
    p01, p12, p23 = p0 + u * (p1 - p0), p1 + u * (p2 - p1), p2 + u * (p3 - p2)
    p012, p123 = p01 + u * (p12 - p01), p12 + u * (p23 - p12)
    p0123 = p012 + u * (p123 - p012)
    return torch.stack([p0, p01, p012, p0123], -2), torch.stack([p0123, p123, p23, p3], -2)


def subdivide_cubic(cps, num_pieces):
    """Exact split of cubic Beziers cps [..., 4, 2] into num_pieces pieces at t = k / num_pieces ->
    piecewise control points [..., 3 * num_pieces + 1, 2] (piece s = rows 3s .. 3s + 3)."""
    pieces, rest = [], cps
    for k in range(1, num_pieces):
        left, rest = split_cubic(rest, 1.0 / float(num_pieces - k + 1))
        pieces.append(left)
    pieces.append(rest)
    return torch.cat([pieces[0]] + [p[..., 1:, :] for p in pieces[1:]], dim=-2)


def pieces_of(q):
    """Piecewise control points q [..., 3S + 1, 2] -> [..., S, 4, 2]."""
    num_pieces = (q.shape[-2] - 1) // 3
    return torch.stack([q[..., 3 * s:3 * s + 4, :] for s in range(num_pieces)], dim=-3)


def piecewise_points(q, per_piece):
    """Points of piecewise cubics q [..., 3S + 1, 2]: per_piece points per piece at local t = i / per_piece
    plus the end point -> [..., S * per_piece + 1, 2]. A single cubic split by subdivide_cubic gives the
    same points as curve_points at S * per_piece + 1 uniform t."""
    pc = pieces_of(q)
    t = torch.arange(per_piece, device=q.device, dtype=q.dtype) / float(per_piece)
    pts = curve_points(pc, t)                                                  # [..., S, n, 2]
    return torch.cat([pts.flatten(-3, -2), q[..., -1:, :]], dim=-2)


def piecewise_eval(q, t):
    """Points of piecewise cubics q [..., 3S + 1, 2] at global parameters t [K] (piece s covers
    [s / S, (s + 1) / S], local u = S t - s) -> [..., K, 2]. On a cubic split by subdivide_cubic this is
    curve_points of the cubic."""
    num_pieces = (q.shape[-2] - 1) // 3
    st = t.to(q.dtype) * float(num_pieces)
    piece = st.floor().clamp(0, num_pieces - 1)
    sel = pieces_of(q)[..., piece.long(), :, :]                                # [..., K, 4, 2]
    return (bernstein_basis(st - piece)[..., None] * sel).sum(-2)


def cubic_tangents(c, t):
    """Unit tangents of cubics c [..., 4, 2] (px) at parameters t [K] -> [..., K, 2]. Falls back to the
    chord, then to 'up', where the derivative vanishes."""
    t = t.to(c.dtype).view(-1, 1)
    d0, d1, d2 = c[..., 1, :] - c[..., 0, :], c[..., 2, :] - c[..., 1, :], c[..., 3, :] - c[..., 2, :]
    der = ((1 - t) ** 2 * d0[..., None, :] + 2 * (1 - t) * t * d1[..., None, :] + t ** 2 * d2[..., None, :])
    chord = (c[..., 3, :] - c[..., 0, :])[..., None, :].expand_as(der)
    up = torch.tensor([0.0, -1.0], device=c.device, dtype=c.dtype).expand_as(der)
    n_der, n_ch = der.norm(dim=-1, keepdim=True), chord.norm(dim=-1, keepdim=True)
    v = torch.where(n_der > 1e-3, der, torch.where(n_ch > 1e-3, chord, up))
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def frame_delta(c_px, upd):
    """Lane-frame update of cubics c_px [..., 4, 2] (px): upd [..., 4, 2] = (along the tangent, along the
    normal) per control point, at that point's parameter k / 3 -> the px deltas [..., 4, 2]."""
    tan = cubic_tangents(c_px.detach(), torch.linspace(0.0, 1.0, 4, device=c_px.device))
    nrm = torch.stack([-tan[..., 1], tan[..., 0]], dim=-1)
    return upd[..., :1] * tan + upd[..., 1:] * nrm


def c1_project(q_px, eps=1e-3):
    """Tangent continuity at the inner joints of piecewise cubics q_px [..., 3S + 1, 2] (px): the two
    handles of each joint go onto the line through the joint along (next handle - previous handle),
    keeping their lengths. Joints whose handles coincide are left as they are."""
    rows = list(q_px.unbind(-2))
    for j in range(3, q_px.shape[-2] - 1, 3):
        a, jt, c = rows[j - 1], rows[j], rows[j + 1]
        d = c - a
        n = d.norm(dim=-1, keepdim=True)
        u = d / n.clamp_min(eps)
        ok = n > eps
        rows[j - 1] = torch.where(ok, jt - (jt - a).norm(dim=-1, keepdim=True) * u, a)
        rows[j + 1] = torch.where(ok, jt + (c - jt).norm(dim=-1, keepdim=True) * u, c)
    return torch.stack(rows, dim=-2)


def curve_code(c_px, h_px):
    """Geometry code of cubics c_px [..., 4, 2] (px): the control points (normalized by the image height)
    plus the chord length and angle -> [..., 10]."""
    chord = c_px[..., 3, :] - c_px[..., 0, :]
    length = chord.norm(dim=-1, keepdim=True) / h_px
    angle = torch.atan2(-chord[..., 1:], chord[..., :1]) / math.pi
    return torch.cat([c_px.flatten(-2) / h_px, length, angle], dim=-1).detach()


# ------------------------------------------------------------------ 2D LaneIoU
def _point_to_polyline(p, poly):
    """Distance of points p [..., P, 2] to polylines poly [..., M, 2] (same leading dims) -> [..., P]."""
    s0, s1 = poly[..., :-1, :], poly[..., 1:, :]
    d = s1 - s0                                                                # [..., M-1, 2]
    rel = p[..., :, None, :] - s0[..., None, :, :]                             # [..., P, M-1, 2]
    den = (d * d).sum(-1).clamp_min(1e-12)[..., None, :]
    t = ((rel * d[..., None, :, :]).sum(-1) / den).clamp(0.0, 1.0)
    diff = rel - t[..., None] * d[..., None, :, :]
    return (diff * diff).sum(-1).add(1e-12).sqrt().min(-1).values


def _arc_weights(poly):
    """Arc length around each vertex of polylines [..., M, 2] (half of each adjacent segment) -> [..., M]."""
    seg = (poly[..., 1:, :] - poly[..., :-1, :]).norm(dim=-1)
    z = seg.new_zeros(seg.shape[:-1] + (1,))
    return 0.5 * (torch.cat([z, seg], -1) + torch.cat([seg, z], -1))


def lane_iou_2d(a, b, half_width, pairwise=False):
    """Symmetric 2D LaneIoU of polylines a [..., P, 2] and b [..., M, 2], in the same px space.

    Every sample of either lane is compared with the nearest point of the other lane (not the same row):
    overlap 2w - d, union 2w + d (CLRerNet's LaneIoU per sample), integrated over both lanes' arc length
    (vertex weights, no gradient). 1 for identical lanes; a parallel shift d gives (2w - d) / (2w + d);
    negative far away. pairwise: a [B, N, P, 2], b [B, G, M, 2] -> [B, N, G]; else paired -> [...]."""
    if pairwise:
        (batch_size, num_n, num_p), (num_g, num_m) = a.shape[:3], b.shape[1:3]
        a = a[:, :, None].expand(batch_size, num_n, num_g, num_p, 2)
        b = b[:, None].expand(batch_size, num_n, num_g, num_m, 2)
    w = float(half_width)
    d_a = _point_to_polyline(a, b)
    d_b = _point_to_polyline(b, a)
    wa, wb = _arc_weights(a).detach(), _arc_weights(b).detach()
    inter = (wa * (2 * w - d_a)).sum(-1) + (wb * (2 * w - d_b)).sum(-1)
    union = (wa * (2 * w + d_a)).sum(-1) + (wb * (2 * w + d_b)).sum(-1)
    return inter / union.clamp_min(1e-6)


# ------------------------------------------------------------------ GT polylines
def gt_polyline(flat, img_w, img_h, num_points, min_len_px=8.0, step_px=1.0):
    """One annotated lane (flat [x0, y0, x1, y1, ...], network px, bottom -> top) -> its longest run
    inside the image, resampled to num_points points uniform in arc length, normalized (x / (w - 1),
    y / (h - 1)) [num_points, 2]; None if shorter than min_len_px or fewer than 2 points."""
    p = np.asarray(flat, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return None
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])
    p = p[keep]
    if len(p) < 2:
        return None
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    n = max(2, int(math.ceil(cum[-1] / step_px)) + 1)
    s = np.linspace(0.0, cum[-1], n)
    dense = np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], axis=1)
    inside = ((dense[:, 0] >= 0) & (dense[:, 0] <= img_w - 1) & (dense[:, 1] >= 0) & (dense[:, 1] <= img_h - 1))
    if not inside.any():
        return None
    # the longest run of inside samples
    edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    k = int(np.argmax(ends - starts))
    run = dense[starts[k]:ends[k]]
    if len(run) < 2:
        return None
    rc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(run, axis=0), axis=1))])
    if rc[-1] < min_len_px:
        return None
    s = np.linspace(0.0, rc[-1], num_points)
    out = np.stack([np.interp(s, rc, run[:, 0]) / (img_w - 1), np.interp(s, rc, run[:, 1]) / (img_h - 1)], 1)
    return out.astype(np.float32)


@torch.no_grad()
def fit_cubic_2d(pts, ridge=1e-2):
    """Least-squares cubic Bezier through polylines pts [G, M, 2] sampled uniformly in arc length
    (chord-length parameters t = i / (M - 1)), with a ridge toward the chord -> [G, 4, 2]."""
    num_g, num_m = pts.shape[:2]
    t = torch.linspace(0.0, 1.0, num_m, device=pts.device, dtype=pts.dtype)
    basis = bernstein_basis(t)                                                 # [M, 4]
    frac = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], device=pts.device, dtype=pts.dtype)
    c_lin = pts[:, :1] + (pts[:, -1:] - pts[:, :1]) * frac.view(1, 4, 1)
    eye = torch.eye(4, device=pts.device, dtype=pts.dtype)
    lhs = (basis.T @ basis + ridge * eye).expand(num_g, 4, 4)
    return torch.linalg.solve(lhs, basis.T @ pts + ridge * c_lin)


def bottom_first(pts, *others):
    """Points [..., P, 2] (image y down) reversed where the first point is above the last; others [..., K, 2]
    are reversed with them. Returns pts alone, or (pts, *others)."""
    flip = (pts[..., 0, 1] < pts[..., -1, 1])[..., None, None]
    out = [torch.where(flip, x.flip(-2), x) for x in (pts,) + others]
    return out[0] if not others else tuple(out)


# ------------------------------------------------------------------ relation attention in 2D
class LaneRelationAttention2D(LaneRelationAttention):
    """LaneRelationAttention with 2D relation features. ``lanes`` are points [B, Q, P, 2] in px along
    each lane (bottom -> top). Neighbours: the nearest lanes by symmetric chamfer distance. Features
    (REL_FEATURES = 11, as the row version): chamfer distance both ways, the distances between the two
    starts and the two ends (in dist_scale units, capped at max_dist), the share of each lane's points
    within dist_scale of the other, the length difference (image widths), the neighbour's score, its
    difference to the query's, has_score, is_self. orientation_free (set by HBLRHead): each lane is read
    from its lower end, so the start / end features do not depend on its direction."""

    orientation_free = False

    @torch.no_grad()
    def neighbours(self, lanes, scores):
        if self.orientation_free:
            lanes = bottom_first(lanes)
        batch_size, num_q, num_p = lanes.shape[:3]
        dev = lanes.device
        d = torch.cdist(lanes.reshape(batch_size, num_q * num_p, 2),
                        lanes.reshape(batch_size, num_q * num_p, 2))
        d = d.view(batch_size, num_q, num_p, num_q, num_p).permute(0, 1, 3, 2, 4)  # [B, Q, Q, P, P]
        near_i = d.min(-1).values                                               # i's points to j
        near_j = d.min(-2).values                                               # j's points to i
        cham_ij, cham_ji = near_i.mean(-1), near_j.mean(-1)
        ov_ij = (near_i < self.dist_scale).float().mean(-1)
        ov_ji = (near_j < self.dist_scale).float().mean(-1)
        d_start, d_end = d[..., 0, 0], d[..., -1, -1]
        length = (lanes[..., 1:, :] - lanes[..., :-1, :]).norm(dim=-1).sum(-1)    # [B, Q]
        idx_q = torch.arange(num_q, device=dev)
        self_idx = idx_q.view(1, num_q, 1).expand(batch_size, num_q, 1)
        k = min(self.num_neighbors, num_q - 1)
        if k > 0:
            sym = 0.5 * (cham_ij + cham_ji)
            dist = torch.where(idx_q[:, None] != idx_q[None, :], sym, torch.full_like(sym, float("inf")))
            _, j = dist.topk(k, dim=-1, largest=False)
            nbr = torch.cat([self_idx, j], dim=-1)
        else:
            nbr = self_idx
        valid = torch.ones_like(nbr, dtype=torch.bool)
        if scores is None:
            s_i, has_score = torch.zeros((batch_size, num_q), device=dev), 0.0
        else:
            s_i, has_score = scores.float(), 1.0
        s_j = s_i.gather(1, nbr.reshape(batch_size, -1)).view_as(nbr).float()
        len_j = length.gather(1, nbr.reshape(batch_size, -1)).view_as(s_j)
        is_self = torch.zeros_like(s_j)
        is_self[..., 0] = 1.0
        h_norm = float(self.x_scale)
        f = torch.stack([
            cham_ij.gather(2, nbr) / self.dist_scale,
            cham_ji.gather(2, nbr) / self.dist_scale,
            d_start.gather(2, nbr) / self.dist_scale,
            d_end.gather(2, nbr) / self.dist_scale,
            ov_ij.gather(2, nbr),
            ov_ji.gather(2, nbr),
            (len_j - length[..., None]) / h_norm,
            s_j,
            s_j - s_i[..., None],
            torch.full_like(s_j, has_score),
            is_self,
        ], dim=-1)
        assert f.shape[-1] == REL_FEATURES
        f[..., :4] = f[..., :4].clamp(max=self.max_dist)
        return nbr, valid, f


class LevelAttention(nn.Module):
    """One lane's tokens at the two levels: queries [M, Lq, D] read keys / values [M, Lk, D] (plus key codes
    [M, Lk, D]), pre-norm; the output is added to the queries. The output projection starts at zero, so
    the exchange starts as the identity."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        for norm in (self.norm_q, self.norm_kv):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, q, kv, k_code):
        kv = self.norm_kv(kv)
        return q + self.attn(self.norm_q(q), kv + k_code, kv, need_weights=False)[0]


# ------------------------------------------------------------------ head
@MODELS.register_module()
class HBLRHead(BLRHead):
    """BLRHead with 2D Bezier lanes and an optional lane -> segment hierarchy (module docstring).

    curve_cfg (all optional): gt_points (48, arc-length samples of each GT lane), out_per_segment (24:
    output samples per third of the lane, so 3 * 24 + 1 points per lane), rel_points (12, per lane in the
    relation attention), cost_points (25, per lane in the matching cost), end_loss_weight (0.05, px
    smooth-L1 on the start and end points), min_gt_len_px (8), orientation_free (False: module docstring).
    The loss_cfg widths are reused:
    lane_width (loss) and lane_width_cost (matching), as half-widths in the iou_eval_shape space.
    hier_cfg: None, or segments (3), seg_iters (3, steps after the lane steps), seg_deform_points (12,
    curve-attention reference points per segment), seg_sample_points (12, re-sampled per segment),
    seg_neighbors (16), seg_rel_points (8), schedule ("sequential": the lane steps, then the segment
    steps; "chain": alternating lane / segment layers with the level exchange, seg_iters = num_iters)."""

    def __init__(self, curve_cfg=None, hier_cfg=None, **kwargs):
        r = dict(kwargs.get("refine_cfg") or {})
        problems = []
        if r.get("point_offsets", True):
            problems.append("refine_cfg.point_offsets must be False (no row residual)")
        if r.get("proposal", False):
            problems.append("refine_cfg.proposal is not supported")
        if not r.get("step_blocks", False):
            problems.append("refine_cfg.step_blocks must be True")
        if r.get("resample", "fine") != "fine":
            problems.append("refine_cfg.resample must be 'fine'")
        if r.get("step_cls") is not None:
            problems.append("refine_cfg.step_cls is not supported")
        for key in ("align_cfg", "strip_cfg", "merge_cfg", "o2m_branch_cfg"):
            if kwargs.get(key) is not None:
                problems.append(f"{key} must be None")
        if kwargs.get("o2o_stop_grad", False):
            problems.append("o2o_stop_grad is not supported")
        if problems:
            raise ValueError("HBLRHead: " + "; ".join(problems))
        cc = dict(curve_cfg or {})
        known = {"gt_points", "out_per_segment", "rel_points", "cost_points", "end_loss_weight", "min_gt_len_px",
                 "orientation_free"}
        if set(cc) - known:
            raise ValueError(f"HBLRHead curve_cfg: unknown keys {sorted(set(cc) - known)}")
        super().__init__(**kwargs)
        self.gt_points = int(cc.get("gt_points", 48))
        self.out_per_segment = int(cc.get("out_per_segment", 24))
        self.rel_points = int(cc.get("rel_points", 12))
        self.cost_points = int(cc.get("cost_points", 25))
        self.end_loss_weight = float(cc.get("end_loss_weight", 0.05))
        self.min_gt_len_px = float(cc.get("min_gt_len_px", 8.0))
        self.orientation_free = bool(cc.get("orientation_free", False))
        self.register_buffer("px_scale", torch.tensor([float(self.img_w - 1), float(self.img_h - 1)]),
                             persistent=False)
        self.register_buffer("eval_scale", torch.tensor([float(self.iou_w), float(self.iou_h)]), persistent=False)
        self.register_buffer("t_rel", torch.linspace(0.0, 1.0, self.rel_points), persistent=False)
        self.num_out = 3 * self.out_per_segment + 1
        self.register_buffer("t_out_2d", torch.linspace(0.0, 1.0, self.num_out), persistent=False)

        # anchors: free control points (the same initial curves as BLRHead)
        cps = self.anchor_control_points_init()
        del self.anchor_params
        self.anchor_cps = nn.Parameter(cps)                                    # [N, 4, 2] normalized

        # the segment level
        self.hier = None
        if hier_cfg is not None:
            self._build_hier(dict(hier_cfg), r, dict(kwargs.get("anchor_cfg") or {}))
        # the same weights; only the relation geometry changes to 2D
        for m in self.modules():
            if type(m) is LaneRelationAttention:
                m.__class__ = LaneRelationAttention2D
                if self.orientation_free:
                    m.orientation_free = True

    def anchor_control_points_init(self):
        return anchor_control_points(self.anchor_params.detach().float()).clone()

    def _build_hier(self, hc, r, a):
        known = {"segments", "seg_iters", "seg_deform_points", "seg_sample_points", "seg_neighbors",
                 "seg_rel_points", "schedule"}
        if set(hc) - known:
            raise ValueError(f"HBLRHead hier_cfg: unknown keys {sorted(set(hc) - known)}")
        segs = int(hc.get("segments", 3))
        if segs < 2:
            raise ValueError("HBLRHead hier_cfg.segments must be >= 2")
        self.hier = dict(segments=segs, iters=int(hc.get("seg_iters", 3)),
                         deform_points=int(hc.get("seg_deform_points", 12)),
                         sample_points=int(hc.get("seg_sample_points", 12)),
                         rel_points=int(hc.get("seg_rel_points", 8)))
        if self.hier["iters"] < 1:
            raise ValueError("HBLRHead hier_cfg.seg_iters must be >= 1")
        self.hier["schedule"] = str(hc.get("schedule", "sequential"))
        if self.hier["schedule"] not in ("sequential", "chain"):
            raise ValueError("HBLRHead hier_cfg.schedule must be 'sequential' or 'chain'")
        chain = self.hier["schedule"] == "chain"
        if chain and self.hier["iters"] != self.num_iters:
            raise ValueError("HBLRHead hier_cfg.schedule 'chain' alternates the levels: seg_iters must equal "
                             f"refine_cfg.num_iters ({self.num_iters})")
        dim = self.dim
        heads = int(a.get("num_heads", 8))
        rel_sb = dict(num_heads=heads, num_neighbors=int(hc.get("seg_neighbors", 16)), n_strips=self.n_strips,
                      x_scale=float(self.img_w - 1), dist_scale=float(a.get("dist_scale", 50.0)),
                      max_dist=float(a.get("max_dist", 10.0)), dropout=float(a.get("dropout", 0.0)))
        deform = dict(value_channels=self.prior_feat_channels, num_levels=self.deform_levels,
                      num_heads=int(r.get("deform_heads", 4)), num_points=int(r.get("deform_points", 4)),
                      x_scale=float(self.img_w - 1))
        self.seg_embed = nn.Parameter(torch.zeros(segs, dim))
        self.seg_geo = _MLP(10, dim)
        self.seg_norm = nn.LayerNorm(dim)
        self.seg_blocks = nn.ModuleList([StepBlock(dim, rel_sb, deform, int(r.get("step_ffn_dim", 1024)),
                                                   rel_sb["dropout"]) for _ in range(self.hier["iters"])])
        self.seg_step_pos = nn.Parameter(torch.zeros(self.hier["deform_points"], dim))
        self.seg_delta_enc = _MLP(8, dim)
        self.seg_resample_enc = nn.Sequential(
            nn.Linear(self.prior_feat_channels * self.hier["sample_points"], dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim))
        self.seg_gru = nn.GRUCell(3 * dim, dim)
        self.seg_update = nn.Linear(dim, 8)
        if chain:
            # low -> high before each lane layer but the first and before scoring; high -> low before
            # each segment layer
            self.up_attn = nn.ModuleList([LevelAttention(dim, heads, rel_sb["dropout"])
                                          for _ in range(self.num_iters)])
            self.down_attn = nn.ModuleList([LevelAttention(dim, heads, rel_sb["dropout"])
                                            for _ in range(self.hier["iters"])])
            self.seg_res_code = _MLP(8, dim)                     # a segment's residual (lane frame) in its key
            self.lane_code = nn.Parameter(torch.zeros(dim))     # the lane token's key code
        else:
            self.seg_pool = nn.Linear(dim, dim)
        self.register_buffer("t_seg_deform", torch.linspace(0.0, 1.0, self.hier["deform_points"]), persistent=False)
        self.register_buffer("t_seg_sample", torch.linspace(0.0, 1.0, self.hier["sample_points"]), persistent=False)
        self.register_buffer("t_seg_rel", torch.linspace(0.0, 1.0, self.hier["rel_points"]), persistent=False)
        self._reset_hier()

    def _reset_hier(self):
        for mod in (self.seg_geo, self.seg_blocks, self.seg_delta_enc, self.seg_resample_enc, self.seg_norm):
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for m in self.seg_blocks.modules():          # after the Linear loop, as in reset_parameters
            if isinstance(m, CurveDeformAttention):
                m.reset_parameters()
        nn.init.normal_(self.seg_embed, std=0.02)
        nn.init.trunc_normal_(self.seg_step_pos, std=0.02)
        nn.init.normal_(self.seg_update.weight, std=1.0e-2)       # as the lane update: sub-pixel at init
        nn.init.zeros_(self.seg_update.bias)
        if self.hier["schedule"] == "chain":
            for m in self.seg_res_code.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            nn.init.normal_(self.lane_code, std=0.02)
            for m in list(self.up_attn) + list(self.down_attn):
                m.reset_parameters()
        else:
            nn.init.zeros_(self.seg_pool.weight)                   # the lane state starts as without segments
            nn.init.zeros_(self.seg_pool.bias)

    # ------------------------------------------------------------------ pieces
    def _px(self, xy):
        return xy * self.px_scale.to(xy.dtype)

    def _norm(self, xy_px):
        return xy_px / self.px_scale.to(xy_px.dtype)

    def _layout(self, pts, cps_ends):
        """Official rows of points [B, N, P, 2] for the test output; cps_ends [B, N, 4, 2] give the end
        tangents (bezier_end_slopes) that continue the lane beyond its ends."""
        zeros2 = pts.new_zeros(pts.shape[:2] + (2,))
        return self.official_layout(zeros2, pts, bezier_end_slopes(cps_ends))

    def _forward(self, feats, img_tok=None, branch="main", strip_maps=None, fine=None):
        batch_size = feats[0].shape[0]
        num_q, dim = self.num_anchors, self.dim
        cps0 = self.anchor_cps[None].expand(batch_size, -1, -1, -1).float()      # [B, N, 4, 2]
        pts_s = curve_points(cps0, self.t_sample)
        sem = self.anchor_embed[None].expand(batch_size, -1, -1) + self.geo_enc(curve_code(self._px(cps0), self.img_h - 1))
        if img_tok is None:
            img_tok = self._image_tokens(feats)
        sem = self.sem_pre_norm[0](sem)
        sem = self.sem_pre_norm[1](sem + self.sem_pre(sem, img_tok, img_tok, need_weights=False)[0])
        raw = torch.cat([sample_points(f, pts_s.detach()).float() for f in feats], dim=-1)
        tok = self.point_norm(self.point_proj(raw) + self.point_pos)
        geo = self.geo_vec(self.geo_point(raw).flatten(2))
        rel0 = self._px(curve_points(cps0, self.t_rel)).detach()
        for block in self.blocks:
            sem, geo = block(sem, geo, tok, img_tok, rel0)
        h = self.fuse(torch.cat([sem, geo], dim=-1))

        # lane steps
        fine = feats[-1] if fine is None else fine
        deform_feats = feats[len(feats) - self.deform_levels:]
        if self.hier is not None and self.hier["schedule"] == "chain":
            return self._forward_chain(h, cps0, raw, feats, fine, deform_feats)
        hf = h.reshape(batch_size * num_q, dim)
        cps, cur_pts = cps0, pts_s.detach()
        curves, cps_list, ends, step_off, step_w = [], [], [], [], []
        for i in range(self.num_iters):
            ref = (cur_pts if self.deform_curve_points == self.sample_points
                   else curve_points(cps, self.t_deform).detach())
            rel = self._px(curve_points(cps, self.t_rel)).detach()
            hb = self.step_blocks[i](hf.view(batch_size, num_q, dim), rel, self.step_pos, ref, deform_feats)
            hf = hb.reshape(batch_size * num_q, dim)
            step_off.append(self.step_blocks[i].curve.last_offset_px)
            step_w.append(self.step_blocks[i].sa.last_self_weight)
            inp = [hf, self.delta_enc((cps - cps0).detach().flatten(-2)).reshape(-1, dim),
                   self.resample_enc(sample_points(fine, cur_pts).float().flatten(2)).reshape(-1, dim)]
            hf = self.gru(torch.cat(inp, dim=-1), hf)
            upd = self.update(hf).view(batch_size, num_q, 4, 2).float() * self.step_scale   # px, lane frame
            cps = self._norm(self._px(cps) + frame_delta(self._px(cps), upd))
            curves.append(curve_points(cps, self.t_out_2d))
            cps_list.append(cps)
            ends.append(cps)
            cur_pts = curve_points(cps, self.t_sample).detach()
        h = hf.view(batch_size, num_q, dim)
        num_lane_steps = len(curves)

        # segment steps
        seg_move = None
        if self.hier is not None:
            hc, segs = self.hier, self.hier["segments"]
            q = subdivide_cubic(cps, segs)                                        # [B, N, 3S+1, 2]
            q_start = q.detach()
            pc0 = pieces_of(q)
            hs = self.seg_norm(h[:, :, None] + self.seg_embed + self.seg_geo(curve_code(self._px(pc0), self.img_h - 1)))
            hs = hs.reshape(batch_size * num_q * segs, dim)
            index = torch.tensor([3 * s + k for s in range(segs) for k in range(4)], device=q.device)
            count = torch.zeros(3 * segs + 1, device=q.device).index_add_(0, index, torch.ones_like(index, dtype=torch.float))
            moves = []
            for j in range(hc["iters"]):
                pc = pieces_of(q)                                                 # [B, N, S, 4, 2]
                rel = self._px(curve_points(pc, self.t_seg_rel)).reshape(batch_size, num_q * segs, -1, 2).detach()
                ref = curve_points(pc, self.t_seg_deform).reshape(batch_size, num_q * segs, -1, 2).detach()
                hb = self.seg_blocks[j](hs.view(batch_size, num_q * segs, dim), rel, self.seg_step_pos, ref,
                                        deform_feats)
                hs = hb.reshape(-1, dim)
                step_off.append(self.seg_blocks[j].curve.last_offset_px)
                step_w.append(self.seg_blocks[j].sa.last_self_weight)
                pts_seg = curve_points(pc, self.t_seg_sample).reshape(batch_size, num_q * segs, -1, 2).detach()
                inp = [hs, self.seg_delta_enc((pc - pieces_of(q_start)).detach().flatten(-2)).reshape(-1, dim),
                       self.seg_resample_enc(sample_points(fine, pts_seg).float().flatten(2)).reshape(-1, dim)]
                hs = self.seg_gru(torch.cat(inp, dim=-1), hs)
                upd = self.seg_update(hs).view(batch_size, num_q, segs, 4, 2).float() * self.step_scale
                delta = frame_delta(self._px(pc), upd)                            # [B, N, S, 4, 2] px
                d_q = delta.new_zeros(batch_size, num_q, 3 * segs + 1, 2).index_add(2, index, delta.flatten(2, 3))
                d_q = d_q / count.view(1, 1, -1, 1)                               # shared joints: the mean
                q = self._norm(c1_project(self._px(q) + d_q))
                moves.append(d_q.detach().norm(dim=-1).mean())
                curves.append(piecewise_points(q, self.out_per_segment))
                cps_list.append(q)
                ends.append(torch.cat([q[..., :2, :], q[..., -2:, :]], dim=-2))
            h = h + self.seg_pool(hs.view(batch_size, num_q, segs, dim).mean(2))
            seg_move = torch.stack(moves).mean()
        lane_cps = cps_list[:num_lane_steps] + [None] * (len(curves) - num_lane_steps)
        return self._finish(h, raw, feats, curves, cps_list, ends, lane_cps, num_lane_steps, step_off, step_w,
                            seg_move=seg_move)

    def _finish(self, h, raw, feats, curves, cps_list, ends, lane_cps, num_lane_steps, step_off, step_w,
                seg_move=None, seg_res=None):
        """Scores and the output dict. lane_cps: per output, the lane cubic [B, N, 4, 2] that gets the
        control-point loss, or None."""
        # scores: o2m per anchor; o2o after the converged lanes have interacted
        cls_o2m, s_o2m = None, None
        if self.cls_o2m is not None:
            cls_o2m = self.cls_o2m(h).float()
            with torch.no_grad():
                s_o2m = torch.softmax(cls_o2m, dim=-1)[..., 1]
        sub = torch.linspace(0, self.num_out - 1, self.rel_points, device=h.device).round().long()
        rel_fin = self._px(curves[-1][:, :, sub]).detach()
        h_fin = self.final_norm(h + self.final_sa(h, rel_fin, s_o2m))
        cls_o2o = self.cls_o2o(h_fin).float()
        if self.orientation_free:
            preds = [self._layout(*bottom_first(c, e)) for c, e in zip(curves, ends)]
        else:
            preds = [self._layout(c, e) for c, e in zip(curves, ends)]
        return dict(preds=preds, curves=curves, cps=cps_list, num_lane_steps=num_lane_steps, lane_cps=lane_cps,
                    cls_o2o=cls_o2o, cls_o2m=cls_o2m, final_pred=None, strip=None, merge=None,
                    h=h, offsets=None, raw_feats=raw, align_feats=raw, feats=feats, prop_px=None,
                    seg_move_px=seg_move, seg_res_px=seg_res,
                    step_off_px=torch.stack(step_off).mean(),
                    step_self_w=(torch.stack([w for w in step_w if w is not None]).mean()
                                 if all(w is not None for w in step_w) else None),
                    enc_rel_delta=(self.encoder.last_rel_delta if self.encoder is not None else None),
                    step_cls=None,
                    self_w=[b.sem_sa.last_self_weight for b in self.blocks] + [self.final_sa.last_self_weight])

    def _joint_index(self, segs, device):
        """Rows of the piecewise control points that each piece's 4 points map to, and how many pieces
        share each row (2 at the inner joints)."""
        index = torch.tensor([3 * s + k for s in range(segs) for k in range(4)], device=device)
        count = torch.zeros(3 * segs + 1, device=device).index_add_(0, index, torch.ones_like(index, dtype=torch.float))
        return index, count

    def _compose(self, c, res, index, count):
        """The chain's curve: the exact split of the lane cubics c [B, N, 4, 2] (normalized) plus the segment
        residual res [B, N, S, 4, 2] (lane frame px per piece control point), shared joints averaged, C1
        projected -> piecewise control points [B, N, 3S + 1, 2] (normalized)."""
        segs = res.shape[2]
        base = self._px(subdivide_cubic(c, segs))
        delta = frame_delta(pieces_of(base), res)                                   # [B, N, S, 4, 2] px
        d_q = delta.new_zeros(base.shape).index_add(2, index, delta.flatten(2, 3)) / count.view(1, 1, -1, 1)
        return self._norm(c1_project(base + d_q))

    def _forward_chain(self, h, cps0, raw, feats, fine, deform_feats):
        """hier_cfg.schedule "chain": lane layer i, then segment layer i, for i < num_iters (module docstring)."""
        batch_size, num_q, dim = h.shape[0], self.num_anchors, self.dim
        segs = self.hier["segments"]
        index, count = self._joint_index(segs, h.device)
        res_norm = float(self.img_h - 1)
        c = cps0
        res = cps0.new_zeros(batch_size, num_q, segs, 4, 2)
        q = self._compose(c, res, index, count)
        hs = self.seg_norm(h[:, :, None] + self.seg_embed
                           + self.seg_geo(curve_code(self._px(pieces_of(q)), self.img_h - 1)))
        hs = hs.reshape(batch_size * num_q * segs, dim)
        hf = h.reshape(batch_size * num_q, dim)

        def seg_keys():
            return (self.seg_embed + self.seg_res_code((res / res_norm).detach().flatten(-2))).reshape(-1, segs, dim)

        curves, cps_list, lane_cps, ends, step_off, step_w, moves = [], [], [], [], [], [], []

        def record(cubic):
            curves.append(piecewise_points(q, self.out_per_segment))
            cps_list.append(q)
            lane_cps.append(cubic)
            ends.append(torch.cat([q[..., :2, :], q[..., -2:, :]], dim=-2))

        for i in range(self.num_iters):
            # lane layer: low -> high, then read along the whole curve and move the cubic
            if i > 0:
                hf = self.up_attn[i - 1](hf[:, None], hs.view(-1, segs, dim), seg_keys())[:, 0]
            pts_q = piecewise_eval(q, self.t_sample).detach()
            ref = (pts_q if self.deform_curve_points == self.sample_points
                   else piecewise_eval(q, self.t_deform).detach())
            rel = self._px(piecewise_eval(q, self.t_rel)).detach()
            hb = self.step_blocks[i](hf.view(batch_size, num_q, dim), rel, self.step_pos, ref, deform_feats)
            hf = hb.reshape(batch_size * num_q, dim)
            step_off.append(self.step_blocks[i].curve.last_offset_px)
            step_w.append(self.step_blocks[i].sa.last_self_weight)
            inp = [hf, self.delta_enc((c - cps0).detach().flatten(-2)).reshape(-1, dim),
                   self.resample_enc(sample_points(fine, pts_q).float().flatten(2)).reshape(-1, dim)]
            hf = self.gru(torch.cat(inp, dim=-1), hf)
            upd = self.update(hf).view(batch_size, num_q, 4, 2).float() * self.step_scale   # px, lane frame
            c = self._norm(self._px(c) + frame_delta(self._px(c), upd))
            q = self._compose(c, res, index, count)
            record(c)

            # segment layer: high -> low, then each segment reads along itself and moves the residual
            kv = torch.cat([hf[:, None], hs.view(-1, segs, dim)], dim=1)
            k_code = torch.cat([self.lane_code.view(1, 1, dim).expand(kv.shape[0], 1, dim), seg_keys()], dim=1)
            hs = self.down_attn[i](hs.view(-1, segs, dim), kv, k_code).reshape(-1, dim)
            pc = pieces_of(q)                                                     # [B, N, S, 4, 2]
            rel = self._px(curve_points(pc, self.t_seg_rel)).reshape(batch_size, num_q * segs, -1, 2).detach()
            ref = curve_points(pc, self.t_seg_deform).reshape(batch_size, num_q * segs, -1, 2).detach()
            hb = self.seg_blocks[i](hs.view(batch_size, num_q * segs, dim), rel, self.seg_step_pos, ref,
                                    deform_feats)
            hs = hb.reshape(-1, dim)
            step_off.append(self.seg_blocks[i].curve.last_offset_px)
            step_w.append(self.seg_blocks[i].sa.last_self_weight)
            pts_seg = curve_points(pc, self.t_seg_sample).reshape(batch_size, num_q * segs, -1, 2).detach()
            inp = [hs, self.seg_delta_enc((res / res_norm).detach().flatten(-2)).reshape(-1, dim),
                   self.seg_resample_enc(sample_points(fine, pts_seg).float().flatten(2)).reshape(-1, dim)]
            hs = self.seg_gru(torch.cat(inp, dim=-1), hs)
            upd_s = self.seg_update(hs).view(batch_size, num_q, segs, 4, 2).float() * self.step_scale
            res = res + upd_s
            q = self._compose(c, res, index, count)
            moves.append(upd_s.detach().norm(dim=-1).mean())
            record(None)

        hf = self.up_attn[-1](hf[:, None], hs.view(-1, segs, dim), seg_keys())[:, 0]   # low -> high, for scoring
        return self._finish(hf.view(batch_size, num_q, dim), raw, feats, curves, cps_list, ends, lane_cps,
                            self.num_iters, step_off, step_w, seg_move=torch.stack(moves).mean(),
                            seg_res=res.detach().norm(dim=-1).mean())

    # ------------------------------------------------------------------ targets
    def extract_lanes_2d(self, batch_data_samples, device):
        """The augmented annotation polylines (``gt_points``) of each image -> list of [G_b, M, 2]
        normalized (at most num_anchors lanes per image)."""
        out = []
        for sample in batch_data_samples:
            value, _ = _lookup(sample, ["gt_points"])
            if value is None:
                raise KeyError("HBLRHead needs 'gt_points' in the train meta_keys (PackCLRNetInputs)")
            lanes = [gt_polyline(lane, self.img_w, self.img_h, self.gt_points, self.min_gt_len_px)
                     for lane in value]
            lanes = [ln for ln in lanes if ln is not None][:self.num_anchors]
            out.append(torch.as_tensor(np.stack(lanes), device=device) if lanes
                       else torch.zeros((0, self.gt_points, 2), device=device))
        return out

    def loss(self, x, batch_data_samples, *args, **kwargs):
        if self.training:
            self._train_step += 1
        feats = self._select_features(x)
        extra = [self._fine_input(x)] if hasattr(self, "_fine_input") else []
        outs = self.forward_train(feats, self._strip_maps(x), *extra)
        device = feats[0].device
        gts = self.extract_lanes_2d(batch_data_samples, device)
        seg_gt = None
        if self.seg_loss_weight > 0 and "seg" in outs:
            seg_gt = self.target_adapter.extract_seg(batch_data_samples, device, outs["seg"].shape[-2:])
        return self.loss_by_outputs_2d(outs, gts, seg_gt)

    # ------------------------------------------------------------------ losses
    def _cost_cache(self, logits, curve, layout, gt_flat):
        """Matching cost terms [B, N, Gmax] of the curves [B, N, P, 2] against the image's GT."""
        batch_size, num_q = curve.shape[:2]
        sub_p = torch.linspace(0, curve.shape[2] - 1, self.cost_points, device=curve.device).round().long()
        sub_g = torch.linspace(0, self.gt_points - 1, self.cost_points, device=curve.device).round().long()
        a = curve[:, :, sub_p].float() * self.eval_scale
        gmax = layout.gmax
        gt = curve.new_zeros(batch_size, gmax, self.cost_points, 2)           # padding: a dummy vertical line
        gt[:, :, :, 0] = 0.5
        gt[:, :, :, 1] = torch.linspace(1.0, 0.0, self.cost_points, device=curve.device)
        if layout.num:
            gt[layout.img, layout.slot] = gt_flat[:, sub_g].float()
        gt = gt * self.eval_scale
        iou_cost = lane_iou_2d(a, gt, self.lane_width_cost * self.iou_w, pairwise=True)
        iou_dyn = lane_iou_2d(a, gt, self.lane_width * self.iou_w, pairwise=True)
        fg = torch.softmax(logits.float(), dim=-1)[..., 1].clamp(1e-6, 1.0 - 1e-6)
        cls_cost = torch.nan_to_num(-torch.log(fg), nan=100.0, posinf=100.0, neginf=100.0)
        return {"cls_cost": cls_cost[..., None].expand(-1, -1, gmax),
                "point_cost": torch.zeros_like(iou_cost),
                "iou_cost_assign": 1.0 - torch.nan_to_num(iou_cost, nan=0.0),
                "lane_iou_dynamic": torch.nan_to_num(iou_dyn, nan=0.0)}

    def _pair_terms(self, outs, b, n, j, gt_flat, gt_cps, batch_size, rev_out=None):
        """Per-image means over the pairs (b, n) <- GT j, averaged over the steps: 1 - LaneIoU, endpoint
        and control-point smooth-L1 (the outputs with a lane cubic in outs["lane_cps"]) -> three [B] tensors.
        orientation_free: the endpoint and control-point terms against the better GT direction per pair and
        step (jointly where both exist); rev_out (a list) receives the last output's reversed-pair share."""
        cnt = per_image_sum(torch.ones_like(b, dtype=torch.float), b, batch_size).clamp_min(1.0)
        g = gt_flat[j].float()
        g_eval, g_px = g * self.eval_scale, self._px(g)
        zero = g.new_zeros(batch_size)
        iou = end = cp = zero
        n_steps = float(len(outs["curves"]))
        n_cp = sum(c is not None for c in outs["lane_cps"])
        for k, (curve, c) in enumerate(zip(outs["curves"], outs["lane_cps"])):
            p = curve[b, n].float()
            l_iou = 1.0 - lane_iou_2d(p * self.eval_scale, g_eval, self.lane_width * self.iou_w)
            iou = iou + per_image_sum(l_iou, b, batch_size) / cnt
            p_px = self._px(p)
            e = torch.stack([p_px[:, 0] - g_px[:, 0], p_px[:, -1] - g_px[:, -1]], dim=1)
            l_end = _smooth_l1(e, 0.0, self.smooth_l1_beta).mean(dim=(-1, -2))
            if c is not None:
                l_cp = _smooth_l1(self._px(c[b, n].float()) - self._px(gt_cps[j]), 0.0,
                                  self.smooth_l1_beta).mean(dim=(-1, -2))
            if self.orientation_free:
                e_r = torch.stack([p_px[:, 0] - g_px[:, -1], p_px[:, -1] - g_px[:, 0]], dim=1)
                l_end_r = _smooth_l1(e_r, 0.0, self.smooth_l1_beta).mean(dim=(-1, -2))
                if c is not None:
                    l_cp_r = _smooth_l1(self._px(c[b, n].float()) - self._px(gt_cps[j].flip(-2)), 0.0,
                                        self.smooth_l1_beta).mean(dim=(-1, -2))
                    rev = (l_end_r + l_cp_r < l_end + l_cp).detach()
                    l_cp = torch.where(rev, l_cp_r, l_cp)
                else:
                    rev = (l_end_r < l_end).detach()
                l_end = torch.where(rev, l_end_r, l_end)
                if rev_out is not None and k == len(outs["curves"]) - 1 and rev.numel():
                    rev_out.append(rev.float().mean())
            end = end + per_image_sum(l_end, b, batch_size) / cnt
            if c is not None:
                cp = cp + per_image_sum(l_cp, b, batch_size) / cnt
        return iou / n_steps, end / n_steps, cp / float(max(1, n_cp))

    def loss_by_outputs_2d(self, outs, gts, seg_gt=None):
        batch_size, num_q = outs["cls_o2o"].shape[:2]
        dev = outs["cls_o2o"].device
        layout = GTLayout([g.reshape(g.shape[0], 2 * self.gt_points) for g in gts], dev)
        gt_flat = layout.flat.view(layout.num, self.gt_points, 2)
        gt_counts = torch.tensor([float(c) if c > 0 else 1.0 for c in layout.counts], device=dev)
        n_img = max(1.0, float(sum(1 for c in layout.counts if c > 0)))
        tgt_o2o = torch.zeros((batch_size, num_q), dtype=torch.long, device=dev)
        tgt_o2m = torch.zeros_like(tgt_o2o)
        zero = outs["cls_o2o"].new_zeros(())
        iou_sum = end_sum = cp_sum = zero
        num_pos_o2o = num_pos_o2m = 0
        rev_o2o = []
        if layout.num:
            gt_cps = fit_cubic_2d(gt_flat.float(), self.cp_fit_ridge)
            final = outs["curves"][-1].detach()
            with torch.no_grad():
                cache = self._cost_cache(outs["cls_o2o"].detach(), final, layout, gt_flat)
            b_o, n_o, g_o = batched_assign(self.o2o_assigner, cache, layout)
            tgt_o2o[b_o, n_o] = 1
            num_pos_o2o = int(n_o.numel())
            iou, end, cp = self._pair_terms(outs, b_o, n_o, layout.flat_index(b_o, g_o), gt_flat, gt_cps, batch_size,
                                            rev_out=rev_o2o)
            iou_sum, end_sum, cp_sum = iou.sum(), end.sum(), cp.sum()
            if self.o2m_assigner is not None:
                with torch.no_grad():
                    cache_m = dict(cache)
                    fg = torch.softmax(outs["cls_o2m"].detach().float(), -1)[..., 1].clamp(1e-6, 1.0 - 1e-6)
                    cache_m["cls_cost"] = torch.nan_to_num(-torch.log(fg), nan=100.0)[..., None].expand_as(cache["cls_cost"])
                b_m, n_m, g_m = batched_assign(self.o2m_assigner, cache_m, layout)
                tgt_o2m[b_m, n_m] = 1
                num_pos_o2m = int(n_m.numel())
                if n_m.numel() and self.o2m_reg_weight > 0:
                    iou, end, cp = self._pair_terms(outs, b_m, n_m, layout.flat_index(b_m, g_m), gt_flat, gt_cps,
                                                    batch_size)
                    w = self.o2m_reg_weight
                    iou_sum, end_sum, cp_sum = iou_sum + w * iou.sum(), end_sum + w * end.sum(), cp_sum + w * cp.sum()
        cls_o2o = (focal_2cls(outs["cls_o2o"], tgt_o2o, self.focal_alpha, self.focal_gamma).sum(-1) / gt_counts).mean()
        losses = dict(loss_cls=cls_o2o * self.cls_loss_weight,
                      loss_iou=iou_sum / n_img * self.iou_loss.loss_weight,
                      loss_end=end_sum / n_img * self.end_loss_weight,
                      loss_cp=cp_sum / n_img * self.cp_loss_weight)
        if self.o2m_assigner is not None:
            cls_o2m = (focal_2cls(outs["cls_o2m"], tgt_o2m, self.focal_alpha, self.focal_gamma).sum(-1)
                       / gt_counts).mean()
            losses["loss_cls_o2m"] = cls_o2m * self.cls_loss_weight * self.o2m_cls_weight
        if seg_gt is not None and "seg" in outs:
            losses["loss_seg"] = F.nll_loss(F.log_softmax(outs["seg"], dim=1), seg_gt, weight=self.seg_class_weights,
                                            ignore_index=self.seg_ignore_label) * self.seg_loss_weight
        with torch.no_grad():
            diag = dict(num_pos_o2o=torch.tensor(float(num_pos_o2o), device=dev),
                        num_pos_o2m=torch.tensor(float(num_pos_o2m), device=dev),
                        num_gt=torch.tensor(float(layout.num), device=dev))
            for i, w in enumerate(outs["self_w"]):
                diag[f"self_w_{'final' if i == len(outs['self_w']) - 1 else i}"] = w
            for key in ("step_off_px", "step_self_w", "enc_rel_delta", "seg_move_px", "seg_res_px"):
                if outs.get(key) is not None:
                    diag[key] = outs[key].detach()
            pos = tgt_o2o.bool()
            if bool(pos.any()):
                p = torch.softmax(outs["cls_o2o"], -1)[..., 1]
                diag["o2o_pos_prob"], diag["o2o_neg_prob"] = p[pos].mean(), p[~pos].mean()
            if self.orientation_free:
                diag["end_rev_frac"] = rev_o2o[0].detach() if rev_o2o else torch.zeros((), device=dev)
        losses.update(diag)
        return losses

    def loss_by_outputs(self, outs, lanes, seg_gt=None):
        raise RuntimeError("HBLRHead trains on 2D lanes: use loss() / loss_by_outputs_2d()")
