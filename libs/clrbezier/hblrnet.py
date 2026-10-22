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

curve_cfg.y_order / frame / match_end_len (YSY, 2026-10-18; defaults keep the head as above):
- frame "image": every update (lane and segment level) is in the image axes, px per control point, instead
  of the lane's own tangent / normal; the chain's residual is then in the image axes too. The levels share
  one frame and the control-point target lives in it.
- y_order "monotone" (needs frame "image"): every curve is monotone in y by construction. A chain of
  control points (the lane's 4, or the 3S + 1 of the segments) keeps its bottom point, and each next
  point's y is the previous minus a softplus increment; x is free. Updates move x (px), the bottom y (px)
  and the increments' pre-activations. A split of a monotone cubic is monotone, so the segments stay one
  monotone chain with shared joints; the G1 handles are shortened where needed to keep the y order. The
  anchors are made monotone (cumulative min of y). Lanes run bottom -> top, as the GT does.
- y_order "penalty": control points stay free and a hinge on every rise of y along them (px, every output,
  every anchor) is added as loss_order (order_loss_weight, 0.05).
- match_end_len: the matching cost's point term (assign_cfg point_weight) becomes the endpoint distance
  (mean of the two ends, px / (img_h - 1); the better direction with orientation_free) plus the length
  term |len_pred / len_gt - 1| (at most 1), as in BezierFormer's matching cost.
- factorized_iou (YSY, 2026-10-22; clrbezierlane/extent_shape/extent_shape_design.md, idea B): a prediction's
  lateral position and its extent are judged apart, along the GT lane (lane_iou_factorized). IoU_lat is the 2D
  LaneIoU over the overlap only (the samples of either lane whose foot on the other lies inside it), IoU_ext the 1D
  IoU of the two arc intervals along the GT (the prediction's ends projected on the GT, its end segments run on as
  lines). Matching cost: 1 - IoU_lat x IoU_ext (1 - IoU_lat where IoU_lat <= 0) at lane_width_cost, in place of
  1 - LaneIoU; the metric's mask IoU of a shifted, cut copy is about that product, while the whole-lane LaneIoU
  charges every uncovered GT sample its distance to the prediction's end. Loss of every step's pairs:
  (1 - IoU_lat) + extent_iou_weight (1.0) x (1 - IoU_ext) at lane_width, in place of 1 - LaneIoU; the lateral
  term gets no pull from the part one lane lacks, and the extent term moves the ends along the GT. The endpoint /
  length terms stay as configured (the B configs set point_weight 0 and end_loss_weight 0). Logs o2o_lat_iou /
  o2o_ext_iou (last output, o2o pairs, at lane_width).

pattern_cfg (None; needs frame "image" and y_order "monotone"; hier schedule "sequential" only): lanes as pose x
pattern (clrbezierlane/pose_pattern_design.md, libs/clrbezier/pattern.py). A lane cubic is its pose (bottom and
top points, image axes, the top above the bottom by a softplus gap) and its pattern (the two inner control points in
the lane's own chord frame: shift, tilt and length removed). The pattern is a softmax mixture of K prototypes (a
codebook from k-means of training lanes, tools/clrbezier/build_pattern_codebook.py) plus a residual. Each lane step
moves the pose (px), adds to the prototype logits and to the residual. The anchors start as their own pose and
pattern (soft-assignment logits + the exact residual). With hier_cfg, each segment is a pose (its two joints,
shared with its neighbours, one monotone chain) and a pattern from a segment codebook in its own chord frame; at
the split the pieces are read exactly; after each step the inner points are clamped monotone and the joint handles
made collinear (g1_project_monotone). Losses: the control-point loss is replaced by the pattern loss (smooth-L1 on
the pattern difference times the GT chord length, px; a segment against the GT between the projections of its
joints) and a soft cross-entropy on the logits (target: soft assignment of the GT pattern), every output; LaneIoU,
endpoint and matching unchanged. Logs pat_acc / seg_pat_acc (o2o pairs whose top prototype is the GT's, last
output of each level) and pat_ent (entropy of the lane mixture weights).

centre_cfg (None; needs pattern_cfg; YSY 2026-10-19, clrbezierlane/dynamic_anchor_design.md): the step-0 curves are
inferred from the image instead of being 16 learned anchors. A light conv head on the finest head level (all levels
upsampled and concatenated) predicts per cell a lane-centre heatmap, a pose (bottom and top points as offsets from the
cell, the bottom below and the top above it by softplus gaps) and a pattern (logits over the lane codebook + a
residual, decoded as the lane steps decode theirs). The num_anchors strongest 3 x 3 peaks are the queries: their
decoded curves (detached) are the step-0 curves, their logits the step-0 pattern logits. No anchor control points and
no anchor embedding: a query's content is the encoding of its curve, the projected head feature of its cell (zero
initialized) and one shared token; everything after step 0 is unchanged. Targets: each GT lane's centre is its
arc-length midpoint; heatmap = a Gaussian at the centre's cell (sigma_cells), CenterNet focal loss; the cells within
radius_cells of a centre (each cell to its nearest centre, weighted by the Gaussian) get the pose smooth-L1 (px, end
weight), the 2D LaneIoU of their decoded curve (iou weight) and the pattern smooth-L1 / soft CE (pattern_cfg weights).
Logs centre_recall (GT lanes with a selected peak within radius_cells of their centre cell). With pattern_cfg (static
anchors or centre_cfg) the loss also logs cov0 / iou0: the share of GT lanes whose best step-0 curve has 2D LaneIoU
>= 0.5, and the mean of that best IoU (negative IoUs of far curves counted as 0).

Not supported here (raise): the row residual (point_offsets), the proposal, the alignment offsets,
strip / merge, the o2m branch, o2o_stop_grad, per-step matching (step_cls). step_blocks and
resample "fine" are required.
"""
import math
import os

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
from .pattern import (assemble_pieces, assign_logits, clamp_inner_monotone, fit_cubic_pinned, from_canon, mixture,
                      pose_chain_step, prior_logits, reverse_canon, segment_targets, soft_ce, to_canon)


# ------------------------------------------------------------------ 2D curve geometry
def split_cubic(c, u):
    """de Casteljau split of cubic Beziers c [..., 4, 2] at parameter u -> (left, right), each [..., 4, 2]."""
    p0, p1, p2, p3 = c.unbind(-2)
    p01, p12, p23 = p0 + u * (p1 - p0), p1 + u * (p2 - p1), p2 + u * (p3 - p2)
    p012, p123 = p01 + u * (p12 - p01), p12 + u * (p23 - p12)
    p0123 = p012 + u * (p123 - p012)
    return torch.stack([p0, p01, p012, p0123], -2), torch.stack([p0123, p123, p23, p3], -2)


def _blossom(c, u1, u2, u3):
    """Polar form of cubic Beziers c [..., 4, 2] at parameters u1, u2, u3 (tensors [...] or floats) -> [..., 2]: the
    de Casteljau levels taken at u1, u2, u3; blossom(t, t, t) is the curve point at t."""
    q = list(c.unbind(-2))
    for u in (u1, u2, u3):
        u = u[..., None] if torch.is_tensor(u) else u
        q = [a + u * (b - a) for a, b in zip(q[:-1], q[1:])]
    return q[0]


def subcurve(c, t0, t1):
    """Control points [..., 4, 2] of cubic Beziers c [..., 4, 2] restricted to [t0, t1] (tensors [...] or floats; any
    values, outside [0, 1] the curve is extended as its polynomial), exactly: blossoms (t0, t0, t0), (t0, t0, t1),
    (t0, t1, t1), (t1, t1, t1). The sub-curve's point at u is c's at t0 + u (t1 - t0). On [t0, t1] inside [0, 1], a
    cubic with y-monotone control points gives y-monotone control points (the derivative's blossoms are convex
    combinations of its control points)."""
    return torch.stack([_blossom(c, t0, t0, t0), _blossom(c, t0, t0, t1), _blossom(c, t0, t1, t1),
                        _blossom(c, t1, t1, t1)], dim=-2)


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


MONO_EPS_PX = 1e-2      # smallest y increment between consecutive control points (px), y_order "monotone"


def inv_softplus(x):
    """Inverse of softplus for x > 0."""
    return torch.where(x > 20.0, x, x + torch.log(-torch.expm1(-x)))


def monotone_step(q_px, d, index, count, eps=MONO_EPS_PX):
    """y-monotone update of chains of control points q_px [..., 3S + 1, 2] (px, image y down, bottom first):
    d [..., S, 4, 2] per piece moves x of the piece's 4 points (d[..., 0]; shared joints averaged), adds to
    the softplus pre-activations of the piece's 3 y increments (d[..., 1:, 1]) and moves the chain's bottom
    y (d[..., 0, 0, 1]). Each point's y is the previous minus a positive increment (in floating point it can
    round to 0 for extreme updates: the chain is then non-increasing, still monotone)."""
    dx = q_px.new_zeros(q_px.shape[:-1]).index_add(-1, index, d[..., 0].flatten(-2)) / count
    y = q_px[..., 1]
    inc = (y[..., :-1] - y[..., 1:]).clamp_min(eps)
    inc = F.softplus(inv_softplus(inc) + d[..., 1:, 1].flatten(-2))
    y0 = y[..., :1] + d[..., 0, 0, 1:2]
    return torch.stack([q_px[..., 0] + dx, torch.cat([y0, y0 - torch.cumsum(inc, dim=-1)], dim=-1)], dim=-1)


def monotone_fix(c_px):
    """Control points [..., K, 2] (px) made non-increasing in y (cumulative min)."""
    return torch.stack([c_px[..., 0], torch.cummin(c_px[..., 1], dim=-1).values], dim=-1)


def g1_project_monotone(q_px, eps=1e-3):
    """c1_project for y-monotone chains: the same collinear handles, each shortened where needed so that the
    chain stays non-increasing in y (no handle goes below the point before it or above the point after)."""
    rows = list(q_px.unbind(-2))
    for j in range(3, q_px.shape[-2] - 1, 3):
        a, jt, c = rows[j - 1], rows[j], rows[j + 1]
        d = c - a
        n = d.norm(dim=-1, keepdim=True)
        u = d / n.clamp_min(eps)
        ok = n > eps
        up = (-u[..., 1:]).clamp_min(0.0)                     # a monotone chain runs up: u_y <= 0
        la, lc = (jt - a).norm(dim=-1, keepdim=True), (c - jt).norm(dim=-1, keepdim=True)
        room_a = (rows[j - 2][..., 1:] - jt[..., 1:]).clamp_min(0.0)
        room_c = (jt[..., 1:] - rows[j + 2][..., 1:]).clamp_min(0.0)
        steep = up > 0
        la = torch.where(steep, torch.minimum(la, room_a / up.clamp_min(1e-12)), la)
        lc = torch.where(steep, torch.minimum(lc, room_c / up.clamp_min(1e-12)), lc)
        rows[j - 1] = torch.where(ok, jt - la * u, a)
        rows[j + 1] = torch.where(ok, jt + lc * u, c)
    return torch.stack(rows, dim=-2)


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


def _project_polyline(p, poly):
    """Foot of points p [..., P, 2] on polylines poly [..., M, 2] (same leading dims), the first and the last
    segment run on as straight lines -> distance [..., P], arc coordinate [..., P] (< 0 before the first point,
    > length past the last) and the polyline's length [...]."""
    s0, d = poly[..., :-1, :], poly[..., 1:, :] - poly[..., :-1, :]                    # [..., M-1, 2]
    len2 = (d * d).sum(-1).clamp_min(1e-12)
    rel = p[..., :, None, :] - s0[..., None, :, :]                                     # [..., P, M-1, 2]
    t = (rel * d[..., None, :, :]).sum(-1) / len2[..., None, :]
    lo = torch.zeros(t.shape[-1], dtype=t.dtype, device=t.device)
    hi = torch.ones_like(lo)
    lo[0], hi[-1] = -float("inf"), float("inf")
    t = torch.maximum(torch.minimum(t, hi), lo)
    diff = rel - t[..., None] * d[..., None, :, :]
    dist, k = (diff * diff).sum(-1).add(1e-12).sqrt().min(-1)
    seg = len2.sqrt()
    cum = torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(-1)], -1)               # [..., M]
    idx = k[..., None]
    s = (cum[..., None, :-1].expand_as(t).gather(-1, idx) + (t * seg[..., None, :]).gather(-1, idx)).squeeze(-1)
    return dist, s, cum[..., -1]


def lane_iou_factorized(a, b, half_width, pairwise=False):
    """curve_cfg.factorized_iou: (IoU_lat, IoU_ext) of prediction polylines a [..., P, 2] against GT polylines
    b [..., M, 2], in the same px space (module docstring). IoU_lat: lane_iou_2d over the overlap only (the
    samples of either lane whose foot on the other lies inside it; the masks carry no gradient; 0 without
    overlap). IoU_ext: 1D IoU of [min, max] of a's two ends projected on b (b's end segments run on as lines)
    against [0, length of b]; its gradient moves a's ends along b. pairwise as lane_iou_2d."""
    iou_lat, ext_inter, ext_union, _ = _factorized_parts(a, b, half_width, pairwise)
    return iou_lat, ext_inter / ext_union.clamp_min(1e-6)


def carrier_terms(a, b, half_width, pairwise=False):
    """FovealLaneHead interval_cfg: (IoU_lat, coverage) of carrier polylines a [..., P, 2] against GT polylines
    b [..., M, 2]: IoU_lat as lane_iou_factorized; coverage = the share of b's length inside a's ends projected on b.
    A carrier that runs past the GT costs nothing; one that stops short loses coverage."""
    iou_lat, ext_inter, _, len_b = _factorized_parts(a, b, half_width, pairwise)
    return iou_lat, ext_inter / len_b.clamp_min(1e-6)


def _factorized_parts(a, b, half_width, pairwise):
    """lane_iou_factorized's IoU_lat, the extent intersection and union (along b) and b's length."""
    if pairwise:
        (batch_size, num_n, num_p), (num_g, num_m) = a.shape[:3], b.shape[1:3]
        a = a[:, :, None].expand(batch_size, num_n, num_g, num_p, 2)
        b = b[:, None].expand(batch_size, num_n, num_g, num_m, 2)
    w = float(half_width)
    d_a, s_a, len_b = _project_polyline(a, b)
    d_b, s_b, len_a = _project_polyline(b, a)
    with torch.no_grad():
        eps = 1e-6
        in_a = ((s_a > eps) & (s_a < len_b[..., None] - eps)).to(a.dtype)
        in_b = ((s_b > eps) & (s_b < len_a[..., None] - eps)).to(a.dtype)
    wa, wb = _arc_weights(a).detach() * in_a, _arc_weights(b).detach() * in_b
    inter = (wa * (2 * w - d_a)).sum(-1) + (wb * (2 * w - d_b)).sum(-1)
    union = (wa * (2 * w + d_a)).sum(-1) + (wb * (2 * w + d_b)).sum(-1)
    iou_lat = inter / union.clamp_min(1e-6)
    lo = torch.minimum(s_a[..., 0], s_a[..., -1])
    hi = torch.maximum(s_a[..., 0], s_a[..., -1])
    ext_inter = (torch.minimum(hi, len_b) - lo.clamp_min(0.0)).clamp_min(0.0)
    ext_union = torch.maximum(hi, len_b) - lo.clamp_max(0.0)
    return iou_lat, ext_inter, ext_union, len_b


def factorized_cost_iou(iou_lat, iou_ext):
    """The matching cost's IoU of curve_cfg.factorized_iou: IoU_lat x IoU_ext, or IoU_lat where it is <= 0 (a
    copy beside the lane is not made cheaper by a worse extent)."""
    return torch.where(iou_lat > 0, iou_lat * iou_ext, iou_lat)


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
    smooth-L1 on the start and end points), min_gt_len_px (8), orientation_free (False), frame ("lane" or
    "image"), y_order (None, "monotone" or "penalty"), order_loss_weight (0.05), match_end_len (False),
    factorized_iou (False), extent_iou_weight (1.0): module docstring.
    The loss_cfg widths are reused:
    lane_width (loss) and lane_width_cost (matching), as half-widths in the iou_eval_shape space.
    hier_cfg: None, or segments (3), seg_iters (3, steps after the lane steps), seg_deform_points (12,
    curve-attention reference points per segment), seg_sample_points (12, re-sampled per segment),
    seg_neighbors (16), seg_rel_points (8), schedule ("sequential": the lane steps, then the segment
    steps; "chain": alternating lane / segment layers with the level exchange, seg_iters = num_iters).
    pattern_cfg: None, or codebook (path of build_pattern_codebook.py's output; the prototypes are buffers, so a
    checkpoint carries them), lane_protos (32), seg_protos (32), ce_weight (0.5), l1_weight (0.05, px smooth-L1 as
    the control-point loss), seg_target_points (16), min_seg_len_px (4: shorter GT pieces get no segment pattern
    loss), tau_scale (1.0, multiplies the codebook's soft-assignment width).
    centre_cfg: None, or channels (64, conv width of the proposal head), sigma_cells (2.0, heatmap Gaussian),
    radius_cells (2, regression cells around each centre), scale_px (100, pose offsets per unit), hm_weight (1.0),
    iou_weight (1.0), end_weight (0.05, px smooth-L1 of the bottom and top points)."""

    def __init__(self, curve_cfg=None, hier_cfg=None, pattern_cfg=None, centre_cfg=None, **kwargs):
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
                 "orientation_free", "frame", "y_order", "order_loss_weight", "match_end_len", "factorized_iou",
                 "extent_iou_weight"}
        if set(cc) - known:
            raise ValueError(f"HBLRHead curve_cfg: unknown keys {sorted(set(cc) - known)}")
        if cc.get("frame", "lane") not in ("lane", "image"):
            raise ValueError("HBLRHead curve_cfg.frame must be 'lane' or 'image'")
        if cc.get("y_order") not in (None, "monotone", "penalty"):
            raise ValueError("HBLRHead curve_cfg.y_order must be None, 'monotone' or 'penalty'")
        if cc.get("y_order") == "monotone" and cc.get("frame", "lane") != "image":
            raise ValueError("HBLRHead curve_cfg.y_order 'monotone' updates in the image axes: set frame='image'")
        if pattern_cfg is not None:
            if cc.get("frame", "lane") != "image" or cc.get("y_order") != "monotone":
                raise ValueError("HBLRHead pattern_cfg needs curve_cfg frame='image' and y_order='monotone'")
            if hier_cfg is not None and dict(hier_cfg).get("schedule", "sequential") != "sequential":
                raise ValueError("HBLRHead pattern_cfg supports hier_cfg schedule 'sequential' only")
        if centre_cfg is not None and pattern_cfg is None:
            raise ValueError("HBLRHead centre_cfg needs pattern_cfg (the proposals predict a pose and a pattern)")
        super().__init__(**kwargs)
        self.gt_points = int(cc.get("gt_points", 48))
        self.out_per_segment = int(cc.get("out_per_segment", 24))
        self.rel_points = int(cc.get("rel_points", 12))
        self.cost_points = int(cc.get("cost_points", 25))
        self.end_loss_weight = float(cc.get("end_loss_weight", 0.05))
        self.min_gt_len_px = float(cc.get("min_gt_len_px", 8.0))
        self.orientation_free = bool(cc.get("orientation_free", False))
        self.frame = str(cc.get("frame", "lane"))
        self.y_order = cc.get("y_order")
        self.order_loss_weight = float(cc.get("order_loss_weight", 0.05))
        self.match_end_len = bool(cc.get("match_end_len", False))
        self.factorized_iou = bool(cc.get("factorized_iou", False))
        self.extent_iou_weight = float(cc.get("extent_iou_weight", 1.0))
        self.carrier_weight = 0.0                    # FovealLaneHead interval_cfg: the carrier loss (_pair_terms)
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
        self.pattern = None
        if pattern_cfg is not None:
            self._build_pattern(dict(pattern_cfg))
        self.centre = None
        if centre_cfg is not None:
            self._build_centre(dict(centre_cfg))
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

    def _build_pattern(self, pc):
        known = {"codebook", "lane_protos", "seg_protos", "ce_weight", "l1_weight", "seg_target_points",
                 "min_seg_len_px", "tau_scale"}
        if set(pc) - known:
            raise ValueError(f"HBLRHead pattern_cfg: unknown keys {sorted(set(pc) - known)}")
        k_lane, k_seg = int(pc.get("lane_protos", 32)), int(pc.get("seg_protos", 32))
        self.pattern = dict(codebook=pc.get("codebook"), k_lane=k_lane, k_seg=k_seg,
                            ce_weight=float(pc.get("ce_weight", 0.5)), l1_weight=float(pc.get("l1_weight", 0.05)),
                            seg_target_points=int(pc.get("seg_target_points", 16)),
                            min_seg_len_px=float(pc.get("min_seg_len_px", 4.0)),
                            tau_scale=float(pc.get("tau_scale", 1.0)))
        self.register_buffer("lane_protos", torch.zeros(k_lane, 4))
        self.register_buffer("lane_sigma", torch.zeros(()))
        self.register_buffer("seg_protos", torch.zeros(k_seg, 4))
        self.register_buffer("seg_sigma", torch.zeros(()))
        path = self.pattern["codebook"]
        if path and os.path.exists(path):
            self.load_codebook(torch.load(path, map_location="cpu"))
        # per lane: pose (x bottom, x top, gap, y bottom; px), prototype logits, pattern residual
        self.update = nn.Linear(self.dim, 4 + k_lane + 4)
        nn.init.normal_(self.update.weight, std=1.0e-2)
        nn.init.zeros_(self.update.bias)
        if self.hier is not None:
            self.seg_update = nn.Linear(self.dim, 4 + k_seg + 4)
            nn.init.normal_(self.seg_update.weight, std=1.0e-2)
            nn.init.zeros_(self.seg_update.bias)

    def _build_centre(self, cc):
        known = {"channels", "sigma_cells", "radius_cells", "scale_px", "hm_weight", "iou_weight", "end_weight"}
        if set(cc) - known:
            raise ValueError(f"HBLRHead centre_cfg: unknown keys {sorted(set(cc) - known)}")
        self.centre = dict(channels=int(cc.get("channels", 64)), sigma=float(cc.get("sigma_cells", 2.0)),
                           radius=int(cc.get("radius_cells", 2)), scale_px=float(cc.get("scale_px", 100.0)),
                           hm_weight=float(cc.get("hm_weight", 1.0)), iou_weight=float(cc.get("iou_weight", 1.0)),
                           end_weight=float(cc.get("end_weight", 0.05)))
        del self.anchor_cps, self.anchor_embed               # no static anchors, no per-index identity
        w, k = self.centre["channels"], self.pattern["k_lane"]
        self.centre_trunk = nn.Sequential(nn.Conv2d(self.prior_feat_channels * self.num_levels, w, 3, padding=1),
                                          nn.ReLU(inplace=True), nn.Conv2d(w, w, 3, padding=1), nn.ReLU(inplace=True))
        self.centre_hm = nn.Conv2d(w, 1, 1)
        self.centre_pose = nn.Conv2d(w, 4, 1)                # x bottom, y bottom gap, x top, y top gap
        self.centre_pat = nn.Conv2d(w, k + 4, 1)             # prototype logits, pattern residual
        nn.init.normal_(self.centre_hm.weight, std=1.0e-2)
        nn.init.constant_(self.centre_hm.bias, -math.log(9.0))   # CenterNet: every cell starts at 0.1
        for m in (self.centre_pose, self.centre_pat):
            nn.init.normal_(m.weight, std=1.0e-2)
            nn.init.zeros_(m.bias)
        self.centre_feat = nn.Linear(w, self.dim)            # the cell's context into the query; starts at zero
        nn.init.zeros_(self.centre_feat.weight)
        nn.init.zeros_(self.centre_feat.bias)
        self.query_token = nn.Parameter(torch.zeros(self.dim))

    def _cell_px(self, idx, hc, wc):
        """Flat cell indices [...] of an hc x wc grid -> cell centres [..., 2] (px)."""
        sx, sy = float(self.img_w) / wc, float(self.img_h) / hc
        return torch.stack([((idx % wc).float() + 0.5) * sx, (torch.div(idx, wc, rounding_mode="floor").float() + 0.5) * sy],
                           dim=-1)

    def _centre_decode(self, pose, pat, cell):
        """Per-cell outputs pose [..., 4], pat [..., K + 4] at cell centres [..., 2] (px) -> bottom, top [..., 2] (px),
        logits [..., K], residual [..., 4], cubic [..., 4, 2] (px)."""
        s, k = self.centre["scale_px"], self.pattern["k_lane"]
        bottom = torch.stack([cell[..., 0] + s * pose[..., 0], cell[..., 1] + s * F.softplus(pose[..., 1])], dim=-1)
        top = torch.stack([cell[..., 0] + s * pose[..., 2], cell[..., 1] - s * F.softplus(pose[..., 3])], dim=-1)
        z, r = pat[..., :k], pat[..., k:]
        c_px = clamp_inner_monotone(from_canon(bottom, top, mixture(z, self.lane_protos) + r))
        return bottom, top, z, r, c_px

    def _centre_proposals(self, feats):
        """centre_cfg: the dense maps and the num_anchors strongest 3 x 3 peaks -> dict(hm [B, 1, H, W], pose
        [B, 4, H, W], pat [B, K + 4, H, W], idx [B, N] flat cells, cps [B, N, 4, 2] normalized (detached), z [B, N, K]
        (detached), feat [B, N, C])."""
        size = feats[-1].shape[-2:]
        x = torch.cat([f if f.shape[-2:] == size else F.interpolate(f, size=size, mode="bilinear", align_corners=False)
                       for f in feats], dim=1)
        t = self.centre_trunk(x)
        hm, pose, pat = self.centre_hm(t).float(), self.centre_pose(t).float(), self.centre_pat(t).float()
        hc, wc = hm.shape[-2:]
        with torch.no_grad():
            s = hm.sigmoid()
            peak = s == F.max_pool2d(s, 3, stride=1, padding=1)
            idx = torch.where(peak, s, s - 2.0).flatten(1).topk(self.num_anchors, dim=1).indices      # peaks first

        def take(m):
            return m.flatten(2).gather(2, idx[:, None].expand(-1, m.shape[1], -1)).transpose(1, 2)

        _, _, z, _, c_px = self._centre_decode(take(pose), take(pat), self._cell_px(idx, hc, wc))
        return dict(hm=hm, pose=pose, pat=pat, idx=idx, cps=self._norm(c_px).detach(), z=z.detach(), feat=take(t))

    def load_codebook(self, cb):
        """Prototypes and soft-assignment widths from build_pattern_codebook.py's output (a dict)."""
        for key, protos, sigma in (("lane", self.lane_protos, self.lane_sigma), ("seg", self.seg_protos, self.seg_sigma)):
            p = torch.as_tensor(cb[key]).float()
            if p.shape != protos.shape:
                raise ValueError(f"HBLRHead pattern codebook: {tuple(p.shape)} {key} prototypes, pattern_cfg asks "
                                 f"for {tuple(protos.shape)}")
            protos.copy_(p)
            sigma.fill_(float(cb[key + "_sigma"]))

    def _sigma(self, key):
        sigma = float(self.lane_sigma if key == "lane" else self.seg_sigma) * self.pattern["tau_scale"]
        if sigma <= 0.0:
            raise RuntimeError("HBLRHead pattern_cfg: no codebook loaded. Run tools/clrbezier/build_pattern_codebook.py "
                               f"CONFIG first (it writes pattern_cfg.codebook = {self.pattern['codebook']})")
        return sigma

    def _pattern_lane_init(self, cps0, z=None):
        """Anchors [B, N, 4, 2] (normalized, monotone) -> pose [B, N, 2, 2] px, logits [B, N, K] (z, the proposals'
        logits with centre_cfg; else their soft assignment with a uniform floor, prior_logits), residual [B, N, 4]:
        they decode exactly to the anchors."""
        c_px = self._px(cps0)
        can = to_canon(c_px)
        if z is None:
            z = prior_logits(can.detach(), self.lane_protos, self._sigma("lane"))
        return c_px[..., [0, 3], :], z, can - mixture(z, self.lane_protos)

    def _pattern_lane_step(self, pose, z, r, out):
        """One lane step from the update out [B, N, 4 + K + 4] -> (pose, z, r, cubic px [B, N, 4, 2])."""
        k = self.pattern["k_lane"]
        pose = pose_chain_step(pose, out[..., None, :4] * self.step_scale, MONO_EPS_PX)
        z = z + out[..., 4:4 + k]
        r = r + out[..., 4 + k:] * (self.step_scale / float(self.img_h - 1))
        c_px = from_canon(pose[..., 0, :], pose[..., 1, :], mixture(z, self.lane_protos) + r)
        return pose, z, r, clamp_inner_monotone(c_px)

    def _pattern_seg_init(self, q):
        """Piecewise control points q [B, N, 3S + 1, 2] (normalized, an exact split) -> joints [B, N, S + 1, 2] px,
        logits [B, N, S, Ks], residual [B, N, S, 4]."""
        pcs = pieces_of(self._px(q))
        can = to_canon(pcs)
        z = prior_logits(can.detach(), self.seg_protos, self._sigma("seg"))
        joints = torch.cat([pcs[..., :, 0, :], pcs[..., -1:, 3, :]], dim=-2)
        return joints, z, can - mixture(z, self.seg_protos)

    def _pattern_seg_step(self, joints, z, r, out):
        """One segment step from the update out [B, N, S, 4 + Ks + 4] -> (joints, z, r, piecewise px)."""
        k = self.pattern["k_seg"]
        joints = pose_chain_step(joints, out[..., :4] * self.step_scale, MONO_EPS_PX)
        z = z + out[..., 4:4 + k]
        r = r + out[..., 4 + k:] * (self.step_scale / float(self.img_h - 1))
        pcs = from_canon(joints[..., :-1, :], joints[..., 1:, :], mixture(z, self.seg_protos) + r)
        return joints, z, r, g1_project_monotone(assemble_pieces(clamp_inner_monotone(pcs)))

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

    def _lane_step(self, c_px, upd):
        """One lane update of cubics c_px [B, N, 4, 2] (px) by upd [B, N, 4, 2] (curve_cfg.frame / y_order)."""
        if self.frame == "lane":
            return c_px + frame_delta(c_px, upd)
        if self.y_order == "monotone":
            index, count = self._joint_index(1, c_px.device)
            return monotone_step(c_px, upd[..., None, :, :], index, count)
        return c_px + upd

    def _chain_step(self, q_px, upd, index, count):
        """One segment update of piecewise control points q_px [B, N, 3S + 1, 2] (px) by upd [B, N, S, 4, 2]
        -> (new points, their move px)."""
        if self.y_order == "monotone":
            q_new = g1_project_monotone(monotone_step(q_px, upd, index, count))
            return q_new, q_new - q_px
        delta = frame_delta(pieces_of(q_px), upd) if self.frame == "lane" else upd
        d_q = delta.new_zeros(q_px.shape).index_add(2, index, delta.flatten(2, 3))
        d_q = d_q / count.view(1, 1, -1, 1)                                     # shared joints: the mean
        return c1_project(q_px + d_q), d_q

    def _layout(self, pts, cps_ends):
        """Official rows of points [B, N, P, 2] for the test output; cps_ends [B, N, 4, 2] give the end
        tangents (bezier_end_slopes) that continue the lane beyond its ends."""
        zeros2 = pts.new_zeros(pts.shape[:2] + (2,))
        return self.official_layout(zeros2, pts, bezier_end_slopes(cps_ends))

    def _forward(self, feats, img_tok=None, branch="main", strip_maps=None, fine=None):
        batch_size = feats[0].shape[0]
        num_q, dim = self.num_anchors, self.dim
        prop = None
        if self.centre is not None:
            prop = self._centre_proposals(feats)
            cps0 = prop["cps"]                                                   # monotone by construction
            sem = (self.query_token + self.geo_enc(curve_code(self._px(cps0), self.img_h - 1))
                   + self.centre_feat(prop["feat"]))
        else:
            cps0 = self.anchor_cps[None].expand(batch_size, -1, -1, -1).float()      # [B, N, 4, 2]
            if self.y_order == "monotone":
                cps0 = self._norm(monotone_fix(self._px(cps0)))
            sem = self.anchor_embed[None].expand(batch_size, -1, -1) + self.geo_enc(curve_code(self._px(cps0), self.img_h - 1))
        pts_s = curve_points(cps0, self.t_sample)
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
        pat_logits = None
        if self.pattern is not None:
            pose, z, r = self._pattern_lane_init(cps0, None if prop is None else prop["z"])
            pat_logits = []
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
            if self.pattern is not None:
                pose, z, r, c_px = self._pattern_lane_step(pose, z, r, self.update(hf).view(batch_size, num_q, -1).float())
                cps = self._norm(c_px)
                pat_logits.append(z)
            else:
                upd = self.update(hf).view(batch_size, num_q, 4, 2).float() * self.step_scale   # px, lane frame
                cps = self._norm(self._lane_step(self._px(cps), upd))
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
            if self.pattern is not None:
                joints, zs, rs = self._pattern_seg_init(q)
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
                if self.pattern is not None:
                    joints, zs, rs, q_px = self._pattern_seg_step(
                        joints, zs, rs, self.seg_update(hs).view(batch_size, num_q, segs, -1).float())
                    d_q = q_px - self._px(q)
                    pat_logits.append(zs)
                else:
                    upd = self.seg_update(hs).view(batch_size, num_q, segs, 4, 2).float() * self.step_scale
                    q_px, d_q = self._chain_step(self._px(q), upd, index, count)
                q = self._norm(q_px)
                moves.append(d_q.detach().norm(dim=-1).mean())
                curves.append(piecewise_points(q, self.out_per_segment))
                cps_list.append(q)
                ends.append(torch.cat([q[..., :2, :], q[..., -2:, :]], dim=-2))
            h = h + self.seg_pool(hs.view(batch_size, num_q, segs, dim).mean(2))
            seg_move = torch.stack(moves).mean()
        lane_cps = cps_list[:num_lane_steps] + [None] * (len(curves) - num_lane_steps)
        if self.pattern is not None:                       # the pattern loss replaces the control-point loss
            lane_cps = [None] * len(curves)
        out = self._finish(h, raw, feats, curves, cps_list, ends, lane_cps, num_lane_steps, step_off, step_w,
                           seg_move=seg_move, pat_logits=pat_logits)
        if self.pattern is not None:
            out["curve0"] = curve_points(cps0, self.t_out_2d).detach()          # the step-0 curves (cov0 log)
        if prop is not None:
            out["centre"] = prop
        return out

    def _finish(self, h, raw, feats, curves, cps_list, ends, lane_cps, num_lane_steps, step_off, step_w,
                seg_move=None, seg_res=None, pat_logits=None):
        """Scores and the output dict. lane_cps: per output, the lane cubic [B, N, 4, 2] that gets the
        control-point loss, or None. pat_logits (pattern_cfg): per output, the prototype logits ([B, N, K] lane
        outputs, [B, N, S, Ks] segment outputs)."""
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
                    step_cls=None, pat_logits=pat_logits,
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
        if self.y_order == "monotone":
            return self._norm(g1_project_monotone(monotone_step(base, res, index, count)))
        delta = frame_delta(pieces_of(base), res) if self.frame == "lane" else res   # [B, N, S, 4, 2] px
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
            c = self._norm(self._lane_step(self._px(c), upd))
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
        if self.factorized_iou:
            fact = lane_iou_factorized(a, gt, self.lane_width_cost * self.iou_w, pairwise=True)
            iou_cost = factorized_cost_iou(*fact)
        else:
            iou_cost = lane_iou_2d(a, gt, self.lane_width_cost * self.iou_w, pairwise=True)
        iou_dyn = lane_iou_2d(a, gt, self.lane_width * self.iou_w, pairwise=True)
        point_cost = torch.zeros_like(iou_cost)
        if self.match_end_len:
            point_cost = self._end_len_cost(a / self.eval_scale, gt / self.eval_scale)
        fg = torch.softmax(logits.float(), dim=-1)[..., 1].clamp(1e-6, 1.0 - 1e-6)
        cls_cost = torch.nan_to_num(-torch.log(fg), nan=100.0, posinf=100.0, neginf=100.0)
        return {"cls_cost": cls_cost[..., None].expand(-1, -1, gmax),
                "point_cost": point_cost,
                "iou_cost_assign": 1.0 - torch.nan_to_num(iou_cost, nan=0.0),
                "lane_iou_dynamic": torch.nan_to_num(iou_dyn, nan=0.0)}

    def _end_len_cost(self, a, gt):
        """curve_cfg.match_end_len: pairwise endpoint + length cost of curves a [B, N, P, 2] against gt
        [B, G, P, 2] (normalized) -> [B, N, G]."""
        a_px, g_px = self._px(a)[:, :, None], self._px(gt)[:, None]
        ends = 0.5 * ((a_px[..., 0, :] - g_px[..., 0, :]).norm(dim=-1) + (a_px[..., -1, :] - g_px[..., -1, :]).norm(dim=-1))
        if self.orientation_free:
            rev = 0.5 * ((a_px[..., 0, :] - g_px[..., -1, :]).norm(dim=-1)
                         + (a_px[..., -1, :] - g_px[..., 0, :]).norm(dim=-1))
            ends = torch.minimum(ends, rev)
        length = (a_px[..., 1:, :] - a_px[..., :-1, :]).norm(dim=-1).sum(-1)
        g_len = (g_px[..., 1:, :] - g_px[..., :-1, :]).norm(dim=-1).sum(-1).clamp_min(1.0)
        return ends / float(self.img_h - 1) + (length / g_len - 1.0).abs().clamp_max(1.0)

    def _pair_terms(self, outs, b, n, j, gt_flat, gt_cps, batch_size, rev_out=None, pat_out=None):
        """Per-image means over the pairs (b, n) <- GT j, averaged over the steps: 1 - LaneIoU, endpoint
        and control-point smooth-L1 (the outputs with a lane cubic in outs["lane_cps"]) -> three [B] tensors.
        orientation_free: the endpoint and control-point terms against the better GT direction per pair and
        step (jointly where both exist); rev_out (a list) receives the last output's reversed-pair share.
        pat_out (a list; pattern_cfg) receives the pattern smooth-L1 and cross-entropy [B] (averaged over the
        steps; lane outputs against the GT lane's pinned cubic, jointly with the endpoint for the direction;
        segment outputs against the GT between their joints) and a dict of logs. outs["carriers"] (FovealLaneHead
        interval_cfg): per output, the carrier polylines; carrier_weight x ((1 - IoU_lat) + (1 - coverage)) of the
        carrier (carrier_terms) joins the LaneIoU term."""
        cnt = per_image_sum(torch.ones_like(b, dtype=torch.float), b, batch_size).clamp_min(1.0)
        g = gt_flat[j].float()
        g_eval, g_px = g * self.eval_scale, self._px(g)
        zero = g.new_zeros(batch_size)
        iou = end = cp = zero
        n_steps = float(len(outs["curves"]))
        n_cp = sum(c is not None for c in outs["lane_cps"])
        pat = pat_out is not None and outs.get("pat_logits") is not None
        if pat:
            n_lane, n_out, beta = outs["num_lane_steps"], len(outs["curves"]), self.smooth_l1_beta
            fit = fit_cubic_pinned(g_px)
            can_gt = to_canon(fit)
            len_gt = (fit[:, 3] - fit[:, 0]).norm(dim=-1, keepdim=True).clamp_min(1.0)
            tgt = torch.softmax(assign_logits(can_gt, self.lane_protos, self._sigma("lane")), dim=-1)
            if self.orientation_free:
                can_gt_r = reverse_canon(can_gt)
                tgt_r = torch.softmax(assign_logits(can_gt_r, self.lane_protos, self._sigma("lane")), dim=-1)
            pat_l1 = pat_ce = zero
            logs = {}
        for k, (curve, c) in enumerate(zip(outs["curves"], outs["lane_cps"])):
            p = curve[b, n].float()
            if self.factorized_iou:
                i_lat, i_ext = lane_iou_factorized(p * self.eval_scale, g_eval, self.lane_width * self.iou_w)
                l_iou = (1.0 - i_lat) + self.extent_iou_weight * (1.0 - i_ext)
            else:
                l_iou = 1.0 - lane_iou_2d(p * self.eval_scale, g_eval, self.lane_width * self.iou_w)
            if outs.get("carriers") is not None and self.carrier_weight > 0:
                c_lat, c_cov = carrier_terms(outs["carriers"][k][b, n].float() * self.eval_scale, g_eval,
                                             self.lane_width * self.iou_w)
                l_iou = l_iou + self.carrier_weight * ((1.0 - c_lat) + (1.0 - c_cov))
            iou = iou + per_image_sum(l_iou, b, batch_size) / cnt
            p_px = self._px(p)
            e = torch.stack([p_px[:, 0] - g_px[:, 0], p_px[:, -1] - g_px[:, -1]], dim=1)
            l_end = _smooth_l1(e, 0.0, self.smooth_l1_beta).mean(dim=(-1, -2))
            if c is not None:
                l_cp = _smooth_l1(self._px(c[b, n].float()) - self._px(gt_cps[j]), 0.0,
                                  self.smooth_l1_beta).mean(dim=(-1, -2))
            l_pl = None
            if pat:
                logits = outs["pat_logits"][k][b, n].float()
                if k < n_lane:
                    can_p = to_canon(self._px(outs["cps"][k][b, n].float()))
                    l_pl = _smooth_l1((can_p - can_gt) * len_gt, 0.0, beta).mean(-1)
                    l_pc = soft_ce(logits, tgt)
                    tgt_used = tgt
                    if self.orientation_free:
                        l_pl_r = _smooth_l1((can_p - can_gt_r) * len_gt, 0.0, beta).mean(-1)
                        l_pc_r = soft_ce(logits, tgt_r)
                else:
                    pcs = pieces_of(self._px(outs["cps"][k][b, n].float()))                   # [P, S, 4, 2]
                    joints = torch.cat([pcs[:, :, 0], pcs[:, -1:, 3]], dim=1)
                    can_t, len_t, valid = segment_targets(joints.detach(), g_px, self.pattern["seg_target_points"],
                                                          self.pattern["min_seg_len_px"])
                    vw = valid.float()
                    den = vw.sum(-1).clamp_min(1.0)
                    l_pl = (_smooth_l1((to_canon(pcs) - can_t) * len_t[..., None], 0.0, beta).mean(-1) * vw).sum(-1) / den
                    t_seg = torch.softmax(assign_logits(can_t, self.seg_protos, self._sigma("seg")), dim=-1)
                    l_pc = (soft_ce(logits, t_seg) * vw).sum(-1) / den
            if self.orientation_free:
                e_r = torch.stack([p_px[:, 0] - g_px[:, -1], p_px[:, -1] - g_px[:, 0]], dim=1)
                l_end_r = _smooth_l1(e_r, 0.0, self.smooth_l1_beta).mean(dim=(-1, -2))
                if c is not None:
                    l_cp_r = _smooth_l1(self._px(c[b, n].float()) - self._px(gt_cps[j].flip(-2)), 0.0,
                                        self.smooth_l1_beta).mean(dim=(-1, -2))
                    rev = (l_end_r + l_cp_r < l_end + l_cp).detach()
                    l_cp = torch.where(rev, l_cp_r, l_cp)
                elif l_pl is not None and k < n_lane:
                    rev = (l_end_r + l_pl_r < l_end + l_pl).detach()
                    l_pl = torch.where(rev, l_pl_r, l_pl)
                    l_pc = torch.where(rev, l_pc_r, l_pc)
                    tgt_used = torch.where(rev[:, None], tgt_r, tgt)
                else:
                    rev = (l_end_r < l_end).detach()
                l_end = torch.where(rev, l_end_r, l_end)
                if rev_out is not None and k == len(outs["curves"]) - 1 and rev.numel():
                    rev_out.append(rev.float().mean())
            end = end + per_image_sum(l_end, b, batch_size) / cnt
            if c is not None:
                cp = cp + per_image_sum(l_cp, b, batch_size) / cnt
            if l_pl is not None:
                pat_l1 = pat_l1 + per_image_sum(l_pl, b, batch_size) / cnt
                pat_ce = pat_ce + per_image_sum(l_pc, b, batch_size) / cnt
                with torch.no_grad():
                    if k == n_lane - 1 and b.numel():
                        logs["pat_acc"] = (logits.argmax(-1) == tgt_used.argmax(-1)).float().mean()
                        prob = torch.softmax(logits, dim=-1)
                        logs["pat_ent"] = -(prob * prob.clamp_min(1e-12).log()).sum(-1).mean()
                    if k == n_out - 1 and k >= n_lane and b.numel():
                        logs["seg_pat_acc"] = (((logits.argmax(-1) == t_seg.argmax(-1)).float() * vw).sum()
                                               / vw.sum().clamp_min(1.0))
        if pat:
            pat_out.extend([pat_l1 / n_steps, pat_ce / n_steps, logs])
        return iou / n_steps, end / n_steps, cp / float(max(1, n_cp))

    def _coverage(self, curve0, layout, gt_flat):
        """Best 2D LaneIoU (loss width) of each GT lane against the step-0 curves curve0 [B, N, P, 2] -> [G]."""
        sub_p = torch.linspace(0, curve0.shape[2] - 1, self.cost_points, device=curve0.device).round().long()
        sub_g = torch.linspace(0, self.gt_points - 1, self.cost_points, device=curve0.device).round().long()
        a = curve0[:, :, sub_p].float() * self.eval_scale
        gt = curve0.new_zeros(curve0.shape[0], layout.gmax, self.cost_points, 2)
        gt[layout.img, layout.slot] = gt_flat[:, sub_g].float()
        iou = lane_iou_2d(a, gt * self.eval_scale, self.lane_width * self.iou_w, pairwise=True)     # [B, N, Gmax]
        return torch.nan_to_num(iou, nan=0.0).max(dim=1).values[layout.img, layout.slot]

    def _centre_loss(self, c, layout, gt_flat, n_img):
        """centre_cfg losses of the dense maps c (outs["centre"]) and the centre_recall log."""
        hm, cc = c["hm"][:, 0], self.centre
        batch_size, hc, wc = hm.shape
        sx, sy = float(self.img_w) / wc, float(self.img_h) / hc
        dev = hm.device
        target = hm.new_zeros(batch_size, hc * wc)
        out = {}
        if layout.num:
            g_px = self._px(bottom_first(gt_flat.float()))                                      # [G, M, 2]
            m = g_px.shape[1]
            centre = 0.5 * (g_px[:, (m - 1) // 2] + g_px[:, m // 2])                             # arc-length midpoint
            ci = (centre[:, 0] / sx).floor().clamp(0, wc - 1).long()
            cj = (centre[:, 1] / sy).floor().clamp(0, hc - 1).long()
            gx, gy = torch.arange(wc, device=dev), torch.arange(hc, device=dev)
            d2 = ((gx[None, None, :] - ci[:, None, None]) ** 2 + (gy[None, :, None] - cj[:, None, None]) ** 2).float()
            gauss = torch.exp(-d2 / (2.0 * cc["sigma"] ** 2)).flatten(1)                       # [G, H * W]
            target = target.scatter_reduce(0, layout.img[:, None].expand_as(gauss), gauss, "amax")
        pos = target == 1.0
        num_pos = float(max(1, int(pos.sum())))
        x = hm.flatten(1)
        l_pos = -F.logsigmoid(x) * (1.0 - torch.sigmoid(x)) ** 2
        l_neg = -F.logsigmoid(-x) * torch.sigmoid(x) ** 2 * (1.0 - target) ** 4
        out["loss_centre_hm"] = torch.where(pos, l_pos, l_neg).sum() / num_pos * cc["hm_weight"]
        zero = (c["pose"].sum() + c["pat"].sum()) * 0.0                                     # keeps the maps in the graph
        out.update(loss_centre_end=zero, loss_centre_iou=zero, loss_centre_pat=zero, loss_centre_pat_ce=zero)
        if not layout.num:
            return out
        # regression cells: within radius_cells of a centre, each cell to its nearest centre in its image
        rad = cc["radius"]
        o = torch.arange(-rad, rad + 1, device=dev)
        oy, ox = torch.meshgrid(o, o, indexing="ij")
        yy, xx = cj[:, None] + oy.flatten()[None], ci[:, None] + ox.flatten()[None]           # [G, (2R + 1)^2]
        dd = (oy.flatten() ** 2 + ox.flatten() ** 2).float()[None].expand_as(yy)
        g_idx = torch.arange(layout.num, device=dev)[:, None].expand_as(yy)
        inside = (yy >= 0) & (yy < hc) & (xx >= 0) & (xx < wc)
        yy, xx, dd, g_idx = yy[inside], xx[inside], dd[inside], g_idx[inside]
        key = layout.img[g_idx] * (hc * wc) + yy * wc + xx
        order = torch.sort(dd, stable=True).indices
        order = order[torch.sort(key[order], stable=True).indices]
        key_s = key[order]
        first = torch.ones_like(key_s, dtype=torch.bool)
        first[1:] = key_s[1:] != key_s[:-1]
        sel = order[first]
        g_idx, yy, xx, w = g_idx[sel], yy[sel], xx[sel], torch.exp(-dd[sel] / (2.0 * cc["sigma"] ** 2))
        b = layout.img[g_idx]
        cell = torch.stack([(xx.float() + 0.5) * sx, (yy.float() + 0.5) * sy], dim=-1)
        bottom, top, z, r, c_px = self._centre_decode(c["pose"][b, :, yy, xx], c["pat"][b, :, yy, xx], cell)
        g_sel = g_px[g_idx]
        beta = self.smooth_l1_beta
        l_end = _smooth_l1(torch.stack([bottom - g_sel[:, 0], top - g_sel[:, -1]], dim=1), 0.0, beta).mean(dim=(-1, -2))
        curve = curve_points(self._norm(c_px), self.t_out_2d)
        l_iou = 1.0 - lane_iou_2d(curve * self.eval_scale, gt_flat[g_idx].float() * self.eval_scale,
                                  self.lane_width * self.iou_w)
        fit = fit_cubic_pinned(g_px)
        can_gt = to_canon(fit)
        len_gt = (fit[:, 3] - fit[:, 0]).norm(dim=-1, keepdim=True).clamp_min(1.0)
        tgt = torch.softmax(assign_logits(can_gt, self.lane_protos, self._sigma("lane")), dim=-1)
        l_pat = _smooth_l1((mixture(z, self.lane_protos) + r - can_gt[g_idx]) * len_gt[g_idx], 0.0, beta).mean(-1)
        l_ce = soft_ce(z, tgt[g_idx])
        den = per_image_sum(w, b, batch_size).clamp_min(1e-6)

        def red(v):
            return (per_image_sum(w * v, b, batch_size) / den).sum() / n_img

        out["loss_centre_end"] = red(l_end) * cc["end_weight"]
        out["loss_centre_iou"] = red(l_iou) * cc["iou_weight"]
        out["loss_centre_pat"] = red(l_pat) * self.pattern["l1_weight"]
        out["loss_centre_pat_ce"] = red(l_ce) * self.pattern["ce_weight"]
        with torch.no_grad():
            picks = c["idx"][layout.img]                                                         # [G, N]
            py, px_ = torch.div(picks, wc, rounding_mode="floor"), picks % wc
            near = torch.maximum((py - cj[:, None]).abs(), (px_ - ci[:, None]).abs()) <= rad
            out["centre_recall"] = near.any(dim=1).float().mean()
        return out

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
        iou_sum = end_sum = cp_sum = pat_l1_sum = pat_ce_sum = zero
        num_pos_o2o = num_pos_o2m = 0
        rev_o2o = []
        fact_log = car_log = None
        pat_o2o = [] if self.pattern is not None else None
        if layout.num:
            gt_cps = fit_cubic_2d(gt_flat.float(), self.cp_fit_ridge)
            final = outs["curves"][-1].detach()
            with torch.no_grad():
                cache = self._cost_cache(outs["cls_o2o"].detach(), final, layout, gt_flat)
            b_o, n_o, g_o = batched_assign(self.o2o_assigner, cache, layout)
            tgt_o2o[b_o, n_o] = 1
            num_pos_o2o = int(n_o.numel())
            iou, end, cp = self._pair_terms(outs, b_o, n_o, layout.flat_index(b_o, g_o), gt_flat, gt_cps, batch_size,
                                            rev_out=rev_o2o, pat_out=pat_o2o)
            iou_sum, end_sum, cp_sum = iou.sum(), end.sum(), cp.sum()
            if self.factorized_iou and n_o.numel():
                with torch.no_grad():
                    fact_log = lane_iou_factorized(final[b_o, n_o].float() * self.eval_scale,
                                                   gt_flat[layout.flat_index(b_o, g_o)].float() * self.eval_scale,
                                                   self.lane_width * self.iou_w)
            if outs.get("carriers") is not None and n_o.numel():
                with torch.no_grad():
                    car_log = carrier_terms(outs["carriers"][-1][b_o, n_o].detach().float() * self.eval_scale,
                                            gt_flat[layout.flat_index(b_o, g_o)].float() * self.eval_scale,
                                            self.lane_width * self.iou_w)
            if pat_o2o:
                pat_l1_sum, pat_ce_sum = pat_o2o[0].sum(), pat_o2o[1].sum()
            if self.o2m_assigner is not None:
                with torch.no_grad():
                    cache_m = dict(cache)
                    fg = torch.softmax(outs["cls_o2m"].detach().float(), -1)[..., 1].clamp(1e-6, 1.0 - 1e-6)
                    cache_m["cls_cost"] = torch.nan_to_num(-torch.log(fg), nan=100.0)[..., None].expand_as(cache["cls_cost"])
                b_m, n_m, g_m = batched_assign(self.o2m_assigner, cache_m, layout)
                tgt_o2m[b_m, n_m] = 1
                num_pos_o2m = int(n_m.numel())
                if n_m.numel() and self.o2m_reg_weight > 0:
                    pat_m = [] if self.pattern is not None else None
                    iou, end, cp = self._pair_terms(outs, b_m, n_m, layout.flat_index(b_m, g_m), gt_flat, gt_cps,
                                                    batch_size, pat_out=pat_m)
                    w = self.o2m_reg_weight
                    iou_sum, end_sum, cp_sum = iou_sum + w * iou.sum(), end_sum + w * end.sum(), cp_sum + w * cp.sum()
                    if pat_m:
                        pat_l1_sum, pat_ce_sum = pat_l1_sum + w * pat_m[0].sum(), pat_ce_sum + w * pat_m[1].sum()
        cls_o2o = (focal_2cls(outs["cls_o2o"], tgt_o2o, self.focal_alpha, self.focal_gamma).sum(-1) / gt_counts).mean()
        losses = dict(loss_cls=cls_o2o * self.cls_loss_weight,
                      loss_iou=iou_sum / n_img * self.iou_loss.loss_weight,
                      loss_end=end_sum / n_img * self.end_loss_weight,
                      loss_cp=cp_sum / n_img * self.cp_loss_weight)
        if self.pattern is not None:
            del losses["loss_cp"]
            losses["loss_pat"] = pat_l1_sum / n_img * self.pattern["l1_weight"]
            losses["loss_pat_ce"] = pat_ce_sum / n_img * self.pattern["ce_weight"]
        if self.o2m_assigner is not None:
            cls_o2m = (focal_2cls(outs["cls_o2m"], tgt_o2m, self.focal_alpha, self.focal_gamma).sum(-1)
                       / gt_counts).mean()
            losses["loss_cls_o2m"] = cls_o2m * self.cls_loss_weight * self.o2m_cls_weight
        if self.y_order == "penalty":
            rise = [F.relu(self._px(c.float())[..., 1:, 1] - self._px(c.float())[..., :-1, 1]).sum(-1).mean()
                    for c in outs["cps"]]
            losses["loss_order"] = torch.stack(rise).mean() * self.order_loss_weight
        if seg_gt is not None and "seg" in outs:
            losses["loss_seg"] = F.nll_loss(F.log_softmax(outs["seg"], dim=1), seg_gt, weight=self.seg_class_weights,
                                            ignore_index=self.seg_ignore_label) * self.seg_loss_weight
        if outs.get("centre") is not None:
            losses.update(self._centre_loss(outs["centre"], layout, gt_flat, n_img))
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
            if self.factorized_iou:
                diag["o2o_lat_iou"], diag["o2o_ext_iou"] = ((fact_log[0].mean(), fact_log[1].mean()) if fact_log
                                                            is not None else (torch.zeros((), device=dev),) * 2)
            if outs.get("carriers") is not None:
                diag["o2o_car_lat"], diag["o2o_car_cov"] = ((car_log[0].mean(), car_log[1].mean()) if car_log
                                                            is not None else (torch.zeros((), device=dev),) * 2)
            if pat_o2o:
                diag.update({k: v.detach() for k, v in pat_o2o[2].items()})
            if outs.get("curve0") is not None and layout.num:
                best = self._coverage(outs["curve0"], layout, gt_flat)
                diag["cov0"], diag["iou0"] = (best >= 0.5).float().mean(), best.clamp_min(0.0).mean()
        losses.update(diag)
        return losses

    def loss_by_outputs(self, outs, lanes, seg_gt=None):
        raise RuntimeError("HBLRHead trains on 2D lanes: use loss() / loss_by_outputs_2d()")
