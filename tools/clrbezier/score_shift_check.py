"""Can a lane head's score see localization? Eval-only checks on CurveLanes valid.

1. Shift. The last decoder layer is run again with every query's input curve translated sideways by d px (along
   the normal of its chord, px of the network input, 800 x 320). All queries move together, each along its own
   normal, so the layer's self-attention sees the same relative layout. Two outputs per d:
     in   the output curve is the shifted input: the score judges a curve d px off, from features read there;
     out  the head's own last update is applied, as at test time.
   Every GT lane that is hit at d = 0 (out, the head's own output) is followed through its best query (IoU > 0.5):
   the metric IoU of that query's curve with the lane, and its score, at each d. If the score stays flat while the
   IoU falls below 0.5, the score cannot see localization at the metric's scale.
   Reported per d: median IoU, share above 0.5, median score and score / score at d = 0. Within-lane AUC: over the
   pairs of shifts of one lane where one curve is a hit (IoU > 0.5) and the other is not, how often the hit scores
   higher (0.5 = blind, 1 = always right). Edge ratio: the score at the smallest shift that loses the lane, over
   the score at d = 0. Both also for short or flat lanes (extent < 0.4 or slant > 4) and the rest.
2. Copies (d = 0, out). Per GT lane: K = queries with IoU > 0.5 (its copies), the best copy's score, the low-score
   rate (best copy below the threshold), and, for K >= 2, how often the best-IoU copy is the top-scored one (chance
   = mean 1 / K). If one-to-one training splits the score among copies the features cannot tell apart, the best
   copy's score falls and the low-score rate rises with K.
3. Near misses (d = 0, out), the real-query version of 1: per GT lane, the queries whose best lane it is with IoU in
   (0.2, 0.5]. AUC: how often a copy (IoU > 0.5) of a lane outscores a near miss of the same lane; and how many near
   misses score above the threshold (each one is a false positive next to a real lane).
4. Linear probe: is the hit / miss signal in the vector the classifier reads? An L2 logistic regression on that
   vector (FovealLaneHead: the o2o classifier's input after the scoring self-attention; the replica: the last layer's
   query) is fit on the even images and scored on the odd ones, for the shifted copies of 1 (hit vs miss, mode in)
   and the real copies and near misses of 3. Its within-lane AUC is printed next to the head's own score's on the
   same lanes. Both classifiers are one linear layer on that vector (cls_o2o, cls_heads[-1]), so the probe has exactly
   the classifier's capacity. Probe well above the score: the information is there and the trained classifier does
   not use it. Probe near the score: the vector does not carry it.
5. F1 bound of rescoring (needs 4): probes fit on every decodable query of the even images rescore the head's own
   outputs at d = 0 on the odd images; swept F1@50 (vega: Hungarian on 1 - IoU, hits IoU > 0.5), each row at its best
   threshold on the odd images. Rows with the official lane NMS (the config's nms_thres, nms_topk) use the target
   hit (IoU > 0.5 with some lane, duplicates included); rows without NMS use the target top copy (per lane the
   head's own top-scored copy, everything else negative), so the head's selection is kept and a perfect probe gives
   the best-copy oracle. Inputs: the vector, the head's logit, explicit geometry (last-step move, distance to the
   nearest and 2nd nearest query, queries within the NMS radius, a higher-scored one among them, start, angle,
   length). The logit alone on top copy is the control (a monotone map of the head score: its row is the head's).
   Two GT oracles on the head's score give the ceilings: every near miss scored 0; the best-IoU copy of each lane 1.
6. What the near misses are: each near miss (above and below the threshold) and the head's top copy of each lane
   above it, against its lane with the error breakdown's extent_lateral: what alone would make it a hit (the GT
   extent, the GT lateral position, either, neither), its median lateral offset, the share of the lane it covers and
   runs past, and its end errors. Check 4 also reports the AUC over copies and near misses both above the threshold.
7. The shape of the offset: the signed normal offset along the lane (bottom / middle / top third), its side (right,
   away from the image centre) and a linear fit of it on the distance from the image centre, for the near misses and
   the hits. Check 5 adds the position oracle: every near miss moved onto its lane, its extent and score kept.
   near_misses.csv lists every near miss (image, GT index, query, score, extent / lateral fields, offsets, its curve
   and the top-scored copy of its lane, original px); draw_near_misses.py draws a sample of them.
8. Evidence along each GT lane: its outcome (hit, near only, low score, missed) against whether it is the outermost
   lane on its side, the marking contrast in its bottom / top third (the line against its sides, and the step
   between its sides for curb-like edges) and the image's brightness. Reads each image at half size.
9. Neighbour prior (premise test of a structure-based fix): each output lane is re-angled about its far end so that
   its near-field line meets the image's bottom row on the lattice of its neighbours (equal spacing), with the other
   output lanes or the other GT lanes as neighbours: near-only near misses fixed, hits broken, F1 before / after.

The metric is the error breakdown's (curvelanes_error_breakdown.py, next to this file): the head's own
predictions_to_lanes, CurvelanesMetric's lane masks (224 x 224, width 5) and IoU.

Heads: FovealLaneHead without interval_cfg / hier_cfg (foveal_lane_off_bfstyle_o2o*, flip, pointcond, factiou) and
BezierFormerHead (bezierformer_replica*). At d = 0, "out" is checked against the head's own forward (max difference
printed; it must be 0).

    python tools/clrbezier/score_shift_check.py CONFIG CHECKPOINT [--max-images 1000] [--shifts 0 2 4 6 8 10 12 16]
        [--thr THR] [--out-dir DIR] [--jobs N] [--no-probe]
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import os.path as osp
import sys
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_HERE = osp.dirname(osp.abspath(__file__))
_spec = importlib.util.spec_from_file_location("curvelanes_error_breakdown",
                                               osp.join(_HERE, "curvelanes_error_breakdown.py"))
eb = importlib.util.module_from_spec(_spec)
sys.modules["curvelanes_error_breakdown"] = eb
_spec.loader.exec_module(eb)
rc = eb.rc

HIT = 0.5
NEAR = 0.2                                     # a near miss: best IoU with its lane in (NEAR, HIT]
MODES = ("in", "out")


# ----------------------------------------------------------------------------
# re-running the last decoder layer
# ----------------------------------------------------------------------------
def chord_normals(cps_px):
    """Unit normals [B, N, 2] of the chords of cubics cps_px [B, N, 4, 2], the chord taken from the lower end up, so
    that +d moves every curve to the right in the image whatever its control-point order."""
    d = cps_px[..., 3, :] - cps_px[..., 0, :]
    d = torch.where(d[..., 1:] > 0, -d, d)
    n = torch.stack([-d[..., 1], d[..., 0]], dim=-1)
    length = n.norm(dim=-1, keepdim=True)
    unit_x = torch.zeros_like(n)
    unit_x[..., 0] = 1.0
    return torch.where(length > 1e-6, n / length.clamp_min(1e-6), unit_x)


class FovealRescorer:
    """FovealLaneHead._forward up to the last decoder step, then that step again on shifted input curves."""

    def __init__(self, head):
        if getattr(head, "interval", None) is not None or getattr(head, "hier", None) is not None:
            raise ValueError("score_shift_check: FovealLaneHead with interval_cfg or hier_cfg is not supported")
        self.head = head
        cfg = head.test_cfg or {}
        source = head.test_score or ("o2m" if bool(cfg.get("use_nms", True)) else "o2o")
        self.classifier = head.cls_o2m if source == "o2m" and head.cls_o2m is not None else head.cls_o2o

    def _step(self, step, h, cps, st):
        hd = self.head
        ref = hd._px(eb_curve_points(cps, hd.t_deform)).detach()
        rel = hd._px(eb_curve_points(cps, hd.t_rel)).detach()
        h = step(h, rel, hd.dec_pos, ref, st["tokens"], st["maps"], st["frame"])
        upd = hd.update(h).view(*cps.shape).float() * hd.step_scale
        return h, upd, step.curve.last_offset_px, step.sa.last_self_weight

    def prefix(self, feats):
        hd = self.head
        tokens, maps, frame, fine = hd._tokens(feats)
        prop = hd._proposals(fine, frame)
        cps = prop["cps"]
        h = hd.query_norm(hd.query_token + hd.geo_enc(eb_curve_code(hd._px(cps), hd.img_h - 1))
                          + hd.tok_proj(prop["tok"]))
        st = dict(tokens=tokens, maps=maps, frame=frame, curves=[], cps_list=[], step_off=[], step_w=[])
        for step in hd.dec[:-1]:
            h, upd, off, w = self._step(step, h, cps, st)
            cps = hd._norm(hd._lane_step(hd._px(cps), upd))
            st["curves"].append(eb_curve_points(cps, hd.t_out_2d))
            st["cps_list"].append(cps)
            st["step_off"].append(off)
            st["step_w"].append(w)
        st.update(h=h, cps=cps)
        return st

    def last(self, st, shift_norm, mode):
        """-> (the official pred dict, the input cubics of the last step [B, N, 4, 2] normalized)."""
        hd = self.head
        cps_in = st["cps"] + shift_norm[:, :, None, :].to(st["cps"].dtype)
        h, upd, off, w = self._step(hd.dec[-1], st["h"], cps_in, st)
        cps_out = hd._norm(hd._lane_step(hd._px(cps_in), upd)) if mode == "out" else cps_in
        curves = st["curves"] + [eb_curve_points(cps_out, hd.t_out_2d)]
        cps_list = st["cps_list"] + [cps_out]
        n = len(curves)
        out = hd._finish(h, None, None, curves, cps_list, list(cps_list), [None] * n, n,
                         st["step_off"] + [off], st["step_w"] + [w])
        return official_foveal(hd, out), cps_in


def official_foveal(hd, out):
    """FovealLaneHead.forward_test's last output: its scores (o2o NMS-free / o2m with NMS, nms_topk) in the rows."""
    cfg = hd.test_cfg or {}
    use_nms = bool(cfg.get("use_nms", True))
    source = hd.test_score or ("o2m" if use_nms else "o2o")
    z = out["cls_o2o"] if source == "o2o" or out["cls_o2m"] is None else out["cls_o2m"]
    if not use_nms:
        topk = cfg.get("nms_topk")
        if topk is not None and 0 < int(topk) < z.shape[1]:
            fg = z[..., 1] - z[..., 0]
            keep = torch.zeros_like(fg, dtype=torch.bool)
            keep.scatter_(1, fg.topk(int(topk), dim=1).indices, True)
            z = torch.stack([z[..., 0], torch.where(keep, z[..., 1], z[..., 1] - 1e4)], dim=-1)
    last = out["preds"][-1].clone()
    last[..., :2] = z.to(last.dtype)
    return hd.to_official_pred_dict(last)


class BezierRescorer:
    """BezierFormerHead._forward up to the last layer, then that layer again on shifted input curves."""

    def __init__(self, head):
        self.head = head
        self.classifier = head.cls_heads[-1]

    def prefix(self, feats):
        hd = self.head
        tokens, maps, frame = hd._tokens(feats)
        batch_size = tokens.shape[0]
        pooled = torch.cat([tokens[:, mp_.start:mp_.start + mp_.n].mean(1) for mp_ in maps], dim=-1)
        c = hd.init_mlp(pooled).float().view(batch_size, hd.num_queries, 4, 2)
        e = hd.query_embed[None].expand(batch_size, -1, -1).to(tokens.dtype)
        for i, layer in enumerate(hd.layers[:-1]):
            c_in = c.detach()
            e = self._layer(layer, e, c_in, tokens, maps, frame)
            c = c_in + hd.reg_heads[i](e).float().view(batch_size, hd.num_queries, 4, 2)
        return dict(tokens=tokens, maps=maps, frame=frame, e=e, cps=c)

    def _layer(self, layer, e, c_in, tokens, maps, frame):
        hd = self.head
        pts = eb_curve_points(c_in, hd.t_ref)
        pe = hd.pe_mlp(eb_sine_embed(pts, hd.dim).flatten(-2).to(e.dtype))
        return layer(e, pe, hd._px(pts), tokens, maps, frame)

    def last(self, st, shift_norm, mode):
        hd = self.head
        c_in = st["cps"].detach() + shift_norm[:, :, None, :].to(st["cps"].dtype)
        e = self._layer(hd.layers[-1], st["e"], c_in, st["tokens"], st["maps"], st["frame"])
        logit = hd.cls_heads[-1](e).squeeze(-1).float()
        c_out = c_in + hd.reg_heads[-1](e).float().view(*c_in.shape) if mode == "out" else c_in
        return hd.to_official_pred_dict(hd._official(logit, c_out)), c_in


def eb_curve_points(cps, t):
    from libs.clrbezier.blrnet import curve_points
    return curve_points(cps, t)


def eb_curve_code(c_px, norm):
    from libs.clrbezier.hblrnet import curve_code
    return curve_code(c_px, norm)


def eb_sine_embed(xy, dim):
    from libs.clrbezier.foveal_lane import sine_embed
    return sine_embed(xy, dim)


def rescorer_for(head):
    name = type(head).__name__
    if name == "BezierFormerHead":
        return BezierRescorer(head)
    if hasattr(head, "dec") and hasattr(head, "_proposals"):
        return FovealRescorer(head)
    raise ValueError(f"score_shift_check: unsupported head {name} (FovealLaneHead or BezierFormerHead)")


@torch.no_grad()
def run_batch(head, rescorer, feats, shifts, keep_states=False):
    """-> {(mode, d): (rows [B, N, 78], scores [B, N])}; keep_states: the vectors the classifier read
    {(mode, d): [B, N, D] float16}, mode in at every d and out at d = 0, else {}; the max difference of (out, 0) from
    head(feats)[-1]."""
    st = rescorer.prefix(feats)
    normals = chord_normals(head._px(st["cps"]))                       # [B, N, 2] px
    scale = head.px_scale.to(normals.dtype)
    res, states, seen = {}, {}, []
    hook = rescorer.classifier.register_forward_hook(lambda mod, inp, out: seen.append(inp[0]))
    try:
        for mode in MODES:
            for d in shifts:
                seen.clear()
                pred, _ = rescorer.last(st, normals * float(d) / scale, mode)
                rows, score = eb.pred_rows(pred)
                res[(mode, float(d))] = (rows.float().cpu().numpy(), score.float().cpu().numpy())
                if keep_states and (mode == "in" or d == 0):
                    states[(mode, float(d))] = seen[-1].detach().to(torch.float16).cpu().numpy()
    finally:
        hook.remove()
    own, _ = eb.pred_rows(head(feats)[-1])
    rows0 = torch.as_tensor(res[("out", 0.0)][0])
    diff = float((own.float().cpu() - rows0).abs().max()) if own.numel() else 0.0
    return res, states, diff


# ----------------------------------------------------------------------------
# metric IoU per image (worker)
# ----------------------------------------------------------------------------
_W = {}


def _init_worker(state):
    torch.set_num_threads(1)
    _W.update(state)


def _lane(row, ori_hw, offset):
    """The metric's lane (list of {x, y}, original px) of one prediction row, or None."""
    pts = _W["decoder"].points(row)
    if pts is None:
        return None
    lane = rc.to_metric_points(pts, ori_hw, offset, _W["st"]["y_step"])
    return lane if len(lane) else None


def _mask(row, ori_hw, offset):
    lane = _lane(row, ori_hw, offset)
    return None if lane is None else rc.lane_mask(lane, ori_hw, _W["st"]["eval"])


EVID_FIELDS = ["outermost", "contrast_bot", "step_bot", "contrast_top", "step_top", "brightness"]
WEAK = 8.0                                               # gray levels: below it, no visible marking


def outermost_lanes(gt):
    """Per GT lane (original px points): True if it lies left of, or right of, every lane it shares rows with (mean
    x difference over the shared rows); a lane that shares rows with none counts as outermost."""
    arr = [np.asarray(ln, np.float64).reshape(-1, 2) for ln in gt]
    arr = [a[np.argsort(a[:, 1])] for a in arr]
    out = []
    for g, a in enumerate(arr):
        side = []
        for h, b in enumerate(arr):
            lo, hi = max(a[0, 1], b[0, 1]), min(a[-1, 1], b[-1, 1])
            if h == g or hi - lo < 10:
                continue
            ys = np.linspace(lo, hi, 10)
            side.append(np.sign(np.mean(np.interp(ys, a[:, 1], a[:, 0]) - np.interp(ys, b[:, 1], b[:, 0]))))
        out.append(not side or all(v < 0 for v in side) or all(v > 0 for v in side))
    return out


def lane_evidence(img, lane, scale, width):
    """Marking evidence along a GT lane (original px points) in a grayscale image reduced by `scale`: per bottom /
    top third of its arc, the median of (on the line, best of 3 samples within 0.4% of the image width) minus (the
    mean of the two sides at 1.5%), and of |left side - right side| (a step edge such as a curb). NaN off the image."""
    nan = (float("nan"),) * 4
    pts = np.asarray(lane, np.float64).reshape(-1, 2)
    if img is None or len(pts) < 2:
        return nan
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if cum[-1] < 12:
        return nan
    sa = np.arange(0.0, cum[-1], 4.0)
    x, y = np.interp(sa, cum, pts[:, 0]), np.interp(sa, cum, pts[:, 1])
    tx, ty = np.gradient(x), np.gradient(y)
    n = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / n, tx / n
    h, w = img.shape[:2]

    def at(off):
        c = np.round((x + off * nx) / scale).astype(int)
        r = np.round((y + off * ny) / scale).astype(int)
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        v = np.full(len(x), np.nan)
        v[ok] = img[r[ok], c[ok]]
        return v
    d, o = 0.004 * width, 0.015 * width
    with warnings.catch_warnings():                      # all-NaN columns (off the image) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        on = np.nanmax(np.stack([at(-d), at(0.0), at(d)]), axis=0)
    left, right = at(-o), at(o)
    contrast, step = on - (left + right) / 2, np.abs(left - right)
    first = sa <= cum[-1] / 3
    last = sa >= 2 * cum[-1] / 3
    bot, top = (first, last) if pts[0, 1] >= pts[-1, 1] else (last, first)

    def med(v, m):
        v = v[m & ~np.isnan(v)]
        return float(np.median(v)) if len(v) else float("nan")
    return med(contrast, bot), med(step, bot), med(contrast, top), med(step, top)


# 9. neighbour prior: equal spacing of straight near fields at the bottom row (flat road, pinhole camera)
MAX_STEEP = np.tan(np.radians(75.0))                    # |dx/dy| of a near field within 75 deg of vertical
MERGE = 0.02                                             # neighbours closer than this share of the width are one lane


def near_line(lane):
    """(a, b) of x = a y + b fitted to the bottom half (by y) of a lane (list of {x, y} or [n, 2], original px), or
    None if it has < 3 points there or lies within 15 deg of horizontal."""
    pts = np.array([[p["x"], p["y"]] for p in lane] if lane and isinstance(lane[0], dict) else lane,
                   np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None
    lo, hi = pts[:, 1].min(), pts[:, 1].max()
    near = pts[pts[:, 1] >= lo + 0.5 * (hi - lo)]
    if len(near) < 3:
        near = pts
    if np.ptp(near[:, 1]) < 1e-6:
        return None
    a, b = np.polyfit(near[:, 1], near[:, 0], 1)
    return None if abs(a) > MAX_STEEP else (float(a), float(b))


def lattice_x(x, xs, width):
    """The point nearest x on the lattice of the neighbours' bottom-row xs (merged within MERGE x width; >= 2 after
    merging), or None. Spacing: the median of gap / round(gap / smallest gap), so a lane missing from the
    neighbours (a double gap) does not double it."""
    xs = np.sort(np.asarray(xs, np.float64))
    if len(xs) < 2:
        return None
    merged = [xs[0]]
    for v in xs[1:]:
        if v - merged[-1] > MERGE * width:
            merged.append(v)
    if len(merged) < 2:
        return None
    gaps = np.diff(merged)
    step = float(np.median(gaps / np.maximum(1.0, np.round(gaps / gaps.min()))))
    nb = merged[int(np.argmin(np.abs(np.asarray(merged) - x)))]
    return nb + round((x - nb) / step) * step


def shear_lane(lane, x_ref, x_new, y_ref):
    """The lane re-angled about its far (top) end so that its near-field line meets row y_ref at x_new instead of
    x_ref: each point moves sideways by (x_new - x_ref) x (y - y_top) / (y_ref - y_top)."""
    y_top = min(p["y"] for p in lane)
    if y_ref - y_top < 1e-6:
        return lane
    dx = x_new - x_ref
    return [{"x": p["x"] + dx * (p["y"] - y_top) / (y_ref - y_top), "y": p["y"]} for p in lane]


def neighbour_prior(lanes0, masks0, score0, iou0, best_gt, gt, gt_masks, hit_above, ori_hw, thr, ev):
    """Each output lane (score >= thr) re-placed by the neighbour prior, with neighbours = the other output lanes
    (pred) or the other GT lanes (gt): tp before / after / gated (the better of the two per query, GT-chosen), and
    the near-only near misses fixed and the hits broken."""
    h, w = ori_hw
    y_ref = h - 1.0
    out_q = [q for q in range(len(score0)) if masks0[q] is not None and score0[q] >= thr]
    lines = {q: near_line(lanes0[q]) for q in out_q}
    gt_lines = [near_line(ln) for ln in gt]
    res = {}
    for mode in ("pred", "gt"):
        after = iou0.copy()
        r = dict(n_out=len(out_q), n_gt=len(gt), app=0, nm=0, nm_app=0, nm_fixed=0, hit=0, hit_app=0, hit_broken=0,
                 move_nm=[], move_hit=[])
        for q in out_q:
            g0 = int(best_gt[q]) if len(gt) else -1
            before = float(iou0[g0, q]) if g0 >= 0 else 0.0
            is_nm = NEAR < before <= HIT and not hit_above[g0]
            is_hit = before > HIT
            r["nm"] += is_nm
            r["hit"] += is_hit
            if lines[q] is None or not len(gt):
                continue
            if mode == "pred":
                xs = [lines[p][0] * y_ref + lines[p][1] for p in out_q if p != q and lines[p] is not None]
            else:
                xs = [gl[0] * y_ref + gl[1] for k, gl in enumerate(gt_lines)
                      if gl is not None and not (k == g0 and before > NEAR)]
            x_ref = lines[q][0] * y_ref + lines[q][1]
            x_new = lattice_x(x_ref, xs, w)
            if x_new is None:
                continue
            r["app"] += 1
            m = rc.lane_mask(shear_lane(lanes0[q], x_ref, x_new, y_ref), ori_hw, ev)
            after[:, q] = rc.iou_matrix(gt_masks, [m])[:, 0]
            if is_nm:
                r["nm_app"] += 1
                r["nm_fixed"] += bool(after[g0, q] > HIT)
                r["move_nm"].append(abs(x_new - x_ref) / w)
            if is_hit:
                r["hit_app"] += 1
                r["hit_broken"] += bool(after[g0, q] <= HIT)
                r["move_hit"].append(abs(x_new - x_ref) / w)
        if len(gt):
            r.update(tp0=image_tp(iou0, out_q), tp1=image_tp(after, out_q),
                     tpg=image_tp(np.maximum(iou0, after), out_q))
        else:
            r.update(tp0=0, tp1=0, tpg=0)
        res[mode] = r
    return res


def _pts_str(lane):
    """A metric lane (list of {x, y}, original px) as "x y x y ..." for near_misses.csv."""
    return " ".join(f"{p['x']:.1f} {p['y']:.1f}" for p in lane) if lane else ""


PROF_FIELDS = ["d_med", "d_bottom", "d_middle", "d_top", "x_mid", "steep", "same_sign"]


def offset_profile(gline, lane, ori_hw, ev):
    """Signed normal offset of a prediction along its GT lane, eval space (224 x 224), in PROF_FIELDS order: median
    over the arc both cover and over its bottom / middle / top thirds, oriented to the image's right (down for a flat
    lane, chord within 45 deg of horizontal); the eval x of the lane's middle; steep (1) or flat (0); the share of
    covered points on the median's side. None if fewer than 3 covered points."""
    if gline is None or lane is None or len(lane) < 2:
        return None
    poly, cum = gline
    pts = eb._eval_space(lane, ori_hw, ev)
    s, d = eb._project(pts, poly, cum)
    inside = (s >= 0) & (s <= cum[-1])
    if inside.sum() < 3:
        return None
    s, d = s[inside], d[inside]
    chord = poly[-1] - poly[0]                              # _project's normal is (-ab_y, ab_x) along the GT polyline
    steep = abs(chord[1]) >= abs(chord[0])
    sign = np.sign(-chord[1]) if steep else np.sign(chord[0])
    d = d * (sign if sign != 0 else 1.0)
    lo, hi = s.min(), s.max()
    e1, e2 = lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3
    thirds = [d[s <= e1], d[(s > e1) & (s < e2)], d[s >= e2]]
    med = [float(np.median(t)) if len(t) else float("nan") for t in thirds]
    if poly[0, 1] < poly[-1, 1]:                            # arc 0 at the top of the image
        med = med[::-1]
    x_mid = float(eb._at(poly, cum, np.array([(lo + hi) / 2]))[0][0, 0])
    m = float(np.median(d))
    return (m, med[0], med[1], med[2], x_mid, float(steep), float(np.mean(np.sign(d) == np.sign(m))))


def analyse_image(rec):
    st = _W["st"]
    ori_hw, offset = tuple(rec["ori_hw"]), int(rec["offset"])
    gt = [ln for ln in rc.read_lines_txt(eb.gt_file(rec["filename"])) if len(ln) > 0]
    attrs, _ = eb.gt_geometry(gt, ori_hw, offset, st)
    gt_masks = [rc.lane_mask([{"x": x, "y": y} for x, y in ln], ori_hw, st["eval"]) for ln in gt]
    rows0, score0 = rec["res"][("out", 0.0)]
    n = len(score0)
    lanes0 = [_lane(rows0[i], ori_hw, offset) for i in range(n)]
    masks0 = [None if ln is None else rc.lane_mask(ln, ori_hw, st["eval"]) for ln in lanes0]
    valid = [i for i in range(n) if masks0[i] is not None]
    iou0 = np.zeros((len(gt), n))
    if gt and valid:
        iou0[:, valid] = rc.iou_matrix(gt_masks, [masks0[i] for i in valid])
    best_gt = iou0.argmax(0) if len(gt) else np.full(n, -1)
    lanes = []
    for g in range(len(gt)):
        copies = np.nonzero(iou0[g] > HIT)[0]
        near = np.nonzero((best_gt == g) & (iou0[g] > NEAR) & (iou0[g] <= HIT))[0]
        sc, sn = score0[copies][:, None], score0[near][None, :]
        row = dict(image=rec["filename"], gt=g, slant=attrs[g]["slant"], extent=attrs[g]["extent"],
                   k=int(len(copies)), best_copy_score=float(score0[copies].max()) if len(copies) else float("nan"),
                   top_iou_is_top_score=(bool(np.argmax(iou0[g, copies]) == np.argmax(score0[copies]))
                                         if len(copies) >= 2 else None),
                   n_near=int(len(near)), n_near_above=int((score0[near] >= _W["thr"]).sum()),
                   near_conc=float((sc > sn).sum() + 0.5 * (sc == sn).sum()), near_pairs=int(sc.size * sn.size))
        if len(copies):
            p = int(copies[np.argmax(iou0[g, copies])])
            row["query"] = p
            for (mode, d), (rows, score) in rec["res"].items():
                if (mode, d) == ("out", 0.0):
                    v = float(iou0[g, p])
                else:
                    m = _mask(rows[p], ori_hw, offset)
                    v = float(rc.iou_matrix([gt_masks[g]], [m])[0, 0]) if m is not None else 0.0
                row[f"iou_{mode}_{d:g}"] = v
                row[f"score_{mode}_{d:g}"] = float(score[p])
        lanes.append(row)
    real = [(int(q), int(best_gt[q]), bool(iou0[best_gt[q], q] > HIT)) for q in range(n)
            if len(gt) and iou0[best_gt[q], q] > NEAR]
    near_nms, dist = eb.kernel_pairs(rows0, st["n_rows_m1"], st["img_w"], _W["nms_thres"])
    # 6. near misses and the head's top copies against their lane: extent vs lateral position
    glines, nm, tc, prof_nm, prof_tc, nm_rows = {}, [], [], [], [], []

    def gline(g):
        if g not in glines:
            glines[g] = eb.gt_polyline([{"x": x, "y": y} for x, y in gt[g]], ori_hw, st["eval"])
        return glines[g]

    def ext(g, q):
        return tuple(float(v) for v in eb.extent_lateral(gline(g), gt_masks[g], lanes0[q], ori_hw, st["eval"]))
    hit_above = [bool(((iou0[g] > HIT) & (score0 >= _W["thr"])).any()) for g in range(len(gt))]
    # 8. evidence for each GT lane: outermost, marking contrast in its bottom / top third, image brightness
    with np.errstate(all="ignore"):
        img = cv2.imread(rec["filename"], cv2.IMREAD_REDUCED_GRAYSCALE_2)
        img = None if img is None else img.astype(np.float32)
        bright = float(img.mean()) if img is not None else float("nan")
        outer = outermost_lanes(gt) if gt else []
        near_above = np.zeros(len(gt), bool)
        for q in range(n):
            if len(gt) and masks0[q] is not None and NEAR < iou0[best_gt[q], q] <= HIT and score0[q] >= _W["thr"]:
                near_above[best_gt[q]] = True
        for g in range(len(gt)):
            ev = (float(outer[g]),) + lane_evidence(img, gt[g], 2, ori_hw[1]) + (bright,)
            status = ("hit" if hit_above[g] else "near only" if near_above[g] else
                      "low score" if lanes[g]["k"] else "missed")
            lanes[g].update(dict(zip(EVID_FIELDS, ev)), status=status)
    prior = neighbour_prior(lanes0, masks0, score0, iou0, best_gt, gt, gt_masks, hit_above, ori_hw, _W["thr"],
                            st["eval"])
    for q in range(n):
        if masks0[q] is not None and len(gt) and NEAR < iou0[best_gt[q], q] <= HIT:
            g = int(best_gt[q])
            above = bool(score0[q] >= _W["thr"])
            e = ext(g, q)
            nm.append((above, short_or_flat(attrs[g]), float(iou0[g, q]), e, hit_above[g], q))
            pr = offset_profile(gline(g), lanes0[q], ori_hw, st["eval"])
            prof_nm.append((above, short_or_flat(attrs[g]), pr))
            copies = np.nonzero(iou0[g] > HIT)[0]
            c = int(copies[np.argmax(score0[copies])]) if len(copies) else -1
            nm_rows.append(dict(image=rec["filename"], ori_h=ori_hw[0], ori_w=ori_hw[1], gt=g, query=q,
                                score=float(score0[q]), above=above, lane_hit_by_copy=hit_above[g],
                                iou=float(iou0[g, q]), short_or_flat=short_or_flat(attrs[g]),
                                **dict(zip(eb.EXT_FIELDS, e)),
                                **dict(zip(PROF_FIELDS, pr if pr is not None else (float("nan"),) * len(PROF_FIELDS))),
                                **{k: lanes[g][k] for k in EVID_FIELDS}, points=_pts_str(lanes0[q]), copy_query=c,
                                copy_score=float(score0[c]) if c >= 0 else float("nan"),
                                copy_iou=float(iou0[g, c]) if c >= 0 else float("nan"),
                                copy_points=_pts_str(lanes0[c]) if c >= 0 else ""))
    for g in range(len(gt)):
        copies = np.nonzero(iou0[g] > HIT)[0]
        if len(copies):
            q = int(copies[np.argmax(score0[copies])])
            if score0[q] >= _W["thr"]:
                tc.append((True, short_or_flat(attrs[g]), float(iou0[g, q]), ext(g, q), True, q))
                prof_tc.append((True, short_or_flat(attrs[g]), offset_profile(gline(g), lanes0[q], ori_hw,
                                                                               st["eval"])))
    return dict(lanes=lanes, real=real, nm=nm, tc=tc, prof_nm=prof_nm, prof_tc=prof_tc, nm_rows=nm_rows,
                prior=prior,
                iou0=iou0.astype(np.float32),
                valid=np.asarray([m is not None for m in masks0]),
                near_nms=near_nms, score0=np.asarray(score0, np.float64),
                geo=geometry(rows0, rec["res"][("in", 0.0)][0], near_nms, dist, score0, st))


GEO_NAMES = ("last-step move", "nearest query", "2nd nearest query", "queries within NMS", "higher-scored query within "
             "NMS", "start y", "start x", "angle", "length")


def geometry(rows, rows_in, near, dist, score, st):
    """Explicit geometry per query [N, 9] (check 5): log(1 + px) of the last step's move (mean |dx| between the output
    and the input of the last step, common rows) and of the distances to the nearest and the 2nd nearest other query
    (mean |dx|, the NMS kernel's); the number of queries within the NMS radius; whether a higher-scored one is among
    them; the official start y, start x, angle and length. Distances without common rows count as 200 px."""
    n = len(rows)
    if n == 0:
        return np.zeros((0, len(GEO_NAMES)), np.float32)
    _, both = eb.kernel_pairs(np.concatenate([rows, rows_in]), st["n_rows_m1"], st["img_w"], _W["nms_thres"])
    move = np.minimum(np.diagonal(both[:n, n:]), 200.0)
    d = np.where(np.eye(n, dtype=bool), np.inf, dist)
    d = np.sort(np.minimum(d, 200.0), axis=1)
    nn1 = d[:, 0]
    nn2 = d[:, 1] if n > 1 else np.full(n, 200.0)
    higher = (near & (score[None, :] > score[:, None])).any(1)
    return np.stack([np.log1p(move), np.log1p(nn1), np.log1p(nn2), near.sum(1), higher,
                     rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def _median(v):
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else float("nan")


def within_lane_auc(lanes, mode, shifts):
    conc = total = 0.0
    for ln in lanes:
        iou = np.array([ln[f"iou_{mode}_{d:g}"] for d in shifts])
        sc = np.array([ln[f"score_{mode}_{d:g}"] for d in shifts])
        for i in np.nonzero(iou > HIT)[0]:
            for j in np.nonzero(iou <= HIT)[0]:
                total += 1
                conc += 1.0 if sc[i] > sc[j] else 0.5 if sc[i] == sc[j] else 0.0
    return (conc / total if total else float("nan")), int(total)


def edge_ratio(lanes, mode, shifts):
    """Per lane: the score at the smallest |d| (> 0) where the lane is lost (IoU <= 0.5), over the score at d = 0."""
    out = []
    pos = sorted({abs(d) for d in shifts if d != 0})
    for ln in lanes:
        s0 = ln[f"score_{mode}_0"]
        if not s0 > 0:
            continue
        for a in pos:
            lost = [d for d in (a, -a) if d in shifts and ln[f"iou_{mode}_{d:g}"] <= HIT]
            if lost:
                out.append(min(ln[f"score_{mode}_{d:g}"] for d in lost) / s0)
                break
    return _median(out), len(out)


def report(lanes, shifts, thr, say):
    tracked = [ln for ln in lanes if "query" in ln]
    groups = groups_all(tracked)
    say(f"GT lanes: {len(lanes)}; with a copy at d = 0 (IoU > {HIT}): {len(tracked)}")
    summary = dict(n_gt=len(lanes), n_tracked=len(tracked), shift={}, copies={})
    for mode in MODES:
        say("")
        say(f"1. shift, output = {'the shifted input' if mode == 'in' else 'the head own last update'} ({mode})")
        say(f"  {'d px':>6} {'med IoU':>8} {'IoU>0.5':>8} {'med score':>10} {'score/s0':>9}")
        rows = []
        for d in sorted(shifts):
            iou = [ln[f"iou_{mode}_{d:g}"] for ln in tracked]
            sc = [ln[f"score_{mode}_{d:g}"] for ln in tracked]
            ratio = [ln[f"score_{mode}_{d:g}"] / ln[f"score_{mode}_0"] for ln in tracked if ln[f"score_{mode}_0"] > 0]
            r = dict(d=d, med_iou=_median(iou), hit=float(np.mean(np.asarray(iou) > HIT)) if iou else float("nan"),
                     med_score=_median(sc), med_ratio=_median(ratio))
            rows.append(r)
            say(f"  {d:>6g} {r['med_iou']:>8.3f} {100 * r['hit']:>7.1f}% {r['med_score']:>10.3f} "
                f"{r['med_ratio']:>9.3f}")
        summary["shift"][mode] = dict(table=rows)
        for name, sub in groups:
            auc, npairs = within_lane_auc(sub, mode, shifts)
            er, ne = edge_ratio(sub, mode, shifts)
            summary["shift"][mode][name] = dict(auc=auc, pairs=npairs, edge_ratio=er, n_edge=ne, n=len(sub))
            say(f"  {name:<14} n {len(sub):>6}  within-lane AUC {auc:.3f} ({npairs} pairs)  "
                f"edge score ratio {er:.3f} ({ne} lanes)")
    say("")
    say(f"2. copies at d = 0 (out): K = queries with IoU > {HIT}; low score = best copy below {thr:.2f}")
    say(f"  {'K':>4} {'GT lanes':>9} {'% of GT':>8} {'med best score':>15} {'low score':>10} "
        f"{'top-IoU top-scored':>19} {'chance':>7}")
    bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 10 ** 6)]
    for lo, hi in bins:
        sub = [ln for ln in lanes if lo <= ln["k"] <= hi]
        lab = f"{lo}" if lo == hi else f"{lo}+"
        if not sub:
            say(f"  {lab:>4} {0:>9}")
            continue
        best = [ln["best_copy_score"] for ln in sub]
        low = float(np.mean([b < thr for b in best])) if lo > 0 else float("nan")
        rk = [ln["top_iou_is_top_score"] for ln in sub if ln["top_iou_is_top_score"] is not None]
        chance = float(np.mean([1.0 / ln["k"] for ln in sub])) if lo >= 2 else float("nan")
        acc = float(np.mean(rk)) if rk else float("nan")
        summary["copies"][lab] = dict(n=len(sub), share=len(sub) / max(1, len(lanes)), med_best=_median(best),
                                      low_score=low, top_iou_top_score=acc, chance=chance)
        say(f"  {lab:>4} {len(sub):>9} {100 * len(sub) / max(1, len(lanes)):>7.1f}% {_median(best):>15.3f} "
            f"{100 * low:>9.1f}% {100 * acc:>18.1f}% {100 * chance:>6.1f}%")
    say("")
    say(f"3. near misses at d = 0 (out): queries whose best lane it is, IoU in ({NEAR}, {HIT}]")
    summary["near"] = {}
    for name, sub in groups_all(lanes):
        pairs = sum(ln["near_pairs"] for ln in sub)
        auc = sum(ln["near_conc"] for ln in sub) / pairs if pairs else float("nan")
        n_near = sum(ln["n_near"] for ln in sub)
        above = sum(ln["n_near_above"] for ln in sub)
        summary["near"][name] = dict(n_gt=len(sub), n_near=n_near, near_above=above, auc=auc, pairs=pairs)
        say(f"  {name:<14} GT {len(sub):>6}  near misses {n_near:>6} ({n_near / max(1, len(sub)):.2f} per lane), "
            f"{above} above {thr:.2f}  copy-over-near-miss AUC {auc:.3f} ({pairs} pairs)")
    return summary


def short_or_flat(ln):
    return ln["extent"] < 0.4 or ln["slant"] > 4


def groups_all(lanes):
    return [("all", lanes), ("short or flat", [ln for ln in lanes if short_or_flat(ln)]),
            ("the rest", [ln for ln in lanes if not short_or_flat(ln)])]


# ----------------------------------------------------------------------------
# 4. linear probe on the classifier's input
# ----------------------------------------------------------------------------
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)


def group_auc(score, label, group):
    """Within-group AUC: over the (positive, negative) pairs of each group, how often the positive scores higher
    (ties 0.5) -> (AUC, pairs)."""
    conc = total = 0.0
    order = np.argsort(group, kind="stable")
    score, label, group = score[order], label[order], group[order]
    cuts = np.flatnonzero(np.diff(group)) + 1
    for s, y in zip(np.split(score, cuts), np.split(label, cuts)):
        pos, neg = s[y], s[~y]
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            conc += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
            total += diff.size
    return (conc / total if total else float("nan")), int(total)


def fit_logistic(x, y, lam, iters=200):
    """L2 logistic regression (LBFGS) on x [M, D] (already standardized) -> (w [D], b)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + lam * (w * w).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def probe(x, y, score, group, image, short, above=None):
    """Fit on even images (lambda chosen on images 2 mod 4 after fitting on 0 mod 4), score on odd images ->
    dict per subset (all / short or flat / the rest): the head score's and the probe's within-group AUC."""
    train, test = image % 2 == 0, image % 2 == 1
    if not (train.any() and test.any()) or y[train].all() or not y[train].any():
        return None

    def norm(sel):
        mu, sd = x[sel].mean(0), x[sel].std(0) + 1e-6
        return lambda z: (z - mu) / sd
    fit, val = image % 4 == 0, image % 4 == 2
    best = (LAMBDAS[1], -1.0)
    if fit.any() and val.any() and y[fit].any() and not y[fit].all():
        f = norm(fit)
        for lam in LAMBDAS:
            w, b = fit_logistic(f(x[fit]), y[fit], lam)
            auc, _ = group_auc(f(x[val]) @ w + b, y[val], group[val])
            if auc == auc and auc > best[1]:
                best = (lam, auc)
    lam = best[0]
    f = norm(train)
    w, b = fit_logistic(f(x[train]), y[train], lam)
    z = f(x) @ w + b
    out = dict(lam=lam, dim=int(x.shape[1]), train_auc=group_auc(z[train], y[train], group[train])[0])
    subsets = [("all", test), ("short or flat", test & short), ("the rest", test & ~short)]
    if above is not None:
        subsets.append(("both above thr", test & above))
    for name, sub in subsets:
        a_s, n = group_auc(score[sub], y[sub], group[sub])
        a_p, _ = group_auc(z[sub], y[sub], group[sub])
        out[name] = dict(score_auc=a_s, probe_auc=a_p, pairs=n)
    return out


def probe_samples(per, records, states, shifts):
    """(shifted: hit vs miss of each tracked lane's shifted copies, mode in; real: copies vs near misses at d = 0) ->
    two tuples (x, y, score, group, image, short)."""
    sh, re_ = [[] for _ in range(6)], [[] for _ in range(6)]
    gid = 0
    for i, (p, rec, stt) in enumerate(zip(per, records, states)):
        base = gid
        for ln in p["lanes"]:
            sf = short_or_flat(ln)
            if "query" in ln:
                q = ln["query"]
                for d in shifts:
                    for lst, v in zip(sh, (stt[("in", d)][q], ln[f"iou_in_{d:g}"] > HIT, ln[f"score_in_{d:g}"],
                                           gid, i, sf)):
                        lst.append(v)
            gid += 1
        score0 = rec["res"][("out", 0.0)][1]
        for q, g, lab in p["real"]:
            for lst, v in zip(re_, (stt[("out", 0.0)][q], lab, float(score0[q]), base + g, i,
                                    short_or_flat(p["lanes"][g]))):
                lst.append(v)

    def arr(cols):
        if not cols[0]:
            return None
        return (np.stack(cols[0]).astype(np.float32), np.asarray(cols[1], bool), np.asarray(cols[2], np.float64),
                np.asarray(cols[3]), np.asarray(cols[4]), np.asarray(cols[5], bool))
    return arr(sh), arr(re_)


def probe_report(per, records, states, shifts, say, thr=None):
    say("")
    say("4. linear probe on the vector the classifier reads: fit on even images, within-lane AUC on odd images "
        "(chance 0.5)")
    res = {}
    for key, title, data in zip(("shifted", "real"),
                                ("shifted copies (in): hit vs miss", "real queries: copy vs near miss"),
                                probe_samples(per, records, states, shifts)):
        above = data[2] >= thr if data is not None and key == "real" and thr is not None else None
        r = probe(*data, above=above) if data is not None else None
        res[key] = r
        if r is None:
            say(f"  {title}: not enough samples")
            continue
        say(f"  {title} (D {r['dim']}, lambda {r['lam']:g}, probe train AUC {r['train_auc']:.3f})")
        say(f"    {'':<14} {'head score':>10} {'probe':>7} {'pairs':>9}")
        for name in ("all", "short or flat", "the rest", "both above thr"):
            if name in r:
                v = r[name]
                say(f"    {name:<14} {v['score_auc']:>10.3f} {v['probe_auc']:>7.3f} {v['pairs']:>9}")
    return res


# ----------------------------------------------------------------------------
# 5. F1 bound of rescoring with a probe
# ----------------------------------------------------------------------------
THRS = np.round(np.arange(0.02, 0.99, 0.01), 2)


def image_tp(iou, out):
    """vega: Hungarian on 1 - IoU over (GT, outputs); hits = matched pairs with IoU > 0.5."""
    return sum(v > HIT for _, v in eb.assignment(iou, out).values())


def sweep_f1(per, scores, nms, top_k):
    """Best F1@50 of per-image scores (list of [N] arrays) over THRS plus 200 quantiles of the decodable queries'
    scores (probe outputs need not share the head's scale) -> dict(f1, p, r, thr, ...)."""
    n_gt = sum(p["iou0"].shape[0] for p in per)
    pool = np.concatenate([sc[p["valid"]] for p, sc in zip(per, scores)] + [np.zeros(0)])
    thrs = np.unique(np.concatenate([THRS, np.quantile(pool, np.linspace(0, 1, 201)) if len(pool) else []]))
    tp, n_out = np.zeros(len(thrs), np.int64), np.zeros(len(thrs), np.int64)
    for p, sc in zip(per, scores):
        prev, prev_tp = None, 0
        for k, t in enumerate(thrs):
            kept, _, _ = eb.select(sc, p["near_nms"], float(t), "nms" if nms else "free", top_k)
            out = sorted(i for i in kept if p["valid"][i])
            if out != prev:                              # the kept set changes only at a few thresholds
                prev, prev_tp = out, image_tp(p["iou0"], out)
            tp[k] += prev_tp
            n_out[k] += len(out)
    best = dict(f1=-1.0)
    for k, t in enumerate(thrs):
        pr, rc_, f1 = eb.prf(int(tp[k]), int(n_out[k]), n_gt)
        if f1 > best["f1"]:
            best = dict(f1=f1, p=pr, r=rc_, thr=float(t), n_out=int(n_out[k]), tp=int(tp[k]))
    best["n_gt"] = n_gt
    return best


def oracle_scores(p, kind):
    """The head's scores with GT knowledge (ceilings): "near" drops every near miss (best IoU in (NEAR, HIT]) to 0;
    "recover" scores the best-IoU copy of each lane 1."""
    sc = p["score0"].copy()
    iou = p["iou0"]
    if iou.shape[0] == 0:
        return sc
    if kind == "near":
        best = iou.max(0)
        sc[p["valid"] & (best > NEAR) & (best <= HIT)] = 0.0
    else:
        for g in range(iou.shape[0]):
            if (iou[g] > HIT).any():
                sc[int(np.argmax(iou[g]))] = 1.0
    return sc


def placed_iou(p):
    """The image's IoU matrix with every near miss moved onto its lane: its IoU with that lane becomes iou_lat (the
    metric IoU given the GT lateral position, extent kept) when that is higher."""
    iou = p["iou0"].copy()
    for e in p["nm"]:
        q, g, lat = e[5], int(np.argmax(iou[:, e[5]])), e[3][6]
        if lat == lat:
            iou[g, q] = max(iou[g, q], lat)
    return iou


def _logit(sc):
    s_ = np.clip(sc, 1e-6, 1.0 - 1e-6)
    return np.log(s_) - np.log1p(-s_)


def fit_probe_scores(per, feats_of, label_of):
    """Fit an L2 logistic probe on every decodable query of the even images (lambda chosen on images 2 mod 4 after
    fitting on 0 mod 4, by within-image AUC) -> (lambda, {odd image: probability [N]})."""
    xs, ys, gs = [], [], []
    for i, p in enumerate(per):
        v = np.nonzero(p["valid"])[0]
        if i % 2 or not len(v):
            continue
        xs.append(feats_of(i)[v])
        ys.append(label_of(p)[v])
        gs.append(np.full(len(v), i))
    if not xs:
        return None, None
    x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
    if y.all() or not y.any():
        return None, None
    mu, sd = x.mean(0), x.std(0) + 1e-6
    fit, val = g % 4 == 0, g % 4 == 2
    lam, best_auc = LAMBDAS[1], -1.0
    if y[fit].any() and not y[fit].all() and val.any():
        for cand in LAMBDAS:
            w, b = fit_logistic((x[fit] - mu) / sd, y[fit], cand)
            auc, _ = group_auc((x[val] - mu) / sd @ w + b, y[val], g[val])
            if auc == auc and auc > best_auc:
                lam, best_auc = cand, auc
    w, b = fit_logistic((x - mu) / sd, y, lam)
    prob = {}
    for i in range(1, len(per), 2):
        z = ((feats_of(i) - mu) / sd @ w + b).astype(np.float64)
        prob[i] = 1.0 / (1.0 + np.exp(-z))
    return lam, prob


def hit_labels(p):
    """A query that hits some lane (IoU > 0.5): duplicates count. For rows with NMS."""
    return p["iou0"].max(0) > HIT if p["iou0"].shape[0] else np.zeros(len(p["score0"]), bool)


def top_copy_labels(p):
    """Per lane, the head's own top-scored copy (IoU > 0.5) is the positive; every other query (lower copies, near
    misses, the rest) is negative. For rows without NMS: a perfect probe on it keeps the head's selection, drops
    everything else, and gives the best-copy oracle."""
    lab = np.zeros(len(p["score0"]), bool)
    for g in range(p["iou0"].shape[0]):
        copies = np.nonzero((p["iou0"][g] > HIT) & p["valid"])[0]
        if len(copies):
            lab[copies[np.argmax(p["score0"][copies])]] = True
    return lab


def f1_bound(per, states, top_k, say):
    say("")
    say("5. F1 bound: the head's outputs at d = 0 rescored by probes fit on all queries of the even images, or by GT "
        "oracles; swept F1@50 on the odd images. Targets: hit = IoU > 0.5 with some lane (rows with NMS); top copy = "
        "the head's top-scored copy of each lane, all else negative (rows without NMS)")

    def logit(i):
        return _logit(per[i]["score0"]).astype(np.float32)[:, None]

    def vec(i):
        return states[i][("out", 0.0)].astype(np.float32)
    feats = dict(vector=vec,
                 logit=logit,
                 logit_vector=lambda i: np.concatenate([logit(i), vec(i)], axis=1),
                 logit_geo=lambda i: np.concatenate([logit(i), per[i]["geo"]], axis=1),
                 logit_vector_geo=lambda i: np.concatenate([logit(i), vec(i), per[i]["geo"]], axis=1))
    fits = {}
    for key, fn, lab in (("vector/hit", "vector", hit_labels), ("logit+vector/hit", "logit_vector", hit_labels),
                         ("logit/top", "logit", top_copy_labels), ("logit+vector/top", "logit_vector", top_copy_labels),
                         ("logit+geo/top", "logit_geo", top_copy_labels),
                         ("logit+vector+geo/top", "logit_vector_geo", top_copy_labels)):
        fits[key] = fit_probe_scores(per, feats[fn], lab)
    if any(prob is None for _, prob in fits.values()):
        say("  not enough samples")
        return None
    odd = sorted(fits["vector/hit"][1])
    sub = [per[i] for i in odd]
    head = [p["score0"] for p in sub]

    def got(key):
        return [fits[key][1][i] for i in odd]
    pr = got("vector/hit")
    rows = [("head score, no NMS (the run's own output)", head, False),
            ("head score + lane NMS", head, True),
            ("vector, hit, no NMS", pr, False),
            ("vector, hit + lane NMS", pr, True),
            ("sqrt(head score x vector/hit), no NMS", [np.sqrt(h * q) for h, q in zip(head, pr)], False),
            ("logit + vector, hit + lane NMS", got("logit+vector/hit"), True),
            ("logit alone, top copy, no NMS (control)", got("logit/top"), False),
            ("logit + vector, top copy, no NMS", got("logit+vector/top"), False),
            ("logit + geometry, top copy, no NMS", got("logit+geo/top"), False),
            ("logit + vector + geometry, top copy, no NMS", got("logit+vector+geo/top"), False),
            ("oracle: near misses scored 0, no NMS", [oracle_scores(p, "near") for p in sub], False),
            ("oracle: best copy of each lane scored 1, no NMS", [oracle_scores(p, "recover") for p in sub], False)]
    placed = [placed_iou(p) for p in sub]
    say("  lambda " + ", ".join(f"{k} {v[0]:g}" for k, v in fits.items()) + f"; odd images {len(sub)}; GT lanes "
        f"{sum(p['iou0'].shape[0] for p in sub)}")
    say(f"  geometry: {', '.join(GEO_NAMES)}")
    say(f"  {'scores':<48} {'F1@50':>6} {'P':>6} {'R':>6} {'thr':>6}")
    res = {}
    for name, sc, nms in rows:
        r = sweep_f1(sub, sc, nms, top_k)
        res[name] = r
        say(f"  {name:<48} {100 * r['f1']:>6.2f} {100 * r['p']:>6.2f} {100 * r['r']:>6.2f} {r['thr']:>6.3f}")
    for name, sc in (("oracle: near misses placed on their lane, no NMS", head),
                     ("  + near misses that stay off scored 0", [oracle_scores(dict(p, iou0=pl), "near")
                                                                 for p, pl in zip(sub, placed)])):
        r = sweep_f1([dict(p, iou0=pl) for p, pl in zip(sub, placed)], sc, False, top_k)
        res[name] = r
        say(f"  {name:<48} {100 * r['f1']:>6.2f} {100 * r['p']:>6.2f} {100 * r['r']:>6.2f} {r['thr']:>6.3f}")
    say("  placed: each near miss moved to its lane's lateral position, its extent kept (its IoU becomes the "
        "breakdown's iou_lat, the metric IoU of that curve); head scores unchanged")
    return dict(lam={k: v[0] for k, v in fits.items()}, rows=res)


# ----------------------------------------------------------------------------
# 6. what the near misses are
# ----------------------------------------------------------------------------
def near_miss_report(per, thr, say):
    """Near misses above / below the threshold and the head's top copies above it (hits, the reference), measured
    against their lane with the error breakdown's extent_lateral (eval space)."""
    say("")
    say(f"6. near misses against their lane (eval space 224 x 224, 1 eval px ~ 3.6 px of the 800-px input). extent / "
        f"lateral: IoU > {HIT} once the GT extent alone / the GT lateral position alone replaces the prediction's")
    nm = [e for p in per for e in p["nm"]]
    groups = [("near misses >= thr", [e for e in nm if e[0]]),
              ("  short or flat", [e for e in nm if e[0] and e[1]]),
              ("  the rest", [e for e in nm if e[0] and not e[1]]),
              ("  lane hit by a copy", [e for e in nm if e[0] and e[4]]),
              ("  lane missed", [e for e in nm if e[0] and not e[4]]),
              ("near misses < thr", [e for e in nm if not e[0]]),
              ("top copies >= thr (hits)", [e for p in per for e in p["tc"]])]
    say(f"  {'':<26} {'n':>6} {'IoU':>5} {'extent':>7} {'lateral':>8} {'either':>7} {'neither':>8} {'lat px':>7} "
        f"{'cover':>6} {'overrun':>8} {'end bot':>8} {'end top':>8}")
    out = {}
    for name, sub in groups:
        if not sub:
            say(f"  {name:<26} {0:>6}")
            continue
        e = np.array([x[3] for x in sub], dtype=np.float64)                    # [n, 7] EXT_FIELDS
        ok = ~(np.isnan(e[:, 5]) | np.isnan(e[:, 6]))
        fe, fl = e[ok, 5] > HIT, e[ok, 6] > HIT
        share = {k: float(np.mean(v)) if ok.any() else float("nan")
                 for k, v in (("extent", fe & ~fl), ("lateral", fl & ~fe), ("either", fe & fl), ("neither", ~fe & ~fl))}

        def med(col):
            v = e[:, col][~np.isnan(e[:, col])]
            return float(np.median(v)) if len(v) else float("nan")
        r = dict(n=len(sub), iou=float(np.median([x[2] for x in sub])), **share, lat_px=med(4), cover=med(0),
                 overrun=med(1), end_bottom=med(2), end_top=med(3), short_share=float(np.mean([x[1] for x in sub])))
        out[name.strip()] = r
        say(f"  {name:<26} {r['n']:>6} {r['iou']:>5.2f} {100 * r['extent']:>6.1f}% {100 * r['lateral']:>7.1f}% "
            f"{100 * r['either']:>6.1f}% {100 * r['neither']:>7.1f}% {r['lat_px']:>7.2f} {r['cover']:>6.2f} "
            f"{r['overrun']:>8.2f} {r['end_bottom']:>8.1f} {r['end_top']:>8.1f}")
    say("  either: each alone turns it into a hit; neither: only both together (or not even that). lat px: median "
        "|normal offset| inside the lane, eval px;")
    say("  cover / overrun: share of the lane's arc covered / run past its ends; end bot / top: signed error at the "
        "lane's bottom / top end, eval px (+ stops short, - runs past).")
    say("  lane hit by a copy: another query hits the lane above the threshold, so the near miss is only a false "
        "positive; lane missed: no copy above it, so it is also a lost lane")
    return out


# ----------------------------------------------------------------------------
# 7. the shape of the offset
# ----------------------------------------------------------------------------
def offset_report(per, say):
    """Signed offsets along the lane (offset_profile): direction, outward from the image centre, size by third of the
    lane, sign consistency, and a linear fit of the offset on the distance from the image centre (a scale error)."""
    say("")
    say("7. offset along the lane, eval px (+ = right; flat lanes: + = down); steep = lanes within 45 deg of vertical")
    groups = [("near misses >= thr", [x[2] for p in per for x in p["prof_nm"] if x[0]]),
              ("  short or flat", [x[2] for p in per for x in p["prof_nm"] if x[0] and x[1]]),
              ("  the rest", [x[2] for p in per for x in p["prof_nm"] if x[0] and not x[1]]),
              ("near misses < thr", [x[2] for p in per for x in p["prof_nm"] if not x[0]]),
              ("top copies >= thr (hits)", [x[2] for p in per for x in p["prof_tc"]])]
    say(f"  {'':<26} {'n':>6} {'|d| bot':>8} {'|d| mid':>8} {'|d| top':>8} {'top/bot':>8} {'one side':>9} "
        f"{'steep n':>8} {'med d':>7} {'right':>6} {'outward':>8} {'slope':>7} {'icpt':>6}")
    out = {}
    for name, sub in groups:
        a = np.array([x for x in sub if x is not None], dtype=np.float64).reshape(-1, len(PROF_FIELDS))
        if not len(a):
            say(f"  {name:<26} {0:>6}")
            continue
        ab = np.abs(a[:, 1:4])

        def med(v):
            v = v[~np.isnan(v)]
            return float(np.median(v)) if len(v) else float("nan")
        ok = ~np.isnan(ab[:, 0]) & ~np.isnan(ab[:, 2]) & (ab[:, 0] > 0)
        st_ = a[a[:, 5] == 1]
        dx = st_[:, 4] - 112.0
        out_sign = np.sign(dx) * np.sign(st_[:, 0])
        if len(st_) >= 3 and np.ptp(dx) > 0:
            slope, icpt = np.polyfit(dx, st_[:, 0], 1)
        else:
            slope, icpt = float("nan"), float("nan")
        r = dict(n=len(a), bot=med(ab[:, 0]), mid=med(ab[:, 1]), top=med(ab[:, 2]),
                 top_over_bot=med(ab[ok, 2] / ab[ok, 0]), one_side=float(np.mean(a[:, 6] >= 0.9)),
                 n_steep=len(st_), med_d=med(st_[:, 0]), right=float(np.mean(st_[:, 0] > 0)) if len(st_) else
                 float("nan"), outward=float(np.mean(out_sign > 0)) if len(st_) else float("nan"),
                 slope=float(slope), icpt=float(icpt))
        out[name.strip()] = r
        say(f"  {name:<26} {r['n']:>6} {r['bot']:>8.2f} {r['mid']:>8.2f} {r['top']:>8.2f} {r['top_over_bot']:>8.2f} "
            f"{100 * r['one_side']:>8.1f}% {r['n_steep']:>8} {r['med_d']:>7.2f} {100 * r['right']:>5.1f}% "
            f"{100 * r['outward']:>7.1f}% {100 * r['slope']:>7.3f} {r['icpt']:>6.2f}")
    say("  |d| bot / mid / top: median |offset| in the lane's bottom / middle / top third; top/bot: median ratio per "
        "query (about 1: constant in the image, the size of a feature cell; well below 1: shrinks toward the "
        "horizon as a")
    say("  parallel line on the road would); one side: share of queries with >= 90% of points on one side (a shift, "
        "not a crossing). Steep lanes only: med d, share to the right, share away from the image centre (x 112),")
    say("  slope (eval px of offset per 100 eval px from the centre) and intercept of a linear fit of d on x - 112: a "
        "coordinate scale error shows as a slope, a fixed shift as an intercept, in the hits as well")
    return out


# ----------------------------------------------------------------------------
# 8. evidence along the lane
# ----------------------------------------------------------------------------
STATUS = ("hit", "near only", "low score", "missed")


def evidence_report(lanes, say):
    """Per GT lane: outcome (hit = a copy above the threshold; near only = none, but a near miss above it; low score =
    a copy below it; missed) against how visible the lane is: outermost on its side, marking contrast in the bottom /
    top third of its arc (gray levels), image brightness."""
    say("")
    say(f"8. evidence along each GT lane (grayscale 0-255, image at half size): outcome vs outermost lane, marking "
        f"contrast (line minus its sides) in the lane's bottom / top third, step |left - right| (curb-like edge), "
        f"image brightness; weak = contrast < {WEAK:g}")
    rows = [ln for ln in lanes if "status" in ln]
    if not rows:
        say("  no lanes")
        return None

    def col(sub, k):
        v = np.array([ln[k] for ln in sub], np.float64)
        return v[~np.isnan(v)]

    def med(sub, k):
        v = col(sub, k)
        return float(np.median(v)) if len(v) else float("nan")
    unread = sum(ln["brightness"] != ln["brightness"] for ln in rows)
    say(f"  GT lanes {len(rows)}; on images that could not be read: {unread}" + ("  <- check the image paths"
                                                                                if unread else ""))
    out = dict(by_status={}, by_bin={}, unread=unread)
    say(f"  {'outcome':<12} {'lanes':>6} {'% GT':>6} {'outermost':>10} {'contr bot':>10} {'weak bot':>9} "
        f"{'step bot':>9} {'contr top':>10} {'bright':>7}")
    for stt in STATUS:
        sub = [ln for ln in rows if ln["status"] == stt]
        if not sub:
            continue
        cb = col(sub, "contrast_bot")
        r = dict(n=len(sub), share=len(sub) / len(rows), outermost=float(np.mean(col(sub, "outermost"))),
                 contrast_bot=med(sub, "contrast_bot"), weak_bot=float(np.mean(cb < WEAK)) if len(cb) else
                 float("nan"), step_bot=med(sub, "step_bot"), contrast_top=med(sub, "contrast_top"),
                 brightness=med(sub, "brightness"))
        out["by_status"][stt] = r
        say(f"  {stt:<12} {r['n']:>6} {100 * r['share']:>5.1f}% {100 * r['outermost']:>9.1f}% "
            f"{r['contrast_bot']:>10.1f} {100 * r['weak_bot']:>8.1f}% {r['step_bot']:>9.1f} {r['contrast_top']:>10.1f} "
            f"{r['brightness']:>7.1f}")
    br = col(rows, "brightness")
    t1, t2 = (np.quantile(br, [1 / 3, 2 / 3]) if len(br) else (float("nan"), float("nan")))
    splits = [("contrast bot", "contrast_bot", [("< 5", lambda v: v < 5), ("5-10", lambda v: 5 <= v < 10),
                                                ("10-20", lambda v: 10 <= v < 20), ("20-40", lambda v: 20 <= v < 40),
                                                (">= 40", lambda v: v >= 40)]),
              ("outermost", "outermost", [("yes", lambda v: v == 1), ("no", lambda v: v == 0)]),
              ("brightness", "brightness", [(f"< {t1:.0f}", lambda v: v < t1), (f"{t1:.0f}-{t2:.0f}",
                                                                                   lambda v: t1 <= v < t2),
                                            (f">= {t2:.0f}", lambda v: v >= t2)])]
    say(f"  {'lanes by':<26} {'lanes':>6} {'% GT':>6} {'hit':>6} {'near only':>10} {'low score':>10} {'missed':>7}")
    for title, key, bins in splits:
        for lab, fn in bins:
            sub = [ln for ln in rows if ln[key] == ln[key] and fn(ln[key])]
            if not sub:
                continue
            r = {stt: float(np.mean([ln["status"] == stt for ln in sub])) for stt in STATUS}
            r.update(n=len(sub), share=len(sub) / len(rows))
            out["by_bin"][f"{title} {lab}"] = r
            say(f"  {title + ' ' + lab:<26} {r['n']:>6} {100 * r['share']:>5.1f}% {100 * r['hit']:>5.1f}% "
                f"{100 * r['near only']:>9.1f}% {100 * r['low score']:>9.1f}% {100 * r['missed']:>6.1f}%")
    say("  near only: no copy above the threshold, but a near miss above it (a false positive and a lost lane). "
        "Lanes whose image could not be read have NaN contrast and drop out of the contrast rows")
    return out


# ----------------------------------------------------------------------------
# 9. neighbour prior
# ----------------------------------------------------------------------------
def prior_report(per, say):
    say("")
    say("9. neighbour prior: each output lane (score >= thr, no NMS) re-angled about its far end so that its "
        "near-field line meets the image's bottom row on the lattice of its neighbours' (equal spacing: flat road, "
        "straight near field); neighbours = the other output lanes (test time) or the other GT lanes (ideal "
        "neighbours)")
    say(f"  {'neighbours':<11} {'applies':>8} {'near-only NMs':>14} {'fixed':>7} {'hits':>7} {'broken':>7} "
        f"{'move NM':>8} {'move hit':>9} {'F1 before':>10} {'after':>7} {'gated':>7}")
    out = {}
    for mode in ("pred", "gt"):
        rs = [p["prior"][mode] for p in per]

        def tot(k):
            return sum(r[k] for r in rs)
        n_gt, n_out = tot("n_gt"), tot("n_out")
        f0, f1, fg = (100 * eb.prf(tot(k), n_out, n_gt)[2] for k in ("tp0", "tp1", "tpg"))
        mv_nm = [v for r in rs for v in r["move_nm"]]
        mv_hit = [v for r in rs for v in r["move_hit"]]
        r = dict(n_out=n_out, app=tot("app") / max(1, n_out), nm=tot("nm"), nm_app=tot("nm_app"),
                 fixed=tot("nm_fixed") / max(1, tot("nm_app")), hit=tot("hit"), hit_app=tot("hit_app"),
                 broken=tot("hit_broken") / max(1, tot("hit_app")),
                 move_nm=float(np.median(mv_nm)) if mv_nm else float("nan"),
                 move_hit=float(np.median(mv_hit)) if mv_hit else float("nan"), f1_before=f0, f1_after=f1, f1_gated=fg)
        out[mode] = r
        say(f"  {mode:<11} {100 * r['app']:>7.1f}% {r['nm_app']:>6} of {r['nm']:<5} {100 * r['fixed']:>6.1f}% "
            f"{r['hit_app']:>7} {100 * r['broken']:>6.1f}% {100 * r['move_nm']:>7.2f}% {100 * r['move_hit']:>8.2f}% "
            f"{f0:>10.2f} {f1:>7.2f} {fg:>7.2f}")
    say("  applies: outputs with a near field within 75 deg of vertical and >= 2 distinct neighbours; near-only NMs: "
        "near misses above thr whose lane has no copy above it (applied of all); fixed: IoU > 0.5 after; hits: the")
    say("  applied outputs with IoU > 0.5, broken: IoU <= 0.5 after; move: median |shift| at the bottom row, % of the "
        "image width; F1 on all images at thr: before, after (prior applied to every output), gated (kept only where "
        "it raises the IoU: a GT-chosen ceiling)")
    return out


# ----------------------------------------------------------------------------
def collect(cfg, args, device):
    from mmengine.runner import Runner
    model = eb.build_model(cfg, args.checkpoint, device)
    head = model.bbox_head
    rescorer = rescorer_for(head)
    meta = eb.head_meta(head)
    loader = Runner.build_dataloader(copy.deepcopy(cfg.test_dataloader))
    records, states, diffs = [], [], []
    tic = time.time()
    for data in loader:
        data = model.data_preprocessor(data, False)
        feats = model.extract_feat(data["inputs"])
        res, stt, diff = run_batch(head, rescorer, feats, args.shifts, keep_states=not args.no_probe)
        diffs.append(diff)
        for b, ds in enumerate(data["data_samples"]):
            m = ds.metainfo
            records.append(dict(filename=str(m["filename"]), ori_hw=(int(m["ori_shape"][0]), int(m["ori_shape"][1])),
                                offset=int(m["crop_offset"][1]),
                                res={k: (v[0][b], v[1][b]) for k, v in res.items()}))
            states.append({k: v[b] for k, v in stt.items()})
        if len(records) % 500 < len(data["data_samples"]):
            print(f"  {len(records)} images ({time.time() - tic:.0f} s)", flush=True)
        if args.max_images is not None and len(records) >= args.max_images:
            break
    n = args.max_images or len(records)
    return records[:n], states[:n], meta, max(diffs) if diffs else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--shifts", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 16],
                    help="px of the network input; each is also run with the opposite sign")
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--thr", type=float, default=None, help="default: the config's test_cfg.conf_threshold")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-probe", action="store_true", help="skip check 4 (keeps no classifier inputs)")
    args = ap.parse_args(argv)
    args.shifts = sorted({float(s) for a in args.shifts for s in (a, -a)} | {0.0})
    missing = [f for f in ("gt_polyline", "extent_lateral") if not hasattr(eb, f)]
    if missing:                                          # check 6 needs the breakdown of 2026-10-22 or later
        sys.exit(f"{eb.__file__} has no {', '.join(missing)}: replace it with the current "
                 f"clrbezierlane/curvelanes_error_breakdown.py (2026-10-22 or later)")

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records, states, meta, diff = collect(cfg, args, device)
    st = eb.settings(cfg, meta, args)
    thr = float(args.thr if args.thr is not None else meta["test_cfg"]["conf_threshold"])
    state = dict(st=st, decoder=eb.make_decoder(meta), thr=thr, nms_thres=float(meta["test_cfg"]["nms_thres"]))
    jobs = args.jobs or max(1, min(16, (os.cpu_count() or 2) - 1))
    if jobs > 1:
        with mp.get_context("fork").Pool(jobs, initializer=_init_worker, initargs=(state,)) as pool:
            per = pool.map(analyse_image, records, chunksize=8)
    else:
        _init_worker(state)
        per = [analyse_image(r) for r in records]
    lanes = [ln for p in per for ln in p["lanes"]]

    out_dir = args.out_dir or osp.join("work_dirs", "score_shift_check", osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"config {args.config}")
    say(f"checkpoint {args.checkpoint}")
    say(f"head {meta['head_cls']}; images {len(records)}; shifts {', '.join(f'{d:g}' for d in args.shifts)} px "
        f"(800 x 320 input); threshold {thr:.2f}")
    say(f"check: d = 0 (out) vs the head's own forward, max difference {diff:.3g} (must be 0)")
    summary = report(lanes, args.shifts, thr, say)
    if not args.no_probe:
        summary["probe"] = probe_report(per, records, states, args.shifts, say, thr)
        summary["f1_bound"] = f1_bound(per, states, int(meta["test_cfg"]["nms_topk"]), say)
    summary["near_misses"] = near_miss_report(per, thr, say)
    summary["offsets"] = offset_report(per, say)
    summary["evidence"] = evidence_report(lanes, say)
    summary["prior"] = prior_report(per, say)
    summary.update(config=args.config, checkpoint=args.checkpoint, images=len(records), threshold=thr,
                   shifts=args.shifts, self_check_max_diff=diff)
    with open(osp.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    keys = sorted({k for ln in lanes for k in ln})
    with open(osp.join(out_dir, "lanes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(lanes)
    nm_rows = [r for p in per for r in p["nm_rows"]]
    if nm_rows:
        with open(osp.join(out_dir, "near_misses.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(nm_rows[0]))
            w.writeheader()
            w.writerows(nm_rows)
    print(f"wrote {out_dir}/summary.txt, summary.json, lanes.csv, near_misses.csv")
    return dict(summary=summary, lanes=lanes, per=per)


if __name__ == "__main__":
    main()
