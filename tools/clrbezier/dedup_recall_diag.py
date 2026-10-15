"""Why does the learned NMS (bbox_head.dedup_cfg) lose recall without NMS?

One inference pass stores, for every image, the last-stage predictions with their
O2M score and their dedup score (the corrected score use_at_test=True puts in).
It then rebuilds the two outputs the way the evaluation does:

  with NMS : dedup score >= --thr-nms, then greedy lane NMS (test_cfg.nms_thres,
             nms_topk); the use_at_test=True + use_nms=True path
  no NMS   : the top nms_topk lanes per image by dedup score, then >= --thr-free;
             the use_at_test=True + use_nms=False path

and matches both to the ground truth as the official CULane metric does
(interpolated lanes drawn 30 px wide on 1640x590, Hungarian, IoU > 0.5).

Every GT lane the NMS output finds and the no-NMS output misses is "lost". The
prediction the NMS output found it with is q; its "copies" are the stored
predictions within the NMS distance of q (mean |dx| < nms_thres over the common
rows, as the NMS kernel computes it). Each lost lane gets one reason:

  o2m_below        q's dedup score is below --thr-free and so is its O2M score: the
                   lane was weak before the dedup head; NMS keeps it only because
                   its threshold is lower
  isolated_lowered q's O2M score is above --thr-free, the dedup head lowered it
                   below, and q has no copy: the head suppressed a lane with no
                   duplicate (miscalibration rather than de-duplication)
  cluster_lowered  as above, q has copies, and every copy also ended below
                   --thr-free: the head could not pick one copy and lowered them all
  copy_kept        as above, and a copy of q was kept instead, but that copy misses
                   the GT at IoU 0.5: the head preferred a worse-localized copy
                   (classification-localization mismatch)
  copy_evicted     as above, a copy stayed above --thr-free but the top-k cap
                   dropped it
  evicted          q's dedup score is above --thr-free, but the top-k cap dropped it
  reassigned       q is in the no-NMS output, but the Hungarian matched it to
                   another GT lane

It also prints P/R/F1 of both outputs (check them against your sweep), the
no-NMS output at the NMS threshold with its duplicate false positives (why the
no-NMS threshold has to sit higher), copy statistics for the lost lanes against
the lanes both outputs find, and a breakdown by CULane category.

Separability: a no-NMS detector needs one global threshold between every true
lane and every duplicate. "True" = the predictions the NMS output matches to a
GT lane; "duplicate" = the other stored predictions within the NMS distance of
one of them. For the O2M score and for the dedup score it prints the AUC
P(true > duplicate) and the score quantiles of both groups. If the dedup AUC is
close to 1, one threshold separates them and the recall loss comes from
elsewhere; if weak true lanes sit inside the duplicates' score range, no
threshold can keep them without keeping duplicates.

Per-lane rows go to lost_lanes_<thr-nms>_<thr-free>.csv.

    python tools/clrbezier/dedup_recall_diag.py CONFIG CKPT --thr-nms 0.38 --thr-free T \\
        --save-dump work_dirs/dedup_diag/dump.pkl
    # other thresholds, no inference (thresholds must stay >= --floor of the dump)
    python tools/clrbezier/dedup_recall_diag.py CONFIG --load-dump work_dirs/dedup_diag/dump.pkl \\
        --thr-nms 0.38 --thr-free T

--thr-free is the best threshold of your no-NMS sweep. The inference pass also
checks, on the first --verify-batches batches, that the rebuilt outputs keep
exactly the lanes the head's own get_lanes keeps on both paths.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import pickle
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ORI_W, ORI_H = 1640, 590       # CULaneMetric.ori_w / ori_h
IOU_WIDTH, IOU_THR = 30, 0.5   # official CULane evaluation
REASONS = ["o2m_below", "isolated_lowered", "cluster_lowered", "copy_kept",
           "copy_evicted", "evicted", "reassigned"]
READING = {
    "o2m_below": "the O2M score was already below the no-NMS threshold; NMS keeps the "
                 "lane only because its threshold is lower",
    "isolated_lowered": "the dedup head lowered a lane that had no duplicate: "
                        "miscalibration, not de-duplication",
    "cluster_lowered": "the dedup head lowered every copy of the lane: it could not "
                       "pick one",
    "copy_kept": "the dedup head kept a worse-localized copy: "
                 "classification-localization mismatch",
    "copy_evicted": "a copy stayed above the threshold but the top-k cap dropped it",
    "evicted": "the top-k cap dropped the lane",
    "reassigned": "the lane was kept but matched to another GT",
}


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def head_outputs(head, pyramid):
    """Last-stage predictions [B, Q, 6+R] (official layout), the RoI features the
    dedup head reads, and the O2M and dedup scores [B, Q], computed as
    CLRBezierHead.forward_test does with dedup_cfg.use_at_test=True."""
    feats = head._select_features(pyramid)
    preds, _, extra = head._refine(feats, head.prior_bank(feats[-1].shape[0]),
                                   head.gsrc_tokens(feats))
    last = preds[-1]
    roi = extra["last_roi"]
    dd_logits = head._dedup_logits(last, head.dedup(last, roi))
    o2m = torch.softmax(last[..., :2].float(), dim=-1)[..., 1]
    dd = torch.softmax(dd_logits.float(), dim=-1)[..., 1]
    return last, roi, o2m, dd


def image_records(last, o2m, dd, names, floor):
    """Per image: the predictions with O2M or dedup score >= floor (CPU, float32).
    Every prediction a threshold >= floor can keep or that can suppress one is
    among them; the top-k rank of a prediction above floor counts only stored
    predictions, so it is exact too."""
    out = []
    keep = (o2m >= floor) | (dd >= floor)
    for b, name in enumerate(names):
        idx = torch.nonzero(keep[b], as_tuple=False).squeeze(1)
        out.append(dict(name=name, idx=idx.cpu().numpy().astype(np.int32),
                        rows=last[b, idx].float().cpu().numpy(),
                        o2m=o2m[b, idx].cpu().numpy(), dd=dd[b, idx].cpu().numpy()))
    return out


# ----------------------------------------------------------------------------
# rebuilding the two outputs (pure torch / numpy)
# ----------------------------------------------------------------------------
def near_matrix(rows, n_strips, x_scale, nms_thres):
    """[n, n] bool: the NMS kernel would suppress one of the pair with the other
    (common rows, mean |dx| < nms_thres). Same geometry as dedup.pairwise_geometry."""
    from libs.clrbezier.dedup import pairwise_geometry
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=bool)
    mean, _, _, count, _, _ = pairwise_geometry(torch.as_tensor(rows), n_strips, x_scale)
    near = ((count > 0) & (mean < nms_thres)).numpy()
    np.fill_diagonal(near, False)
    return near


def cap_mask(dd, k):
    """Top-k by score (the no-NMS cap of CLRBezierHead._dedup_scores)."""
    keep = np.zeros(len(dd), dtype=bool)
    if k is None or k <= 0 or k >= len(dd):
        keep[:] = True
        return keep
    keep[np.argsort(-dd, kind="stable")[:k]] = True
    return keep


def select_nms(dd, near, thr, top_k, cap=None):
    """Indices kept by threshold + greedy lane NMS, in score order."""
    cand = np.nonzero((dd >= thr) & cap_mask(dd, cap))[0]
    cand = cand[np.argsort(-dd[cand], kind="stable")]
    kept = []
    for i in cand:
        if any(near[i, j] for j in kept):
            continue
        kept.append(int(i))
        if len(kept) >= top_k:
            break
    return kept


def select_free(dd, thr, cap):
    """Indices kept without NMS: top-k cap, then threshold, in score order."""
    cand = np.nonzero((dd >= thr) & cap_mask(dd, cap))[0]
    return [int(i) for i in cand[np.argsort(-dd[cand], kind="stable")]]


# ----------------------------------------------------------------------------
# official CULane matching
# ----------------------------------------------------------------------------
class _AttrDict(dict):
    __getattr__ = dict.__getitem__


class PredDecoder:
    """CPU copy of what CLRerHead.predictions_to_lanes reads, to decode single
    predictions in worker processes exactly as get_lanes does."""

    def __init__(self, head, y_step=2):
        self.prior_ys = head.prior_ys.detach().cpu()
        self.n_strips = head.n_strips
        self.test_cfg = _AttrDict(dict(head.test_cfg))
        self.extend_bottom = bool(self.test_cfg.get("extend_bottom", True))
        self.y_step = y_step
        self._decode = type(head).predictions_to_lanes

    def points(self, row):
        """One prediction row -> its lane as CULaneMetric writes it, [(x, y), ...]
        in 1640x590 pixels, or None when the metric would not write it."""
        r = torch.as_tensor(row)[None]
        lanes = self._decode(self, r[:, 6:], r[:, 2:5], torch.round(r[:, 5:6] * self.n_strips),
                             torch.zeros(1), True, self.extend_bottom)
        if not lanes:
            return None
        ys = np.arange(0, ORI_H, self.y_step) / ORI_H
        xs = lanes[0](ys)
        valid = (xs >= 0) & (xs < 1)
        lx = (xs[valid] * ORI_W)[::-1]
        ly = (ys[valid] * ORI_H)[::-1]
        if len(lx) < 2:
            return None
        # the values the metric reads back from its "{:.5f}" text
        return [(float(f"{x:.5f}"), float(f"{y:.5f}")) for x, y in zip(lx.tolist(), ly.tolist())]


def lane_mask(points):
    """The official discrete_cross_iou drawing (one channel instead of three: the
    channels are identical, so the IoU is the same). (y0, y1, x0, x1, crop)."""
    from libs.utils.lane_utils import interp
    from libs.utils.visualizer import draw_lane
    m = draw_lane(interp(points, n=5), img_shape=(ORI_H, ORI_W), width=IOU_WIDTH) > 0
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    if len(ys) == 0:
        return (0, 0, 0, 0, m[:0, :0])
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    return (y0, y1, x0, x1, m[y0:y1, x0:x1])


def mask_iou(a, b):
    ay0, ay1, ax0, ax1, am = a
    by0, by1, bx0, bx1, bm = b
    y0, y1, x0, x1 = min(ay0, by0), max(ay1, by1), min(ax0, bx0), max(ax1, bx1)
    ua = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    ub = np.zeros_like(ua)
    ua[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = am
    ub[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = bm
    union = (ua | ub).sum()
    return float((ua & ub).sum()) / float(union) if union else 0.0


def match(pred_ids, masks, gt_masks):
    """Official culane_metric matching. Returns ({gt: pred id}, IoU matrix)."""
    ious = np.zeros((len(pred_ids), len(gt_masks)))
    for i, p in enumerate(pred_ids):
        for j, g in enumerate(gt_masks):
            ious[i, j] = mask_iou(masks[p], g)
    if len(pred_ids) == 0 or len(gt_masks) == 0:
        return {}, ious
    row, col = linear_sum_assignment(1 - ious)
    return {int(c): pred_ids[r] for r, c in zip(row, col) if ious[r, c] > IOU_THR}, ious


# ----------------------------------------------------------------------------
# per-image analysis
# ----------------------------------------------------------------------------
_W = {}


def _init_worker(state):
    torch.set_num_threads(1)
    _W.update(state)


def load_gt(data_root, name):
    from libs.datasets.metrics.culane_metric import load_culane_img_data
    return load_culane_img_data(osp.join(data_root, name.lstrip("/").replace(".jpg", ".lines.txt")))


def analyse_image(rec):
    w = _W
    gt = load_gt(w["data_root"], rec["name"])
    rows, o2m, dd = rec["rows"], rec["o2m"], rec["dd"]
    near = near_matrix(rows, w["n_strips"], w["x_scale"], w["nms_thres"])
    sel_a = select_nms(dd, near, w["thr_nms"], w["top_k"], cap=w["cap_nms"])
    sel_b = select_free(dd, w["thr_free"], w["cap_free"])
    sel_b0 = select_free(dd, w["thr_nms"], w["cap_free"])   # no NMS at the NMS threshold

    masks, bad = {}, 0
    for i in set(sel_a) | set(sel_b) | set(sel_b0):
        pts = w["decoder"].points(rows[i])
        if pts is None:
            continue
        try:
            masks[i] = lane_mask(pts)
        except Exception:          # degenerate spline: the official metric would raise
            bad += 1
    gt_masks = [lane_mask(lane) for lane in gt]
    out_a = [i for i in sel_a if i in masks]
    out_b = [i for i in sel_b if i in masks]
    out_b0 = [i for i in sel_b0 if i in masks]
    hit_a, _ = match(out_a, masks, gt_masks)
    hit_b, iou_b = match(out_b, masks, gt_masks)
    hit_b0, _ = match(out_b0, masks, gt_masks)

    n_gt = len(gt)
    counts = dict(
        gt=n_gt, bad=bad,
        tp_a=len(hit_a), fp_a=len(out_a) - len(hit_a),
        tp_b=len(hit_b), fp_b=len(out_b) - len(hit_b),
        tp_b0=len(hit_b0), fp_b0=len(out_b0) - len(hit_b0))
    # false positives of the no-NMS output at the NMS threshold that are copies of
    # one of its true positives: what the higher no-NMS threshold has to remove
    tps0 = set(hit_b0.values())
    counts["dup_fp_b0"] = sum(1 for i in out_b0 if i not in tps0 and any(near[i, j] for j in tps0))

    # separability: true positives of the NMS output vs their copies
    tp_ids = set(hit_a.values())
    dup_ids = {int(j) for q in tp_ids for j in np.nonzero(near[q])[0]} - tp_ids
    sep = dict(true_o2m=[float(o2m[i]) for i in tp_ids], true_dd=[float(dd[i]) for i in tp_ids],
               dup_o2m=[float(o2m[j]) for j in dup_ids], dup_dd=[float(dd[j]) for j in dup_ids])

    def cluster(q):
        return [int(j) for j in np.nonzero(near[q])[0]]

    def stats(q):
        cp = cluster(q)
        best = max((o2m[j] for j in cp), default=float("nan"))
        return dict(o2m_q=float(o2m[q]), dd_q=float(dd[q]), n_copies=len(cp),
                    o2m_gap=float(o2m[q] - best) if cp else float("nan"),
                    q_top=bool(not cp or o2m[q] >= best))

    lanes = []
    in_b = set(sel_b)
    for g in range(n_gt):
        a, b = g in hit_a, g in hit_b
        if a and b:
            lanes.append(dict(outcome="kept", **stats(hit_a[g])))
        elif not a and not b:
            lanes.append(dict(outcome="missed"))
        elif b:
            p = hit_b[g]
            lanes.append(dict(outcome="gained",
                              nms_suppressed=bool(dd[p] >= w["thr_nms"] and
                                                  any(near[p, j] for j in sel_a))))
        else:
            q = hit_a[g]
            row = dict(outcome="lost", image=rec["name"], gt=g, **stats(q))
            cp = cluster(q)
            kept_copy = None
            if q in in_b:
                reason = "reassigned"
            elif dd[q] >= w["thr_free"]:
                reason = "evicted"
            elif o2m[q] < w["thr_free"]:
                reason = "o2m_below"
            elif not cp:
                reason = "isolated_lowered"
            else:
                kept = [j for j in cp if j in in_b]
                if kept:
                    reason = "copy_kept"
                    kept_copy = max(kept, key=lambda j: dd[j])
                elif all(dd[j] < w["thr_free"] for j in cp):
                    reason = "cluster_lowered"
                else:
                    reason = "copy_evicted"
            row["reason"] = reason
            if kept_copy is not None:
                row["copy_o2m"] = float(o2m[kept_copy])
                row["copy_dd"] = float(dd[kept_copy])
                row["copy_iou"] = (float(iou_b[out_b.index(kept_copy), g])
                                   if kept_copy in out_b else float("nan"))
                row["dedup_reversed_o2m"] = bool(o2m[q] > o2m[kept_copy])
            lanes.append(row)
    return rec["name"], counts, lanes, sep


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _med(values):
    v = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return "   -              "
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:6.3f} [{q1:.3f}, {q3:.3f}]"


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(pos == neg) over all pairs."""
    pos, neg = np.sort(np.asarray(pos, float)), np.sort(np.asarray(neg, float))
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def report(results, cats, args, info):
    tot = Counter()
    lanes, by_cat_gt, by_cat_lost = [], Counter(), Counter()
    sep = defaultdict(list)
    for _, _, _, s in results:
        for k, v in s.items():
            sep[k] += v
    for name, counts, img_lanes, _ in results:
        tot.update(counts)
        cat = cats.get(name.lstrip("/"), "-") if cats else "-"
        by_cat_gt[cat] += counts["gt"]
        for ln in img_lanes:
            ln["category"] = cat
            if ln["outcome"] == "lost":
                by_cat_lost[cat] += 1
        lanes += img_lanes
    out = []
    say = out.append
    say("=" * 78)
    say(f"learned NMS recall diagnosis  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for k, v in info.items():
        say(f"  {k}: {v}")
    say(f"images {len(results)}, GT lanes {tot['gt']}" +
        (f", {tot['bad']} degenerate predicted lanes skipped" if tot["bad"] else ""))
    say("")
    say(f"{'output':34s} {'TP':>7} {'FP':>7} {'FN':>7} {'P':>8} {'R':>8} {'F1':>8}")
    for label, k, thr in (("with NMS", "a", args.thr_nms), ("no NMS", "b", args.thr_free),
                          ("no NMS at the NMS threshold", "b0", args.thr_nms)):
        tp, fp = tot[f"tp_{k}"], tot[f"fp_{k}"]
        fn = tot["gt"] - tp
        p, r, f = prf(tp, fp, fn)
        say(f"{label + f' @ {thr:.2f}':34s} {tp:7d} {fp:7d} {fn:7d} {p:8.4f} {r:8.4f} {f * 100:8.2f}")
    say(f"  of the {tot['fp_b0']} false positives without NMS at {args.thr_nms:.2f}, "
        f"{tot['dup_fp_b0']} are copies (within the NMS distance) of a true positive")
    oc = Counter(ln["outcome"] for ln in lanes)
    say("")
    say(f"GT lanes: found by both {oc['kept']}, lost without NMS {oc['lost']}, "
        f"gained without NMS {oc['gained']}, missed by both {oc['missed']}")
    gained = [ln for ln in lanes if ln["outcome"] == "gained"]
    if gained:
        say(f"  gained lanes that NMS suppressed: {sum(ln['nms_suppressed'] for ln in gained)}")
    lost = [ln for ln in lanes if ln["outcome"] == "lost"]
    kept = [ln for ln in lanes if ln["outcome"] == "kept"]
    rc = Counter(ln["reason"] for ln in lost)
    say("")
    say(f"why the {len(lost)} lost lanes were lost (q = the prediction NMS found them with):")
    for r in REASONS:
        say(f"  {r:17s} {rc[r]:6d} {100 * rc[r] / max(len(lost), 1):6.1f}%   {READING[r]}")
    ck = [ln for ln in lost if ln["reason"] == "copy_kept"]
    if ck:
        say(f"  copy_kept: IoU of the kept copy with the GT {_med([ln['copy_iou'] for ln in ck])}, "
            f"q had the higher O2M score in {sum(ln['dedup_reversed_o2m'] for ln in ck)} of {len(ck)}")
    say("")
    say(f"{'median [IQR]':34s} {'lost':>24} {'found by both':>24}")
    for label, key in (("copies of q within the NMS distance", "n_copies"),
                       ("O2M score of q", "o2m_q"),
                       ("dedup score of q", "dd_q"),
                       ("O2M score of q - best copy", "o2m_gap")):
        say(f"{label:34s} {_med([ln[key] for ln in lost]):>24} {_med([ln[key] for ln in kept]):>24}")
    for label, sub in (("lost", lost), ("found by both", kept)):
        if sub:
            say(f"  {label}: q has a copy in {100 * np.mean([ln['n_copies'] > 0 for ln in sub]):.1f}%, "
                f"q is the O2M top copy in {100 * np.mean([ln['q_top'] for ln in sub]):.1f}%")
    say("")
    say(f"separability, true lanes ({len(sep['true_dd'])}) vs their duplicates ({len(sep['dup_dd'])}):")
    say(f"  {'score':6s} {'AUC':>7}   {'true p10 / p25 / p50':>24}   {'duplicate p50 / p75 / p90':>27}")
    for label, key in (("O2M", "o2m"), ("dedup", "dd")):
        t, d = np.asarray(sep[f"true_{key}"]), np.asarray(sep[f"dup_{key}"])
        tq = np.percentile(t, [10, 25, 50]) if len(t) else [float("nan")] * 3
        dq = np.percentile(d, [50, 75, 90]) if len(d) else [float("nan")] * 3
        say(f"  {label:6s} {auc(t, d):7.4f}   {' / '.join(f'{v:.3f}' for v in tq):>24}   "
            f"{' / '.join(f'{v:.3f}' for v in dq):>27}")
    for label, key in (("O2M", "o2m"), ("dedup", "dd")):
        t, d = np.asarray(sep[f"true_{key}"]), np.asarray(sep[f"dup_{key}"])
        if len(t) and len(d):
            say(f"  {label}: at {args.thr_free:.2f}, {100 * np.mean(t < args.thr_free):.1f}% of true lanes "
                f"are below and {100 * np.mean(d >= args.thr_free):.1f}% of duplicates above; at "
                f"{args.thr_nms:.2f}, {100 * np.mean(t < args.thr_nms):.1f}% and "
                f"{100 * np.mean(d >= args.thr_nms):.1f}%")
    if cats:
        say("")
        say("by category: lost / GT lanes")
        for cat in sorted(by_cat_gt):
            say(f"  {cat:14s} {by_cat_lost[cat]:6d} / {by_cat_gt[cat]:6d}  "
                f"({100 * by_cat_lost[cat] / max(by_cat_gt[cat], 1):5.2f}%)")
    if lost:
        top = rc.most_common(1)[0][0]
        say("")
        say(f"largest reason: {top} ({100 * rc[top] / len(lost):.0f}% of lost lanes): {READING[top]}.")
    return "\n".join(out), lost


# ----------------------------------------------------------------------------
def build_model(cfg, checkpoint, device):
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS
    init_default_scope(cfg.get("default_scope", "mmdet"))
    model = MODELS.build(cfg.model)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location="cpu")
    return model.to(device).eval()


def check_head(head):
    if getattr(head, "dedup", None) is None:
        raise SystemExit("this config has no bbox_head.dedup_cfg")
    if getattr(head, "diffusion", None) is not None:
        raise SystemExit("diffusion_cfg is not supported")


def caps(head, test_cfg):
    """(cap with NMS, cap without NMS) of CLRBezierHead._dedup_scores."""
    k = head.dedup_test_topk
    return (int(k) if k is not None else None,
            int(k) if k is not None else int(test_cfg["nms_topk"]))


@torch.no_grad()
def verify(head, last, roi, recs, args, near_fn):
    """Kept dedup scores of the head's own get_lanes == the rebuilt selections."""
    cfg = head.test_cfg
    saved = {k: cfg[k] for k in ("conf_threshold", "use_nms")}
    bad = 0
    try:
        for use_nms, thr in ((True, args.thr_nms), (False, args.thr_free)):
            cfg["conf_threshold"], cfg["use_nms"] = thr, use_nms
            pd = head.to_official_pred_dict(head._dedup_scores(last, roi))
            _, scores = head.get_lanes(pd, as_lanes=False, extend_bottom=cfg.get("extend_bottom", True))
            for b, rec in enumerate(recs):
                if use_nms:
                    near = near_fn(rec["rows"])
                    sel = select_nms(rec["dd"], near, thr, int(cfg["nms_topk"]), cap=args._cap_nms)
                else:
                    sel = select_free(rec["dd"], thr, args._cap_free)
                ref = np.sort(np.asarray([float(s) for s in scores[b]], dtype=np.float64))
                got = np.sort(rec["dd"][sel].astype(np.float64))
                if len(ref) != len(got) or not np.allclose(ref, got, rtol=0, atol=1e-6):
                    bad += 1
    finally:
        cfg["conf_threshold"], cfg["use_nms"] = saved["conf_threshold"], saved["use_nms"]
    return bad


@torch.no_grad()
def run_inference(cfg, args, device):
    from mmengine.runner import Runner
    model = build_model(cfg, args.checkpoint, device)
    head = model.bbox_head
    check_head(head)
    tc = head.test_cfg
    args._cap_nms, args._cap_free = caps(head, tc)
    near_fn = lambda rows: near_matrix(rows, head.n_strips, float(head.img_w - 1), float(tc["nms_thres"]))
    loader_cfg = copy.deepcopy(cfg.test_dataloader if args.split == "test" else cfg.val_dataloader)
    loader = Runner.build_dataloader(loader_cfg)
    records, seen, verified, mismatched = [], 0, 0, 0
    tic = time.time()
    for bi, data in enumerate(loader):
        data = model.data_preprocessor(data, False)
        last, roi, o2m, dd = head_outputs(head, model.extract_feat(data["inputs"]))
        names = [s.metainfo["sub_img_name"] for s in data["data_samples"]]
        recs = image_records(last, o2m, dd, names, args.floor)
        if bi < args.verify_batches:
            mismatched += verify(head, last, roi, recs, args, near_fn)
            verified += len(recs)
        records += recs
        seen += len(recs)
        if seen % 2000 < len(recs):
            print(f"  {seen} images ({time.time() - tic:.0f} s)", flush=True)
        if args.max_images is not None and seen >= args.max_images:
            break
    print(f"verified {verified} images against get_lanes on both paths: "
          f"{mismatched} image-path mismatches", flush=True)
    return model, records, dict(verified_images=verified, verify_mismatches=mismatched)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--thr-nms", type=float, required=True, help="best threshold with NMS")
    ap.add_argument("--thr-free", type=float, required=True, help="best threshold without NMS")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--floor", type=float, default=0.10,
                    help="store predictions with O2M or dedup score >= this (thresholds must be >= it)")
    ap.add_argument("--save-dump", default=None)
    ap.add_argument("--load-dump", default=None, help="reuse a saved pass, skip inference")
    ap.add_argument("--out-dir", default=None, help="default: work_dirs/dedup_recall_diag/<config name>")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--verify-batches", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=None, help="debug only; P/R are then partial")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    tc = cfg.model.test_cfg
    dump_meta = None
    if args.load_dump:
        with open(args.load_dump, "rb") as f:
            dump = pickle.load(f)
        records, dump_meta = dump["records"], dump["meta"]
        args.floor = dump_meta["floor"]
    if min(args.thr_nms, args.thr_free) < args.floor:
        raise SystemExit(f"thresholds must be >= the storage floor {args.floor}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.load_dump:
        model = build_model(cfg, None, torch.device("cpu"))
        check_head(model.bbox_head)
        args._cap_nms, args._cap_free = caps(model.bbox_head, tc)
        vinfo = dict(verify="(from the dump) " + json.dumps(dump_meta.get("verify", {})))
    else:
        if not args.checkpoint:
            raise SystemExit("a checkpoint is needed unless --load-dump is given")
        model, records, vinfo = run_inference(cfg, args, device)
        if args.save_dump:
            os.makedirs(osp.dirname(osp.abspath(args.save_dump)), exist_ok=True)
            with open(args.save_dump, "wb") as f:
                pickle.dump(dict(records=records, meta=dict(
                    config=args.config, checkpoint=args.checkpoint, split=args.split,
                    floor=args.floor, verify=vinfo)), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved {args.save_dump}")
    head = model.bbox_head

    evaluator = cfg.test_evaluator if args.split == "test" else cfg.val_evaluator
    data_root = evaluator["data_root"]
    cats = {}
    cat_dir = osp.join(data_root, "list/test_split")
    if args.split == "test" and osp.isdir(cat_dir):
        from libs.datasets.metrics.culane_metric import load_categories
        data_cats, _ = load_categories(cat_dir)
        cats = {k.lstrip("/"): v for k, v in data_cats.items()}

    state = dict(data_root=data_root, decoder=PredDecoder(head, evaluator.get("y_step", 2)),
                 n_strips=head.n_strips, x_scale=float(head.img_w - 1),
                 nms_thres=float(tc["nms_thres"]), top_k=int(tc["nms_topk"]),
                 thr_nms=args.thr_nms, thr_free=args.thr_free,
                 cap_nms=args._cap_nms, cap_free=args._cap_free)
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    tic = time.time()
    if jobs == 1:
        _init_worker(state)
        results = [analyse_image(r) for r in records]
    else:
        with mp.get_context("fork").Pool(jobs, initializer=_init_worker, initargs=(state,)) as pool:
            results = list(pool.imap(analyse_image, records, chunksize=64))
    print(f"analysed {len(results)} images in {time.time() - tic:.0f} s")

    info = dict(config=args.config, checkpoint=(dump_meta or {}).get("checkpoint", args.checkpoint),
                split=args.split, nms_thres=state["nms_thres"], nms_topk=state["top_k"],
                cap_without_nms=args._cap_free, floor=args.floor, **vinfo)
    text, lost = report(results, cats, args, info)
    print(text)
    out_dir = args.out_dir or osp.join("work_dirs", "dedup_recall_diag",
                                       osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "summary.txt"), "a") as f:
        f.write(text + "\n")
    keys = ["image", "gt", "category", "reason", "o2m_q", "dd_q", "n_copies", "o2m_gap", "q_top",
            "copy_o2m", "copy_dd", "copy_iou", "dedup_reversed_o2m"]
    with open(osp.join(out_dir, f"lost_lanes_{args.thr_nms:.2f}_{args.thr_free:.2f}.csv"), "w",
              newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for ln in lost:
            wr.writerow({k: ln.get(k, "") for k in keys})
    print(f"wrote {out_dir}/summary.txt and the lost-lane CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
