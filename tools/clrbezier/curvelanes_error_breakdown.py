"""Where does a CurveLanes lane detector lose its F1?

One inference pass on CurveLanes valid stores, for every image, the 64
highest-scored last-stage predictions (--max-per-image; the head's output before
threshold and NMS, low scores included, so a lane the head found but scored low
is told apart from a lane it never proposed). The evaluation is then rebuilt as
the test loop runs it: threshold, lane NMS as the CUDA kernel computes it (or
none), predictions_to_lanes, CurvelanesMetric and vega's LaneMetricCore (lanes
drawn 5 px wide on 224x224, Hungarian on 1 - IoU). Works for CLRBezierHead, the
official CLRerHead (line anchors) and the 2D heads (BLRHead, HBLRHead, FovealLaneHead
with its 2x configs): they all emit the official rows. For the BLR-family heads the
stored scores are the ones the config's test path uses (o2o when use_nms=False, the
header says which), and --mode defaults to that path. For FovealLaneHead every
stored prediction also carries the view its query's proposal came from (context or
the second view: window / full / polar), reported per bin as hit-v2 / low-v2.

The report has:

1. a threshold sweep with NMS and without NMS (P, R, F1@50, F1@75). The row at
   the config threshold must equal the validation log;
2. at the operating threshold (best F1@50 of the sweep, or --thr-nms /
   --thr-free), why each missed GT lane was missed:
     absent        no stored prediction has IoU > 0.3 with it, even at a low
                   score: the head never proposed the lane
     mislocalized  the best stored prediction has IoU in (0.3, 0.5]: proposed,
                   not localized well enough
     low_score     a stored prediction has IoU > 0.5, but every such one scored
                   below the threshold: a classification miss
     suppressed    one scored above the threshold and NMS removed it
     capped        one scored above the threshold and the nms_topk cap dropped it
     taken         one is in the output, but the Hungarian gave it to another
                   GT lane
   and what each false positive is:
     duplicate     IoU > 0.5 with a GT lane that another prediction took
     mislocalized  best IoU with any GT lane in (0.3, 0.5]
     ghost         best IoU <= 0.3: nothing annotated there
3. F1@50 if one error class were fixed alone (oracle bounds);
4. why hits at IoU 0.5 miss IoU 0.75: row coverage, rows beyond the annotation,
   and x error of the matched prediction against the annotated rows;
5. recall@50, recall@75 and the miss reasons by GT lane attribute: slant,
   vertical extent, bend, distance to the nearest other GT lane, forks / merges,
   lanes per image, resolution, y-reversing lanes, lanes reaching above the crop;
   false positives by predicted slant and extent.

GT attributes are measured in the network input (800x320 for the CurveLanes
configs). The rows of a GT lane are its training target (the config's
LaneAlbumentation cut_unsorted and PackCLRNetInputs lane_interp, as
repr_ceiling.py builds them); "annotated rows" leave out the rows the target
extends below the lowest annotated point.

    # one inference pass (GPU); the dump lets later runs skip inference
    python tools/clrbezier/curvelanes_error_breakdown.py CONFIG CKPT \\
        --save-dump work_dirs/err/RUN.pkl --out-dir work_dirs/err/RUN
    # other modes / thresholds from the dump (CPU)
    python tools/clrbezier/curvelanes_error_breakdown.py CONFIG --load-dump work_dirs/err/RUN.pkl \\
        --mode both --out-dir work_dirs/err/RUN
    # side by side (e.g. A-mono and the foveal runs off / window / polar; per bin R@50, R@75, low_score)
    python tools/clrbezier/curvelanes_error_breakdown.py --compare A/summary.json B/summary.json

Outputs in --out-dir: summary.txt, summary.json (for --compare), gt_lanes.csv
(one row per GT lane and mode), false_positives.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import pickle
import sys
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_HERE = osp.dirname(osp.abspath(__file__))
_REPO = osp.dirname(osp.dirname(_HERE))
if _REPO not in sys.path:      # libs.* for the head class when run as tools/clrbezier/...
    sys.path.append(_REPO)


def _load_ceiling():
    """repr_ceiling.py next to this file: the metric pieces and the training-target path."""
    if "repr_ceiling" in sys.modules:
        return sys.modules["repr_ceiling"]
    spec = importlib.util.spec_from_file_location("repr_ceiling", osp.join(_HERE, "repr_ceiling.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["repr_ceiling"] = mod
    spec.loader.exec_module(mod)
    return mod


rc = _load_ceiling()

BELOW, KEPT, SUPPRESSED, CAPPED = 0, 1, 2, 3
MODES = ("nms", "free")
MODE_LABEL = {"nms": "with NMS", "free": "without NMS"}
HIT, TIGHT, NEAR = 0.5, 0.75, 0.3
FN_REASONS = ["absent", "mislocalized", "low_score", "suppressed", "capped", "taken"]
FN_SHORT = {"absent": "absent", "mislocalized": "misloc", "low_score": "low_sc",
            "suppressed": "suppr", "capped": "capped", "taken": "taken"}
FN_READING = {
    "absent": "no stored prediction has IoU > 0.3 with the lane, even at a low score",
    "mislocalized": "the best stored prediction has IoU in (0.3, 0.5]",
    "low_score": "a prediction with IoU > 0.5 scored below the threshold",
    "suppressed": "NMS removed a prediction with IoU > 0.5",
    "capped": "the nms_topk cap dropped a prediction with IoU > 0.5",
    "taken": "a prediction with IoU > 0.5 is in the output, matched to another GT lane",
}
FP_KINDS = ["duplicate", "mislocalized", "ghost"]
FP_READING = {
    "duplicate": "IoU > 0.5 with a GT lane another prediction took",
    "mislocalized": "best IoU with any GT lane in (0.3, 0.5]",
    "ghost": "best IoU <= 0.3: nothing annotated there",
}
FORK_PX = 10.0            # two GT lanes this close at their lowest common annotated row
ABOVE_CROP_FRAC = 0.05    # share of the lane length above the crop line
EXTENT_COVERAGE, EXTENT_OVERSHOOT = 0.85, 0.15
INF = math.inf


def gt_attr_specs(nms_thres):
    t = float(nms_thres)
    return [
        ("slant", "slant: sum|dx| / sum|dy| along the lane, network px (1 = 45 deg)",
         [(0, 1, "<1"), (1, 2, "1-2"), (2, 4, "2-4"), (4, INF, ">4")]),
        ("extent", "vertical extent inside the crop / crop height",
         [(0, .2, "<0.2"), (.2, .4, "0.2-0.4"), (.4, .7, "0.4-0.7"), (.7, INF, ">=0.7")]),
        ("bend", "bend: largest distance of the lane from its chord inside the crop, network px",
         [(0, 3, "<3"), (3, 10, "3-10"), (10, 30, "10-30"), (30, INF, ">=30")]),
        ("nn_dist", f"nearest other GT lane: mean |dx| over the common rows of the two training "
                    f"targets, network px, as the NMS kernel measures it (< {t:g}: NMS keeps only one "
                    f"of two perfect predictions)",
         [(0, t, f"<{t:g}"), (t, 2 * t, f"{t:g}-{2 * t:g}"), (2 * t, 4 * t, f"{2 * t:g}-{4 * t:g}"),
          (4 * t, INF, f">={4 * t:g}/none")]),
        ("fork", f"fork / merge: another GT lane within {FORK_PX:g} px at their lowest common "
                 f"annotated row", "bool"),
        ("lanes", "GT lanes in the image", [(1, 4, "1-3"), (4, 7, "4-6"), (7, 10, "7-9"), (10, INF, ">=10")]),
        ("resolution", "original image size (w x h)", "cat"),
        ("reversing", "y-reversing annotation (x(y) cannot hold it)", "bool"),
        ("above_crop", f"more than {100 * ABOVE_CROP_FRAC:g}% of the lane length above the crop line "
                       f"(the network never sees it)", "bool"),
    ]


PRED_ATTR_SPECS = [
    ("slant", "predicted slant (visible part, network px)",
     [(0, 1, "<1"), (1, 2, "1-2"), (2, 4, "2-4"), (4, INF, ">4")]),
    ("extent", "predicted vertical extent / crop height",
     [(0, .2, "<0.2"), (.2, .4, "0.2-0.4"), (.4, .7, "0.4-0.7"), (.7, INF, ">=0.7")]),
]


def bin_label(spec, v):
    if spec == "bool":
        return "yes" if v else "no"
    if spec == "cat":
        return str(v)
    v = float(v)
    if math.isnan(v):
        v = 0.0
    for lo, hi, lab in spec:
        if lo <= v < hi:
            return lab
    return spec[-1][2] if v >= spec[-1][0] else spec[0][2]


def bin_order(spec, labels):
    if spec == "bool":
        return [b for b in ("yes", "no") if b in labels]
    if spec == "cat":
        return sorted(labels, key=lambda b: -labels[b])
    return [lab for _, _, lab in spec]


# ----------------------------------------------------------------------------
# NMS exactly as libs/models/layers/nms/src/nms_kernel.cu on get_lanes' boxes
# ----------------------------------------------------------------------------
def nms_boxes(rows, n_strips, img_w):
    """The [n, 5 + R] float32 boxes get_lanes hands to nms(): cls logits,
    1 - start_y, start_x, length * n_strips, xs * (img_w - 1). (get_lanes takes
    the logit columns from image 0 of the batch; the kernel reads them only for a
    start row below -0.5 / n_strips, so they are taken from the image itself.)"""
    f32 = np.float32
    r = np.asarray(rows, dtype=f32)
    box = np.empty((len(r), 5 + r.shape[1] - 6), dtype=f32)
    box[:, 0:2] = r[:, 0:2]
    box[:, 2] = f32(1.0) - r[:, 2]
    box[:, 3] = r[:, 3]
    box[:, 4] = r[:, 5] * f32(n_strips)
    box[:, 5:] = r[:, 6:] * f32(img_w - 1)
    return box


def kernel_pairs(rows, n_strips, img_w, nms_thres):
    """devIoU for every pair: ([n, n] bool "would suppress", [n, n] mean |dx| over the
    common rows (inf without common rows)). float32 where the kernel is float32."""
    n = len(rows)
    if n == 0:
        return np.zeros((0, 0), bool), np.zeros((0, 0), np.float32)
    f32 = np.float32
    box = nms_boxes(rows, n_strips, img_w)
    a2, a4 = box[:, 2], box[:, 4]
    start = np.trunc((a2 * f32(n_strips)).astype(np.float64) + 0.5).astype(np.int64)
    end = np.trunc(((start.astype(f32) + a4) - f32(1)).astype(np.float64) + 0.5
                   - ((a4 - f32(1)) < 0).astype(np.float64)).astype(np.int64)
    s = np.maximum(start[:, None], start[None, :])
    e = np.minimum(np.minimum(end[:, None], end[None, :]), n_strips)
    overlap = e >= s
    lo = 5 + s                                  # the loop starts at box column 5 + start
    cols = np.arange(box.shape[1])
    inside = (cols[None, None, :] >= lo[:, :, None]) & (cols[None, None, :] <= (5 + e)[:, :, None])
    inside &= (lo >= 0)[:, :, None]             # unsigned char index: 5 + start < 0 never loops
    diff = np.abs(box[:, None, :] - box[None, :, :])
    dist = np.cumsum(np.where(inside, diff, f32(0)), axis=2, dtype=f32)[:, :, -1]
    width = (e - s + 1).astype(f32)
    near = overlap & (dist < f32(nms_thres) * width)
    np.fill_diagonal(near, False)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(overlap & (width > 0), dist / np.maximum(width, f32(1)), f32(np.inf))
    return near, mean.astype(np.float32)


def select(score, near, thr, mode, top_k):
    """What get_lanes keeps at this threshold. Returns (kept indices in kernel order,
    fate per prediction (BELOW / KEPT / SUPPRESSED / CAPPED), suppressor per prediction)."""
    n = len(score)
    fate = np.full(n, BELOW, dtype=np.int8)
    by = np.full(n, -1, dtype=np.int64)
    cand = np.nonzero(score >= thr)[0]
    cand = cand[np.argsort(-score[cand], kind="stable")]
    if mode == "free":
        fate[cand] = KEPT
        return [int(i) for i in cand], fate, by
    removed = np.zeros(n, dtype=bool)
    kept = []
    for pos, i in enumerate(cand):
        if removed[i]:
            continue
        kept.append(int(i))
        fate[i] = KEPT
        rest = cand[pos + 1:]
        hit = rest[near[i, rest] & ~removed[rest]]
        removed[hit] = True
        fate[hit] = SUPPRESSED
        by[hit] = i
        if len(kept) == top_k:
            left = rest[~removed[rest]]
            fate[left] = CAPPED
            break
    return kept, fate, by


def output_order(kept, valid, mode, idx):
    """The lanes CurvelanesMetric receives, in get_lanes' order (NMS: score order;
    no NMS: prior order); lanes the decoding or the metric drops are left out."""
    out = [i for i in kept if valid[i]]
    if mode == "free":
        out.sort(key=lambda i: idx[i])
    return out


def assignment(iou, out):
    """vega evaluate_core: Hungarian on 1 - IoU over (GT, output). {gt: (pred, IoU)}."""
    if iou.shape[0] == 0 or not out:
        return {}
    sub = iou[:, out]
    rows, cols = linear_sum_assignment(1 - sub)
    return {int(g): (out[c], float(sub[g, c])) for g, c in zip(rows, cols)}


# ----------------------------------------------------------------------------
# decoding: the head's own predictions_to_lanes, then CurvelanesMetric
# ----------------------------------------------------------------------------
class _AttrDict(dict):
    __getattr__ = dict.__getitem__


class Decoder:
    """Calls the head class's predictions_to_lanes on one prediction, as get_lanes
    does (lengths rounded by torch.round first)."""

    def __init__(self, decode_fn, prior_ys, n_strips, test_cfg):
        self.prior_ys = torch.as_tensor(prior_ys, dtype=torch.float32)
        self.n_strips = int(n_strips)
        self.test_cfg = _AttrDict(test_cfg)
        self.extend_bottom = bool(test_cfg.get("extend_bottom", True))
        self._decode = decode_fn

    def points(self, row):
        """[N, 2] (x, y) normalized as predictions_to_lanes returns them, or None."""
        r = torch.as_tensor(np.asarray(row, dtype=np.float32))[None]
        lengths = torch.round(r[:, 5:6] * self.n_strips)
        lanes = self._decode(self, r[:, 6:], r[:, 2:5], lengths, torch.zeros(1), False,
                             self.extend_bottom)
        return lanes[0].cpu().numpy() if lanes else None


def resolve(path):
    mod, _, name = path.partition(":")
    obj = importlib.import_module(mod)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def head_meta(head):
    owner = next(c for c in type(head).__mro__ if "predictions_to_lanes" in c.__dict__)
    tc = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
          for k, v in dict(head.test_cfg).items()}
    return dict(img_w=int(head.img_w), img_h=int(head.img_h), n_strips=int(head.n_strips),
                prior_ys=head.prior_ys.detach().float().cpu().numpy(),
                test_cfg=tc, decode_cls=f"{owner.__module__}:{owner.__qualname__}",
                head_cls=f"{type(head).__module__}:{type(head).__qualname__}")


def make_decoder(meta):
    cls = resolve(meta["decode_cls"])
    return Decoder(cls.predictions_to_lanes, meta["prior_ys"], meta["n_strips"], meta["test_cfg"])


# ----------------------------------------------------------------------------
# geometry in the network input
# ----------------------------------------------------------------------------
def chord_bend(pts):
    if len(pts) < 3:
        return 0.0
    a, b = pts[0], pts[-1]
    d = b - a
    norm = float(np.hypot(*d))
    if norm < 1e-9:
        return float(np.max(np.linalg.norm(pts - a, axis=1)))
    cross = np.abs(d[0] * (pts[:, 1] - a[1]) - d[1] * (pts[:, 0] - a[0])) / norm
    return float(cross.max())


def gt_geometry(gt, ori_hw, offset, st):
    """Per GT lane: attributes (dict) and its training-target rows
    (xs_px [R], first, last, n_ext) or None."""
    ori_h, ori_w = ori_hw
    sx, sy = st["img_w"] / ori_w, st["img_h"] / (ori_h - offset)
    attrs = []
    for ln in gt:
        net = np.array([(x * sx, (y - offset) * sy) for x, y in ln], dtype=np.float64)
        if len(net) >= 2 and net[0, 1] < net[-1, 1]:
            net = net[::-1]
        rev, _, slant = rc.lane_flags(net)
        if len(net) >= 2:
            seg = np.linalg.norm(np.diff(net, axis=0), axis=1)
            mid = (net[1:, 1] + net[:-1, 1]) / 2
            above = float(seg[mid < 0].sum() / seg.sum()) if seg.sum() > 0 else float(net[0, 1] < 0)
            inside = rc.clip_to_crop(net)
        else:
            above = float(net[0, 1] < 0)
            inside = net[net[:, 1] >= 0]
        extent = float((inside[:, 1].max() - inside[:, 1].min()) / st["img_h"]) if len(inside) >= 2 else 0.0
        attrs.append(dict(slant=float(slant), extent=extent, bend=chord_bend(inside),
                          reversing=bool(rev), above_frac=above, above_crop=above > ABOVE_CROP_FRAC,
                          n_points=len(ln), resolution=f"{ori_w}x{ori_h}", lanes=len(gt),
                          nn_dist=INF, fork=False))
    # training targets (CurvelanesDataset >= 2 points, Resize, cut_unsorted, PackCLRNetInputs)
    rows = [None] * len(gt)
    train = [i for i, ln in enumerate(gt) if len(ln) >= 2]
    flat = [[c for x, y in gt[i] for c in (x * sx, (y - offset) * sy)] for i in train]
    if st["cut_unsorted"] and not all(rc.is_sorted(f) for f in flat):
        flat = rc.cut_unsorted_points(flat)
    for i, f in zip(train, flat):
        if len(f) > 2 and rc.is_sorted(f):
            try:
                rows[i] = rc.row_targets(f, st)
            except Exception:
                rows[i] = None
    # nearest other lane (kernel distance on the targets) and forks (annotated rows)
    w = st["img_w"]
    for i in range(len(gt)):
        if rows[i] is None:
            continue
        xi, fi, li, ei = rows[i]
        for j in range(len(gt)):
            if j == i or rows[j] is None:
                continue
            xj, fj, lj, ej = rows[j]
            r = np.arange(max(fi, fj), min(li, lj) + 1)
            r = r[(xi[r] >= 0) & (xi[r] < w) & (xj[r] >= 0) & (xj[r] < w)] if len(r) else r
            if len(r):
                attrs[i]["nn_dist"] = min(attrs[i]["nn_dist"], float(np.abs(xi[r] - xj[r]).mean()))
            ra = r[(r >= ei) & (r >= ej)]
            if len(ra) >= 3 and abs(xi[ra[0]] - xj[ra[0]]) <= FORK_PX:
                attrs[i]["fork"] = True
    return attrs, rows


def annotated_rows(grow, img_w):
    if grow is None:
        return None
    xs, first, last, n_ext = grow
    r = np.arange(max(first, n_ext), last + 1)
    r = r[(xs[r] >= 0) & (xs[r] < img_w)] if len(r) else r
    return (r, xs[r]) if len(r) else None


def pred_geometry(pts, st):
    """Visible part of a decoded prediction (0 <= x < 1): ({row: x_px}, slant, extent)."""
    x, y = pts[:, 0], pts[:, 1]
    vis = (x >= 0) & (x < 1)
    rows = np.rint((1.0 - y[vis]) * st["n_rows_m1"]).astype(int)
    xpx = x[vis] * (st["img_w"] - 1)
    rowmap = dict(zip(rows.tolist(), xpx.tolist()))
    if vis.sum() < 2:
        return rowmap, 0.0, 0.0
    px = np.stack([xpx, y[vis] * st["img_h"]], 1)
    d = np.abs(np.diff(px, axis=0))
    dy = d[:, 1].sum()
    slant = float(d[:, 0].sum() / dy) if dy > 0 else INF
    return rowmap, slant, float(y[vis].max() - y[vis].min())


def pair_metrics(ann, rowmap, st):
    """(row coverage, rows beyond the annotation, mean |dx|, mean normal distance),
    the prediction's visible rows against the GT's annotated rows; NaN if unknown."""
    nan = (math.nan,) * 4
    if ann is None or rowmap is None:
        return nan
    r, xs = ann
    rset = set(r.tolist())
    common = [k for k, row in enumerate(r.tolist()) if row in rowmap]
    cov = len(common) / len(r)
    over = len(set(rowmap) - rset) / len(r)
    if not common:
        return cov, over, math.nan, math.nan
    dx = np.abs(np.array([rowmap[int(r[k])] for k in common]) - xs[common])
    if len(r) >= 2:
        slope = np.gradient(xs.astype(np.float64), r.astype(np.float64)) / (st["img_h"] / st["n_rows_m1"])
    else:
        slope = np.zeros(1)
    perp = dx / np.sqrt(1.0 + slope[common] ** 2)
    return cov, over, float(dx.mean()), float(perp.mean())


# ----------------------------------------------------------------------------
# one image (worker): everything threshold-independent, plus the sweep
# ----------------------------------------------------------------------------
_W = {}


def _init_worker(state):
    torch.set_num_threads(1)
    _W.update(state)


def gt_file(filename):
    return filename.replace(".jpg", ".lines.txt")      # CurvelanesMetric.parse_anno


def analyse_image(rec):
    w = _W
    st, dec = w["st"], w["decoder"]
    ori_hw, offset = tuple(rec["ori_hw"]), int(rec["offset"])
    gt = [ln for ln in rc.read_lines_txt(gt_file(rec["filename"])) if len(ln) > 0]
    attrs, gt_rows = gt_geometry(gt, ori_hw, offset, st)
    gt_masks = [rc.lane_mask([{"x": x, "y": y} for x, y in ln], ori_hw, st["eval"]) for ln in gt]

    rows, score = rec["rows"], rec["score"]
    n = len(score)
    valid = np.zeros(n, dtype=bool)
    pred_attr = np.zeros((n, 2), dtype=np.float64)
    rowmaps, masks = [None] * n, []
    for i in range(n):
        pts = dec.points(rows[i])
        if pts is None:
            continue
        lane = rc.to_metric_points(pts, ori_hw, offset, st["y_step"])
        if len(lane) == 0:                     # vega drops empty lanes
            continue
        valid[i] = True
        masks.append(rc.lane_mask(lane, ori_hw, st["eval"]))
        rowmaps[i], pred_attr[i, 0], pred_attr[i, 1] = pred_geometry(pts, st)
    iou = np.zeros((len(gt), n), dtype=np.float64)
    vi = np.nonzero(valid)[0]
    if len(gt) and len(vi):
        iou[:, vi] = rc.iou_matrix(gt_masks, masks)
    near, _ = kernel_pairs(rows, st["n_rows_m1"], st["img_w"], w["nms_thres"])
    anns = [annotated_rows(r, st["img_w"]) for r in gt_rows]
    pairs = {}
    for g, p in zip(*np.nonzero(iou > NEAR)):
        pairs[(int(g), int(p))] = pair_metrics(anns[g], rowmaps[p], st)

    sweep = {}
    for mode in MODES:
        h50, h75, nout = (np.zeros(len(w["thresholds"]), dtype=np.int64) for _ in range(3))
        for k, thr in enumerate(w["thresholds"]):
            kept, _, _ = select(score, near, thr, mode, w["top_k"])
            out = output_order(kept, valid, mode, rec["idx"])
            m = np.array([v for _, v in assignment(iou, out).values()])
            h50[k], h75[k], nout[k] = int((m > HIT).sum()), int((m > TIGHT).sum()), len(out)
        sweep[mode] = (h50, h75, nout)
    return dict(filename=rec["filename"], n_gt=len(gt), attrs=attrs, iou=iou, valid=valid,
                near=np.packbits(near, axis=1), n=n, pred_attr=pred_attr, pairs=pairs, sweep=sweep)


def unpack_near(c):
    n = c["n"]
    if n == 0:
        return np.zeros((0, 0), bool)
    return np.unpackbits(c["near"], axis=1, count=n).astype(bool)


# ----------------------------------------------------------------------------
# one image at the operating threshold (main process)
# ----------------------------------------------------------------------------
def evaluate_image(c, rec, thr, mode, ctx):
    score, valid, iou = rec["score"], c["valid"], c["iou"]
    view = rec.get("view")
    near = unpack_near(c)
    kept, fate, by = select(score, near, thr, mode, ctx["top_k"])
    out = output_order(kept, valid, mode, rec["idx"])
    matched = assignment(iou, out)
    hit_of = {p: g for g, (p, v) in matched.items() if v > HIT}
    lanes = []
    for g in range(c["n_gt"]):
        p, v = matched.get(g, (-1, 0.0))
        row = dict(image=c["filename"], gt=g, iou=v, hit50=v > HIT, hit75=v > TIGHT,
                   **c["attrs"][g])
        src = -1                                 # the prediction the lane's row describes (view column)
        if v > HIT:
            row["reason"] = "hit"
            row["pair"] = c["pairs"].get((g, p))
            row["score"] = float(score[p])
            src = p
        else:
            cand = np.nonzero(valid & (iou[g] > HIT))[0]
            f = fate[cand]
            if (f == KEPT).any():
                row["reason"] = "taken"
            elif (f == SUPPRESSED).any():
                row["reason"] = "suppressed"
                q = cand[f == SUPPRESSED]
                q = q[np.argmax(score[q])]
                row["suppressor_is_hit"] = int(by[q]) in hit_of
                row["score"] = float(score[q])
                src = q
            elif (f == CAPPED).any():
                row["reason"] = "capped"
            elif len(cand):
                row["reason"] = "low_score"
                src = cand[np.argmax(score[cand])]
                row["score"] = float(score[src])
            else:
                best_p, best = -1, 0.0
                if valid.any():
                    vv = np.where(valid, iou[g], -1.0)
                    best_p = int(np.argmax(vv))
                    best = float(vv[best_p])
                row["best_iou"] = best
                if best > NEAR:
                    row["reason"] = "mislocalized"
                    row["pair"] = c["pairs"].get((g, best_p))
                    row["best_in_output"] = best_p in out
                    row["score"] = float(score[best_p])
                    src = best_p
                else:
                    row["reason"] = "absent"
        if src >= 0:
            row["pred"] = int(rec["idx"][src])
            if view is not None:
                row["view"] = int(view[src])
        lanes.append(row)
    fps = []
    for p in out:
        if p in hit_of:
            continue
        col = iou[:, p]
        g = int(np.argmax(col)) if len(col) else -1
        best = float(col[g]) if len(col) else 0.0
        kind = "duplicate" if best > HIT else "mislocalized" if best > NEAR else "ghost"
        row = dict(image=c["filename"], pred=int(rec["idx"][p]), score=float(score[p]), kind=kind,
                   best_iou=best, slant=float(c["pred_attr"][p, 0]), extent=float(c["pred_attr"][p, 1]),
                   twin_dist=math.nan, twin_near=False)
        if view is not None:
            row["view"] = int(view[p])
        if kind == "duplicate" and g in matched and matched[g][1] > HIT:
            twin = matched[g][0]
            nr, mean = kernel_pairs(rec["rows"][[p, twin]], ctx["n_strips"], ctx["img_w"], ctx["nms_thres"])
            row["twin_dist"], row["twin_near"] = float(mean[0, 1]), bool(nr[0, 1])
        fps.append(row)
    counts = dict(gt=c["n_gt"], out=len(out), tp50=len(hit_of),
                  tp75=sum(1 for v in matched.values() if v[1] > TIGHT))
    return lanes, fps, counts


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def prf(tp, n_out, n_gt):
    eps = sys.float_info.epsilon            # vega LaneMetricCore.summary
    p = tp / (n_out + eps)
    r = tp / (n_gt + eps)
    return p, r, 2 * p * r / (p + r + eps)


def sweep_table(results, thresholds):
    tab = {}
    n_gt = sum(c["n_gt"] for c in results)
    for mode in MODES:
        h50 = sum(c["sweep"][mode][0] for c in results)
        h75 = sum(c["sweep"][mode][1] for c in results)
        nout = sum(c["sweep"][mode][2] for c in results)
        rows = []
        for k, thr in enumerate(thresholds):
            p, r, f = prf(int(h50[k]), int(nout[k]), n_gt)
            f75 = prf(int(h75[k]), int(nout[k]), n_gt)[2]
            rows.append(dict(thr=float(thr), P=p, R=r, F1_50=f, F1_75=f75, out=int(nout[k])))
        tab[mode] = rows
    return tab


def _q(values):
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return "      -          "
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:6.2f} [{q1:.2f}, {q3:.2f}]"


def _v2(items):
    """Share of the items (lane / false-positive rows) whose prediction came from the second view (NaN if none)."""
    v = [it["view"] for it in items if "view" in it]
    return float(np.mean(np.asarray(v) == 1)) if v else math.nan


def breakdown(lanes, fps, counts, mode, thr, specs, second_view=None):
    """Everything the report and summary.json show for one mode. second_view: the FovealLaneHead's second view
    (window / full / polar) when the records carry the view of each prediction, else None."""
    G = sum(k["gt"] for k in counts)
    n_out = sum(k["out"] for k in counts)
    tp = sum(k["tp50"] for k in counts)
    tp75 = sum(k["tp75"] for k in counts)
    p, r, f = prf(tp, n_out, G)
    f75 = prf(tp75, n_out, G)[2]
    rc_ = Counter(ln["reason"] for ln in lanes)
    fk = Counter(fp["kind"] for fp in fps)
    mis_out = sum(1 for ln in lanes if ln["reason"] == "mislocalized" and ln.get("best_in_output"))
    supp = [ln for ln in lanes if ln["reason"] == "suppressed"]

    def f1(tp_, out_):
        return 100 * prf(tp_, out_, G)[2]
    oracles = [
        ("keep the lanes NMS removed (suppressed + capped)",
         f1(tp + rc_["suppressed"] + rc_["capped"], n_out + rc_["suppressed"] + rc_["capped"])),
        ("score every low_score lane above the threshold",
         f1(tp + rc_["low_score"], n_out + rc_["low_score"])),
        ("localize every mislocalized lane",
         f1(tp + rc_["mislocalized"], n_out + rc_["mislocalized"] - mis_out)),
        ("propose every absent lane", f1(tp + rc_["absent"], n_out + rc_["absent"])),
        ("drop every duplicate", f1(tp, n_out - fk["duplicate"])),
        ("drop every mislocalized false positive", f1(tp, n_out - fk["mislocalized"])),
        ("drop every ghost", f1(tp, n_out - fk["ghost"])),
    ]
    hits = [ln for ln in lanes if ln["hit50"] and ln.get("pair")]
    loose = [ln for ln in hits if not ln["hit75"]]
    tight = [ln for ln in hits if ln["hit75"]]

    def ext_err(ln):
        cov, over = ln["pair"][0], ln["pair"][1]
        return (not math.isnan(cov)) and (cov < EXTENT_COVERAGE or over > EXTENT_OVERSHOOT)

    loose_stats = dict(
        n_loose=sum(1 for ln in lanes if ln["hit50"] and not ln["hit75"]), n_hits=tp,
        rows=[(lab, [ln["pair"][k] for ln in loose], [ln["pair"][k] for ln in tight])
              for k, lab in enumerate(["row coverage of the annotation", "rows beyond the annotation",
                                       "mean |dx|, px", "mean normal distance, px"])],
        ext_loose=float(np.mean([ext_err(ln) for ln in loose])) if loose else math.nan,
        ext_tight=float(np.mean([ext_err(ln) for ln in tight])) if tight else math.nan)

    tables = {}
    for name, desc, spec in specs:
        labels = Counter(bin_label(spec, ln[name]) for ln in lanes)
        order = bin_order(spec, labels)
        misses = max(1, sum(1 for ln in lanes if not ln["hit50"]))
        rows = []
        for b in order:
            sub = [ln for ln in lanes if bin_label(spec, ln[name]) == b]
            n = len(sub)
            if n == 0:
                rows.append(dict(bin=b, n=0))
                continue
            rcb = Counter(ln["reason"] for ln in sub)
            rows.append(dict(bin=b, n=n, share=n / max(G, 1),
                             r50=sum(ln["hit50"] for ln in sub) / n, r75=sum(ln["hit75"] for ln in sub) / n,
                             reasons={k: rcb[k] / n for k in FN_REASONS},
                             miss_share=sum(1 for ln in sub if not ln["hit50"]) / misses))
            if second_view:
                rows[-1].update(v2_hits=_v2([ln for ln in sub if ln["reason"] == "hit"]),
                                v2_low=_v2([ln for ln in sub if ln["reason"] == "low_score"]))
        tables[name] = dict(desc=desc, rows=rows)
    fp_tables = {}
    for name, desc, spec in PRED_ATTR_SPECS:
        rows = []
        for b in bin_order(spec, {}):
            sub = [fp for fp in fps if bin_label(spec, fp[name]) == b]
            kinds = Counter(fp["kind"] for fp in sub)
            d = [fp["twin_dist"] for fp in sub if fp["kind"] == "duplicate"]
            rows.append(dict(bin=b, n=len(sub), share=len(sub) / max(len(fps), 1),
                             kinds={k: kinds[k] / max(len(sub), 1) for k in FP_KINDS},
                             twin_dist=_q(d).strip()))
        fp_tables[name] = dict(desc=desc, rows=rows)
    dups = [fp for fp in fps if fp["kind"] == "duplicate" and not math.isnan(fp["twin_dist"])]
    return dict(mode=mode, thr=thr, gt=G, out=n_out, tp50=tp, tp75=tp75, P=p, R=r, F1_50=f, F1_75=f75,
                reasons={k: rc_[k] / max(G, 1) for k in FN_REASONS}, reason_counts=dict(rc_),
                fp={k: fk[k] / max(n_out, 1) for k in FP_KINDS}, fp_counts=dict(fk),
                mis_in_output=mis_out / max(rc_["mislocalized"], 1),
                suppressed_by_hit=(sum(ln.get("suppressor_is_hit", False) for ln in supp) / len(supp))
                if supp else math.nan,
                dup_twin=dict(n=len(dups), median=float(np.median([fp["twin_dist"] for fp in dups]))
                              if dups else math.nan,
                              near=float(np.mean([fp["twin_near"] for fp in dups])) if dups else math.nan),
                oracles=oracles, loose=loose_stats, tables=tables, fp_tables=fp_tables,
                views=(dict(second=second_view, hits=_v2([ln for ln in lanes if ln["reason"] == "hit"]),
                            low=_v2([ln for ln in lanes if ln["reason"] == "low_score"]), fp=_v2(fps))
                       if second_view else None))


def format_mode(b, cfg_thr, chosen, say, nms_thres):
    say("=" * 100)
    say(f"{MODE_LABEL[b['mode']]} at {b['thr']:.2f} ({chosen}; config {cfg_thr:.2f}):  "
        f"P {100 * b['P']:.2f}  R {100 * b['R']:.2f}  F1@50 {100 * b['F1_50']:.2f}  F1@75 {100 * b['F1_75']:.2f}")
    say("-" * 100)
    say(f"missed GT lanes, in points of recall@50 (of {100 * (1 - b['R']):.2f} lost):")
    for k in FN_REASONS:
        if b["mode"] == "free" and k in ("suppressed", "capped"):
            continue
        extra = ""
        if k == "mislocalized" and b["reason_counts"].get(k):
            extra = f"; {100 * b['mis_in_output']:.0f}% of them are in the output (also false positives)"
        if k == "suppressed" and b["reason_counts"].get(k):
            extra = f"; the suppressor is a hit on another GT lane in {100 * b['suppressed_by_hit']:.0f}%"
        say(f"  {k:13s} {100 * b['reasons'][k]:6.2f}   {FN_READING[k]}{extra}")
    say(f"false positives, in % of the output (of {100 * (1 - b['P']):.2f}):")
    for k in FP_KINDS:
        say(f"  {k:13s} {100 * b['fp'][k]:6.2f}   {FP_READING[k]}")
    dt = b["dup_twin"]
    if dt["n"]:
        say(f"  duplicates vs the prediction that took their GT lane: median kernel distance "
            f"{dt['median']:.1f} px (nms_thres {nms_thres:g}); {100 * dt['near']:.0f}% within it")
    vw = b.get("views")
    if vw:
        say(f"second view ({vw['second']}): its queries made {_pct(vw['hits'])}% of the hits, "
            f"{_pct(vw['fp'])}% of the false positives, and the best candidate of {_pct(vw['low'])}% of the "
            f"low_score misses (columns hit-v2 / low-v2 below)")
    say(f"F1@50 if one error class were fixed alone (from {100 * b['F1_50']:.2f}):")
    for lab, v in b["oracles"]:
        say(f"  {lab:48s} {v:6.2f}  ({v - 100 * b['F1_50']:+.2f})")
    ls = b["loose"]
    say(f"hits at IoU 0.5 that miss 0.75: {ls['n_loose']} of {ls['n_hits']} hits "
        f"({100 * ls['n_loose'] / max(ls['n_hits'], 1):.1f}%); median [IQR], 800x320 px:")
    say(f"  {'':34s} {'IoU 0.5-0.75':>22} {'IoU > 0.75':>22}")
    for lab, lo, ti in ls["rows"]:
        say(f"  {lab:34s} {_q(lo):>22} {_q(ti):>22}")
    say(f"  extent error (coverage < {EXTENT_COVERAGE:g} or beyond > {EXTENT_OVERSHOOT:g}): "
        f"{100 * ls['ext_loose']:.0f}% of IoU 0.5-0.75 hits, {100 * ls['ext_tight']:.0f}% of IoU > 0.75 hits")
    reasons = [k for k in FN_REASONS if not (b["mode"] == "free" and k in ("suppressed", "capped"))]
    for name, t in b["tables"].items():
        say("")
        say(f"by {t['desc']}")
        say(f"  {'bin':>12} {'GT%':>6} {'R@50':>6} {'R@75':>6} |" +
            "".join(f"{FN_SHORT[k]:>7}" for k in reasons) + f" | {'misses%':>7}" +
            (f" | {'hit-v2':>6} {'low-v2':>6}" if vw else ""))
        for row in t["rows"]:
            if row["n"] == 0:
                say(f"  {row['bin']:>12} {0:6.2f}      -      - |")
                continue
            say(f"  {row['bin']:>12} {100 * row['share']:6.2f} {100 * row['r50']:6.2f} {100 * row['r75']:6.2f} |" +
                "".join(f"{100 * row['reasons'][k]:7.2f}" for k in reasons) +
                f" | {100 * row['miss_share']:7.2f}" +
                (f" | {_pct(row['v2_hits']):>6} {_pct(row['v2_low']):>6}" if vw else ""))
    for name, t in b["fp_tables"].items():
        say("")
        say(f"false positives by {t['desc']}: kinds in % of the bin's false positives")
        say(f"  {'bin':>12} {'FP':>7} {'FP%':>6} | {'dup':>6} {'misloc':>6} {'ghost':>6} | dup kernel dist, px")
        for row in t["rows"]:
            say(f"  {row['bin']:>12} {row['n']:7d} {100 * row['share']:6.2f} | " +
                " ".join(f"{100 * row['kinds'][k]:6.2f}" for k in FP_KINDS) + f" | {row['twin_dist']}")


def _pct(v):
    return "-" if v is None or math.isnan(v) else f"{100 * v:.1f}"


def dataset_lines(lanes, say, nms_thres):
    G = len(lanes)
    if not G:
        return
    conflict = sum(ln["nn_dist"] < nms_thres for ln in lanes)
    say(f"GT lanes {G}: {100 * conflict / G:.2f}% have another GT lane within the NMS distance "
        f"(mean |dx| < {nms_thres:g} px), so NMS keeps only one of two perfect predictions; "
        f"{100 * np.mean([ln['fork'] for ln in lanes]):.2f}% fork / merge, "
        f"{100 * np.mean([ln['reversing'] for ln in lanes]):.2f}% y-reversing, "
        f"{100 * np.mean([ln['above_crop'] for ln in lanes]):.2f}% reach above the crop")


def write_csvs(out_dir, per_mode):
    gt_keys = ["mode", "image", "gt", "reason", "pred", "iou", "hit50", "hit75", "score", "best_iou",
               "best_in_output", "suppressor_is_hit", "coverage", "beyond", "dx", "normal",
               "slant", "extent", "bend", "nn_dist", "fork", "lanes", "resolution", "reversing",
               "above_frac", "n_points", "view"]
    with open(osp.join(out_dir, "gt_lanes.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=gt_keys, extrasaction="ignore")
        wr.writeheader()
        for mode, (lanes, _) in per_mode.items():
            for ln in lanes:
                row = dict(ln, mode=mode)
                pair = ln.get("pair") or (math.nan,) * 4
                row.update(coverage=pair[0], beyond=pair[1], dx=pair[2], normal=pair[3])
                wr.writerow({k: _fmt(row.get(k, "")) for k in gt_keys})
    fp_keys = ["mode", "image", "pred", "score", "kind", "best_iou", "slant", "extent", "twin_dist", "twin_near",
               "view"]
    with open(osp.join(out_dir, "false_positives.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fp_keys, extrasaction="ignore")
        wr.writeheader()
        for mode, (_, fps) in per_mode.items():
            for fp in fps:
                wr.writerow({k: _fmt(dict(fp, mode=mode).get(k, "")) for k in fp_keys})


def _fmt(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return "" if math.isnan(v) else ("inf" if math.isinf(v) else f"{v:.4f}")
    return v


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return None if math.isnan(o) else (1e30 if math.isinf(o) else o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ----------------------------------------------------------------------------
# --compare
# ----------------------------------------------------------------------------
def compare(paths, say):
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    labels = [r["label"] for r in runs]
    width = max(20, max(len(lab) for lab in labels) + 2)
    say("=" * 100)
    say("comparison")
    for lab, p in zip(labels, paths):
        say(f"  {lab}: {p}")
    for mode in MODES:
        have = [r for r in runs if mode in r["modes"]]
        if not have:
            continue
        say("")
        say(f"{MODE_LABEL[mode]}: operating point")
        say(f"  {'':34s}" + "".join(f"{lab:>{width}}" for lab in labels))

        def line(label, get):
            say(f"  {label:34s}" + "".join(
                f"{(get(r['modes'][mode]) if mode in r['modes'] else '-'):>{width}}" for r in runs))
        line("threshold", lambda b: f"{b['thr']:.2f}")
        line("P / R", lambda b: f"{100 * b['P']:.2f} / {100 * b['R']:.2f}")
        line("F1@50 / F1@75", lambda b: f"{100 * b['F1_50']:.2f} / {100 * b['F1_75']:.2f}")
        say("  recall@50 lost (points):")
        for k in FN_REASONS:
            line(f"    {k}", lambda b, k=k: f"{100 * b['reasons'][k]:.2f}")
        say("  false positives (% of output):")
        for k in FP_KINDS:
            line(f"    {k}", lambda b, k=k: f"{100 * b['fp'][k]:.2f}")
        if any(r["modes"][mode].get("views") for r in have):
            line("second view: % hits / low_sc / FP", lambda b: (
                f"{_pct(b['views']['hits'])} / {_pct(b['views']['low'])} / {_pct(b['views']['fp'])}"
                if b.get("views") else "-"))
        first = have[0]["modes"][mode]

        def table(name, desc, rows, get):
            say("")
            say(f"  {desc}")
            for row in rows:
                def cell(b, binname=row["bin"]):
                    for rr in b["tables"].get(name, {}).get("rows", []):
                        if rr["bin"] == binname and rr.get("n"):
                            return get(rr)
                    return "-"
                share = f"({100 * row.get('share', 0):.1f}% GT)"
                say(f"  {row['bin'] + ' ' + share:34s}" + "".join(
                    f"{(cell(r['modes'][mode]) if mode in r['modes'] else '-'):>{width}}" for r in runs))
        for name, t in first["tables"].items():
            table(name, f"R@50 / R@75 / low_score (points of the bin's recall) by {t['desc']}", t["rows"],
                  lambda rr: f"{100 * rr['r50']:.1f} / {100 * rr['r75']:.1f} / "
                             f"{100 * rr['reasons']['low_score']:.1f}")
        if any(r["modes"][mode].get("views") for r in have):
            for name in ("slant", "extent"):
                if name in first["tables"]:
                    t = first["tables"][name]
                    table(name, f"second view: % of the bin's hits / % of its low_score misses whose best "
                                f"candidate came from it, by {t['desc']}", t["rows"],
                          lambda rr: (f"{_pct(rr['v2_hits'])} / {_pct(rr['v2_low'])}" if "v2_hits" in rr else "-"))


# ----------------------------------------------------------------------------
# inference
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


def pred_rows(pred_dict):
    """[B, Q, 6 + R] rows in the official layout and the get_lanes scores [B, Q]."""
    logits = pred_dict["cls_logits"]
    score = torch.nn.Softmax(dim=2)(logits)[:, :, 1]
    rows = torch.cat([logits, pred_dict["anchor_params"], pred_dict["lengths"], pred_dict["xs"]], dim=-1)
    return rows, score


def image_records(pred_dict, samples, floor, max_n, view=None):
    """view [B, Q] (FovealLaneHead): the view each query's proposal came from, stored per prediction."""
    rows, score = pred_rows(pred_dict)
    out = []
    for b, ds in enumerate(samples):
        s = score[b]
        keep = torch.nonzero(s >= floor, as_tuple=False).squeeze(1)
        order = keep[torch.argsort(s[keep], descending=True, stable=True)]
        trunc = float(s[order[max_n]]) if len(order) > max_n else None
        order = order[:max_n]
        meta = ds.metainfo
        out.append(dict(filename=str(meta["filename"]),
                        ori_hw=(int(meta["ori_shape"][0]), int(meta["ori_shape"][1])),
                        offset=int(meta["crop_offset"][1]),
                        idx=order.cpu().numpy().astype(np.int64),
                        rows=rows[b, order].float().cpu().numpy(),
                        score=s[order].float().cpu().numpy(), n_above=int(len(keep)), trunc=trunc))
        if view is not None:
            out[-1]["view"] = view[b, order.to(view.device)].cpu().numpy().astype(np.int8)
    return out


class ViewCapture:
    """FovealLaneHead: the view (0 the context, 1 the second view) of each query's proposal, read from the head's
    internal output during head(feats); None for every other head."""

    def __init__(self, head):
        self.head, self.view = head, None
        self.on = hasattr(head, "fovea_mode") and hasattr(head, "_forward")

    def __enter__(self):
        if self.on:
            fwd = self.head._forward

            def capture(*a, **k):
                out = fwd(*a, **k)
                prop = out.get("prop") if isinstance(out, dict) else None
                self.view = None if prop is None else prop["view"]
                return out
            self.head._forward = capture
        return self

    def __exit__(self, *exc):
        if self.on:
            del self.head._forward                   # back to the class method
        return False


def score_source(head):
    """Which scores head(feats)[-1] carries for the BLR-family heads (BLRHead, HBLRHead, FovealLaneHead): the o2o
    scores on the NMS-free test path, the o2m scores with NMS, unless test_score fixes one; None for others."""
    if not hasattr(head, "cls_o2o"):
        return None
    return getattr(head, "test_score", None) or ("o2m" if bool(head.test_cfg.get("use_nms", True)) else "o2o")


@torch.no_grad()
def verify(head, pred_dict, recs, decoder, meta, top_k):
    """get_lanes on both paths at the config threshold keeps exactly the rebuilt
    selection (same scores) and decodes the same number of lanes. Returns
    (images checked, mismatches)."""
    cfg = head.test_cfg
    saved = cfg["use_nms"]
    thr = float(cfg["conf_threshold"])
    checked = bad = 0
    try:
        for mode in MODES:
            cfg["use_nms"] = mode == "nms"
            lanes, scores = head.get_lanes(pred_dict, as_lanes=False,
                                           extend_bottom=bool(cfg.get("extend_bottom", True)))
            for b, rec in enumerate(recs):
                if rec["trunc"] is not None and rec["trunc"] >= thr:
                    continue
                near, _ = kernel_pairs(rec["rows"], meta["n_strips"], meta["img_w"], float(cfg["nms_thres"]))
                kept, _, _ = select(rec["score"], near, thr, mode, top_k)
                ref = np.sort(np.asarray([float(s) for s in scores[b]], dtype=np.float64))
                got = np.sort(rec["score"][kept].astype(np.float64))
                n_dec = sum(decoder.points(rec["rows"][i]) is not None for i in kept)
                checked += 1
                if len(ref) != len(got) or not np.allclose(ref, got, rtol=0, atol=1e-7) or len(lanes[b]) != n_dec:
                    bad += 1
    finally:
        cfg["use_nms"] = saved
    return checked, bad


@torch.no_grad()
def run_inference(cfg, args, device):
    from mmengine.runner import Runner
    model = build_model(cfg, args.checkpoint, device)
    head = model.bbox_head
    meta = head_meta(head)
    meta.update(score_source=score_source(head), second_view=getattr(head, "fovea_mode", None))
    decoder = Decoder(type(head).predictions_to_lanes, meta["prior_ys"], meta["n_strips"], meta["test_cfg"])
    top_k = int(head.test_cfg["nms_topk"])
    loader = Runner.build_dataloader(copy.deepcopy(cfg.test_dataloader if args.split == "test"
                                                   else cfg.val_dataloader))
    records, checked, bad = [], 0, 0
    tic = time.time()
    for bi, data in enumerate(loader):
        data = model.data_preprocessor(data, False)
        feats = model.extract_feat(data["inputs"])
        with ViewCapture(head) as cap:
            pred_dict = head(feats)[-1]
        recs = image_records(pred_dict, data["data_samples"], args.floor, args.max_per_image, cap.view)
        if bi < args.verify_batches:
            c, m = verify(head, pred_dict, recs, decoder, meta, top_k)
            checked, bad = checked + c, bad + m
        records += recs
        if len(records) % 2000 < len(recs):
            print(f"  {len(records)} images ({time.time() - tic:.0f} s)", flush=True)
        if args.max_images is not None and len(records) >= args.max_images:
            break
    print(f"verified {checked} image-path pairs against get_lanes (with and without NMS): "
          f"{bad} mismatches", flush=True)
    return records, meta, dict(checked=checked, mismatches=bad)


# ----------------------------------------------------------------------------
def settings(cfg, meta, args):
    ns = SimpleNamespace(split="valid", data_root=None, data_list=None, lane_interp=None, curve_points=100)
    st = rc.settings_from_config(cfg, ns)
    ev = cfg.get("test_evaluator", {}) or {}
    st.update(img_w=meta["img_w"], img_h=meta["img_h"], n_rows_m1=meta["n_strips"],
              y_step=int(ev.get("y_step", 8)))
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--mode", choices=["nms", "free", "both"], default=None,
                    help="which output to break down (the sweep always has both); default: the config's "
                         "test_cfg.use_nms (free for the NMS-free BLR / HBLR / foveal configs)")
    ap.add_argument("--thr-nms", type=float, default=None, help="default: best F1@50 of the sweep")
    ap.add_argument("--thr-free", type=float, default=None, help="default: best F1@50 of the sweep")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="test = the config's test_dataloader (CurveLanes valid)")
    ap.add_argument("--floor", type=float, default=0.0, help="store only predictions with score >= this")
    ap.add_argument("--max-per-image", type=int, default=64,
                    help="store the highest-scored predictions of each image, at most this many")
    ap.add_argument("--save-dump", default=None)
    ap.add_argument("--load-dump", default=None, help="reuse a saved inference pass")
    ap.add_argument("--out-dir", default=None,
                    help="default: work_dirs/curvelanes_error_breakdown/<config name>")
    ap.add_argument("--label", default=None, help="name of this run in --compare")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--verify-batches", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=None, help="quick look only")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--compare", nargs="+", default=None, metavar="SUMMARY_JSON")
    args = ap.parse_args(argv)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    if args.compare:
        compare(args.compare, say)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            with open(osp.join(args.out_dir, "comparison.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        return None
    if not args.config:
        ap.error("CONFIG is required (or --compare)")

    from mmengine.config import Config
    if args.load_dump:
        cfg = Config.fromfile(args.config, import_custom_modules=False)
        with open(args.load_dump, "rb") as f:
            dump = pickle.load(f)
        records, meta, vinfo = dump["records"], dump["meta"], dump["meta"]["verify"]
        args.floor = dump["meta"]["floor"]
        decoder = make_decoder(meta)
        checkpoint = dump["meta"].get("checkpoint")
    else:
        if not args.checkpoint:
            ap.error("a checkpoint is needed unless --load-dump is given")
        cfg = Config.fromfile(args.config)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        records, meta, vinfo = run_inference(cfg, args, device)
        checkpoint = args.checkpoint
        decoder = Decoder(resolve(meta["decode_cls"]).predictions_to_lanes, meta["prior_ys"],
                          meta["n_strips"], meta["test_cfg"])
        if args.save_dump:
            os.makedirs(osp.dirname(osp.abspath(args.save_dump)), exist_ok=True)
            with open(args.save_dump, "wb") as f:
                pickle.dump(dict(records=records, meta=dict(
                    meta, config=args.config, checkpoint=checkpoint, split=args.split, floor=args.floor,
                    max_per_image=args.max_per_image, verify=vinfo)), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved {args.save_dump}")
    if args.max_images is not None:
        records = records[:args.max_images]

    tc = meta["test_cfg"]
    if args.mode is None:
        args.mode = "nms" if bool(tc.get("use_nms", True)) else "free"
    second_view = meta.get("second_view")
    if second_view in (None, "off") or not any("view" in r for r in records):
        second_view = None
    cfg_thr = float(tc["conf_threshold"])
    nms_thres, top_k = float(tc["nms_thres"]), int(tc["nms_topk"])
    grid = set(np.round(np.arange(0.05, 0.951, 0.05), 2).tolist()) | {0.96, 0.97, 0.98, 0.99, cfg_thr}
    grid |= {t for t in (args.thr_nms, args.thr_free) if t is not None}
    thresholds = sorted(t for t in grid if t >= args.floor - 1e-12)
    st = settings(cfg, meta, args)
    state = dict(st=st, decoder=decoder, nms_thres=nms_thres, top_k=top_k, thresholds=thresholds)
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    tic = time.time()
    if jobs == 1:
        _init_worker(state)
        results = [analyse_image(r) for r in records]
    else:
        with mp.get_context("fork").Pool(jobs, initializer=_init_worker, initargs=(state,)) as pool:
            results = list(pool.imap(analyse_image, records, chunksize=32))
    print(f"analysed {len(results)} images in {time.time() - tic:.0f} s", flush=True)

    sweep = sweep_table(results, thresholds)
    exact_from = max([r["trunc"] for r in records if r["trunc"] is not None], default=None)
    n_stored = [len(r["score"]) for r in records]
    say("=" * 100)
    say(f"CurveLanes error breakdown  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"  config: {args.config}")
    say(f"  checkpoint: {checkpoint}")
    say(f"  head: {meta.get('head_cls', '?')}; NMS distance {nms_thres:g} px over common rows, "
        f"nms_topk {top_k}; metric {st['eval']['eval_width']}x{st['eval']['eval_height']} "
        f"width {st['eval']['lane_width']} (vega LaneMetricCore), y_step {st['y_step']}")
    say(f"  images {len(records)}; stored predictions per image (the highest-scored, at most "
        f"{dump['meta']['max_per_image'] if args.load_dump else args.max_per_image}, score >= "
        f"{args.floor:g}): median {int(np.median(n_stored)) if n_stored else 0}, max {max(n_stored, default=0)}"
        + (f"; the highest unstored score is {exact_from:.3f}: rows of the sweep below it are approximate"
           if exact_from is not None and exact_from >= thresholds[0] else ""))
    src = meta.get("score_source")
    if src == "o2o":
        say("  scores: o2o (cls_o2o, the head's NMS-free test path); the with-NMS columns run NMS on these "
            "o2o scores, not on the o2m scores the head uses with use_nms=True")
    elif src == "o2m":
        say("  scores: o2m (cls_o2m, the head's test path with NMS); the without-NMS columns use these o2m "
            "scores, not the o2o scores the head uses with use_nms=False")
    if second_view:
        say(f"  FovealLaneHead second view: {second_view}; each stored prediction carries the view its query's "
            f"proposal came from (0 context, 1 {second_view}; view column of the CSVs)")
    say(f"  verify: {json.dumps(vinfo)}")
    say("")

    modes = ["nms", "free"] if args.mode == "both" else [args.mode]
    chosen, ops = {}, {}
    for mode in MODES:
        given = args.thr_nms if mode == "nms" else args.thr_free
        best = max(sweep[mode], key=lambda r: (r["F1_50"], -r["thr"]))
        ops[mode] = given if given is not None else best["thr"]
        chosen[mode] = "given" if given is not None else "best F1@50 of the sweep"

    say("threshold sweep: P / R / F1 at IoU 0.5, F1 at IoU 0.75   (c = config threshold, "
        "* = operating threshold)")
    say(f"  {'':8s} {'---------- with NMS -----------':>34}   {'--------- without NMS ---------':>34}")
    say(f"  {'thr':8s} " + "   ".join(f"{'P':>8}{'R':>8}{'F1@50':>9}{'F1@75':>9}" for _ in MODES))
    for k, thr in enumerate(thresholds):
        tag = ("c" if abs(thr - cfg_thr) < 1e-9 else " ")
        cells = []
        for mode in MODES:
            r = sweep[mode][k]
            star = "*" if abs(thr - ops[mode]) < 1e-9 else " "
            cells.append(f"{100 * r['P']:8.2f}{100 * r['R']:8.2f}{100 * r['F1_50']:8.2f}{star}{100 * r['F1_75']:9.2f}")
        say(f"  {thr:.2f} {tag}   " + "   ".join(cells))
    say("")

    ctx = dict(top_k=top_k, nms_thres=nms_thres, n_strips=meta["n_strips"], img_w=meta["img_w"])
    specs = gt_attr_specs(nms_thres)
    per_mode, summaries = {}, {}
    for mode in modes:
        lanes, fps, counts = [], [], []
        for c, rec in zip(results, records):
            ln, fp, cn = evaluate_image(c, rec, ops[mode], mode, ctx)
            lanes += ln
            fps += fp
            counts.append(cn)
        if mode == modes[0]:
            dataset_lines(lanes, say, nms_thres)
            say("")
        b = breakdown(lanes, fps, counts, mode, ops[mode], specs, second_view)
        k = [i for i, t in enumerate(thresholds) if abs(t - ops[mode]) < 1e-9]
        if k:
            sw = sweep[mode][k[0]]
            assert abs(sw["F1_50"] - b["F1_50"]) < 1e-9, "breakdown and sweep disagree"
        format_mode(b, cfg_thr, chosen[mode], say, nms_thres)
        say("")
        per_mode[mode] = (lanes, fps)
        summaries[mode] = b

    out_dir = args.out_dir or osp.join("work_dirs", "curvelanes_error_breakdown",
                                       osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    label = args.label or (osp.splitext(osp.basename(args.config))[0] +
                           (f" ({osp.basename(checkpoint)})" if checkpoint else ""))
    with open(osp.join(out_dir, "summary.json"), "w") as f:
        json.dump(_json_safe(dict(label=label, config=args.config, checkpoint=checkpoint,
                                  config_threshold=cfg_thr, nms_thres=nms_thres, nms_topk=top_k,
                                  sweep=sweep, modes=summaries)), f, indent=1)
    write_csvs(out_dir, per_mode)
    print(f"wrote {out_dir}/summary.txt, summary.json, gt_lanes.csv, false_positives.csv")
    return dict(results=results, sweep=sweep, summaries=summaries, per_mode=per_mode)


if __name__ == "__main__":
    main()
