"""How well can each lane representation fit the CurveLanes ground truth?

CPU only, no checkpoint. Every GT lane is turned into the output a perfectly
trained detector of each representation would produce, and those outputs are
scored against the GT with the official CurveLanes metric (vega LaneMetricCore:
lanes drawn 5 px wide on 224x224, Hungarian, IoU > 0.5; also IoU > 0.75). The
F1 of each row is that representation's ceiling.

  gt                the GT lanes themselves (sanity: must be 1.0)
  rows_trained      the CLR target as training builds it, decoded as the head
                    decodes it: crop, resize to the network input, cut_unsorted,
                    sample_lane_rows (lane_interp, extended to the image bottom),
                    start/length rows, predictions_to_lanes, CurvelanesMetric.
                    The ceiling of CLRerNet and of CLRBezierLane (curve + free
                    row residual can output any row target).
  rows_no_ext       the same rows without the extension below the lowest
                    annotated point: x(y) itself, without the bottom convention
  cubic_xy_ls       one cubic x(y) (= any fixed-y cubic Bezier), plain least
                    squares on the rows_no_ext rows, decoded over them
  cubic_2d          one free 2D cubic Bezier (BezierLaneNet / BezierFormer),
                    least squares on the annotated polyline in network pixels
                    with parameter correction, no cut, no extension; its points
                    go to the metric directly (it is not a function of y)
Anchor states: the curve a stage carries to the next one and samples features
along, i.e. the output without the per-row residual, at its training target
and over the rows_trained start/length:
  line_trained      CLRerNet's line anchor at its target (start point, mean
                    angle, as PackCLRNetInputs builds them; generate_anchors)
  cp_clamp          the head's GT control-point fit (global frame, ridge toward
                    the best line) with each control point clamped on its own
                    to [-cp_x_margin, 1 + cp_x_margin]  (brr_cp_fit_box=False)
  cp_box            the same objective solved inside that box (brr_cp_fit_box=True)
  cp_box_m1/_m2     box fit with cp_x_margin 1.0 / 2.0
  cp_unbounded      the same ridge fit without any margin
The predicted control points are clamped to the same margin, so cp_box is the
closest curve the head's state can hold.

Read:
  gt -> rows_no_ext             cost of x(y) itself (cut_unsorted + interpolation)
  rows_no_ext -> rows_trained   cost of extending lanes to the image bottom
  cubic_xy_ls vs cubic_2d       fixed-y vs free 2D control points, same lanes
  anchor state -> rows_trained  what the row residual has to fix

Lanes whose y is not strictly decreasing along the annotation (they bend
sideways or back down) are reported separately: x(y) cannot represent them.
Recall is also split by slant, sum|dx| / sum|dy| along the lane in network px.

    python tools/clrbezier/repr_ceiling.py CONFIG [--split valid] [--jobs 8] [--skip-2d]

CONFIG is a CurveLanes config (data root, list, lane_interp, max_lanes, the
head's CP-fit settings and the metric are read from it). Output:
work_dirs/repr_ceiling/<config>/summary_<split>.txt and lanes_<split>.csv.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import multiprocessing as mp
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
# (ori_h, ori_w) -> crop top, as CurvelanesDataset (libs/datasets/datasets.py)
CROP_OFFSETS = {(1440, 2560): 640, (660, 1570): 180, (720, 1280): 368}
REPS = ["gt", "rows_trained", "rows_no_ext", "cubic_xy_ls", "cubic_2d",
        "line_trained", "cp_clamp", "cp_box", "cp_box_m1", "cp_box_m2", "cp_unbounded"]
# (name, cp_x_margin (None = the config's), box-constrained fit)
CP_VARIANTS = (("cp_clamp", None, False), ("cp_box", None, True), ("cp_box_m1", 1.0, True),
               ("cp_box_m2", 2.0, True), ("cp_unbounded", 1e3, False))
SLANT_BINS = ((0.0, 1.0, "<1"), (1.0, 2.0, "1-2"), (2.0, 4.0, "2-4"), (4.0, math.inf, ">4"))
INVALID = -1e5


def _load(name, rel):
    """Import a repo file by path (no package __init__, so no mmdet needed)."""
    path = osp.join(REPO, rel)
    if not osp.exists(path):
        path = osp.join(os.getcwd(), rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lane_utils = _load("_ceiling_lane_utils", "libs/utils/lane_utils.py")
_GEOMETRY = None


def geometry():
    """libs/clrbezier/geometry.py, loaded once per process (torch only)."""
    global _GEOMETRY
    if _GEOMETRY is None:
        _GEOMETRY = _load("_ceiling_geometry", "libs/clrbezier/geometry.py")
    return _GEOMETRY


# ----------------------------------------------------------------------------
# official CurveLanes metric pieces (vega LaneMetricCore)
# ----------------------------------------------------------------------------
try:  # the metric YSY's evaluation uses
    from vega.metrics.pytorch.lane_metric import resize_lane, spline_interp  # noqa: F401
    VEGA = "vega"
except Exception:  # verbatim copies (vega/metrics/pytorch/lane_metric.py, Apache-2.0)
    VEGA = "vendored"

    def _calc_params(lane):
        params = []
        n_pt = len(lane)
        if n_pt < 2:
            return params
        if n_pt == 2:
            h0 = np.sqrt((lane[0]['x'] - lane[1]['x']) * (lane[0]['x'] - lane[1]['x'])
                         + (lane[0]['y'] - lane[1]['y']) * (lane[0]['y'] - lane[1]['y']))
            a_x = lane[0]['x']
            a_y = lane[0]['y']
            b_x = (lane[1]['x'] - a_x) / h0
            b_y = (lane[1]['y'] - a_y) / h0
            params.append({"a_x": a_x, "b_x": b_x, "c_x": 0, "d_x": 0,
                           "a_y": a_y, "b_y": b_y, "c_y": 0, "d_y": 0, "h": h0})
            return params
        h = []
        for i in range(n_pt - 1):
            dx = lane[i]['x'] - lane[i + 1]['x']
            dy = lane[i]['y'] - lane[i + 1]['y']
            h.append(np.sqrt(dx * dx + dy * dy))
        A, B, C, D_x, D_y = [], [], [], [], []
        for i in range(n_pt - 2):
            A.append(h[i])
            B.append(2 * (h[i] + h[i + 1]))
            C.append(h[i + 1])
            dx1 = (lane[i + 1]['x'] - lane[i]['x']) / h[i]
            dx2 = (lane[i + 2]['x'] - lane[i + 1]['x']) / h[i + 1]
            tmpx = 6 * (dx2 - dx1)
            dy1 = (lane[i + 1]['y'] - lane[i]['y']) / h[i]
            dy2 = (lane[i + 2]['y'] - lane[i + 1]['y']) / h[i + 1]
            tmpy = 6 * (dy2 - dy1)
            if i == 0:
                C[i] /= B[i]
                D_x.append(tmpx / B[i])
                D_y.append(tmpy / B[i])
            else:
                base_v = B[i] - A[i] * C[i - 1]
                C[i] /= base_v
                D_x.append((tmpx - A[i] * D_x[i - 1]) / base_v)
                D_y.append((tmpy - A[i] * D_y[i - 1]) / base_v)
        Mx = np.zeros(n_pt)
        My = np.zeros(n_pt)
        Mx[n_pt - 2] = D_x[n_pt - 3]
        My[n_pt - 2] = D_y[n_pt - 3]
        for i in range(n_pt - 4, -1, -1):
            Mx[i + 1] = D_x[i] - C[i] * Mx[i + 2]
            My[i + 1] = D_y[i] - C[i] * My[i + 2]
        Mx[0] = 0
        Mx[-1] = 0
        My[0] = 0
        My[-1] = 0
        for i in range(n_pt - 1):
            a_x = lane[i]['x']
            b_x = (lane[i + 1]['x'] - lane[i]['x']) / h[i] - (2 * h[i] * Mx[i] + h[i] * Mx[i + 1]) / 6
            c_x = Mx[i] / 2
            d_x = (Mx[i + 1] - Mx[i]) / (6 * h[i])
            a_y = lane[i]['y']
            b_y = (lane[i + 1]['y'] - lane[i]['y']) / h[i] - (2 * h[i] * My[i] + h[i] * My[i + 1]) / 6
            c_y = My[i] / 2
            d_y = (My[i + 1] - My[i]) / (6 * h[i])
            params.append({"a_x": a_x, "b_x": b_x, "c_x": c_x, "d_x": d_x,
                           "a_y": a_y, "b_y": b_y, "c_y": c_y, "d_y": d_y, "h": h[i]})
        return params

    def spline_interp(*, lane, step_t=1):
        interp_lane = []
        if len(lane) < 2:
            return lane
        for f in _calc_params(lane):
            t = 0
            while t < f['h']:
                x = f['a_x'] + f['b_x'] * t + f['c_x'] * t * t + f['d_x'] * t * t * t
                y = f['a_y'] + f['b_y'] * t + f['c_y'] * t * t + f['d_y'] * t * t * t
                interp_lane.append({"x": x, "y": y})
                t += step_t
        interp_lane.append(lane[-1])
        return interp_lane

    def resize_lane(lane, x_ratio, y_ratio):
        return [{"x": float(p['x']) / x_ratio, "y": float(p['y']) / y_ratio} for p in lane]


def lane_mask(lane, ori_hw, ev):
    """The mask vega's calc_iou draws for one lane (same resize, spline, cv2.line)."""
    h, w = ev["eval_height"], ev["eval_width"]
    im = np.zeros((h, w), np.uint8)
    lane = resize_lane(lane, np.true_divide(ori_hw[1], w), np.true_divide(ori_hw[0], h))
    pts = spline_interp(lane=lane, step_t=1)
    for i in range(len(pts) - 1):
        cv2.line(im, (int(pts[i]['x']), int(pts[i]['y'])),
                 (int(pts[i + 1]['x']), int(pts[i + 1]['y'])), 255, ev["lane_width"])
    return im > 0


def iou_matrix(gt_masks, pr_masks):
    out = np.zeros((len(gt_masks), len(pr_masks)))
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pr_masks):
            union = np.count_nonzero(g | p)
            out[i, j] = np.count_nonzero(g & p) / float(union) if union else 0.0
    return out


def match(iou):
    """Per-GT IoU of its Hungarian partner (vega: linear_sum_assignment on 1 - IoU)."""
    best = np.zeros(iou.shape[0])
    if iou.size:
        rows, cols = linear_sum_assignment(1.0 - iou)
        best[rows] = iou[rows, cols]
    return best


def dedup_consecutive(lane):
    out = []
    for p in lane:
        if not out or (p["x"], p["y"]) != (out[-1]["x"], out[-1]["y"]):
            out.append(p)
    return out


# ----------------------------------------------------------------------------
# training-target path
# ----------------------------------------------------------------------------
def read_lines_txt(path):
    """CurvelanesMetric.parse_anno (the eval GT): every line, original pixels."""
    lanes = []
    with open(path) as f:
        for line in f:
            v = [float(t) for t in line.strip().split(" ") if t]
            lanes.append([(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)])
    return lanes


def cut_unsorted_points(lanes):
    """LaneAlbumentation.cut_unsorted_points (curvelane branch), flat lists."""
    out_points = []
    for lane in lanes:
        out_points.append([])
        prev_y = 1e8
        for x, y in zip(lane[0::2], lane[1::2]):
            if y < prev_y:
                out_points[-1].extend([x, y])
                prev_y = y
    return out_points


def is_sorted(lane):
    y = np.asarray(lane[1::2])
    return bool(np.all(y[1:] < y[:-1]))


def row_targets(flat_lane, st):
    """PackCLRNetInputs.convert_targets for one lane: (xs_px [R], first, last, n_ext)
    or None when the packer drops it. n_ext = rows below the lowest point."""
    try:
        all_xs = lane_utils.sample_lane_rows(flat_lane, st["offsets_ys"], st["img_w"],
                                             interp=st["lane_interp"])
    except AssertionError:
        return None
    inside = np.nonzero((all_xs >= 0) & (all_xs < st["img_w"]))[0]
    if len(inside) <= 1:
        return None
    xs = np.full(st["n_rows"], INVALID)
    xs[:len(all_xs)] = all_xs
    max_y = max(flat_lane[1::2])
    n_ext = int(np.sum(st["offsets_ys"] > max_y))
    return xs, int(inside[0]), int(inside[-1]), n_ext


def decode_rows(pred_xs_norm, first, last, st):
    """CLRerHead.predictions_to_lanes with the CurveLanes test_cfg
    (ori_img_h=1, cut_height=0): (x, y) points, crop-normalized."""
    n = st["n_rows"] - 1
    start_y = 1.0 - first / n
    length = last - first + 1
    start = min(max(0, int(round((1 - start_y) * n))), n)
    end = min(start + int(round(length)) - 1, n)
    xs = np.asarray(pred_xs_norm, dtype=np.float64)
    if st["extend_bottom"]:
        edge = (xs[:start] >= 0.0) & (xs[:start] <= 1.0)
        start -= int(np.cumprod(edge[::-1]).sum())
    ys = st["prior_ys"][start:end + 1][::-1]
    lx = xs[start:end + 1][::-1]
    if len(lx) <= 1:
        return None
    return np.stack([lx, ys], axis=1)


def to_metric_points(points, ori_hw, offset, y_step):
    """CurvelanesMetric.crop_to_original + convert_coords_laneatt for one lane."""
    ori_h, ori_w = ori_hw
    p = np.array(points, dtype=np.float64, copy=True)
    p[:, 1] = (p[:, 1] * (ori_h - offset) + offset) / ori_h
    lane = lane_utils.Lane(points=p)
    ys = np.arange(0, ori_h, y_step) / ori_h
    xs = lane(ys)
    valid = (xs >= 0) & (xs < 1)
    lx = (xs * ori_w)[valid][::-1]
    ly = (ys[valid] * ori_h)[::-1]
    return [{"x": x, "y": y} for x, y in zip(lx, ly)]


# ----------------------------------------------------------------------------
# curve fits
# ----------------------------------------------------------------------------
def _bern(t):
    o = 1.0 - t
    return np.stack([o ** 3, 3 * o * o * t, 3 * o * t * t, t ** 3], axis=-1)


def _bern_d1(t):
    o = 1.0 - t
    return np.stack([-3 * o * o, 3 * o * o - 6 * o * t, 6 * o * t - 3 * t * t, 3 * t * t], axis=-1)


def _bern_d2(t):
    o = 1.0 - t
    return np.stack([6 * o, -12 * o + 6 * t, 6 * o - 12 * t, 6 * t], axis=-1)


def fit_free_cubic(pts, iters=20):
    """Least-squares cubic Bezier in 2D to an ordered polyline: the 4 control
    points and every point's curve parameter are fitted jointly (end parameters
    pinned at 0 and 1), so this is the closest cubic, not a chord-length
    approximation. The annotated points are the data, weighted by the polyline
    length around each (a densely annotated stretch does not dominate); sparse
    lanes (< 6 points) are resampled along the polyline first. Starts from
    chord-length, centripetal and uniform parameters (Hoschek correction, then
    Levenberg-Marquardt) and keeps the best. Returns cp [4, 2] and the weighted
    mean point distance, or (None, inf)."""
    from scipy.optimize import least_squares
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], seg > 1e-9])]
    if len(pts) < 2:
        return None, math.inf
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if seg.sum() < 1e-6:
        return None, math.inf
    if len(pts) < 6:
        s = np.concatenate([[0.0], np.cumsum(seg)])
        u = np.linspace(0.0, s[-1], 20)
        pts = np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], axis=1)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    w = np.concatenate([[seg[0] / 2], (seg[:-1] + seg[1:]) / 2, [seg[-1] / 2]])
    w = w / w.sum()
    sw = np.sqrt(w)[:, None]

    n = len(pts)

    def residual(z):
        t = np.concatenate([[0.0], np.clip(z[8:], 0.0, 1.0), [1.0]])
        return ((_bern(t) @ z[:8].reshape(4, 2) - pts) * sw).ravel()

    def jacobian(z):
        t = np.concatenate([[0.0], np.clip(z[8:], 0.0, 1.0), [1.0]])
        basis = _bern(t) * sw                         # [n, 4]
        jac = np.zeros((2 * n, 8 + n - 2))
        jac[0::2, 0:8:2] = basis                      # x residuals vs cp x
        jac[1::2, 1:8:2] = basis                      # y residuals vs cp y
        tan = (_bern_d1(t) @ z[:8].reshape(4, 2)) * sw  # [n, 2]
        inner = np.arange(1, n - 1)
        jac[2 * inner, 8 + inner - 1] = tan[inner, 0]
        jac[2 * inner + 1, 8 + inner - 1] = tan[inner, 1]
        return jac

    def score(cp, t):
        return float((w * np.linalg.norm(_bern(t) @ cp - pts, axis=1)).sum())

    best, best_err = None, math.inf
    for alpha in (1.0, 0.5, 0.0):
        d = seg ** alpha
        t = np.concatenate([[0.0], np.cumsum(d)]) / d.sum()
        for _ in range(iters):
            basis = _bern(t)
            cp = np.linalg.lstsq(basis * sw, pts * sw, rcond=None)[0]
            res = basis @ cp - pts
            d1 = _bern_d1(t) @ cp
            d2 = _bern_d2(t) @ cp
            num = (res * d1).sum(1)
            den = (d1 * d1).sum(1) + (res * d2).sum(1)
            ok = np.abs(den) > 1e-9
            t = np.clip(t - np.where(ok, num / np.where(ok, den, 1.0), 0.0), 0.0, 1.0)
            t[0], t[-1] = 0.0, 1.0
            t = np.maximum.accumulate(t)  # keep the order along the lane
        cp = np.linalg.lstsq(_bern(t) * sw, pts * sw, rcond=None)[0]
        z = least_squares(residual, np.concatenate([cp.ravel(), t[1:-1]]), jac=jacobian,
                          method="lm", max_nfev=200).x
        t_lm = np.concatenate([[0.0], np.clip(z[8:], 0.0, 1.0), [1.0]])
        for c, tt in ((cp, t), (z[:8].reshape(4, 2), t_lm)):
            e = score(c, tt)
            if e < best_err:
                best, best_err = c, e
    return best, best_err


def clip_to_crop(pts):
    """The part of a network-space polyline (bottom -> top) inside y >= 0, cut at
    the crossing; the first contiguous run from the bottom."""
    out = [tuple(pts[0])] if pts[0][1] >= 0 else []
    if not out:
        return np.zeros((0, 2))
    for a, b in zip(pts[:-1], pts[1:]):
        if b[1] >= 0:
            out.append(tuple(b))
            continue
        r = a[1] / (a[1] - b[1])
        out.append((a[0] + r * (b[0] - a[0]), 0.0))
        break
    return np.asarray(out)


def cubic_xy_rows(xs_px, valid):
    """Plain least-squares cubic x(y) on the valid rows (degree drops with fewer rows)."""
    r = np.nonzero(valid)[0]
    y = r.astype(np.float64)
    deg = min(3, len(r) - 1)
    coef = np.polyfit(y, xs_px[r], deg)
    out = np.full(len(xs_px), INVALID)
    out[r[0]:r[-1] + 1] = np.polyval(coef, np.arange(r[0], r[-1] + 1, dtype=np.float64))
    return out


def line_rows(xs_px, st):
    """CLRerNet's line anchor at its training target, x at every row normalized by
    img_w - 1: PackCLRNetInputs' start row, start x and mean angle theta_far, then
    CLRerNetAnchorGenerator.generate_anchors."""
    inside = np.nonzero((xs_px >= 0) & (xs_px < st["img_w"]))[0]
    first = int(inside[0])
    n = st["n_rows"] - 1
    x_in = xs_px[inside].astype(np.float64)
    theta = np.arctan((inside[1:] - first) * (st["img_h"] / n) / (x_in[1:] - x_in[0] + 1e-5)) / math.pi
    theta = np.where(theta > 0, theta, 1 - np.abs(theta))
    # the target row is float32 (PackCLRNetInputs.lanes)
    y0, x0, th = np.float32(1 - first / n), np.float32(x_in[0]), np.float32(theta.mean())
    tan = math.tan(float(th) * math.pi + 1e-5)
    xs = float(x0) - (st["prior_ys"] - float(y0)) * st["img_h"] / tan
    return xs / float(st["img_w"] - 1)


# ----------------------------------------------------------------------------
# one image
# ----------------------------------------------------------------------------
def lane_flags(net_pts):
    """y-reversing: y not strictly decreasing along the annotation (x(y) cannot
    hold it). back_frac: share of the lane's length on segments going down.
    slant: sum|dx| / sum|dy| along the lane (network px; 1 = 45 degrees)."""
    p = np.asarray(net_pts, dtype=np.float64)
    if len(p) < 2:
        return False, 0.0, 0.0
    d = np.diff(p, axis=0)
    seg = np.linalg.norm(d, axis=1)
    total = seg.sum()
    back = seg[d[:, 1] >= 0].sum()
    dy = np.abs(d[:, 1]).sum()
    slant = float(np.abs(d[:, 0]).sum() / dy) if dy > 0 else math.inf
    return (bool(np.any(d[:, 1] >= 0)), float(back / total) if total > 0 else 0.0, slant)


def process_image(task):
    import torch
    torch.set_num_threads(1)
    rel, img_path, ori_hw, st = task
    ori_h, ori_w = ori_hw
    offset = CROP_OFFSETS[(ori_h, ori_w)]
    crop_h = ori_h - offset
    sx, sy = st["img_w"] / ori_w, st["img_h"] / crop_h
    gt = read_lines_txt(img_path.replace(".jpg", ".lines.txt"))
    ev = st["eval"]
    max_lanes = st["max_lanes"]

    # training lanes: CurvelanesDataset._read_lines_txt (>= 2 points), Resize, then
    # cut_unsorted when some lane is unsorted (no augmentation changes the order)
    train_idx = [i for i, ln in enumerate(gt) if len(ln) >= 2]
    flat = [[c for x, y in gt[i] for c in (x * sx, (y - offset) * sy)] for i in train_idx]
    if st["cut_unsorted"] and not all(is_sorted(f) for f in flat):
        flat = cut_unsorted_points(flat)
    packed = [(i, f) for i, f in zip(train_idx, flat) if len(f) > 2][:max_lanes]

    reps = st["reps"]
    out = {r: [] for r in reps}
    out["gt"] = [[{"x": x, "y": y} for x, y in ln] for ln in gt]
    rows = []
    for i, f in packed:
        tgt = row_targets(f, st)
        if tgt is None:
            continue
        xs, first, last, n_ext = tgt
        rows.append((i, xs, first, last, n_ext))

    norm = float(max(1, st["img_w"] - 1))
    fitted = {}
    if rows:
        tx = torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.float32)
        for name, margin, box in CP_VARIANTS:
            cp, _ = geometry().fit_global_cubic_to_clr_rows(
                tx, st["prior_ys_t"], st["img_w"], st["ridge"],
                st["margin"] if margin is None else margin,
                cp_frame="global", n_strips=st["n_rows"] - 1, box=box)
            fitted[name] = geometry().eval_global_cubic(cp, st["prior_ys_t"]).double().numpy()
    for k, (i, xs, first, last, n_ext) in enumerate(rows):
        states = [("rows_trained", xs / norm), ("line_trained", line_rows(xs, st))]
        states += [(name, fitted[name][k]) for name, _, _ in CP_VARIANTS]
        for name, px in states:
            pts = decode_rows(px, first, last, st)
            if pts is not None:
                out[name].append(to_metric_points(pts, ori_hw, offset, st["y_step"]))
        # the same rows without the extension below the lowest annotated point
        xs_ne = xs.copy()
        xs_ne[:n_ext] = INVALID
        inside = np.nonzero((xs_ne >= 0) & (xs_ne < st["img_w"]))[0]
        if len(inside) > 1:
            ext_first, ext_last = int(inside[0]), int(inside[-1])
            pts = decode_rows(xs_ne / norm, ext_first, ext_last, st)
            if pts is not None:
                out["rows_no_ext"].append(to_metric_points(pts, ori_hw, offset, st["y_step"]))
            valid = (xs_ne >= 0) & (xs_ne < st["img_w"])
            cub = cubic_xy_rows(xs_ne, valid)
            pts = decode_rows(cub / norm, ext_first, ext_last, st)
            if pts is not None:
                out["cubic_xy_ls"].append(to_metric_points(pts, ori_hw, offset, st["y_step"]))

    # free 2D cubic on the annotated polyline (no cut, no extension)
    for i in (train_idx[:max_lanes] if "cubic_2d" in reps else []):
        net = np.array([(x * sx, (y - offset) * sy) for x, y in gt[i]], dtype=np.float64)
        if net[0, 1] < net[-1, 1]:
            net = net[::-1]
        net = clip_to_crop(net)
        if len(net) < 2:
            continue
        cp2, _ = fit_free_cubic(net)
        if cp2 is None:
            continue
        curve = _bern(np.linspace(0.0, 1.0, st["n_curve"])) @ cp2
        ox, oy = curve[:, 0] / sx, curve[:, 1] / sy + offset
        keep = (ox >= 0) & (ox < ori_w) & (oy >= 0) & (oy < ori_h)
        lane = [{"x": float(x), "y": float(y)} for x, y in zip(ox[keep], oy[keep])]
        if len(lane) >= 2:
            out["cubic_2d"].append(lane)

    gt_lanes = [ln for ln in out["gt"] if len(ln) > 0]
    gt_masks = [lane_mask(ln, ori_hw, ev) for ln in gt_lanes]
    res = {"rel": rel, "n_gt": len(gt_lanes), "per_rep": {}}
    for name in reps:
        pr = [dedup_consecutive(ln) for ln in out[name] if len(ln) > 0]
        pr_masks = [lane_mask(ln, ori_hw, ev) for ln in pr]
        best = match(iou_matrix(gt_masks, pr_masks)) if gt_masks and pr_masks \
            else np.zeros(len(gt_masks))
        res["per_rep"][name] = dict(n_pr=len(pr), iou=best.tolist())
    flags = []
    for ln in gt:
        if len(ln) == 0:
            continue
        net = [(x * sx, (y - offset) * sy) for x, y in ln]
        if len(net) >= 2 and net[0][1] < net[-1][1]:
            net = net[::-1]
        rev, back, slant = lane_flags(net)
        flags.append(dict(n_points=len(ln), reversing=rev, back_frac=back, slant=slant))
    res["flags"] = flags
    res["capped"] = max(0, len(train_idx) - max_lanes)
    return res


# ----------------------------------------------------------------------------
# settings, image list, report
# ----------------------------------------------------------------------------
def _find(pipeline, type_name):
    for t in pipeline or []:
        if isinstance(t, dict) and t.get("type") == type_name:
            return t
    return {}


def settings_from_config(cfg, args):
    import torch
    head = cfg.model.bbox_head
    loss = head.get("loss_cfg", {})
    brr = head.get("brr_cfg", {}) or {}
    split_dl = cfg.train_dataloader if args.split == "train" else cfg.test_dataloader
    pack = _find(cfg.train_dataloader.dataset.pipeline, "PackCLRNetInputs")
    alaug = _find(cfg.train_dataloader.dataset.pipeline, "LaneAlbumentation")
    ev = cfg.get("test_evaluator", {}) or {}
    n_rows = int(head.get("num_points", 72))
    img_w, img_h = int(head.get("img_w", 800)), int(head.get("img_h", 320))
    tcfg = cfg.model.get("test_cfg", {}) or {}
    st = dict(
        data_root=args.data_root or split_dl.dataset.data_root,
        data_list=args.data_list or split_dl.dataset.data_list,
        lane_interp=args.lane_interp or pack.get("lane_interp", "spline"),
        max_lanes=int(pack.get("max_lanes", 16)),
        cut_unsorted=bool(alaug.get("cut_unsorted", False)),
        img_w=img_w, img_h=img_h, n_rows=n_rows,
        offsets_ys=np.arange(img_h, -1, -img_h / (n_rows - 1)),
        prior_ys=torch.linspace(1, 0, steps=n_rows).double().numpy(),  # as the head (float32)
        prior_ys_t=torch.linspace(1, 0, steps=n_rows),
        ridge=float(loss.get("brr_cp_fit_ridge", 0.01)),
        margin=float(brr.get("cp_x_margin", head.get("cp_x_margin", 0.5))),
        box=bool(loss.get("brr_cp_fit_box", False)),
        extend_bottom=bool(tcfg.get("extend_bottom", True)),
        y_step=8, n_curve=args.curve_points,
        eval=dict(eval_width=int(ev.get("eval_width", 224)),
                  eval_height=int(ev.get("eval_height", 224)),
                  lane_width=int(ev.get("lane_width", 5))),
    )
    return st


def image_tasks(st, max_images):
    from PIL import Image
    tasks, skipped = [], 0
    with open(st["data_list"]) as f:
        rels = [ln.strip().split(" ")[0].lstrip("/") for ln in f if ln.strip()]
    if max_images:
        rels = rels[:max_images]
    for rel in rels:
        path = osp.join(st["data_root"], rel)
        with Image.open(path) as im:
            w, h = im.size
        if (h, w) not in CROP_OFFSETS:  # CurvelanesDataset returns None for these
            skipped += 1
            continue
        tasks.append((rel, path, (h, w)))
    return tasks, skipped


def run(tasks, st, jobs):
    items = [(rel, path, hw, st) for rel, path, hw in tasks]
    if jobs <= 1:
        return [process_image(t) for t in items]
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs) as pool:
        return list(pool.imap(process_image, items, chunksize=16))


def summarize(results, st, skipped, split, say):
    reps = st["reps"]
    n_img = len(results)
    n_gt = sum(r["n_gt"] for r in results)
    flags = [f for r in results for f in r["flags"]]
    rev = np.array([f["reversing"] for f in flags], dtype=bool)
    slant = np.array([f["slant"] for f in flags], dtype=np.float64)
    config_cp = "cp_box" if st["box"] else "cp_clamp"
    say(f"images {n_img} (skipped {skipped} of unknown size), GT lanes {n_gt}, "
        f"lanes beyond max_lanes={st['max_lanes']}: {sum(r['capped'] for r in results)}")
    say(f"lane_interp={st['lane_interp']} cut_unsorted={st['cut_unsorted']} "
        f"extend_bottom={st['extend_bottom']} cp fit: ridge {st['ridge']} margin "
        f"{st['margin']} box {st['box']} (= {config_cp}); metric {VEGA} "
        f"{st['eval']['eval_width']}x{st['eval']['eval_height']} w{st['eval']['lane_width']}")
    say(f"y-reversing GT lanes (x(y) cannot hold them): {int(rev.sum())} "
        f"({100.0 * rev.mean() if len(rev) else 0:.2f}%)")
    say("")
    say(f"{'representation':<18}{'F1@50':>8}{'P':>8}{'R':>8}{'F1@75':>8}{'R@75':>8}"
        f"{'IoU med':>9}   recall@50 monotone / y-reversing")
    table, ious = {}, {}
    for name in reps:
        if name == "line_trained":
            say("anchor state (no row residual):")
        iou = np.array([v for r in results for v in r["per_rep"][name]["iou"]])
        ious[name] = iou
        n_pr = sum(r["per_rep"][name]["n_pr"] for r in results)
        row = {}
        for thr in (0.5, 0.75):
            hit = int((iou > thr).sum())
            p = hit / (n_pr + sys.float_info.epsilon)
            rcl = hit / (n_gt + sys.float_info.epsilon)
            row[thr] = (p, rcl, 2 * p * rcl / (p + rcl + sys.float_info.epsilon))
        mono = iou[~rev] if len(rev) == len(iou) else iou
        bend = iou[rev] if len(rev) == len(iou) else np.zeros(0)
        r_mono = (mono > 0.5).mean() if len(mono) else float("nan")
        r_bend = (bend > 0.5).mean() if len(bend) else float("nan")
        table[name] = row
        say(f"{name:<18}{100 * row[0.5][2]:>8.2f}{100 * row[0.5][0]:>8.2f}"
            f"{100 * row[0.5][1]:>8.2f}{100 * row[0.75][2]:>8.2f}{100 * row[0.75][1]:>8.2f}"
            f"{np.median(iou) if len(iou) else float('nan'):>9.3f}   "
            f"{100 * r_mono:.2f} / {100 * r_bend:.2f}")
    say("")

    say("recall@50 by slant, sum|dx| / sum|dy| along the lane in network px (1 = 45 deg):")
    bins = [(lo, hi, lab) for lo, hi, lab in SLANT_BINS]
    masks = [(slant >= lo) & ((slant < hi) | (hi == math.inf)) for lo, hi, _ in bins]
    say(f"{'':<18}" + "".join(f"{lab:>9}" for _, _, lab in bins))
    say(f"{'GT lanes (%)':<18}" + "".join(
        f"{100 * m.mean() if len(m) else 0:>9.2f}" for m in masks))
    for name in reps:
        iou = ious[name]
        if len(iou) != len(slant):
            continue
        say(f"{name:<18}" + "".join(
            f"{100 * (iou[m] > 0.5).mean():>9.2f}" if m.any() else f"{'-':>9}" for m in masks))
    say("")

    def d(a, b, thr=0.5):
        return 100 * (table[b][thr][2] - table[a][thr][2])

    def line(label, a, b):
        if a in table and b in table:
            say(f"  {label:<26}{a + ' -> ' + b:<32}{d(a, b):+.2f}, {d(a, b, 0.75):+.2f}")
    say("F1 differences (points; @50, @75):")
    line("x(y) itself", "gt", "rows_no_ext")
    line("bottom extension", "rows_no_ext", "rows_trained")
    line("free 2D vs fixed-y", "cubic_xy_ls", "cubic_2d")
    line("free 2D cubic vs rows", "rows_trained", "cubic_2d")
    say("what the row residual has to fix (anchor state -> rows_trained):")
    labels = {"line_trained": "CLRerNet line",
              "cp_clamp": f"margin {st['margin']:g}, clamp",
              "cp_box": f"margin {st['margin']:g}, box",
              "cp_box_m1": "margin 1, box", "cp_box_m2": "margin 2, box",
              "cp_unbounded": "no margin"}
    for name, lab in labels.items():
        line(lab + (" (config)" if name == config_cp else ""), name, "rows_trained")
    return table


def write_csv(results, st, path):
    reps = st["reps"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "lane", "n_points", "y_reversing", "back_frac", "slant"]
                   + [f"iou_{n}" for n in reps])
        for r in results:
            for j, fl in enumerate(r["flags"]):
                w.writerow([r["rel"], j, fl["n_points"], int(fl["reversing"]),
                            f"{fl['back_frac']:.4f}", f"{fl['slant']:.4f}"]
                           + [f"{r['per_rep'][n]['iou'][j]:.4f}" for n in reps])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--split", choices=["valid", "train"], default="valid",
                    help="valid = the config's test list (default)")
    ap.add_argument("--data-root", default=None, help="override the config's data_root")
    ap.add_argument("--data-list", default=None, help="override the config's data_list")
    ap.add_argument("--lane-interp", choices=["spline", "linear", "pchip"], default=None,
                    help="override PackCLRNetInputs.lane_interp of the config")
    ap.add_argument("--curve-points", type=int, default=100,
                    help="points sampled on each free 2D curve for the metric")
    ap.add_argument("--max-images", type=int, default=None, help="quick look only")
    ap.add_argument("--skip-2d", action="store_true",
                    help="skip the free 2D cubic fit (most of the run time)")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    from mmengine.config import Config
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    st = settings_from_config(cfg, args)
    st["reps"] = [r for r in REPS if not (args.skip_2d and r == "cubic_2d")]
    name = osp.splitext(osp.basename(args.config))[0]
    out_dir = args.out_dir or osp.join("work_dirs", "repr_ceiling", name)
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    tasks, skipped = image_tasks(st, args.max_images)
    say("=" * 78)
    say(f"representation ceiling  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"  config: {args.config}")
    say(f"  split: {args.split}  list: {st['data_list']}")
    results = run(tasks, st, args.jobs)
    summarize(results, st, skipped, args.split, say)
    say(f"({time.time() - t0:.0f} s)")
    with open(osp.join(out_dir, f"summary_{args.split}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    write_csv(results, st, osp.join(out_dir, f"lanes_{args.split}.csv"))
    print(f"wrote {out_dir}/summary_{args.split}.txt and lanes_{args.split}.csv")
    return results


if __name__ == "__main__":
    main()
