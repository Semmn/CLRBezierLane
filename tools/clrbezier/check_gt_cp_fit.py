"""How well do the GT Bezier control points reproduce the GT lanes? Read-only.

Builds the CLR target rows exactly as training does (libs/utils/lane_utils.py
sample_lane_rows, PackCLRNetInputs.convert_targets rules) from the annotations,
after the crop + resize of the val pipeline (no random augmentation), and fits
the GT control points with libs/clrbezier/geometry.py in each cp_frame. For
every lane it reports the pixel error between the fitted curve and the GT on
the visible rows, for

  clamp: the current target (fit, then each control point clamped to
         [-cp_x_margin, 1 + cp_x_margin] on its own);
  box:   the same fit solved under that bound (loss_cfg brr_cp_fit_box=True),
         only if this repo's geometry.py has it.

Run from the repo root:

    python tools/clrbezier/check_gt_cp_fit.py culane /path/to/CULane --limit 20000
    python tools/clrbezier/check_gt_cp_fit.py curvelanes /path/to/CurveLanes --limit 20000

Options: --list (annotation list, default the train list), --margin 0.5,
--ridge 1e-2, --min-span 0.1, --workers 8, --out (JSON summary).
"""
import argparse
import importlib.util
import inspect
import json
import os
from multiprocessing import Pool

import numpy as np
import torch

IMG_W, IMG_H, N_OFF = 800, 320, 72
N_STRIPS = N_OFF - 1
ROW_YS = np.arange(IMG_H, -1, -IMG_H / N_STRIPS)  # PackCLRNetInputs.offsets_ys
MAX_LANES = {"culane": 4, "curvelanes": 16}
# CurvelanesDataset crops: original (h, w) -> rows cut from the top
CURVELANES_CUT = {(1440, 2560): 640, (660, 1570): 180, (720, 1280): 368}
CULANE_HW, CULANE_CUT = (590, 1640), 270

_LU = None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init(repo):
    global _LU
    _LU = _load("_lane_utils", os.path.join(repo, "libs/utils/lane_utils.py"))


def image_hw(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        return h, w
    except ImportError:
        import cv2
        return cv2.imread(path).shape[:2]


def cut_unsorted(lane):
    """LaneAlbumentation.cut_unsorted_points for one flat lane."""
    out, prev = [], 1e8
    for x, y in zip(lane[0::2], lane[1::2]):
        if y < prev:
            out += [x, y]
            prev = y
    return out


def target_rows(lane):
    """One lane's 72 target x values (network px, bottom -> top, -1e5 above the top)
    with the convert_targets rules, or None when training would drop the lane."""
    try:
        xs = _LU.sample_lane_rows(lane, ROW_YS, IMG_W)
    except (AssertionError, Exception):
        return None
    inside = np.nonzero((xs >= 0) & (xs < IMG_W))[0]
    if len(inside) <= 1:
        return None
    row = np.full(N_OFF, -1e5, dtype=np.float32)
    row[: len(xs)] = xs
    return row


def process(job):
    dataset, img_path = job
    if dataset == "curvelanes":
        hw = image_hw(img_path)
        cut = CURVELANES_CUT.get(tuple(hw))
        if cut is None:
            return None
    else:
        hw, cut = CULANE_HW, CULANE_CUT
    sx, sy = IMG_W / hw[1], IMG_H / (hw[0] - cut)
    lanes = []
    with open(img_path.replace(".jpg", ".lines.txt")) as f:
        for line in f:
            v = [float(t) for t in line.split()]
            if len(v) < 4:
                continue
            lane = []
            for x, y in zip(v[0::2], v[1::2]):
                lane += [x * sx, (y - cut) * sy]
            lane = cut_unsorted(lane)
            if len(lane) > 2:  # convert_targets filter
                lanes.append(lane)
    rows = [r for r in (target_rows(l) for l in lanes[: MAX_LANES[dataset]]) if r is not None]
    return np.stack(rows) if rows else np.zeros((0, N_OFF), np.float32)


def lane_errors(geo, xs, frame, ridge, margin, min_span, box):
    kw = dict(cp_frame=frame, n_strips=N_STRIPS, min_span=min_span, return_frame=True)
    if box:
        kw["box"] = True
    ys = torch.linspace(1, 0, N_OFF)
    cp, ok, fr = geo.fit_global_cubic_to_clr_rows(xs, ys, IMG_W, ridge, margin, **kw)
    kind = "global" if frame == "global" else "support"   # support form covers anchored (top = 0)
    curve = geo.eval_cubic(cp, ys.view(1, -1).expand(len(xs), -1), fr[:, 1], kind, y_top=fr[:, 0])
    valid = (xs >= 0) & (xs < IMG_W)
    err = ((curve * (IMG_W - 1) - xs).abs()).where(valid, torch.zeros(()))
    mean = err.sum(1) / valid.sum(1).clamp_min(1)
    return mean, err.max(1).values, ok


def summarize(mean, mx, sel):
    m, x = mean[sel], mx[sel]
    if m.numel() == 0:
        return {}
    return dict(lanes=int(m.numel()), mean_px=float(m.mean()), p95_lane_mean_px=float(m.quantile(.95)),
                mean_worst_row_px=float(x.mean()), lanes_over_5px_pct=float(100 * (m > 5).float().mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["culane", "curvelanes"])
    ap.add_argument("data_root")
    ap.add_argument("--list", default=None, help="annotation list (default: the train list)")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.5, help="brr_cfg cp_x_margin")
    ap.add_argument("--ridge", type=float, default=1e-2, help="loss_cfg brr_cp_fit_ridge")
    ap.add_argument("--min-span", type=float, default=0.1, help="brr_cfg min_span (support)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    geo = _load("_geometry", os.path.join(repo, "libs/clrbezier/geometry.py"))
    has_box = "box" in inspect.signature(geo.fit_global_cubic_to_clr_rows).parameters

    if args.dataset == "culane":
        prefix = args.data_root
        lst = args.list or os.path.join(args.data_root, "list/train_gt.txt")
    else:
        prefix = os.path.join(args.data_root, "train")
        lst = args.list or os.path.join(args.data_root, "train/train_seg.txt")
    with open(lst) as f:
        imgs = [ln.split()[0].lstrip("/") for ln in f if ln.strip()]
    if args.limit:
        imgs = imgs[: args.limit]
    jobs = [(args.dataset, os.path.join(prefix, im)) for im in imgs]
    with Pool(args.workers, initializer=_init, initargs=(repo,)) as pool:
        parts = [p for p in pool.imap(process, jobs, chunksize=64) if p is not None]
    xs = torch.from_numpy(np.concatenate(parts)).float()
    print(f"{len(parts)} images, {len(xs)} lanes ({args.dataset}, margin {args.margin}, ridge {args.ridge})")

    out = {}
    for frame in ("global", "anchored", "support"):
        mean_c, mx_c, ok = lane_errors(geo, xs, frame, args.ridge, args.margin, args.min_span, False)
        mean_u, mx_u, _ = lane_errors(geo, xs, frame, args.ridge, 1e6, args.min_span, False)
        # lanes whose unconstrained fit leaves the margin = lanes the clamp changes
        cp_u, _ = geo.fit_global_cubic_to_clr_rows(xs, torch.linspace(1, 0, N_OFF), IMG_W, args.ridge,
                                                   1e6, cp_frame=frame, n_strips=N_STRIPS,
                                                   min_span=args.min_span)
        hit = ok & ((cp_u < -args.margin) | (cp_u > 1 + args.margin)).any(-1)
        res = dict(clamp_hit_pct=float(100 * hit.float().mean()),
                   all_lanes=dict(current=summarize(mean_c, mx_c, ok),
                                  unbounded_fit=summarize(mean_u, mx_u, ok)),
                   clamped_lanes=dict(current=summarize(mean_c, mx_c, hit)))
        line = (f"{frame:8s} lanes the clamp changes: {res['clamp_hit_pct']:5.1f}% | all lanes mean px: "
                f"current {mean_c[ok].mean():6.2f}, unbounded {mean_u[ok].mean():6.2f}")
        if has_box:
            mean_b, mx_b, _ = lane_errors(geo, xs, frame, args.ridge, args.margin, args.min_span, True)
            res["all_lanes"]["box"] = summarize(mean_b, mx_b, ok)
            res["clamped_lanes"]["box"] = summarize(mean_b, mx_b, hit)
            line += f", box {mean_b[ok].mean():6.2f}"
            if hit.any():
                line += (f" | clamped lanes: current {mean_c[hit].mean():6.2f} px (worst row "
                         f"{mx_c[hit].mean():6.1f}), box {mean_b[hit].mean():6.2f} px (worst row "
                         f"{mx_b[hit].mean():6.1f})")
        elif hit.any():
            line += f" | clamped lanes: current {mean_c[hit].mean():6.2f} px (worst row {mx_c[hit].mean():6.1f})"
        print(line)
        out[frame] = res
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
