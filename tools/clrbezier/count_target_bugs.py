"""Count CurveLanes lanes hit by the two CLRBezierLane target bugs. Read-only: changes nothing.

Bug 1  PackCLRNetInputs.convert_targets writes hstack(outside, inside) rows, so a lane
       that leaves the image anywhere above its first visible row gets its GT shifted up
       by the number of outside rows and a wrong start_y.
Bug 2  cp_frame="anchored": head.py reparam_cp_to_frame with ratio > 1 extrapolates the
       GT cubic; for curved lanes the target leaves the state's [-0.5, 1.5] range.
       Reported as "susceptible" lanes: whether a matched prediction actually starts
       lower than the GT depends on training, so this is an upper bound per start gap.

Run from the repo root (the folder holding libs/ and configs/):

    # geometry after crop + resize only (no mmdet needed)
    python count_target_bugs.py configs/clrbezier/curvelanes/dataset_curvelanes_clrernet.py

    # also through the real training augmentation, K random draws per image (needs mmdet env)
    python count_target_bugs.py configs/clrbezier/curvelanes/dataset_curvelanes_clrernet.py --aug 3

Options: --limit N (first N images), --workers W, --out stats.json
"""
import argparse
import ast
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

IMG_W, IMG_H, N_OFF = 800, 320, 72
N_STRIPS = N_OFF - 1
ROW_YS = np.arange(IMG_H, -1, -IMG_H / N_STRIPS)  # PackCLRNetInputs.offsets_ys
CROP_OFFSETS = {(1440, 2560): 640, (660, 1570): 180, (720, 1280): 368}  # CurvelanesDataset
MAX_LANES = 16
MARGIN = 0.5
GAPS = (0.2, 0.4, 1.0)  # pred_start - gt_start values checked for bug 2


def _load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LU = _GEO = _TORCH = None


def _init_modules(repo):
    global _LU, _GEO, _TORCH
    _LU = _load_file_module("_lane_utils", os.path.join(repo, "libs/utils/lane_utils.py"))
    try:
        import torch
        _TORCH = torch
        _GEO = _load_file_module("_geometry", os.path.join(repo, "libs/clrbezier/geometry.py"))
    except ImportError:
        _TORCH = _GEO = None


def read_config_values(path):
    """data_root and train data_list from the dataset config, without mmengine."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("data_root",):
                env[name] = ast.literal_eval(node.value)
    data_root = env["data_root"]
    return data_root + "/train/", data_root + "/train/train_seg.txt"


def image_shape(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        return h, w
    except ImportError:
        import cv2
        return cv2.imread(path).shape[:2]


def read_lines_txt(path, offset_y):
    """CurvelanesDataset._read_lines_txt."""
    shapes = []
    with open(path) as f:
        for line in f:
            vals = line.strip().split(" ")
            coords = []
            for i in range(len(vals) // 2):
                coords += [float(vals[2 * i]), float(vals[2 * i + 1]) + offset_y]
            if len(coords) > 3:
                shapes.append(coords)
    return shapes


def is_sorted(lane):
    y = np.array(lane[1::2])
    return bool(np.all(y[1:] < y[:-1]))


def cut_unsorted(lane):
    out, prev = [], 1e8
    for x, y in zip(lane[0::2], lane[1::2]):
        if y < prev:
            out += [x, y]
            prev = y
    return out


def row_xs(lane):
    """lane_utils.sample_lane without the inside/outside split: x per row, bottom -> top."""
    from scipy.interpolate import InterpolatedUnivariateSpline
    pts = np.array([lane[0::2], lane[1::2]]).T
    x, y = pts[:, 0], pts[:, 1]
    interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(pts) - 1))
    m = (ROW_YS >= y.min()) & (ROW_YS <= y.max())
    if not m.any():
        return np.zeros(0)
    extrap = np.polyfit(pts[:2, 1], pts[:2, 0], deg=1)
    return np.hstack((np.polyval(extrap, ROW_YS[ROW_YS > y.max()]), interp(ROW_YS[m])))


def analyse_lanes(lanes):
    """lanes: list of flat [x0, y0, ...] in 800x320 input pixels, after augmentation."""
    res = dict(lanes=0, bug1=0, shift_rows=[], fixed_rows=[], ok_lanes=0)
    lanes = [l for l in lanes if len(l) > 2]  # convert_targets filter
    for lane in lanes[:MAX_LANES]:
        try:
            out_x, in_x = _LU.sample_lane(lane, ROW_YS, IMG_W)
        except (AssertionError, Exception):
            continue
        if len(in_x) <= 1:
            continue
        res["lanes"] += 1
        xs = row_xs(lane)
        inside = (xs >= 0) & (xs < IMG_W)
        first = int(np.argmax(inside))
        if np.any(~inside[first:]):
            res["bug1"] += 1
            res["shift_rows"].append(len(out_x) - first)
        row = np.full(N_OFF, -1e5)
        row[: len(xs)] = xs
        res["fixed_rows"].append(row)
    return res


def bug2_susceptible(fixed_rows):
    """For each lane, does the anchored CP target leave [-M, 1+M] for a pred start GAP lower?"""
    if _GEO is None or not fixed_rows:
        return None
    torch = _TORCH
    xs = torch.tensor(np.stack(fixed_rows), dtype=torch.float32)
    ys = torch.linspace(1, 0, N_OFF)
    cp, ok = _GEO.fit_global_cubic_to_clr_rows(xs, ys, IMG_W, 1e-2, MARGIN, cp_frame="anchored")
    valid = (xs >= 0) & (xs < IMG_W)
    s_gt = (ys[None] * valid).max(-1).values.clamp_min(1e-3)
    out = {}
    for gap in GAPS:
        s_pred = (s_gt + gap).clamp(1.0 / N_STRIPS, 1.0)
        tgt = _GEO.reparam_cp_to_frame(cp, (s_pred / s_gt).clamp(0.2, 5.0))
        over = ((tgt - tgt.clamp(-MARGIN, 1 + MARGIN)).abs() * (IMG_W - 1)).mean(-1)
        out[gap] = (((over > 0) & ok).sum().item(), (over * 0.1)[ok].tolist())
    return out


def _resize(lanes, crop_hw):
    sx, sy = IMG_W / crop_hw[1], IMG_H / crop_hw[0]
    return [[v * (sx if i % 2 == 0 else sy) for i, v in enumerate(l)] for l in lanes]


_AUG = None


def _init_worker(repo, cfg_path, aug):
    global _AUG
    sys.path.insert(0, repo)
    _init_modules(repo)
    if aug:
        from mmengine.config import Config
        from libs.datasets.common import LaneAlbumentation
        cfg = Config.fromfile(cfg_path)
        _AUG = LaneAlbumentation(cfg.train_al_pipeline, cut_unsorted=True)


def process(args):
    img_prefix, rel_img, n_aug = args
    img_path = os.path.join(img_prefix, rel_img)
    hw = image_shape(img_path)
    offset = CROP_OFFSETS.get(tuple(hw))
    rec = dict(img=rel_img, skipped=offset is None)
    if offset is None:
        return rec
    crop_hw = (hw[0] - offset, hw[1])
    lanes = read_lines_txt(img_path.replace(".jpg", ".lines.txt"), -offset)
    rec["raw_unsorted"] = sum(not is_sorted(l) for l in lanes)
    rec["over_max"] = len([l for l in lanes if len(l) > 2]) > MAX_LANES
    # no augmentation: the val geometry (crop + resize); unsorted lanes cut as training does
    plain = _resize([l if is_sorted(l) else cut_unsorted(l) for l in lanes], crop_hw)
    r = analyse_lanes(plain)
    rec["plain"] = {k: r[k] for k in ("lanes", "bug1", "shift_rows")}
    rec["plain_bug2"] = bug2_susceptible(r["fixed_rows"])
    rec["aug"] = []
    for _ in range(n_aug):
        data = dict(img=np.zeros((crop_hw[0], crop_hw[1], 3), np.uint8), gt_points=lanes)
        data = _AUG(data)
        r = analyse_lanes(data["gt_points"])
        rec["aug"].append({k: r[k] for k in ("lanes", "bug1", "shift_rows")})
    return rec


def summarize(recs, n_aug):
    recs = [r for r in recs if not r["skipped"]]
    s = dict(images=len(recs), skipped_unknown_resolution=None)
    s["images_with_raw_unsorted_lane"] = sum(r["raw_unsorted"] > 0 for r in recs)
    s["images_over_max_lanes"] = sum(r["over_max"] for r in recs)

    def block(items):
        lanes = sum(i["lanes"] for i in items)
        bug1 = sum(i["bug1"] for i in items)
        imgs = sum(i["bug1"] > 0 for i in items)
        shifts = np.array([x for i in items for x in i["shift_rows"]] or [0])
        p = imgs / max(1, len(items))
        return dict(valid_lanes=lanes, bug1_lanes=bug1, bug1_lane_pct=100 * bug1 / max(1, lanes),
                    bug1_images=imgs, bug1_image_pct=100 * p,
                    shift_rows_median=float(np.median(shifts)), shift_rows_p90=float(np.percentile(shifts, 90)),
                    shift_rows_max=int(shifts.max()),
                    batches_with_bug1_pct_b24=100 * (1 - (1 - p) ** 24),
                    batches_with_bug1_pct_b32=100 * (1 - (1 - p) ** 32))
    s["no_aug"] = block([r["plain"] for r in recs])
    if n_aug:
        s["with_train_aug"] = block([a for r in recs for a in r["aug"]])
    if recs and recs[0]["plain_bug2"] is not None:
        b2 = {}
        lanes = s["no_aug"]["valid_lanes"]
        for gap in GAPS:
            n = sum(r["plain_bug2"][gap][0] for r in recs if r["plain_bug2"])
            vals = np.array([v for r in recs if r["plain_bug2"] for v in r["plain_bug2"][gap][1]] or [0.0])
            b2[f"pred_start_lower_by_{gap}"] = dict(
                lanes_out_of_range=n, pct=100 * n / max(1, lanes),
                cp_loss_floor_x0p1_mean=float(vals.mean()), p99=float(np.percentile(vals, 99)),
                max=float(vals.max()))
        s["bug2_susceptible_no_aug"] = b2
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="CurveLanes dataset config (for data_root / train list)")
    ap.add_argument("--aug", type=int, default=0, help="random training-augmentation draws per image")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--data-list", default=None, help="override the train list")
    ap.add_argument("--img-prefix", default=None)
    ap.add_argument("--out", default="curvelanes_target_bug_stats.json")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    img_prefix, data_list = read_config_values(args.config)
    img_prefix = args.img_prefix or img_prefix
    data_list = args.data_list or data_list
    with open(data_list) as f:
        imgs = [ln.strip().split(" ")[0].lstrip("/") for ln in f if ln.strip()]
    if args.limit:
        imgs = imgs[: args.limit]
    jobs = [(img_prefix, im, args.aug) for im in imgs]
    init = (repo, os.path.abspath(args.config), args.aug > 0)
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker, initargs=init) as pool:
            recs = []
            for i, r in enumerate(pool.imap_unordered(process, jobs, chunksize=32)):
                recs.append(r)
                if (i + 1) % 5000 == 0:
                    print(f"{i + 1}/{len(jobs)}", flush=True)
    else:
        _init_worker(*init)
        recs = [process(j) for j in jobs]
    summary = summarize(recs, args.aug)
    summary["skipped_unknown_resolution"] = sum(r["skipped"] for r in recs)
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary,
                       bug1_images_no_aug=[r["img"] for r in recs if not r["skipped"] and r["plain"]["bug1"]]),
                  f, indent=1)
    print(f"wrote {args.out} (includes the list of affected images)")


if __name__ == "__main__":
    main()
