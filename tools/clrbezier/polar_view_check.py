"""Polar-view check on GT lanes, no training: do lanes become rays from the vanishing point?

Idea under test (thread, 2026-10-20): resample the image around the predicted vanishing point (VP) into polar
coordinates (angle theta around the VP, log radius). A lane that points at the VP is a vertical line at one theta in
that view, whatever its slant in the image, so flat side lanes would look like centre lanes. The idea only works if
the lanes the detectors lose (flat: slant > 4; short: vertical extent < 0.2) are mostly radial in their image.

Per image, the VP is the median pairwise intersection of lines x = a y + b fitted to the upper half of each lane (as
FovealLaneHead's vanishing_point). Every lane is measured twice: against the VP of all lanes, and against the VP of
the OTHER lanes only ("leave one out", the stricter test: a lane cannot pull the VP onto itself).

Per lane (network px, img_w x img_h):
    radial dev   length-weighted mean of the angle between the lane's direction (chords over an eighth of the
                 lane) and the ray from the VP to the chord's middle (0 = points at the VP; 90 = runs across it).
                 This is the lane's orientation as the polar view shows it.
    vertical dev the same against the image's vertical axis: the orientation the image shows (for contrast).
    theta spread max - min of the angle around the VP over the lane's points (0 for a straight radial lane; grows
                 with curvature that the polar view does not remove).
    above VP     share of the lane's points less than 2 px below the VP row (they leave the polar view's half plane).
Bins as curvelanes_error_breakdown.py: slant = sum|dx| / sum|dy| (<1, 1-2, 2-4, >4), vertical extent / img_h
(<0.2, 0.2-0.4, 0.4-0.7, >=0.7), and the "short or flat" group (extent < 0.4 or slant > 4).

Reading: if flat lanes have a small radial dev (say a median under 10 deg, against ~80 deg from vertical) the polar
view normalizes them and is worth building; if their radial dev is large (curving roads, lanes across the VP), it
does not, and the flat-lane problem is not one of orientation around the VP.

Usage (CPU, minutes):
    python tools/clrbezier/polar_view_check.py CONFIG [--images 2000] [--split train|val] [--batch 32]
        [--workers 8] [--seed 0] [--out work_dirs/polar_view/<config stem>.txt]
Lanes come from the config's pipeline as the head reads them (gt_points -> gt_polyline), as road_frame_ceiling.py
loads them (self-contained: needs only libs/ with the hier patches). Use a config whose head frame is the network input
(A-mono: hblrnet_a16_2d_mono_curvelanes_r34_bgrecipe.py). The train pipeline is augmented (flips, shifts, rotations,
crops); these keep straight lines straight and concurrent lines concurrent, so radial lanes stay radial."""
import argparse
import copy
import os
import os.path as osp
import sys
import time

import numpy as np

SLANT_BINS = ((0.0, 1.0, "slant <1"), (1.0, 2.0, "slant 1-2"), (2.0, 4.0, "slant 2-4"), (4.0, np.inf, "slant >4"))
EXT_BINS = ((0.0, 0.2, "extent <0.2"), (0.2, 0.4, "extent 0.2-0.4"), (0.4, 0.7, "extent 0.4-0.7"),
            (0.7, np.inf, "extent >=0.7"))


# ------------------------------------------------------------------ geometry
def vanishing_point(lanes_px, img_h):
    """Median pairwise intersection of lines x = a y + b fitted to the upper half of each lane -> (x, y) or None
    (as libs/clrbezier/foveal_lane.py vanishing_point)."""
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


def _line_angle(u, v):
    """Unsigned angle (deg, 0-90) between the lines along vectors u and v [..., 2]."""
    c = np.abs((u * v).sum(-1)) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


def lane_attrs(p, img_h):
    """Image attributes of one lane [M, 2] px: slant, vertical extent / img_h, length px."""
    d = np.diff(p, axis=0)
    dy = np.abs(d[:, 1]).sum()
    slant = float(np.abs(d[:, 0]).sum() / dy) if dy > 0 else np.inf
    return slant, float(np.ptp(p[:, 1]) / img_h), float(np.linalg.norm(d, axis=1).sum())


def polar_measures(p, vp, span=None):
    """radial dev, vertical dev, theta spread (deg), above-VP share of one lane [M, 2] px against vp (x, y).
    Directions are chords over span points (default M // 8, an eighth of the lane), so annotation noise of a px does
    not read as a turn."""
    v = np.asarray(vp, dtype=np.float64)
    k = max(1, min(len(p) - 1, span if span is not None else len(p) // 8))
    d = p[k:] - p[:-k]
    w = np.linalg.norm(d, axis=1)
    mid = 0.5 * (p[k:] + p[:-k])
    radial = float((w * _line_angle(d, mid - v)).sum() / max(w.sum(), 1e-9))
    vertical = float((w * _line_angle(d, np.array([0.0, 1.0]))).sum() / max(w.sum(), 1e-9))
    below = p[:, 1] >= v[1] + 2.0
    th = np.degrees(np.arctan2(p[below, 1] - v[1], p[below, 0] - v[0])) if below.any() else np.zeros(0)
    spread = float(np.ptp(th)) if len(th) >= 2 else float("nan")
    return radial, vertical, spread, float(1.0 - below.mean())


def analyse_image(lanes, img_h):
    """Rows (one per lane) of attributes and polar measures against the all-lane and leave-one-out VPs."""
    rows = []
    vp_all = vanishing_point(lanes, img_h)
    for i, p in enumerate(lanes):
        slant, ext, length = lane_attrs(p, img_h)
        r = dict(slant=slant, extent=ext, length=length, n_lanes=len(lanes))
        for tag, vp in (("all", vp_all),
                        ("loo", vanishing_point(lanes[:i] + lanes[i + 1:], img_h) if len(lanes) >= 3 else None)):
            if vp is not None:
                rad, ver, spread, above = polar_measures(p, vp)
                r.update({f"{tag}_radial": rad, f"{tag}_vertical": ver, f"{tag}_spread": spread,
                          f"{tag}_above": above})
        rows.append(r)
    return rows, vp_all is not None


# ------------------------------------------------------------------ report
def _q(a, q):
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=np.float64)
    return float(np.quantile(a, q)) if len(a) else float("nan")


def _table(rows, tag, groups, say):
    say(f"{'lanes':16s} {'n':>7s} {'% GT':>6s} {'vert dev':>9s} {'radial dev':>11s} {'<5deg':>6s} {'<10deg':>7s} "
        f"{'<20deg':>7s} {'radial p90':>10s} {'th spread':>10s} {'above VP':>9s}")
    total = sum(1 for r in rows if f"{tag}_radial" in r)
    for name, sel in groups:
        rs = [r for r in rows if f"{tag}_radial" in r and sel(r)]
        if not rs:
            say(f"{name:16s} {0:7d}")
            continue
        rad = np.array([r[f"{tag}_radial"] for r in rs])
        say(f"{name:16s} {len(rs):7d} {100 * len(rs) / max(total, 1):6.1f} "
            f"{_q([r[f'{tag}_vertical'] for r in rs], 0.5):9.1f} {_q(rad, 0.5):11.1f} {100 * (rad < 5).mean():6.1f} "
            f"{100 * (rad < 10).mean():7.1f} {100 * (rad < 20).mean():7.1f} {_q(rad, 0.9):10.1f} "
            f"{_q([r[f'{tag}_spread'] for r in rs], 0.5):10.1f} "
            f"{100 * np.mean([r[f'{tag}_above'] > 0.1 for r in rs]):9.1f}")


def report(images_rows, n_images, n_vp, say):
    rows = [r for rs in images_rows for r in rs]
    say(f"images {n_images}, with a VP from all lanes {n_vp} ({100 * n_vp / max(n_images, 1):.1f}%); lanes {len(rows)}")
    groups = [("all", lambda r: True)]
    groups += [(n, (lambda lo, hi: lambda r: lo <= r["slant"] < hi)(lo, hi)) for lo, hi, n in SLANT_BINS]
    groups += [(n, (lambda lo, hi: lambda r: lo <= r["extent"] < hi)(lo, hi)) for lo, hi, n in EXT_BINS]
    groups += [("short or flat", lambda r: r["extent"] < 0.4 or r["slant"] > 4),
               ("the rest", lambda r: not (r["extent"] < 0.4 or r["slant"] > 4))]
    for tag, title in (("all", "VP from all lanes of the image"),
                       ("loo", "VP from the OTHER lanes only (images with >= 3 lanes)")):
        n = sum(1 for r in rows if f"{tag}_radial" in r)
        say("")
        say(f"== {title}: {n} lanes measured")
        say("   vert dev / radial dev: median deg from the image vertical / from the ray to the VP (arc-length mean "
            "per lane); <5/<10/<20deg: % of lanes with radial dev below; th spread: median deg of angle around the "
            "VP covered by the lane; above VP: % of lanes with >10% of points above the VP row")
        _table(rows, tag, groups, say)


# ------------------------------------------------------------------ data
def _seed_all(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def ensure_gt_points(node):
    """Adds "gt_points" to the meta_keys of every PackCLRNetInputs in a (nested) config node; returns how many
    were changed (as build_pattern_codebook.py, inlined so this file needs nothing else from tools/)."""
    changed = 0
    if isinstance(node, dict):
        if node.get("type") == "PackCLRNetInputs":
            keys = list(node.get("meta_keys") or [])
            if "gt_points" not in keys:
                node["meta_keys"] = keys + ["gt_points"]
                changed += 1
        for v in node.values():
            changed += ensure_gt_points(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            changed += ensure_gt_points(v)
    return changed


def load_images(cfg, args, head):
    """GT polylines (px, bottom first) per image from the config's train or val pipeline (road_frame_ceiling.py)."""
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from torch.utils.data import DataLoader, Subset
    from libs.clrbezier.data_adapters import _lookup
    from libs.clrbezier.hblrnet import gt_polyline
    init_default_scope(cfg.get("default_scope", "mmdet"))
    dl_cfg = copy.deepcopy(cfg.train_dataloader if args.split == "train" else cfg.val_dataloader)
    dl_cfg["batch_size"], dl_cfg["num_workers"] = args.batch, 0
    dl_cfg["persistent_workers"] = False
    if ensure_gt_points(dl_cfg):
        print("added gt_points to the pipeline's meta_keys", flush=True)
    base = Runner.build_dataloader(dl_cfg, seed=args.seed)
    ds = base.dataset
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(ds), size=min(args.images, len(ds)), replace=False).tolist())

    def init(wid):
        _seed_all(args.seed + 1 + wid)
    _seed_all(args.seed)
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=base.collate_fn, worker_init_fn=init)
    cc = dict(head.get("curve_cfg") or {})
    img_w, img_h = int(head["img_w"]), int(head["img_h"])
    num_points, min_len = int(cc.get("gt_points", 48)), float(cc.get("min_gt_len_px", 8.0))
    scale = np.array([img_w - 1.0, img_h - 1.0])
    images, empty = [], 0
    for data in loader:
        for sample in data["data_samples"]:
            value, _ = _lookup(sample, ["gt_points"])
            empty += not value
            lanes = []
            for lane in value or []:
                p = gt_polyline(lane, img_w, img_h, num_points, min_len)
                if p is not None:
                    p = p[::-1].copy() if p[0, 1] < p[-1, 1] else p
                    lanes.append(p.astype(np.float64) * scale)
            images.append(lanes)
    if empty == len(images):
        raise SystemExit(f"no gt_points in any {args.split} sample: use --split train")
    return images, img_h


def analyse(images, img_h, say):
    tic, per_image, n_vp = time.time(), [], 0
    for k, lanes in enumerate(images):
        rows, has_vp = analyse_image(lanes, img_h)
        per_image.append(rows)
        n_vp += has_vp
        if (k + 1) % 500 == 0:
            print(f"  {k + 1} images, {time.time() - tic:.0f} s", flush=True)
    report(per_image, len(images), n_vp, say)
    return per_image


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--images", type=int, default=2000)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    head = cfg.model["bbox_head"]
    out = args.out or osp.join("work_dirs", "polar_view", osp.splitext(osp.basename(args.config))[0] + ".txt")
    os.makedirs(osp.dirname(out) or ".", exist_ok=True)
    images, img_h = load_images(cfg, args, head)
    lines = [f"config: {args.config}", f"split {args.split}, images loaded: {len(images)} (seed {args.seed})"]

    def say(s):
        print(s, flush=True)
        lines.append(s)
    for s in lines[:2]:
        print(s)
    analyse(images, img_h, say)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("report:", out)


if __name__ == "__main__":
    main()
