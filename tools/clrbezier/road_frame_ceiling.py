"""Road-frame check on GT lanes, no training (clrbezierlane/road_frame_design.md, section 5).

Question: can a lane's shape be predicted from the OTHER lanes of its image? If yes, a head that shares one road
frame between the lanes lets a lane with little paint inherit its shape from its neighbours.

Model, per image, in the network input (img_w x img_h): s = 1 - y / (img_h - 1) runs up the image; the frame is
x = F(u, s) = Bu(u) X Bs(s), a Bernstein polynomial of degree du in u (--du, 1) and ds in s (--ds, 3); X is
(du + 1) x (ds + 1) px. du = 1 by default: on a flat road, lanes that share the road's curvature are, at every image
row, at an x linear in their lateral offset, so the frame is two boundary curves and a lane is a blend of them.
A lane is u(s) = u0 + u1 s inside the frame (or u0 alone). The frame is fitted by alternating least squares, every
fitted lane a line of constant u (X is linear given the u's; each u by Gauss-Newton given X).

Leave one lane out: for every lane of an image with at least 3 lanes, the frame is fitted on the other lanes and
the held-out lane gets only its u: 1 number (const) or 2 (linear), over its own y range. Its error is the mean distance (px) from its GT points to that curve. Compared, per lane, with
    chord      the straight line through its own two end points (4 numbers),
    own cubic  a free cubic Bezier fitted to the lane alone (8 numbers): the ceiling of the current heads.
If "frame, linear" is well below "chord" (and near "own cubic") for most lanes, especially the slanted and short
ones, the neighbours carry the lane's shape and the road-frame head has something to use. If it is near or above
the chord, the lanes of an image do not share a usable frame and the head should not be built.

Report (stdout and <out>.txt): mean / median / p90 px and the share of lanes under 2 and 5 px for the four models,
then the same by chord slant |dx / dy| (<1, 1-2, 2-4, >4), by chord length (<100, 100-200, >200 px) and by lanes
per image (3, 4, 5+). Images with fewer than 3 lanes are counted and skipped.

On synthetic perspective roads (blrnet_tests/test_road_frame_ceiling.py: 40 images of 3-5 lanes, shared
curvature, 0.3 px noise) "frame, const" has median 0.85 px against 2.00 for the chord; with an independent bend per
lane it has 3.76 against 2.05, so the check tells the two apart. "frame, linear" has a long tail (a free u slope fitted on a
short lane), so read its median and p90 rather than its mean.

Usage:
    python tools/clrbezier/road_frame_ceiling.py CONFIG [--images 2000] [--batch 32] [--workers 8] [--seed 0]
        [--du 1] [--ds 3] [--out work_dirs/road_frame/<config stem>.txt]
Lanes come from the config's train pipeline (augmented, as the head reads them: gt_points, gt_polyline), as in
build_pattern_codebook.py."""
import argparse
import copy
import math
import os
import os.path as osp
import sys
import time

import numpy as np

SLANT_BINS = ((0.0, 1.0, "<1"), (1.0, 2.0, "1-2"), (2.0, 4.0, "2-4"), (4.0, np.inf, ">4"))
LEN_BINS = ((0.0, 100.0, "<100 px"), (100.0, 200.0, "100-200 px"), (200.0, np.inf, ">200 px"))
MODELS = ("chord", "own cubic", "frame, const", "frame, linear")


# ------------------------------------------------------------------ Bernstein bases
def bern(deg, t):
    """Bernstein basis of degree deg at t [...] -> [..., deg + 1]."""
    t = np.asarray(t, dtype=np.float64)[..., None]
    k = np.arange(deg + 1)
    comb = np.array([float(math.comb(deg, int(i))) for i in k])
    return comb * t ** k * (1.0 - t) ** (deg - k)


def dbern(deg, t):
    """d/dt of the Bernstein basis -> [..., deg + 1]."""
    t = np.asarray(t, dtype=np.float64)
    if deg == 0:
        return np.zeros(t.shape + (1,))
    b = bern(deg - 1, t)
    out = np.zeros(t.shape + (deg + 1,))
    out[..., :-1] -= deg * b
    out[..., 1:] += deg * b
    return out


# ------------------------------------------------------------------ frame and lanes
def frame_x(X, u, s):
    """x = Bu(u) X Bs(s) for u, s of one shape."""
    return np.einsum("...a,ab,...b->...", bern(X.shape[0] - 1, u), X, bern(X.shape[1] - 1, s))


def frame_dx_du(X, u, s):
    return np.einsum("...a,ab,...b->...", dbern(X.shape[0] - 1, u), X, bern(X.shape[1] - 1, s))


def fit_frame(lanes, params, du, ds=3, ridge=1e-3):
    """X [(du + 1), (ds + 1)] by least squares from lanes [(s, x)] and their u parameters (u0, u1)."""
    rows, xs = [], []
    for (s, x), p in zip(lanes, params):
        u = p[0] + p[1] * s
        rows.append(np.einsum("ka,kb->kab", bern(du, u), bern(ds, s)).reshape(len(s), -1))
        xs.append(x)
    a, b = np.concatenate(rows), np.concatenate(xs)
    n = a.shape[1]
    sol = np.linalg.solve(a.T @ a + ridge * np.eye(n), a.T @ b)
    return sol.reshape(du + 1, ds + 1)


def fit_u(X, s, x, p, linear, iters=12, damp=1e-6):
    """Gauss-Newton on a lane's (u0, u1) given the frame (u1 stays 0 unless linear)."""
    p = np.array(p, dtype=np.float64)
    for _ in range(iters):
        u = np.clip(p[0] + p[1] * s, -1.0, 2.0)
        r = frame_x(X, u, s) - x
        g = frame_dx_du(X, u, s)
        jac = np.stack([g, g * s], axis=1) if linear else g[:, None]
        step = np.linalg.solve(jac.T @ jac + damp * np.eye(jac.shape[1]), jac.T @ r)
        p[0] -= step[0]
        if linear:
            p[1] -= step[1]
        p[0] = float(np.clip(p[0], -1.0, 2.0))
        p[1] = float(np.clip(p[1], -2.0, 2.0))
    return p


def fit_scene(lanes, du, ds, rounds=15):
    """Frame X for lanes [(s, x)], each a line of constant u in it (alternating least squares)."""
    order = np.argsort([float(np.mean(x)) for _, x in lanes])
    params = [None] * len(lanes)
    for rank, i in enumerate(order):
        params[i] = np.array([rank / max(1, len(lanes) - 1), 0.0])
    X = fit_frame(lanes, params, du, ds)
    for _ in range(rounds):
        params = [fit_u(X, s, x, p, linear=False) for (s, x), p in zip(lanes, params)]
        X = fit_frame(lanes, params, du, ds)
    return X


# ------------------------------------------------------------------ distances and own fits
def point_to_polyline(p, poly):
    """Mean distance of points p [M, 2] to the polyline poly [K, 2]."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(np.einsum("mkd,kd->mk", p[:, None] - a[None], ab) / np.maximum((ab ** 2).sum(-1), 1e-12), 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return float(np.sqrt(((p[:, None] - proj) ** 2).sum(-1)).min(1).mean())


def own_cubic(p, ridge=1e-2, samples=100):
    """Free cubic Bezier through p [M, 2] (arc-length parameters, ridge toward the chord) -> curve [samples, 2]."""
    t = np.linspace(0.0, 1.0, len(p))
    basis = bern(3, t)
    c_lin = p[0] + (p[-1] - p[0]) * np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])[:, None]
    cps = np.linalg.solve(basis.T @ basis + ridge * np.eye(4), basis.T @ p + ridge * c_lin)
    return bern(3, np.linspace(0.0, 1.0, samples)) @ cps


def lane_errors(lanes_px, img_h, du=1, ds=3, samples=100):
    """Per lane of one image (>= 3 lanes, px [M, 2] bottom first): (errors dict, slant, chord length)."""
    sx = [(1.0 - p[:, 1] / (img_h - 1), p[:, 0]) for p in lanes_px]
    out = []
    for j, p in enumerate(lanes_px):
        others = [sx[i] for i in range(len(lanes_px)) if i != j]
        X = fit_scene(others, min(du, len(others) - 1), ds)
        s, x = sx[j]
        s_dense = np.linspace(s.min(), s.max(), samples)
        y_dense = (1.0 - s_dense) * (img_h - 1)
        err = {"chord": point_to_polyline(p, np.stack([p[0], p[-1]])),
               "own cubic": point_to_polyline(p, own_cubic(p))}
        best, q = np.inf, None
        for u_start in np.linspace(-0.5, 1.5, 9):           # the held-out lane may sit outside the others
            cand = fit_u(X, s, x, [u_start, 0.0], linear=False)
            r = float(np.mean((frame_x(X, np.full_like(s, cand[0]), s) - x) ** 2))
            if r < best:
                best, q = r, cand
        err["frame, const"] = point_to_polyline(p, np.stack([frame_x(X, np.full_like(s_dense, q[0]), s_dense),
                                                             y_dense], 1))
        q = fit_u(X, s, x, q, linear=True)
        u_dense = np.clip(q[0] + q[1] * s_dense, -1.0, 2.0)
        err["frame, linear"] = point_to_polyline(p, np.stack([frame_x(X, u_dense, s_dense), y_dense], 1))
        d = p[-1] - p[0]
        out.append((err, abs(d[0]) / max(abs(d[1]), 1e-6), float(np.hypot(*d))))
    return out


# ------------------------------------------------------------------ report
def summarize(rows, say, title):
    say(f"{title}: {len(rows)} lanes")
    if not rows:
        return
    say(f"  {'model':16s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'<2 px':>7s} {'<5 px':>7s}")
    for m in MODELS:
        e = np.array([r[0][m] for r in rows])
        say(f"  {m:16s} {e.mean():7.2f} {np.median(e):7.2f} {np.quantile(e, 0.9):7.2f} "
            f"{100 * (e < 2).mean():6.1f}% {100 * (e < 5).mean():6.1f}%")


def report(per_image, say, skipped):
    rows = [r for img in per_image for r in img]
    say(f"images with >= 3 lanes: {len(per_image)} (skipped with fewer: {skipped}); lanes: {len(rows)}")
    say("px = mean distance from a lane's GT points to the model's curve, network input")
    summarize(rows, say, "all lanes")
    for lo, hi, name in SLANT_BINS:
        summarize([r for r in rows if lo <= r[1] < hi], say, f"slant |dx/dy| {name}")
    for lo, hi, name in LEN_BINS:
        summarize([r for r in rows if lo <= r[2] < hi], say, f"chord {name}")
    for n, name in ((3, "3"), (4, "4")):
        summarize([r for img in per_image if len(img) == n for r in img], say, f"{name} lanes per image")
    summarize([r for img in per_image if len(img) >= 5 for r in img], say, "5+ lanes per image")


def analyse(images, img_h, say, min_lanes=3, du=1, ds=3):
    per_image, skipped, tic = [], 0, time.time()
    for k, lanes in enumerate(images):
        if len(lanes) < min_lanes:
            skipped += 1
            continue
        per_image.append(lane_errors(lanes, img_h, du, ds))
        if (k + 1) % 200 == 0:
            print(f"  {k + 1} images, {time.time() - tic:.0f} s", flush=True)
    report(per_image, say, skipped)
    return per_image


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
    """GT polylines (px, bottom first) of random augmented training images, grouped per image."""
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from torch.utils.data import DataLoader, Subset
    from libs.clrbezier.data_adapters import _lookup
    from libs.clrbezier.hblrnet import gt_polyline
    init_default_scope(cfg.get("default_scope", "mmdet"))
    dl_cfg = copy.deepcopy(cfg.train_dataloader)
    dl_cfg["batch_size"], dl_cfg["num_workers"] = args.batch, 0
    dl_cfg["persistent_workers"] = False
    if ensure_gt_points(dl_cfg):
        print("added gt_points to the train pipeline's meta_keys", flush=True)
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
    images = []
    for data in loader:
        for sample in data["data_samples"]:
            value, _ = _lookup(sample, ["gt_points"])
            lanes = []
            for lane in value or []:
                p = gt_polyline(lane, img_w, img_h, num_points, min_len)
                if p is not None:
                    p = p[::-1].copy() if p[0, 1] < p[-1, 1] else p
                    lanes.append(p.astype(np.float64) * scale)
            images.append(lanes)
    return images, img_h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--images", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--du", type=int, default=1, help="frame degree across the road (1)")
    ap.add_argument("--ds", type=int, default=3, help="frame degree along the road (3)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    head = cfg.model["bbox_head"]
    out = args.out or osp.join("work_dirs", "road_frame", osp.splitext(osp.basename(args.config))[0] + ".txt")
    os.makedirs(osp.dirname(out) or ".", exist_ok=True)
    images, img_h = load_images(cfg, args, head)
    lines = [f"config: {args.config}", f"images loaded: {len(images)} (seed {args.seed}); frame degree du {args.du}, "
             f"ds {args.ds}"]

    def say(s):
        print(s, flush=True)
        lines.append(s)
    for s in lines[:2]:
        print(s)
    analyse(images, img_h, say, du=args.du, ds=args.ds)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("report:", out)


if __name__ == "__main__":
    main()
