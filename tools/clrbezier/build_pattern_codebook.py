"""Pattern codebook for HBLRHead pattern_cfg (lanes as pose x pattern, clrbezierlane/pose_pattern_design.md).

Reads augmented training lanes through the config's own train pipeline (the lanes the head trains on), fits each
lane with a cubic pinned to its ends and reads its pattern in its own chord frame (libs/clrbezier/pattern.py);
k-means gives the lane prototypes. The same for each lane's equal arc pieces (hier_cfg.segments, 3 by default)
gives the segment prototypes. Prints how compact the patterns are, then writes the codebook to
pattern_cfg.codebook (or --out) and the report next to it (.txt).

    python tools/clrbezier/build_pattern_codebook.py CONFIG [--images 20000] [--batch 32] [--workers 8]
        [--seed 0] [--out PATH]

Report, per level (lanes, segments). px = mean distance between the pinned cubic and the cubic decoded from a
pattern with the true pose (25 points along the curve):
- floor: the pinned cubic against the GT polyline (the cubic's own error, not a pattern error);
- straight: every curve as its chord (pattern (0, 1/3, 0, 2/3));
- K = 8 / 16 / 32 / 64 prototypes, no residual;
- usage of the codebook written: the largest prototype's share and the effective number of prototypes
  (exp of the usage entropy).
Reading it: if K = 32 removes little of the straight error, lane bends are not compact in the chord frame and
pattern_cfg is unlikely to help. A largest share far above 1 / K means the codebook is dominated by straight lanes.
"""
import argparse
import copy
import math
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch

_HERE = osp.dirname(osp.abspath(__file__))
_REPO = osp.dirname(osp.dirname(_HERE))
if _REPO not in sys.path:      # libs.* when run as tools/clrbezier/...
    sys.path.append(_REPO)

KS = (8, 16, 32, 64)
EVAL_POINTS = 25


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def load_lanes(cfg, args, head):
    """GT polylines of random augmented training images [L, M, 2] (normalized, bottom first), as the head reads
    them (gt_polyline with its img_w, img_h, gt_points, min_gt_len_px)."""
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from torch.utils.data import DataLoader, Subset
    from libs.clrbezier.data_adapters import _lookup
    from libs.clrbezier.hblrnet import gt_polyline
    init_default_scope(cfg.get("default_scope", "mmdet"))
    dl_cfg = copy.deepcopy(cfg.train_dataloader)
    dl_cfg["batch_size"], dl_cfg["num_workers"] = args.batch, 0
    dl_cfg["persistent_workers"] = False
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
    lanes, images, tic = [], 0, time.time()
    for data in loader:
        for sample in data["data_samples"]:
            images += 1
            value, _ = _lookup(sample, ["gt_points"])
            for lane in value or []:
                p = gt_polyline(lane, img_w, img_h, num_points, min_len)
                if p is not None:
                    lanes.append(p[::-1].copy() if p[0, 1] < p[-1, 1] else p)
    print(f"loaded {images} augmented images, {len(lanes)} lanes in {time.time() - tic:.0f} s", flush=True)
    return np.stack(lanes), images


def curve_error(fit_px, pattern):
    """Mean px distance between cubics fit_px [L, 4, 2] and the cubics with the same poses and patterns [L, 4]."""
    from libs.clrbezier.blrnet import curve_points
    from libs.clrbezier.pattern import from_canon
    t = torch.linspace(0.0, 1.0, EVAL_POINTS, dtype=fit_px.dtype)
    dec = from_canon(fit_px[:, 0], fit_px[:, 3], pattern)
    return (curve_points(fit_px, t) - curve_points(dec, t)).norm(dim=-1).mean(-1)


def floor_error(fit_px, poly_px):
    """Mean px distance of the GT polyline points to their pinned cubic (dense)."""
    from libs.clrbezier.blrnet import curve_points
    from libs.clrbezier.hblrnet import _point_to_polyline
    dense = curve_points(fit_px, torch.linspace(0.0, 1.0, 200, dtype=fit_px.dtype))
    return torch.cat([_point_to_polyline(poly_px[i:i + 512], dense[i:i + 512]) for i in range(0, len(fit_px), 512)]
                     ).mean(-1)


def level_report(name, fit_px, k_cfg, seed, say, floor=None):
    """Prints the compactness table of one level and returns (prototypes [k_cfg, 4], sigma, usage counts)."""
    from libs.clrbezier.pattern import STRAIGHT, kmeans, proto_sigma, to_canon
    pat = to_canon(fit_px)
    chord = (fit_px[:, 3] - fit_px[:, 0]).norm(dim=-1)
    say(f"\n{name}: {len(pat)} curves, chord length px p10 / p50 / p90 "
        f"{chord.quantile(0.1):.0f} / {chord.median():.0f} / {chord.quantile(0.9):.0f}")
    say(f"  pattern mean (a1, b1, a2, b2) {[round(float(v), 3) for v in pat.mean(0)]}, std "
        f"{[round(float(v), 3) for v in pat.std(0)]}")

    def row(label, err):
        say(f"  {label:22s} mean px {err.mean():6.2f}   p50 {err.median():6.2f}   p90 {err.quantile(0.9):6.2f}   "
            f"<= 2 px {100 * (err <= 2).float().mean():5.1f}%   <= 5 px {100 * (err <= 5).float().mean():5.1f}%")
    if floor is not None:
        row("floor (cubic vs GT)", floor)
    straight = curve_error(fit_px, torch.tensor(STRAIGHT, dtype=pat.dtype).expand_as(pat))
    row("straight chord", straight)
    keep = None
    for k in sorted(set(KS) | {k_cfg}):
        centers, assign = kmeans(pat, k, seed=seed)
        err = curve_error(fit_px, centers[assign])
        gain = 100.0 * (1.0 - float(err.mean()) / max(1e-9, float(straight.mean())))
        row(f"K = {k}" + (" (codebook)" if k == k_cfg else ""), err)
        say(f"  {'':22s} removes {gain:5.1f}% of the straight-chord error")
        if k == k_cfg:
            keep = (centers, proto_sigma(centers), torch.bincount(assign, minlength=k))
    counts = keep[2].float()
    share = counts / counts.sum()
    ent = -(share[share > 0] * share[share > 0].log()).sum()
    say(f"  codebook usage (K = {k_cfg}): largest share {100 * share.max():.1f}% (uniform {100.0 / k_cfg:.1f}%), "
        f"effective prototypes {math.exp(float(ent)):.1f}; soft-target sigma {keep[1]:.4f}")
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--images", type=int, default=20000, help="random training images (augmented)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: the config's pattern_cfg.codebook")
    args = ap.parse_args(argv)
    from mmengine.config import Config
    from libs.clrbezier.pattern import arc_pieces, fit_cubic_pinned
    cfg = Config.fromfile(args.config)
    head = cfg.model["bbox_head"]
    pc = dict(head.get("pattern_cfg") or {})
    out = args.out or pc.get("codebook")
    if not out:
        raise SystemExit("no --out and no pattern_cfg.codebook in the config")
    k_lane, k_seg = int(pc.get("lane_protos", 32)), int(pc.get("seg_protos", 32))
    segs = int(dict(head.get("hier_cfg") or {}).get("segments", 3))
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    _seed_all(args.seed)
    lanes, images = load_lanes(cfg, args, head)
    scale = torch.tensor([float(head["img_w"] - 1), float(head["img_h"] - 1)], dtype=torch.float64)
    poly = torch.from_numpy(lanes).double() * scale                                 # [L, M, 2] px
    fit = fit_cubic_pinned(poly)
    pieces = arc_pieces(poly, segs).reshape(-1, 4, 2)
    say(f"config {args.config}; {images} augmented training images (seed {args.seed}); image {head['img_w']} x "
        f"{head['img_h']} px")
    lane_keep = level_report("lanes", fit, k_lane, args.seed, say, floor=floor_error(fit, poly))
    seg_keep = level_report(f"segments ({segs} equal arc pieces per lane)", pieces, k_seg, args.seed, say)
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    torch.save(dict(lane=lane_keep[0].float(), lane_sigma=float(lane_keep[1]), lane_count=lane_keep[2],
                    seg=seg_keep[0].float(), seg_sigma=float(seg_keep[1]), seg_count=seg_keep[2],
                    meta=dict(config=args.config, images=images, lanes=len(lanes), segments=segs, seed=args.seed,
                              img_w=int(head["img_w"]), img_h=int(head["img_h"]))), out)
    with open(osp.splitext(out)[0] + ".txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out} and {osp.splitext(out)[0]}.txt")


if __name__ == "__main__":
    main()
