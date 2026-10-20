"""Look at the VP-polar view P on real images before run 4 (CPU, no weights needed). For a few images of the config's
train or val pipeline: the context view with the GT lanes and their vanishing point, and P around that point with the
same lanes mapped into it (polar_view.patch: PolarFrame, FovealLaneHead.polar_image). A lane through the vanishing
point is a vertical line of P whatever its slant in the image.

    python tools/clrbezier/polar_view_show.py configs/clrbezier/curvelanes/foveal_lane_polar_curvelanes_r34.py \
        --images 12 --split train --out work_dirs/polar_view_show

Writes <out>/polar_<k>.png (top: the context view, 800 x 320; bottom: P, 800 x 256; flat lanes, slant > 4, in red,
the others in green; the vanishing point in yellow) and <out>/polar_lanes.txt: per lane its slant in the image and
its column spread in P (columns between its leftmost and rightmost point, 1 column = 0.25 deg: 0 for a straight lane
through the point), with the medians of flat lanes and of the rest. P is centred on the GT vanishing point of the
image, clamped to the head's range, as in training; images without one are listed and skipped."""
import argparse
import copy
import os
import sys
import types

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())


def _seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def ensure_gt_points(node):
    """Adds "gt_points" to the meta_keys of every PackCLRNetInputs in a (nested) config node; returns how many
    were changed (as polar_view_check.py)."""
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


def load_samples(cfg, args, img_w, img_h, num_points, min_len):
    """(2x image [3, 2 img_h, 2 img_w] uint8, GT lanes in context px, bottom first) for args.images images."""
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from libs.clrbezier.data_adapters import _lookup
    from libs.clrbezier.hblrnet import gt_polyline
    init_default_scope(cfg.get("default_scope", "mmdet"))
    dl_cfg = copy.deepcopy(cfg.train_dataloader if args.split == "train" else cfg.val_dataloader)
    dl_cfg["batch_size"], dl_cfg["num_workers"], dl_cfg["persistent_workers"] = 1, 0, False
    ensure_gt_points(dl_cfg)
    base = Runner.build_dataloader(dl_cfg, seed=args.seed)
    ds = base.dataset
    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(ds), size=min(args.images, len(ds)), replace=False).tolist())
    scale = np.array([img_w - 1.0, img_h - 1.0])
    _seed_all(args.seed)
    for i in idx:
        data = base.collate_fn([ds[i]])
        img, sample = data["inputs"][0], data["data_samples"][0]
        value, _ = _lookup(sample, ["gt_points"])
        lanes = []
        for lane in value or []:
            p = gt_polyline(lane, img_w, img_h, num_points, min_len)
            if p is not None:
                p = p[::-1].copy() if p[0, 1] < p[-1, 1] else p
                lanes.append(p.astype(np.float64) * scale)
        yield i, img, lanes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="the polar config (foveal_lane_polar_curvelanes_r34.py: 2x pipeline, fovea_cfg)")
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="work_dirs/polar_view_show")
    args = ap.parse_args(argv)
    from mmengine.config import Config
    from libs.clrbezier.foveal_lane import FovealLaneHead, PolarFrame, vanishing_point
    cfg = Config.fromfile(args.config)
    head = cfg.model.bbox_head
    fc, cc = dict(head.get("fovea_cfg") or {}), dict(head.get("curve_cfg") or {})
    if fc.get("mode") != "polar":
        print(f"fovea_cfg.mode is {fc.get('mode')!r}: using the polar defaults for P", flush=True)
    img_w, img_h = int(head["img_w"]), int(head["img_h"])
    ns = types.SimpleNamespace(polar_ss=tuple(int(k) for k in fc.get("supersample", (2, 4))),
                               win_w=float(fc.get("polar_w", 800)), win_h=float(fc.get("polar_h", 256)),
                               img_w=img_w, img_h=img_h)                  # what FovealLaneHead.polar_image reads
    theta = tuple(float(a) for a in fc.get("theta_deg", (-10.0, 190.0)))
    r0 = float(fc.get("r0_px", 24.0))
    os.makedirs(args.out, exist_ok=True)
    lines, flat_spread, rest_spread, skipped = [], [], [], []
    for k, (i, img, lanes) in enumerate(load_samples(cfg, args, img_w, img_h, int(cc.get("gt_points", 48)),
                                                     float(cc.get("min_gt_len_px", 8.0)))):
        if tuple(img.shape[-2:]) != (2 * img_h, 2 * img_w):
            raise SystemExit(f"image {i}: {tuple(img.shape[-2:])}, expected the 2x input {(2 * img_h, 2 * img_w)} "
                             f"(use the polar config: dataset_curvelanes_clrbezier_2x.py)")
        v = vanishing_point(lanes, img_h)
        if v is None:
            skipped.append(i)
            continue
        vp = (min(max(v[0], 0.0), float(img_w)), min(max(v[1], -0.5 * img_h), float(img_h)))
        frame = PolarFrame(img_w, img_h, ns.win_w, ns.win_h, theta, r0, torch.tensor([vp], dtype=torch.float32))
        x2 = img[None].float()
        with torch.no_grad():
            pimg = FovealLaneHead.polar_image(ns, x2, frame)[0]
        ctx = torch.nn.functional.avg_pool2d(x2, 2)[0]
        top = np.ascontiguousarray(ctx.permute(1, 2, 0).clamp(0, 255).byte().numpy())
        bot = np.ascontiguousarray(pimg.permute(1, 2, 0).clamp(0, 255).byte().numpy())
        lines.append(f"image {i} (polar_{k}.png): vanishing point ({v[0]:.1f}, {v[1]:.1f}) px, P centred on "
                     f"({vp[0]:.1f}, {vp[1]:.1f})")
        for j, p in enumerate(lanes):
            slant = abs(p[-1, 0] - p[0, 0]) / max(abs(p[0, 1] - p[-1, 1]), 1e-6)
            colour = (0, 0, 255) if slant > 4 else (0, 255, 0)
            cv2.polylines(top, [np.round(p * 16).astype(np.int32)], False, colour, 2, cv2.LINE_AA, 4)
            seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
            s = np.linspace(0, seg[-1], 200)
            dense = np.stack([np.interp(s, seg, p[:, 0]), np.interp(s, seg, p[:, 1])], 1)
            q, th = frame.polar(torch.tensor(dense, dtype=torch.float32)[None])
            q, keep = q[0].numpy(), (th[0] <= frame.theta[1]).numpy()
            for a in range(len(q) - 1):
                if keep[a] and keep[a + 1]:
                    cv2.line(bot, tuple(np.round(q[a] * 16).astype(int)), tuple(np.round(q[a + 1] * 16).astype(int)),
                             colour, 2, cv2.LINE_AA, 4)
            spread = float(q[keep, 0].max() - q[keep, 0].min()) if keep.any() else float("nan")
            (flat_spread if slant > 4 else rest_spread).append(spread)
            lines.append(f"  lane {j}: slant {slant:5.2f}, column spread in P {spread:6.1f} "
                         f"({100 * keep.mean():.0f}% of it in P's angle range)")
        if 0 <= vp[0] < img_w and 0 <= vp[1] < img_h:
            cv2.circle(top, (int(round(vp[0])), int(round(vp[1]))), 4, (0, 255, 255), -1)
        pic = np.concatenate([top, np.full((6, img_w, 3), 255, np.uint8), bot], 0)
        cv2.imwrite(os.path.join(args.out, f"polar_{k}.png"), pic)
    med = (lambda v: f"{np.nanmedian(v):.1f}" if len(v) else "-")
    lines.append(f"\nmedian column spread in P: flat lanes (slant > 4) {med(flat_spread)} over {len(flat_spread)}, "
                 f"the rest {med(rest_spread)} over {len(rest_spread)}; skipped (no vanishing point): {skipped}")
    with open(os.path.join(args.out, "polar_lanes.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]) + f"\nwrote {args.out}/polar_*.png and polar_lanes.txt", flush=True)


if __name__ == "__main__":
    main()
