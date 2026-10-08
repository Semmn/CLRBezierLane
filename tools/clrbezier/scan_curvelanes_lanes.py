"""Scan CurveLanes lane annotations for shapes that a row-based target handles badly.

Each lane is read from its .lines.txt, cropped like CurvelanesDataset and scaled
to the 800x320 network input, then checked for:

  zigzag      the x of consecutive points (in file order) reverses direction by
              more than --zigzag-px. A lane whose points were sorted by y after it
              bent back looks like this: the row-sampled target jumps between the
              two branches.
  unsorted    y is not strictly decreasing in file order, so the training
              pipeline retries augmentation 30 times and then cuts the lane at the
              first reversal (cut_unsorted).
  flat        some segment is flatter than --flat-slope (|dx/dy| in network px),
              i.e. nearly horizontal: a few rows carry a long x run.
  short       the lane covers fewer than --min-rows of the 72 target rows.
  too_many    the image has more lanes than max_lanes (16); extra ones are dropped.

Usage:
  # whole training list: rates and the worst images
  python tools/clrbezier/scan_curvelanes_lanes.py DATA_ROOT/train train/train_seg.txt --out scan.csv
  # images from the batches LossSpikeHook flagged, compared with the base rates
  python tools/clrbezier/scan_curvelanes_lanes.py DATA_ROOT/train train/train_seg.txt \
      --spikes work_dirs/.../loss_spikes_rank*.jsonl
"""
import argparse
import csv
import glob
import json
import os.path as osp
from collections import Counter

import numpy as np

CROP = {(2560, 1440): 640, (1570, 660): 180, (1280, 720): 368}   # (w, h) -> top offset
IMG_W, IMG_H, N_ROWS = 800, 320, 72
FLAGS = ("zigzag", "unsorted", "flat", "short", "too_many")


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def read_lanes(path):
    lanes = []
    with open(path) as f:
        for line in f:
            v = [float(t) for t in line.split()]
            if len(v) >= 4:
                lanes.append(np.asarray(v[: len(v) // 2 * 2], dtype=np.float64).reshape(-1, 2))
    return lanes


def lane_flags(pts, args):
    """pts: [K, 2] in network px (file order). Returns (flags, stats)."""
    flags, x, y = set(), pts[:, 0], pts[:, 1]
    order = np.argsort(-y, kind="stable")           # bottom -> top, as the target is built
    xs, ys = x[order], y[order]
    dx = np.diff(xs)
    reversals = 0
    if np.any(np.diff(y) >= 0):
        # the pipeline cuts such a lane at its first y reversal (cut_unsorted),
        # so its zig-zag never reaches the target
        flags.add("unsorted")
    else:
        big = dx[np.abs(dx) > args.zigzag_px]
        reversals = int(np.sum(np.sign(big[1:]) != np.sign(big[:-1]))) if big.size > 1 else 0
        if reversals:
            flags.add("zigzag")
    dys = np.abs(np.diff(ys))
    slope = np.abs(dx) / np.maximum(dys, 1e-6)
    seg_ok = np.abs(dx) > 2.0                        # ignore sub-pixel jitter
    max_slope = float(slope[seg_ok].max()) if seg_ok.any() else 0.0
    if max_slope > args.flat_slope:
        flags.add("flat")
    vis = (ys >= 0) & (ys <= IMG_H)
    rows = (ys[vis].max() - ys[vis].min()) / (IMG_H / (N_ROWS - 1)) if vis.sum() > 1 else 0.0
    if rows < args.min_rows:
        flags.add("short")
    return flags, dict(reversals=reversals, max_slope=max_slope, rows=float(rows))


def scan_image(root, rel, args):
    img_path = osp.join(root, rel)
    anno = osp.splitext(img_path)[0] + ".lines.txt"
    try:
        w, h = image_size(img_path)
    except OSError:
        return None
    off = CROP.get((w, h))
    if off is None or not osp.exists(anno):
        return None
    sx, sy = IMG_W / w, IMG_H / (h - off)
    lanes = read_lanes(anno)
    per, flags = [], Counter()
    for pts in lanes:
        p = np.stack([pts[:, 0] * sx, (pts[:, 1] - off) * sy], 1)
        f, s = lane_flags(p, args)
        flags.update(f)
        per.append(s)
    if len(lanes) > args.max_lanes:
        flags["too_many"] += 1
    score = (3 * flags["zigzag"] + 2 * flags["flat"] + flags["unsorted"]
             + flags["short"] + flags["too_many"])
    return dict(image=rel, lanes=len(lanes), score=score,
                max_reversals=max((s["reversals"] for s in per), default=0),
                max_slope=round(max((s["max_slope"] for s in per), default=0.0), 1),
                **{k: flags[k] for k in FLAGS})


def rates(rows):
    n_img = max(1, len(rows))
    return {k: sum(r[k] > 0 for r in rows) / n_img for k in FLAGS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="directory the list's image paths are relative to")
    ap.add_argument("list", help="image list, e.g. train/train_seg.txt (first column used)")
    ap.add_argument("--spikes", nargs="*", default=None, help="LossSpikeHook jsonl files")
    ap.add_argument("--out", default="curvelanes_lane_scan.csv")
    ap.add_argument("--zigzag-px", type=float, default=8.0)
    ap.add_argument("--flat-slope", type=float, default=6.0)
    ap.add_argument("--min-rows", type=float, default=3.0)
    ap.add_argument("--max-lanes", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="scan only the first N images")
    args = ap.parse_args()

    with open(args.list) as f:
        names = [ln.split()[0].lstrip("/") for ln in f if ln.strip()]
    if args.limit:
        names = names[: args.limit]
    rows = []
    for i, rel in enumerate(names):
        r = scan_image(args.root, rel, args)
        if r is not None:
            rows.append(r)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(names)}", flush=True)
    rows.sort(key=lambda r: -r["score"])
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    base = rates(rows)
    print(f"scanned {len(rows)} images -> {args.out}")
    print("share of images with at least one flagged lane:")
    for k in FLAGS:
        print(f"  {k:9s} {100 * base[k]:6.2f}%")
    print("worst 15:")
    for r in rows[:15]:
        print("  ", {k: r[k] for k in ("image", "lanes", "score", "max_reversals", "max_slope") + FLAGS})

    if args.spikes:
        by_name = {r["image"]: r for r in rows}
        spike_rows, recs = [], []
        for path in sorted(set(p for s in args.spikes for p in glob.glob(s))):
            with open(path) as f:
                recs += [json.loads(ln) for ln in f if ln.strip()]
        for rec in recs:
            for im in rec["images"]:
                key = im["name"].lstrip("/")
                hit = by_name.get(key) or next((r for k, r in by_name.items() if key.endswith(k)), None)
                if hit:
                    spike_rows.append(hit)
        sp = rates(spike_rows)
        print(f"\nspike batches: {len(recs)} records, {len(spike_rows)} images matched")
        print(f"  {'flag':9s} {'spikes':>8s} {'all':>8s}  ratio")
        for k in FLAGS:
            ratio = sp[k] / base[k] if base[k] > 0 else float("nan")
            print(f"  {k:9s} {100 * sp[k]:7.2f}% {100 * base[k]:7.2f}%  {ratio:5.2f}x")
        worst = sorted(spike_rows, key=lambda r: -r["score"])[:15]
        print("worst images inside spike batches:")
        for r in worst:
            print("  ", {k: r[k] for k in ("image", "lanes", "score", "max_reversals", "max_slope") + FLAGS})


if __name__ == "__main__":
    main()
