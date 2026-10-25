"""Draw near misses from score_shift_check.py's near_misses.csv over their images: the near miss (red), its lane
(green), the top-scored copy of that lane if there is one (blue, any score), the other GT lanes (thin white).

    python tools/clrbezier/draw_near_misses.py work_dirs/score_shift_check/<config>/near_misses.csv \
        [--which above_missed] [--n 40] [--seed 0] [--out-dir DIR] [--width 1280]

--which: above_missed (default: scored above the threshold and the lane has no copy above it, the 990 of the o2o
run), above, below, all. The sample is random (seeded), not the worst cases. Writes one image per near miss
(NNN_<image>_q<query>.jpg, with score, IoU and the offsets in the corner), contact sheets of 8 (sheet_NN.jpg) and
picked.csv. A missing image is drawn on grey.
"""
from __future__ import annotations

import argparse
import csv
import os
import os.path as osp
import random

import cv2
import numpy as np

WHICH = dict(above_missed=lambda r: r["above"] and not r["lane_hit_by_copy"], above=lambda r: r["above"],
             below=lambda r: not r["above"], all=lambda r: True)
RED, GREEN, BLUE, WHITE = (0, 0, 255), (0, 200, 0), (255, 120, 0), (255, 255, 255)


def _bool(v):
    return str(v).strip().lower() in ("true", "1")


def _pts(text):
    v = [float(t) for t in str(text).split()]
    return np.array(v, dtype=np.float64).reshape(-1, 2)


def read_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["above"], r["lane_hit_by_copy"] = _bool(r["above"]), _bool(r["lane_hit_by_copy"])
    return rows


def gt_lanes(image):
    """CurvelanesMetric.parse_anno: the .lines.txt next to the image, original px."""
    path = image.replace(".jpg", ".lines.txt")
    if not osp.exists(path):
        return []
    with open(path) as f:
        return [p for p in (_pts(line) for line in f) if len(p) >= 2]


def _line(im, pts, color, t):
    if len(pts) >= 2:
        cv2.polylines(im, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, color, t, cv2.LINE_AA)


def _f(r, k, fmt="{:.1f}"):
    try:
        v = float(r[k])
    except (KeyError, ValueError):
        return "-"
    return "-" if v != v else fmt.format(v)


def draw(r, width):
    h, w = int(r["ori_h"]), int(r["ori_w"])
    im = cv2.imread(r["image"])
    found = im is not None
    if not found:
        im = np.full((h, w, 3), 96, np.uint8)
    scale = width / im.shape[1]
    t = max(1, int(round(2 / scale)))
    lanes = gt_lanes(r["image"])
    g = int(r["gt"])
    for k, ln in enumerate(lanes):
        if k != g:
            _line(im, ln, WHITE, max(1, t // 2))
    if g < len(lanes):
        _line(im, lanes[g], GREEN, t)
    if r.get("copy_points"):
        _line(im, _pts(r["copy_points"]), BLUE, t)
    _line(im, _pts(r["points"]), RED, t)
    im = cv2.resize(im, (width, int(round(im.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    text = [f"near miss (red) score {_f(r, 'score', '{:.2f}')} IoU {_f(r, 'iou', '{:.2f}')} lat {_f(r, 'lat_px')} "
            f"eval px; d bot / mid / top {_f(r, 'd_bottom')} / {_f(r, 'd_middle')} / {_f(r, 'd_top')}; "
            f"one side {_f(r, 'same_sign', '{:.2f}')}",
            "lane (green); its top copy (blue) " + (f"score {_f(r, 'copy_score', '{:.2f}')} IoU "
                                                       f"{_f(r, 'copy_iou', '{:.2f}')}" if r.get("copy_points")
                                                       else "none") + ("" if found else "; IMAGE NOT FOUND")]
    for k, s in enumerate(text):
        y = 22 + 22 * k
        cv2.putText(im, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return im, found


def sheets(images, out_dir, per_sheet=8, cols=2, width=640):
    names = []
    for k in range(0, len(images), per_sheet):
        tiles = [cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1])))) for im in
                 images[k:k + per_sheet]]
        th = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        grid = np.concatenate([np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)], axis=0)
        name = osp.join(out_dir, f"sheet_{k // per_sheet:02d}.jpg")
        cv2.imwrite(name, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        names.append(name)
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--which", choices=sorted(WHICH), default="above_missed")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args(argv)
    rows = [r for r in read_rows(args.csv) if WHICH[args.which](r)]
    picked = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    out_dir = args.out_dir or osp.join(osp.dirname(osp.abspath(args.csv)), f"near_misses_{args.which}")
    os.makedirs(out_dir, exist_ok=True)
    images, missing = [], 0
    for k, r in enumerate(picked):
        im, found = draw(r, args.width)
        missing += not found
        base = osp.splitext(osp.basename(r["image"]))[0]
        cv2.imwrite(osp.join(out_dir, f"{k:03d}_{base}_q{r['query']}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        images.append(im)
    names = sheets(images, out_dir) if images else []
    keys = ["image", "gt", "query", "score", "iou", "lat_px", "d_bottom", "d_middle", "d_top", "same_sign", "steep",
            "copy_score", "copy_iou"]
    with open(osp.join(out_dir, "picked.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(picked)
    print(f"{len(picked)} of {len(rows)} near misses ({args.which}) drawn to {out_dir}: {len(names)} contact sheets"
          + (f"; {missing} images not found (drawn on grey)" if missing else ""))
    return dict(picked=picked, out_dir=out_dir, sheets=names, missing=missing, n_rows=len(rows))


if __name__ == "__main__":
    main()
