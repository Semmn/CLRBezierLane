"""The static vanishing point of a dataset, for Polar R-CNN's global pole (MultiAnchorFormerHead anchor_cfg.pole_px):
"positioned near the static vanishing point observed across the entire lane image dataset" (arXiv 2411.01499, Sec.
III.A). Per image, the vanishing point of its GT lanes (median pairwise intersection of lines fitted to their upper
halves, polar_view_check.vanishing_point); over the images, the median x and y, in the head's frame (img_w x img_h).

Usage (CPU, minutes; lanes from the config's pipeline as the head reads them, polar_view_check.load_images):
    python tools/clrbezier/static_pole.py CONFIG [--images 4000] [--split train|val] [--batch 32] [--workers 8]
Prints the median, quartiles and the share of images with a vanishing point; paste the median into
anchor_cfg.pole_px. The train pipeline's augmentation (flips, shifts, rotations, crops) keeps concurrent lines
concurrent, so each image's point is that image's as the network sees it; flips centre the median's x."""
import argparse
import os.path as osp
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--images", type=int, default=4000)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    from mmengine.config import Config
    from polar_view_check import load_images, vanishing_point
    cfg = Config.fromfile(args.config)
    head = cfg.model["bbox_head"]
    images, img_h = load_images(cfg, args, head)
    vps = [vanishing_point(lanes, img_h) for lanes in images]
    v = np.array([p for p in vps if p is not None], dtype=np.float64)
    if not len(v):
        raise SystemExit("no image with a vanishing point (needs 2 non-parallel lanes)")
    q = np.percentile(v, [25, 50, 75], axis=0)
    print(f"images {len(images)}, with a vanishing point {len(v)} ({100.0 * len(v) / len(images):.1f}%)")
    print(f"x: median {q[1, 0]:.1f} px (quartiles {q[0, 0]:.1f} .. {q[2, 0]:.1f}); "
          f"y: median {q[1, 1]:.1f} px (quartiles {q[0, 1]:.1f} .. {q[2, 1]:.1f}); frame {head['img_w']} x {img_h}")
    print(f"anchor_cfg.pole_px=({q[1, 0]:.1f}, {q[1, 1]:.1f})")


if __name__ == "__main__":
    main()
