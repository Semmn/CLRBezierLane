"""F1 against confidence threshold, from a single inference pass.

Why this is exact
-----------------
``get_lanes`` filters by ``conf_threshold`` *before* NMS, so it looks as though
the sweep would need one inference pass per threshold. It does not. Greedy NMS
keeps the highest-scoring member of each cluster, and a candidate can only be
suppressed by one scoring above it, so if the suppressor falls below a
threshold the suppressed one is below it too. Filtering before NMS and
filtering after therefore leave the same set, and so does the ``nms_topk`` cap
(the top-k overall that clear ``t`` are exactly the top-k of those that clear
``t``). One dump at a low threshold gives every higher threshold exactly.

Cost
----
Rasterizing 34,680 images is the expensive part, and it does not depend on the
threshold, so it happens once: see ``fast_sweep.py`` for why that is exact and
for the test that checks it against ``eval_predictions``. The earlier version of
this script called the official evaluator once per threshold -- 44 full CULane
evaluations for the default grid -- which is what made it look like it was
"evaluating again and again".

What it is for
--------------
A model whose optimum sits at 0.90 has its scores packed against 1.0, where the
F1-versus-threshold curve is steep, so a threshold chosen on validation
transfers badly to test. A model optimal at 0.50 sits on a flat part of the
curve and transfers well. Comparing two models at their *validation-chosen*
thresholds conflates "worse model" with "more fragile operating point"; this
separates them by reporting, for each model, F1 at the validation-chosen
threshold and F1 at the test-optimal one. The difference between those two
numbers is the transfer loss, and it belongs in the paper.

Usage:
    python tools/clrbezier/sweep_threshold.py \\
        configs/clrbezier/culane/clrbezier_collab_perturb_r34.py \\
        work_dirs/.../epoch_24.pth --picked 0.90
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.registry import MODELS

from libs.datasets.metrics.culane_metric import CULaneMetric

from fast_sweep import lane_scores, run_sweep, verify_official


def _get(result, *names):
    for name in names:
        if isinstance(result, dict) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    raise KeyError(
        f"none of {names} in the inference result. Present: "
        f"{list(result.keys()) if isinstance(result, dict) else dir(result)}"
    )


def collect(model, loader, device, max_images=None):
    """One pass at the dump threshold; keep every lane with its score."""
    out = []
    seen = 0
    with torch.no_grad():
        for data in loader:
            results = model.test_step(data)
            for result in results:
                lanes = _get(result, "lanes")
                scores = _get(result, "scores")
                meta = _get(result, "metainfo", "meta")
                scores = lane_scores(lanes, scores)
                out.append((meta["sub_img_name"], lanes, scores))
            seen += len(results)
            if seen % 2000 < len(results):
                print(f"  {seen} images", flush=True)
            if max_images is not None and seen >= max_images:
                break
    return out


def evaluate_at(dump, threshold, metric, data_root, data_list, categories_dir):
    """One official evaluation at one threshold, through the filesystem.

    Correct, and the right call when you want a single reportable number with
    its category breakdown. Do NOT put it in a loop: it re-parses all 34,680
    annotations and re-rasterizes every lane on each call. Use ``run_sweep``
    for a grid.
    """
    return verify_official(dump, threshold, metric, data_root, data_list,
                           categories_dir,
                           logger=MMLogger.get_current_instance())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--dump-threshold", type=float, default=0.10,
                    help="inference threshold for the single pass; every swept "
                         "threshold must be at or above it")
    ap.add_argument("--grid", type=float, nargs=3, default=[0.10, 0.96, 0.02],
                    metavar=("START", "STOP", "STEP"))
    ap.add_argument("--picked", type=float, default=None,
                    help="the threshold you selected on validation, for the "
                         "transfer-loss line")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes for the rasterization pass; 1 keeps "
                         "it in-process, which is what you want if it crashes")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the single official eval_predictions check at "
                         "the peak threshold")
    ap.add_argument("--rgb-masks", action="store_true",
                    help="rasterize onto the official 3-channel canvas instead "
                         "of 1 channel; 2.3x slower, provably the same IoU")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))

    # Decode everything above the dump threshold in one pass.
    cfg.model.setdefault("test_cfg", {})
    cfg.model["test_cfg"]["conf_threshold"] = args.dump_threshold

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(torch.device(args.device if torch.cuda.is_available() else "cpu")).eval()

    loader_cfg = copy.deepcopy(
        cfg.test_dataloader if args.split == "test" else cfg.val_dataloader)
    loader = Runner.build_dataloader(loader_cfg)
    evaluator_cfg = cfg.test_evaluator if args.split == "test" else cfg.val_evaluator
    data_root = evaluator_cfg["data_root"]
    data_list = evaluator_cfg["data_list"]
    categories_dir = os.path.join(data_root, "list/test_split/")

    print(f"inference at conf >= {args.dump_threshold} ({args.split} split)...")
    dump = collect(model, loader, args.device, args.max_images)
    total = sum(len(s) for _, _, s in dump)
    print(f"{len(dump)} images, {total} surviving lanes "
          f"({total / max(1, len(dump)):.2f} per image)\n")

    metric = CULaneMetric(data_root=data_root, data_list=data_list)

    start, stop, step = args.grid
    thresholds = np.round(np.arange(start, stop + 1e-9, step), 4)
    thresholds = [t for t in thresholds if t >= args.dump_threshold]
    if args.picked is not None and args.picked not in thresholds:
        thresholds.append(round(args.picked, 4))
        thresholds.sort()

    swept, _ = run_sweep({"model": dump}, thresholds, data_root, data_list,
                         categories_dir, metric, jobs=args.jobs,
                         verify=not args.no_verify,
                         logger=MMLogger.get_current_instance(),
                         rgb_masks=args.rgb_masks)

    print(f"\n{'conf':>6} {'F1':>8} {'precision':>10} {'recall':>8}")
    print("-" * 36)
    rows = []
    for result in swept["model"]:
        threshold = result["threshold"]
        rows.append((threshold, result["F1"], result["Precision"], result["Recall"]))
        mark = "  <- picked on val" if (args.picked is not None
                                        and abs(threshold - args.picked) < 1e-9) else ""
        print(f"{threshold:6.2f} {result['F1'] * 100:8.2f} "
              f"{result['Precision'] * 100:10.2f} "
              f"{result['Recall'] * 100:8.2f}{mark}")

    best = max(rows, key=lambda r: r[1])
    print(f"\ntest-optimal: conf {best[0]:.2f} -> F1 {best[1] * 100:.2f}")
    if args.picked is not None:
        picked = min(rows, key=lambda r: abs(r[0] - args.picked))
        loss = (best[1] - picked[1]) * 100
        print(f"at the validation-chosen {picked[0]:.2f} -> F1 {picked[1] * 100:.2f}")
        print(f"transfer loss: {loss:.2f} F1")
        if loss > 0.5:
            print("\nThe operating point, not the model, is costing you most of that gap.")
            print("A model whose optimum sits high has its scores packed against 1.0,")
            print("where this curve is steep and a threshold does not transfer.")
        else:
            print("\nThe threshold transferred. The gap is the model, not the operating point.")

    # Flatness around the peak: how much does being 0.05 off cost?
    peak = best[0]
    near = [r for r in rows if abs(r[0] - peak) <= 0.05 + 1e-9]
    if len(near) > 1:
        drop = (best[1] - min(r[1] for r in near)) * 100
        print(f"\nwithin +-0.05 of the optimum the curve varies by {drop:.2f} F1 "
              f"({'steep — fragile' if drop > 0.4 else 'flat — robust'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
