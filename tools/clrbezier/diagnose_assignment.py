"""Does the LaneIoU assigner starve curved lanes?

The hypothesis
--------------
Assignment cost and dynamic-k both come from LaneIoU between the *predicted*
lanes and the GT. The anchor priors are straight (``prior_delta`` sits at ~0.007
logits, so they never moved from the straight initializer). A curved GT lane
therefore overlaps every near-straight prediction poorly, which raises its cost
and shrinks its dynamic k, which means fewer assigned positives, which means
less gradient pushing anything to bend, which keeps predictions straight. A
self-reinforcing loop, and it would produce the measured 2.7x under-bending
without any reference to features or read-out capacity.

It is also upstream of everything else under discussion: no MoE router fixes a
starved expert, and loss weighting cannot help a lane that was never matched.

What this measures
------------------
For every GT lane, bucketed by how far that lane departs from a straight line:

  * assigned positives, per stage      -- the quantity the hypothesis predicts falls
  * dynamic k (SimOTA assigners)       -- the mechanism, if that is the assigner
  * best LaneIoU any prediction got    -- the cause, upstream of both
  * best 1 - iou_cost_assign           -- the wide-width variant the cost uses

Reading it: if positives and dynamic k fall monotonically as the bow grows while
the straight buckets stay flat, the assignment is the bottleneck. If they are
flat, the assigner is fine and the deficiency is downstream -- run
``probe_bow.py`` next to decide between representation and read-out.

    python tools/clrbezier/diagnose_assignment.py CONFIG CKPT --max-images 2000
    python tools/clrbezier/diagnose_assignment.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Bow buckets in pixels at img_w. The IoU-0.75 reachability boundary is ~16 px
# and the IoU-0.50 boundary ~36 px (derived in test_curvature.py), so the
# buckets straddle both decision points instead of splitting the range evenly.
BOW_EDGES = (0.0, 2.0, 5.0, 10.0, 16.0, 36.0, float("inf"))


def bucket_label(lo: float, hi: float) -> str:
    return f"{lo:>5.0f}-{hi:<5.0f}" if np.isfinite(hi) else f"{lo:>5.0f}+     "


def straight_fit_bow(xs: np.ndarray, ys: np.ndarray, img_w: int):
    """Max |straight-fit residual| for one lane, in pixels at ``img_w``.

    ``xs`` is the GT row vector in pixels with invalid rows outside [0, img_w-1],
    matching ``build_cost_cache``'s validity rule exactly, so this bucketing and
    the assigner see the same lanes.
    """
    valid = (xs >= 0.0) & (xs <= float(img_w - 1))
    if valid.sum() < 3:
        return None
    x, y = xs[valid], ys[valid]
    span = float(y.max() - y.min())
    if span < 1e-6:
        return None
    u = (y - y.min()) / span
    coef = np.polyfit(u, x, 1)
    return float(np.abs(np.polyval(coef, u) - x).max())


def report(rows, img_w, stages):
    """rows: list of dicts with bow, per-stage positives, dyn_k, best_iou, best_cost_iou."""
    bow = np.array([r["bow"] for r in rows])
    print(f"\n{len(rows)} GT lanes. Bow is the max straight-fit residual in px at "
          f"img_w={img_w}; multiply by {1640/img_w:.2f} for full CULane width.")
    print("A bow above ~16 px is where a straight fit stops reaching the lane at "
          "IoU 0.75, above ~36 px at IoU 0.50.\n")
    head = f"{'bow (px)':<13} {'lanes':>7} {'share':>7}"
    for s in stages:
        head += f" {'pos s' + str(s):>8}"
    head += f" {'dyn k':>7} {'best IoU':>9} {'best cIoU':>10}"
    print(head)
    print("-" * len(head))
    for lo, hi in zip(BOW_EDGES[:-1], BOW_EDGES[1:]):
        sel = (bow >= lo) & (bow < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        line = f"{bucket_label(lo, hi):<13} {n:7d} {n / len(rows) * 100:6.2f}%"
        for s in stages:
            vals = np.array([r["pos"][s] for r, m in zip(rows, sel) if m], dtype=float)
            line += f" {vals.mean():8.2f}"
        for key, w in (("dyn_k", 7), ("best_iou", 9), ("best_cost_iou", 10)):
            vals = np.array([r[key] for r, m in zip(rows, sel) if m], dtype=float)
            vals = vals[~np.isnan(vals)]
            line += f" {vals.mean():{w}.3f}" if len(vals) else f" {'n/a':>{w}}"
        print(line)

    last = max(stages)
    straight = np.array([r["pos"][last] for r in rows if r["bow"] < 16.0], dtype=float)
    curved = np.array([r["pos"][last] for r in rows if r["bow"] >= 16.0], dtype=float)
    iou_s = np.array([r["best_iou"] for r in rows if r["bow"] < 16.0], dtype=float)
    iou_c = np.array([r["best_iou"] for r in rows if r["bow"] >= 16.0], dtype=float)
    if not len(curved) or not len(straight):
        print("\nnot enough lanes on one side of the 16 px boundary to compare")
        return
    ratio = curved.mean() / max(straight.mean(), 1e-9)
    print(f"\nbow < 16 px : {len(straight):6d} lanes, {straight.mean():.2f} positives, "
          f"best IoU {iou_s.mean():.3f}")
    print(f"bow >= 16 px: {len(curved):6d} lanes, {curved.mean():.2f} positives, "
          f"best IoU {iou_c.mean():.3f}")
    print(f"\ncurved lanes receive {ratio:.2f}x the positives of straight ones "
          f"and reach {iou_c.mean() / max(iou_s.mean(), 1e-9):.2f}x the LaneIoU")
    if ratio < 0.8:
        print("\nVERDICT: curved lanes are starved. The assignment is the bottleneck, "
              "upstream of features and read-out alike.\nThe fix is in the cost, not "
              "in capacity: a curvature-tolerant assignment cost, or a warmup where\n"
              "dynamic k comes from mean row distance instead of LaneIoU so a curved GT "
              "can recruit anchors\nbefore any prediction overlaps it. No MoE router "
              "repairs a starved expert.")
    elif ratio > 1.2:
        print("\nVERDICT: curved lanes get MORE positives, not fewer. The hypothesis is "
              "refuted -- and this is worth\nknowing, because it means the under-bending "
              "survives abundant supervision, which points at the loss\n(LaneIoU "
              "saturating inside its half-width) rather than at the assignment.")
    else:
        print("\nVERDICT: assignment is roughly even across curvature. Not the bottleneck.\n"
              "Run probe_bow.py next to split representation from read-out.")


# --------------------------------------------------------------------------


def run(args):
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS

    from libs.clrbezier.assigners import SimOTALaneAssigner, build_cost_cache

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # eval() for determinism: dropout off, gate thresholds at their eval values.
    # The assignment itself does not depend on training mode.
    model.to(device).eval()
    head = model.bbox_head

    loader_cfg = copy.deepcopy(cfg.train_dataloader)
    loader_cfg["batch_size"] = args.batch_size
    loader_cfg.setdefault("sampler", {})
    loader = Runner.build_dataloader(loader_cfg)

    prior_ys = head.prior_ys.detach().cpu().numpy() * float(head.img_h)
    stages = sorted(set(args.stages)) if args.stages else list(range(head.refine_layers))
    rows, seen = [], 0

    with torch.no_grad():
        for data in loader:
            data = model.data_preprocessor(data, True)
            feats = head._select_features(model.extract_feat(data["inputs"]))
            outs = head.forward_train(feats)
            lanes = head.target_adapter.extract_lanes(data["data_samples"], device)
            valid_targets = [t[t[:, 1] == 1] for t in lanes]

            for b, target in enumerate(valid_targets):
                if target.shape[0] == 0:
                    continue
                t_xs = target[:, 6:].detach().cpu().numpy()
                bows = [straight_fit_bow(t_xs[g], prior_ys, head.img_w)
                        for g in range(target.shape[0])]
                per_gt = [{"bow": bw, "pos": {}, "dyn_k": np.nan,
                           "best_iou": np.nan, "best_cost_iou": np.nan}
                          for bw in bows]

                for stage in stages:
                    pred = outs["main_preds"][stage][b].detach()
                    assigner = head.main_stage_assigners.get(stage, head.main_assigner)
                    required = set(assigner.required_cache_keys) | {"lane_iou_dynamic"}
                    cache = build_cost_cache(pred, target, head.img_w, head.img_h,
                                             head.lane_width, head.lane_width_cost,
                                             required, iou_fns=head.iou_fns)
                    _, cols = assigner.assign(cache)
                    counts = np.bincount(cols.detach().cpu().numpy().astype(int),
                                         minlength=target.shape[0])
                    iou_dyn = cache["lane_iou_dynamic"].clamp(0, 1)
                    cost_iou = 1.0 - cache["iou_cost_assign"]
                    # dynamic k, recomputed exactly as SimOTALaneAssigner does
                    dyn = None
                    if isinstance(assigner, SimOTALaneAssigner):
                        k = min(assigner.candidate_topk, iou_dyn.shape[0])
                        dyn = iou_dyn.topk(k, dim=0).values.sum(0).int().clamp(
                            min=assigner.min_dynamic_k, max=iou_dyn.shape[0])
                        dyn = dyn.detach().cpu().numpy()
                    best_iou = iou_dyn.max(0).values.detach().cpu().numpy()
                    best_cost = cost_iou.max(0).values.detach().cpu().numpy()
                    for g in range(target.shape[0]):
                        per_gt[g]["pos"][stage] = int(counts[g])
                        if stage == max(stages):
                            per_gt[g]["best_iou"] = float(best_iou[g])
                            per_gt[g]["best_cost_iou"] = float(best_cost[g])
                            if dyn is not None:
                                per_gt[g]["dyn_k"] = float(dyn[g])
                rows.extend(r for r in per_gt if r["bow"] is not None)

            seen += len(valid_targets)
            if seen % 500 < len(valid_targets):
                print(f"  {seen} images, {len(rows)} GT lanes", flush=True)
            if args.max_images is not None and seen >= args.max_images:
                break

    if not rows:
        print("no usable GT lanes found")
        return 1
    report(rows, head.img_w, stages)
    return 0


# --------------------------------------------------------------------------


def self_test():
    """Checks the bucketing and the bow fit against constructed lanes."""
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""))
        ok &= bool(cond)

    img_w, rows = 800, 72
    ys = np.linspace(320.0, 0.0, rows)

    # A straight lane must read exactly zero bow, whatever its slope.
    for slope in (0.0, 0.3, -0.5):
        xs = 400.0 + slope * (ys - ys.mean())
        bow = straight_fit_bow(xs, ys, img_w)
        chk(f"straight lane, slope {slope:+.1f} -> bow 0", bow is not None and bow < 1e-6,
            f"{bow:.2e}" if bow is not None else "None")

    # A parabolic bow of B has a max least-squares residual of 2/3 B, the same
    # relation test_curvature.py derives and verifies.
    for B in (6.0, 15.0, 30.0):
        u = (ys - ys.min()) / (ys.max() - ys.min())
        xs = 400.0 + B * 4.0 * u * (1.0 - u)
        bow = straight_fit_bow(xs, ys, img_w)
        chk(f"parabolic bow {B:.0f} px -> residual 2/3 of it",
            bow is not None and abs(bow / B - 2 / 3) < 0.02, f"ratio {bow / B:.4f}")

    # Rows outside [0, img_w-1] are invalid, matching build_cost_cache.
    xs = np.full(rows, -1e5)
    xs[:40] = 400.0
    chk("invalid rows are excluded from the fit",
        straight_fit_bow(xs, ys, img_w) is not None
        and straight_fit_bow(xs, ys, img_w) < 1e-6)
    chk("fewer than 3 valid rows returns None",
        straight_fit_bow(np.where(np.arange(rows) < 2, 400.0, -1e5), ys, img_w) is None)

    # Bucket edges must straddle both reachability boundaries.
    chk("buckets straddle the IoU-0.75 boundary (16 px)", 16.0 in BOW_EDGES)
    chk("buckets straddle the IoU-0.50 boundary (36 px)", 36.0 in BOW_EDGES)

    # The report must run on synthetic rows without a model.
    fake = [{"bow": b, "pos": {0: 3, 1: 3, 2: 3}, "dyn_k": 3.0,
             "best_iou": max(0.0, 0.9 - b / 60), "best_cost_iou": 0.8}
            for b in (0.5, 1.0, 3.0, 8.0, 20.0, 50.0)]
    try:
        report(fake, img_w, [0, 1, 2])
        chk("report renders on synthetic input", True)
    except Exception as exc:                                        # pragma: no cover
        chk(f"report renders on synthetic input ({exc})", False)

    print("\nPASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stages", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.config and args.checkpoint):
        ap.error("CONFIG and CKPT are required unless --self-test")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
