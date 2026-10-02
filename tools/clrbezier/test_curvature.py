"""Does curvature actually buy anything on CULane?

Three questions, in increasing cost, and the first one settles most of it.

1. ``--mode gt`` (no model, no GPU). Fit every ground-truth lane twice, once
   with a straight line and once with a cubic, and score each fit against the
   lane it came from using the *official* CULane IoU (30 px width, 590x1640
   canvas). If a straight-line fit already clears IoU 0.5, then a model that
   predicted that line perfectly would score the lane as a true positive, and
   curvature cannot add it. The gap between the cubic's reachable fraction and
   the line's is a hard upper bound on the recall any curve representation can
   contribute. This is the representation analogue of CLRerNet's oracle study.

2. ``--mode model`` (one forward pass). How far the refined control points
   actually depart from straight, per refinement stage and per category. Says
   whether the model *uses* the freedom it has.

3. ``--mode ablate`` (one inference pass). Project every predicted curve onto
   its own best straight-line fit over its valid rows, keeping the score and
   extent, and re-evaluate F1 with the official metric. If F1 does not drop,
   the curvature the model predicts is contributing nothing to the metric.
   This is the causal test, and it needs no retraining.

A note on what counts as "straight", because the obvious yardstick is wrong by
about 5x. Lanes are rasterized 30 px wide, which invites the guess that a
deviation past the 15 px half-width matters. It does not. Two *parallel* lanes
cross IoU 0.5 at an offset of 10 px -- that part is exact, IoU = (W-d)/(W+d) --
but a curve is not offset uniformly from its own best straight line. For a
parabolic bow of B px the least-squares line splits the bow, leaving a mean
residual of only ~0.27B, so the lane stays reachable until **B is around 40 px**.
``--self-test`` derives and prints that threshold rather than assuming it. Any
curvature below it is real geometry the metric cannot reward.

    python tools/clrbezier/test_curvature.py --mode gt \\
        --data-root dataset/culane --data-list dataset/culane/list/test.txt
    python tools/clrbezier/test_curvature.py --self-test
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libs.datasets.metrics.culane_metric import (  # noqa: E402
    discrete_cross_iou,
    load_categories,
    load_culane_data,
)
from libs.utils.lane_utils import interp  # noqa: E402

# Single-channel canvas: draw_lane paints all three channels identically, so
# intersection and union both triple and the IoU is unchanged. Verified exactly
# in test_fast_sweep.py.
MASK_SHAPE = (590, 1640)
WIDTH = 30
IOU_LEVELS = (0.5, 0.75)
# The metric's own scale, in pixels at full CULane width.
HALF_WIDTH = WIDTH / 2.0


# --------------------------------------------------------------------------
# geometry


def fit_x_of_y(points, deg: int):
    """Least-squares x = p(y) of the given degree, sampled at the lane's own ys.

    Evaluating at the source lane's y values keeps the comparison about *shape*:
    the fit covers exactly the same vertical extent, so the IoU that follows
    cannot be moved by a difference in length.

    y is normalized to [0, 1] over the lane's span before fitting, or a cubic in
    raw pixel y (range ~590) is badly conditioned and numpy warns.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < deg + 1:
        return None
    x, y = pts[:, 0], pts[:, 1]
    span = float(y.max() - y.min())
    if span < 1e-6:
        return None
    u = (y - y.min()) / span
    coef = np.polyfit(u, x, deg)
    return np.stack([np.polyval(coef, u), y], axis=1)


def official_iou(lane_a, lane_b) -> float:
    """The CULane metric's own IoU between two single lanes."""
    a = np.array([interp(np.asarray(lane_a, dtype=np.float64), n=5)], dtype=object)
    b = np.array([interp(np.asarray(lane_b, dtype=np.float64), n=5)], dtype=object)
    return float(discrete_cross_iou(a, b, width=WIDTH, img_shape=MASK_SHAPE)[0, 0])


def chord_deviation(cp_x, n: int = 101):
    """Max |curve - chord| for cubic Bezier control points, in the same units.

    ``cp_x`` is ``[..., 4]``. Closed form at the midpoint is (3/8)|a + b| with
    a = P0 - 2P1 + P2 and b = P1 - 2P2 + P3; this takes the max over t, which is
    within ~15% of it.
    """
    cp = np.asarray(cp_x, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    basis = np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2,
                      3 * t ** 2 * (1 - t), t ** 3], axis=-1)      # [n, 4]
    curve = cp @ basis.T                                            # [..., n]
    chord = cp[..., :1] * (1 - t) + cp[..., 3:4] * t
    return np.abs(curve - chord).max(axis=-1)


# --------------------------------------------------------------------------
# 1. ground truth only


def analyze_lane(gt_points):
    """Straight-line and cubic fit quality for one ground-truth lane."""
    gt = np.asarray(gt_points, dtype=np.float64)
    if gt.shape[0] < 2:
        return None
    line = fit_x_of_y(gt, 1)
    if line is None:
        return None
    row = {
        "n_points": int(gt.shape[0]),
        "y_span": float(gt[:, 1].max() - gt[:, 1].min()),
        "dev_line_max": float(np.abs(line[:, 0] - gt[:, 0]).max()),
        "dev_line_rms": float(np.sqrt(((line[:, 0] - gt[:, 0]) ** 2).mean())),
        "iou_line": official_iou(line, gt),
        "iou_cubic": np.nan,
        "dev_cubic_max": np.nan,
    }
    # A cubic through 4 points is exact and would report IoU 1.0 for free, which
    # says nothing. Require a 5th point so the fit is actually constrained.
    if gt.shape[0] >= 5:
        cubic = fit_x_of_y(gt, 3)
        if cubic is not None:
            row["dev_cubic_max"] = float(np.abs(cubic[:, 0] - gt[:, 0]).max())
            row["iou_cubic"] = official_iou(cubic, gt)
    return row


def _image_rows(task):
    lanes, cat = task
    out = []
    for lane in lanes:
        row = analyze_lane(lane)
        if row is not None:
            row["cat"] = cat
            out.append(row)
    return out


def run_gt(data_root, data_list, categories_dir=None, jobs=None, limit=None):
    categories_dir = categories_dir or os.path.join(data_root, "list/test_split/")
    try:
        data_cats, categories = load_categories(categories_dir)
    except FileNotFoundError:
        print(f"no category lists under {categories_dir}; reporting overall only")
        data_cats, categories = {}, []
    print(f"loading annotations from {data_root}", flush=True)
    annos, cats = load_culane_data(data_root, data_list, data_cats)
    if limit is not None:
        annos, cats = annos[:limit], cats[:limit]

    tasks = list(zip(annos, cats))
    if jobs == 1:
        results = [_image_rows(t) for t in tasks]
    else:
        from p_tqdm import p_map
        results = p_map(_image_rows, tasks, num_cpus=jobs, desc="fitting lanes")
    rows = [r for group in results for r in group]
    if not rows:
        print("no usable lanes found")
        return 1
    report_gt(rows, categories)
    return 0


def report_gt(rows, categories):
    def block(name, subset):
        if not subset:
            return
        dev = np.array([r["dev_line_max"] for r in subset])
        iou_l = np.array([r["iou_line"] for r in subset])
        cub = [r for r in subset if not np.isnan(r["iou_cubic"])]
        iou_c = np.array([r["iou_cubic"] for r in cub]) if cub else np.zeros(0)
        cells = [f"{name:<14}", f"{len(subset):7d}",
                 f"{np.median(dev):8.2f}", f"{np.percentile(dev, 95):8.2f}"]
        for level in IOU_LEVELS:
            cells.append(f"{(iou_l > level).mean() * 100:8.2f}")
        for level in IOU_LEVELS:
            cells.append(f"{(iou_c > level).mean() * 100:8.2f}" if len(iou_c)
                         else f"{'   n/a':>8}")
        head = (iou_c > IOU_LEVELS[0]).mean() - (iou_l > IOU_LEVELS[0]).mean() \
            if len(iou_c) else float("nan")
        cells.append(f"{head * 100:+9.2f}")
        print(" ".join(cells), flush=True)

    print(f"\n{len(rows)} lanes. Deviation is |straight-line fit - GT| in pixels "
          f"at full 1640 px width.")
    print(f"'reachable' = the fit scores above that IoU against its own GT lane, "
          f"i.e. a model predicting it perfectly would get a true positive.\n")
    print(f"{'category':<14} {'lanes':>7} {'dev p50':>8} {'dev p95':>8} "
          f"{'line@.5':>8} {'line@.75':>8} {'cub@.5':>8} {'cub@.75':>8} "
          f"{'headroom':>9}")
    print("-" * 94)
    block("ALL", rows)
    for cat in categories:
        block(cat, [r for r in rows if r["cat"] == cat])

    dev = np.array([r["dev_line_max"] for r in rows])
    iou_l = np.array([r["iou_line"] for r in rows])
    cub = [r for r in rows if not np.isnan(r["iou_cubic"])]
    print(f"\nstraight-line deviation exceeds the {HALF_WIDTH:.1f} px half-width "
          f"on {(dev > HALF_WIDTH).mean() * 100:.2f}% of lanes, "
          f"and {WIDTH} px on {(dev > WIDTH).mean() * 100:.2f}%")
    if not cub:
        return
    iou_c = np.array([r["iou_cubic"] for r in cub])
    # Report EVERY IoU level. The headroom at 0.5 and at 0.75 can differ by
    # several times over, because a straight fit tolerates a far larger bow when
    # only 0.5 overlap is demanded -- and quoting the 0.5 figure alone is how a
    # curve representation gets written off on a metric that cannot see it.
    print()
    for level in IOU_LEVELS:
        reach = float((iou_l > level).mean())
        gain = float((iou_c > level).mean()) - reach
        print(f"IoU {level}: a perfect straight-line predictor reaches "
              f"{reach * 100:.2f}% of lanes (F1 ceiling "
              f"{200 * reach / (1 + reach):.2f} at perfect precision); "
              f"the cubic adds {gain * 100:+.2f}")
    gains = {lvl: float((iou_c > lvl).mean()) - float((iou_l > lvl).mean())
             for lvl in IOU_LEVELS}
    worst, best_level = max((g, lvl) for lvl, g in gains.items())
    if worst < 0.005:
        print("\nVERDICT: curvature is below the metric's resolution at every "
              "level tested. A curve representation cannot pay for itself here, "
              "and a straight-anchor baseline is giving up nothing.")
    else:
        per_cat = []
        for cat in categories:
            sub = [r for r in rows if r["cat"] == cat and not np.isnan(r["iou_cubic"])]
            if len(sub) < 50:
                continue
            l = np.array([r["iou_line"] for r in sub])
            c = np.array([r["iou_cubic"] for r in sub])
            per_cat.append((float((c > best_level).mean() - (l > best_level).mean()),
                            cat, len(sub)))
        per_cat.sort(reverse=True)
        print(f"\nVERDICT: real headroom, and it is concentrated. At IoU "
              f"{best_level} the cubic is worth {worst * 100:+.2f} overall, and "
              f"by category:")
        for gain, cat, n in per_cat[:4]:
            share = n / len(rows)
            print(f"    {cat:<14} {gain * 100:+7.2f} on {share * 100:5.2f}% of "
                  f"lanes -> {gain * share * 100:+.2f} overall")
        print("Weight the subset that carries it, and report F1 at the level that "
              "can see it. Optimizing F1@50 alone will never reveal this.")


# --------------------------------------------------------------------------
# self test


def synthetic_lane(x0=800.37, bow=0.0, n=30, y_lo=250.0, y_hi=580.0):
    """A lane with a known parabolic bow of exactly ``bow`` px at mid-height.

    x0 is deliberately fractional. ``draw_lane`` rasterizes with
    ``lane.astype(np.int32)``, which truncates, so a coordinate sitting exactly
    on an integer can fall either side of it under 1e-13 of float noise and
    shift a whole pixel column. At x0 = 800.0 an *exact* fit scores IoU 0.9707
    against its own source; at any fractional x0 it scores exactly 1.0. Real
    CULane annotations are fractional, so this is a property of the test, not of
    the measurement -- but it is why the self-test must not sit on an integer.
    """
    y = np.linspace(y_hi, y_lo, n)
    u = (y - y.min()) / (y.max() - y.min())
    x = x0 + bow * 4.0 * u * (1.0 - u)          # 0 at both ends, `bow` at u=0.5
    return np.stack([x, y], axis=1)


def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + name)
        ok &= bool(cond)

    # 1. Calibrate the rasterization against closed form. Two parallel lanes of
    #    width W offset by d have IoU exactly (W - d) / (W + d). Everything else
    #    here is read through this measurement, so if it does not reproduce the
    #    law, no later number means anything.
    print(f"parallel lanes offset by d, against (W-d)/(W+d) at W={WIDTH}:\n")
    print(f"{'d (px)':>7} {'measured':>9} {'theory':>8} {'delta':>8}")
    print("-" * 35)
    worst = 0.0
    for d in (2, 5, 10, 15, 20):
        base = synthetic_lane()
        shifted = base.copy()
        shifted[:, 0] += d
        got = official_iou(base, shifted)
        want = (WIDTH - d) / (WIDTH + d)
        worst = max(worst, abs(got - want))
        print(f"{d:7d} {got:9.4f} {want:8.4f} {got - want:+8.4f}")
    check(f"rasterized IoU matches the closed form (max delta {worst:.4f}; the "
          f"small positive bias is the thick-line end caps)", worst < 0.02)
    check("and the IoU-0.5 boundary for parallel lanes sits at d = 10 px",
          official_iou(synthetic_lane(),
                       synthetic_lane() + np.array([10.0, 0.0])) > 0.5
          > official_iou(synthetic_lane(), synthetic_lane() + np.array([11.0, 0.0])))

    # 2. Recover a known bow.
    print(f"\nrecovering a known parabolic bow:\n")
    print(f"{'bow':>6} {'dev_max':>8} {'mean|r|':>8} {'iou_line':>9} "
          f"{'predicted':>10} {'iou_cubic':>10}")
    print("-" * 56)
    measured = []
    for bow in (0.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0):
        gt = synthetic_lane(bow=bow)
        row = analyze_lane(gt)
        line = fit_x_of_y(gt, 1)
        mean_r = float(np.abs(line[:, 0] - gt[:, 0]).mean())
        # The same law as step 1, with the lane's own mean residual standing in
        # for a constant offset. This is what ties the bow case to closed form
        # instead of to a hardcoded crossing point.
        predicted = (WIDTH - mean_r) / (WIDTH + mean_r)
        measured.append((bow, row, mean_r, predicted))
        print(f"{bow:6.1f} {row['dev_line_max']:8.2f} {mean_r:8.2f} "
              f"{row['iou_line']:9.4f} {predicted:10.4f} {row['iou_cubic']:10.4f}")

    check("an exact fit scores IoU 1", measured[0][1]["iou_line"] > 0.9999)
    check("a straight lane has no deviation", measured[0][1]["dev_line_max"] < 1e-6)

    # For a parabola x0 + B*4u(1-u) the least-squares line is the constant
    # 2B/3, so the residual runs from -2B/3 at the ends to +B/3 at the middle
    # and the max is 2/3 of the bow. Derived, not eyeballed.
    ratios = [r["dev_line_max"] / b for b, r, _, _ in measured[1:]]
    check(f"max deviation is 2/3 of the bow (measured {np.mean(ratios):.4f}, "
          f"spread {np.ptp(ratios):.1e})",
          abs(np.mean(ratios) - 2.0 / 3.0) < 0.03 and np.ptp(ratios) < 1e-3)

    check("the cubic captures a parabolic bow exactly",
          all(r["iou_cubic"] > 0.999 for _, r, _, _ in measured))

    ious = [r["iou_line"] for _, r, _, _ in measured]
    check("line IoU falls monotonically with the bow",
          all(a >= b - 1e-9 for a, b in zip(ious, ious[1:])))

    gaps = [abs(r["iou_line"] - pred) for _, r, _, pred in measured[1:]]
    check(f"bowed-lane IoU follows the same offset law (max deviation "
          f"{max(gaps):.4f})", max(gaps) < 0.02)

    # The bow at which a lane stops being reachable by a straight line, read off
    # rather than assumed: it is where mean|r| reaches the 10 px parallel bound.
    crossing = next((b for b, r, _, _ in measured if r["iou_line"] < 0.5), None)
    print(f"\na straight line stops reaching the lane at a bow of ~{crossing:.0f} px "
          f"(mean residual ~{WIDTH / 3:.0f} px). Note this is far above the "
          f"{HALF_WIDTH:.1f} px half-width:\nthe metric tolerates much more "
          f"curvature than the raster width alone suggests.")
    check("that crossing is above the half-width, not at it",
          crossing is not None and crossing > 2 * HALF_WIDTH)

    # 3. chord_deviation. At t = 1/2 the deviation from the chord is exactly
    #    (3/8)|a + b|; over all t it can be arbitrarily larger, because an
    #    S-shaped curve with a = -b crosses the chord at the midpoint while
    #    bulging on both sides. So assert the identity and the bound, not a
    #    ratio -- an upper bound does not exist.
    rng = np.random.default_rng(0)
    cp = rng.normal(size=(2000, 4))
    a = cp[:, 0] - 2 * cp[:, 1] + cp[:, 2]
    b = cp[:, 1] - 2 * cp[:, 2] + cp[:, 3]
    mid_closed = 0.375 * np.abs(a + b)
    t = np.linspace(0.0, 1.0, 101)
    basis = np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2,
                      3 * t ** 2 * (1 - t), t ** 3], axis=-1)
    curve = cp @ basis.T
    chord = cp[:, :1] * (1 - t) + cp[:, 3:4] * t
    mid_numeric = np.abs(curve - chord)[:, 50]
    check(f"midpoint deviation equals (3/8)|a+b| "
          f"(max error {np.abs(mid_numeric - mid_closed).max():.2e})",
          float(np.abs(mid_numeric - mid_closed).max()) < 1e-12)
    dev = chord_deviation(cp)
    check("chord_deviation never falls below the midpoint value",
          bool((dev >= mid_closed - 1e-9).all()))
    s_shape = np.abs(a + b) < 0.01
    if s_shape.any():
        ratio = (dev[s_shape] / np.maximum(mid_closed[s_shape], 1e-12)).max()
        check(f"and exceeds it without bound on S-shaped curves (max ratio "
              f"{ratio:.0f}, so no upper bound can be asserted)", ratio > 10)
    line_cp = np.stack([np.zeros(5), np.ones(5) / 3,
                        2 * np.ones(5) / 3, np.ones(5)], -1)
    check("collinear control points deviate by 0",
          float(np.abs(chord_deviation(line_cp)).max()) < 1e-12)

    print("\nPASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "model", "ablate"], default="gt")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-root")
    ap.add_argument("--data-list")
    ap.add_argument("--categories-dir")
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--dump-threshold", type=float, default=0.10)
    ap.add_argument("--grid", type=float, nargs=3, default=[0.10, 0.96, 0.02],
                    metavar=("START", "STOP", "STEP"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="gt mode: only the first N images")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--expect-f1", type=float, default=None,
                    help="ablate mode: the F1@50 (in percent) this checkpoint is "
                         "known to score. The unmodified arm must reproduce it, "
                         "or the run aborts instead of reporting a delta.")
    ap.add_argument("--expect-tol", type=float, default=0.5)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--rgb-masks", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.mode == "gt":
        if not (args.data_root and args.data_list):
            ap.error("--mode gt needs --data-root and --data-list")
        return run_gt(args.data_root, args.data_list, args.categories_dir,
                      args.jobs, args.limit)
    from curvature_model import run_model, run_ablate   # torch-only half
    if not (args.config and args.checkpoint):
        ap.error(f"--mode {args.mode} needs --config and --checkpoint")
    return (run_model if args.mode == "model" else run_ablate)(args)


if __name__ == "__main__":
    raise SystemExit(main())
