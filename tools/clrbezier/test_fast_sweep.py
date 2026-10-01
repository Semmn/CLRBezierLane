"""Proof that the cached-IoU sweep equals the official metric, threshold by
threshold.

The whole speedup rests on one claim: the IoU between a prediction and an
annotation does not depend on the confidence threshold, so a sweep can
rasterize once and then only delete rows of the cached matrix. If that claim is
wrong anywhere, every number the sweep reports is wrong — and wrong quietly,
which is the worst failure mode a metric tool has. So it is checked here
against the official ``culane_metric`` itself, not against a reimplementation
of it, on cases chosen to break it:

* an image with no annotations at all (test7_cross is this, 2,000 times over)
* an image where every prediction is filtered out
* scores exactly equal to the threshold, where ``>=`` and ``>`` differ
* **a reassignment case**: two predictions compete for the same annotation and
  the loser is matched elsewhere. Remove the winner and the Hungarian
  assignment must change. A sweep that cached the *assignment* instead of the
  IoU matrix would get this wrong; caching the matrix and re-solving gets it
  right, and this is the test that tells the two apart.

    python tools/clrbezier/test_fast_sweep.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from libs.datasets.metrics.culane_metric import (
    culane_metric,
    load_culane_img_data,
)
from fast_sweep import (
    EPS,
    FAST_MASK_SHAPE,
    IMG_SHAPE,
    IOU_THRESHOLDS,
    MAIN_IOU,
    _image_ious,
    _parse_prediction_text,
    sweep,
)

ORI_W, ORI_H = 1640, 590


def lane(x_bottom, x_top, n=40):
    """A straight lane from the image bottom to the top, as (x, y) points."""
    ys = np.linspace(ORI_H - 1, 0, n)
    xs = np.linspace(x_bottom, x_top, n)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def official_counts(images, threshold, iou_thr=MAIN_IOU):
    """TP/FP/FN the way ``eval_predictions`` computes them, at one threshold."""
    tp = n_pred = n_gt = 0
    k = list(IOU_THRESHOLDS).index(iou_thr)
    for preds, scores, anno, cat in images:
        keep = [p for p, s in zip(preds, scores) if s >= threshold]
        result = culane_metric(keep, anno, cat, iou_thresholds=list(IOU_THRESHOLDS))
        hits = result["hits"][k]
        tp += int(hits.sum())
        n_pred += int(hits.size)
        n_gt += result["n_gt"]
    return tp, n_pred - tp, n_gt - tp


def fast_records(images):
    """What ``build_records`` produces, without needing a model or Lane objects."""
    records = []
    for preds, scores, anno, cat in images:
        records.append({
            "name": f"img{len(records)}",
            "ious": _image_ious((preds, anno, FAST_MASK_SHAPE)),
            "scores": np.asarray(scores, dtype=np.float64),
            "n_gt": len(anno),
            "cat": cat,
        })
    return records


def f1_of(tp, fp, fn):
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    return 2 * prec * rec / (prec + rec + EPS)


def report(name, ok):
    print(("  ok   " if ok else "  FAIL ") + name)
    return ok


# --------------------------------------------------------------------------
# end to end, on a synthetic split


class FakeLane:
    """Stands in for ``libs.core.lane.Lane``: callable, normalized ys -> xs."""

    def __init__(self, x_bottom, x_top):
        self.x_bottom = x_bottom / ORI_W
        self.x_top = x_top / ORI_W

    def __call__(self, ys):
        # ys ascending from 0 (image top) to just under 1 (bottom).
        return self.x_top + (self.x_bottom - self.x_top) * np.asarray(ys)


def write_split(root, images):
    """A miniature CULane layout: annotations, data list, category lists."""
    names = []
    os.makedirs(os.path.join(root, "list/test_split"), exist_ok=True)
    for i, (_, _, anno, _) in enumerate(images):
        name = f"driver_0_test/seq_{i}.MP4/00000.jpg"
        names.append(name)
        path = os.path.join(root, name.replace(".jpg", ".lines.txt"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            for points in anno:
                handle.write(" ".join(f"{x:.3f} {y:.3f}" for x, y in points) + "\n")
    with open(os.path.join(root, "list/test.txt"), "w") as handle:
        for name in names:
            handle.write("/" + name + "\n")
    by_cat = {}
    for name, (_, _, _, cat) in zip(names, images):
        by_cat.setdefault(cat, []).append(name)
    for cat in ("test0_normal", "test1_crowd", "test2_hlight", "test3_shadow",
                "test4_noline", "test5_arrow", "test6_curve", "test7_cross",
                "test8_night"):
        with open(os.path.join(root, f"list/test_split/{cat}.txt"), "w") as handle:
            for name in by_cat.get(cat, []):
                handle.write(name + "\n")
    return names


def end_to_end():
    """``GroundTruth`` + ``build_records`` + ``sweep`` against ``eval_predictions``.

    The in-memory checks above validate the arithmetic. This validates the
    plumbing around it -- name keying between ``sub_img_name`` and the data
    list, the ``get_prediction_string`` sampling, the category files -- which is
    where a sweep goes wrong without the numbers looking obviously broken.
    """
    from fast_sweep import GroundTruth, build_records, sweep as fast
    try:
        from libs.datasets.metrics.culane_metric import CULaneMetric
    except Exception as exc:                                  # pragma: no cover
        print(f"  skip  end-to-end (CULaneMetric unavailable: {exc})")
        return True

    images = [
        ([FakeLane(400, 500), FakeLane(900, 820)], [0.95, 0.40],
         [lane(405, 505), lane(905, 825)], "test0_normal"),
        ([FakeLane(500, 600)], [0.70], [], "test7_cross"),
        ([FakeLane(600, 640), FakeLane(608, 648)], [0.90, 0.55],
         [lane(600, 640), lane(622, 662)], "test1_crowd"),
        ([], [], [lane(1200, 1150)], "test8_night"),
    ]

    ok = True
    with tempfile.TemporaryDirectory() as root:
        names = write_split(root, images)
        data_list = os.path.join(root, "list/test.txt")
        categories_dir = os.path.join(root, "list/test_split/")
        metric = CULaneMetric(data_root=root, data_list=data_list)
        dump = [(name, lanes, np.asarray(scores, dtype=np.float64))
                for name, (lanes, scores, _, _) in zip(names, images)]

        gt = GroundTruth(root, data_list, categories_dir)
        ok &= report(f"ground truth loaded ({len(gt)} images)", len(gt) == len(images))

        records = build_records(dump, gt, metric, jobs=1)
        ok &= report("one record per image", len(records) == len(images))
        ok &= report("categories came through",
                     [r["cat"] for r in records] == [im[3] for im in images])

        from fast_sweep import verify_official
        for threshold in (0.10, 0.50, 0.80):
            mine = fast(records, [threshold])[0]
            theirs = verify_official(dump, threshold, metric, root, data_list,
                                     categories_dir)
            same = all(int(mine[k]) == int(theirs[k])
                       for k in ("TP0.5", "FP0.5", "FN0.5"))
            same &= abs(mine["F1_0.5"] - theirs["F1_0.5"]) < 1e-9
            ok &= report(
                f"conf {threshold:.2f}: fast "
                f"({mine['TP0.5']}, {mine['FP0.5']}, {mine['FN0.5']}, "
                f"F1 {mine['F1_0.5'] * 100:.2f}) == eval_predictions "
                f"({theirs['TP0.5']}, {theirs['FP0.5']}, {theirs['FN0.5']}, "
                f"F1 {theirs['F1_0.5'] * 100:.2f})", same)
    return ok


def main():
    ok = True

    # ---------------------------------------------------------------- images
    images = []

    # 1. The ordinary case: three predictions, two annotations.
    images.append((
        [lane(400, 500), lane(900, 820), lane(1300, 1100)],
        [0.95, 0.60, 0.22],
        [lane(405, 505), lane(905, 825)],
        "test0_normal",
    ))

    # 2. No annotations. Every prediction is a false positive, and the IoU
    #    matrix has zero columns -- the shape the official code handles by
    #    an empty assignment and the fast path must handle the same way.
    images.append(([lane(500, 600), lane(1000, 950)], [0.80, 0.30], [],
                   "test7_cross"))

    # 3. Annotations with nothing predicted above most thresholds: pure recall
    #    loss, and at high thresholds a zero-row matrix.
    images.append(([lane(700, 700)], [0.15], [lane(700, 700), lane(1200, 1200)],
                   "test4_noline"))

    # 4. Scores sitting exactly on grid thresholds, where >= and > disagree.
    images.append((
        [lane(300, 360), lane(1100, 1040)], [0.50, 0.30],
        [lane(302, 362), lane(1098, 1038)], "test8_night",
    ))

    # 5. The reassignment case, and the reason this sweep caches the IoU matrix
    #    rather than the assignment. Prediction 0 lies exactly on annotation 0;
    #    prediction 1 lies 8 px off it, with annotation 1 a further 14 px away:
    #
    #        ious = [[1.000, 0.168],
    #                [0.588, 0.376]]
    #
    #    The Hungarian maximizes the total, so with both rows present it gives
    #    annotation 0 to prediction 0 and leaves prediction 1 with annotation 1
    #    at 0.376 -- below 0.5, so prediction 1 is a false positive. Raise the
    #    threshold past 0.90 and prediction 0 is gone; prediction 1 is now free
    #    to take annotation 0 at 0.588 and becomes a true positive. The matched
    #    column changes, and so does the hit. A sweep that cached each image's
    #    assignment would report the stale 0.376 forever.
    images.append((
        [lane(600, 640), lane(608, 648)], [0.95, 0.55],
        [lane(600, 640), lane(622, 662)], "test1_crowd",
    ))

    thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]
    records = fast_records(images)
    fast_rows = sweep(records, thresholds, per_category=True)

    print(f"{'conf':>6} {'fast TP/FP/FN':>18} {'official TP/FP/FN':>20} "
          f"{'fast F1':>9} {'official F1':>12}")
    print("-" * 70)
    all_match = True
    for threshold, row in zip(thresholds, fast_rows):
        want = official_counts(images, threshold)
        got = (row[f"TP{MAIN_IOU}"], row[f"FP{MAIN_IOU}"], row[f"FN{MAIN_IOU}"])
        match = tuple(int(v) for v in got) == tuple(int(v) for v in want)
        all_match &= match
        print(f"{threshold:6.2f} {str(got):>18} {str(want):>20} "
              f"{row['F1'] * 100:9.4f} {f1_of(*want) * 100:12.4f}"
              f"{'' if match else '   <- MISMATCH'}")
    ok &= report("counts match the official metric at every threshold", all_match)

    f1_match = all(
        abs(row["F1"] - f1_of(*official_counts(images, t))) < 1e-12
        for t, row in zip(thresholds, fast_rows))
    ok &= report("F1 matches to 1e-12", f1_match)

    # Case 5 only proves something if the assignment really moves. Assert the
    # column change *and* the hit flip, or a vacuous pass would hide the point.
    from scipy.optimize import linear_sum_assignment
    ious = records[4]["ious"]
    _, col_full = linear_sum_assignment(1 - ious)
    _, col_sub = linear_sum_assignment(1 - ious[np.array([False, True])])
    moved = int(col_full[1]) != int(col_sub[0])
    iou_before = float(ious[1, col_full[1]])
    iou_after = float(ious[1, col_sub[0]])
    print(f"\nreassignment: prediction 1 matched column {col_full[1]} at "
          f"IoU {iou_before:.3f} with the top row present, column {col_sub[0]} "
          f"at IoU {iou_after:.3f} without it")
    ok &= report("the reassignment case reassigns", moved)
    ok &= report("and the reassignment flips the hit, so it is observable",
                 (iou_before > MAIN_IOU) != (iou_after > MAIN_IOU))

    # Other IoU thresholds come free off the same cached matrix.
    for iou_thr in IOU_THRESHOLDS:
        t = 0.30
        want = official_counts(images, t, iou_thr)
        row = sweep(records, [t])[0]
        got = (row[f"TP{iou_thr}"], row[f"FP{iou_thr}"], row[f"FN{iou_thr}"])
        ok &= report(f"IoU {iou_thr} agrees as well {got}",
                     tuple(int(v) for v in got) == tuple(int(v) for v in want))

    # ------------------------------------------------------- the round trip
    # build_records parses get_prediction_string's text in-process instead of
    # writing it to disk. That shortcut is only safe if it agrees with
    # load_culane_img_data, including the 5-decimal rounding and the
    # fewer-than-2-points rule.
    text = "\n".join(
        " ".join(f"{x:.5f} {y:.5f}" for x, y in points)
        for points in (lane(400, 500, n=6), lane(900, 820, n=2)))
    text += "\n123.45678 99.00000"           # one point: must be dropped
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a.lines.txt")
        with open(path, "w") as handle:
            handle.write(text + "\n")
        want_lanes = load_culane_img_data(path)
    got_lanes = _parse_prediction_text(text)
    same = (len(got_lanes) == len(want_lanes)
            and all(np.allclose(np.array(a), np.array(b), atol=0, rtol=0)
                    for a, b in zip(got_lanes, want_lanes)))
    ok &= report(f"in-memory parse equals load_culane_img_data "
                 f"({len(got_lanes)} lanes, single-point lane dropped)", same)

    # Per-category aggregation, against the official formula on one category.
    cat = "test0_normal"
    sel = [im for im in images if im[3] == cat]
    tp, fp, fn = official_counts(sel, 0.10)
    ok &= report("per-category F1 matches",
                 abs(fast_rows[0][f"F1_{cat}_{MAIN_IOU}"] - f1_of(tp, fp, fn)) < 1e-12)

    # The single-channel canvas. This is the one optimization that touches the
    # IoU value itself rather than how often it is computed, so it is checked on
    # random geometry, not on the handful of cases above.
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        a = lane(*rng.uniform(0, ORI_W, 2), n=int(rng.integers(5, 40)))
        b = lane(*rng.uniform(0, ORI_W, 2), n=int(rng.integers(5, 40)))
        rgb = _image_ious(([a], [b], IMG_SHAPE))
        gray = _image_ious(([a], [b], FAST_MASK_SHAPE))
        worst = max(worst, float(np.abs(rgb - gray).max()))
    ok &= report(f"single-channel canvas gives identical IoU "
                 f"(max delta {worst:.1e} over 200 random pairs)", worst == 0.0)

    print("\nend to end, through GroundTruth / build_records / eval_predictions:")
    ok &= end_to_end()

    print("\nPASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
