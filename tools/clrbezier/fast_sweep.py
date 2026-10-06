"""Sweep a confidence threshold with the official CULane metric, once.

The bug this fixes
------------------
``sweep_threshold.evaluate_at`` writes every surviving prediction to a fresh
temp directory and calls the official ``eval_predictions``. That is correct,
and on one threshold it is the right thing to do. Called inside a sweep it is
quadratic waste: ``eval_predictions`` re-parses all 34,680 *annotation* files
and re-rasterizes every lane on every call, so the default grid
``[0.10, 0.96, 0.02]`` runs 44 full CULane evaluations — and ``fuse_eval``,
which sweeps two decode modes, runs 88. Each one prints its own
"Loading annotation data... / Calculating metric..." progress, which is the
"it keeps evaluating again and again" you are seeing. It is not stuck; it is
doing the same 34,680-image evaluation 88 times.

Why one pass is enough
----------------------
Look at what the threshold actually changes in ``culane_metric``:

    ious = discrete_cross_iou(interp_pred, interp_anno)   # <- threshold-free
    row, col = linear_sum_assignment(1 - ious)
    hits = pred_ious > iou_thr

The IoU between prediction *i* and annotation *j* is a property of two curves.
It does not depend on the confidence threshold. Raising the threshold only
*deletes rows* of that matrix. So the whole sweep is:

    rasterize once per image  ->  ious            (the expensive part)
    per threshold:  sub = ious[scores >= t]; Hungarian on sub; count

The Hungarian runs on a matrix of about 4x4, so a threshold costs microseconds
instead of a full evaluation. The arithmetic is not an approximation of the
official metric — it is the official metric's own code path applied to the
filtered row set, which is exactly the set ``eval_predictions`` would have been
given had the dump been written at that threshold. ``verify_official`` below
checks that claim against the real thing at one threshold, and the sweep should
not be trusted until it has.

Cost, measured in rasterizations of the 34,680-image test split: 88 before, 1
per decode mode now.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from libs.datasets.metrics.culane_metric import (
    discrete_cross_iou,
    eval_predictions,
    load_categories,
    load_culane_data,
)
from libs.utils.lane_utils import interp

IMG_SHAPE = (590, 1640, 3)
# The canvas `discrete_cross_iou` rasterizes onto for the *fast* path. The
# official shape has three channels, and `draw_lane` paints every one of them
# identically with color=(255,255,255), so dropping to a single channel triples
# both |A & B| and |A | B| -- and the ratio is unchanged. Verified exactly
# (max |delta| = 0 over 400 random lane pairs) in test_fast_sweep.py, and again
# end to end by `check_against_official`, which runs the real three-channel
# `eval_predictions`. It is 2.3x faster, and the IoU matrix is the whole cost of
# this module, so it is on by default; pass --rgb-masks to force the official
# canvas if you ever want to rule it out.
FAST_MASK_SHAPE = (590, 1640)
WIDTH = 30
IOU_THRESHOLDS = (0.1, 0.5, 0.75)
MAIN_IOU = 0.5
EPS = 1e-8


# --------------------------------------------------------------------------
# ground truth, loaded once



def lane_scores(lanes, scores) -> np.ndarray:
    """One score per returned lane.

    ``get_lanes`` returns every post-NMS score, but ``predictions_to_lanes``
    drops lanes with <= 1 row, so the two lists can differ in length and
    pairing them by position mis-scores the lanes after a dropped one. Each
    ``Lane`` carries its own score in ``metadata["conf"]``; use that.
    """
    confs = [(getattr(lane, "metadata", None) or {}).get("conf") for lane in lanes]
    if all(c is not None for c in confs):
        return np.asarray([float(c.item() if hasattr(c, "item") else c) for c in confs],
                          dtype=np.float64)
    scores = [float(s) for s in scores]
    if len(scores) != len(lanes):
        raise ValueError(f"{len(lanes)} lanes but {len(scores)} scores and no "
                         "metadata['conf'] to align them (use as_lanes=True)")
    return np.asarray(scores, dtype=np.float64)

def _key(name) -> str:
    """Data-list path and ``sub_img_name`` reduced to one comparable form."""
    name = str(name).lstrip("/")
    for suffix in (".lines.txt", ".jpg", ".png"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class GroundTruth:
    """Annotations and category labels for a split, parsed exactly once.

    This is the object whose repeated construction inside ``eval_predictions``
    was most of the cost: 34,680 file reads per threshold.
    """

    def __init__(self, data_root: str, data_list: str, categories_dir: str = None):
        categories_dir = categories_dir or os.path.join(data_root, "list/test_split/")
        data_cats, self.categories = load_categories(categories_dir)
        print(f"loading annotations once from {data_root}", flush=True)
        self.annos, self.cats = load_culane_data(data_root, data_list, data_cats)
        with open(data_list) as handle:
            names = [line.split()[0] for line in handle if line.strip()]
        self.index = {_key(n): i for i, n in enumerate(names)}
        if len(self.index) != len(self.annos):
            raise RuntimeError(
                f"data list has {len(names)} entries ({len(self.index)} unique) but "
                f"{len(self.annos)} annotations were loaded")

    def __len__(self):
        return len(self.annos)

    def lookup(self, sub_img_name):
        pos = self.index.get(_key(sub_img_name))
        if pos is None:
            raise KeyError(
                f"{sub_img_name!r} is not in the data list. The dump and the "
                f"evaluator are pointing at different splits.")
        return self.annos[pos], self.cats[pos]


# --------------------------------------------------------------------------
# the one expensive pass


def _parse_prediction_text(text: str):
    """``get_prediction_string`` output back to point lists.

    This is the round trip the official path performs through the filesystem
    (``get_prediction_string`` -> ``.lines.txt`` -> ``load_culane_img_data``).
    It is reproduced rather than bypassed because it is lossy in two ways that
    change the metric: coordinates are rounded to 5 decimals, and lanes with
    fewer than 2 points are dropped. Feeding the model's float64 lanes straight
    into ``interp`` would silently compute a slightly different number from the
    one the official evaluator reports.
    """
    lanes = []
    for line in text.splitlines():
        values = [float(v) for v in line.split()]
        lane = [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]
        if len(lane) >= 2:
            lanes.append(lane)
    return lanes


def _image_ious(task):
    """IoU matrix for one image. Runs in a worker; keep the payload small.

    ``img_shape`` travels in the task rather than through a module global
    because a spawned worker would not inherit the parent's global.
    """
    pred_pts, anno_pts, img_shape = task
    if len(pred_pts) == 0 or len(anno_pts) == 0:
        # Matches the official path, where an empty prediction or annotation
        # list yields a degenerate matrix and an empty assignment. The second
        # case is not hypothetical: test7_cross has no annotated lanes, so every
        # prediction there is a false positive by construction.
        return np.zeros((len(pred_pts), len(anno_pts)), dtype=np.float32)
    interp_pred = np.array([interp(lane, n=5) for lane in pred_pts], dtype=object)
    interp_anno = np.array([interp(lane, n=5) for lane in anno_pts], dtype=object)
    # The official function, unmodified -- only the canvas it draws on changes.
    return discrete_cross_iou(
        interp_pred, interp_anno, width=WIDTH, img_shape=img_shape
    ).astype(np.float32)


def build_records(dump, gt: GroundTruth, metric, jobs: int = None,
                  chunk: int = 1000, label: str = "",
                  img_shape=FAST_MASK_SHAPE):
    """Rasterize and cross-IoU every image once.

    Args:
        dump: ``[(sub_img_name, lanes, scores)]`` from one inference pass at the
            dump threshold. ``lanes`` are ``Lane`` objects, ``scores`` their
            confidences as a float array.
        gt: the shared :class:`GroundTruth`.
        metric: a ``CULaneMetric``, used only for ``get_prediction_string`` so
            the point sampling matches the official writer exactly.
        jobs: worker processes; ``None`` lets p_tqdm choose, ``1`` stays
            in-process (use it when debugging, the traceback is readable).
        chunk: images per batch handed to the pool. The point lists are far
            larger than the matrices they produce, so streaming in chunks keeps
            peak memory flat instead of holding the whole split at once.

    Returns:
        ``[dict(name, ious, scores, n_gt, cat)]`` — a few hundred bytes per
        image, cheap to sweep over as many times as you like.
    """
    records = []
    total = len(dump)
    for start in range(0, total, chunk):
        batch = dump[start:start + chunk]
        tasks, meta = [], []
        for name, lanes, scores in batch:
            anno, cat = gt.lookup(name)
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            if len(lanes) != len(scores):
                raise ValueError(
                    f"{name}: {len(lanes)} lanes but {len(scores)} scores")
            # One lane at a time, so a dropped lane drops its score with it and
            # the rows of `ious` stay aligned with `keep` in the sweep.
            pred_pts, kept_scores = [], []
            for lane, score in zip(lanes, scores):
                text = metric.get_prediction_string([lane])
                parsed = _parse_prediction_text(text) if text else []
                if parsed:
                    pred_pts.append(parsed[0])
                    kept_scores.append(score)
            tasks.append((pred_pts, anno, img_shape))
            meta.append((name, np.asarray(kept_scores, dtype=np.float64),
                         len(anno), cat))

        if jobs == 1:
            ious = [_image_ious(t) for t in tasks]
        else:
            from p_tqdm import p_map
            ious = p_map(_image_ious, tasks, num_cpus=jobs,
                         desc=f"{label}rasterize {start + len(batch)}/{total}")

        for (name, scores, n_gt, cat), matrix in zip(meta, ious):
            records.append({"name": name, "ious": matrix, "scores": scores,
                            "n_gt": n_gt, "cat": cat})
    return records


# --------------------------------------------------------------------------
# the cheap part


def _hits(record, threshold: float, iou_thresholds=IOU_THRESHOLDS):
    """Per-prediction TP flags, by the official rule, on the filtered rows."""
    ious, scores = record["ious"], record["scores"]
    sub = ious[scores >= threshold] if scores.size else ious[:0]
    pred_ious = np.zeros(sub.shape[0])
    if sub.shape[0] and sub.shape[1]:
        row, col = linear_sum_assignment(1 - sub)
        pred_ious[row] = sub[row, col]
    return [pred_ious > thr for thr in iou_thresholds]


def sweep(records, thresholds, iou_thresholds=IOU_THRESHOLDS, per_category=False):
    """F1 / precision / recall at each threshold, aggregated as the official
    evaluator does: pooled over the whole split, not averaged over images.

    Returns ``[dict]``, one per threshold, keyed like ``eval_predictions``'
    result dict (``F1_0.5``, ``Precision0.5``, ...) plus a plain ``F1`` alias
    for the main IoU threshold.
    """
    rows = []
    for threshold in thresholds:
        hits = [_hits(r, threshold, iou_thresholds) for r in records]
        out = {"threshold": float(threshold)}
        for k, iou_thr in enumerate(iou_thresholds):
            tp = sum(int(h[k].sum()) for h in hits)
            n_pred = sum(int(h[k].size) for h in hits)
            n_gt = sum(r["n_gt"] for r in records)
            fp = n_pred - tp
            prec = tp / (tp + fp + EPS)
            rec = tp / (n_gt + EPS)
            f1 = 2 * prec * rec / (prec + rec + EPS)
            out.update({f"TP{iou_thr}": tp, f"FP{iou_thr}": fp,
                        f"FN{iou_thr}": n_gt - tp,
                        f"Precision{iou_thr}": prec, f"Recall{iou_thr}": rec,
                        f"F1_{iou_thr}": f1})
            if iou_thr == MAIN_IOU:
                out.update({"F1": f1, "Precision": prec, "Recall": rec})
            # Every IoU level, not just the main one. Gating this on MAIN_IOU
            # emitted only F1_<cat>_0.5 keys, so a caller asking for the 0.75
            # breakdown got None for every category and printed an empty table --
            # the per-level hits are already computed, so the restriction bought
            # nothing and silently withheld the level that matters most here.
            if per_category:
                cats = sorted({r["cat"] for r in records})
                for cat in cats:
                    sel = [(r, h) for r, h in zip(records, hits) if r["cat"] == cat]
                    c_tp = sum(int(h[k].sum()) for _, h in sel)
                    c_pred = sum(int(h[k].size) for _, h in sel)
                    c_gt = sum(r["n_gt"] for r, _ in sel)
                    c_prec = c_tp / (c_pred + EPS)
                    c_rec = c_tp / (c_gt + EPS)
                    out[f"F1_{cat}_{iou_thr}"] = (
                        2 * c_prec * c_rec / (c_prec + c_rec + EPS))
        rows.append(out)
    return rows


# --------------------------------------------------------------------------
# the audit


def verify_official(dump, threshold, metric, data_root, data_list,
                    categories_dir, logger=None):
    """The official ``eval_predictions`` at one threshold, via the filesystem.

    Kept for two jobs: confirming the fast sweep (``--verify``) and producing
    the per-category table at the single threshold you end up reporting. Never
    call it in a loop — that is the mistake this module exists to undo.
    """
    result_dir = tempfile.mkdtemp(prefix="sweep_")
    try:
        for sub_name, lanes, scores in dump:
            keep = [lane for lane, score in zip(lanes, np.asarray(scores).reshape(-1))
                    if score >= threshold]
            path = os.path.join(result_dir, _key(sub_name) + ".lines.txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            text = metric.get_prediction_string(keep)
            with open(path, "w") as handle:
                if text:
                    handle.write(text + "\n")
        return eval_predictions(result_dir, data_root, data_list, categories_dir,
                                logger=logger)
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


def check_against_official(records, dump, threshold, metric, data_root,
                           data_list, categories_dir, logger=None,
                           tol: float = 1e-6) -> bool:
    """Fast sweep against the official evaluator at one threshold.

    A sweep that silently disagrees with the official number is worse than a
    slow one, so this prints both and says plainly whether they match.
    """
    fast = sweep(records, [threshold])[0]
    print(f"\nverifying the fast sweep at conf {threshold:.2f} against "
          f"eval_predictions...", flush=True)
    official = verify_official(dump, threshold, metric, data_root, data_list,
                               categories_dir, logger=logger)
    ok = True
    print(f"\n{'':>12} {'fast':>12} {'official':>12} {'delta':>10}")
    for key in (f"TP{MAIN_IOU}", f"FP{MAIN_IOU}", f"FN{MAIN_IOU}", f"F1_{MAIN_IOU}"):
        if key not in official:
            continue
        a, b = float(fast[key]), float(official[key])
        delta = a - b
        ok &= abs(delta) <= tol * max(1.0, abs(b))
        print(f"{key:>12} {a:12.6f} {b:12.6f} {delta:10.2e}")
    print("fast sweep reproduces the official metric" if ok else
          "MISMATCH — do not trust the sweep; the records are built wrong")
    return ok


# --------------------------------------------------------------------------
# the driver the three tools share


def run_sweep(dumps: dict, thresholds, data_root: str, data_list: str,
              categories_dir: str, metric, jobs: int = None,
              verify: bool = True, logger=None, rgb_masks: bool = False,
              return_records: bool = False):
    """Sweep one or more dumps over one threshold grid.

    ``dumps`` maps a label ("nms", "fuse", "baseline", "probe") to a dump from
    the same inference pass. Annotations are loaded once for all of them, each
    dump is rasterized once, and every threshold after that is arithmetic.

    Returns ``(rows_by_label, best_by_label)`` where ``rows_by_label[label]`` is
    the list of per-threshold dicts and ``best_by_label[label]`` is
    ``(threshold, f1)``.
    """
    gt = GroundTruth(data_root, data_list, categories_dir)
    labels = list(dumps)
    records, rows, best = {}, {}, {}
    for label in labels:
        prefix = f"[{label}] " if len(labels) > 1 else ""
        print(f"\n{prefix}computing IoU matrices once over "
              f"{len(dumps[label])} images", flush=True)
        records[label] = build_records(
            dumps[label], gt, metric, jobs=jobs, label=prefix,
            img_shape=IMG_SHAPE if rgb_masks else FAST_MASK_SHAPE)
        rows[label] = sweep(records[label], thresholds)
        peak = max(rows[label], key=lambda r: r["F1"])
        best[label] = (peak["threshold"], peak["F1"])

    if verify:
        label = labels[0]
        partial = len(dumps[label]) < len(gt)
        if partial:
            # eval_predictions walks the whole data list and reads a prediction
            # file for every entry, so it cannot score a --max-images subset:
            # the missing images would either crash it or silently count as
            # pure recall loss. The fast path scores exactly the images it has,
            # which is what you want for a quick look, but it means the two are
            # measuring different sets and the comparison would be meaningless.
            print(f"\nskipping the official cross-check: the dump covers "
                  f"{len(dumps[label])} of {len(gt)} images, and "
                  f"eval_predictions always scores the full list. Drop "
                  f"--max-images for a verifiable run.")
        else:
            check_against_official(records[label], dumps[label], best[label][0],
                                   metric, data_root, data_list, categories_dir,
                                   logger=logger)
    # The cached IoU matrices cost one rasterization pass each; handing them back
    # lets a caller re-aggregate (per category, at another IoU level) for free
    # instead of paying for that pass again.
    if return_records:
        return rows, best, records
    return rows, best
