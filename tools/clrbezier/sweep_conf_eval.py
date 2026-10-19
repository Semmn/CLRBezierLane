"""Evaluate one checkpoint over a confidence-threshold range with a fixed step.

    python tools/clrbezier/sweep_conf_eval.py CONFIG CHECKPOINT --range 0.80 0.90 --step 0.02

Inference runs ONCE, at the lowest threshold of the range. Every higher threshold
is then obtained by dropping the lanes whose score is below it.

The evaluation is also done once for CurveLanes and CULane. Their metrics match
predictions to annotations per image by the Hungarian algorithm on 1 - IoU, and
the IoU of a prediction/annotation pair does not depend on the threshold: a
higher threshold only removes predictions. So the IoU matrices are computed once
(the expensive part: drawing every lane) and each threshold only re-solves the
matching on the surviving predictions (re-solved, not reused: removing a winner
can hand its lane to another prediction).

    CurvelanesMetric  cached here, with vega's own lane drawing; checked against
                      CurvelanesMetric on a subset of images before the sweep
                      is reported (--check-images), exact to the last bit; also
                      its subclasses that only map the lanes in process(), e.g.
                      FoveatedCurvelanesMetric (libs/datasets/foveation.py)
    CULaneMetric      tools/clrbezier/fast_sweep.py (build_records + sweep)
    LLAMAS, TuSimple  the config's evaluator re-run per threshold

``--reference`` re-runs the config's evaluator per threshold for every dataset
(the old behaviour), and ``--verify`` re-runs it once on the whole split at the
best threshold to confirm the cached number.

Why filtering afterwards is exact
---------------------------------
``CLRerHead.get_lanes`` drops scores below ``conf_threshold`` and then runs greedy
lane NMS with ``nms_topk``. Greedy NMS visits candidates in score order and a
candidate can only be suppressed by a kept candidate with a higher score, so the
decisions for lanes above t never depend on lanes below t; the ``nms_topk`` cap
keeps a score-ordered prefix, so the lanes above t in the low-threshold output
are exactly the first ones the high-threshold run would keep. (The NMS kernel
reads start, length and xs only; the batch-0 ``cls_logits`` it is handed are not
used.) ``predictions_to_lanes`` drops lanes with <= 1 row regardless of the
threshold. Each surviving ``Lane`` carries its own score in ``metadata["conf"]``,
which is what the filter uses, so scores stay aligned even when a lane is dropped.

Use ``--save-dump`` once and ``--load-dump`` afterwards to sweep other ranges
without running the model again (the dump must come from a run whose lowest
threshold is at or below the new range).

CPU workers for the one IoU pass: ``--workers`` (default: up to 8).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import os.path as osp
import pickle
import re
import shutil
import sys
import time

import numpy as np


# ----------------------------------------------------------------------------
# threshold grid / dump handling (no mmdet needed: unit-testable)
# ----------------------------------------------------------------------------
def make_thresholds(start, stop, step):
    if step <= 0:
        raise ValueError("--step must be positive")
    if stop < start:
        raise ValueError("--range STOP must be >= START")
    n = int(math.floor((stop - start) / step + 1e-6)) + 1
    ts = [round(start + k * step, 6) for k in range(n)]
    if (stop - start) > 1e-9 and n == 1:
        print(f"[warn] step {step} is larger than the range {start}-{stop}; "
              f"only {start} will be evaluated")
    return ts


def lane_score(lane):
    meta = getattr(lane, "metadata", None) or {}
    conf = meta.get("conf")
    if conf is None:
        return None
    return float(conf.item() if hasattr(conf, "item") else conf)


def _plain(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.tolist() if v.ndim else v.item()
    if isinstance(v, (list, tuple)):
        return type(v)(_plain(x) for x in v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def compact_result(result):
    """One predict() output -> (metainfo, [(points [N,2] float64, score), ...])."""
    lanes = result["lanes"]
    scores = result.get("scores", [])
    per_lane = [lane_score(l) for l in lanes]
    if any(s is None for s in per_lane):
        # Lanes without metadata (as_lanes=False): fall back to positional scores,
        # which is only valid if predictions_to_lanes dropped nothing.
        scores = [float(s) for s in scores]
        if len(scores) != len(lanes):
            raise RuntimeError(
                "lanes carry no metadata['conf'] and len(lanes) != len(scores); "
                "set model.test_cfg.as_lanes=True")
        per_lane = scores
    out = []
    for lane, s in zip(lanes, per_lane):
        pts = getattr(lane, "points", lane)
        if hasattr(pts, "detach"):
            pts = pts.detach().cpu().numpy()
        out.append((np.asarray(pts, dtype=np.float64), float(s)))
    meta = {k: _plain(v) for k, v in dict(result["metainfo"]).items()}
    return meta, out


def filtered_samples(dump, threshold, lane_cls):
    """Rebuild predict()-style outputs keeping lanes with score >= threshold."""
    samples = []
    for meta, lanes in dump:
        kept = [(p, s) for p, s in lanes if s >= threshold]
        samples.append(dict(
            lanes=[lane_cls(points=p.copy(), metadata={"conf": s}) for p, s in kept],
            scores=[s for _, s in kept],
            metainfo=meta,
        ))
    return samples


def pick_key(metrics, requested=None):
    if requested:
        if requested not in metrics:
            raise KeyError(f"--metric-key {requested!r} not in {sorted(metrics)}")
        return requested
    # CULaneMetric reports F1_0.1 / F1_0.5 / F1_0.75 and no plain F1, so F1_0.5
    # must come before the alphabetical fallback (which would pick F1_0.1).
    for key in ("F1", "F1_0.5", "F1_0.50", "F1_measure", "F1@50", "f1"):
        if key in metrics:
            return key
    f1s = [k for k in metrics if "f1" in k.lower()]
    if f1s:
        return sorted(f1s)[0]
    raise KeyError(f"no F1-like key in {sorted(metrics)}; pass --metric-key")


def sweep(dump, thresholds, metric_factory, lane_cls, metric_key=None, log=print):
    rows = []
    key = metric_key
    for t in thresholds:
        tic = time.time()
        metric = metric_factory()
        metric.results = []
        samples = filtered_samples(dump, t, lane_cls)
        metric.process({}, samples)
        try:
            res = metric.compute_metrics(metric.results)
        finally:
            # CULaneMetric writes one .lines.txt per image to a temp dir it never
            # removed (official behaviour); one copy per threshold fills /tmp.
            tmp = getattr(metric, "result_dir", None)
            if tmp and not getattr(metric, "output_dir", None):
                shutil.rmtree(tmp, ignore_errors=True)
        res = {k: float(v) for k, v in res.items()
               if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)}
        key = key or pick_key(res, metric_key)
        n_lanes = sum(len(s["lanes"]) for s in samples)
        rows.append(dict(threshold=t, lanes=n_lanes, seconds=round(time.time() - tic, 1), **res))
        log(f"conf {t:.4f}: {key} = {res.get(key, float('nan')):.4f}  ({n_lanes} lanes)")
    return rows, key


def lanes_above(dump, t):
    """Lanes of the dump with score >= t (the ``lanes`` column of the reference sweep)."""
    return sum(1 for _, lanes in dump for _, s in lanes if s >= t)


# ----------------------------------------------------------------------------
# CurveLanes: IoU matrices once, Hungarian per threshold
# ----------------------------------------------------------------------------
_CL = {}   # read by the forked workers: metric, lane class, dump


def curvelanes_cache_supported(metric):
    """CurvelanesMetric, with or without the F1@ thresholds of curvelanes_f1_thresholds.patch, or a subclass
    that only maps the lanes in process() (FoveatedCurvelanesMetric): the cache calls process() per image."""
    return any(c.__name__ == "CurvelanesMetric" for c in type(metric).__mro__) and all(
        hasattr(metric, a) for a in ("process", "parse_anno", "iou_thresh", "eval_width",
                                     "eval_height", "lane_width"))


def _vega_prf(hits, pr_num, gt_num):
    """vega LaneMetricCore.summary (CurvelanesMetric._prf where the patch adds it)."""
    eps = sys.float_info.epsilon
    precision = hits / (pr_num + eps)
    recall = hits / (gt_num + eps)
    return precision, recall, 2 * precision * recall / (precision + recall + eps)


def _vega_mask(lane, height, width, lane_width):
    """One lane drawn exactly as vega's ``calc_iou`` draws it, flattened."""
    import cv2
    from vega.metrics.pytorch.lane_metric import spline_interp

    im = np.zeros((height, width), np.uint8)
    pts = spline_interp(lane=lane, step_t=1)
    for i in range(0, len(pts) - 1):
        cv2.line(im, (int(pts[i]["x"]), int(pts[i]["y"])),
                 (int(pts[i + 1]["x"]), int(pts[i + 1]["y"])), 255, lane_width)
    return im.reshape(-1) > 0


def curvelanes_iou_matrix(gt_lines, pr_lines, ori_shape, metric):
    """vega ``calc_iou`` for every (GT, prediction) pair, each lane drawn once.

    ``calc_iou`` divides sum(A & B) by sum(A | B) on 0/255 masks (0 when the union
    is empty); the factor 255 cancels exactly in float64, so pixel counts give the
    same numbers bit for bit (checked in test_sweep_conf_eval_cached.py).
    """
    iou = np.zeros((len(gt_lines), len(pr_lines)))
    if not gt_lines or not pr_lines:   # vega draws nothing for these images either
        return iou
    from vega.metrics.pytorch.lane_metric import resize_lane

    h, w, lw = int(metric.eval_height), int(metric.eval_width), int(metric.lane_width)
    y_ratio = np.true_divide(ori_shape[0], metric.eval_height)
    x_ratio = np.true_divide(ori_shape[1], metric.eval_width)
    gt = np.stack([_vega_mask(resize_lane(l, x_ratio, y_ratio), h, w, lw) for l in gt_lines])
    pr = np.stack([_vega_mask(resize_lane(l, x_ratio, y_ratio), h, w, lw) for l in pr_lines])
    cols = np.flatnonzero(gt.any(0))   # an intersection can only lie on GT pixels
    inter = gt[:, cols].astype(np.int64) @ pr[:, cols].astype(np.int64).T
    union = gt.sum(1, dtype=np.int64)[:, None] + pr.sum(1, dtype=np.int64)[None, :] - inter
    np.divide(inter, union, out=iou, where=union > 0)
    return iou


def _curvelanes_image(i):
    """(GT lane count, scores [P], IoU [G, P]) of image i, P = the predictions
    CurvelanesMetric would score (lanes that convert to no points are dropped)."""
    metric, lane_cls = _CL["metric"], _CL["lane_cls"]
    sample = filtered_samples([_CL["dump"][i]], -math.inf, lane_cls)[0]
    metric.results = []
    metric.process({}, [sample])   # the metric's own crop/convert, one entry per lane
    r = metric.results.pop()
    keep = [j for j, line in enumerate(r["pred"]) if len(line) > 0]
    gt_lines = [lane for lane in metric.parse_anno(r["filename"]) if len(lane) > 0]
    iou = curvelanes_iou_matrix(gt_lines, [r["pred"][j] for j in keep], r["ori_shape"], metric)
    return len(gt_lines), np.asarray([sample["scores"][j] for j in keep], dtype=np.float64), iou


def _worker_init():
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass


def build_curvelanes_cache(dump, metric, lane_cls, workers=1, log=print):
    """One pass over the dump: draws every lane once per image."""
    _CL.update(metric=metric, lane_cls=lane_cls, dump=dump)
    tic = time.time()
    try:
        ctx = None
        if workers > 1 and len(dump) > 1:
            import multiprocessing as mp
            try:
                ctx = mp.get_context("fork")   # workers inherit _CL; nothing is pickled in
            except ValueError:
                ctx = None
        cache = []
        if ctx is None:
            items = map(_curvelanes_image, range(len(dump)))
            pool = None
        else:
            pool = ctx.Pool(workers, initializer=_worker_init)
            items = pool.imap(_curvelanes_image, range(len(dump)), chunksize=16)
        try:
            for item in items:
                cache.append(item)
                if len(cache) % 2000 == 0:
                    log(f"  IoU matrices: {len(cache)}/{len(dump)} images", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    finally:
        _CL.clear()
    log(f"IoU matrices for {len(cache)} images in {time.time() - tic:.0f} s", flush=True)
    return cache


def curvelanes_sweep(cache, dump, thresholds, metric, metric_key=None, log=print):
    """CurvelanesMetric.compute_metrics at every threshold from the cached matrices."""
    from scipy.optimize import linear_sum_assignment

    prf = getattr(metric, "_prf", _vega_prf)
    main = float(metric.iou_thresh)
    extra = sorted(set(float(t) for t in (getattr(metric, "iou_thresholds", None) or ())))
    levels = sorted(set(extra) | {main})
    gt_num = sum(n_gt for n_gt, _, _ in cache)
    rows, key = [], metric_key
    for t in thresholds:
        tic = time.time()
        hits = dict.fromkeys(levels, 0)
        pr_num = 0
        for n_gt, scores, iou in cache:
            keep = scores >= t
            n = int(keep.sum())
            pr_num += n
            if n_gt and n:
                sub = iou[:, keep]
                r, c = linear_sum_assignment(1 - sub)   # vega: cost 1 - IoU, GT x prediction
                matched = sub[r, c]
                for lv in levels:
                    hits[lv] += int((matched > lv).sum())   # vega: iou_val > iou_thresh
        precision, recall, f1 = prf(hits[main], pr_num, gt_num)
        res = dict(f1_measure=f1, precision=precision, recall=recall)
        for lv in extra:
            res[f"F1@{int(round(lv * 100))}"] = prf(hits[lv], pr_num, gt_num)[2]
        res = {k: float(v) for k, v in res.items()}
        key = key or pick_key(res, metric_key)
        n_lanes = lanes_above(dump, t)
        rows.append(dict(threshold=t, lanes=n_lanes, seconds=round(time.time() - tic, 1), **res))
        log(f"conf {t:.4f}: {key} = {res.get(key, float('nan')):.4f}  ({n_lanes} lanes)")
    return rows, key


def _metric_values(row):
    return {k: v for k, v in row.items() if k not in ("threshold", "seconds")}


def compare_rows(cached, reference, tol=0.0):
    """[(threshold, key, cached, reference)] where the two sweeps disagree."""
    bad = []
    for a, b in zip(cached, reference):
        va, vb = _metric_values(a), _metric_values(b)
        for k in sorted(set(va) | set(vb)):
            x, y = va.get(k), vb.get(k)
            if x is None or y is None or abs(float(x) - float(y)) > tol * max(1.0, abs(float(y))):
                bad.append((a["threshold"], k, x, y))
    return bad


def check_curvelanes_cache(dump, cache, thresholds, metric, factory, lane_cls, n_images,
                           log=print):
    """Cached vs CurvelanesMetric on the first ``n_images`` images, at the lowest,
    middle and highest threshold. Must agree exactly, or nothing is reported."""
    n = min(int(n_images), len(dump))
    if n <= 0:
        return
    probe = sorted({thresholds[0], thresholds[len(thresholds) // 2], thresholds[-1]})
    tic = time.time()
    quiet = lambda *a, **k: None   # noqa: E731
    ref, _ = sweep(dump[:n], probe, factory, lane_cls, log=quiet)
    fast, _ = curvelanes_sweep(cache[:n], dump[:n], probe, metric, log=quiet)
    bad = compare_rows(fast, ref)
    if bad:
        lines = "\n".join(f"  conf {t:.4f} {k}: cached {x} vs CurvelanesMetric {y}"
                          for t, k, x, y in bad[:10])
        raise RuntimeError(
            f"the cached CurveLanes evaluation differs from CurvelanesMetric on the first "
            f"{n} images:\n{lines}\nRun with --reference (evaluator per threshold) and "
            "report this.")
    log(f"check: cached == CurvelanesMetric on {n} images at conf "
        f"{', '.join(f'{t:g}' for t in probe)} ({time.time() - tic:.0f} s)", flush=True)


# ----------------------------------------------------------------------------
# CULane: tools/clrbezier/fast_sweep.py
# ----------------------------------------------------------------------------
def import_fast_sweep():
    try:
        import fast_sweep   # this script's folder (tools/clrbezier/) is on sys.path
        return fast_sweep
    except ImportError:
        pass
    try:
        from tools.clrbezier import fast_sweep
        return fast_sweep
    except ImportError:
        return None


def culane_cache_supported(metric):
    return type(metric).__name__ == "CULaneMetric" and all(
        hasattr(metric, a) for a in ("img_prefix", "list_path", "test_categories_dir",
                                     "get_prediction_string"))


def culane_fast_dump(dump, lane_cls):
    """fast_sweep's dump layout: [(sub_img_name, [Lane], scores)]."""
    return [(meta["sub_img_name"],
             [lane_cls(points=p.copy(), metadata={"conf": s}) for p, s in lanes],
             np.asarray([s for _, s in lanes], dtype=np.float64))
            for meta, lanes in dump]


def culane_sweep(fs, records, dump, thresholds, metric_key=None, log=print):
    """fast_sweep.sweep per threshold, keyed like CULaneMetric (eval_predictions)."""
    rows, key = [], metric_key
    for t in thresholds:
        tic = time.time()
        out = fs.sweep(records, [t], per_category=True)[0]
        # fast_sweep adds F1 / Precision / Recall aliases of the 0.5 numbers;
        # eval_predictions does not, so the report keeps ranking by F1_0.5
        res = {k: float(v) for k, v in out.items()
               if k not in ("threshold", "F1", "Precision", "Recall")}
        key = key or pick_key(res, metric_key)
        n_lanes = lanes_above(dump, t)
        rows.append(dict(threshold=t, lanes=n_lanes, seconds=round(time.time() - tic, 1), **res))
        log(f"conf {t:.4f}: {key} = {res.get(key, float('nan')):.4f}  ({n_lanes} lanes)")
    return rows, key


# ----------------------------------------------------------------------------
# inference (mmdet / mmengine)
# ----------------------------------------------------------------------------
def run_inference(cfg_path, checkpoint, conf, split, cfg_options, max_images, work_dir):
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(cfg_path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint
    cfg.model.setdefault("test_cfg", {})
    cfg.model.test_cfg["conf_threshold"] = conf
    cfg.model.test_cfg["as_lanes"] = True
    if split == "val":
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator
    # Runner handles custom_imports, default_scope, data_preprocessor and the
    # checkpoint exactly as tools/test.py does.
    runner = Runner.from_cfg(cfg)
    import torch
    # Inference only. The configs set randomness.deterministic=True, which makes the Runner
    # call torch.use_deterministic_algorithms(True); tools/train.py turns it off again after
    # building its runner, and so does this (BLRNet's CUDA cumsum has no deterministic kernel).
    torch.use_deterministic_algorithms(False)
    runner.load_or_resume()
    model = runner.model
    model.eval()

    dump, seen = [], 0
    with torch.no_grad():
        for data in runner.test_dataloader:
            for result in model.test_step(data):
                dump.append(compact_result(result))
            seen = len(dump)
            if seen % 2000 < runner.test_dataloader.batch_size:
                print(f"  {seen} images", flush=True)
            if max_images and seen >= max_images:
                break
    evaluator_cfg = cfg.test_evaluator
    return dump, evaluator_cfg


def metric_factory_from(evaluator_cfg, index):
    from mmdet.registry import METRICS
    import libs.datasets  # noqa: F401  (registers the lane metrics)
    try:   # FoveatedCurvelanesMetric; inference imports it through custom_imports, --load-dump does not
        import libs.datasets.foveation  # noqa: F401
    except ImportError:
        pass

    cfgs = evaluator_cfg if isinstance(evaluator_cfg, (list, tuple)) else [evaluator_cfg]
    metric_cfg = copy.deepcopy(cfgs[index])
    return lambda: METRICS.build(copy.deepcopy(metric_cfg))


def default_work_dir(config):
    """configs/clrbezier/culane/x.py -> work_dirs/clrbezier/culane/x (as tools/train.py lays out runs)."""
    parts = osp.normpath(osp.splitext(config)[0]).split(os.sep)
    if "configs" in parts:
        parts = parts[len(parts) - 1 - parts[::-1].index("configs") + 1:]
    else:
        parts = parts[-1:]
    return osp.join("work_dirs", *parts)


def format_report(rows, key, best, info):
    # precision/recall (either capitalization) and any other F1@xx the metric
    # reports, e.g. CurvelanesMetric's F1@75 next to the ranked F1@50
    cand = ["Precision", "Recall", "precision", "recall"]
    # CULaneMetric: Precision0.5 / Recall0.5 of the ranked F1_0.5, and the
    # overall F1 at its other IoU thresholds (F1_0.1, F1_0.75)
    m = re.fullmatch(r"F1_(\d+(?:\.\d+)?)", key)
    if m:
        cand += [f"Precision{m.group(1)}", f"Recall{m.group(1)}"]
    cand += sorted({k for r in rows for k in r
                    if k.startswith("F1@") or re.fullmatch(r"F1_\d+(?:\.\d+)?", k)})
    extra = [k for k in cand if k != key and any(k in r for r in rows)]
    width = {k: max(10, len(k)) for k in extra}
    lines = ["=" * 64, f"threshold sweep  {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    for name, value in info.items():
        if value not in (None, ""):
            lines.append(f"{name:>10}: {value}")
    lines.append("-" * 64)
    lines.append(f"{'conf':>8} {key:>12}" + "".join(f" {k:>{width[k]}}" for k in extra) + f" {'lanes':>8}")
    for r in rows:
        mark = "  <- best" if r is best else ""
        lines.append(f"{r['threshold']:8.4f} {r.get(key, float('nan')):12.4f}"
                     + "".join(f" {r.get(k, float('nan')):{width[k]}.4f}" for k in extra)
                     + f" {r['lanes']:8d}{mark}")
    lines.append("-" * 64)
    lines.append(f"best in range: conf {best['threshold']:.4f} -> {key} {best[key]:.4f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("checkpoint", nargs="?", default=None,
                    help="not needed with --load-dump")
    ap.add_argument("--range", type=float, nargs=2, required=True, metavar=("START", "STOP"))
    ap.add_argument("--step", type=float, required=True)
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="which dataloader/evaluator from the config")
    ap.add_argument("--metric-key", default=None,
                    help="result key to rank by (default: the F1 key the metric reports)")
    ap.add_argument("--metric-index", type=int, default=0,
                    help="which evaluator entry, if the config lists several")
    ap.add_argument("--save-dump", default=None, help="pickle the single inference pass")
    ap.add_argument("--load-dump", default=None, help="reuse a saved pass, skip inference")
    ap.add_argument("--max-images", type=int, default=None, help="debug only")
    ap.add_argument("--work-dir", default=None,
                    help="default: work_dirs/<config path under configs/>, e.g. "
                         "work_dirs/clrbezier/culane/clrbezier_anchored_o2m_r34")
    ap.add_argument("--out", default=None,
                    help="text report path (default: <work-dir>/threshold_sweep.txt); "
                         "a .csv and .json with the same stem are written next to it")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the text report instead of appending this run to it")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="CPU processes for the one IoU pass (CurveLanes, CULane)")
    ap.add_argument("--reference", action="store_true",
                    help="re-run the config's evaluator at every threshold (no IoU cache)")
    ap.add_argument("--verify", action="store_true",
                    help="also run the config's evaluator on the whole split at the best "
                         "threshold and compare (costs one full evaluation)")
    ap.add_argument("--check-images", type=int, default=50,
                    help="CurveLanes: images on which the cached evaluation is checked "
                         "against CurvelanesMetric before the sweep (0: skip)")
    from mmengine.config import DictAction
    ap.add_argument("--cfg-options", nargs="+", action=DictAction,
                    help="key=value overrides, as in tools/test.py")
    args = ap.parse_args()

    thresholds = make_thresholds(args.range[0], args.range[1], args.step)
    print(f"thresholds ({len(thresholds)}): {', '.join(f'{t:g}' for t in thresholds)}")

    cfg_options = args.cfg_options

    work_dir = args.work_dir or default_work_dir(args.config)
    os.makedirs(work_dir, exist_ok=True)
    out = args.out or osp.join(work_dir, "threshold_sweep.txt")
    if osp.dirname(out):
        os.makedirs(osp.dirname(out), exist_ok=True)

    if args.load_dump:
        with open(args.load_dump, "rb") as f:
            saved = pickle.load(f)
        if saved["conf"] > thresholds[0] + 1e-9:
            raise ValueError(f"dump was made at conf >= {saved['conf']}, above the range start "
                             f"{thresholds[0]}; lanes below it are missing")
        dump, evaluator_cfg = saved["dump"], saved["evaluator_cfg"]
        # the metric comes from the dump, so label the report with its split
        if saved.get("split") and saved["split"] != args.split:
            print(f"[info] the dump is from the {saved['split']} split; reporting it as such")
        args.split = saved.get("split", args.split)
        if cfg_options:
            print("[warn] --cfg-options is ignored with --load-dump (no inference is run)")
        print(f"loaded {len(dump)} images from {args.load_dump} (conf >= {saved['conf']})")
    else:
        if not args.checkpoint:
            ap.error("checkpoint is required unless --load-dump is given")
        print(f"inference once at conf >= {thresholds[0]} ({args.split} split)...")
        dump, evaluator_cfg = run_inference(args.config, args.checkpoint, thresholds[0],
                                            args.split, cfg_options, args.max_images, work_dir)
        evaluator_cfg = _plain(dict(evaluator_cfg)) if not isinstance(evaluator_cfg, (list, tuple)) \
            else [_plain(dict(e)) for e in evaluator_cfg]
        if args.save_dump:
            with open(args.save_dump, "wb") as f:
                pickle.dump(dict(conf=thresholds[0], dump=dump, evaluator_cfg=evaluator_cfg,
                                 config=args.config, checkpoint=args.checkpoint,
                                 split=args.split), f)
            print(f"saved the pass to {args.save_dump}")
    if args.max_images:
        print("[warn] --max-images evaluates against the FULL annotation set; numbers are not valid")

    from libs.utils.lane_utils import Lane
    factory = metric_factory_from(evaluator_cfg, args.metric_index)
    metric = factory()
    # CULaneMetric makes a temp dir when it is built; this instance never scores
    tmp = getattr(metric, "result_dir", None)
    if tmp and not getattr(metric, "output_dir", None):
        shutil.rmtree(tmp, ignore_errors=True)
    fs = None
    if args.reference:
        mode = "reference"
    elif curvelanes_cache_supported(metric):
        mode = "curvelanes"
    elif culane_cache_supported(metric) and (fs := import_fast_sweep()) is not None:
        mode = "culane"
    else:
        mode = "reference"
        print(f"[info] no cached evaluation for {type(metric).__name__}; "
              "running it once per threshold")

    if mode == "curvelanes":
        print(f"IoU matrices once ({len(dump)} images, {args.workers} workers)...", flush=True)
        cache = build_curvelanes_cache(dump, metric, Lane, args.workers)
        check_curvelanes_cache(dump, cache, thresholds, metric, factory, Lane, args.check_images)
        rows, key = curvelanes_sweep(cache, dump, thresholds, metric, args.metric_key)
        evaluation = "IoU matrices once, Hungarian per threshold (CurveLanes)"
    elif mode == "culane":
        print(f"IoU matrices once with fast_sweep ({len(dump)} images)...", flush=True)
        fs_dump = culane_fast_dump(dump, Lane)
        gt = fs.GroundTruth(metric.img_prefix, metric.list_path, metric.test_categories_dir)
        records = fs.build_records(fs_dump, gt, metric, jobs=args.workers)
        rows, key = culane_sweep(fs, records, dump, thresholds, args.metric_key)
        evaluation = "IoU matrices once, Hungarian per threshold (fast_sweep)"
    else:
        rows, key = sweep(dump, thresholds, factory, Lane, args.metric_key)
        evaluation = f"{type(metric).__name__} per threshold"

    best = max(rows, key=lambda r: r.get(key, -1))
    if args.verify and mode != "reference":
        print(f"\nverifying conf {best['threshold']:g} against the config's evaluator "
              "on the whole split...", flush=True)
        if mode == "culane":
            ok = fs.check_against_official(records, fs_dump, best["threshold"], metric,
                                           metric.img_prefix, metric.list_path,
                                           metric.test_categories_dir)
        else:
            ref, _ = sweep(dump, [best["threshold"]], factory, Lane, key)
            bad = compare_rows([best], ref)
            ok = not bad
            for t, k, x, y in bad:
                print(f"  {k}: cached {x} vs evaluator {y}")
        print("verify: the cached sweep matches the evaluator" if ok else
              "verify: MISMATCH, use --reference and report this")
        evaluation += ", verified at best" if ok else ", VERIFY FAILED"
    dump_conf = saved["conf"] if args.load_dump else thresholds[0]
    report = format_report(rows, key, best, dict(
        config=args.config, checkpoint=args.checkpoint or (saved.get("checkpoint") if args.load_dump else None),
        split=args.split, range=f"{args.range[0]:g} ~ {args.range[1]:g}", step=f"{args.step:g}",
        inference=f"conf >= {dump_conf:g}" + (f" (from {args.load_dump})" if args.load_dump else ""),
        evaluation=evaluation, max_images=args.max_images))
    print("\n" + report)

    with open(out, "w" if args.overwrite else "a") as f:
        f.write(report + "\n")
    base = osp.splitext(out)[0]
    fields = ["threshold", key, "lanes", "seconds"] + sorted(
        {k for r in rows for k in r} - {"threshold", key, "lanes", "seconds"})
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(base + ".json", "w") as f:
        json.dump(dict(config=args.config,
                       checkpoint=args.checkpoint or (saved.get("checkpoint") if args.load_dump else None),
                       split=args.split, evaluation=evaluation,
                       metric_key=key, best=best, rows=rows), f, indent=1)
    print(f"{'wrote' if args.overwrite else 'appended to'} {out} (+ {base}.csv, {base}.json for this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
