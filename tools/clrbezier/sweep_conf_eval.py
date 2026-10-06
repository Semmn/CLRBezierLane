"""Evaluate one checkpoint over a confidence-threshold range with a fixed step.

    python tools/clrbezier/sweep_conf_eval.py CONFIG CHECKPOINT --range 0.80 0.90 --step 0.02

Inference runs ONCE, at the lowest threshold of the range. Every higher threshold
is then obtained by dropping the lanes whose score is below it, and the dataset's
own evaluator (``test_evaluator`` or ``val_evaluator`` from the config:
CULaneMetric, CurvelanesMetric, LLAMASMetric, TuSimpleMetric) is run on the
filtered predictions.

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

For CULane, tools/clrbezier/sweep_threshold.py is much faster on large grids
(rasterizes once); this script re-runs the official evaluation per threshold,
which is the reference behaviour and works for every dataset.
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
import shutil
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
    for key in ("F1", "F1_0.50", "F1_measure", "F1@50", "f1"):
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
    runner.load_or_resume()
    model = runner.model
    model.eval()
    import torch

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
    extra = [k for k in ("Precision", "Recall") if any(k in r for r in rows)]
    lines = ["=" * 64, f"threshold sweep  {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    for name, value in info.items():
        if value not in (None, ""):
            lines.append(f"{name:>10}: {value}")
    lines.append("-" * 64)
    lines.append(f"{'conf':>8} {key:>12}" + "".join(f" {k:>10}" for k in extra) + f" {'lanes':>8}")
    for r in rows:
        mark = "  <- best" if r is best else ""
        lines.append(f"{r['threshold']:8.4f} {r.get(key, float('nan')):12.4f}"
                     + "".join(f" {r.get(k, float('nan')):10.4f}" for k in extra)
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
    rows, key = sweep(dump, thresholds, metric_factory_from(evaluator_cfg, args.metric_index),
                      Lane, args.metric_key)

    best = max(rows, key=lambda r: r.get(key, -1))
    dump_conf = saved["conf"] if args.load_dump else thresholds[0]
    report = format_report(rows, key, best, dict(
        config=args.config, checkpoint=args.checkpoint or (saved.get("checkpoint") if args.load_dump else None),
        split=args.split, range=f"{args.range[0]:g} ~ {args.range[1]:g}", step=f"{args.step:g}",
        inference=f"conf >= {dump_conf:g}" + (f" (from {args.load_dump})" if args.load_dump else ""),
        max_images=args.max_images))
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
                       split=args.split,
                       metric_key=key, best=best, rows=rows), f, indent=1)
    print(f"{'wrote' if args.overwrite else 'appended to'} {out} (+ {base}.csv, {base}.json for this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
