"""Metric grouping, smoothing and plateau analysis."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .runs import Run

# Panels, in display order. `keys` are matched as exact names first, then as
# regexes, so a run that logs extra diagnostics still lands somewhere sensible.
GROUPS: List[dict] = [
    dict(
        id="val_f1",
        title="Validation F1 / precision / recall",
        keys=["F1_0.5", "Precision0.5", "Recall0.5"],
        note="IoU threshold 0.5. Depends on the run's confidence threshold.",
        prefer_max=True,
    ),
    dict(
        id="val_thresholds",
        # CULane's metric emits F1/Precision/Recall at several IoU thresholds and
        # one set per test category, so match by shape rather than listing them.
        title="Validation F1 — other IoU thresholds and per-category splits",
        keys=[r"^F1_", r"^(Precision|Recall)[\d.]+$"],
        prefer_max=True,
    ),
    dict(
        id="val_counts",
        title="Validation TP / FP / FN",
        keys=[r"^(TP|FP|FN)[\d.]*$"],
    ),
    dict(
        id="loss_total",
        title="Total loss",
        keys=["loss"],
        log_default=True,
    ),
    dict(
        id="loss_parts",
        title="Loss components",
        keys=[r"^loss_.*"],
        log_default=True,
    ),
    dict(
        id="iou",
        title="IoU loss per branch",
        keys=["main_iou", "aux_iou"],
    ),
    dict(
        id="alignment",
        title="Confidence / localization alignment",
        keys=[r"^(main|aux)_conf_iou_l1$", r"^(main|aux)_conf_iou_rank$"],
        note="L1 is CLRerNet Fig. 7 (lower is better); rank is Spearman (higher is better).",
    ),
    dict(
        id="assignment",
        title="Assigned positives",
        keys=[r"^(main|aux)_num_pos$", r"^main_pos_s\d+$"],
    ),
    dict(
        id="reproject",
        title="Reference re-projection",
        keys=[r"^reproj_.*"],
    ),
    dict(
        id="system",
        title="Schedule and cost",
        keys=["lr", "time", "data_time", "memory"],
    ),
]

PRIMARY_METRIC = "F1_0.5"
LOWER_IS_BETTER = re.compile(r"(^|_)(loss|l1|FP|FN|time|memory|lr)", re.IGNORECASE)

# A key logged by both loops becomes two series; the val one carries this
# suffix so the two never share a panel.
VAL_SUFFIX = "@val"


def split_id(series_id: str):
    """``"time@val"`` -> ``("time", "val")``; ``"loss"`` -> ``("loss", None)``."""
    if series_id.endswith(VAL_SUFFIX):
        return series_id[: -len(VAL_SUFFIX)], "val"
    return series_id, None


def series_ids(run) -> List[str]:
    """Series identifiers for one run, splitting dual-logged keys."""
    ids: List[str] = []
    for key in run.keys():
        splits = run.splits(key)
        if len(splits) > 1:
            ids.append(key)                 # train
            ids.append(key + VAL_SUFFIX)    # val
        else:
            ids.append(key)
    return ids


def run_series(run, series_id: str, x_axis: str = "epoch"):
    key, suffix = split_id(series_id)
    if suffix == "val":
        return run.series(key, x_axis=x_axis, split="val")
    if len(run.splits(key)) > 1:
        return run.series(key, x_axis=x_axis, split="train")
    return run.series(key, x_axis=x_axis)


def is_val_series(run, series_id: str) -> bool:
    key, suffix = split_id(series_id)
    if suffix == "val":
        return True
    if len(run.splits(key)) > 1:
        return False
    return run.is_val_key(key)


def metric_prefers_max(key: str) -> bool:
    if key.startswith(("F1", "Precision", "Recall", "TP")) or key.endswith("_rank"):
        return True
    return not bool(LOWER_IS_BETTER.search(key))


def assign_groups(keys: Sequence[str]) -> List[dict]:
    """Bucket the metric keys actually present into panels; keep the leftovers.

    Matching ignores the ``@val`` suffix so a dual-logged key lands in the same
    group as its training half, as two panels.
    """
    remaining = list(keys)
    panels: List[dict] = []
    for group in GROUPS:
        matched: List[str] = []
        for pattern in group["keys"]:
            exact = [k for k in remaining if split_id(k)[0] == pattern]
            if exact:
                for hit in exact:
                    matched.append(hit)
                    remaining.remove(hit)
                continue
            if any(ch in pattern for ch in ".*^$[]()\\+?"):
                compiled = re.compile(pattern)
                hits = [k for k in remaining if compiled.search(split_id(k)[0])]
                for hit in hits:
                    matched.append(hit)
                    remaining.remove(hit)
        if matched:
            panels.append({**group, "keys": matched})
    if remaining:
        panels.append(dict(id="other", title="Other logged metrics", keys=remaining))
    return panels


def ema(values: Sequence[float], weight: float) -> List[float]:
    """TensorBoard's debiased exponential moving average.

    ``weight`` in [0, 1): 0 returns the input unchanged. The debias term keeps
    the first points from being dragged toward zero, which matters for metrics
    that start high (a loss) as much as for ones that start near zero.
    """
    if weight <= 0 or not values:
        return list(values)
    weight = min(weight, 0.999)
    out: List[float] = []
    last = 0.0
    debias = 0.0
    for value in values:
        last = last * weight + (1.0 - weight) * value
        debias = debias * weight + (1.0 - weight)
        out.append(last / debias if debias > 0 else value)
    return out


@dataclass
class RunSummary:
    run: str
    label: str
    metric: str
    conf_threshold: Optional[float]
    epochs: int
    num_points: int
    best: Optional[float]
    best_epoch: Optional[float]
    final: Optional[float]
    final_epoch: Optional[float]
    plateau: Dict[float, Optional[float]]

    @property
    def drift(self) -> Optional[float]:
        if self.best is None or self.final is None:
            return None
        return self.final - self.best


def summarize(
    run: Run,
    metric: str = PRIMARY_METRIC,
    tolerances: Sequence[float] = (0.002, 0.005),
) -> RunSummary:
    """Best value, where it happened, and when the curve first got close.

    ``plateau[tol]`` is the earliest epoch whose running best is within ``tol``
    of the run's overall best — i.e. the first epoch you could have stopped at
    and given up no more than ``tol``. That is the quantity an
    epoch-budget decision needs, and it is not the same as the argmax.
    """
    xs, ys = run.series(metric, x_axis="epoch")
    prefers_max = metric_prefers_max(metric)
    best = best_epoch = None
    plateau: Dict[float, Optional[float]] = {float(t): None for t in tolerances}

    if ys:
        best = max(ys) if prefers_max else min(ys)
        best_epoch = xs[ys.index(best)]
        running = None
        for x, y in zip(xs, ys):
            running = y if running is None else (max(running, y) if prefers_max else min(running, y))
            for tol in plateau:
                if plateau[tol] is None:
                    close = (running >= best - tol) if prefers_max else (running <= best + tol)
                    if close:
                        plateau[tol] = x

    return RunSummary(
        run=run.name,
        label=run.label,
        metric=metric,
        conf_threshold=run.conf_threshold,
        epochs=run.num_epochs,
        num_points=len(ys),
        best=best,
        best_epoch=best_epoch,
        final=ys[-1] if ys else None,
        final_epoch=xs[-1] if xs else None,
        plateau=plateau,
    )


def decimate(xs: Sequence[float], ys: Sequence[float], max_points: int):
    """Uniform stride, keeping the first and last sample."""
    n = len(xs)
    if max_points <= 0 or n <= max_points:
        return list(xs), list(ys)
    stride = (n + max_points - 1) // max_points
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [xs[i] for i in idx], [ys[i] for i in idx]
