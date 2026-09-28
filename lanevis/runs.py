"""Discover mmengine runs and load their scalar logs.

An mmengine work_dir looks like::

    work_dirs/<cfg_name>/<timestamp>/vis_data/scalars.json
    work_dirs/<cfg_name>/<timestamp>/vis_data/config.py
    work_dirs/<cfg_name>/<timestamp>/<timestamp>.log

``scalars.json`` is JSON Lines, not JSON: one object per line. Two kinds of line
appear, and they do not share an x axis:

* **train** lines carry ``loss`` and a global iteration in ``step``/``iter``,
  plus the 1-based ``epoch``;
* **val** lines carry the metric keys (``F1_0.5`` ...) and ``step`` is the
  **epoch number**, because the validation loop runs on an epoch interval.

Plotting both against raw ``step`` puts a 36-epoch validation curve inside the
first 36 iterations of training. Everything here is converted to a common
fractional-epoch axis instead.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

SCALAR_FILENAMES = ("scalars.json",)

# Keys that are bookkeeping rather than measurements.
AXIS_KEYS = frozenset({"step", "iter", "epoch"})


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves a truncated final line.
                if lineno > 1:
                    continue
                raise
            if isinstance(row, dict):
                rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Config scraping
# --------------------------------------------------------------------------
# The confidence threshold used for validation is not logged in scalars.json,
# so it is read out of the config mmengine dumps beside the log. These patterns
# are deliberately loose: the dumped config is a merged dict literal, and
# exec-ing it would drag in mmengine's registries.
_SCRAPE_PATTERNS = {
    "conf_threshold": r"conf_threshold['\"]?\s*[=:]\s*([0-9]*\.?[0-9]+)",
    "num_priors": r"num_priors['\"]?\s*[=:]\s*([0-9]+)",
    "max_epochs": r"max_epochs['\"]?\s*[=:]\s*([0-9]+)",
    "nms_thres": r"nms_thres['\"]?\s*[=:]\s*([0-9]*\.?[0-9]+)",
    "lr": r"['\"]?lr['\"]?\s*[=:]\s*([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)",
    "batch_size": r"batch_size['\"]?\s*[=:]\s*([0-9]+)",
}
_HEAD_PATTERN = r"type\s*[=:]\s*['\"]((?:CLRBezier|CLRer|CLR)[A-Za-z]*Head)['\"]"


def _find_config(scalars_path: str) -> Optional[str]:
    vis_dir = os.path.dirname(scalars_path)
    run_dir = os.path.dirname(vis_dir)
    candidates = [os.path.join(vis_dir, "config.py")]
    for directory in (run_dir, os.path.dirname(run_dir)):
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        candidates += [os.path.join(directory, n) for n in names if n.endswith(".py")]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def scrape_config(path: Optional[str]) -> Dict[str, object]:
    """Pull the few config values worth showing next to a curve."""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return {}

    meta: Dict[str, object] = {"config_path": path}
    for key, pattern in _SCRAPE_PATTERNS.items():
        matches = re.findall(pattern, text)
        if not matches:
            continue
        # The merged config repeats keys (base then override); the last wins.
        value = matches[-1]
        meta[key] = float(value) if ("." in value or "e" in value.lower()) else int(value)
    head = re.findall(_HEAD_PATTERN, text)
    if head:
        meta["head"] = head[0]
    return meta


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
@dataclass
class Run:
    name: str
    path: str
    train: List[dict] = field(default_factory=list)
    val: List[dict] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)
    iters_per_epoch: Optional[float] = None
    iters_per_epoch_exact: bool = True
    val_step_is_epoch: bool = True

    # -- axes -------------------------------------------------------------
    @property
    def num_epochs(self) -> int:
        epochs = [int(r["epoch"]) for r in self.train if "epoch" in r]
        if epochs:
            return max(epochs)
        if self.val:
            return int(max(r.get("step", 0) for r in self.val))
        return 0

    def train_epoch_axis(self) -> List[float]:
        n = self.iters_per_epoch or 1.0
        return [float(r["step"]) / n for r in self.train]

    def train_iter_axis(self) -> List[float]:
        return [float(r["step"]) for r in self.train]

    def val_epoch_axis(self) -> List[float]:
        if self.val_step_is_epoch:
            return [float(r["step"]) for r in self.val]
        n = self.iters_per_epoch or 1.0
        return [float(r["step"]) / n for r in self.val]

    def val_iter_axis(self) -> List[float]:
        n = self.iters_per_epoch or 1.0
        if self.val_step_is_epoch:
            return [float(r["step"]) * n for r in self.val]
        return [float(r["step"]) for r in self.val]

    # -- series -----------------------------------------------------------
    def keys(self) -> List[str]:
        found: List[str] = []
        for rows in (self.train, self.val):
            for row in rows:
                for key in row:
                    if key not in AXIS_KEYS and key not in found:
                        if isinstance(row[key], (int, float)) and not isinstance(row[key], bool):
                            found.append(key)
        return found

    def is_val_key(self, key: str) -> bool:
        return any(key in row for row in self.val)

    def splits(self, key: str) -> List[str]:
        """Which halves of the log carry this key.

        ``time`` and ``data_time`` are logged by both the train loop and the
        val loop, and they mean different things in each. Resolving such a key
        to a single split would put one run's training cost and another run's
        validation cost on the same axis.
        """
        out = []
        if any(key in row for row in self.train):
            out.append("train")
        if any(key in row for row in self.val):
            out.append("val")
        return out

    def series(self, key: str, x_axis: str = "epoch", split: Optional[str] = None):
        """Return ``(xs, ys)`` for one metric, dropping rows that lack it."""
        if split is None:
            from_val = self.is_val_key(key)
        else:
            from_val = split == "val"
        rows = self.val if from_val else self.train
        if from_val:
            xs_all = self.val_epoch_axis() if x_axis == "epoch" else self.val_iter_axis()
        else:
            xs_all = self.train_epoch_axis() if x_axis == "epoch" else self.train_iter_axis()
        xs: List[float] = []
        ys: List[float] = []
        for x, row in zip(xs_all, rows):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                xs.append(x)
                ys.append(float(value))
        return xs, ys

    @property
    def conf_threshold(self) -> Optional[float]:
        value = self.meta.get("conf_threshold")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def label(self) -> str:
        return str(self.meta.get("label") or self.name)


def _infer_iters_per_epoch(train: Sequence[dict]) -> tuple[Optional[float], bool]:
    """Iterations per epoch, from the first logged step of each epoch.

    mmengine's LoggerHook fires on an epoch-local interval while ``step`` counts
    globally, so the gap between the first logged step of consecutive epochs is
    exactly the epoch length. That is more reliable than ``max_step /
    max_epoch``, which is short by the logging remainder (it gives 435 where the
    true length is 436).
    """
    firsts: Dict[int, int] = {}
    for row in train:
        if "epoch" not in row or "step" not in row:
            continue
        epoch = int(row["epoch"])
        step = int(row["step"])
        firsts[epoch] = min(step, firsts.get(epoch, step))
    if len(firsts) >= 2:
        epochs = sorted(firsts)
        diffs = [
            firsts[b] - firsts[a]
            for a, b in zip(epochs, epochs[1:])
            if b == a + 1 and firsts[b] > firsts[a]
        ]
        if diffs:
            return float(statistics.median(diffs)), True
    steps = [int(r["step"]) for r in train if "step" in r]
    epochs_seen = [int(r["epoch"]) for r in train if "epoch" in r]
    if steps and epochs_seen and max(epochs_seen) > 0:
        return max(steps) / max(epochs_seen), False
    if steps:
        return float(max(steps)), False
    return None, False


def load_run(scalars_path: str, name: Optional[str] = None) -> Run:
    rows = _read_jsonl(scalars_path)
    train = [r for r in rows if "loss" in r or "lr" in r]
    train_ids = {id(r) for r in train}
    val = [r for r in rows if id(r) not in train_ids]

    iters_per_epoch, exact = _infer_iters_per_epoch(train)

    # Does the val row's ``step`` count epochs or iterations? On an
    # epoch-interval ValLoop it counts epochs, and then it never exceeds the
    # epoch count. Compare against the training range rather than assuming.
    max_epoch = max((int(r["epoch"]) for r in train if "epoch" in r), default=0)
    max_val_step = max((float(r.get("step", 0)) for r in val), default=0.0)
    val_step_is_epoch = True
    if val and max_epoch:
        val_step_is_epoch = max_val_step <= max_epoch + 1

    run = Run(
        name=name or os.path.basename(os.path.dirname(os.path.dirname(scalars_path))),
        path=scalars_path,
        train=train,
        val=val,
        iters_per_epoch=iters_per_epoch,
        iters_per_epoch_exact=exact,
        val_step_is_epoch=val_step_is_epoch,
    )
    run.meta.update(scrape_config(_find_config(scalars_path)))
    run.meta["mtime"] = os.path.getmtime(scalars_path)
    return run


def discover(
    targets: Iterable[str],
    latest_only: bool = False,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
) -> List[Run]:
    """Find runs under one or more roots (or accept scalars.json paths directly)."""
    found: List[tuple[str, str]] = []  # (run name, scalars path)
    for target in targets:
        if os.path.isfile(target):
            found.append((_run_name_for(target, os.path.dirname(target)), target))
            continue
        if not os.path.isdir(target):
            raise FileNotFoundError(target)
        root = os.path.normpath(target)
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename in SCALAR_FILENAMES:
                    path = os.path.join(dirpath, filename)
                    found.append((_run_name_for(path, root), path))

    if include:
        pattern = re.compile(include)
        found = [f for f in found if pattern.search(f[0])]
    if exclude:
        pattern = re.compile(exclude)
        found = [f for f in found if not pattern.search(f[0])]

    runs = [load_run(path, name) for name, path in sorted(set(found))]

    if latest_only:
        newest: Dict[str, Run] = {}
        for run in runs:
            group = run.name.split("@", 1)[0]
            current = newest.get(group)
            if current is None or run.meta.get("mtime", 0) > current.meta.get("mtime", 0):
                newest[group] = run
        runs = [newest[key] for key in sorted(newest)]

    return runs


def _run_name_for(scalars_path: str, root: str) -> str:
    """``work_dirs/cfg/20260928_120000/vis_data/scalars.json`` -> ``cfg@20260928_120000``."""
    vis_dir = os.path.dirname(scalars_path)
    run_dir = os.path.dirname(vis_dir) if os.path.basename(vis_dir) == "vis_data" else vis_dir
    relative = os.path.relpath(run_dir, root)
    parts = [p for p in relative.split(os.sep) if p not in (".", "")]
    if not parts:
        return os.path.basename(run_dir) or "run"
    if len(parts) == 1:
        return parts[0]
    return "/".join(parts[:-1]) + "@" + parts[-1]


def auto_label(runs: Sequence[Run]) -> None:
    """Drop the ``@timestamp`` from labels when the config name is unambiguous.

    Run *names* stay unique (they key the overrides and the colour slots); only
    what gets shown shortens, because a tooltip that reads
    ``clrbezier_collab_pertu…`` twice is no legend at all.
    """
    bases = [run.name.split("@", 1)[0] for run in runs]
    if len(set(bases)) != len(bases):
        return
    for run, base in zip(runs, bases):
        run.meta.setdefault("label", base)


def apply_overrides(runs: Sequence[Run], overrides: Dict[str, dict]) -> None:
    """Merge a ``runs.json`` sidecar into run metadata.

    Keys are matched exactly, then as a prefix of the run name (so a config name
    matches every timestamped run of it), then as a regex.
    """
    for run in runs:
        # The longest matching key wins, so an entry for
        # "clrbezier_collab_perturb_r34" does not also capture
        # "clrbezier_collab_perturb_r34_ep24".
        candidates = []
        for key, values in overrides.items():
            if not isinstance(values, dict):
                continue
            # A prefix must end on a name boundary, so "..._r34" matches the
            # config of that name (any timestamp) but not "..._r34_ep24".
            if key == run.name or (
                run.name.startswith(key) and run.name[len(key):len(key) + 1] in ("@", "/")
            ):
                candidates.append((len(key), key, values))
                continue
            # Only treat a key as a pattern when it looks like one; otherwise a
            # plain config name would also match every longer name containing it.
            if not any(ch in key for ch in ".*^$[]()\\+?|{}"):
                continue
            try:
                if re.search(key, run.name):
                    candidates.append((len(key), key, values))
            except re.error:
                continue
        if not candidates:
            continue
        _, _, values = max(candidates, key=lambda item: item[0])
        values = dict(values)
        if "conf" in values and "conf_threshold" not in values:
            values["conf_threshold"] = values.pop("conf")
        run.meta.update(values)
