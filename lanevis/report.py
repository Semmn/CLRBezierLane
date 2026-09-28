"""Text table and tidy CSV export."""
from __future__ import annotations

import csv
import os
from typing import List, Optional, Sequence

from . import metrics as M
from .runs import Run


def _cell(value, digits=2, scale=1.0, plus=False) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value * scale:.{digits}f}"
        return ("+" + text) if (plus and value >= 0) else text
    return str(value)


def summary_table(runs: Sequence[Run], metric: str = M.PRIMARY_METRIC,
                  tolerances: Sequence[float] = (0.002, 0.005)) -> str:
    rows = [M.summarize(run, metric, tolerances) for run in runs]
    prefers_max = M.metric_prefers_max(metric)
    rows.sort(key=lambda s: (s.best is None, -(s.best or 0) if prefers_max else (s.best or 0)))

    percent = metric.startswith(("F1", "Precision", "Recall"))
    scale = 100.0 if percent else 1.0
    unit = " (%)" if percent else ""

    header = ["run", "conf", "ep", f"best{unit}", "@ep"]
    header += [f"w/in {t * scale:g}" for t in tolerances]
    header += [f"final{unit}", "final-best"]

    body: List[List[str]] = []
    for s in rows:
        line = [s.label, _cell(s.conf_threshold), str(s.epochs or "-"),
                _cell(s.best, 2 if percent else 4, scale),
                _cell(s.best_epoch, 0)]
        for tol in tolerances:
            line.append(_cell(s.plateau.get(float(tol)), 0))
        line += [_cell(s.final, 2 if percent else 4, scale),
                 _cell(s.drift, 2 if percent else 4, scale, plus=True)]
        body.append(line)

    widths = [len(h) for h in header]
    for line in body:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], len(cell))

    def render(cells, pad="-"):
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return "  ".join(out)

    lines = [render(header), "  ".join("-" * w for w in widths)]
    lines += [render(line) for line in body]
    lines.append("")
    lines.append(
        f"'w/in x' = earliest epoch whose running best is within x{' percentage points' if percent else ''} of that run's own"
        f" best: the first epoch you could have stopped at."
    )
    if any(s.conf_threshold is None for s in rows):
        lines.append(
            "conf '-' = the threshold was not found in the run's config. F1 is not "
            "comparable across runs at different thresholds; set it per run in runs.json."
        )
    return "\n".join(lines)


def write_csv(runs: Sequence[Run], out_path: str, x_axis: str = "epoch") -> str:
    """Tidy long format: one row per (run, metric, x)."""
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "label", "conf_threshold", "split", "metric",
                         "epoch", "iteration", "value"])
        for run in runs:
            n = run.iters_per_epoch or 1.0
            for sid in M.series_ids(run):
                xs, ys = M.run_series(run, sid, x_axis="epoch")
                split = "val" if M.is_val_series(run, sid) else "train"
                base = M.split_id(sid)[0]
                for x, y in zip(xs, ys):
                    writer.writerow([run.name, run.label,
                                     "" if run.conf_threshold is None else run.conf_threshold,
                                     split, base, f"{x:.5f}", int(round(x * n)), repr(y)])
    return out_path


def describe(runs: Sequence[Run]) -> str:
    lines = []
    for run in runs:
        conf = "?" if run.conf_threshold is None else f"{run.conf_threshold:.2f}"
        exact = "" if run.iters_per_epoch_exact else " (estimated)"
        val_axis = "epoch" if run.val_step_is_epoch else "iteration"
        lines.append(
            f"{run.name}\n"
            f"    path            {run.path}\n"
            f"    epochs          {run.num_epochs}\n"
            f"    iters/epoch     {run.iters_per_epoch:g}{exact}\n"
            f"    train points    {len(run.train)}\n"
            f"    val points      {len(run.val)} (step counts {val_axis})\n"
            f"    conf threshold  {conf}\n"
            f"    head            {run.meta.get('head', '?')}"
            f"    priors {run.meta.get('num_priors', '?')}\n"
            f"    metrics         {len(run.keys())}: {', '.join(run.keys()[:8])}"
            + (" ..." if len(run.keys()) > 8 else "")
        )
    return "\n".join(lines)
