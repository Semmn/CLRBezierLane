"""Static figures for the paper (matplotlib).

Same palette, same grouping and the same epoch axis as the dashboard, so a
figure in the paper matches what you looked at while training. Thin marks,
hairline grid, no chart junk; ``--pdf`` gives vector output for LaTeX.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

from . import metrics as M
from . import palette as P
from .dashboard import PRETTY
from .runs import Run


def _style(mode: str):
    import matplotlib as mpl

    chrome = P.CHROME[mode]
    mpl.rcParams.update({
        "figure.facecolor": chrome["surface"],
        "axes.facecolor": chrome["surface"],
        "savefig.facecolor": chrome["surface"],
        "text.color": chrome["ink"],
        "axes.labelcolor": chrome["ink_secondary"],
        "axes.edgecolor": chrome["axis"],
        "xtick.color": chrome["ink_muted"],
        "ytick.color": chrome["ink_muted"],
        "grid.color": chrome["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 0,
    })


def plot_metrics(
    runs: Sequence[Run],
    keys: Sequence[str],
    out_path: str,
    x_axis: str = "epoch",
    smoothing: float = 0.6,
    ncols: int = 3,
    mode: str = "light",
    log_keys: Sequence[str] = (),
    dpi: int = 200,
    title: Optional[str] = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(mode)
    slots = P.assign_slots([r.name for r in runs])

    keys = [k for k in keys if any(k in M.series_ids(r) for r in runs)]
    if not keys:
        raise ValueError("none of the requested metrics are present in these runs")

    ncols = max(1, min(ncols, len(keys)))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.35 * nrows), squeeze=False)

    handles: dict = {}
    for index, key in enumerate(keys):
        ax = axes[index // ncols][index % ncols]
        for run in runs:
            xs, ys = M.run_series(run, key, x_axis=x_axis)
            if not ys:
                continue
            is_val = M.is_val_series(run, key)
            if not is_val and smoothing > 0:
                ys = M.ema(ys, smoothing)
            color = P.color(slots[run.name], mode)
            (line,) = ax.plot(
                xs, ys, color=color,
                marker="o" if (is_val and len(ys) <= 60) else None,
                markersize=3.2, markeredgecolor=P.CHROME[mode]["surface"], markeredgewidth=0.7,
            )
            handles.setdefault(run.label, line)
        base, suffix = M.split_id(key)
        name = PRETTY.get(base, base) + (" — validation" if suffix == "val" else "")
        ax.set_title(name, loc="left")
        ax.set_xlabel("epoch" if x_axis == "epoch" else "iteration")
        if key in log_keys:
            ax.set_yscale("log")
        ax.margins(x=0.02)

    for blank in range(len(keys), nrows * ncols):
        axes[blank // ncols][blank % ncols].axis("off")

    if len(handles) >= 2:
        fig.legend(list(handles.values()), list(handles.keys()),
                   loc="lower center", ncol=min(4, len(handles)),
                   bbox_to_anchor=(0.5, -0.01))
        bottom = 0.10 + 0.03 * ((len(handles) + 3) // 4)
    else:
        bottom = 0.06
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="semibold")

    fig.tight_layout(rect=(0, bottom, 1, 0.97 if title else 1))

    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_groups(
    runs: Sequence[Run],
    out_dir: str,
    x_axis: str = "epoch",
    smoothing: float = 0.6,
    mode: str = "light",
    ext: str = "png",
    dpi: int = 200,
) -> List[str]:
    """One figure per metric group."""
    keys: List[str] = []
    for run in runs:
        for key in M.series_ids(run):
            if key not in keys:
                keys.append(key)
    written: List[str] = []
    for group in M.assign_groups(keys):
        log_keys = group["keys"] if group.get("log_default") else ()
        path = os.path.join(out_dir, f"{group['id']}.{ext}")
        plot_metrics(runs, group["keys"], path, x_axis=x_axis, smoothing=smoothing,
                     mode=mode, log_keys=log_keys, dpi=dpi, title=group["title"])
        written.append(path)
    return written
