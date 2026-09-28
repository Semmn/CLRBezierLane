"""Build a self-contained interactive HTML dashboard.

No CDN, no build step: the data, CSS and JS are all inlined, so the file works
over SSH (``scp`` it back, or open it through a port-forwarded ``python -m
http.server``) and keeps working offline.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Dict, List, Optional, Sequence

from . import metrics as M
from . import palette as P
from .runs import Run

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

PRETTY = {
    "F1_0.5": "F1 @ IoU 0.5",
    "F1_0.1": "F1 @ IoU 0.1",
    "F1_0.75": "F1 @ IoU 0.75",
    "Precision0.5": "Precision @ IoU 0.5",
    "Recall0.5": "Recall @ IoU 0.5",
    "TP0.5": "True positives",
    "FP0.5": "False positives",
    "FN0.5": "False negatives",
    "loss": "Total loss",
    "loss_cls": "Classification loss",
    "loss_iou": "LaneIoU regression loss",
    "loss_brr_support": "BRR support loss",
    "loss_brr_cp": "BRR control-point loss",
    "loss_seg": "Segmentation loss",
    "main_iou": "main branch IoU loss",
    "aux_iou": "aux branch IoU loss",
    "main_num_pos": "main positives",
    "aux_num_pos": "aux positives",
    "main_pos_s0": "main positives, stage 0",
    "main_pos_s1": "main positives, stage 1",
    "main_pos_s2": "main positives, stage 2",
    "main_conf_iou_l1": "main |conf − IoU|",
    "aux_conf_iou_l1": "aux |conf − IoU|",
    "main_conf_iou_rank": "main conf/IoU Spearman",
    "aux_conf_iou_rank": "aux conf/IoU Spearman",
    "lr": "Learning rate",
    "time": "Iteration time (s)",
    "data_time": "Data time (s)",
    "memory": "GPU memory (MB)",
}


def _read_asset(name: str) -> str:
    with open(os.path.join(_ASSETS, name), "r", encoding="utf-8") as handle:
        return handle.read()


def build_payload(
    runs: Sequence[Run],
    primary: str = M.PRIMARY_METRIC,
    max_points: int = 2000,
    title: str = "Training curves",
) -> dict:
    slots = P.assign_slots([r.name for r in runs])
    warnings: List[str] = []

    if len(runs) > P.MAX_SERIES:
        warnings.append(
            f"{len(runs)} runs loaded but the palette has {P.MAX_SERIES} validated slots; "
            f"colours repeat past the {P.MAX_SERIES}th. Narrow with --include / "
            "--latest-only, or render separate pages."
        )

    inexact = [r.name for r in runs if not r.iters_per_epoch_exact]
    if inexact:
        warnings.append(
            "Iterations per epoch had to be estimated for: " + ", ".join(inexact)
            + ". The epoch axis is approximate for those; the iteration axis is exact."
        )
    no_conf = [r.name for r in runs if r.conf_threshold is None]
    if no_conf:
        warnings.append(
            "No confidence threshold found in the config for: " + ", ".join(no_conf)
            + ". Validation F1 is only comparable at a known threshold — set it in a "
            "runs.json sidecar (\"conf\": 0.40) or with --conf NAME=0.40."
        )

    keys: List[str] = []
    for run in runs:
        for key in M.series_ids(run):
            if key not in keys:
                keys.append(key)
    panels = M.assign_groups(keys)

    series: Dict[str, Dict[str, dict]] = {}
    for run in runs:
        per_run: Dict[str, dict] = {}
        for key in M.series_ids(run):
            xs, ys = M.run_series(run, key, x_axis="epoch")
            if not ys:
                continue
            xs, ys = M.decimate(xs, ys, max_points)
            per_run[key] = dict(
                x=[round(x, 5) for x in xs],
                y=[_round(y) for y in ys],
                val=M.is_val_series(run, key),
            )
        series[run.name] = per_run

    pretty = dict(PRETTY)
    for key in keys:
        if key not in pretty:
            base, suffix = M.split_id(key)
            label = PRETTY.get(base, base)
            pretty[key] = f"{label} — validation" if suffix == "val" else label

    summaries = []
    for run in runs:
        s = M.summarize(run, primary)
        summaries.append(dict(
            run=s.run, label=s.label, conf=s.conf_threshold, epochs=s.epochs,
            best=s.best, best_epoch=s.best_epoch, final=s.final,
            plateau_002=s.plateau.get(0.002), plateau_005=s.plateau.get(0.005),
            drift=s.drift,
        ))
    summaries.sort(key=lambda d: (d["best"] is None, -(d["best"] or 0)))

    return dict(
        title=title,
        generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        primary=primary,
        palette=dict(light=P.SERIES_LIGHT, dark=P.SERIES_DARK),
        pretty=pretty,
        warnings=warnings,
        panels=[dict(id=p["id"], title=p["title"], note=p.get("note", ""),
                     log_default=bool(p.get("log_default")), keys=p["keys"]) for p in panels],
        runs=[dict(
            name=run.name,
            label=run.label,
            path=run.path,
            slot=slots[run.name],
            conf=run.conf_threshold,
            epochs=run.num_epochs,
            iters_per_epoch=run.iters_per_epoch,
            head=run.meta.get("head"),
            num_priors=run.meta.get("num_priors"),
        ) for run in runs],
        series=series,
        summary=summaries,
    )


def _round(value: float) -> float:
    try:
        return float(f"{value:.6g}")
    except (TypeError, ValueError):
        return value


def render(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    # </script> cannot appear inside an inline script.
    data = data.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(payload.get("title", "Training curves"))}</title>
<style>
{_read_asset("dashboard.css")}
</style>
</head>
<body>
<script id="lanevis-data" type="application/json">{data}</script>
<script>window.LANEVIS = JSON.parse(document.getElementById("lanevis-data").textContent);</script>
<script>
{_read_asset("dashboard.js")}
</script>
</body>
</html>
"""


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write(runs: Sequence[Run], out_path: str, primary: str = M.PRIMARY_METRIC,
          max_points: int = 2000, title: Optional[str] = None) -> str:
    payload = build_payload(runs, primary=primary, max_points=max_points,
                            title=title or "Training curves")
    html = render(payload)
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return out_path
