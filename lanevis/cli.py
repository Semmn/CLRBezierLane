"""lanevis command line.

    python -m lanevis ls    work_dirs
    python -m lanevis dash  work_dirs -o curves.html
    python -m lanevis table work_dirs --metric F1_0.5
    python -m lanevis png   work_dirs -o figs/
    python -m lanevis csv   work_dirs -o metrics.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

from . import dashboard, figures, metrics as M, report
from .runs import Run, apply_overrides, auto_label, discover


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("targets", nargs="*", default=["work_dirs"],
                        help="work_dir roots, run directories, or scalars.json paths "
                             "(default: work_dirs)")
    parser.add_argument("--latest-only", action="store_true",
                        help="keep only the newest timestamped run per config")
    parser.add_argument("--include", help="regex the run name must match")
    parser.add_argument("--exclude", help="regex the run name must not match")
    parser.add_argument("--runs-json", help="sidecar with per-run label/conf/notes")
    parser.add_argument("--conf", action="append", default=[], metavar="NAME=VALUE",
                        help="set a run's validation confidence threshold "
                             "(repeatable; NAME is matched as a prefix or regex)")
    parser.add_argument("--label", action="append", default=[], metavar="NAME=TEXT",
                        help="override a run's display label (repeatable)")


def _load(args) -> List[Run]:
    runs = discover(args.targets, latest_only=args.latest_only,
                    include=args.include, exclude=args.exclude)
    overrides: Dict[str, dict] = {}
    if getattr(args, "runs_json", None):
        with open(args.runs_json, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise SystemExit("--runs-json must contain an object keyed by run name")
        overrides.update(loaded)
    for item in getattr(args, "conf", []):
        name, _, value = item.partition("=")
        overrides.setdefault(name, {})["conf_threshold"] = float(value)
    for item in getattr(args, "label", []):
        name, _, value = item.partition("=")
        overrides.setdefault(name, {})["label"] = value
    auto_label(runs)
    if overrides:
        apply_overrides(runs, overrides)
    if not runs:
        raise SystemExit(
            "No scalars.json found under: " + ", ".join(args.targets)
            + "\nmmengine writes it to <work_dir>/<cfg>/<timestamp>/vis_data/scalars.json."
        )
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lanevis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)

    p_ls = subs.add_parser("ls", help="list the runs found and what they contain")
    _common(p_ls)

    p_dash = subs.add_parser("dash", help="write a self-contained interactive HTML page")
    _common(p_dash)
    p_dash.add_argument("-o", "--out", default="lanevis.html")
    p_dash.add_argument("--metric", default=M.PRIMARY_METRIC,
                        help="metric for the summary table (default F1_0.5)")
    p_dash.add_argument("--max-points", type=int, default=2000,
                        help="decimate each curve to at most this many points")
    p_dash.add_argument("--title", default="Training curves")

    p_table = subs.add_parser("table", help="print the best/plateau table")
    _common(p_table)
    p_table.add_argument("--metric", default=M.PRIMARY_METRIC)
    p_table.add_argument("--tol", type=float, nargs="+", default=[0.002, 0.005],
                         help="plateau tolerances in metric units (0.002 = 0.2 pp F1)")

    p_png = subs.add_parser("png", help="write static figures (matplotlib)")
    _common(p_png)
    p_png.add_argument("-o", "--out", default="figs",
                       help="output directory, or a single file when --metric is given")
    p_png.add_argument("--metric", nargs="+", help="plot only these metrics, in one figure")
    p_png.add_argument("--x", choices=["epoch", "iter"], default="epoch")
    p_png.add_argument("--smoothing", type=float, default=0.6)
    p_png.add_argument("--mode", choices=["light", "dark"], default="light")
    p_png.add_argument("--pdf", action="store_true", help="vector output for LaTeX")
    p_png.add_argument("--dpi", type=int, default=200)
    p_png.add_argument("--ncols", type=int, default=3)

    p_csv = subs.add_parser("csv", help="export every series as tidy long-format CSV")
    _common(p_csv)
    p_csv.add_argument("-o", "--out", default="metrics.csv")

    args = parser.parse_args(argv)
    runs = _load(args)

    if args.command == "ls":
        print(report.describe(runs))
        return 0

    if args.command == "table":
        print(report.summary_table(runs, args.metric, args.tol))
        return 0

    if args.command == "csv":
        path = report.write_csv(runs, args.out)
        print(f"wrote {path}")
        return 0

    if args.command == "dash":
        path = dashboard.write(runs, args.out, primary=args.metric,
                              max_points=args.max_points, title=args.title)
        payload_warnings = dashboard.build_payload(runs, primary=args.metric,
                                                   max_points=args.max_points)["warnings"]
        for warning in payload_warnings:
            print("note: " + warning, file=sys.stderr)
        print(f"wrote {path}  ({os.path.getsize(path) / 1024:.0f} kB, {len(runs)} runs)")
        print(report.summary_table(runs, args.metric))
        return 0

    if args.command == "png":
        ext = "pdf" if args.pdf else "png"
        if args.metric:
            available = sorted({k for r in runs for k in M.series_ids(r)})
            missing = [m for m in args.metric if m not in available]
            if len(missing) == len(args.metric):
                raise SystemExit(
                    "None of those metrics are logged in these runs.\nAvailable: "
                    + ", ".join(available)
                )
            if missing:
                print("skipping metrics not present: " + ", ".join(missing), file=sys.stderr)
                args.metric = [m for m in args.metric if m in available]
            out = args.out
            if os.path.isdir(out) or not os.path.splitext(out)[1]:
                out = os.path.join(out, "metrics." + ext)
            log_keys = [k for k in args.metric if k.startswith("loss")]
            path = figures.plot_metrics(runs, args.metric, out, x_axis=args.x,
                                        smoothing=args.smoothing, ncols=args.ncols,
                                        mode=args.mode, log_keys=log_keys, dpi=args.dpi)
            print(f"wrote {path}")
        else:
            written = figures.plot_groups(runs, args.out, x_axis=args.x,
                                          smoothing=args.smoothing, mode=args.mode,
                                          ext=ext, dpi=args.dpi)
            for path in written:
                print(f"wrote {path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
