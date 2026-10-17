"""Merge a lane's copies instead of choosing one: a test-time check on a blrnet_copy_diag.py dump.

BLRNet's o2o score orders the copies of a lane by fit at chance (copy diag of the 84.82 run:
pairwise 51%), so the copy it outputs is close to a random pick. This tool replays the head's
output from a dump (blrnet_copy_diag.py --save-dump keeps every candidate's output rows and both
scores) with the same decoding and CurvelanesMetric, and evaluates it again with the copies
merged. CPU only, no training.

Geometry merge. The copies of a candidate are the candidates whose drawn rows lie within R px of
its own (mean |dx| over the rows both draw inside the image, network px) and cover at least half
of its drawn rows. Each row the candidate draws inside the image becomes the weighted mean of its
own x and the x of the copies that draw that row inside the image. Its start, length and the rows
it does not draw stay as they are, so decoding keeps its extent. Weights: uniform, the o2o score
or the o2m score. R: --radii (default 5 10 15). With one member (no copy) a row is unchanged.

Selection.
  head      the o2o score without NMS, with the head's nms_topk cap (the evaluated test path);
  o2o_nms   NMS on the o2o score;
  or_nms    NMS on the noisy-OR of the o2o scores of the candidate and its NMS-kernel neighbours,
            1 - prod(1 - s): are the low-score misses probability split across copies?
  o2m_nms   NMS on the o2m score (the head's NMS path).
Every selection runs with every geometry merge. NMS uses the kernel of the rows it selects
(test_cfg.nms_thres) and the nms_topk cap.

Outputs in --out-dir: copy_merge.txt (the table) and copy_merge.json (every sweep).

Usage (next to curvelanes_error_breakdown.py, repr_ceiling.py and blrnet_copy_diag.py):
    python tools/clrbezier/blrnet_copy_diag.py CONFIG CKPT \\
        --save-dump work_dirs/copy/RUN.pkl --out-dir work_dirs/copy/RUN
    python tools/clrbezier/blrnet_copy_merge.py CONFIG work_dirs/copy/RUN.pkl --out-dir work_dirs/copy/RUN
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np

_HERE = osp.dirname(osp.abspath(__file__))


def _load_breakdown():
    if "curvelanes_error_breakdown" in sys.modules:
        return sys.modules["curvelanes_error_breakdown"]
    spec = importlib.util.spec_from_file_location(
        "curvelanes_error_breakdown", osp.join(_HERE, "curvelanes_error_breakdown.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["curvelanes_error_breakdown"] = mod
    spec.loader.exec_module(mod)
    return mod


eb = _load_breakdown()            # also puts the repo root on sys.path
rc = eb.rc
HIT, TIGHT = eb.HIT, eb.TIGHT
WEIGHTS = ("uniform", "o2o", "o2m")
SELECTIONS = {"head": "o2o score, no NMS (the head's test path)",
              "o2o_nms": "o2o score, NMS",
              "or_nms": "noisy-OR of the copies' o2o scores, NMS",
              "o2m_nms": "o2m score, NMS"}
MIN_SHARE = 0.5                   # a copy draws at least this share of the candidate's drawn rows
W_EPS = 1e-6                      # keeps a zero score from dropping a member's weight to exactly 0
COPY_COUNT_FLOOR = 0.3            # "copies merged" is averaged over candidates with a head score above this


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def drawn_rows(rows, n_strips):
    """[N, R] bool: the rows predictions_to_lanes draws for each candidate (start .. end,
    extended to the bottom while x stays in [0, 1]) where x is in [0, 1]."""
    rows = np.asarray(rows, dtype=np.float32)
    xs = rows[:, 6:]
    n, n_rows = xs.shape
    inside = (xs >= 0.0) & (xs <= 1.0)
    out = np.zeros((n, n_rows), dtype=bool)
    for i in range(n):
        start = min(max(0, int(round((1 - float(rows[i, 2])) * n_strips))), n_strips)
        length = int(round(float(np.round(rows[i, 5] * np.float32(n_strips)))))
        end = min(start + length - 1, n_rows - 1)
        while start > 0 and inside[i, start - 1]:
            start -= 1
        if end - start + 1 > 1:                     # predictions_to_lanes drops a lane of <= 1 row
            out[i, start:end + 1] = inside[i, start:end + 1]
    return out


def copy_matrix(rows, drawn, radius, img_w):
    """[N, N] bool: j is a copy of i (the diagonal: i draws something)."""
    xs = np.asarray(rows, dtype=np.float64)[:, 6:]
    both = drawn[:, None, :] & drawn[None, :, :]
    common = both.sum(-1)
    own = drawn.sum(-1)
    dist = np.where(both, np.abs(xs[:, None, :] - xs[None, :, :]), 0.0).sum(-1) * (img_w - 1)
    mean = np.where(common > 0, dist / np.maximum(common, 1), np.inf)
    copies = (common > 0) & (mean < radius) & (common >= MIN_SHARE * own[:, None])
    np.fill_diagonal(copies, own > 0)
    return copies


def merge_rows(rows, drawn, weights, copies):
    """The candidates' rows with each drawn row replaced by the weighted mean over the
    candidate and its copies that draw that row (rows with one member are kept as they are)."""
    rows = np.asarray(rows, dtype=np.float32)
    xs = rows[:, 6:].astype(np.float64)
    member = copies[:, :, None] & drawn[None, :, :]                       # [i, j, row]
    wt = member * np.asarray(weights, dtype=np.float64)[None, :, None]
    den = wt.sum(1)
    num = (wt * xs[None, :, :]).sum(1)
    merge = drawn & (member.sum(1) > 1) & (den > 0)
    out = rows.copy()
    out[:, 6:] = np.where(merge, (num / np.where(den > 0, den, 1.0)).astype(np.float32), rows[:, 6:])
    return out


def noisy_or(score, near):
    """1 - prod(1 - s) over each candidate and its NMS-kernel neighbours."""
    s = np.asarray(score, dtype=np.float64)
    group = near | np.eye(len(s), dtype=bool)
    return 1.0 - np.prod(np.where(group, 1.0 - s[None, :], 1.0), axis=1)


# ----------------------------------------------------------------------------
# one image (worker)
# ----------------------------------------------------------------------------
def _iou(rows, need, gt_masks, ori_hw, offset):
    st, dec = eb._W["st"], eb._W["decoder"]
    n = len(rows)
    valid = np.zeros(n, dtype=bool)
    masks = []
    for i in np.nonzero(need)[0]:
        pts = dec.points(rows[i])
        if pts is None:
            continue
        lane = rc.to_metric_points(pts, ori_hw, offset, st["y_step"])
        if len(lane) == 0:                      # vega drops empty lanes
            continue
        valid[i] = True
        masks.append(rc.lane_mask(lane, ori_hw, st["eval"]))
    iou = np.zeros((len(gt_masks), n), dtype=np.float64)
    vi = np.nonzero(valid)[0]
    if len(gt_masks) and len(vi):
        iou[:, vi] = rc.iou_matrix(gt_masks, masks)
    return iou, valid


def scores_of(rec, near):
    out = {"head": (np.asarray(rec["s_eval"], np.float64), "free"),
           "o2o_nms": (np.asarray(rec["s_o2o"], np.float64), "nms"),
           "or_nms": (noisy_or(rec["s_o2o"], near), "nms")}
    if rec.get("s_o2m") is not None:
        out["o2m_nms"] = (np.asarray(rec["s_o2m"], np.float64), "nms")
    return out


def merge_image(item):
    rec, c = item
    w = eb._W
    st, thresholds, top_k = w["st"], w["thresholds"], w["top_k"]
    ori_hw, offset = tuple(rec["ori_hw"]), int(rec["offset"])
    gt = [ln for ln in rc.read_lines_txt(eb.gt_file(rec["filename"])) if len(ln) > 0]
    gt_masks = [rc.lane_mask([{"x": x, "y": y} for x, y in ln], ori_hw, st["eval"]) for ln in gt]
    rows = np.asarray(rec["rows"], dtype=np.float32)
    n = len(rows)
    near0 = eb.unpack_near(c)
    scores = scores_of(rec, near0)
    need = np.zeros(n, dtype=bool)
    for s, _ in scores.values():
        need |= s >= thresholds[0]
    drawn = drawn_rows(rows, st["n_rows_m1"])
    s_o2o = np.asarray(rec["s_o2o"], np.float64)
    s_o2m = s_o2o if rec.get("s_o2m") is None else np.asarray(rec["s_o2m"], np.float64)
    weights = {"uniform": np.ones(n), "o2o": s_o2o + W_EPS, "o2m": s_o2m + W_EPS}
    counted = np.asarray(rec["s_eval"]) >= COPY_COUNT_FLOOR
    sweeps, merged, iou_ok = {}, {}, None
    copy_cache = {}
    for gname, wname, radius in w["geoms"]:
        if wname is None:
            g_rows, near = rows, near0
        else:
            if radius not in copy_cache:
                copy_cache[radius] = copy_matrix(rows, drawn, radius, st["img_w"])
            copies = copy_cache[radius]
            g_rows = merge_rows(rows, drawn, weights[wname], copies)
            near, _ = eb.kernel_pairs(g_rows, st["n_rows_m1"], st["img_w"], w["nms_thres"])
            merged[gname] = (float((copies.sum(1) - 1)[counted].sum()), int(counted.sum()))
        iou, valid = _iou(g_rows, need, gt_masks, ori_hw, offset)
        if wname is None:
            iou_ok = bool(np.array_equal(valid[need], c["valid"][need])
                          and np.array_equal(iou[:, need], c["iou"][:, need]))
        for sname, (s, mode) in scores.items():
            h50, h75, nout = (np.zeros(len(thresholds), dtype=np.int64) for _ in range(3))
            for k, thr in enumerate(thresholds):
                kept, _, _ = eb.select(s, near, thr, mode, top_k)
                out = eb.output_order(kept, valid, mode, rec["idx"])
                m = np.array([v for _, v in eb.assignment(iou, out).values()])
                h50[k], h75[k], nout[k] = int((m > HIT).sum()), int((m > TIGHT).sum()), len(out)
            sweeps[(sname, gname)] = (h50, h75, nout)
    return dict(n_gt=len(gt), sweeps=sweeps, merged=merged, iou_ok=iou_ok)


def run_images(items, state, jobs):
    if jobs == 1:
        eb._init_worker(state)
        return [merge_image(it) for it in items]
    pool = mp.get_context("fork").Pool(jobs, initializer=eb._init_worker, initargs=(state,))
    try:
        return pool.map(merge_image, items, chunksize=16)
    finally:
        pool.close()
        pool.join()


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def tables(results, thresholds):
    n_gt = sum(r["n_gt"] for r in results)
    keys = list(results[0]["sweeps"]) if results else []
    tab = {}
    for key in keys:
        h50 = sum(r["sweeps"][key][0] for r in results)
        h75 = sum(r["sweeps"][key][1] for r in results)
        nout = sum(r["sweeps"][key][2] for r in results)
        rows = []
        for k, thr in enumerate(thresholds):
            p, r_, f = eb.prf(int(h50[k]), int(nout[k]), n_gt)
            rows.append(dict(thr=float(thr), P=p, R=r_, F1_50=f, F1_75=eb.prf(int(h75[k]), int(nout[k]), n_gt)[2],
                             out=int(nout[k])))
        tab[key] = rows
    return tab, n_gt


def at_recall(rows, recall, field):
    """`field` interpolated at `recall` along the sweep (NaN outside the swept recalls)."""
    r = np.array([x["R"] for x in rows])
    v = np.array([x[field] for x in rows])
    o = np.argsort(r, kind="stable")
    if not len(r) or recall < r.min() - 1e-12 or recall > r.max() + 1e-12:
        return math.nan
    return float(np.interp(recall, r[o], v[o]))


def summarize(tab, merged_tot):
    head = tab[("head", "none")]
    ref = max(head, key=lambda x: (x["F1_50"], -x["thr"]))
    out = []
    for (sname, gname), rows in tab.items():
        b = max(rows, key=lambda x: (x["F1_50"], -x["thr"]))
        b75 = max(rows, key=lambda x: (x["F1_75"], -x["thr"]))
        s, c = merged_tot.get(gname, (0.0, 0))
        out.append(dict(selection=sname, geometry=gname, thr=b["thr"], F1_50=b["F1_50"], P=b["P"], R=b["R"],
                        F1_75=b["F1_75"], P_at_ref_R=at_recall(rows, ref["R"], "P"),
                        F1_75_at_ref_R=at_recall(rows, ref["R"], "F1_75"),
                        max_F1_75=b75["F1_75"], max_F1_75_thr=b75["thr"],
                        copies=(s / c) if c else (0.0 if gname == "none" else math.nan),
                        edge=b["thr"] in (rows[0]["thr"], rows[-1]["thr"])))
    return ref, out


def _pct(v):
    return "   n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:6.2f}"


def format_report(ref, summary, meta, n_img, n_gt, iou_ok, thresholds, say):
    say("=" * 112)
    say(f"BLRNet copy merge (test time)  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"  dump: config {meta.get('config')}, checkpoint {meta.get('checkpoint')}")
    say(f"  images {n_img}, GT lanes {n_gt}; thresholds {thresholds[0]:.2f}..{thresholds[-1]:.2f}; "
        f"nms_thres {meta['test_cfg']['nms_thres']}, nms_topk {meta['test_cfg']['nms_topk']}")
    say(f"  verify: unmerged IoU equals the dump's in {iou_ok[0]} of {iou_ok[1]} images")
    say(f"  reference = head / none at its best F1@50: thr {ref['thr']:.2f}, P {100 * ref['P']:.2f}, "
        f"R {100 * ref['R']:.2f}, F1@50 {100 * ref['F1_50']:.2f}, F1@75 {100 * ref['F1_75']:.2f}")
    say("")
    say("columns: best F1@50 of the sweep (thr, P, R, F1@75 there); P and F1@75 at the reference recall;")
    say("best F1@75 of the sweep; copies = mean copies merged into a candidate with head score >= "
        f"{COPY_COUNT_FLOOR}. '!' = the best F1@50 sits at the end of the swept thresholds.")
    for sname, label in SELECTIONS.items():
        rows = [r for r in summary if r["selection"] == sname]
        if not rows:
            continue
        say("")
        say(f"{sname}: {label}")
        say(f"  {'geometry':<14}{'thr':>6}{'F1@50':>8}{'P':>8}{'R':>8}{'F1@75':>8}   |{'P@refR':>8}{'F1@75@refR':>11}"
            f"   |{'max F1@75':>10}{'thr':>6}{'copies':>8}")
        for r in rows:
            say(f"  {r['geometry']:<14}{r['thr']:>6.2f}{_pct(r['F1_50']):>8}{_pct(r['P']):>8}{_pct(r['R']):>8}"
                f"{_pct(r['F1_75']):>8}{'!' if r['edge'] else ' '}  |{_pct(r['P_at_ref_R']):>8}"
                f"{_pct(r['F1_75_at_ref_R']):>11}   |{_pct(r['max_F1_75']):>10}{r['max_F1_75_thr']:>6.2f}"
                f"{r['copies']:>8.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("dump", help="blrnet_copy_diag.py --save-dump output")
    ap.add_argument("--radii", type=float, nargs="+", default=[5.0, 10.0, 15.0],
                    help="copy radius, network px (mean |dx| over common drawn rows)")
    ap.add_argument("--thr-min", type=float, default=0.10)
    ap.add_argument("--thr-max", type=float, default=0.94)
    ap.add_argument("--thr-step", type=float, default=0.02)
    ap.add_argument("--out-dir", default=None, help="default: the dump's directory")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=None, help="quick look only")
    args = ap.parse_args(argv)

    from mmengine.config import Config
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    with open(args.dump, "rb") as f:
        dump = pickle.load(f)
    meta, slims, results = dump["meta"], dump["records"], dump["results"]
    if args.max_images is not None:
        slims, results = slims[:args.max_images], results[:args.max_images]
    if not slims:
        raise SystemExit("no images in the dump")
    tc = meta["test_cfg"]
    n_steps = int(round((args.thr_max - args.thr_min) / args.thr_step))
    thresholds = [round(args.thr_min + k * args.thr_step, 4) for k in range(n_steps + 1)]
    geoms = [("none", None, 0.0)] + [(f"{wn} R{r:g}", wn, float(r)) for wn in WEIGHTS for r in args.radii]
    state = dict(st=eb.settings(cfg, meta, args), decoder=eb.make_decoder(meta), nms_thres=float(tc["nms_thres"]),
                 top_k=int(tc["nms_topk"]), thresholds=thresholds, geoms=geoms)
    keep_rec = ("filename", "ori_hw", "offset", "rows", "s_o2o", "s_o2m", "s_eval", "idx")
    items = [({k: r.get(k) for k in keep_rec}, {k: res["c"][k] for k in ("iou", "valid", "near", "n")})
             for r, res in zip(slims, results)]
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    tic = time.time()
    out = run_images(items, state, jobs)
    print(f"merged and evaluated {len(out)} images in {time.time() - tic:.0f} s", flush=True)

    tab, n_gt = tables(out, thresholds)
    merged_tot = {}
    for r in out:
        for g, (s, c) in r["merged"].items():
            a, b = merged_tot.get(g, (0.0, 0))
            merged_tot[g] = (a + s, b + c)
    iou_ok = (sum(bool(r["iou_ok"]) for r in out), len(out))
    ref, summary = summarize(tab, merged_tot)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    format_report(ref, summary, meta, len(out), n_gt, iou_ok, thresholds, say)
    out_dir = args.out_dir or osp.dirname(osp.abspath(args.dump))
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "copy_merge.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(out_dir, "copy_merge.json"), "w") as f:
        json.dump(eb._json_safe(dict(dump=args.dump, config=meta.get("config"), checkpoint=meta.get("checkpoint"),
                                     thresholds=thresholds, radii=args.radii, n_gt=n_gt, verify=iou_ok,
                                     reference=ref, summary=summary,
                                     sweeps={f"{s}|{g}": rows for (s, g), rows in tab.items()})), f, indent=1)
    print(f"wrote {osp.join(out_dir, 'copy_merge.txt')}")
    return dict(ref=ref, summary=summary, tab=tab, n_gt=n_gt, iou_ok=iou_ok, merged=merged_tot)


if __name__ == "__main__":
    main()
