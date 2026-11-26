"""Do two differently anchored lane detectors fail on different lanes? (multi-anchor premise test)

Eval-only, CPU. Reads two inference dumps written by curvelanes_error_breakdown.py (--save-dump) on the same
CurveLanes split, e.g. a Bezier head and a start-anchored row head (CLRBezierHead / BLRHead), and asks whether
taking the NEAR field of each lane from the second model and the FAR field from the first would place lanes that the
first model misses. Every head the breakdown supports works (they all emit the official rows).

  base   the model whose output is kept: its lanes, scores, threshold and extent
  donor  the model that lends its near field

Fusion (fixed rule, no GT, no score):
  1. each base output lane is paired with the donor output lane that agrees with it on the base lane's far part:
     the donor's points projected on the base lane (eval space, 224 x 224), median |normal offset| over the base
     lane's far region (u >= u_far, u = 0 at the near end, 1 at the far end, along the base lane's arc) must be
     <= --pair-tol eval px; the closest such donor lane is taken;
  2. the base lane's points move along its normal by w(u) * d_donor(u), where d_donor is the donor lane's offset
     from the base lane at the same arc position and w = 1 for u <= u_near, 0 for u >= u_far, linear between
     (--blend u_near u_far); where the donor lane is shorter than the base lane, its offset at its own end is
     held (no step in the fused lane); the report counts the paired lanes whose donor does not reach the base
     lane's near region, and the donor lanes shared by two or more base lanes (pairing is not one-to-one).
The output is scored with the CurveLanes metric (lanes drawn 5 px wide on 224 x 224, Hungarian on 1 - IoU).

Rows of the report:
  base / donor       each model alone at its own threshold (base must equal its breakdown sweep row)
  fused              the base output after fusion (realizable at test time)
  GT-gated oracle    fusion kept only where it raises the lane's best IoU with a GT lane (uses the GT; a
                     near-upper bound of a fuse / don't-fuse decision, not a strict one under the Hungarian)
  union recall       a GT lane counts if either model hits it (bound of any combination)
Per GT lane: base outcome (hit / near-only: not hit, a base output lane has it as best GT with IoU in
(--near-lo, 0.5], default 0.2 = the 2026-10-23 near-miss definition of score_shift_check check 3 / other miss) x donor hit; signed normal offset of each model's best lane from the GT lane, median
over the GT lane's near third and far third (eval px; + = the GT normal's side, the same for both models) for each
model's best lane on it if that lane's IoU > --near-lo, and the correlation of the near-third offsets.

Read (decided before the run, multi_anchor_analysis.md sec. 3):
  fused - base >= +1.0 F1@50 and the donor places >= 30% of the base's near-only lanes -> errors are complementary;
  design the joint multi-anchored model. This rule wins if both fire (the F1 gain is the outcome; r is a diagnostic).
  fused - base < +0.5 F1@50, or near-third offset correlation r > 0.8 ON THE BASE NEAR-ONLY LANES -> both
  anchorings fail on the same lanes; drop multi-anchoring.

    # the dumps (one GPU pass each; reuse if they exist)
    python tools/clrbezier/curvelanes_error_breakdown.py CFG_A CKPT_A --save-dump work_dirs/err/A.pkl --out-dir work_dirs/err/A
    python tools/clrbezier/curvelanes_error_breakdown.py CFG_B CKPT_B --save-dump work_dirs/err/B.pkl --out-dir work_dirs/err/B
    # the check (CPU); thresholds = the best F1@50 rows of the two summary.txt files
    python tools/clrbezier/anchor_fusion_check.py --base-config CFG_A --base-dump work_dirs/err/A.pkl --base-thr T_A \\
        --donor-config CFG_B --donor-dump work_dirs/err/B.pkl --donor-thr T_B --out-dir work_dirs/anchor_fusion/A_from_B

Outputs in --out-dir: summary.txt, summary.json, gt_lanes.csv (one row per GT lane).
"""
from __future__ import annotations

import argparse
import csv
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
from scipy.optimize import linear_sum_assignment

_HERE = osp.dirname(osp.abspath(__file__))


def _load_breakdown():
    """curvelanes_error_breakdown.py next to this file (it loads repr_ceiling.py and puts the repo on sys.path)."""
    name = "curvelanes_error_breakdown"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, osp.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eb = _load_breakdown()
rc = eb.rc

HIT, TIGHT = eb.HIT, eb.TIGHT
SIDES = ("base", "donor")


# ----------------------------------------------------------------------------
# geometry in the eval space
# ----------------------------------------------------------------------------
def arc_frame(pts):
    """(poly, cum, near_first) of a lane's eval-space points: the polyline without repeated points, its arc
    length, and whether poly[0] is the near end (the larger y); None if it has < 2 distinct points."""
    poly = eb._distinct(np.asarray(pts, dtype=np.float64).reshape(-1, 2), 1e-3)
    if len(poly) < 2:
        return None
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    if cum[-1] <= 0:
        return None
    return poly, cum, bool(poly[0, 1] >= poly[-1, 1])


def u_of(s, frame):
    """Position along the lane, 0 at its near end, 1 at its far end."""
    _, cum, near_first = frame
    u = np.asarray(s, dtype=np.float64) / cum[-1]
    return u if near_first else 1.0 - u


def blend_weight(u, u_near, u_far):
    """1 on the near region (u <= u_near), 0 on the far region (u >= u_far), linear between."""
    return np.clip((u_far - np.asarray(u, dtype=np.float64)) / (u_far - u_near), 0.0, 1.0)


def donor_offsets(base_frame, donor_pts):
    """Arc coordinate s and signed normal offset d of the donor's points on the base lane, sorted by s, only
    those whose foot lies on the base lane (0 <= s <= L), repeated s dropped."""
    poly, cum, _ = base_frame
    s, d = eb._project(np.asarray(donor_pts, dtype=np.float64).reshape(-1, 2), poly, cum)
    tol = 1e-6 * max(cum[-1], 1.0)                 # rounding at the two ends
    on = (s >= -tol) & (s <= cum[-1] + tol)
    s, d = np.clip(s[on], 0.0, cum[-1]), d[on]
    o = np.argsort(s, kind="stable")
    s, d = s[o], d[o]
    if len(s) > 1:
        keep = np.concatenate([[True], np.diff(s) > 1e-9])
        s, d = s[keep], d[keep]
    return s, d


def far_agreement(base_frame, donor_pts, u_far, min_points=3):
    """Median |offset| of the donor lane from the base lane over the base lane's far region, or inf."""
    s, d = donor_offsets(base_frame, donor_pts)
    far = u_of(s, base_frame) >= u_far
    if far.sum() < min_points:
        return math.inf
    return float(np.median(np.abs(d[far])))


def reaches_near(base_frame, donor_pts, u_near):
    """Whether the donor lane has a point on the base lane's near region (u <= u_near)."""
    s, _ = donor_offsets(base_frame, donor_pts)
    return bool((u_of(s, base_frame) <= u_near).any())


def fuse(base_pts, donor_pts, u_near, u_far, eps=1e-9):
    """The base lane's points with its near field moved onto the donor lane (see the module docstring).
    Returns a new [n, 2] array (the same points where nothing moves)."""
    pts = np.asarray(base_pts, dtype=np.float64).reshape(-1, 2)
    frame = arc_frame(pts)
    if frame is None:
        return pts.copy()
    poly, cum, _ = frame
    s_d, d_d = donor_offsets(frame, donor_pts)
    out = pts.copy()
    if len(s_d) < 2:
        return out
    s_b, _ = eb._project(pts, poly, cum)
    s_b = np.clip(s_b, 0.0, cum[-1])
    w = blend_weight(u_of(s_b, frame), u_near, u_far)
    move = w * np.interp(s_b, s_d, d_d)          # beyond the donor's ends np.interp holds its end offset
    sel = np.abs(move) > eps
    if sel.any():
        _, nrm = eb._at(poly, cum, s_b[sel])
        out[sel] = pts[sel] + move[sel, None] * nrm
    return out


def signed_offsets(gframe, pred_pts):
    """Median signed normal offset of a prediction from the GT lane over the GT's near third and far third
    (eval px), NaN where the prediction has no point there."""
    s, d = donor_offsets(gframe, pred_pts)
    u = u_of(s, gframe)
    near, far = u <= 1.0 / 3.0, u >= 2.0 / 3.0
    return (float(np.median(d[near])) if near.any() else math.nan,
            float(np.median(d[far])) if far.any() else math.nan)


# ----------------------------------------------------------------------------
# one image (worker)
# ----------------------------------------------------------------------------
_W = {}


def _init_worker(state):
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _W.update(state)


def output_lanes(rec, side):
    """The lanes CurvelanesMetric receives from one model at its threshold: list of (eval-space points [n, 2],
    original-pixel metric lane, score)."""
    w = _W[side]
    dec, st = w["decoder"], w["st"]
    ori_hw, offset = tuple(rec["ori_hw"]), int(rec["offset"])
    rows, score = rec["rows"], np.asarray(rec["score"])
    n = len(score)
    valid = np.zeros(n, dtype=bool)
    metric = [None] * n
    for i in range(n):
        if score[i] < w["thr"]:
            continue                          # never selected; skip the decode
        pts = dec.points(rows[i])
        if pts is None:
            continue
        lane = rc.to_metric_points(pts, ori_hw, offset, st["y_step"])
        if len(lane) == 0:                     # vega drops empty lanes
            continue
        valid[i] = True
        metric[i] = lane
    near = None
    if w["mode"] == "nms":
        near, _ = eb.kernel_pairs(rows, w["n_strips"], w["img_w"], w["nms_thres"])
    kept, _, _ = eb.select(score, near, w["thr"], w["mode"], w["top_k"])
    out = eb.output_order(kept, valid, w["mode"], rec["idx"])
    ev = _W["ev"]
    return [(eb._eval_space(metric[i], ori_hw, ev), metric[i], float(score[i])) for i in out]


def eval_mask(pts, ev):
    """The metric's mask of eval-space points (identical to lane_mask of the original-pixel lane)."""
    return rc.lane_mask([{"x": float(x), "y": float(y)} for x, y in pts],
                        (ev["eval_height"], ev["eval_width"]), ev)


def match(iou):
    """vega evaluate_core: Hungarian on 1 - IoU. Per GT lane the IoU of its partner and the partner's index."""
    g_iou = np.zeros(iou.shape[0])
    g_pred = np.full(iou.shape[0], -1, dtype=np.int64)
    if iou.size:
        r, c = linear_sum_assignment(1.0 - iou)
        g_iou[r], g_pred[r] = iou[r, c], c
    return g_iou, g_pred


def analyse_image(pair):
    rec_b, rec_d = pair
    ev, u_near, u_far = _W["ev"], _W["u_near"], _W["u_far"]
    ori_hw = tuple(rec_b["ori_hw"])
    gt = [ln for ln in rc.read_lines_txt(eb.gt_file(rec_b["filename"])) if len(ln) > 0]
    gt_dicts = [[{"x": x, "y": y} for x, y in ln] for ln in gt]
    gt_masks = [rc.lane_mask(ln, ori_hw, ev) for ln in gt_dicts]
    gframes = []
    for ln in gt_dicts:
        g = eb.gt_polyline(ln, ori_hw, ev)
        gframes.append(None if g is None else (g[0], g[1], bool(g[0][0, 1] >= g[0][-1, 1])))

    base = output_lanes(rec_b, "base")
    donor = output_lanes(rec_d, "donor")
    base_masks = [rc.lane_mask(m, ori_hw, ev) for _, m, _ in base]
    donor_masks = [rc.lane_mask(m, ori_hw, ev) for _, m, _ in donor]

    # fusion: pair each base lane with the donor lane that agrees on its far part
    fused_pts, paired, n_short = [], [], 0
    for pts, _, _ in base:
        frame = arc_frame(pts)
        best, best_j = math.inf, -1
        if frame is not None:
            for j, (dp, _, _) in enumerate(donor):
                a = far_agreement(frame, dp, u_far)
                if a < best:
                    best, best_j = a, j
        if best_j >= 0 and best <= _W["pair_tol"]:
            fused_pts.append(fuse(pts, donor[best_j][0], u_near, u_far))
            paired.append(best_j)
            n_short += not reaches_near(frame, donor[best_j][0], u_near)
        else:
            fused_pts.append(pts)
            paired.append(-1)
    fused_masks = [eval_mask(p, ev) if paired[k] >= 0 else base_masks[k] for k, p in enumerate(fused_pts)]

    n_gt = len(gt)
    iou_b = rc.iou_matrix(gt_masks, base_masks) if n_gt and base else np.zeros((n_gt, len(base)))
    iou_d = rc.iou_matrix(gt_masks, donor_masks) if n_gt and donor else np.zeros((n_gt, len(donor)))
    iou_f = rc.iou_matrix(gt_masks, fused_masks) if n_gt and base else np.zeros((n_gt, len(base)))
    # gated: per base lane, fused only where it raises the lane's best IoU with a GT lane
    gate = np.zeros(len(base), dtype=bool)
    if n_gt and base:
        gate = iou_f.max(axis=0) > iou_b.max(axis=0)
    iou_g = np.where(gate[None, :], iou_f, iou_b)

    res = {}
    for key, iou in (("base", iou_b), ("donor", iou_d), ("fused", iou_f), ("gated", iou_g)):
        g_iou, _ = match(iou)
        res[key] = dict(h50=int((g_iou > HIT).sum()), h75=int((g_iou > TIGHT).sum()), n_out=iou.shape[1],
                        g_iou=g_iou)

    lanes = []
    hb, hd = res["base"]["g_iou"] > HIT, res["donor"]["g_iou"] > HIT
    for g in range(n_gt):
        if hb[g]:
            status = "hit"
        else:
            own = [k for k in range(len(base)) if int(np.argmax(iou_b[:, k])) == g
                   and _W["near_lo"] < iou_b[g, k] <= HIT]
            status = "near_only" if own else "other_miss"
        row = dict(image=rec_b["filename"], gt=g, base=status, donor_hit=bool(hd[g]),
                   base_iou=float(res["base"]["g_iou"][g]), donor_iou=float(res["donor"]["g_iou"][g]),
                   fused_hit=bool(res["fused"]["g_iou"][g] > HIT), gated_hit=bool(res["gated"]["g_iou"][g] > HIT),
                   fused_iou=float(res["fused"]["g_iou"][g]),
                   base_near=math.nan, base_far=math.nan, donor_near=math.nan, donor_far=math.nan, paired=False)
        if gframes[g] is not None:
            for side, iou, lanes_s in (("base", iou_b, base), ("donor", iou_d, donor)):
                if iou.shape[1] and iou[g].max() > _W["near_lo"]:
                    k = int(np.argmax(iou[g]))
                    row[f"{side}_near"], row[f"{side}_far"] = signed_offsets(gframes[g], lanes_s[k][0])
                    if side == "base":
                        row["paired"] = paired[k] >= 0
        lanes.append(row)
    for key in res:
        res[key].pop("g_iou")
    used = [p for p in paired if p >= 0]
    n_shared = sum(1 for p in used if used.count(p) > 1)
    return dict(filename=rec_b["filename"], n_gt=n_gt, res=res, lanes=lanes,
                n_paired=len(used), n_base=len(base), n_short=int(n_short), n_shared=int(n_shared))


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def prf(h, n_out, n_gt):
    p = h / n_out if n_out else 0.0
    r = h / n_gt if n_gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return math.nan, int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def _med_abs(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(np.abs(v))) if len(v) else math.nan


def report(results, args, say):
    n_gt = sum(r["n_gt"] for r in results)
    tot = {k: dict(h50=0, h75=0, n_out=0) for k in ("base", "donor", "fused", "gated")}
    for r in results:
        for k in tot:
            for f in ("h50", "h75", "n_out"):
                tot[k][f] += r["res"][k][f]
    lanes = [ln for r in results for ln in r["lanes"]]
    table = {}
    n_paired = sum(r["n_paired"] for r in results)
    n_short = sum(r["n_short"] for r in results)
    n_shared = sum(r["n_shared"] for r in results)
    say(f"images {len(results)}; GT lanes {n_gt}; base lanes paired with a donor lane "
        f"{n_paired} / {sum(r['n_base'] for r in results)}; of the paired, donor does not reach the near region "
        f"{n_short} (its end offset held), share a donor lane with another base lane {n_shared}")
    say(f"blend u_near {args.blend[0]:g} u_far {args.blend[1]:g}; pair tolerance {args.pair_tol:g} eval px; "
        f"near-only IoU ({args.near_lo:g}, 0.5]")
    say("")
    say(f"  {'':16s}{'P':>8}{'R':>8}{'F1@50':>8}{'F1@75':>8}   (CurveLanes metric, %)")
    for k, label in (("base", "base"), ("donor", "donor"), ("fused", "fused"), ("gated", "GT-gated oracle")):
        p, rr, f1 = prf(tot[k]["h50"], tot[k]["n_out"], n_gt)
        f75 = prf(tot[k]["h75"], tot[k]["n_out"], n_gt)[2]
        table[k] = dict(P=p, R=rr, F1_50=f1, F1_75=f75)
        say(f"  {label:16s}{100 * p:8.2f}{100 * rr:8.2f}{100 * f1:8.2f}{100 * f75:8.2f}")
    union = sum(1 for ln in lanes if ln["base"] == "hit" or ln["donor_hit"])
    say(f"  {'union recall':16s}{'':8s}{100 * union / max(n_gt, 1):8.2f}")
    say("")
    cross = {}
    say("GT lanes by base outcome x donor hit")
    say(f"  {'base':12s}{'n':>8}{'donor hit':>11}{'fused hit':>11}{'gated hit':>11}"
        f"{'|base near|':>13}{'|donor near|':>14}")
    for st in ("hit", "near_only", "other_miss"):
        sub = [ln for ln in lanes if ln["base"] == st]
        n = len(sub)
        c = dict(n=n, donor_hit=sum(ln["donor_hit"] for ln in sub) / n if n else math.nan,
                 fused_hit=sum(ln["fused_hit"] for ln in sub) / n if n else math.nan,
                 gated_hit=sum(ln["gated_hit"] for ln in sub) / n if n else math.nan,
                 base_near_abs=_med_abs([ln["base_near"] for ln in sub]),
                 donor_near_abs=_med_abs([ln["donor_near"] for ln in sub]))
        cross[st] = c
        say(f"  {st:12s}{n:8d}{100 * c['donor_hit']:10.1f}%{100 * c['fused_hit']:10.1f}%"
            f"{100 * c['gated_hit']:10.1f}%{c['base_near_abs']:13.2f}{c['donor_near_abs']:14.2f}")
    say("  (|near| = median |signed offset| from the GT lane over its near third, eval px, of each model's best lane)")
    say("")
    corr = {}
    for label, sub in (("all lanes both models reach", lanes),
                       ("base near-only lanes", [ln for ln in lanes if ln["base"] == "near_only"])):
        r_near, n_near = _pearson([ln["base_near"] for ln in sub], [ln["donor_near"] for ln in sub])
        r_far, n_far = _pearson([ln["base_far"] for ln in sub], [ln["donor_far"] for ln in sub])
        both = [ln for ln in sub if np.isfinite(ln["base_near"]) and np.isfinite(ln["donor_near"])]
        same = (sum(1 for ln in both if np.sign(ln["base_near"]) == np.sign(ln["donor_near"])) / len(both)
                if both else math.nan)
        corr[label] = dict(r_near=r_near, n_near=n_near, r_far=r_far, n_far=n_far, same_side_near=same)
        say(f"offset correlation, {label}: near third r {r_near:.3f} (n {n_near}), far third r {r_far:.3f} "
            f"(n {n_far}); near offsets on the same side {100 * same:.1f}%")
    say("")
    gain = table["fused"]["F1_50"] - table["base"]["F1_50"]
    donor_places = cross["near_only"]["donor_hit"]
    r_near = corr["base near-only lanes"]["r_near"]
    high_r = bool(np.isfinite(r_near) and r_near > 0.8)
    if gain >= 0.01 and donor_places >= 0.30:
        verdict = "COMPLEMENTARY: fused >= +1.0 F1@50 and the donor places >= 30% of near-only lanes -> design it"
        if high_r:
            verdict += " (note: near-only near-third r > 0.8; the gain rule wins)"
    elif gain < 0.005 or high_r:
        verdict = "SAME FAILURE: fused < +0.5 F1@50 or near-only near-third r > 0.8 -> drop multi-anchoring"
    else:
        verdict = "IN BETWEEN: neither rule fires; look at the gated row and the near-only line"
    say(f"fused - base = {100 * gain:+.2f} F1@50; donor places {100 * donor_places:.1f}% of base near-only lanes")
    say(f"VERDICT: {verdict}")
    return dict(table=table, cross=cross, corr=corr, union_recall=union / max(n_gt, 1), gain_f1_50=gain,
                n_paired=n_paired, n_short=n_short, n_shared=n_shared,
                verdict=verdict, n_gt=n_gt, n_images=len(results))


# ----------------------------------------------------------------------------
def side_state(cfg_path, dump_path, thr, mode, args):
    from types import SimpleNamespace
    from mmengine.config import Config
    cfg = Config.fromfile(cfg_path, import_custom_modules=False)
    with open(dump_path, "rb") as f:
        dump = pickle.load(f)
    meta = dump["meta"]
    tc = meta["test_cfg"]
    mode = mode or ("nms" if bool(tc.get("use_nms", True)) else "free")
    thr = float(tc["conf_threshold"]) if thr is None else float(thr)
    st = eb.settings(cfg, meta, SimpleNamespace())
    floor = float(meta.get("floor", 0.0))
    if thr < floor - 1e-12:
        raise SystemExit(f"{dump_path}: threshold {thr} is below the dump's floor {floor}")
    trunc = [r["trunc"] for r in dump["records"] if r["trunc"] is not None and r["trunc"] >= thr]
    state = dict(decoder=eb.make_decoder(meta), st=st, thr=thr, mode=mode, n_strips=int(meta["n_strips"]),
                 img_w=int(meta["img_w"]), nms_thres=float(tc["nms_thres"]), top_k=int(tc["nms_topk"]))
    info = dict(config=cfg_path, dump=dump_path, checkpoint=meta.get("checkpoint"), head=meta.get("head_cls"),
                thr=thr, mode=mode, truncated_images=len(trunc))
    return state, dump["records"], info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for side in SIDES:
        ap.add_argument(f"--{side}-config", required=True)
        ap.add_argument(f"--{side}-dump", required=True, help="curvelanes_error_breakdown.py --save-dump file")
        ap.add_argument(f"--{side}-thr", type=float, default=None,
                        help="default: the config's conf_threshold (pass the swept best F1@50)")
        ap.add_argument(f"--{side}-mode", choices=["nms", "free"], default=None,
                        help="default: the config's test_cfg.use_nms")
    ap.add_argument("--blend", type=float, nargs=2, default=[1.0 / 3.0, 2.0 / 3.0], metavar=("U_NEAR", "U_FAR"))
    ap.add_argument("--pair-tol", type=float, default=2.0, help="far-part agreement to pair two lanes, eval px")
    ap.add_argument("--near-lo", type=float, default=0.2, help="near-only: best base IoU in (near_lo, 0.5]")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out-dir", default="work_dirs/anchor_fusion")
    args = ap.parse_args(argv)
    if not 0.0 <= args.blend[0] < args.blend[1] <= 1.0:
        ap.error("--blend needs 0 <= U_NEAR < U_FAR <= 1")

    state, info, recs = {}, {}, {}
    for side in SIDES:
        st, records, inf = side_state(getattr(args, f"{side}_config"), getattr(args, f"{side}_dump"),
                                      getattr(args, f"{side}_thr"), getattr(args, f"{side}_mode"), args)
        state[side], info[side], recs[side] = st, inf, records
    ev_b, ev_d = state["base"]["st"]["eval"], state["donor"]["st"]["eval"]
    if ev_b != ev_d:
        raise SystemExit(f"the two configs evaluate differently: {ev_b} vs {ev_d}")
    state.update(ev=ev_b, u_near=args.blend[0], u_far=args.blend[1], pair_tol=args.pair_tol, near_lo=args.near_lo)
    donor_by = {r["filename"]: r for r in recs["donor"]}
    pairs = [(r, donor_by[r["filename"]]) for r in recs["base"] if r["filename"] in donor_by]
    missing = len(recs["base"]) - len(pairs)
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    if not pairs:
        raise SystemExit("no image is in both dumps")

    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    tic = time.time()
    if jobs == 1:
        _init_worker(state)
        results = [analyse_image(p) for p in pairs]
    else:
        with mp.get_context("fork").Pool(jobs, initializer=_init_worker, initargs=(state,)) as pool:
            results = list(pool.imap(analyse_image, pairs, chunksize=16))
    print(f"analysed {len(results)} images in {time.time() - tic:.0f} s", flush=True)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 100)
    say(f"anchor fusion check  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for side in SIDES:
        i = info[side]
        say(f"  {side}: {i['config']}  ({i['head']}, thr {i['thr']:g}, {i['mode']}, ckpt {i['checkpoint']})")
        if i["truncated_images"]:
            say(f"    WARNING: {i['truncated_images']} images of this dump stored only some predictions above the "
                f"threshold (--max-per-image); its row is approximate")
    if missing:
        say(f"  {missing} base images are not in the donor dump (left out)")
    say("")
    summary = report(results, args, say)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(eb._json_safe(dict(info=info, args=vars(args), **summary)), f, indent=1)
    rows = [ln for r in results for ln in r["lanes"]]
    if rows:
        with open(osp.join(args.out_dir, "gt_lanes.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"wrote {args.out_dir}/summary.txt, summary.json, gt_lanes.csv")
    return dict(results=results, summary=summary)


if __name__ == "__main__":
    main()
