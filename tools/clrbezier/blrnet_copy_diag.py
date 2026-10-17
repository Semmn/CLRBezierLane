"""Do BLRNet's scores pick the best-fitting copy of a lane, and when is that copy decided?

One inference pass on CurveLanes valid keeps, for every image, every anchor's lane
after every refinement step (GRU steps 0..K-1; the proposal first when
refine_cfg.proposal; the strip output last with strip_cfg mode "final") and both
classifier scores, o2m and o2o, before any threshold, top-k or NMS. The static
anchors are kept once. The evaluation of the head's output is rebuilt exactly as
curvelanes_error_breakdown.py does it (same selection, the head's own decoding,
CurvelanesMetric and vega's mask IoU, 224x224, width 5), and the report adds what
the error breakdown cannot see:

1. coverage by step: share of GT lanes with some anchor at IoU > 0.5 / 0.75 after
   each step (does refinement create the correct candidates?);
2. the error breakdown of the evaluated output at the operating threshold (the
   standard curvelanes_error_breakdown report) and the share of missed GT lanes
   that had a correct copy in the output step (low_score, suppressed, capped,
   taken);
3. copy ranking at the output step, for the o2m and the o2o score (with
   strip_cfg.quality also the strip quality and o2o x quality). A candidate
   is a copy of the GT lane it overlaps most (IoU > 0.3). For every GT lane with
   two or more copies:
     winner = best        the top-scored copy is the best-fitting one
     IoU lost             best copy's IoU minus the winner's
     lost@50 / lost@75    the best copy clears IoU 0.5 (0.75), the winner does not
     pairwise accuracy    of the copy pairs whose IoU differs by >= 0.1, the share
                          the score orders the same way (a score tie counts half);
                          split into both correct (IoU > 0.5: decides F1@75), one
                          correct (decides F1@50), both wrong, and by IoU gap
     Spearman             score vs IoU over the copies of one lane (>= 3 copies)
4. ownership by step, for GT lanes with a correct copy (IoU > 0.5) in the output:
     settled at step k    the output's best copy was already the best fit after
                          step k
     nearest anchor       the anchor whose static lane lay closest to the GT lane
                          (mean |dx| over the GT's annotated rows, network px)
   with the share of lanes whose o2m / o2o winner is the output's best copy, the
   step-0 best copy, or the nearest anchor, and recall at the operating
   threshold for lanes settled at the first step vs settled later ("late").
   In the CLR cascade, late lanes were the ones the score lost.

    # one inference pass (GPU); the dump lets later runs skip inference
    python tools/clrbezier/blrnet_copy_diag.py CONFIG CKPT \\
        --save-dump work_dirs/copy/RUN.pkl --out-dir work_dirs/copy/RUN
    # another threshold or mode from the dump (CPU)
    python tools/clrbezier/blrnet_copy_diag.py CONFIG --load-dump work_dirs/copy/RUN.pkl --thr 0.4

--mode defaults to the config's test path: without NMS (test_cfg.use_nms False)
the o2o score with the head's nms_topk cap, with NMS the o2m score (o2o with
o2m_branch_cfg). Without NMS a correct copy outside the top-k cap scores 0 on the
test path, so the breakdown counts it as low_score. Needs curvelanes_error_breakdown.py and repr_ceiling.py next to
this file.

Outputs in --out-dir: copy_diag.txt, copy_diag.json, copy_lanes.csv (one row per
GT lane).
"""
from __future__ import annotations

import argparse
import copy
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
import torch

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
HIT, TIGHT, NEAR = eb.HIT, eb.TIGHT, eb.NEAR
PAIR_GAP = 0.1                    # copy pairs closer than this in IoU are not ranked
GAP_BINS = ((0.1, 0.2, "0.1-0.2"), (0.2, 0.4, "0.2-0.4"), (0.4, 1.01, ">=0.4"))
PAIR_KINDS = ("both correct", "one correct", "both wrong")
MIN_COVERAGE = 0.3                # nearest anchor: must cover this share of the GT's annotated rows
LOST = ("low_score", "suppressed", "capped", "taken")


# ----------------------------------------------------------------------------
# inference: everything the head computes for every anchor
# ----------------------------------------------------------------------------
def static_anchor_rows(head):
    """[N, 6 + R] official-layout rows of the static anchors (image independent)."""
    mod = sys.modules[type(head).__module__]
    with torch.no_grad():
        params0 = head.anchor_params[None].float()
        cps0 = mod.anchor_control_points(params0)
        zeros2 = params0.new_zeros((1, params0.shape[1], 2))
        lanes = head.official_layout(zeros2, mod.curve_points(cps0, head.t_out), mod.bezier_end_slopes(cps0))
    return lanes[0].float().cpu().numpy()


def step_labels(head, n_preds, has_final):
    labels = []
    if getattr(head, "proposal", None) is not None:
        labels.append("proposal")
    labels += [f"step{i}" for i in range(n_preds - len(labels))]
    if has_final:
        labels.append("merge" if getattr(head, "merge", None) is not None else "strip")
    return labels


class Capture:
    """Wraps head._forward to keep the main branch's output dict."""

    def __init__(self, head):
        self.head, self.out = head, None
        self._orig = head._forward

        def wrapped(*a, **k):
            out = self._orig(*a, **k)
            if k.get("branch", "main") == "main":
                self.out = out
            return out
        head._forward = wrapped

    def pop(self):
        out, self.out = self.out, None
        if out is None:
            raise RuntimeError("BLRHead._forward was not called for the main branch")
        return out

    def restore(self):
        self.head._forward = self._orig


def eval_source(head, out):
    """Which score the head's test path uses: 'o2o' or 'o2m' (forward_test's rule)."""
    use_nms = bool((head.test_cfg or {}).get("use_nms", True))
    source = head.test_score or ("o2m" if use_nms else "o2o")
    return "o2o" if source == "o2o" or out["cls_o2m"] is None else "o2m"


@torch.no_grad()
def batch_records(head, x, samples, cap):
    """Run the head on neck features x (eval mode) and return (records, the test path's
    pred dicts). One record per image: every output step's rows for every anchor, the
    o2o and o2m scores, the evaluated score with the head's top-k cap applied."""
    preds_test = head(x)
    out = cap.pop()
    cfg = head.test_cfg or {}
    steps = list(out["preds"])
    if out.get("final_pred") is not None:
        steps.append(out["final_pred"])
    labels = step_labels(head, len(out["preds"]), out.get("final_pred") is not None)
    out_idx = len(steps) - 1
    if cfg.get("output_step") is not None:
        out_idx = int(cfg["output_step"])
    s_o2o = torch.softmax(out["cls_o2o"].float(), -1)[..., 1]
    if out["cls_o2m"] is not None:
        s_o2m = torch.softmax(out["cls_o2m"].float(), -1)[..., 1]
    elif getattr(head, "cls_o2m", None) is not None:        # o2m_branch_cfg: the neighbour score final_sa uses
        s_o2m = torch.softmax(head.cls_o2m(out["h"]).float(), -1)[..., 1]
    else:
        s_o2m = None
    strip = out.get("strip")
    s_q = (torch.sigmoid(strip["q_logit"].float()) if isinstance(strip, dict) and "q_logit" in strip
           else None)                                       # strip_cfg.quality
    source = eval_source(head, out)
    z = out["cls_o2o"] if source == "o2o" else out["cls_o2m"]
    s_eval = torch.softmax(z.float(), -1)[..., 1]
    if bool(cfg.get("quality_score", False)) and s_q is not None:   # forward_test's joint score
        s_eval = (s_eval * s_q).clamp(1e-6, 1.0 - 1e-6)
    if not bool(cfg.get("use_nms", True)):                   # forward_test's nms_topk cap
        topk = cfg.get("nms_topk")
        if topk is not None and 0 < int(topk) < z.shape[1]:
            if s_q is not None and cfg.get("quality_score"):
                fg = torch.log(s_eval) - torch.log1p(-s_eval)
            else:
                fg = z[..., 1] - z[..., 0]
            keep = torch.zeros_like(fg, dtype=torch.bool)
            keep.scatter_(1, fg.topk(int(topk), dim=1).indices, True)
            s_eval = torch.where(keep, s_eval, torch.zeros_like(s_eval))
    stack = torch.stack([s.float() for s in steps], dim=1).cpu().numpy()          # [B, T, N, 6 + R]
    recs = []
    for b, ds in enumerate(samples):
        meta = ds.metainfo
        recs.append(dict(filename=str(meta["filename"]),
                         ori_hw=(int(meta["ori_shape"][0]), int(meta["ori_shape"][1])),
                         offset=int(meta["crop_offset"][1]),
                         steps=stack[b], labels=labels, out_idx=out_idx, source=source,
                         s_o2o=s_o2o[b].cpu().numpy().astype(np.float64),
                         s_o2m=None if s_o2m is None else s_o2m[b].cpu().numpy().astype(np.float64),
                         s_eval=s_eval[b].cpu().numpy().astype(np.float64),
                         s_q=None if s_q is None else s_q[b].cpu().numpy().astype(np.float64),
                         quality_score=bool(cfg.get("quality_score", False)) and s_q is not None))
    return recs, preds_test


def check_records(head, preds_test, recs, decoder, meta):
    """The rebuilt test path equals the head's: the output rows equal the pred dict's
    rows; the selection at the config threshold with the rebuilt score equals the one
    with the pred dict's score; get_lanes keeps exactly that selection (the error
    breakdown's verify). Returns (images checked, mismatches)."""
    pd = preds_test[-1]
    rows_t, score_t = eb.pred_rows(pd)
    cfg = head.test_cfg
    thr = float(cfg["conf_threshold"])
    mode = "nms" if bool(cfg.get("use_nms", True)) else "free"
    top_k = int(cfg["nms_topk"])
    bad = 0
    for b, rec in enumerate(recs):
        r_t = rows_t[b].float().cpu().numpy()
        r_m = rec["steps"][rec["out_idx"]]
        same_rows = np.array_equal(r_t[:, 2:], r_m[:, 2:])
        st = score_t[b].float().cpu().numpy().astype(np.float64)
        near, _ = eb.kernel_pairs(r_m, meta["n_strips"], meta["img_w"], float(cfg["nms_thres"]))
        k_t, _, _ = eb.select(st, near, thr, mode, top_k)
        k_m, _, _ = eb.select(rec["s_eval"], near, thr, mode, top_k)
        bad += (not same_rows) or k_t != k_m
    eb_recs = eb.image_records(pd, [_Sample(r) for r in recs], 0.0, rows_t.shape[1])
    c, m = eb.verify(head, pd, eb_recs, decoder, meta, top_k)
    return len(recs) + c, bad + m


class _Sample:
    def __init__(self, rec):
        h, w = rec["ori_hw"]
        self.metainfo = dict(filename=rec["filename"], ori_shape=(h, w, 3), crop_offset=[0, rec["offset"]])


def run_inference(cfg, args, device):
    from mmengine.runner import Runner
    model = eb.build_model(cfg, args.checkpoint, device)
    head = model.bbox_head
    meta = eb.head_meta(head)
    meta["anchor_rows"] = static_anchor_rows(head)
    decoder = eb.Decoder(type(head).predictions_to_lanes, meta["prior_ys"], meta["n_strips"], meta["test_cfg"])
    loader = Runner.build_dataloader(copy.deepcopy(cfg.test_dataloader if args.split == "test"
                                                   else cfg.val_dataloader))
    cap = Capture(head)
    records, checked, bad = [], 0, 0
    tic = time.time()
    try:
        for bi, data in enumerate(loader):
            data = model.data_preprocessor(data, False)
            x = model.extract_feat(data["inputs"])
            recs, preds_test = batch_records(head, x, data["data_samples"], cap)
            if bi < args.verify_batches:
                c, m = check_records(head, preds_test, recs, decoder, meta)
                checked, bad = checked + c, bad + m
            records += recs
            if len(records) % 2000 < len(recs):
                print(f"  {len(records)} images ({time.time() - tic:.0f} s)", flush=True)
            if args.max_images is not None and len(records) >= args.max_images:
                break
    finally:
        cap.restore()
    print(f"verified {checked} checks against the head's test path: {bad} mismatches", flush=True)
    return records, meta, dict(checked=checked, mismatches=bad)


# ----------------------------------------------------------------------------
# one image (worker)
# ----------------------------------------------------------------------------
def _iou_rows(rows, gt_masks, ori_hw, offset):
    st, dec = eb._W["st"], eb._W["decoder"]
    n = len(rows)
    valid = np.zeros(n, dtype=bool)
    masks, rowmaps = [], [None] * n
    for i in range(n):
        pts = dec.points(rows[i])
        if pts is None:
            continue
        rowmaps[i] = eb.pred_geometry(pts, st)[0]
        lane = rc.to_metric_points(pts, ori_hw, offset, st["y_step"])
        if len(lane) == 0:
            continue
        valid[i] = True
        masks.append(rc.lane_mask(lane, ori_hw, st["eval"]))
    iou = np.zeros((len(gt_masks), n), dtype=np.float64)
    vi = np.nonzero(valid)[0]
    if len(gt_masks) and len(vi):
        iou[:, vi] = rc.iou_matrix(gt_masks, masks)
    return iou, valid, rowmaps


def _anchor_dx(rowmaps, anns):
    """[G, N] mean |dx| (network px) of each anchor over each GT lane's annotated rows;
    inf where the anchor covers less than MIN_COVERAGE of them."""
    dx = np.full((len(anns), len(rowmaps)), np.inf)
    for g, ann in enumerate(anns):
        if ann is None:
            continue
        r, xs = ann
        for p, rm in enumerate(rowmaps):
            if not rm:
                continue
            common = [k for k, row in enumerate(r.tolist()) if row in rm]
            if len(common) < MIN_COVERAGE * len(r) or not common:
                continue
            dx[g, p] = float(np.mean(np.abs(np.array([rm[int(r[k])] for k in common]) - xs[common])))
    return dx


def analyse(rec):
    st = eb._W["st"]
    n = rec["steps"].shape[1]
    out_rows = rec["steps"][rec["out_idx"]]
    c = eb.analyse_image(dict(filename=rec["filename"], ori_hw=rec["ori_hw"], offset=rec["offset"],
                              rows=out_rows, score=rec["s_eval"], idx=np.arange(n)))
    ori_hw, offset = tuple(rec["ori_hw"]), int(rec["offset"])
    gt = [ln for ln in rc.read_lines_txt(eb.gt_file(rec["filename"])) if len(ln) > 0]
    _, gt_rows = eb.gt_geometry(gt, ori_hw, offset, st)
    gt_masks = [rc.lane_mask([{"x": x, "y": y} for x, y in ln], ori_hw, st["eval"]) for ln in gt]
    step_iou = []
    for k, rows in enumerate(rec["steps"]):
        if k == rec["out_idx"]:
            step_iou.append(c["iou"].astype(np.float32))
        else:
            step_iou.append(_iou_rows(rows, gt_masks, ori_hw, offset)[0].astype(np.float32))
    a_iou, _, a_maps = _iou_rows(eb._W["anchor_rows"], gt_masks, ori_hw, offset)
    anns = [eb.annotated_rows(r, st["img_w"]) for r in gt_rows]
    return dict(c=c, step_iou=np.stack(step_iou) if step_iou else np.zeros((0, len(gt), n), np.float32),
                anchor_iou=a_iou.astype(np.float32), anchor_dx=_anchor_dx(a_maps, anns).astype(np.float32))


def slim(rec):
    """What the report needs from a record once it is analysed (the dump keeps this)."""
    return dict(filename=rec["filename"], ori_hw=rec["ori_hw"], offset=rec["offset"], labels=rec["labels"],
                out_idx=rec["out_idx"], source=rec["source"], s_o2o=rec["s_o2o"], s_o2m=rec["s_o2m"],
                s_eval=rec["s_eval"], score=rec["s_eval"], idx=np.arange(rec["steps"].shape[1]),
                s_q=rec.get("s_q"), quality_score=rec.get("quality_score", False),
                rows=rec["steps"][rec["out_idx"]])


def analyse_records(records, state, jobs, chunk):
    """analyse() over the records in chunks (the per-step rows are dropped from
    `records` as each chunk is done). Returns (results, slim records)."""
    results, slims = [], []
    pool = None if jobs == 1 else mp.get_context("fork").Pool(jobs, initializer=eb._init_worker,
                                                              initargs=(state,))
    if pool is None:
        eb._init_worker(state)
    try:
        for i in range(0, len(records), chunk):
            part = records[i:i + chunk]
            results += pool.map(analyse, part, chunksize=16) if pool else [analyse(r) for r in part]
            slims += [slim(r) for r in part]
            for j in range(i, min(i + chunk, len(records))):
                records[j] = None
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return results, slims


def resweep(res, rec, thresholds, mode, top_k):
    """The threshold sweep of one image for `mode` from its stored IoU and score (the
    sweep analyse_image stored used the grid and score of the inference run)."""
    c = res["c"]
    near = eb.unpack_near(c)
    h50, h75, nout = (np.zeros(len(thresholds), dtype=np.int64) for _ in range(3))
    for k, thr in enumerate(thresholds):
        kept, _, _ = eb.select(rec["score"], near, thr, mode, top_k)
        out = eb.output_order(kept, c["valid"], mode, rec["idx"])
        m = np.array([v for _, v in eb.assignment(c["iou"], out).values()])
        h50[k], h75[k], nout[k] = int((m > HIT).sum()), int((m > TIGHT).sum()), len(out)
    zero = tuple(np.zeros(len(thresholds), dtype=np.int64) for _ in range(3))
    c["sweep"] = {m: ((h50, h75, nout) if m == mode else zero) for m in eb.MODES}


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def copies_of(iou_out, valid):
    """owner [N]: the GT lane each candidate overlaps most if IoU > NEAR, else -1."""
    n = iou_out.shape[1]
    owner = np.full(n, -1, dtype=np.int64)
    if iou_out.shape[0] == 0:
        return owner
    g = np.argmax(iou_out, axis=0)
    best = iou_out[g, np.arange(n)]
    ok = valid & (best > NEAR)
    owner[ok] = g[ok]
    return owner


def _spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _argmax_first(v):
    return int(np.argmax(v))


def ranking_stats(lanes):
    """Aggregate section 3 for one score from the per-lane ranking rows."""
    multi = [ln for ln in lanes if ln["n_copies"] >= 2]
    out = dict(gt=len(lanes), with_copy=sum(ln["n_copies"] >= 1 for ln in lanes), multi=len(multi))
    if not multi:
        return out
    out["winner_is_best"] = float(np.mean([ln["win_is_best"] for ln in multi]))
    out["within_005"] = float(np.mean([ln["best_iou"] - ln["win_iou"] <= 0.05 for ln in multi]))
    out["iou_lost"] = float(np.mean([ln["best_iou"] - ln["win_iou"] for ln in multi]))
    b50 = [ln for ln in multi if ln["best_iou"] > HIT]
    b75 = [ln for ln in multi if ln["best_iou"] > TIGHT]
    out["lost50"] = (sum(ln["win_iou"] <= HIT for ln in b50), len(b50))
    out["lost75"] = (sum(ln["win_iou"] <= TIGHT for ln in b75), len(b75))
    pairs = {k: [0.0, 0] for k in PAIR_KINDS}
    gaps = {lab: [0.0, 0] for _, _, lab in GAP_BINS}
    for ln in multi:
        for kind, gap, agree in ln["pairs"]:
            pairs[kind][0] += agree
            pairs[kind][1] += 1
            for lo, hi, lab in GAP_BINS:
                if lo <= gap < hi:
                    gaps[lab][0] += agree
                    gaps[lab][1] += 1
    tot = [sum(v[0] for v in pairs.values()), sum(v[1] for v in pairs.values())]
    out["pairs"] = {k: (v[0] / v[1] if v[1] else math.nan, v[1]) for k, v in pairs.items()}
    out["pairs_all"] = (tot[0] / tot[1] if tot[1] else math.nan, tot[1])
    out["gaps"] = {k: (v[0] / v[1] if v[1] else math.nan, v[1]) for k, v in gaps.items()}
    sp = [ln["spearman"] for ln in multi if ln["n_copies"] >= 3 and not math.isnan(ln["spearman"])]
    out["spearman"] = (float(np.mean(sp)) if sp else math.nan, len(sp))
    return out


def lane_rows(res, rec, eb_rows):
    """Per GT lane: hit status (eb_rows: evaluate_image's rows of this image at the
    operating threshold), copies, ranking for each score, ownership."""
    c, step_iou = res["c"], res["step_iou"]
    k_out = rec["out_idx"]
    iou_out = step_iou[k_out].astype(np.float64)
    valid = c["valid"]
    by_gt = {ln["gt"]: ln for ln in eb_rows}
    owner = copies_of(iou_out, valid)
    scores = {"o2o": rec["s_o2o"]}
    if rec["s_o2m"] is not None:
        scores["o2m"] = rec["s_o2m"]
    if rec.get("s_q") is not None:                          # strip_cfg.quality and the joint test score
        scores["quality"] = rec["s_q"]
        scores["o2o_x_quality"] = rec["s_o2o"] * rec["s_q"]
    first = 1 if rec["labels"] and rec["labels"][0] == "proposal" else 0
    rows = []
    for g in range(c["n_gt"]):
        e = by_gt[g]
        cps = np.nonzero(owner == g)[0]
        best_all = np.where(valid, iou_out[g], -1.0)
        f = _argmax_first(best_all) if valid.any() else -1
        row = dict(image=c["filename"], gt=g, slant=c["attrs"][g]["slant"], extent=c["attrs"][g]["extent"],
                   hit50=bool(e["hit50"]), hit75=bool(e["hit75"]), matched_iou=float(e["iou"]),
                   reason=e["reason"],
                   n_copies=len(cps), best_out=f, best_out_iou=float(best_all[f]) if f >= 0 else 0.0)
        for k, lab in enumerate(rec["labels"]):
            row[f"best_iou_{lab}"] = float(step_iou[k][g].max()) if step_iou.shape[2] else 0.0
            row[f"best_idx_{lab}"] = int(np.argmax(step_iou[k][g])) if step_iou.shape[2] else -1
        adx = res["anchor_dx"][g]
        row["nearest_anchor"] = int(np.argmin(adx)) if np.isfinite(adx).any() else -1
        row["nearest_anchor_dx"] = float(adx.min()) if np.isfinite(adx).any() else math.inf
        row["anchor_best_iou"] = float(res["anchor_iou"][g].max()) if res["anchor_iou"].shape[1] else 0.0
        # ownership: when did the output's best copy become the best fit?
        settled = None
        if f >= 0 and row["best_out_iou"] > HIT:
            for k in range(k_out + 1):
                if all(int(np.argmax(step_iou[j][g])) == f for j in range(k, k_out + 1)):
                    settled = k
                    break
        row["settled"] = rec["labels"][settled] if settled is not None else ""
        row["late"] = settled is not None and settled > first
        for name, s in scores.items():
            if len(cps):
                w = int(cps[np.argmax(s[cps])])
                b = int(cps[np.argmax(iou_out[g, cps])])
                row[f"{name}_winner"] = w
                row[f"{name}_win_iou"] = float(iou_out[g, w])
                row[f"{name}_win_is_out_best"] = w == f
                row[f"{name}_win_is_step0_best"] = w == row[f"best_idx_{rec['labels'][first]}"]
                row[f"{name}_win_is_nearest_anchor"] = w == row["nearest_anchor"]
                pairs = []
                for a in range(len(cps)):
                    for bb in range(a + 1, len(cps)):
                        i, j = cps[a], cps[bb]
                        gap = abs(iou_out[g, i] - iou_out[g, j])
                        if gap < PAIR_GAP:
                            continue
                        ds = s[i] - s[j]
                        agree = 0.5 if ds == 0 else float(ds * (iou_out[g, i] - iou_out[g, j]) > 0)
                        nc = int(iou_out[g, i] > HIT) + int(iou_out[g, j] > HIT)
                        pairs.append((PAIR_KINDS[2 - nc], gap, agree))
                row[f"{name}_rank"] = dict(n_copies=len(cps), win_is_best=w == b, best_iou=float(iou_out[g, b]),
                                           win_iou=float(iou_out[g, w]), pairs=pairs,
                                           spearman=_spearman(s[cps], iou_out[g, cps]) if len(cps) >= 3 else math.nan)
        rows.append(row)
    return rows


def ownership_stats(lanes, names, labels):
    own = [ln for ln in lanes if ln["best_out_iou"] > HIT]
    out = dict(n=len(own))
    if not own:
        return out
    labels = [lab for lab in labels if any(ln["settled"] == lab for ln in own)]
    out["settled"] = {lab: sum(ln["settled"] == lab for ln in own) / len(own) for lab in labels}
    late = [ln for ln in own if ln["late"]]
    early = [ln for ln in own if not ln["late"]]
    out["late_share"] = len(late) / len(own)
    out["recall_early"] = (float(np.mean([ln["hit50"] for ln in early])) if early else math.nan, len(early))
    out["recall_late"] = (float(np.mean([ln["hit50"] for ln in late])) if late else math.nan, len(late))
    out["nearest_is_out_best"] = float(np.mean([ln["nearest_anchor"] == ln["best_out"] for ln in own]))
    for name in names:
        w = [ln for ln in own if f"{name}_winner" in ln]
        if not w:
            continue
        out[name] = dict(
            win_is_out_best=float(np.mean([ln[f"{name}_win_is_out_best"] for ln in w])),
            win_is_step0_best=float(np.mean([ln[f"{name}_win_is_step0_best"] for ln in w])),
            win_is_nearest_anchor=float(np.mean([ln[f"{name}_win_is_nearest_anchor"] for ln in w])),
            late_win_is_out_best=float(np.mean([ln[f"{name}_win_is_out_best"] for ln in w if ln["late"]]))
            if any(ln["late"] for ln in w) else math.nan)
    return out


def coverage_stats(results, labels):
    n_gt = sum(r["step_iou"].shape[1] for r in results)
    rows = []
    for k, lab in enumerate(labels):
        best = np.concatenate([r["step_iou"][k].max(axis=1) if r["step_iou"].shape[2] else
                               np.zeros(r["step_iou"].shape[1]) for r in results]) if n_gt else np.zeros(0)
        rows.append(dict(label=lab, c50=float((best > HIT).mean()) if n_gt else math.nan,
                         c75=float((best > TIGHT).mean()) if n_gt else math.nan,
                         mean=float(best.mean()) if n_gt else math.nan))
    a = np.concatenate([r["anchor_iou"].max(axis=1) if r["anchor_iou"].shape[1] else
                        np.zeros(r["anchor_iou"].shape[0]) for r in results]) if n_gt else np.zeros(0)
    rows.insert(0, dict(label="anchors", c50=float((a > HIT).mean()) if n_gt else math.nan,
                        c75=float((a > TIGHT).mean()) if n_gt else math.nan, mean=float(a.mean()) if n_gt else math.nan))
    return dict(n_gt=n_gt, rows=rows)


def _pct(v):
    return "     -" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:6.2f}"


def format_report(cov, b, lost, rank, own, names, out_label, say):
    say("1. coverage by step: GT lanes with some anchor at IoU > 0.5 / > 0.75, mean best IoU")
    for r in cov["rows"]:
        tag = "  <- evaluated" if r["label"] == out_label else ""
        say(f"  {r['label']:>10}  {_pct(r['c50'])}  {_pct(r['c75'])}   {r['mean']:.3f}{tag}")
    say("")
    say(f"2. missed GT lanes that had a correct copy (IoU > 0.5) in the evaluated step: "
        f"{lost[0]} of {lost[1]} misses ({_pct(lost[0] / lost[1] if lost[1] else math.nan).strip()}%); "
        f"the standard breakdown is above")
    say("")
    say("3. copy ranking at the evaluated step (copies: candidates whose most-overlapped GT lane is "
        "this one, IoU > 0.3)")
    for name in names:
        r = rank[name]
        say(f"  {name} score: {r['multi']} GT lanes with >= 2 copies "
            f"(of {r['gt']}; {r['with_copy']} with >= 1)")
        if not r["multi"]:
            continue
        say(f"    winner = best copy          {_pct(r['winner_is_best'])}%   within 0.05 IoU {_pct(r['within_005'])}%")
        say(f"    IoU lost to the ranking     {r['iou_lost']:.3f} (best copy - winner, mean)")
        say(f"    lost@50: best > 0.5, winner not   {r['lost50'][0]} of {r['lost50'][1]}")
        say(f"    lost@75: best > 0.75, winner not  {r['lost75'][0]} of {r['lost75'][1]}")
        say(f"    pairwise accuracy (IoU gap >= {PAIR_GAP:g}): all {_pct(r['pairs_all'][0])}% of {r['pairs_all'][1]}")
        for k in PAIR_KINDS:
            acc, n = r["pairs"][k]
            say(f"      {k:14s} {_pct(acc)}% of {n}")
        for k, (acc, n) in r["gaps"].items():
            say(f"      gap {k:10s} {_pct(acc)}% of {n}")
        sp, n = r["spearman"]
        say(f"    Spearman(score, IoU) over the copies of one lane: {sp:.3f} (mean over {n} lanes with >= 3 copies)")
    say("")
    say(f"4. ownership by step: {own['n']} GT lanes with a correct copy in the evaluated step")
    if own["n"]:
        say("  the evaluated step's best copy has been the best fit since:  " +
            "  ".join(f"{lab} {_pct(v).strip()}%" for lab, v in own["settled"].items()))
        say(f"  late (settled after the first step): {_pct(own['late_share']).strip()}%")
        say(f"  recall@50 at the threshold: settled at the first step {_pct(own['recall_early'][0]).strip()}% "
            f"(n {own['recall_early'][1]}), late {_pct(own['recall_late'][0]).strip()}% (n {own['recall_late'][1]})")
        say(f"  the nearest static anchor is the best copy: {_pct(own['nearest_is_out_best']).strip()}%")
        for name in names:
            if name not in own:
                continue
            o = own[name]
            say(f"  {name} winner is: the best copy {_pct(o['win_is_out_best']).strip()}%, the first step's best "
                f"{_pct(o['win_is_step0_best']).strip()}%, the nearest anchor {_pct(o['win_is_nearest_anchor']).strip()}%; "
                f"for late lanes the best copy {_pct(o['late_win_is_out_best']).strip()}%")


CSV_KEYS = ["image", "gt", "slant", "extent", "hit50", "hit75", "matched_iou", "reason", "n_copies",
            "best_out", "best_out_iou", "settled", "late", "nearest_anchor", "nearest_anchor_dx",
            "anchor_best_iou"]


def write_csv(path, lanes, labels, names):
    keys = CSV_KEYS + [f"best_iou_{l}" for l in labels] + [f"best_idx_{l}" for l in labels]
    for n in names:
        keys += [f"{n}_winner", f"{n}_win_iou", f"{n}_win_is_out_best", f"{n}_win_is_step0_best",
                 f"{n}_win_is_nearest_anchor"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for ln in lanes:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in ln.items() if k in keys})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--mode", choices=["nms", "free"], default=None,
                    help="default: the config's test path (test_cfg.use_nms)")
    ap.add_argument("--thr", type=float, default=None, help="default: best F1@50 of the sweep")
    ap.add_argument("--split", choices=["test", "val"], default="test",
                    help="test = the config's test_dataloader (CurveLanes valid)")
    ap.add_argument("--save-dump", default=None)
    ap.add_argument("--load-dump", default=None, help="reuse a saved pass (CPU)")
    ap.add_argument("--out-dir", default=None, help="default: work_dirs/blrnet_copy_diag/<config name>")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=512, help="images analysed per pool round during inference")
    ap.add_argument("--verify-batches", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=None, help="quick look only")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    from mmengine.config import Config
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    if args.load_dump:
        cfg = Config.fromfile(args.config, import_custom_modules=False)
        with open(args.load_dump, "rb") as f:
            dump = pickle.load(f)
        meta, vinfo, slims, results = dump["meta"], dump["meta"]["verify"], dump["records"], dump["results"]
        decoder = eb.make_decoder(meta)
        checkpoint = meta.get("checkpoint")
        if args.max_images is not None:
            slims, results = slims[:args.max_images], results[:args.max_images]
    else:
        if not args.checkpoint:
            ap.error("a checkpoint is needed unless --load-dump is given")
        cfg = Config.fromfile(args.config)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        records, meta, vinfo = run_inference(cfg, args, device)
        checkpoint = args.checkpoint
        decoder = eb.make_decoder(meta)
    tc = meta["test_cfg"]
    cfg_thr = float(tc["conf_threshold"])
    nms_thres, top_k = float(tc["nms_thres"]), int(tc["nms_topk"])
    cfg_mode = "nms" if bool(tc.get("use_nms", True)) else "free"
    grid = set(np.round(np.arange(0.05, 0.951, 0.05), 2).tolist()) | {0.96, 0.97, 0.98, 0.99, cfg_thr}
    if args.thr is not None:
        grid.add(args.thr)
    thresholds = sorted(grid)
    st = eb.settings(cfg, meta, args)
    state = dict(st=st, decoder=decoder, nms_thres=nms_thres, top_k=top_k, thresholds=thresholds,
                 anchor_rows=meta["anchor_rows"])
    if not args.load_dump:
        tic = time.time()
        results, slims = analyse_records(records, state, jobs, args.chunk)
        print(f"analysed {len(results)} images in {time.time() - tic:.0f} s", flush=True)
        if args.save_dump:
            os.makedirs(osp.dirname(osp.abspath(args.save_dump)), exist_ok=True)
            with open(args.save_dump, "wb") as f:
                pickle.dump(dict(records=slims, results=results, meta=dict(
                    meta, config=args.config, checkpoint=checkpoint, split=args.split, verify=vinfo)),
                    f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved {args.save_dump}")
    if not slims:
        raise SystemExit("no images")

    labels, k_out = slims[0]["labels"], slims[0]["out_idx"]
    source = slims[0]["source"]
    mode = args.mode or cfg_mode
    if args.mode is not None and mode != cfg_mode:
        # the evaluated score follows the mode as forward_test would choose it; the head's top-k
        # cap was applied for the config's mode only, so re-derive the score from o2o / o2m
        source = "o2m" if mode == "nms" and slims[0]["s_o2m"] is not None else "o2o"
        for r in slims:
            s = (r["s_o2m"] if source == "o2m" else r["s_o2o"]).copy()
            if r.get("quality_score"):
                s = np.clip(s * r["s_q"], 1e-6, 1.0 - 1e-6)
            if mode == "free" and 0 < top_k < len(s):
                s[np.argsort(-s, kind="stable")[top_k:]] = 0.0
            r["s_eval"] = r["score"] = s
    for res, r in zip(results, slims):
        resweep(res, r, thresholds, mode, top_k)
    names = ["o2m", "o2o"] if slims[0]["s_o2m"] is not None else ["o2o"]
    if slims[0].get("s_q") is not None:
        names += ["quality", "o2o_x_quality"]

    sweep = eb.sweep_table([res["c"] for res in results], thresholds)
    best = max(sweep[mode], key=lambda r: (r["F1_50"], -r["thr"]))
    thr = args.thr if args.thr is not None else best["thr"]
    chosen = "given" if args.thr is not None else "best F1@50 of the sweep"
    ctx = dict(top_k=top_k, nms_thres=nms_thres, n_strips=meta["n_strips"], img_w=meta["img_w"])

    say("=" * 100)
    say(f"BLRNet copy diagnosis  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"  config: {args.config}")
    say(f"  checkpoint: {checkpoint}")
    say(f"  head: {meta.get('head_cls', '?')}; outputs {', '.join(labels)}; evaluated: {labels[k_out]}; "
        f"{eb.MODE_LABEL[mode]}, score {source}" + (" x strip quality" if slims[0].get("quality_score") else "") + (f" (nms_topk {top_k} cap)" if mode == "free" else
                                                    f" (NMS {nms_thres:g} px, nms_topk {top_k})"))
    say(f"  images {len(slims)}; anchors {len(slims[0]['s_o2o'])}; verify: {json.dumps(vinfo)}")
    say("")
    say(f"threshold sweep {eb.MODE_LABEL[mode]}: P / R / F1@50 / F1@75   (c = config, * = operating)")
    for r in sweep[mode]:
        tag = ("c" if abs(r["thr"] - cfg_thr) < 1e-9 else " ") + ("*" if abs(r["thr"] - thr) < 1e-9 else " ")
        say(f"  {r['thr']:.2f} {tag} {100 * r['P']:8.2f}{100 * r['R']:8.2f}{100 * r['F1_50']:8.2f}{100 * r['F1_75']:8.2f}")
    say("")

    eb_lanes, eb_fps, counts, lanes = [], [], [], []
    for res, rec in zip(results, slims):
        ln, fp, cn = eb.evaluate_image(res["c"], rec, thr, mode, ctx)
        eb_lanes += ln
        eb_fps += fp
        counts.append(cn)
        lanes += lane_rows(res, rec, ln)
    b = eb.breakdown(eb_lanes, eb_fps, counts, mode, thr, eb.gt_attr_specs(nms_thres))
    eb.format_mode(b, cfg_thr, chosen, say, nms_thres)
    say("")
    misses = [ln for ln in lanes if not ln["hit50"]]
    lost = (sum(ln["reason"] in LOST for ln in misses), len(misses))
    cov = coverage_stats(results, labels)
    rank = {n: ranking_stats([dict(ln[f"{n}_rank"]) if f"{n}_rank" in ln else dict(n_copies=0) for ln in lanes])
            for n in names}
    own = ownership_stats(lanes, names, labels)
    say("=" * 100)
    format_report(cov, b, lost, rank, own, names, labels[k_out], say)

    out_dir = args.out_dir or osp.join("work_dirs", "blrnet_copy_diag", osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "copy_diag.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(out_dir, "copy_diag.json"), "w") as f:
        json.dump(eb._json_safe(dict(config=args.config, checkpoint=checkpoint, mode=mode, thr=thr, source=source,
                                     labels=labels, evaluated=labels[k_out], sweep=sweep[mode],
                                     breakdown={k: b[k] for k in ("P", "R", "F1_50", "F1_75", "reasons", "fp")},
                                     lost_with_copy=lost, coverage=cov, ranking=rank, ownership=own)), f, indent=1)
    write_csv(osp.join(out_dir, "copy_lanes.csv"), lanes, labels, names)
    print(f"wrote {out_dir}/copy_diag.txt, copy_diag.json, copy_lanes.csv")
    return dict(results=results, records=slims, lanes=lanes, sweep=sweep, breakdown=b, coverage=cov,
                ranking=rank, ownership=own, lost=lost, thr=thr, mode=mode)


if __name__ == "__main__":
    main()
