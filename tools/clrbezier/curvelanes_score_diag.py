"""Why does the head score correctly localized CurveLanes lanes below the threshold?

Reads a dump of curvelanes_error_breakdown.py (CPU only, no model, no GPU) and
puts three verdicts on the same stored predictions side by side:

  eval      the vega mask IoU, exactly as curvelanes_error_breakdown.py computes it
  training  what the config's last-stage main assigner labels positive among the
            stored predictions, against each GT lane's training target
            (PackCLRNetInputs on the annotation, as repr_ceiling.py builds it), with
            the training cost cache of libs/clrbezier/assigners.py and the config's
            LaneIoU geometry (loss_cfg lane_width, lane_width_cost, iou_eval_shape)
  score     the head's score

Each missed lane with a correct prediction (low_score in the error breakdown) is
called "hedged" when its best correct prediction scored >= 0.05 and "dismissed"
below that. The report tells apart:

  H1 split     several correct copies share the belief: their noisy-OR reaches
               the threshold although none does alone
  H2 evidence  the assigner labels a correct copy positive, yet every copy is low
               and the score ranks poorly in that bin: the label is learnable
               in principle, the classifier cannot see the lane
  H3 label     the training label disagrees with the eval: the lane has no
               training target, or the assigner's positive for it is a copy the
               eval calls wrong while the eval's best copy is labelled negative
  mismatch     a copy that is wrong for every lane (best IoU in (0.3, 0.5])
               outranks every correct copy of the lane: classification-
               localization mismatch in the strict sense

Approximations: the assigner sees the stored predictions only (the top
--max-per-image of the dump per image, inference mode, no augmentation),
while training sees all priors of an augmented image. For heads without
bbox_head.main_assigner (the CLRerNet line head) pass --assigner, or the
training section is left out.

    python tools/clrbezier/curvelanes_score_diag.py CONFIG --load-dump work_dirs/err/o2m.pkl \\
        --mode nms --out-dir work_dirs/err/o2m_score

Outputs in --out-dir: score_diag.txt, score_diag.json, score_lanes.csv.
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
HIT, NEAR = eb.HIT, eb.NEAR
DISMISS = 0.05                    # a missed lane's best correct copy below this is "dismissed"
CLASSES = ["hit", "hedged", "dismissed", "taken", "suppressed", "capped", "mislocalized", "absent"]


# ----------------------------------------------------------------------------
# training geometry and assigner from the config
# ----------------------------------------------------------------------------
def training_setup(cfg, meta, args):
    """The last-stage main assigner and the LaneIoU geometry the head trains with,
    or None when the head has no main assigner and --assigner is not given."""
    head = cfg.model.bbox_head
    loss_cfg = dict(head.get("loss_cfg", {}) or {})
    if args.assigner:
        a, source = json.loads(args.assigner), "--assigner"
    else:
        stage = int(head.get("refine_layers", 3)) - 1
        per = dict(head.get("main_stage_assigners") or {})
        a = per.get(stage) or per.get(str(stage))
        source = f"main_stage_assigners[{stage}]"
        if a is None:
            a, source = head.get("main_assigner"), "main_assigner"
        if a is None:
            return None
    shape = args.iou_shape or loss_cfg.get("iou_eval_shape") or (meta["img_h"], meta["img_w"])
    return dict(
        assigner=dict(a), source=source,
        lane_width=float(args.lane_width if args.lane_width is not None
                         else loss_cfg.get("lane_width", 7.5 / 800)),
        lane_width_cost=float(args.lane_width_cost if args.lane_width_cost is not None
                              else loss_cfg.get("lane_width_cost", 30.0 / 800)),
        iou_h=int(shape[0]), iou_w=int(shape[1]),
        cost_iou_type=str(loss_cfg.get("cost_iou_type", "laneiou")))


def target_tensor(xs_rows, n_rows):
    """PackCLRNetInputs layout [M, 6 + R]; the cost cache reads only the xs (px)."""
    t = torch.full((len(xs_rows), 6 + n_rows), -1e5, dtype=torch.float32)
    t[:, 0], t[:, 1] = 0.0, 1.0
    for j, xs in enumerate(xs_rows):
        t[j, 6:] = torch.as_tensor(np.asarray(xs, dtype=np.float32))
    return t


def training_labels(rec, gt_rows, st, tr):
    """Assigner positives (pred -> GT index or -1), training LaneIoU [G, n]
    (narrow width, the dynamic-k IoU) and the assignment-cost IoU [G, n]."""
    from libs.clrbezier.assigners import build_cost_cache, build_lane_assigner
    G, n = len(gt_rows), len(rec["score"])
    pos = np.full(n, -1, dtype=np.int64)
    dyn = np.full((G, n), np.nan, dtype=np.float32)
    cost_iou = np.full((G, n), np.nan, dtype=np.float32)
    idx = [g for g in range(G) if gt_rows[g] is not None]
    if not idx or n == 0:
        return pos, dyn, cost_iou
    assigner = build_lane_assigner(tr["assigner"])
    required = set(assigner.required_cache_keys) | {"lane_iou_dynamic", "iou_cost_assign"}
    pred = torch.as_tensor(np.asarray(rec["rows"], dtype=np.float32))
    target = target_tensor([gt_rows[g][0] for g in idx], st["n_rows"])
    cache = build_cost_cache(pred, target, st["img_w"], st["img_h"], tr["lane_width"],
                             tr["lane_width_cost"], required, iou_shape=(tr["iou_w"], tr["iou_h"]))
    rows, cols = assigner.assign(cache)
    for r, c in zip(rows.tolist(), cols.tolist()):
        pos[r] = idx[c]
    dyn[idx] = cache["lane_iou_dynamic"].T.numpy()
    cost_iou[idx] = (1.0 - cache["iou_cost_assign"]).T.numpy()
    return pos, dyn, cost_iou


# ----------------------------------------------------------------------------
# one image (worker)
# ----------------------------------------------------------------------------
_S = {}


def _init(state):
    torch.set_num_threads(1)
    eb._init_worker(state["eb"])
    _S.update(state)


def analyse(rec):
    c = eb.analyse_image(rec)
    st, tr = _S["eb"]["st"], _S["train"]
    gt = [ln for ln in rc.read_lines_txt(eb.gt_file(rec["filename"])) if len(ln) > 0]
    _, gt_rows = eb.gt_geometry(gt, tuple(rec["ori_hw"]), int(rec["offset"]), st)
    c["has_target"] = np.array([r is not None for r in gt_rows], dtype=bool)
    if tr is not None:
        c["pos"], c["dyn"], c["cost_iou"] = training_labels(rec, gt_rows, st, tr)
    return c


# ----------------------------------------------------------------------------
# per lane and per prediction at the operating point
# ----------------------------------------------------------------------------
def lane_class(ln):
    if ln["reason"] != "low_score":
        return ln["reason"]
    return "dismissed" if ln["score"] < DISMISS else "hedged"


def lane_stats(c, rec, ln, thr):
    g = ln["gt"]
    score, valid, iou = rec["score"].astype(np.float64), c["valid"], c["iou"]
    best_any = iou.max(0) if iou.shape[0] else np.zeros(len(score))
    corr = np.nonzero(valid & (iou[g] > HIT))[0]
    s = np.sort(score[corr])[::-1]
    wrong = np.nonzero(valid & (iou[g] > NEAR) & (iou[g] <= HIT) & (best_any <= HIT))[0]
    row = dict(cls=lane_class(ln), n_correct=len(corr),
               s1=float(s[0]) if len(s) else math.nan, s2=float(s[1]) if len(s) > 1 else math.nan,
               s_sum=float(s.sum()), noisy_or=float(1.0 - np.prod(1.0 - s)) if len(s) else 0.0,
               mismatch=bool(len(s) and len(wrong) and score[wrong].max() > s[0]),
               has_target=bool(c["has_target"][g]))
    if "pos" in c:
        mine = np.nonzero(c["pos"] == g)[0]
        best = int(corr[np.argmax(iou[g, corr])]) if len(corr) else -1
        row.update(n_pos=len(mine),
                   pos_correct=int((valid[mine] & (iou[g, mine] > HIT)).sum()),
                   pos_eval_iou=float(iou[g, mine].max()) if len(mine) else math.nan,
                   best_is_pos=bool(best >= 0 and c["pos"][best] == g),
                   best_eval_iou=float(iou[g, best]) if best >= 0 else math.nan,
                   best_train_iou=float(c["dyn"][g, best]) if best >= 0 else math.nan,
                   max_train_iou=float(np.nanmax(c["dyn"][g])) if c["has_target"][g] and len(score) else math.nan)
        # the training label points at a copy the eval calls wrong, the eval's best is negative
        row["label_off"] = bool(best >= 0 and not row["best_is_pos"] and len(mine)
                                and row["pos_eval_iou"] <= HIT)
    return row


def pred_labels(c, rec):
    """Per stored prediction: score, eval-correct, assigner positive, training IoU > 0.5,
    and the eval-best GT lane (for binning) or -1."""
    n = len(rec["score"])
    iou = c["iou"]
    if iou.shape[0]:
        best_g = np.argmax(iou, axis=0)
        best = iou[best_g, np.arange(n)]
    else:
        best_g, best = np.full(n, -1), np.zeros(n)
    ev = c["valid"] & (best > HIT)
    out = dict(score=rec["score"].astype(np.float64), eval=ev, gt=np.where(ev, best_g, -1))
    if "pos" in c:
        dyn = np.nan_to_num(c["dyn"], nan=0.0)
        out["asg"] = c["pos"] >= 0
        out["train"] = (dyn.max(0) > HIT) if dyn.shape[0] else np.zeros(n, bool)
    return out


def auc(score, label):
    """P(score of a positive > score of a negative), ties counted half."""
    from scipy.stats import rankdata
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def _med(v):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    return float(np.median(v)) if len(v) else math.nan


def _share(v):
    v = list(v)
    return float(np.mean(v)) if v else math.nan


def _f(v, fmt="{:.2f}"):
    return "   -" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


BINS = [
    ("slant", "slant", [1, 2, 4], ["<1", "1-2", "2-4", ">4"]),
    ("extent", "vertical extent", [0.2, 0.4, 0.7], ["<0.2", "0.2-0.4", "0.4-0.7", ">=0.7"]),
    ("n_points", "annotated points", [2.5, 5.5, 8.5], ["<=2", "3-5", "6-8", ">8"]),
]


def report(lanes, preds, thr, mode, train, say):
    G = len(lanes)
    by = {k: [ln for ln in lanes if ln["cls"] == k] for k in CLASSES}
    say(f"1. GT lanes by outcome ({eb.MODE_LABEL[mode]}, threshold {thr:.2f}; hedged / dismissed = "
        f"low_score with the best correct copy >= / < {DISMISS})")
    say("   correct copy = a stored prediction with eval IoU > 0.5 with this lane; wrong copy = best IoU with "
        "every lane in (0.3, 0.5]")
    say(f"   {'':13s}{'lanes':>7}{'% GT':>7}{'copies':>8}{'>=2':>7}{'top':>7}{'2nd':>7}{'sum':>7}"
        f"{'OR>=thr':>9}{'wrong on top':>14}")
    tab1 = {}
    for k in CLASSES:
        L = by[k]
        if not L:
            continue
        has = [ln for ln in L if ln["n_correct"] > 0]
        t = dict(n=len(L), pct=100 * len(L) / G, copies=_med(ln["n_correct"] for ln in has),
                 two=_share(ln["n_correct"] >= 2 for ln in has), s1=_med(ln["s1"] for ln in has),
                 s2=_med(ln["s2"] for ln in has), ssum=_med(ln["s_sum"] for ln in has),
                 or_thr=_share(ln["noisy_or"] >= thr for ln in has), mismatch=_share(ln["mismatch"] for ln in has))
        tab1[k] = t
        if not has:
            say(f"   {k:13s}{t['n']:7d}{t['pct']:7.1f}   (no correct copy)")
            continue
        say(f"   {k:13s}{t['n']:7d}{t['pct']:7.1f}{_f(t['copies'], '{:8.0f}')}{100 * t['two']:6.0f}%"
            f"{_f(t['s1'], '{:7.2f}')}{_f(t['s2'], '{:7.2f}')}{_f(t['ssum'], '{:7.2f}')}"
            f"{100 * t['or_thr']:8.0f}%{100 * t['mismatch']:13.0f}%")
    say("   copies / top / 2nd / sum: medians over the lanes; >=2, OR>=thr, wrong on top: share of the lanes")
    say("")
    tab2 = {}
    if train is not None:
        say(f"2. what the training assigner says about the same lanes ({train['assigner']['type']}, "
            f"from {train['source']})")
        say(f"   {'':13s}{'no target':>11}{'positives':>11}{'pos correct':>13}{'eval-best pos':>15}"
            f"{'pos eval IoU':>14}{'label off':>11}{'best: eval/train IoU':>22}")
        for k in CLASSES:
            L = by[k]
            if not L:
                continue
            has = [ln for ln in L if ln["n_correct"] > 0]
            tgt = [ln for ln in L if ln["has_target"]]
            t = dict(no_target=1 - _share(ln["has_target"] for ln in L),
                     n_pos=_med(ln["n_pos"] for ln in tgt),
                     pos_correct=_share(ln["pos_correct"] > 0 for ln in tgt),
                     best_is_pos=_share(ln["best_is_pos"] for ln in has),
                     pos_eval_iou=_med(ln["pos_eval_iou"] for ln in tgt),
                     label_off=_share(ln["label_off"] for ln in has),
                     best_eval=_med(ln["best_eval_iou"] for ln in has),
                     best_train=_med(ln["best_train_iou"] for ln in has if ln["has_target"]))
            tab2[k] = t
            pct = lambda v: "     -" if math.isnan(v) else f"{100 * v:5.0f}%"
            say(f"   {k:13s}{pct(t['no_target']):>11}{_f(t['n_pos'], '{:11.0f}')}{pct(t['pos_correct']):>13}"
                f"{pct(t['best_is_pos']):>15}{_f(t['pos_eval_iou'], '{:14.2f}')}{pct(t['label_off']):>11}"
                f"{_f(t['best_eval'], '{:13.2f}')} / {_f(t['best_train'])}")
        say("   no target: the training pipeline builds no target for the lane (never a positive in training)")
        say("   positives: stored predictions the assigner gives the lane (median); pos correct: share of lanes "
            "where one of them has eval IoU > 0.5")
        say("   eval-best pos: the eval's best copy is a positive; label off: the lane's positives all have eval "
            "IoU <= 0.5 while a correct copy is negative")
        say("   best: eval IoU / training LaneIoU of the eval's best copy (medians)")
        say("")
    say("3. how well the score ranks the stored predictions (AUC; 0.5 = chance)")
    labels = [("eval", "eval IoU > 0.5")]
    if train is not None:
        labels += [("asg", "assigner positive"), ("train", "training IoU > 0.5")]
    sc = np.concatenate([p["score"] for p in preds]) if preds else np.zeros(0)
    lab = {k: np.concatenate([p[k] for p in preds]) for k, _ in labels} if preds else {}
    tab3 = {"all": {k: auc(sc, lab[k]) for k, _ in labels}}
    say(f"   {'':22s}" + "".join(f"{name:>22s}" for _, name in labels))
    say(f"   {'all predictions':22s}" + "".join(f"{tab3['all'][k]:22.3f}" for k, _ in labels))
    # per bin: positives = eval-correct predictions of the bin's lanes, negatives = predictions
    # correct for no lane (the same for every bin)
    gt_attr = np.concatenate([p["gt_attr"] for p in preds]) if preds else np.zeros((0, 3))
    neg = ~lab["eval"] if preds else np.zeros(0, bool)
    for col, (key, title, inner, names) in enumerate(BINS):
        vals = gt_attr[:, col]
        which = np.where(np.isnan(vals), -1, np.digitize(np.nan_to_num(vals, nan=0.0), inner))
        say(f"   eval-correct vs correct for no lane, by the lane's {title}:")
        tab3[key] = {}
        for b, name in enumerate(names):
            inbin = lab["eval"] & (which == b)
            m = inbin | neg
            v = auc(sc[m], inbin[m])
            tab3[key][name] = dict(auc=v, n=int(inbin.sum()))
            say(f"     {name:>8s} ({int(inbin.sum()):7d} correct predictions)  AUC {_f(v, '{:.3f}')}")
    say("")
    return dict(lanes=tab1, training=tab2, auc=tab3)


def reading(say, train):
    say("4. reading (hedged and dismissed rows against the hit row)")
    if train is None:
        say("   (H2 and H3 need the training labels, left out here: pass --assigner)")
    say("   H1 split     hedged lanes have >= 2 copies and OR>=thr is high, hits have one dominant copy: the")
    say("                belief is shared among look-alikes. Points to fewer, interacting proposals or soft")
    say("                (IoU-quality) classification targets (loss_cfg.cls_target_mode = 'iou').")
    say("   H3 label     'no target' or 'label off' is high for hedged / dismissed lanes and low for hits, or the")
    say("                AUC for 'assigner positive' is well above the AUC for 'eval IoU > 0.5': the classifier")
    say("                learned its training label, and the label is not what the eval counts.")
    say("   H2 evidence  'eval-best pos' is high (the label says positive) but every copy is low, and the AUC")
    say("                drops in the short / flat / sparse bins: the classifier cannot see these lanes.")
    say("   mismatch     'wrong on top' is high: a worse copy outranks the correct one (strict cls-loc mismatch).")
    say("   Dismissed lanes the label calls positive with ordinary shapes point at the data (annotation")
    say("   policy) as much as at the head; compare with the line baseline's run of this tool.")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else
                            "" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--load-dump", required=True, help="dump of curvelanes_error_breakdown.py --save-dump")
    ap.add_argument("--mode", choices=["nms", "free"], default="nms")
    ap.add_argument("--thr", type=float, default=None, help="default: best F1@50 of the sweep")
    ap.add_argument("--assigner", default=None,
                    help='JSON, e.g. \'{"type": "SimOTALaneAssigner", "candidate_topk": 4}\'; '
                         "default: the config's last-stage main assigner")
    ap.add_argument("--iou-shape", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="LaneIoU geometry; default loss_cfg.iou_eval_shape")
    ap.add_argument("--lane-width", type=float, default=None, help="default loss_cfg.lane_width")
    ap.add_argument("--lane-width-cost", type=float, default=None, help="default loss_cfg.lane_width_cost")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=None, help="quick look only")
    args = ap.parse_args(argv)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    from mmengine.config import Config
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    with open(args.load_dump, "rb") as f:
        dump = pickle.load(f)
    records, meta = dump["records"], dump["meta"]
    if args.max_images is not None:
        records = records[:args.max_images]
    st = eb.settings(cfg, meta, args)
    tc = meta["test_cfg"]
    cfg_thr, nms_thres, top_k = float(tc["conf_threshold"]), float(tc["nms_thres"]), int(tc["nms_topk"])
    floor = float(meta["floor"])
    grid = set(np.round(np.arange(0.05, 0.951, 0.05), 2).tolist()) | {0.96, 0.97, 0.98, 0.99, cfg_thr}
    if args.thr is not None:
        grid.add(args.thr)
    thresholds = sorted(t for t in grid if t >= floor - 1e-12)
    train = training_setup(cfg, meta, args)
    state = dict(eb=dict(st=st, decoder=eb.make_decoder(meta), nms_thres=nms_thres, top_k=top_k,
                         thresholds=thresholds), train=train)
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    tic = time.time()
    if jobs == 1:
        _init(state)
        results = [analyse(r) for r in records]
    else:
        with mp.get_context("fork").Pool(jobs, initializer=_init, initargs=(state,)) as pool:
            results = list(pool.imap(analyse, records, chunksize=32))
    print(f"analysed {len(results)} images in {time.time() - tic:.0f} s", flush=True)

    sweep = eb.sweep_table(results, thresholds)
    if args.thr is not None:
        thr, how = args.thr, "given"
    else:
        best = max(sweep[args.mode], key=lambda r: (r["F1_50"], -r["thr"]))
        thr, how = best["thr"], "best F1@50 of the sweep"
    k = [i for i, t in enumerate(thresholds) if abs(t - thr) < 1e-9][0]
    sw = sweep[args.mode][k]
    ctx = dict(top_k=top_k, nms_thres=nms_thres, n_strips=meta["n_strips"], img_w=meta["img_w"])
    lanes, preds = [], []
    for c, rec in zip(results, records):
        ln_rows, _, _ = eb.evaluate_image(c, rec, thr, args.mode, ctx)
        for ln in ln_rows:
            row = dict(image=ln["image"], gt=ln["gt"], reason=ln["reason"], score=ln.get("score", math.nan),
                       slant=ln["slant"], extent=ln["extent"], n_points=ln["n_points"], nn_dist=ln["nn_dist"],
                       lanes=ln["lanes"])
            row.update(lane_stats(c, rec, dict(ln, score=ln.get("score", math.nan)), thr))
            lanes.append(row)
        p = pred_labels(c, rec)
        attrs = c["attrs"]
        p["gt_attr"] = np.array([[attrs[g]["slant"], attrs[g]["extent"], attrs[g]["n_points"]] if g >= 0
                                 else [math.nan] * 3 for g in p["gt"]], dtype=np.float64).reshape(-1, 3)
        preds.append(p)

    say("=" * 100)
    say(f"CurveLanes score diagnosis  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"  config: {args.config}")
    say(f"  dump: {args.load_dump} (checkpoint {meta.get('checkpoint')}); images {len(records)}; "
        f"stored predictions per image: at most {meta.get('max_per_image')}, score >= {floor:g}")
    say(f"  operating point: {eb.MODE_LABEL[args.mode]}, threshold {thr:.2f} ({how}): "
        f"P {100 * sw['P']:.2f}  R {100 * sw['R']:.2f}  F1@50 {100 * sw['F1_50']:.2f}")
    if train is not None:
        say(f"  training labels: {json.dumps(train['assigner'])} ({train['source']}); LaneIoU geometry "
            f"{train['iou_h']}x{train['iou_w']}, lane_width {train['lane_width']:.5f}, "
            f"lane_width_cost {train['lane_width_cost']:.5f}")
        if train["cost_iou_type"] != "laneiou":
            say(f"  NOTE: the config assigns with cost_iou_type={train['cost_iou_type']!r}; this report uses LaneIoU")
        say("  the assigner sees the stored predictions only (inference mode, no augmentation): an approximation")
    else:
        say("  training labels: left out (no bbox_head.main_assigner; pass --assigner)")
    say("")
    tabs = report(lanes, preds, thr, args.mode, train, say)
    reading(say, train)

    out_dir = args.out_dir or osp.join("work_dirs", "curvelanes_score_diag",
                                       osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "score_diag.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(out_dir, "score_diag.json"), "w") as f:
        json.dump(eb._json_safe(dict(config=args.config, dump=args.load_dump, mode=args.mode, thr=thr,
                                     train=train, **tabs)), f, indent=1)
    write_csv(osp.join(out_dir, "score_lanes.csv"), lanes)
    print(f"wrote {out_dir}/score_diag.txt, score_diag.json, score_lanes.csv")
    return dict(results=results, lanes=lanes, preds=preds, tabs=tabs, thr=thr, sweep=sweep)


if __name__ == "__main__":
    main()
