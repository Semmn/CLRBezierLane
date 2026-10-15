"""Where does a CurveLanes prediction get its score, and does the score judge the lane it outputs?

One inference pass (GPU) stores what curvelanes_error_breakdown.py stores (the 64
highest-scored last-stage predictions per image, low scores included) and, for the
same predictions, what every refine stage did to them:

  stage rows   each stage's output lane (score, start, length, 72 row xs)
  residual     the dense row offsets each stage adds to its curve (the last
               n_offsets outputs of reg_layers). curve = output - residual; the
               curve of stage s is the lane stage s+1 pools its RoI from
  anchor       the lane stage 0 pools from (the prior)
  features     the last stage's classifier input (after the cls tower) and the
               cls tower input (the stage's RoI feature)
  rescore      a second pass of the head in which the last stage pools its RoI on
               the curve it outputs instead of the curve it was given: the score
               the head gives when it looks at its own output (stages before the
               last are unchanged; checked)
  fresh        a third pass in which the last stage also re-pools every earlier
               stage's RoI slot on that curve (each on its own feature level), so
               the classifier sees no trace of the anchor or the earlier lanes:
               the score the lane would get had it been there from the start

On the first batches it checks that the samples each stage pooled at are the
previous stage's curve (the anchor for stage 0), as the trace assumes.
Works for CLRBezierHead and for the official CLRerHead (line anchors).

The analysis (CPU) rebuilds the evaluation exactly as curvelanes_error_breakdown.py
does (vega mask IoU, NMS kernel, Hungarian) for every geometry of the trace and
reports:

  1. the cascade stage by stage: GT lanes with a correct copy in each geometry
     (anchor, each stage's output, the last stage's input, the curve without the
     residual) and F1 for each (geometry, score) pair: each stage alone, the final
     lanes with the previous stage's score, the residual zeroed, the rescore;
  2. input or output: is the last-stage score about the lane it was given or the
     lane it outputs (2x2 of input correct x output correct, AUC per stage);
  3. missed lanes through the cascade: for each outcome of the error breakdown
     (hit, hedged, dismissed, mislocalized, ...), the best-scored correct copy's
     input IoU, how far the last stage moved it, its score at every stage and
     its rescore; the wrong copy that outranks it (mismatch);
  4. the rescore and the fresh score: which missed lanes they recover and which
     hits they lose;
  4b. early ownership: lanes by the stage at which their best-scored correct copy
     first became correct; their scores per stage, rescore and fresh score, and
     whether the training assigner gives the eval's best copy the positive, with
     and without the score in the matching cost (cls_weight 0);
  5. curve vs residual: hits that need the residual, residual size per outcome;
  6. what the RoI samples see: share of the 36 samples inside the GT lane's
     annotated rows and within one feature cell of it, per outcome and lane shape;
  7. feature probes (cross-fitted by image halves): the best linear readout of
     the classifier's own input feature, trained on the eval label (IoU > 0.5)
     or on the training assigner's label, an MLP on the RoI feature (the cls
     tower retrained), and the linear probe with geometry descriptors; AUC by
     lane shape and F1. Head ~ probe: the feature lacks the information;
     eval probe >> assigner probe ~ head: the training label is the problem;
  8. score calibration (share correct per score bin);
  9. anchors: correct copies and their scores by where the anchor starts.

    # one inference pass (GPU); the dump also works with the error breakdown and score tools
    python tools/clrbezier/curvelanes_stage_trace.py CONFIG CKPT \\
        --save-dump work_dirs/trace/RUN.pkl --out-dir work_dirs/trace/RUN
    # analysis only, from the dump (CPU)
    python tools/clrbezier/curvelanes_stage_trace.py CONFIG --load-dump work_dirs/trace/RUN.pkl \\
        --mode nms --out-dir work_dirs/trace/RUN
    # training images (does the head fit its own training set?), a random subset
    python tools/clrbezier/curvelanes_stage_trace.py CONFIG CKPT --split train --subset 5000 \\
        --save-dump work_dirs/trace/RUN_train.pkl --out-dir work_dirs/trace/RUN_train
    # side by side
    python tools/clrbezier/curvelanes_stage_trace.py --compare A/trace.json B/trace.json

Outputs in --out-dir: trace.txt, trace.json (for --compare), trace_lanes.csv (one
row per GT lane), trace_preds.npz (one entry per stored prediction, for slicing).
"""
from __future__ import annotations

import argparse
import copy
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


def _sibling(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, osp.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eb = _sibling("curvelanes_error_breakdown")      # also puts the repo root on sys.path
sd = _sibling("curvelanes_score_diag")
rc = eb.rc
HIT, NEAR, TIGHT = eb.HIT, eb.NEAR, eb.TIGHT
MODES = eb.MODES


# ============================================================================
# capture (inference)
# ============================================================================
class StageTrace:
    """Hooks on one head that record, per forward pass and per stage, the reg_layers
    output, the RoI sample xs the stage pooled at, the classifier input and the cls
    tower input; plus the stage-0 input lane. With ``override`` set, the last stage
    pools at those sample xs instead (the rescore pass)."""

    def __init__(self, head):
        self.head = head
        self.L = int(head.refine_layers)
        self.R = int(len(head.prior_ys))
        self.kind = "bezier" if any(c.__name__ == "CLRBezierHead" for c in type(head).__mro__) else "clr"
        self._check_supported()
        self._undo = []
        self.override = None
        self.fresh = None
        self._pool0 = None
        self.reset()

    def _check_supported(self):
        h = self.head
        bad = [name for name in ("diffusion", "slot_experts", "curv_moe")
               if getattr(h, name, None) is not None]
        if bad:
            raise SystemExit(f"stage trace: heads with {', '.join(bad)} are not supported "
                             "(the classifier or the RoI path is not one pass per stage)")
        if getattr(h, "reproject", None) is not None:
            raise SystemExit("stage trace: reproject_cfg moves the next stage's input away from the "
                             "stage's curve; not supported")
        if getattr(h, "roi_gather_deformable", False):
            raise SystemExit("stage trace: the deformable RoI gather reads the geometry outside "
                             "pool_prior_features; not supported")
        if getattr(h, "lateral", None) is not None:
            raise SystemExit("stage trace: lateral modules pool on their own; not supported")

    def reset(self):
        self.reg, self.samples, self.cls_in, self.tower_in = [], [], [], []
        self.pool_calls = []                     # per stage: (feature map, extra args, kwargs) of its pooling
        self.fresh_applied, self.fresh_in = 0, None
        self.anchor = None

    # ---- hooks -------------------------------------------------------------
    def _pool_xs(self, prior_xs):
        if self.override is not None and len(self.samples) == self.L - 1:
            prior_xs = self.override.to(device=prior_xs.device, dtype=prior_xs.dtype)
        self.samples.append(prior_xs.detach())
        return prior_xs

    def install(self):
        h = self.head
        self._undo.append(h.reg_layers.register_forward_hook(
            lambda m, i, o: self.reg.append(o.detach())).remove)
        self._undo.append(h.cls_layers.register_forward_pre_hook(
            lambda m, i: self.cls_in.append(i[0].detach().clone())).remove)
        first = h.cls_modules[0] if len(h.cls_modules) else h.cls_layers
        self._undo.append(first.register_forward_pre_hook(
            lambda m, i: self.tower_in.append(i[0].detach().clone())).remove)
        if self.kind == "bezier":
            mod = sys.modules[type(h).__module__]
            pool0, ref0 = mod.pool_prior_features, mod.brr_reference

            def pool(feature_map, prior_xs, *a, **k):
                self.pool_calls.append((feature_map, a, k))
                return pool0(feature_map, self._pool_xs(prior_xs), *a, **k)

            def reference(*a, **k):
                out = ref0(*a, **k)
                if self.anchor is None:          # _refine decodes the anchor state first
                    self.anchor = out[0].detach()
                return out

            mod.pool_prior_features, mod.brr_reference = pool, reference
            self._undo.append(lambda: (setattr(mod, "pool_prior_features", pool0),
                                       setattr(mod, "brr_reference", ref0)))
        else:
            pool0 = h.pool_prior_features
            gen = h.anchor_generator
            gen0 = gen.generate_anchors

            def pool(batch_features, prior_xs, *a, **k):
                self.pool_calls.append((batch_features, a, k))
                return pool0(batch_features, self._pool_xs(prior_xs), *a, **k)

            def generate(params, *a, **k):
                out = gen0(params, *a, **k)
                if self.anchor is None:          # forward generates the priors first
                    p = params.detach().float()
                    xs = out[0].detach().float()
                    rows = xs.new_zeros(p.shape[0], 6 + xs.shape[-1])
                    rows[:, 2:5] = p
                    rows[:, 5] = 1.0             # the line runs to the top of the image
                    rows[:, 6:] = xs
                    self.anchor = rows
                return out

            h.pool_prior_features = pool
            gen.generate_anchors = generate
            self._undo.append(lambda: (h.__dict__.pop("pool_prior_features", None),
                                       gen.__dict__.pop("generate_anchors", None)))
        self._pool0 = pool0
        # the fresh pass: the last stage's RoI gather receives every earlier stage's slot
        # re-pooled at self.fresh[s] (on that stage's own feature map)
        gather = getattr(h, "roi_gather" if self.kind == "bezier" else "attention", None)
        if not isinstance(gather, torch.nn.Module):
            raise SystemExit("stage trace: the head has no RoI gather module (roi_gather / attention)")

        def gather_pre(m, args):
            if self.fresh is None or len(self.samples) != self.L:
                return None
            slots = list(args[0])
            for s in range(self.L - 1):
                fm, a, k = self.pool_calls[s]
                x = self.fresh[s].to(device=fm.device, dtype=fm.dtype)
                slots[s] = self._pool0(fm, x, *a, **k)
            self.fresh_applied += 1
            self.fresh_in = [f.detach() for f in slots]
            return (slots,) + tuple(args[1:])

        self._undo.append(gather.register_forward_pre_hook(gather_pre).remove)
        return self

    def remove(self):
        while self._undo:
            self._undo.pop()()

    # ---- one forward pass --------------------------------------------------
    def run(self, feats, override=None, fresh=None):
        """Head forward with the hooks; returns (per-stage official dicts, capture).
        override: sample xs the last stage pools at. fresh: one sample-xs tensor per
        earlier stage; the last stage's RoI gather gets those slots re-pooled there."""
        self.reset()
        self.override, self.fresh = override, fresh
        try:
            preds = self.head(feats)
        finally:
            self.override, self.fresh = None, None
        L = self.L
        if fresh is not None and self.fresh_applied != 1:
            raise RuntimeError(f"stage trace: the fresh slots were applied {self.fresh_applied} times (expected 1)")
        if not (len(self.reg) == len(self.samples) == len(self.cls_in) == len(self.tower_in) == L
                and len(preds) == L):
            raise RuntimeError(f"stage trace: expected one call per stage ({L}); got reg {len(self.reg)}, "
                               f"pool {len(self.samples)}, cls {len(self.cls_in)}, tower {len(self.tower_in)}, "
                               f"stages {len(preds)}")
        B, Q = preds[-1]["cls_logits"].shape[:2]
        cap = dict(residual=[r.view(B, Q, -1)[..., -self.R:].float() for r in self.reg],
                   samples=[s.reshape(B, Q, -1).float() for s in self.samples],
                   cls_feat=[f.reshape(B, Q, -1) for f in self.cls_in],
                   roi=[f.reshape(B, Q, -1) for f in self.tower_in])
        anchor = self.anchor.float()
        cap["anchor"] = anchor if anchor.dim() == 3 else anchor[None].expand(B, -1, -1)
        return preds, cap


def sample_view(xs, idx, flip, clamp):
    """The RoI sample xs a stage pools at for a lane whose 72 row xs are ``xs``."""
    s = xs[..., idx]
    if clamp:
        s = torch.nan_to_num(s, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return torch.flip(s, dims=[-1]) if flip else s


def stage_inputs(rows, cap):
    """Per stage s, the 72 row xs of the lane stage s pooled from: the anchor for s = 0,
    else the previous stage's output minus its residual."""
    out = [cap["anchor"][..., 6:]]
    for s in range(len(rows) - 1):
        out.append(rows[s][..., 6:] - cap["residual"][s])
    return out


def detect_view(head, rows, cap):
    """(flip, clamp) that turn each stage's input lane into the samples it pooled at,
    and the largest error over stages."""
    idx = head.sample_x_indices.to(rows[0].device).long()
    ins = stage_inputs(rows, cap)
    best = None
    for flip in (True, False):
        for clamp in (True, False):
            err = max(float((sample_view(x, idx, flip, clamp).clamp(-2, 3)
                             - cap["samples"][s].clamp(-2, 3)).abs().max()) for s, x in enumerate(ins))
            if best is None or err < best[2]:
                best = (flip, clamp, err)
    return best


@torch.no_grad()
def run_trace(cfg, args, device):
    from mmengine.runner import Runner
    from torch.utils.data import DataLoader, Subset
    model = eb.build_model(cfg, args.checkpoint, device)
    head = model.bbox_head
    meta = eb.head_meta(head)
    decoder = eb.Decoder(eb.resolve(meta["decode_cls"]).predictions_to_lanes, meta["prior_ys"],
                         meta["n_strips"], meta["test_cfg"])
    top_k = int(head.test_cfg["nms_topk"])
    if args.split == "train":
        dl = copy.deepcopy(cfg.val_dataloader)
        tr = cfg.train_dataloader.dataset
        dl.dataset.data_root, dl.dataset.data_list = tr.data_root, tr.data_list
        dl.dataset.test_mode = True
        dl.dataset.pop("diff_thr", None)
    else:
        dl = copy.deepcopy(cfg.test_dataloader if args.split == "test" else cfg.val_dataloader)
    loader = Runner.build_dataloader(dl)
    if args.subset:
        n_all = len(loader.dataset)
        pick = np.sort(np.random.default_rng(args.seed).choice(n_all, min(args.subset, n_all), replace=False))
        loader = DataLoader(Subset(loader.dataset, pick.tolist()), batch_size=loader.batch_size,
                            num_workers=loader.num_workers, collate_fn=loader.collate_fn, shuffle=False)
    tracer = StageTrace(head).install()
    L, R = tracer.L, tracer.R
    idx_t = head.sample_x_indices.long()
    records, checked, bad, view = [], 0, 0, None
    checks = dict(sample_err=0.0, rescore_stage_err=0.0, rescore_sample_err=0.0, fresh_stage_err=0.0)
    tic = time.time()
    try:
        for bi, data in enumerate(loader):
            data = model.data_preprocessor(data, False)
            feats = model.extract_feat(data["inputs"])
            preds, cap = tracer.run(feats)
            rows = [eb.pred_rows(p)[0].float() for p in preds]
            scores = [eb.pred_rows(p)[1].float() for p in preds]
            if view is None:
                flip, clamp, err = detect_view(head, rows, cap)
                view = dict(flip=flip, clamp=clamp)
                fmap = [tuple(f.shape[-2:]) for f in head._select_features(feats)] \
                    if hasattr(head, "_select_features") else \
                    [tuple(f.shape[-2:]) for f in list(feats[len(feats) - L:])[::-1]]
                strides = [float(meta["img_w"]) / w for _, w in fmap]
            idx = idx_t.to(rows[0].device)
            ins = stage_inputs(rows, cap)
            # the rescore pass: the last stage pools on the curve it outputs
            final_curve = rows[-1][..., 6:] - cap["residual"][-1]
            over = sample_view(final_curve, idx, view["flip"], view["clamp"])
            preds2, cap2 = tracer.run(feats, override=over)
            rows2 = [eb.pred_rows(p)[0].float() for p in preds2]
            rescore = eb.pred_rows(preds2[-1])[1].float()
            # the fresh pass: every RoI slot of the last stage pooled on that curve
            preds3, cap3 = tracer.run(feats, override=over, fresh=[over] * (L - 1))
            rows3 = [eb.pred_rows(p)[0].float() for p in preds3]
            fresh = eb.pred_rows(preds3[-1])[1].float()
            if bi < args.verify_batches:
                for s, x in enumerate(ins):
                    e = float((sample_view(x, idx, view["flip"], view["clamp"]).clamp(-2, 3)
                               - cap["samples"][s].clamp(-2, 3)).abs().max())
                    checks["sample_err"] = max(checks["sample_err"], e)
                checks["rescore_stage_err"] = max(checks["rescore_stage_err"], max(
                    [float((rows2[s] - rows[s]).abs().max()) for s in range(L - 1)], default=0.0))
                checks["rescore_sample_err"] = max(checks["rescore_sample_err"],
                                                   float((cap2["samples"][-1] - over).abs().max()))
                checks["fresh_stage_err"] = max(checks["fresh_stage_err"], max(
                    [float((rows3[s] - rows[s]).abs().max()) for s in range(L - 1)], default=0.0))
            recs = eb.image_records(preds[-1], data["data_samples"], args.floor, args.max_per_image)
            if bi < args.verify_batches:
                c, m = eb.verify(head, preds[-1], recs, decoder, meta, top_k)
                checked, bad = checked + c, bad + m
            for b, rec in enumerate(recs):
                o = torch.as_tensor(rec["idx"], device=rows[0].device)
                rec["stage_rows"] = torch.stack([rows[s][b, o] for s in range(L - 1)]).cpu().numpy() \
                    if L > 1 else np.zeros((0, len(o), 6 + R), np.float32)
                rec["anchor_rows"] = cap["anchor"][b, o].cpu().numpy().astype(np.float32)
                rec["residual"] = torch.stack([cap["residual"][s][b, o] for s in range(L)]).cpu().numpy() \
                    .astype(np.float16)
                rec["stage_score"] = torch.stack([scores[s][b, o] for s in range(L)]).cpu().numpy()
                rec["rescore"] = rescore[b, o].cpu().numpy()
                rec["feat"] = cap["cls_feat"][-1][b, o].float().cpu().numpy().astype(np.float16)
                rec["feat_rescore"] = cap2["cls_feat"][-1][b, o].float().cpu().numpy().astype(np.float16)
                rec["fresh"] = fresh[b, o].cpu().numpy()
                rec["feat_fresh"] = cap3["cls_feat"][-1][b, o].float().cpu().numpy().astype(np.float16)
                rec["roi"] = cap["roi"][-1][b, o].float().cpu().numpy().astype(np.float16)
            records += recs
            if len(records) % 2000 < len(recs):
                print(f"  {len(records)} images ({time.time() - tic:.0f} s)", flush=True)
            if args.max_images is not None and len(records) >= args.max_images:
                break
    finally:
        tracer.remove()
    print(f"verified {checked} image-path pairs against get_lanes (with and without NMS): {bad} mismatches; "
          f"stage inputs == pooled samples to {checks['sample_err']:.2e}; rescore pass: earlier stages "
          f"unchanged to {checks['rescore_stage_err']:.2e}, last stage pooled at the output curve to "
          f"{checks['rescore_sample_err']:.2e}; fresh pass: earlier stages unchanged to "
          f"{checks['fresh_stage_err']:.2e}", flush=True)
    meta["trace"] = dict(kind=tracer.kind, refine_layers=L, n_rows=R,
                         sample_x_indices=idx_t.cpu().numpy().astype(np.int64),
                         flip=bool(view["flip"]) if view else True,
                         clamp=bool(view["clamp"]) if view else True,
                         strides=strides if view else None, checks=checks, fresh=True,
                         dedup=bool(getattr(head, "dedup", None) is not None
                                    and getattr(head, "dedup_use_at_test", False)),
                         decoder=getattr(head, "decoder", None) is not None,
                         num_fc=sum(isinstance(m, torch.nn.Linear) for m in head.cls_modules),
                         split=args.split, subset=args.subset, seed=args.seed)
    return records, meta, dict(checked=checked, mismatches=bad)


# ============================================================================
# analysis, one image (worker)
# ============================================================================
_T = {}
_JOB = {}


def _init(state):
    sd._init(state["sd"])
    _T.update(state["trace"])


def trace_geometries(rec, L):
    """name -> rows [n, 6 + R] (official layout) for each geometry of the trace other
    than the final lanes: the anchor, the earlier stages' outputs, the last stage's
    input lane (the previous stage's curve over its start and length) and the last
    stage's curve (its output without the residual)."""
    res = rec["residual"].astype(np.float32)
    g = {"anchor": rec["anchor_rows"].astype(np.float32)}
    for s in range(L - 1):
        g[f"out{s}"] = rec["stage_rows"][s].astype(np.float32)
    inp = rec["stage_rows"][L - 2].astype(np.float32).copy()
    inp[:, 6:] -= res[L - 2]
    g["input"] = inp
    cur = rec["rows"].astype(np.float32).copy()
    cur[:, 6:] -= res[L - 1]
    g["curve"] = cur
    return g


def geometry_iou(rows, gt_masks, ori_hw, offset, st, dec):
    """(decodable [n], vega IoU [G, n]) as curvelanes_error_breakdown.analyse_image."""
    n = len(rows)
    valid = np.zeros(n, dtype=bool)
    masks = []
    for i in range(n):
        pts = dec.points(rows[i])
        if pts is None:
            continue
        lane = rc.to_metric_points(pts, ori_hw, offset, st["y_step"])
        if len(lane) == 0:
            continue
        valid[i] = True
        masks.append(rc.lane_mask(lane, ori_hw, st["eval"]))
    iou = np.zeros((len(gt_masks), n), dtype=np.float32)
    vi = np.nonzero(valid)[0]
    if len(gt_masks) and len(vi):
        iou[:, vi] = rc.iou_matrix(gt_masks, masks)
    return valid, iou


def sample_coverage(xs, ann, rows, st, tol):
    """RoI samples (the lane's xs at ``rows``) against a GT lane's annotated rows:
    (share inside the GT's annotated rows, share within ``tol`` px normal distance of
    the GT, share off the image)."""
    x = xs[rows]
    off = float(np.mean((x < 0) | (x > 1)))
    if ann is None:
        return math.nan, math.nan, off
    r, gx = ann
    inside = (rows >= r[0]) & (rows <= r[-1])
    if not inside.any():
        return 0.0, 0.0, off
    gxi = np.interp(rows[inside], r, gx)
    if len(r) >= 2:
        slope = np.gradient(gx.astype(np.float64), r.astype(np.float64)) / (st["img_h"] / st["n_rows_m1"])
        sl = np.interp(rows[inside], r, slope)
    else:
        sl = np.zeros(int(inside.sum()))
    d = np.abs(x[inside] * (st["img_w"] - 1) - gxi) / np.sqrt(1.0 + sl ** 2)
    return float(inside.mean()), float((d <= tol).sum() / len(rows)), off


def sweep_scores(score, near, valid, iou, idx, thresholds, top_k):
    """h50 / h75 / outputs per threshold, both modes (the error breakdown's sweep)."""
    out = {}
    for mode in MODES:
        h50, h75, nout = (np.zeros(len(thresholds), dtype=np.int64) for _ in range(3))
        for k, thr in enumerate(thresholds):
            kept, _, _ = eb.select(score, near, thr, mode, top_k)
            o = eb.output_order(kept, valid, mode, idx)
            m = np.array([v for _, v in eb.assignment(iou, o).values()])
            h50[k], h75[k], nout[k] = int((m > HIT).sum()), int((m > TIGHT).sum()), len(o)
        out[mode] = (h50, h75, nout)
    return out


def score_arrays(rec):
    L = rec["stage_score"].shape[0]
    s = {f"s{k}": rec["stage_score"][k].astype(np.float64) for k in range(L)}
    s["rescore"] = rec["rescore"].astype(np.float64)
    if "fresh" in rec:
        s["fresh"] = rec["fresh"].astype(np.float64)
    return s


def analyse(rec):
    c = sd.analyse(rec)
    T, W = _T, eb._W
    st, dec, L = W["st"], W["decoder"], T["L"]
    ori_hw, offset = tuple(rec["ori_hw"]), int(rec["offset"])
    gt = [ln for ln in rc.read_lines_txt(eb.gt_file(rec["filename"])) if len(ln) > 0]
    _, gt_rows = eb.gt_geometry(gt, ori_hw, offset, st)
    tr = sd._S.get("train")
    if tr is not None and "cls_weight" in tr["assigner"]:
        # the same assigner with the score left out of the matching cost
        c["pos_noscore"] = sd.training_labels(rec, gt_rows, st,
                                              dict(tr, assigner=dict(tr["assigner"], cls_weight=0.0)))[0]
    gt_masks = [rc.lane_mask([{"x": x, "y": y} for x, y in ln], ori_hw, st["eval"]) for ln in gt]
    n, G = len(rec["score"]), len(gt)
    geoms = trace_geometries(rec, L)
    t_valid, t_iou, t_near = {"final": c["valid"]}, {"final": c["iou"].astype(np.float32)}, {}
    for k, rows in geoms.items():
        t_valid[k], t_iou[k] = geometry_iou(rows, gt_masks, ori_hw, offset, st, dec)
        t_near[k] = np.packbits(eb.kernel_pairs(rows, st["n_rows_m1"], st["img_w"], W["nms_thres"])[0], axis=1) \
            if n else np.zeros((0, 0), np.uint8)
    # per prediction, against the GT lane its final lane overlaps most (g*)
    fin = c["iou"]
    gstar = np.argmax(fin, axis=0) if G else np.full(n, -1)
    near_any = (fin[gstar, np.arange(n)] > NEAR) if G else np.zeros(n, bool)
    w1 = st["img_w"] - 1
    xin, res = geoms["input"][:, 6:], rec["residual"][L - 1].astype(np.float32)
    move, resid = np.full(n, np.nan), np.full(n, np.nan)
    cov = np.full((n, 4), np.nan)              # span, on (input samples), off (input), on (output samples)
    anns = [eb.annotated_rows(r, st["img_w"]) for r in gt_rows]
    srows = T["sample_rows"]
    for i in range(n):
        pts = dec.points(rec["rows"][i])
        if pts is not None:
            rm = eb.pred_geometry(pts, st)[0]
            if rm:
                r = np.fromiter(rm.keys(), dtype=np.int64)
                x = np.fromiter(rm.values(), dtype=np.float64)
                r = np.clip(r, 0, xin.shape[1] - 1)
                move[i] = float(np.abs(x - xin[i, r] * w1).mean())
                resid[i] = float(np.abs(res[i, r]).mean() * w1)
        ann = anns[gstar[i]] if near_any[i] else None
        span, on, off = sample_coverage(xin[i], ann, srows, st, T["tol"])
        _, on_out, _ = sample_coverage(geoms["curve"][i, 6:], ann, srows, st, T["tol"])
        cov[i] = (span, on, off, on_out)
    # F1 sweeps of the other (geometry, score) pairs on the same stored predictions
    scores = score_arrays(rec)
    scores["oracle"] = np.where(c["valid"], fin.max(0), 0.0) if G else np.zeros(n)
    sweeps = {}
    for name, geom, skey in T["combos"]:
        if geom == "final":
            near, valid, iou = eb.unpack_near(c), c["valid"], c["iou"]
        else:
            near = np.unpackbits(t_near[geom], axis=1, count=n).astype(bool) if n else np.zeros((0, 0), bool)
            valid, iou = t_valid[geom], t_iou[geom]
        sweeps[name] = sweep_scores(scores[skey], near, valid, iou, rec["idx"], W["thresholds"], W["top_k"])
    c.update(t_valid=t_valid, t_iou=t_iou, t_near=t_near, gstar=gstar, move=move, resid=resid, cov=cov,
             t_sweeps=sweeps, has_ann=np.array([a is not None for a in anns], dtype=bool))
    return c


def lane_hits_job(i):
    """Per GT lane of image i: hit@50 under each alternative of _JOB["alts"]
    (name, geometry, score per image, threshold, mode)."""
    c, rec = _JOB["results"][i], _JOB["records"][i]
    n = len(rec["score"])
    out = {}
    for name, geom, scores, thr, mode in _JOB["alts"]:
        score = scores[i] if isinstance(scores, list) else score_arrays(rec)[scores]
        if geom == "final":
            near, valid, iou = eb.unpack_near(c), c["valid"], c["iou"]
        else:
            near = np.unpackbits(c["t_near"][geom], axis=1, count=n).astype(bool) if n else np.zeros((0, 0), bool)
            valid, iou = c["t_valid"][geom], c["t_iou"][geom]
        kept, _, _ = eb.select(score, near, thr, mode, _JOB["top_k"])
        o = eb.output_order(kept, valid, mode, rec["idx"])
        m = eb.assignment(iou, o)
        out[name] = np.array([m.get(g, (-1, 0.0))[1] > HIT for g in range(c["n_gt"])], dtype=bool)
    return out


def probe_sweep_job(i):
    """F1 sweep (both modes) of image i under each probe score of _JOB["probes"]."""
    c, rec = _JOB["results"][i], _JOB["records"][i]
    near = eb.unpack_near(c)
    return {name: sweep_scores(sc[i], near, c["valid"], c["iou"], rec["idx"], _JOB["thresholds"], _JOB["top_k"])
            for name, sc in _JOB["probes"].items()}


def run_jobs(fn, n, jobs):
    if jobs == 1:
        return [fn(i) for i in range(n)]
    with mp.get_context("fork").Pool(jobs) as pool:
        return list(pool.imap(fn, range(n), chunksize=64))


# ============================================================================
# feature probes (main process)
# ============================================================================
def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd_ = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd_, (Xte - mu) / sd_


def fit_logistic(X, y, l2=1e-4, iters=300):
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda Z: torch.sigmoid(torch.as_tensor(Z, dtype=torch.float32) @ w.detach() + b.detach()).numpy()


def fit_mlp(X, y, num_fc, epochs=6, batch=8192, lr=2e-3, seed=0):
    """num_fc x (Linear + ReLU) + Linear, the cls tower and classifier shape."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    d = X.shape[1]
    layers = []
    for _ in range(max(num_fc, 1)):
        layers += [torch.nn.Linear(d, d), torch.nn.ReLU()]
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = epochs * max(1, math.ceil(len(X) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for k in range(0, len(X), batch):
            j = perm[k:k + batch]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[j]).squeeze(1), y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()

    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.as_tensor(Z, dtype=torch.float32)).squeeze(1)).numpy()
    return predict


def cross_fit(X, y, fold, kind, num_fc=2):
    """Out-of-fold probability for every row: fit on one image half, predict the other."""
    p = np.full(len(y), np.nan)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        Xtr, Xte = _standardize(X[tr].astype(np.float32), X[te].astype(np.float32))
        model = fit_logistic(Xtr, y[tr]) if kind == "linear" else fit_mlp(Xtr, y[tr], num_fc)
        p[te] = model(Xte)
    return p


def geometry_descriptors(results, records, L):
    """Per stored prediction, what the head could know about its own lane without the
    GT: start, length, extent, slant, residual and move, the previous stage's logit,
    the share of its RoI samples off the image, bottom x."""
    out = []
    for c, rec in zip(results, records):
        r = rec["rows"]
        prev = np.clip(rec["stage_score"][L - 2].astype(np.float64), 1e-6, 1 - 1e-6)
        slant = np.log1p(np.clip(np.nan_to_num(c["pred_attr"][:, 0], nan=0.0, posinf=50.0), 0, 50))
        out.append(np.stack([r[:, 2], r[:, 5], c["pred_attr"][:, 1], slant,
                             np.log1p(np.nan_to_num(c["resid"], nan=0.0)),
                             np.log1p(np.nan_to_num(c["move"], nan=0.0)),
                             np.log(prev / (1 - prev)), c["cov"][:, 2], np.abs(r[:, 3] - 0.5)], 1))
    return np.concatenate(out) if out else np.zeros((0, 9))


# ============================================================================
# report helpers
# ============================================================================
_med, _share, _f, auc = sd._med, sd._share, sd._f, sd.auc
NAN = math.nan
LANE_CLASSES = ["hit", "hedged", "dismissed", "taken", "suppressed", "capped", "mislocalized", "absent"]


def geom_names(L):
    return ["anchor"] + [f"out{s}" for s in range(L - 1)] + ["input", "final", "curve"]


def geom_label(k, L):
    if k == "anchor":
        return "anchor (stage 0 input)"
    if k.startswith("out"):
        return f"stage {k[3:]} output"
    return {"input": f"stage {L - 1} input (stage {L - 2} output minus its residual)",
            "final": f"stage {L - 1} output (final)",
            "curve": f"stage {L - 1} output minus its residual"}[k]


FRESH_NAME = "final lanes, fresh (every RoI slot on the output)"


def combos_for(L, fresh=False):
    out = [(f"stage {s} alone (its lanes, its score)", f"out{s}", f"s{s}") for s in range(L - 1)]
    out += [(f"final lanes, stage {L - 2} score", "final", f"s{L - 2}"),
            ("final curve, residual zeroed", "curve", f"s{L - 1}"),
            (f"stage {L - 1} input lanes, final score", "input", f"s{L - 1}"),
            ("final lanes, rescore (RoI on the output)", "final", "rescore")]
    if fresh:
        out.append((FRESH_NAME, "final", "fresh"))
    out += [("anchors, stage 0 score", "anchor", "s0"),
            ("final lanes, oracle score (its eval IoU)", "final", "oracle")]
    return out


def agg_sweep(results, get, thresholds, n_gt):
    tab = {}
    for mode in MODES:
        h50 = sum(get(c)[mode][0] for c in results)
        h75 = sum(get(c)[mode][1] for c in results)
        nout = sum(get(c)[mode][2] for c in results)
        rows = []
        for k, thr in enumerate(thresholds):
            p, r, f = eb.prf(int(h50[k]), int(nout[k]), n_gt)
            rows.append(dict(thr=float(thr), P=p, R=r, F1_50=f, F1_75=eb.prf(int(h75[k]), int(nout[k]), n_gt)[2],
                             out=int(nout[k])))
        tab[mode] = rows
    return tab


def best_row(rows):
    return max(rows, key=lambda r: (r["F1_50"], -r["thr"]))


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def jsafe(o):
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsafe(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return None if math.isnan(o) else (1e30 if math.isinf(o) else o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def best_copy_label(c, pos, g):
    """(the eval's best copy of lane g is an assigner positive, label off) under the
    positives ``pos``; label off as curvelanes_score_diag: every positive of g is
    eval-wrong while a correct copy exists."""
    valid, iou = c["valid"], c["iou"]
    corr = np.nonzero(valid & (iou[g] > HIT))[0]
    if not len(corr):
        return NAN, NAN
    best = int(corr[np.argmax(iou[g, corr])])
    mine = np.nonzero(pos == g)[0]
    is_pos = bool(pos[best] == g)
    return is_pos, bool(not is_pos and len(mine) and iou[g, mine].max() <= HIT)


def lane_trace_rows(results, records, thr, mode, ctx, L):
    """One row per GT lane: the error breakdown's outcome, the score tool's copy stats,
    and the trace of its best-scored correct copy (p_*), of the wrong copy that outranks
    it (w_*) and of its best copy by IoU when no copy is correct (b_*)."""
    G_NAMES = geom_names(L)
    rows = []
    for c, rec in zip(results, records):
        ln_rows, _, _ = eb.evaluate_image(c, rec, thr, mode, ctx)
        score, valid, iou = rec["score"].astype(np.float64), c["valid"], c["iou"]
        sc, resc = rec["stage_score"].astype(np.float64), rec["rescore"].astype(np.float64)
        frs = rec["fresh"].astype(np.float64) if "fresh" in rec else None
        best_any = iou.max(0) if iou.shape[0] else np.zeros(len(score))
        ti, tv = c["t_iou"], c["t_valid"]
        for ln in ln_rows:
            g = ln["gt"]
            base = sd.lane_stats(c, rec, dict(ln, score=ln.get("score", NAN)), thr)
            row = dict(image=ln["image"], gt=g, reason=ln["reason"], cls=base["cls"], slant=ln["slant"],
                       extent=ln["extent"], n_points=ln["n_points"], nn_dist=ln["nn_dist"], lanes=ln["lanes"],
                       n_correct=base["n_correct"], mismatch=base["mismatch"], has_target=base["has_target"],
                       label_off=base.get("label_off", NAN))
            if "pos" in c:
                row["best_pos"] = best_copy_label(c, c["pos"], g)[0]
            if "pos_noscore" in c:
                row["best_pos_noscore"], row["label_off_noscore"] = best_copy_label(c, c["pos_noscore"], g)
            for k in G_NAMES:
                v = ti[k][g][tv[k]]
                row[f"best_{k}"] = float(v.max()) if len(v) else 0.0
            corr = np.nonzero(valid & (iou[g] > HIT))[0]
            if len(corr):
                p = int(corr[np.argmax(score[corr])])
                for k in G_NAMES:
                    row[f"p_iou_{k}"] = float(ti[k][g, p]) if tv[k][p] else 0.0
                outs = [row[f"p_iou_out{s}"] for s in range(L - 1)] + [row["p_iou_final"]]
                row["p_first"] = next(s for s, v in enumerate(outs) if v > HIT)
                row["p_pulled_in"] = row["p_iou_input"] <= HIT
                row.update(p_move=float(c["move"][p]), p_res=float(c["resid"][p]),
                           p_rescore=float(resc[p]), p_dlogit=float(logit(sc[L - 1, p]) - logit(sc[L - 2, p])),
                           p_span=float(c["cov"][p, 0]), p_on=float(c["cov"][p, 1]), p_off=float(c["cov"][p, 2]),
                           p_on_out=float(c["cov"][p, 3]))
                for s in range(L):
                    row[f"p_s{s}"] = float(sc[s, p])
                if frs is not None:
                    row["p_fresh"] = float(frs[p])
                wrong = np.nonzero(valid & (iou[g] > NEAR) & (iou[g] <= HIT) & (best_any <= HIT))[0]
                if base["mismatch"] and len(wrong):
                    w = int(wrong[np.argmax(score[wrong])])
                    row.update(w_iou_input=float(ti["input"][g, w]) if tv["input"][w] else 0.0,
                               w_move=float(c["move"][w]), w_score=float(score[w]), w_rescore=float(resc[w]),
                               w_prev=float(sc[L - 2, w]))
                    if frs is not None:
                        row["w_fresh"] = float(frs[w])
            elif valid.any():
                b = int(np.argmax(np.where(valid, iou[g], -1.0)))
                for k in G_NAMES:
                    row[f"b_iou_{k}"] = float(ti[k][g, b]) if tv[k][b] else 0.0
            rows.append(row)
    return rows


def pred_table(results, records, L):
    """Concatenated per-prediction arrays."""
    keys = geom_names(L)
    P = {k: [] for k in ["image", "score", "rescore", "valid", "gstar_iou", "move", "resid", "span", "on", "off",
                         "on_out", "idx", "gt_slant", "gt_extent", "gt_npts", "pos"]
         + [f"s{s}" for s in range(L)] + [f"correct_{k}" for k in keys] + [f"iou_{k}" for k in keys]}
    for i, (c, rec) in enumerate(zip(results, records)):
        n = len(rec["score"])
        gs = c["gstar"]
        P["image"].append(np.full(n, i))
        P["score"].append(rec["score"].astype(np.float64))
        P["rescore"].append(rec["rescore"].astype(np.float64))
        if "fresh" in rec:
            P.setdefault("fresh", []).append(rec["fresh"].astype(np.float64))
        P["valid"].append(c["valid"])
        P["idx"].append(rec["idx"])
        for s in range(L):
            P[f"s{s}"].append(rec["stage_score"][s].astype(np.float64))
        for k in keys:
            I = c["t_iou"][k]
            P[f"correct_{k}"].append(c["t_valid"][k] & ((I.max(0) if I.shape[0] else np.zeros(n)) > HIT))
            P[f"iou_{k}"].append(np.where(c["t_valid"][k], I[gs, np.arange(n)], 0.0) if I.shape[0] else np.zeros(n))
        P["gstar_iou"].append(P["iou_final"][-1])
        P["move"].append(c["move"])
        P["resid"].append(c["resid"])
        for j, k in enumerate(["span", "on", "off", "on_out"]):
            P[k].append(c["cov"][:, j])
        at = c["attrs"]
        ok = P["correct_final"][-1]
        for key, name in (("gt_slant", "slant"), ("gt_extent", "extent"), ("gt_npts", "n_points")):
            P[key].append(np.array([at[g][name] if (ok[j] and g >= 0) else NAN for j, g in enumerate(gs)],
                                   dtype=np.float64) if n else np.zeros(0))
        P["pos"].append(c["pos"] >= 0 if "pos" in c else np.zeros(n, bool))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in P.items()}


def anchor_groups(records, dec, st):
    """Per stored prediction: where its anchor enters the image (bottom / left / right
    edge) and the anchor's slant bin."""
    cache, out = {}, []
    for rec in records:
        g = []
        for q, row in zip(rec["idx"].tolist(), rec["anchor_rows"]):
            if q not in cache:
                pts = dec.points(row)
                name = "none"
                if pts is not None:
                    vis = pts[(pts[:, 0] >= 0) & (pts[:, 0] <= 1)]
                    if len(vis) >= 2:
                        lo = vis[np.argmax(vis[:, 1])]
                        edge = "bottom" if lo[1] >= 0.95 else ("left edge" if lo[0] < 0.5 else "right edge")
                        d = np.abs(np.diff(vis, axis=0))
                        sl = d[:, 0].sum() * (st["img_w"] - 1) / max(d[:, 1].sum() * (st["img_h"] - 1), 1e-6)
                        name = f"{edge}, slant {'<1' if sl < 1 else '1-3' if sl < 3 else '>3'}"
                cache[q] = name
            g.append(cache[q])
        out.append(np.array(g, dtype=object))
    return np.concatenate(out) if out else np.zeros(0, dtype=object)


# ============================================================================
# report
# ============================================================================
def pct(v, w=6):
    return f"{'-':>{w}}" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:{w - 1}.0f}%"


def num(v, w=7, d=2):
    return f"{'-':>{w}}" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:{w}.{d}f}"


def sec_cascade(lanes, sweeps, L, say, key):
    G = len(lanes)
    say("1. the cascade, stage by stage")
    say("   a) GT lanes with a correct copy (eval IoU > 0.5, any score) among the stored predictions, per geometry")
    say(f"   {'geometry':52s}{'lanes':>8}{'% GT':>8}{'median best IoU':>18}")
    cover = {}
    for k in geom_names(L):
        v = [ln[f"best_{k}"] for ln in lanes]
        n = sum(x > HIT for x in v)
        cover[k] = n / max(G, 1)
        say(f"   {geom_label(k, L):52s}{n:8d}{100 * n / max(G, 1):8.1f}{_med(v):18.3f}")
    key["correct_copy_input"], key["correct_copy_final"] = cover["input"], cover["final"]
    key["correct_copy_curve"] = cover["curve"]
    say("")
    say("   b) F1 of each (geometry, score) pair at its own best threshold (same stored predictions)")
    say(f"   {'':52s}{'----------- with NMS -----------':>34}  {'---------- without NMS ---------':>34}")
    say(f"   {'':52s}" + "  ".join(f"{'thr':>6}{'P':>7}{'R':>7}{'F1@50':>7}{'F1@75':>7}" for _ in MODES))
    f1 = {}
    for name, tab in sweeps.items():
        cells = []
        for mode in MODES:
            b = best_row(tab[mode])
            f1[(name, mode)] = b
            cells.append(f"{b['thr']:6.2f}{100 * b['P']:7.2f}{100 * b['R']:7.2f}{100 * b['F1_50']:7.2f}"
                         f"{100 * b['F1_75']:7.2f}")
        say(f"   {name:52s}" + "  ".join(cells))
        for mode in MODES:
            key[f"F1 {mode}: {name}"] = f1[(name, mode)]["F1_50"]
    say("   the scores other than the final one rank the stored predictions only (the 64 highest by final "
        "score): an approximation")
    say("   oracle: each prediction scored by its own eval IoU, the most any scoring could get from these lanes")
    say("")
    return cover, f1


def sec_input_output(P, thr, L, say, key):
    say("2. is the final score about the lane the last stage was given or the lane it outputs?")
    say("   every stored prediction, against the GT lane its final lane overlaps most; input = the last stage's "
        "input lane, output = the final lane")
    ok_in = P["iou_input"] > HIT
    ok_out = P["valid"] & (P["iou_final"] > HIT)
    say(f"   {'input':>8}{'output':>8}{'preds':>10}{'share':>8}{'score':>8}{'>=thr':>8}{'rescore':>9}{'>=thr':>8}"
        f"{'move px':>9}")
    tab = {}
    for a, b, name in ((True, True, "stayed correct"), (False, True, "pulled in"), (True, False, "pushed out"),
                       (False, False, "stayed wrong")):
        m = (ok_in == a) & (ok_out == b)
        t = dict(n=int(m.sum()), share=float(m.mean()) if len(m) else NAN,
                 score=_med(P["score"][m]), above=_share(P["score"][m] >= thr),
                 rescore=_med(P["rescore"][m]), re_above=_share(P["rescore"][m] >= thr), move=_med(P["move"][m]))
        tab[name] = t
        say(f"   {'right' if a else 'wrong':>8}{'right' if b else 'wrong':>8}{t['n']:10d}{pct(t['share'], 8)}"
            f"{num(t['score'], 8)}{pct(t['above'], 8)}{num(t['rescore'], 9)}{pct(t['re_above'], 8)}"
            f"{num(t['move'], 9, 1)}   {name}")
    key["pulled_in score"], key["stayed_correct score"] = tab["pulled in"]["score"], tab["stayed correct"]["score"]
    key["pushed_out score"] = tab["pushed out"]["score"]
    say("   score / rescore / move: medians; >=thr: share at or above the operating threshold")
    say("")
    say("   AUC of each score (rows) for 'correct for some GT lane' in each geometry (columns)")
    cols = [f"out{s}" for s in range(L - 1)] + ["input", "final", "curve"]
    say(f"   {'':22s}" + "".join(f"{k:>10s}" for k in cols))
    rows = [f"s{s}" for s in range(L)] + ["rescore"] + (["fresh"] if "fresh" in P else [])
    mat = {}
    for r in rows:
        mat[r] = {k: auc(P[r], P[f"correct_{k}"]) for k in cols}
        name = (f"stage {r[1:]} score" + (" (final)" if r == f"s{L - 1}" else "")) if r.startswith("s") else r
        say(f"   {name:22s}" + "".join(f"{mat[r][k]:10.3f}" for k in cols))
    key["AUC final score vs final"], key["AUC final score vs input"] = mat[f"s{L - 1}"]["final"], mat[f"s{L - 1}"]["input"]
    key["AUC rescore vs final"] = mat["rescore"]["final"]
    if "fresh" in mat:
        key["AUC fresh vs final"] = mat["fresh"]["final"]
    say("")
    return tab, mat


def sec_classes(lanes, thr, L, say, key):
    say("3. missed lanes through the cascade (the best-scored correct copy of each lane)")
    by = {k: [ln for ln in lanes if ln["cls"] == k] for k in LANE_CLASSES}
    G = len(lanes)
    say(f"   a) geometry   {'lanes':>7}{'% GT':>7}{'in IoU':>8}{'pulled':>8}{'move px':>9}{'resid px':>9}"
        + "".join(f"{'1st s' + str(s):>8}" for s in range(L)))
    tabs = {}
    for k in LANE_CLASSES:
        has = [ln for ln in by[k] if "p_iou_final" in ln]
        if not has:
            continue
        t = dict(n=len(by[k]), pct=len(by[k]) / max(G, 1), in_iou=_med(ln["p_iou_input"] for ln in has),
                 pulled=_share(ln["p_pulled_in"] for ln in has), move=_med(ln["p_move"] for ln in has),
                 res=_med(ln["p_res"] for ln in has),
                 first=[_share(ln["p_first"] == s for ln in has) for s in range(L)],
                 s=[_med(ln[f"p_s{s}"] for ln in has) for s in range(L)],
                 dlogit=_med(ln["p_dlogit"] for ln in has), rescore=_med(ln["p_rescore"] for ln in has),
                 re_above=_share(ln["p_rescore"] >= thr for ln in has))
        tabs[k] = t
        say(f"   {k:13s}{t['n']:7d}{100 * t['pct']:7.1f}{num(t['in_iou'], 8)}{pct(t['pulled'], 8)}"
            f"{num(t['move'], 9, 1)}{num(t['res'], 9, 1)}" + "".join(pct(v, 8) for v in t["first"]))
    say("   in IoU: the copy's last-stage input lane against this GT lane; pulled: share of lanes whose copy was "
        "wrong (IoU <= 0.5) at the input")
    say("   move: mean |x_final - x_input| over the final lane's rows; resid: mean |residual| (px, network input); "
        "1st s: first stage whose output is correct")
    say("")
    say("   b) scores     " + "".join(f"{'s' + str(s):>8}" for s in range(L))
        + f"{'dlogit':>9}{'rescore':>9}{'>=thr':>8}")
    for k, t in tabs.items():
        say(f"   {k:13s}" + "".join(num(v, 8) for v in t["s"]) + f"{num(t['dlogit'], 9)}{num(t['rescore'], 9)}"
            f"{pct(t['re_above'], 8)}")
    say("   s<k>: score at stage k (medians); dlogit: logit change at the last stage; rescore: the last stage's score "
        "when it pools on its own output")
    say("")
    for k in ("hedged", "dismissed"):
        if k in tabs:
            key[f"{k} pulled_in"] = tabs[k]["pulled"]
            key[f"{k} rescore>=thr"] = tabs[k]["re_above"]
    key["hit pulled_in"] = tabs.get("hit", {}).get("pulled", NAN)
    say("   c) the wrong copy that outranks the correct one (mismatch lanes)")
    say(f"   {'':13s}{'lanes':>7}{'wrong in IoU':>14}{'wrong was right':>17}{'wrong move':>12}{'scores w / c':>16}"
        f"{'rescore w / c':>16}{'rescore fixes':>15}")
    mis = {}
    for k in ("hedged", "dismissed", "hit", "taken"):
        L2 = [ln for ln in by[k] if "w_score" in ln]
        if not L2:
            continue
        t = dict(n=len(L2), w_in=_med(ln["w_iou_input"] for ln in L2), w_right=_share(ln["w_iou_input"] > HIT for ln in L2),
                 w_move=_med(ln["w_move"] for ln in L2), w_s=_med(ln["w_score"] for ln in L2),
                 c_s=_med(ln["p_s" + str(L - 1)] for ln in L2), w_r=_med(ln["w_rescore"] for ln in L2),
                 c_r=_med(ln["p_rescore"] for ln in L2), fix=_share(ln["p_rescore"] > ln["w_rescore"] for ln in L2))
        mis[k] = t
        say(f"   {k:13s}{t['n']:7d}{num(t['w_in'], 14)}{pct(t['w_right'], 17)}{num(t['w_move'], 12, 1)}"
            f"{num(t['w_s'], 9)} /{num(t['c_s'], 5)}{num(t['w_r'], 9)} /{num(t['c_r'], 5)}{pct(t['fix'], 15)}")
    say("   wrong was right: the wrong copy's input lane was correct (the last stage pushed it out); rescore fixes: "
        "the correct copy outranks the wrong one under the rescore")
    say("")
    say("   d) lanes with a correct copy in each geometry (any score), per outcome")
    names = geom_names(L)
    say(f"   {'':13s}" + "".join(f"{k:>9s}" for k in names))
    cov = {}
    for k in LANE_CLASSES:
        if not by[k]:
            continue
        cov[k] = {g: _share(ln[f"best_{g}"] > HIT for ln in by[k]) for g in names}
        say(f"   {k:13s}" + "".join(pct(cov[k][g], 9) for g in names))
    say("   mislocalized lanes with 'input' or an earlier 'out' high: a copy was correct before the last stage "
        "moved it")
    say("")
    return dict(classes=tabs, mismatch=mis, geometry=cov)


def sec_alternatives(lanes_mode, hits, alt_names, say, title):
    say(title)
    say(f"   {'':13s}{'lanes':>7}" + "".join(f"{n[:22]:>24s}" for n in alt_names))
    out = {}
    cls = np.array([ln["cls"] for ln in lanes_mode], dtype=object)
    for k in LANE_CLASSES:
        m = cls == k
        if not m.any():
            continue
        out[k] = {n: float(hits[n][m].mean()) for n in alt_names}
        say(f"   {k:13s}{int(m.sum()):7d}" + "".join(pct(out[k][n], 24) for n in alt_names))
    say("   cells: share of the lanes of that outcome (at the head's operating point) that are hits under the "
        "alternative at its own best threshold")
    say("")
    return out


def _known(ln, k):
    v = ln.get(k)
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def sec_ownership(lanes, hits, thr, L, say, key):
    """4b: lanes by the stage at which their best-scored correct copy first became
    correct (p_first); hits under the alternatives of section 4; training labels of
    the eval's best copy with and without the score in the matching cost."""
    say("4b. early ownership: lanes by the stage at which their best-scored correct copy first became correct")
    G = len(lanes)
    first = np.array([ln.get("p_first", -1) for ln in lanes], dtype=np.int64)
    is_hit = np.array([ln["cls"] == "hit" for ln in lanes], dtype=bool)
    fresh = "fresh" in hits
    lab = any("best_pos" in ln for ln in lanes)
    lab0 = any("best_pos_noscore" in ln for ln in lanes)
    head = (f"   {'first correct':14s}{'lanes':>7}{'% GT':>7}{'hit':>6}" + "".join(f"{'s' + str(s):>7}" for s in range(L))
            + f"{'rescore':>9}" + (f"{'fresh':>7}" if fresh else "") + f"{'mismatch':>10}"
            + (f"{'best +':>8}" if lab else "") + (f"{'best +, no score':>18}" if lab0 else "")
            + f"{'hit: rescore':>14}" + (f"{'hit: fresh':>12}" if fresh else ""))
    say(head)
    tab = {}
    for s_ in range(L):
        m = first == s_
        sel = [ln for ln, k in zip(lanes, m) if k]
        if not sel:
            continue
        t = dict(n=int(m.sum()), pct=m.mean() if G else NAN, hit=float(is_hit[m].mean()),
                 s=[_med(ln[f"p_s{k}"] for ln in sel) for k in range(L)], rescore=_med(ln["p_rescore"] for ln in sel),
                 fresh=_med(ln["p_fresh"] for ln in sel if "p_fresh" in ln) if fresh else NAN,
                 mismatch=_share(ln["mismatch"] for ln in sel),
                 best_pos=_share(ln["best_pos"] for ln in sel if _known(ln, "best_pos")) if lab else NAN,
                 best_pos0=_share(ln["best_pos_noscore"] for ln in sel if _known(ln, "best_pos_noscore"))
                 if lab0 else NAN,
                 hit_rescore=float(hits["rescore"][m].mean()),
                 hit_fresh=float(hits["fresh"][m].mean()) if fresh else NAN)
        tab[s_] = t
        say(f"   {'stage ' + str(s_):14s}{t['n']:7d}{100 * t['pct']:7.1f}{pct(t['hit'], 6)}"
            + "".join(num(v, 7) for v in t["s"]) + num(t["rescore"], 9) + (num(t["fresh"], 7) if fresh else "")
            + pct(t["mismatch"], 10) + (pct(t["best_pos"], 8) if lab else "") + (pct(t["best_pos0"], 18) if lab0 else "")
            + pct(t["hit_rescore"], 14) + (pct(t["hit_fresh"], 12) if fresh else ""))
    say("   s<k>, rescore, fresh: medians of the copy's scores; mismatch: a never-correct copy outranks it; best +: the "
        "eval's best copy of the lane is a positive of the training assigner (best +, no score: the same assigner "
        "with cls_weight 0); hit: rescore / fresh: hits under that score at its own best threshold (section 4)")
    late = first >= 1
    key["late lanes % GT"] = float(late.mean()) if G else NAN
    key["late lanes hit"] = float(is_hit[late].mean()) if late.any() else NAN
    key["early lanes hit"] = float(is_hit[first == 0].mean()) if (first == 0).any() else NAN
    if fresh and late.any():
        key["late lanes hit under fresh"] = float(hits["fresh"][late].mean())
    if fresh and (first == 0).any():
        key["early lanes hit under fresh"] = float(hits["fresh"][first == 0].mean())
    for name, k in (("best copy positive", "best_pos"), ("best copy positive, no score", "best_pos_noscore")):
        for nm, msk in (("late", late), ("early", first == 0)):
            v = [ln[k] for ln, w in zip(lanes, msk) if w and _known(ln, k)]
            if v:
                key[f"{nm} lanes {name}"] = _share(v)
    say("")
    mm = [(ln, f) for ln, f in zip(lanes, first) if "w_score" in ln and ln["cls"] != "hit"]
    if mm:
        say("   missed lanes where a never-correct copy outranks the correct one: does the rescore / fresh score "
            "reorder the two?")
        say(f"   {'first correct':14s}{'lanes':>7}{'wrong / correct':>17}{'rescore fixes':>15}"
            + (f"{'fresh w / c':>14}{'fresh fixes':>13}" if fresh else ""))
        for s_ in range(L):
            sel = [ln for ln, f in mm if f == s_]
            if not sel:
                continue
            say(f"   {'stage ' + str(s_):14s}{len(sel):7d}{num(_med(ln['w_score'] for ln in sel), 9)} /"
                f"{num(_med(ln['p_s' + str(L - 1)] for ln in sel), 6)}"
                f"{pct(_share(ln['p_rescore'] > ln['w_rescore'] for ln in sel), 15)}"
                + (f"{num(_med(ln['w_fresh'] for ln in sel), 7)} /{num(_med(ln['p_fresh'] for ln in sel), 5)}"
                   f"{pct(_share(ln['p_fresh'] > ln['w_fresh'] for ln in sel), 13)}" if fresh else ""))
        if fresh:
            key["mismatch fresh fixes"] = _share(ln["p_fresh"] > ln["w_fresh"] for ln, _ in mm)
        say("")
    good = [ln for ln in lanes if ln.get("best_final", 0) > 0.8]
    if good:
        say("   lanes whose best final copy has IoU > 0.8: hit rate by correct copies (stored) x stage-0 output")
        say(f"   {'copies':10s}{'stage 0 had a correct copy':>30s}{'had none':>18s}")
        bins = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 10 ** 9, ">= 5")]
        for lo, hi, name in bins:
            cells = []
            for early in (True, False):
                sel = [ln["cls"] == "hit" for ln in good if lo <= ln["n_correct"] <= hi
                       and (ln.get("best_out0", 0) > HIT) == early]
                cells.append(f"{pct(_share(sel), 6)} of {len(sel):6d}")
            say(f"   {name:10s}{cells[0]:>30s}{cells[1]:>18s}")
        say("   correct copies count the stored predictions only (the top --max-per-image by final score)")
        say("")
    return tab


def sec_coverage(lanes, P, thr, tol, say, key):
    say(f"6. what the last stage's RoI samples see (36 rows over the full height; on lane = within {tol:.0f} px of the "
        f"GT lane, one feature cell)")
    say(f"   {'':13s}{'lanes':>7}{'in span':>9}{'on lane':>9}{'on (out)':>10}{'off image':>11}")
    tab = {}
    for k in LANE_CLASSES:
        L2 = [ln for ln in lanes if ln["cls"] == k and "p_span" in ln]
        if not L2:
            continue
        t = dict(n=len(L2), span=_med(ln["p_span"] for ln in L2), on=_med(ln["p_on"] for ln in L2),
                 on_out=_med(ln["p_on_out"] for ln in L2), off=_med(ln["p_off"] for ln in L2))
        tab[k] = t
        say(f"   {k:13s}{t['n']:7d}{pct(t['span'], 9)}{pct(t['on'], 9)}{pct(t['on_out'], 10)}{pct(t['off'], 11)}")
    say("   medians over the lanes' best-scored correct copies. in span: samples inside the GT lane's annotated "
        "rows; on (out): the samples the rescore uses")
    for key_, title, edges, names in sd.BINS[:2]:
        say(f"   on-lane share by the lane's {title} (hits / hedged + dismissed):")
        vals = np.array([ln[key_] for ln in lanes], dtype=np.float64)
        which = np.digitize(np.nan_to_num(vals, nan=-1), edges)
        cells = []
        for b, name in enumerate(names):
            h = [ln["p_on"] for ln, w in zip(lanes, which) if w == b and ln["cls"] == "hit" and "p_on" in ln]
            m = [ln["p_on"] for ln, w in zip(lanes, which) if w == b and ln["cls"] in ("hedged", "dismissed")
                 and "p_on" in ln]
            cells.append(f"{name}: {pct(_med(h), 4).strip()} / {pct(_med(m), 4).strip()}")
        say("     " + "   ".join(cells))
    from scipy.stats import spearmanr
    m = P["correct_final"] & np.isfinite(P["on"])
    rho = float(spearmanr(P["score"][m], P["on"][m])[0]) if m.sum() > 2 else NAN
    key["spearman score vs on-lane"] = rho
    say(f"   Spearman correlation of the final score with the on-lane share, over correct copies: {num(rho, 6, 3)}")
    say("")
    return dict(classes=tab, spearman=rho)


def sec_calibration(P, say):
    say("8. score calibration: share of stored predictions correct for some GT lane (eval IoU > 0.5) per score bin")
    edges = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001]
    say(f"   {'score':>12}{'preds':>10}{'correct':>9}{'rescore: preds':>16}{'correct':>9}")
    tab = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (P["score"] >= a) & (P["score"] < b)
        r = (P["rescore"] >= a) & (P["rescore"] < b)
        t = dict(lo=a, hi=min(b, 1.0), n=int(m.sum()), correct=_share(P["correct_final"][m]),
                 n_re=int(r.sum()), correct_re=_share(P["correct_final"][r]))
        tab.append(t)
        say(f"   {a:5.2f}-{min(b, 1.0):4.2f}{t['n']:10d}{pct(t['correct'], 9)}{t['n_re']:16d}{pct(t['correct_re'], 9)}")
    say("")
    return tab


def sec_anchors(P, groups, thr, say):
    say("9. anchors: correct copies and their final scores by where the anchor enters the image")
    say(f"   {'anchor':26s}{'priors':>7}{'preds':>9}{'correct':>9}{'score':>8}{'>=thr':>8}{'AUC':>7}")
    tab = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        ok = m & P["correct_final"]
        t = dict(priors=int(len(np.unique(P["idx"][m]))), n=int(m.sum()), correct=int(ok.sum()),
                 score=_med(P["score"][ok]), above=_share(P["score"][ok] >= thr),
                 auc=auc(P["score"][m], P["correct_final"][m]))
        tab[g] = t
        say(f"   {g:26s}{t['priors']:7d}{t['n']:9d}{t['correct']:9d}{num(t['score'], 8)}{pct(t['above'], 8)}"
            f"{num(t['auc'], 7, 3)}")
    say("   score / >=thr over the correct copies; AUC within the group")
    say("")
    return tab


def reading(say):
    say("10. reading: which outcome points at which change of the network")
    for line in [
        "  - section 2 'pulled in' copies score like 'stayed wrong' ones and 'pushed out' like 'stayed correct', and",
        "    the rescore recovers hedged / dismissed lanes (section 4): the score judges the lane the last stage was",
        "    given, not the lane it outputs. Change: score after refinement (a scoring step that pools on the",
        "    output lane), or a quality branch fed with the output geometry.",
        "  - section 7: P1 (eval label on the classifier's own feature) is about the head (AUC, F1): the",
        "    feature does not carry correctness. Look at section 6 (samples off the lane for flat / short lanes)",
        "    and change what the RoI sees (sampling along the lane's own span, more context).",
        "  - P1 >> P2 ~ head: the information is there, the training label",
        "    teaches something else. Change the assignment / classification target.",
        "  - P1 and P2 >> head: the classifier does not reach what its feature allows on this data; run the",
        "    tool on a training-image dump (--split train): if the head scores its training images well, it is",
        "    generalization (augmentation, overfitting); if not, the loss (focal / weights) or optimization.",
        "  - P3 >> P1: the cls tower throws the information away (it is in the RoI feature).",
        "  - P4 >> P1: the score should see the lane's geometry explicitly. P5 >> P1: pool on the output lane.",
        "  - section 1b: final curve (residual zeroed) close to the final lanes: the Bezier curve carries the shape;",
        "    far below: the residual does the work. A stage alone close to the final: the stages after it add little.",
        "  - section 3d: mislocalized lanes with a correct input: the last stage moved a correct lane away.",
        "  - section 9: one anchor group with low scores for its correct copies: a per-anchor prior in the classifier.",
        "  - section 4b: rows 'stage 1' / 'stage 2' far below 'stage 0' in hit and score: early ownership (the lane is",
        "    trusted only if a copy was correct at stage 0). 'fresh' lifts the late rows towards the stage-0 row while",
        "    the stage-0 row keeps its hits: the classifier reads the anchor / earlier lanes from its concatenated RoI",
        "    slots (feature memory); change: score from features of the final lane only. If the stage-0 row also",
        "    collapses under 'fresh', the counterfactual is out of the classifier's training distribution: inconclusive.",
        "  - section 4b: 'best +, no score' >> 'best +' on the late rows: the score term of the matching cost hands the",
        "    positive to the early owner (label lock-in); change: match on geometry. Neither 'fresh' nor 'no score'",
        "    moves the late rows: the ownership is in the classifier's weights (the prior's identity) or the training",
        "    dynamics of earlier stages.",
    ]:
        say(line)


def sec_residual(lanes, say, key):
    say("5. curve vs residual (the best-scored correct copy of each lane)")
    say(f"   {'':13s}{'lanes':>7}{'final IoU':>11}{'curve IoU':>11}{'curve > 0.5':>13}{'resid px':>10}")
    tab = {}
    for k in LANE_CLASSES:
        L2 = [ln for ln in lanes if ln["cls"] == k and "p_iou_curve" in ln]
        if not L2:
            continue
        t = dict(n=len(L2), final=_med(ln["p_iou_final"] for ln in L2), curve=_med(ln["p_iou_curve"] for ln in L2),
                 curve_ok=_share(ln["p_iou_curve"] > HIT for ln in L2), res=_med(ln["p_res"] for ln in L2))
        tab[k] = t
        say(f"   {k:13s}{t['n']:7d}{num(t['final'], 11, 3)}{num(t['curve'], 11, 3)}{pct(t['curve_ok'], 13)}"
            f"{num(t['res'], 10, 1)}")
    if "hit" in tab:
        key["hits correct without residual"] = tab["hit"]["curve_ok"]
    say("   curve IoU: the same prediction with its residual zeroed (start and length kept); F1 of the zeroed "
        "lanes in section 1b")
    say("")
    return tab


def sec_probes(results, records, P, L, meta, train, say, key, jobs, thresholds, top_k, seed):
    num_fc = int(meta["trace"].get("num_fc", 2))
    say("7. feature probes: can the head's own features rank correct lanes better than its score does?")
    say("   cross-fitted: fit on one half of the images, score the other half (and the reverse); "
        "label 'eval' = eval IoU > 0.5 with some GT lane")
    say("   P1 linear on the classifier's input feature (the head's own last layer is linear on it), eval label")
    if train is not None:
        say("   P2 the same with the training assigner's label (positives among the stored predictions)")
    say(f"   P3 an MLP of the cls tower's shape ({num_fc} x Linear+ReLU, then Linear) on the RoI feature (the tower "
        f"input), eval label")
    say("   P4 P1 plus geometry the head could know (start, length, extent, slant, residual, move, previous "
        "logit, samples off image)")
    say("   P5 P1 on the classifier feature of the rescore pass (RoI on the output lane)")
    has_fresh = "feat_fresh" in records[0]
    if has_fresh:
        say("   P6 P1 on the classifier feature of the fresh pass (every RoI slot on the output lane)")
    n_img = len(records)
    perm = np.random.default_rng(seed).permutation(n_img)
    fold = (perm[P["image"].astype(np.int64)] % 2) if len(P["image"]) else np.zeros(0, np.int64)
    y = P["correct_final"].astype(np.float32)
    F = np.concatenate([r["feat"] for r in records]).astype(np.float32)
    Fr = np.concatenate([r["feat_rescore"] for r in records]).astype(np.float32)
    Rf = np.concatenate([r["roi"] for r in records]).astype(np.float32)
    D = geometry_descriptors(results, records, L).astype(np.float32)
    tic = time.time()
    probes = {"P1 linear cls/eval": cross_fit(F, y, fold, "linear")}
    if train is not None:
        probes["P2 linear cls/assigner"] = cross_fit(F, P["pos"].astype(np.float32), fold, "linear")
    probes["P3 MLP roi/eval"] = cross_fit(Rf, y, fold, "mlp", num_fc)
    probes["P4 linear cls+geo/eval"] = cross_fit(np.hstack([F, D]), y, fold, "linear")
    probes["P5 linear rescored/eval"] = cross_fit(Fr, y, fold, "linear")
    if has_fresh:
        Ff = np.concatenate([r["feat_fresh"] for r in records]).astype(np.float32)
        probes["P6 linear fresh/eval"] = cross_fit(Ff, y, fold, "linear")
    probes = {k: np.nan_to_num(v, nan=0.0) for k, v in probes.items()}
    print(f"probes fitted in {time.time() - tic:.0f} s", flush=True)
    rows = {"head score": P["score"], "rescore": P["rescore"], **({"fresh": P["fresh"]} if "fresh" in P else {}),
            **probes}
    cols = [("all", None)]
    for key_, title, edges, names in sd.BINS:
        if key_ == "slant":
            cols.append(("slant > 4", (key_, P["gt_slant"] > edges[-1])))
        if key_ == "extent":
            cols.append(("extent < 0.2", (key_, P["gt_extent"] < edges[0])))
        if key_ == "n_points":
            cols.append(("<= 2 points", (key_, P["gt_npts"] <= edges[0])))
    neg = ~P["correct_final"]
    say("   AUC (eval-correct copies of the lanes in the column vs every prediction correct for no lane)")
    say(f"   {'':26s}" + "".join(f"{c:>14s}" for c, _ in cols) + (f"{'vs assigner':>13s}" if train else ""))
    aucs = {}
    for name, sc in rows.items():
        a = {}
        for cname, sel in cols:
            if sel is None:
                a[cname] = auc(sc, P["correct_final"])
            else:
                pos = P["correct_final"] & sel[1]
                m = pos | neg
                a[cname] = auc(sc[m], pos[m])
        if train is not None:
            a["vs assigner"] = auc(sc, P["pos"])
        aucs[name] = a
        say(f"   {name:26s}" + "".join(f"{a[c]:14.3f}" for c, _ in cols)
            + (f"{a['vs assigner']:13.3f}" if train else ""))
        key[f"AUC {name}"] = a["all"]
    say("   'vs assigner': AUC for the training assigner's label on the same predictions (how well each score "
        "follows the training label)")
    say("")
    split = np.cumsum([len(r["score"]) for r in records])[:-1]
    per_img = {k: np.split(v.astype(np.float64), split) for k, v in probes.items()}
    _JOB.update(results=results, records=records, probes=per_img, thresholds=thresholds, top_k=top_k)
    sw = run_jobs(probe_sweep_job, n_img, jobs)
    f1 = {}
    say("   F1 with each probe as the score (best threshold of the sweep)")
    say(f"   {'':26s}{'NMS thr':>8}{'F1@50':>8}{'F1@75':>8}{'free thr':>10}{'F1@50':>8}{'F1@75':>8}")
    for name in probes:
        tab = agg_sweep(sw, lambda c, n=name: c[n], thresholds, sum(c["n_gt"] for c in results))
        b = {m: best_row(tab[m]) for m in MODES}
        f1[name] = b
        say(f"   {name:26s}{b['nms']['thr']:8.2f}{100 * b['nms']['F1_50']:8.2f}{100 * b['nms']['F1_75']:8.2f}"
            f"{b['free']['thr']:10.2f}{100 * b['free']['F1_50']:8.2f}{100 * b['free']['F1_75']:8.2f}")
        key[f"F1 nms: probe {name}"] = b["nms"]["F1_50"]
    say("   without NMS the eval-label probes keep duplicates (they label every correct copy positive); compare "
        "with NMS")
    say("")
    return dict(auc=aucs, f1=f1), per_img


def write_lane_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else
                            "" if isinstance(v, float) else int(v) if isinstance(v, (bool, np.bool_)) else v)
                        for k, v in r.items()})


def compare_traces(paths, say):
    runs = [json.load(open(p)) for p in paths]
    keys = []
    for r in runs:
        for k in r["key"]:
            if k not in keys:
                keys.append(k)
    width = max(12, max(len(r["label"]) for r in runs) + 2)
    say("=" * 100)
    say("stage trace comparison")
    for r, p in zip(runs, paths):
        say(f"  {r['label']}: {p}")
    say(f"  {'':58s}" + "".join(f"{r['label'][:width - 2]:>{width}s}" for r in runs))
    for k in keys:
        cells = []
        for r in runs:
            v = r["key"].get(k)
            cells.append(f"{'-':>{width}}" if v is None else f"{v:{width}.3f}")
        say(f"  {k[:58]:58s}" + "".join(cells))


# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--mode", choices=["nms", "free"], default="nms", help="the output the lane tables use")
    ap.add_argument("--thr", type=float, default=None, help="operating threshold; default: best F1@50 of the sweep")
    ap.add_argument("--split", choices=["test", "val", "train"], default="test",
                    help="test = the config's test_dataloader (CurveLanes valid); train = the training list "
                         "with the test pipeline")
    ap.add_argument("--subset", type=int, default=None, help="a random subset of this many images (--seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--max-per-image", type=int, default=64)
    ap.add_argument("--save-dump", default=None)
    ap.add_argument("--load-dump", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--label", default=None, help="name of this run in --compare")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--verify-batches", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=None, help="quick look only")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tol", type=float, default=None,
                    help="on-lane distance of a RoI sample, network px; default: the last stage's feature stride")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--assigner", default=None, help="training assigner JSON (see curvelanes_score_diag.py)")
    ap.add_argument("--iou-shape", type=int, nargs=2, default=None, metavar=("H", "W"))
    ap.add_argument("--lane-width", type=float, default=None)
    ap.add_argument("--lane-width-cost", type=float, default=None)
    ap.add_argument("--compare", nargs="+", default=None, metavar="TRACE_JSON")
    args = ap.parse_args(argv)

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    if args.compare:
        compare_traces(args.compare, say)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            with open(osp.join(args.out_dir, "comparison.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        return None
    if not args.config:
        ap.error("CONFIG is required (or --compare)")

    from mmengine.config import Config
    if args.load_dump:
        cfg = Config.fromfile(args.config, import_custom_modules=False)
        with open(args.load_dump, "rb") as f:
            dump = pickle.load(f)
        records, meta = dump["records"], dump["meta"]
        vinfo, checkpoint = meta["verify"], meta.get("checkpoint")
        if "trace" not in meta:
            raise SystemExit("this dump has no stage trace (made by curvelanes_error_breakdown.py?); "
                             "run this tool with the checkpoint")
    else:
        if not args.checkpoint:
            ap.error("a checkpoint is needed unless --load-dump is given")
        cfg = Config.fromfile(args.config)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        records, meta, vinfo = run_trace(cfg, args, device)
        checkpoint = args.checkpoint
        if args.save_dump:
            os.makedirs(osp.dirname(osp.abspath(args.save_dump)), exist_ok=True)
            with open(args.save_dump, "wb") as f:
                pickle.dump(dict(records=records, meta=dict(
                    meta, config=args.config, checkpoint=checkpoint, split=args.split, floor=args.floor,
                    max_per_image=args.max_per_image, verify=vinfo)), f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saved {args.save_dump}")
        meta = dict(meta, floor=args.floor, max_per_image=args.max_per_image, verify=vinfo)
    if args.max_images is not None:
        records = records[:args.max_images]
    tm = meta["trace"]
    L = int(tm["refine_layers"])
    if L < 2:
        raise SystemExit("stage trace needs refine_layers >= 2")
    st = eb.settings(cfg, meta, args)
    tc = meta["test_cfg"]
    cfg_thr, nms_thres, top_k = float(tc["conf_threshold"]), float(tc["nms_thres"]), int(tc["nms_topk"])
    floor = float(meta.get("floor", 0.0))
    grid = set(np.round(np.arange(0.05, 0.951, 0.05), 2).tolist()) | {0.96, 0.97, 0.98, 0.99, cfg_thr}
    if args.thr is not None:
        grid.add(args.thr)
    thresholds = sorted(t for t in grid if t >= floor - 1e-12)
    train = sd.training_setup(cfg, meta, args)
    strides = tm.get("strides") or [8.0]
    tol = float(args.tol if args.tol is not None else strides[-1])
    has_fresh = bool(tm.get("fresh")) and bool(records) and "fresh" in records[0]
    combos = combos_for(L, has_fresh)
    state = dict(sd=dict(eb=dict(st=st, decoder=eb.make_decoder(meta), nms_thres=nms_thres, top_k=top_k,
                                 thresholds=thresholds), train=train),
                 trace=dict(L=L, sample_rows=np.asarray(tm["sample_x_indices"], dtype=np.int64), tol=tol,
                            combos=combos))
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    tic = time.time()
    if jobs == 1:
        _init(state)
        results = [analyse(r) for r in records]
    else:
        with mp.get_context("fork").Pool(jobs, initializer=_init, initargs=(state,)) as pool:
            results = list(pool.imap(analyse, records, chunksize=16))
    print(f"analysed {len(results)} images in {time.time() - tic:.0f} s", flush=True)
    _init(state)                                   # the main process decodes too (anchor groups)

    n_gt = sum(c["n_gt"] for c in results)
    base = eb.sweep_table(results, thresholds)
    sweeps = {"final lanes, final score": base}
    for name, _, _ in combos:
        sweeps[name] = agg_sweep(results, lambda c, n=name: c["t_sweeps"][n], thresholds, n_gt)
    ops = {}
    for mode in MODES:
        ops[mode] = args.thr if (args.thr is not None and mode == args.mode) else best_row(base[mode])["thr"]
    thr, mode = ops[args.mode], args.mode
    ctx = dict(top_k=top_k, nms_thres=nms_thres, n_strips=meta["n_strips"], img_w=meta["img_w"])
    lanes = {m: lane_trace_rows(results, records, ops[m], m, ctx, L) for m in MODES}
    P = pred_table(results, records, L)
    sw = [r for r in base[mode] if abs(r["thr"] - thr) < 1e-9][0]

    key = {}
    say("=" * 100)
    say(f"CurveLanes stage trace  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"  config: {args.config}")
    say(f"  checkpoint: {checkpoint}; head {meta.get('head_cls', '?')} ({tm['kind']}), {L} stages"
        + ("; lane decoder" if tm.get("decoder") else "") + ("; learned NMS scores" if tm.get("dedup") else ""))
    say(f"  split {tm.get('split')}" + (f" (random subset of {tm['subset']}, seed {tm['seed']})" if tm.get("subset")
                                         else "") + f"; images {len(records)}; stored predictions per image: the "
        f"{meta.get('max_per_image')} highest by final score")
    ck = tm.get("checks", {})
    say(f"  checks: get_lanes {json.dumps(vinfo)}; stage inputs == pooled samples to {ck.get('sample_err', NAN):.1e}; "
        f"rescore pass: earlier stages unchanged to {ck.get('rescore_stage_err', NAN):.1e}"
        + (f"; fresh pass: earlier stages unchanged to {ck.get('fresh_stage_err', NAN):.1e}" if has_fresh else ""))
    say(f"  RoI samples: {len(tm['sample_x_indices'])} rows; feature strides by stage "
        f"{', '.join(f'{s:g}' for s in strides)} px; on-lane tolerance {tol:g} px")
    say(f"  operating point: {eb.MODE_LABEL[mode]}, threshold {thr:.2f}: P {100 * sw['P']:.2f}  R {100 * sw['R']:.2f}  "
        f"F1@50 {100 * sw['F1_50']:.2f}  F1@75 {100 * sw['F1_75']:.2f}")
    if train is not None:
        say(f"  training labels: {json.dumps(train['assigner'])} ({train['source']}), on the stored predictions only"
            + ("; also with cls_weight 0 (section 4b)" if "cls_weight" in train["assigner"] else ""))
    say("")
    key["F1 operating point"] = sw["F1_50"]
    out = dict(label=args.label or osp.splitext(osp.basename(args.config))[0], config=args.config,
               checkpoint=checkpoint, mode=mode, thr=thr)
    out["cascade_cover"], out["cascade_f1"] = sec_cascade(lanes[mode], sweeps, L, say, key)
    out["cascade_f1"] = {f"{k[0]} | {k[1]}": v for k, v in out["cascade_f1"].items()}
    out["input_output"], out["auc_matrix"] = sec_input_output(P, thr, L, say, key)
    out["lanes"] = sec_classes(lanes[mode], thr, L, say, key)
    G = len(lanes[mode])
    for k in ("hedged", "dismissed", "mislocalized"):
        key[f"{k} % GT"] = sum(ln["cls"] == k for ln in lanes[mode]) / max(G, 1)

    # 4. alternatives, lane by lane
    alt_specs = [("rescore", "final", "rescore")] + ([("fresh", "final", "fresh")] if has_fresh else []) + [
        ("final curve", "curve", f"s{L - 1}"), (f"stage {L - 2} alone", f"out{L - 2}", f"s{L - 2}"),
        (f"final + s{L - 2} score", "final", f"s{L - 2}")]
    name_of = {"rescore": "final lanes, rescore (RoI on the output)", "fresh": FRESH_NAME,
               "final curve": "final curve, residual zeroed",
               f"stage {L - 2} alone": f"stage {L - 2} alone (its lanes, its score)",
               f"final + s{L - 2} score": f"final lanes, stage {L - 2} score"}
    alts = [(a, g, s, best_row(sweeps[name_of[a]][mode])["thr"], mode) for a, g, s in alt_specs]
    _JOB.clear()
    _JOB.update(results=results, records=records, alts=alts, top_k=top_k)
    hits = run_jobs(lane_hits_job, len(records), jobs)
    hit_arr = {a[0]: np.concatenate([h[a[0]] for h in hits]) if hits else np.zeros(0, bool) for a in alts}
    out["alternatives"] = sec_alternatives(
        lanes[mode], hit_arr, [a[0] for a in alts], say,
        f"4. what each alternative recovers and loses, lane by lane ({eb.MODE_LABEL[mode]})")
    out["ownership"] = sec_ownership(lanes[mode], hit_arr, thr, L, say, key)
    out["residual"] = sec_residual(lanes[mode], say, key)
    out["coverage"] = sec_coverage(lanes[mode], P, thr, tol, say, key)
    if not args.no_probes and len(P["score"]):
        out["probes"], per_img = sec_probes(results, records, P, L, meta, train, say, key, jobs, thresholds, top_k,
                                            args.seed)
        palts = [(k, "final", per_img[k], out["probes"]["f1"][k]["nms"]["thr"], "nms") for k in per_img]
        _JOB.clear()
        _JOB.update(results=results, records=records, alts=palts, top_k=top_k)
        ph = run_jobs(lane_hits_job, len(records), jobs)
        parr = {k: np.concatenate([h[k] for h in ph]) for k in per_img}
        out["probe_lanes"] = sec_alternatives(
            lanes["nms"], parr, list(per_img), say,
            "   probes lane by lane (with NMS, against the head's own NMS operating point)")
    out["calibration"] = sec_calibration(P, say)
    out["anchors"] = sec_anchors(P, anchor_groups(records, eb._W["decoder"], st), thr, say)
    reading(say)
    out["key"] = key

    out_dir = args.out_dir or osp.join("work_dirs", "curvelanes_stage_trace",
                                       osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "trace.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(out_dir, "trace.json"), "w") as f:
        json.dump(jsafe(out), f, indent=1)
    write_lane_csv(osp.join(out_dir, "trace_lanes.csv"), lanes[mode])
    np.savez_compressed(osp.join(out_dir, "trace_preds.npz"),
                        **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in P.items()})
    print(f"wrote {out_dir}/trace.txt, trace.json, trace_lanes.csv, trace_preds.npz")
    return dict(results=results, records=records, lanes=lanes, P=P, out=out, sweeps=sweeps, ops=ops)


if __name__ == "__main__":
    main()
