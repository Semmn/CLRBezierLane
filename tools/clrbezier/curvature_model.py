"""Model-side half of ``test_curvature.py``: does the model use curvature, and
does the curvature it predicts change F1?

Split out so the ground-truth analysis runs with numpy and scipy alone — no
torch, no mmdet, no GPU — and can be run while the GPUs are busy training.

Two measurements:

``run_model``   How far the refined control points depart from a straight line,
                per refinement stage and per CULane category. ``forward_test``
                throws the control points away (it keeps only the sampled xs),
                so this calls ``head._refine`` directly, which returns the BRR
                states whose ``[..., 2:6]`` slice is the control-point x.

``run_ablate``  The causal test. Project every predicted curve onto its own
                least-squares straight line over its valid rows, keep the score
                and the extent, decode and re-evaluate with the official metric.
                If F1 does not move, the curvature the model predicts is not
                paying for itself, whatever its magnitude. Requires no
                retraining and no GT beyond the usual evaluation.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.registry import MODELS

from libs.datasets.metrics.culane_metric import CULaneMetric, load_categories

from fast_sweep import run_sweep
from test_curvature import WIDTH, chord_deviation


# --------------------------------------------------------------------------


def build(args):
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    cfg.model.setdefault("test_cfg", {})
    cfg.model["test_cfg"]["conf_threshold"] = args.dump_threshold
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    loader_cfg = copy.deepcopy(
        cfg.test_dataloader if args.split == "test" else cfg.val_dataloader)
    loader = Runner.build_dataloader(loader_cfg)
    ev = cfg.test_evaluator if args.split == "test" else cfg.val_evaluator
    return cfg, model, loader, ev["data_root"], ev["data_list"]


def category_lookup(data_root):
    try:
        data_cats, categories = load_categories(
            os.path.join(data_root, "list/test_split/"))
    except FileNotFoundError:
        return {}, []
    return {k.lstrip("/"): v for k, v in data_cats.items()}, categories


# --------------------------------------------------------------------------
# does the model use curvature?


@torch.no_grad()
def run_model(args):
    cfg, model, loader, data_root, data_list = build(args)
    head = model.bbox_head
    cats, categories = category_lookup(data_root)
    topk = int(head.test_cfg.get("nms_topk", 4))
    print(f"measuring the top-{topk} queries per image, the set that roughly "
          f"survives NMS\n")

    rows = []     # (stage, category, deviation in px)
    seen = 0
    for data in loader:
        data = model.data_preprocessor(data, False)
        feats = head._select_features(model.extract_feat(data["inputs"]))
        preds, states, _ = head._refine(
            feats, head.prior_bank(feats[-1].shape[0]), head.gsrc_tokens(feats))
        for stage, (pred, state) in enumerate(zip(preds, states)):
            score = torch.softmax(pred[..., :2].float(), -1)[..., 1]
            k = min(topk, score.shape[-1])
            idx = score.topk(k, dim=-1).indices                        # [B, k]
            cp = torch.gather(state[..., 2:6], 1,
                              idx[..., None].expand(*idx.shape, 4))    # [B, k, 4]
            dev = chord_deviation(cp.float().cpu().numpy()) * head.img_w
            for b, sample in enumerate(data["data_samples"]):
                cat = cats.get(str(sample.metainfo["sub_img_name"]).lstrip("/"),
                               "test0_normal")
                rows.extend((stage, cat, float(v)) for v in dev[b])
        seen += len(data["data_samples"])
        if seen % 2000 < len(data["data_samples"]):
            print(f"  {seen} images", flush=True)
        if args.max_images is not None and seen >= args.max_images:
            break

    report_model(rows, categories, head.img_w)
    return 0


# The bow at which a straight-line fit stops reaching a lane is NOT a single
# number -- it depends on the IoU level being scored, and strongly. Parallel
# lanes of width W cross IoU level I at a mean lateral offset of W(1-I)/(1+I),
# and a parabolic bow of B leaves a mean residual of ~0.274 B after the
# least-squares line splits it (0.2566 in closed form; the difference is discrete
# row sampling). So:
#
#     IoU 0.50  ->  mean offset 10.0 px  ->  bow ~36 px
#     IoU 0.75  ->  mean offset  4.3 px  ->  bow ~16 px
#
# Reporting only the 0.50 threshold makes every model look straight, which is
# how an earlier version of this file reached a "curvature is decorative"
# verdict on a model whose curve-subset p90 clears the 0.75 threshold twice over.
RESIDUAL_PER_BOW = 0.274


def bow_threshold(iou_level: float, width: float = WIDTH) -> float:
    return width * (1.0 - iou_level) / (1.0 + iou_level) / RESIDUAL_PER_BOW


def report_model(rows, categories, img_w):
    arr = np.array([(s, d) for s, _, d in rows])
    stages = sorted({int(s) for s, _, _ in rows})
    levels = [(lvl, bow_threshold(lvl)) for lvl in (0.5, 0.75)]
    cols = "".join(f"{f'>{t:.0f}px':>9}" for _, t in levels)

    print(f"\ndeviation of the refined control points from a straight line, in "
          f"pixels at img_w={img_w}.")
    for lvl, thr in levels:
        print(f"  a straight fit stops reaching a lane at a bow of ~{thr:.0f} px "
              f"when scoring at IoU {lvl}")
    print()
    print(f"{'stage':>6} {'queries':>8} {'p50':>7} {'p90':>7} {'p99':>7} "
          f"{'max':>7}{cols}")
    print("-" * (43 + 9 * len(levels)))
    for stage in stages:
        d = arr[arr[:, 0] == stage][:, 1]
        frac = "".join(f"{(d > t).mean() * 100:8.2f}%" for _, t in levels)
        print(f"{stage:6d} {len(d):8d} {np.percentile(d, 50):7.2f} "
              f"{np.percentile(d, 90):7.2f} {np.percentile(d, 99):7.2f} "
              f"{d.max():7.2f}{frac}")

    last = max(stages)
    first = min(stages)
    print(f"\nby category at the final stage:")
    print(f"{'category':<14} {'queries':>8} {'p50':>7} {'p90':>7} {'max':>7}{cols}")
    print("-" * (46 + 9 * len(levels)))
    by_cat = {}
    for cat in categories or sorted({c for _, c, _ in rows}):
        d = np.array([v for s, c, v in rows if s == last and c == cat])
        if not len(d):
            continue
        by_cat[cat] = d
        frac = "".join(f"{(d > t).mean() * 100:8.2f}%" for _, t in levels)
        print(f"{cat:<14} {len(d):8d} {np.percentile(d, 50):7.2f} "
              f"{np.percentile(d, 90):7.2f} {d.max():7.2f}{frac}")

    # Is the refinement actually adding curvature, and is it aimed anywhere?
    p90_first = np.percentile(arr[arr[:, 0] == first][:, 1], 90)
    p90_last = np.percentile(arr[arr[:, 0] == last][:, 1], 90)
    print(f"\nrefinement: p90 bow grows {p90_first:.2f} -> {p90_last:.2f} px "
          f"across stages {first}..{last} ({p90_last / max(p90_first, 1e-9):.2f}x)")
    curve = by_cat.get("test6_curve")
    straight = np.concatenate([d for c, d in by_cat.items()
                               if c in ("test0_normal", "test1_crowd")]) \
        if {"test0_normal", "test1_crowd"} & set(by_cat) else None
    if curve is not None and straight is not None:
        ratio = np.median(curve) / max(np.median(straight), 1e-9)
        print(f"targeting: median bow is {ratio:.1f}x larger on test6_curve than "
              f"on normal/crowd")

    final = arr[arr[:, 0] == last][:, 1]
    print()
    for lvl, thr in levels:
        print(f"at IoU {lvl}: {(final > thr).mean() * 100:.2f}% of surviving "
              f"predictions carry curvature the metric could notice")

    visible = float((final > bow_threshold(0.75)).mean())
    if visible < 0.001:
        print("\nVERDICT: the predicted curves are straight even at IoU 0.75. The "
              "per-row offsets are doing all the shape work and the Bezier "
              "parameterization is not where your F1 is.")
    elif float((final > bow_threshold(0.5)).mean()) < 0.001:
        print("\nVERDICT: the model predicts curvature that is invisible at IoU "
              "0.50 but material at IoU 0.75. That is the regime where a curve "
              "anchor can pay, and F1@50 alone will never show it -- run "
              "--mode ablate and read F1@75, per category.")
    else:
        print("\nVERDICT: the model predicts curvature at a scale that matters "
              "even at IoU 0.50. --mode ablate says whether it helps or hurts.")


# --------------------------------------------------------------------------
# does the curvature change F1?


def straighten(xs: torch.Tensor, prior_ys: torch.Tensor) -> torch.Tensor:
    """Least-squares straight line through each lane's valid rows.

    ``xs`` is ``[K, R]`` normalized lateral position, ``prior_ys`` ``[R]``. Rows
    outside [0, 1] are invalid and excluded from the fit, matching how the
    decoder reads the prediction. A lane with fewer than two valid rows is
    returned unchanged -- there is no line to fit, and forcing one would inject
    an error the curvature question has nothing to do with.
    """
    valid = ((xs >= 0.0) & (xs <= 1.0)).to(xs.dtype)
    y = prior_ys.to(xs.device, xs.dtype)[None, :].expand_as(xs)
    n = valid.sum(-1, keepdim=True)
    my = (valid * y).sum(-1, keepdim=True) / n.clamp_min(1.0)
    mx = (valid * xs).sum(-1, keepdim=True) / n.clamp_min(1.0)
    dy, dx = y - my, xs - mx
    syy = (valid * dy * dy).sum(-1, keepdim=True)
    sxy = (valid * dy * dx).sum(-1, keepdim=True)
    slope = sxy / syy.clamp_min(1e-12)
    line = mx + slope * dy
    return torch.where(n >= 2.0, line, xs)


def nms_keep(head, pred, b, keep_conf, scores):
    """The post-NMS indices, computed exactly as ``get_lanes`` computes them.

    This block is the reason an earlier version of this file was wrong by 29 F1.
    ``get_lanes`` filters by confidence, runs **lane NMS**, and only then calls
    ``predictions_to_lanes``; going straight from the confidence filter to the
    decoder dumps every near-duplicate among the 192 anchors as its own
    prediction, which destroys precision and drags the apparent optimum up to a
    threshold that also costs recall.

    The indices are computed once and shared by both arms, so straightening
    cannot change *which* lanes are kept -- only their geometry. That separation
    is the whole point: otherwise the ablation would conflate "curvature helps
    localization" with "curvature changes what NMS selects".
    """
    if not head.test_cfg.get("use_nms", True):
        return torch.arange(int(keep_conf.sum()), device=scores.device)
    from nms import nms
    anchor_params = pred["anchor_params"][b][keep_conf]
    lengths = pred["lengths"][b][keep_conf]
    xs = pred["xs"][b][keep_conf]
    nms_ap = anchor_params[..., :2].detach().clone()
    nms_ap[..., 0] = 1 - nms_ap[..., 0]
    nms_predictions = torch.cat([
        pred["cls_logits"][b, keep_conf].detach().clone(),
        nms_ap[..., :2],
        lengths.detach().clone() * head.n_strips,
        xs.detach().clone() * (head.img_w - 1),
    ], dim=-1)
    keep, num_to_keep, _ = nms(nms_predictions, scores,
                               overlap=head.test_cfg.nms_thres,
                               top_k=head.test_cfg.nms_topk)
    return keep[:num_to_keep]


@torch.no_grad()
def run_ablate(args):
    cfg, model, loader, data_root, data_list = build(args)
    head = model.bbox_head
    categories_dir = os.path.join(data_root, "list/test_split/")
    print("decoding twice from one NMS selection: the model's curves, and the "
          "same kept curves projected onto their own best straight line\n")

    dumps = {"predicted": [], "straightened": []}
    seen = 0
    for data in loader:
        data = model.data_preprocessor(data, False)
        outs = head(model.extract_feat(data["inputs"]))
        pred = outs[-1] if isinstance(outs, (list, tuple)) else outs
        prob = torch.softmax(pred["cls_logits"].float(), -1)[..., 1]

        for b, sample in enumerate(data["data_samples"]):
            name = sample.metainfo["sub_img_name"]
            keep_conf = prob[b] >= args.dump_threshold
            if not bool(keep_conf.any()):
                for arm in dumps:
                    dumps[arm].append((name, [], np.zeros(0)))
                continue
            sc_all = prob[b][keep_conf]
            idx = nms_keep(head, pred, b, keep_conf, sc_all)
            xs = pred["xs"][b][keep_conf][idx].float()
            sc = sc_all[idx]
            ap = pred["anchor_params"][b][keep_conf][idx].float()
            ln = pred["lengths"][b][keep_conf][idx].float()
            for arm, lane_xs in (("predicted", xs),
                                 ("straightened", straighten(xs, head.prior_ys))):
                lanes = head.predictions_to_lanes(
                    lane_xs, ap, torch.round(ln * head.n_strips), sc,
                    head.test_cfg.as_lanes, head.test_cfg.extend_bottom)
                dumps[arm].append(
                    (name, lanes, sc.detach().cpu().numpy().astype(np.float64)))
        seen += len(data["data_samples"])
        if seen % 2000 < len(data["data_samples"]):
            print(f"  {seen} images", flush=True)
        if args.max_images is not None and seen >= args.max_images:
            break

    per_image = sum(len(s) for _, _, s in dumps["predicted"]) / max(1, len(dumps["predicted"]))
    print(f"\n{per_image:.2f} lanes per image after NMS "
          f"(nms_topk={head.test_cfg.nms_topk}); if this is far above topk the "
          f"NMS step is not running and the numbers below are meaningless")

    start, stop, step = args.grid
    thresholds = [t for t in np.round(np.arange(start, stop + 1e-9, step), 4)
                  if t >= args.dump_threshold]
    metric = CULaneMetric(data_root=data_root, data_list=data_list)
    rows, best, records = run_sweep(
        dumps, thresholds, data_root, data_list, categories_dir, metric,
        jobs=args.jobs, verify=not getattr(args, "no_verify", False),
        logger=MMLogger.get_current_instance(),
        rgb_masks=getattr(args, "rgb_masks", False), return_records=True)

    # Both IoU levels, side by side. F1@50 is the headline CULane number and the
    # one least able to see curvature; F1@75 is where a curve representation has
    # roughly five times the headroom (see --mode gt).
    print(f"\n{'':>6} {'----- F1@50 -----':>26} {'----- F1@75 -----':>26}")
    print(f"{'conf':>6} {'pred':>8} {'straight':>9} {'delta':>7} "
          f"{'pred':>8} {'straight':>9} {'delta':>7}")
    print("-" * 60)
    for i, t in enumerate(thresholds):
        cells = []
        for key in ("F1_0.5", "F1_0.75"):
            a = rows["predicted"][i][key] * 100
            b = rows["straightened"][i][key] * 100
            cells.append(f"{a:8.2f} {b:9.2f} {a - b:+7.2f}")
        print(f"{t:6.2f} " + " ".join(cells))

    # GUARD. The `predicted` arm applies no modification, so its peak F1@50 must
    # reproduce the model's own evaluation. If it does not, the decode path is
    # broken and every delta below is one broken decode against another -- which
    # the sweep's own verification will NOT catch, because that check confirms
    # the metric arithmetic against eval_predictions while the arithmetic
    # faithfully scores wrong predictions.
    peak = max(r["F1_0.5"] for r in rows["predicted"]) * 100
    if args.expect_f1 is not None and abs(peak - args.expect_f1) > args.expect_tol:
        print(f"\nABORT: the unmodified arm peaks at F1@50 {peak:.2f}, but you "
              f"said this model scores {args.expect_f1:.2f} "
              f"(tolerance {args.expect_tol:.2f}).")
        print("The decode path is not reproducing your evaluation, so the "
              "straightening delta is meaningless. Check, in order: that NMS ran "
              "(the lanes-per-image line above), that the config's test_cfg "
              "matches the one you evaluate with, and that the checkpoint loaded "
              "without missing keys.")
        return 1
    if args.expect_f1 is None:
        print(f"\nthe unmodified arm peaks at F1@50 {peak:.2f}. Confirm this "
              f"matches your model's known score before reading anything below; "
              f"pass --expect-f1 to make that check automatic.")

    print()
    verdicts = {}
    for key, label in (("F1_0.5", "F1@50"), ("F1_0.75", "F1@75")):
        pk = max(rows["predicted"], key=lambda r: r[key])
        sk = max(rows["straightened"], key=lambda r: r[key])
        gap = (pk[key] - sk[key]) * 100
        verdicts[label] = gap
        print(f"{label}: predicted {pk[key] * 100:.2f} at conf "
              f"{pk['threshold']:.2f}, straightened {sk[key] * 100:.2f} at conf "
              f"{sk['threshold']:.2f}  ->  curvature worth {gap:+.2f}")

    # Where does the curvature pay? Re-aggregate the cached matrices by category
    # at each arm's own peak threshold -- no extra rasterization. The GT analysis
    # puts most of the IoU-0.75 headroom in test6_curve, on ~1% of lanes, so a
    # gain that is NOT concentrated there means something other than real road
    # curvature is being rewarded and is worth understanding before trusting it.
    from fast_sweep import sweep as resweep
    print(f"\nwhere the curvature pays, at each arm's own peak threshold:")
    print(f"{'category':<14} {'share':>7} {'pred@75':>9} {'straight@75':>12} "
          f"{'delta':>7}")
    print("-" * 54)
    peaks = {arm: max(rows[arm], key=lambda r: r["F1_0.75"])["threshold"]
             for arm in dumps}
    cat_rows = {arm: resweep(records[arm], [peaks[arm]], per_category=True)[0]
                for arm in dumps}
    total = len(records["predicted"])
    cats = sorted({r["cat"] for r in records["predicted"]})
    contributions = []
    for cat in cats:
        share = sum(1 for r in records["predicted"] if r["cat"] == cat) / max(total, 1)
        a = cat_rows["predicted"].get(f"F1_{cat}_0.75")
        b = cat_rows["straightened"].get(f"F1_{cat}_0.75")
        if a is None or b is None:
            continue
        contributions.append(((a - b) * 100, cat, share))
        print(f"{cat:<14} {share * 100:6.2f}% {a * 100:9.2f} {b * 100:12.2f} "
              f"{(a - b) * 100:+7.2f}")
    if not contributions:
        have = sorted(k for k in cat_rows["predicted"] if k.startswith("F1_test"))
        print(f"  (no rows: sweep() returned no per-category keys at IoU 0.75. "
              f"Keys present: {have[:4]}{'...' if len(have) > 4 else ''})")
    else:
        contributions.sort(reverse=True)
        top = contributions[0]
        print(f"\nlargest per-category gain: {top[1]} {top[0]:+.2f} on "
              f"{top[2] * 100:.2f}% of images")
        curve = next((c for c in contributions if c[1] == "test6_curve"), None)
        if curve is not None:
            weighted = curve[0] * curve[2]
            print(f"test6_curve contributes {weighted:+.2f} of the overall "
                  f"{verdicts['F1@75']:+.2f} ({100 * weighted / verdicts['F1@75']:.0f}%)"
                  if verdicts["F1@75"] else "")

    hi = verdicts["F1@75"]
    lo = verdicts["F1@50"]
    print()
    if abs(lo) < 0.1 and abs(hi) < 0.1:
        print("Curvature is worth nothing at either level. Every curve the model\n"
              "predicts could be its own straight-line fit at no cost, so the\n"
              "Bezier parameterization is not where your remaining F1 is.")
    elif abs(lo) < 0.1 <= hi:
        print(f"Invisible at F1@50 ({lo:+.2f}) and real at F1@75 ({hi:+.2f}).\n"
              "This is the expected signature of a curve representation on a\n"
              "straight-dominated dataset: the headline metric cannot resolve the\n"
              "thing your anchor was built to model. Report F1@75 alongside F1@50\n"
              "and break out test6_curve -- that is where the design earns its keep.")
    elif min(lo, hi) < -0.1:
        print("Straightening *improves* F1. The curve freedom is actively costing\n"
              "you, which is the case for a bending penalty or for restricting the\n"
              "cubic to the final refinement stage.")
    else:
        print("Curvature pays at both levels. Protect it with a bending penalty\n"
              "rather than constraining it away.")
    return 0
