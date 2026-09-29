"""Is the localization-quality information in the features at all?

Epistemic vs aleatoric
----------------------
The oracle experiment says a perfect scorer would reach ~98 F1 where the model
reaches ~80. Either the features contain enough to rank a good lane above a bad
one and the head fails to extract it (epistemic — fixable by a better scoring
objective), or the evidence is not there at all (aleatoric — a lane behind a
bus, a marking too faint to see, an annotation the labeller extrapolated from
road context; no head recovers that).

This freezes the trained model, captures the exact vector the classifier sees
for every prediction, and trains a deliberately over-parameterized probe on it
with nothing to do but this one task. If the probe cannot out-rank the model's
own score, the information is not there.

Two things this version fixes, both of which invalidated the first one
----------------------------------------------------------------------
1. **The probe is fitted on the training split, not on validation.** Fitting on
   val gave the probe a few thousand images from ~19 sequences while the score
   it was being compared against had been trained on 88k images. That is not a
   test of what the features contain, it is a test of sample size, and the
   probe loses it by construction.

2. **The probe is trained on the pairwise objective it is evaluated on.**
   Training it to regress absolute LaneIoU and then scoring it on within-cluster
   ordering is exactly the calibration-versus-ranking mistake this whole line of
   work is about: absolute-IoU regression spends its capacity on the easy global
   axis (lane vs background) and none on the hard local one (which of these
   near-identical duplicates is better). ``--objective absolute`` keeps the old
   behaviour for comparison.

Two controls run by default, because a probe result with no control is not
evidence:

* **shuffled** — the same probe on permuted targets. Must land at 0.5. Anything
  else means sequence leakage between fit and evaluation.
* **score-as-feature** — the probe with the model's own score appended to its
  input. Must be at least as good as the score alone. If it is worse, the probe
  is underfit and its headline number means nothing.

Usage:
    python tools/clrbezier/probe_information.py \\
        configs/clrbezier/culane/clrbezier_collab_perturb_r34.py \\
        work_dirs/.../epoch_15.pth \\
        --max-fit-images 12000 --max-eval-images 4000
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.registry import MODELS

from libs.clrbezier.lane_iou import pairwise_lane_iou


# ---------------------------------------------------------------- statistics
def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def auroc(score, label):
    label = np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(score)).astype(np.float64) + 1.0
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pair_accuracy(score, pairs):
    """Fraction of (better, worse) pairs the score orders correctly."""
    if len(pairs) == 0:
        return float("nan")
    return float((score[pairs[:, 0]] > score[pairs[:, 1]]).mean())


def bootstrap_pair_accuracy(score, pairs, image_of_pair, iters=200, seed=0):
    """CI by resampling *images*, since pairs inside one image are dependent."""
    if len(pairs) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    images = np.unique(image_of_pair)
    by_image = {img: np.flatnonzero(image_of_pair == img) for img in images}
    samples = []
    for _ in range(iters):
        picked = rng.choice(images, size=len(images), replace=True)
        idx = np.concatenate([by_image[img] for img in picked])
        samples.append(pair_accuracy(score, pairs[idx]))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ------------------------------------------------------------------ dataflow
def _with_gt_and_val_aug(loader_cfg, cfg, shuffle):
    """Deterministic augmentation *and* ground truth, from any dataloader cfg.

    The val pipeline drops the GT keys because inference does not need them, so
    this borrows the training pipeline (which packs them) and swaps its
    albumentation list for the validation one.
    """
    loader = copy.deepcopy(loader_cfg)
    train_pipeline = copy.deepcopy(cfg.train_dataloader.dataset.pipeline)
    val_al = next((t.get("pipelines") for t in cfg.val_dataloader.dataset.pipeline
                   if t.get("type") == "albumentation"), None)
    if val_al is not None:
        for transform in train_pipeline:
            if transform.get("type") == "albumentation":
                transform["pipelines"] = copy.deepcopy(val_al)
    loader.dataset.pipeline = train_pipeline
    loader.dataset.test_mode = False
    loader.sampler = dict(type="DefaultSampler", shuffle=shuffle)
    return Runner.build_dataloader(loader)


def sequence_of(filename):
    """CULane path -> the driving session it came from.

    ``.../driver_23_30frame/05151649_0422.MP4/00030.jpg`` ->
    ``driver_23_30frame/05151649_0422.MP4``. Consecutive frames share it, which
    is what must not straddle a split.
    """
    parts = os.path.normpath(str(filename)).split(os.sep)
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else str(filename)


@torch.no_grad()
def collect(model, head, loader, device, feature_key, captured, max_images, tag):
    feats, scores, quals, xs_all, groups, image_ids = [], [], [], [], [], []
    seen = 0
    for data in loader:
        data = model.data_preprocessor(data, False)
        pyramid = model.extract_feat(data["inputs"])
        outs = head(pyramid)                       # eval -> StagePredictions

        pred_xs = outs["xs"].float()
        score = torch.softmax(outs["cls_logits"].float(), dim=-1)[..., 1]
        batch, num_q = score.shape

        lanes = head.target_adapter.extract_lanes(data["data_samples"], device)
        quality = torch.zeros_like(score)
        for b in range(batch):
            target = lanes[b][lanes[b][:, 1] == 1]
            if target.shape[0] == 0:
                continue
            geo_p = pred_xs[b] * (float(head.img_w - 1) / float(head.img_w))
            geo_t = target[:, 6:] / float(head.img_w)
            iou = pairwise_lane_iou(geo_p, geo_t, head.lane_width, head.img_w, head.img_h)
            quality[b] = torch.nan_to_num(iou, nan=0.0).max(dim=1).values

        feats.append(captured[feature_key].view(batch, num_q, -1).cpu().numpy())
        scores.append(score.cpu().numpy())
        quals.append(quality.cpu().numpy())
        xs_all.append(pred_xs.cpu().numpy())
        for b in range(batch):
            name = data["data_samples"][b].metainfo.get("filename", f"{tag}{seen + b}")
            groups.extend([sequence_of(name)] * num_q)
            image_ids.extend([f"{tag}{seen + b}"] * num_q)
        seen += batch
        if seen >= max_images:
            break
        if seen % 1000 < batch:
            print(f"  [{tag}] {seen} images", flush=True)

    dim = feats[0].shape[-1]
    return dict(
        feat=np.concatenate(feats).reshape(-1, dim),
        score=np.concatenate(scores).reshape(-1),
        quality=np.concatenate(quals).reshape(-1),
        xs=np.concatenate(xs_all).reshape(-1, xs_all[0].shape[-1]),
        groups=np.array(groups),
        image_ids=np.array(image_ids),
        num_images=seen,
    )


def build_pairs(data, head, device, cluster_iou_thr, margin, max_images=None):
    """Oriented (better, worse) pairs among predictions that compete in NMS.

    A global correlation is inflated by easy lane-vs-background comparisons.
    NMS never makes those: it chooses between near-duplicates, and that is the
    only comparison that decides which lane survives.
    """
    pairs, pair_image, decisive = [], [], []
    images = np.unique(data["image_ids"])
    if max_images is not None:
        images = images[:max_images]
    for image in images:
        sel = np.flatnonzero(data["image_ids"] == image)
        if sel.size < 2:
            continue
        rows = torch.from_numpy(data["xs"][sel]).float().to(device)
        with torch.no_grad():
            mutual = pairwise_lane_iou(rows, rows, head.lane_width,
                                       head.img_w, head.img_h)
            mutual = torch.nan_to_num(mutual, nan=0.0).cpu().numpy()
        q = data["quality"][sel]
        gap = q[:, None] - q[None, :]
        competes = (mutual > cluster_iou_thr) & ~np.eye(sel.size, dtype=bool)
        better = np.argwhere(competes & (gap > margin))
        for i, j in better:
            pairs.append((sel[i], sel[j]))
            pair_image.append(image)
            # A pair only changes the metric if choosing wrongly turns a true
            # positive into a false one. Both members above 0.5 -> NMS can pick
            # either and still score a TP.
            decisive.append(bool(q[i] > 0.5 >= q[j]))
    return (np.array(pairs, dtype=np.int64).reshape(-1, 2),
            np.array(pair_image), np.array(decisive, dtype=bool))


# --------------------------------------------------------------------- probe
class Probe(nn.Module):
    """Deliberately over-parameterized: the question is whether the information
    is present, not whether a small head can find it."""

    def __init__(self, in_dim, width=512, depth=3, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_probe(fit_feat, fit_target, fit_pairs, eval_feat, device,
              objective="pairwise", epochs=30, batch=8192, lr=1e-3, tau=1.0, seed=0):
    torch.manual_seed(seed)
    x = torch.from_numpy(fit_feat).float()
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-5)
    x = ((x - mean) / std).to(device)

    model = Probe(fit_feat.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if objective == "pairwise":
        if len(fit_pairs) == 0:
            raise SystemExit("no cluster pairs in the fit split; lower --cluster-iou")
        pairs = torch.from_numpy(fit_pairs).long().to(device)
        n = pairs.shape[0]
        for _ in range(epochs):
            model.train()
            order = torch.randperm(n, device=device)
            for start in range(0, n, batch):
                idx = pairs[order[start:start + batch]]
                out = model(x[idx.reshape(-1)]).view(-1, 2)
                loss = F.softplus(-(out[:, 0] - out[:, 1]) / tau).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
    else:
        y = torch.from_numpy(fit_target).float().to(device)
        n = x.shape[0]
        for _ in range(epochs):
            model.train()
            order = torch.randperm(n, device=device)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                loss = F.binary_cross_entropy_with_logits(model(x[idx]), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

    model.eval()
    with torch.no_grad():
        xe = ((torch.from_numpy(eval_feat).float() - mean) / std).to(device)
        out = torch.cat([model(xe[i:i + 65536]) for i in range(0, xe.shape[0], 65536)])
    return out.cpu().numpy(), (model, mean, std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--max-fit-images", type=int, default=12000,
                    help="images from the TRAIN split used to fit the probe")
    ap.add_argument("--max-eval-images", type=int, default=4000,
                    help="images from the VAL split used to score it")
    ap.add_argument("--objective", choices=["pairwise", "absolute"], default="pairwise")
    ap.add_argument("--feature", choices=["roi", "tower"], default="roi")
    ap.add_argument("--cluster-iou", type=float, default=0.35)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dump", help="optional .npz of the raw arrays")
    ap.add_argument("--save-probe", help="save the fitted probe for rescore_eval.py")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(device).eval()
    head = model.bbox_head

    captured = {}

    def grab(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()
        return hook

    handles = [head.cls_modules[0].register_forward_pre_hook(grab("roi")),
               head.cls_layers.register_forward_pre_hook(grab("tower"))]

    # Fit on train (shuffled so a capped subset spans many sequences),
    # evaluate on val (ordered, deterministic).
    fit_loader = _with_gt_and_val_aug(cfg.train_dataloader, cfg, shuffle=True)
    eval_loader = _with_gt_and_val_aug(cfg.val_dataloader, cfg, shuffle=False)

    print("collecting fit split (train)...")
    fit = collect(model, head, fit_loader, device, args.feature, captured,
                  args.max_fit_images, "fit")
    print("collecting eval split (val)...")
    ev = collect(model, head, eval_loader, device, args.feature, captured,
                 args.max_eval_images, "ev")
    for handle in handles:
        handle.remove()

    overlap = set(fit["groups"]) & set(ev["groups"])
    print(f"\nfit : {fit['num_images']} images, {fit['feat'].shape[0]} predictions, "
          f"{len(set(fit['groups']))} sequences")
    print(f"eval: {ev['num_images']} images, {ev['feat'].shape[0]} predictions, "
          f"{len(set(ev['groups']))} sequences")
    print(f"sequences in both splits: {len(overlap)}")
    print(f"eval predictions with LaneIoU > 0.5: {(ev['quality'] > 0.5).mean():.1%} "
          f"({(ev['quality'] > 0.5).sum() / max(1, ev['num_images']):.1f} per image)")

    print("\nbuilding NMS clusters...")
    fit_pairs, _, _ = build_pairs(fit, head, device, args.cluster_iou, args.margin)
    eval_pairs, eval_pair_image, eval_decisive = build_pairs(
        ev, head, device, args.cluster_iou, args.margin)
    print(f"fit pairs {len(fit_pairs)}, eval pairs {len(eval_pairs)} "
          f"({eval_decisive.sum()} decisive, {eval_decisive.mean():.1%})")

    print(f"training probe ({args.objective})...")
    probe, probe_bundle = fit_probe(fit["feat"], fit["quality"], fit_pairs, ev["feat"],
                                    device, objective=args.objective, epochs=args.epochs)
    if args.save_probe:
        model_, mean_, std_ = probe_bundle
        torch.save(dict(state_dict=model_.state_dict(), mean=mean_, std=std_,
                        in_dim=fit["feat"].shape[1], feature=args.feature,
                        objective=args.objective), args.save_probe)
        print(f"probe -> {args.save_probe}")

    # Null 1: identical data, identical clusters, random *orientation*. The
    # probe cannot beat chance on this, so anything above 0.5 is leakage.
    # (The previous version trained on a symmetric pair set instead, which is
    # a contradictory objective rather than a random one, and read 0.54.)
    rng = np.random.RandomState(0)
    flipped = fit_pairs.copy()
    if len(flipped):
        flip = rng.rand(len(flipped)) < 0.5
        flipped[flip] = flipped[flip][:, ::-1]
    control_shuffled, _ = fit_probe(fit["feat"], fit["quality"], flipped, ev["feat"],
                                    device, objective=args.objective, epochs=args.epochs)

    # Null 2: no training at all. A randomly initialized network is still a
    # function of the features, so this measures how much ordering the feature
    # geometry gives away for free. It is the floor every other number should
    # be read against — not 0.5.
    torch.manual_seed(1234)
    untrained = Probe(ev["feat"].shape[1]).to(device).eval()
    with torch.no_grad():
        xe = torch.from_numpy(ev["feat"]).float()
        xe = ((xe - xe.mean(0, keepdim=True)) / xe.std(0, keepdim=True).clamp_min(1e-5)).to(device)
        control_random = torch.cat([untrained(xe[i:i + 65536])
                                    for i in range(0, xe.shape[0], 65536)]).cpu().numpy()

    fit_plus = np.concatenate([fit["feat"], fit["score"][:, None]], axis=1)
    ev_plus = np.concatenate([ev["feat"], ev["score"][:, None]], axis=1)
    control_score, _ = fit_probe(fit_plus, fit["quality"], fit_pairs, ev_plus, device,
                                 objective=args.objective, epochs=args.epochs)

    if args.dump:
        np.savez_compressed(args.dump, **{f"eval_{k}": v for k, v in ev.items()
                                          if isinstance(v, np.ndarray)},
                            probe=probe, eval_pairs=eval_pairs)
        print(f"raw arrays -> {args.dump}")

    label = ev["quality"] > 0.5
    print(f"\n{'':26} {'Spearman':>10} {'AUROC>0.5':>11}   (global — easy comparisons)")
    print("-" * 64)
    series = (("model score", ev["score"]), ("probe", probe),
              ("probe + score feature", control_score),
              ("null: random orientation", control_shuffled),
              ("null: untrained network", control_random))
    for name, s in series:
        print(f"{name:26} {spearman(s, ev['quality']):10.4f} {auroc(s, label):11.4f}")

    print(f"\nPairwise accuracy inside NMS clusters — {len(eval_pairs)} pairs")
    print("this is the ordering F1 depends on; 0.5 = coin flip")
    print("-" * 64)
    results, decisive_results = {}, {}
    for name, s in series:
        acc = pair_accuracy(s, eval_pairs)
        lo, hi = bootstrap_pair_accuracy(s, eval_pairs, eval_pair_image)
        results[name] = acc
        print(f"  {name:26} {acc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    if eval_decisive.any():
        print(f"\nRestricted to DECISIVE pairs — {int(eval_decisive.sum())} of them, "
              f"where one member is above IoU 0.5 and the other is not.")
        print("Only these can turn a true positive into a false one; the rest are")
        print("both-good pairs where NMS cannot hurt the metric whichever it keeps.")
        print("-" * 64)
        dec_pairs = eval_pairs[eval_decisive]
        dec_image = eval_pair_image[eval_decisive]
        for name, s in series:
            acc = pair_accuracy(s, dec_pairs)
            lo, hi = bootstrap_pair_accuracy(s, dec_pairs, dec_image)
            decisive_results[name] = acc
            print(f"  {name:26} {acc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 64)
    n_eval_seq = len(set(ev["groups"]))
    if overlap:
        print(f"INVALID: {len(overlap)} sequences appear in both splits. CULane frames")
        print("are consecutive, so the probe has effectively seen the eval data.")
        return 1
    leak = results["null: random orientation"] - 0.5
    if abs(leak) > 0.02:
        print(f"INVALID: the random-orientation null reads "
              f"{results['null: random orientation']:.3f}, not ~0.5 ({leak:+.3f}).")
        print("Something leaks between fit and evaluation; the headline is meaningless.")
        return 1
    if results["probe + score feature"] < results["model score"] - 0.02:
        print("INCONCLUSIVE: the probe cannot even match the score when handed the score")
        print("as an input feature, so it is underfit. Raise --epochs or --max-fit-images")
        print("before reading anything into the comparison.")
        return 1
    if n_eval_seq < 8:
        print(f"UNDERPOWERED: only {n_eval_seq} evaluation sequences. Raise")
        print("--max-eval-images, or point the eval loader at the test list.")

    gain = results["probe"] - results["model score"]
    floor = results["null: untrained network"]
    print(f"NMS picks the better duplicate {results['model score']:.1%} of the time.")
    print(f"one untrained draw reads {floor:.1%}. That is a single random function, not")
    print("an expectation, so it can land either side of 0.5 and is NOT a floor to")
    print("normalize against; the random-orientation null "
          f"({results['null: random orientation']:.3f}) is the one that must be 0.5.")
    print(f"probe - model, inside clusters: {gain:+.4f} "
          f"({gain / max(1e-6, 1 - results['model score']):.1%} of what is left)")
    if decisive_results:
        dgain = decisive_results["probe"] - decisive_results["model score"]
        print(f"probe - model, DECISIVE pairs only: {dgain:+.4f}   <- the number that")
        print("   bounds what a ranking loss can buy in F1")
    if gain > 0.05:
        print("\nEPISTEMIC: the features carry ordering information the score does not use.")
        print("A ranking objective on the score has something to reach.")
    elif gain > 0.02:
        print("\nMIXED: some headroom, most of the oracle gap is not in these features.")
    else:
        print("\nALEATORIC: a probe trained directly on this ordering, on the same data the")
        print("detector saw, cannot beat the score. The gap is the dataset, not the head.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
