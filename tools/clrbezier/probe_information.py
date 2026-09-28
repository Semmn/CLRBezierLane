"""Is the localization-quality information in the features at all?

The oracle experiment says a perfect scorer would reach ~98 F1 where the model
reaches ~80. That gap has two very different halves:

* **epistemic** — the features contain enough to tell a good lane from a bad
  one, but the scoring head does not extract it. Architecture and loss design
  can recover this.
* **aleatoric** — the evidence genuinely is not there. A lane behind a bus, a
  marking too faint to see, an annotation the labeller extrapolated from road
  context. No head can recover this, and no amount of architecture search will.

This script measures the split without retraining the detector. It freezes the
trained model, captures the exact feature vector the classifier sees for every
prediction, and fits a deliberately over-parameterized quality regressor on it
with nothing to do but this one task. Then it compares, on held-out *sequences*:

    Spearman(model score, true LaneIoU)   vs   Spearman(probe, true LaneIoU)

and the same restricted to NMS duplicate clusters, which is the ordering F1
actually depends on.

Reading the result
------------------
* Probe clearly beats the model's own score -> epistemic. The information is
  in the features and a better scoring objective can reach it. The cluster-
  restricted number tells you how much is reachable where it matters.
* Probe barely beats it -> aleatoric. The ceiling is the dataset, not the
  head. That is a reportable finding, and a reason to stop adding modules.

Sequence-grouped splitting matters: CULane frames are consecutive, so a random
per-image split leaks near-duplicate frames into the holdout and the probe
scores itself on data it has effectively seen.

Usage:
    python tools/clrbezier/probe_information.py \
        configs/clrbezier/culane/clrbezier_collab_perturb_r34.py \
        work_dirs/.../epoch_15.pth --max-images 4000
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
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
    """Mann-Whitney AUC: P(score of a true positive > score of a negative)."""
    label = label.astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(score)).astype(np.float64) + 1.0
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cluster_spearman(score, quality, xs, image_id, cluster_iou_thr, lane_width,
                     img_w, img_h, device, max_images=2000):
    """Rank correlation restricted to predictions that compete in NMS.

    A global correlation counts easy comparisons between an obvious lane and
    obvious background. NMS never makes those; it chooses between near
    duplicates, and that is the only comparison that decides F1.
    """
    concordant = discordant = 0
    for image in np.unique(image_id)[:max_images]:
        sel = np.flatnonzero(image_id == image)
        if sel.size < 2:
            continue
        rows = torch.from_numpy(xs[sel]).float().to(device)
        with torch.no_grad():
            mutual = pairwise_lane_iou(rows, rows, lane_width, img_w, img_h)
            mutual = torch.nan_to_num(mutual, nan=0.0).cpu().numpy()
        s, q = score[sel], quality[sel]
        n = sel.size
        for i in range(n):
            for j in range(i + 1, n):
                if mutual[i, j] <= cluster_iou_thr:
                    continue
                if abs(q[i] - q[j]) < 0.05:
                    continue
                right = (s[i] > s[j]) == (q[i] > q[j])
                concordant += int(right)
                discordant += int(not right)
    total = concordant + discordant
    if total == 0:
        return float("nan"), 0
    return concordant / total, total


# ------------------------------------------------------------------ dataflow
def build_gt_dataloader(cfg):
    """Validation data with deterministic augmentation *and* ground truth.

    The val pipeline drops the GT keys because inference does not need them, so
    this borrows the training pipeline (which packs them) and swaps its
    albumentation list for the validation one. Nothing random is left.
    """
    loader = copy.deepcopy(cfg.val_dataloader)
    train_pipeline = copy.deepcopy(cfg.train_dataloader.dataset.pipeline)
    val_pipeline = cfg.val_dataloader.dataset.pipeline

    val_al = next((t.get("pipelines") for t in val_pipeline
                   if t.get("type") == "albumentation"), None)
    if val_al is not None:
        for transform in train_pipeline:
            if transform.get("type") == "albumentation":
                transform["pipelines"] = copy.deepcopy(val_al)
    loader.dataset.pipeline = train_pipeline
    loader.dataset.test_mode = False
    loader.sampler = dict(type="DefaultSampler", shuffle=False)
    return Runner.build_dataloader(loader)


def sequence_of(filename):
    """CULane path -> the driving session it came from.

    e.g. ``.../driver_23_30frame/05151649_0422.MP4/00030.jpg`` ->
    ``driver_23_30frame/05151649_0422.MP4``. Consecutive frames share it, which
    is exactly what must not straddle the split.
    """
    parts = os.path.normpath(str(filename)).split(os.sep)
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else str(filename)


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


def train_probe(feat, target, groups, device, epochs=40, batch=4096, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    holdout = set(uniq[: max(1, len(uniq) // 5)].tolist())
    is_hold = np.array([g in holdout for g in groups])

    xf = torch.from_numpy(feat[~is_hold]).float()
    yf = torch.from_numpy(target[~is_hold]).float()
    mean, std = xf.mean(0, keepdim=True), xf.std(0, keepdim=True).clamp_min(1e-5)

    model = Probe(feat.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = xf.shape[0]
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb = ((xf[idx] - mean) / std).to(device)
            yb = yf[idx].to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        xh = ((torch.from_numpy(feat[is_hold]).float() - mean) / std).to(device)
        pred = torch.sigmoid(model(xh)).cpu().numpy()
    return pred, is_hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--max-images", type=int, default=4000)
    ap.add_argument("--feature", choices=["roi", "tower"], default="roi",
                    help="roi = the pooled feature entering the cls tower; "
                         "tower = the vector the final cls layer sees")
    ap.add_argument("--cluster-iou", type=float, default=0.35)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dump", help="optional .npz to save the raw arrays")
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

    loader = build_gt_dataloader(cfg)

    feats, scores, quals, xs_all, groups, image_ids = [], [], [], [], [], []
    seen = 0
    with torch.no_grad():
        for data in loader:
            data = model.data_preprocessor(data, False)
            pyramid = model.extract_feat(data["inputs"])
            outs = head(pyramid)                      # eval -> StagePredictions

            pred_xs = outs["xs"].float()              # [B, K, R], final stage
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
                iou = pairwise_lane_iou(geo_p, geo_t, head.lane_width,
                                        head.img_w, head.img_h)
                quality[b] = torch.nan_to_num(iou, nan=0.0).max(dim=1).values

            feature = captured[args.feature].view(batch, num_q, -1)
            feats.append(feature.cpu().numpy())
            scores.append(score.cpu().numpy())
            quals.append(quality.cpu().numpy())
            xs_all.append(pred_xs.cpu().numpy())
            for b in range(batch):
                name = data["data_samples"][b].metainfo.get("filename", f"img{seen + b}")
                groups.extend([sequence_of(name)] * num_q)
                image_ids.extend([seen + b] * num_q)
            seen += batch
            if seen >= args.max_images:
                break
    for handle in handles:
        handle.remove()

    feat = np.concatenate(feats).reshape(-1, feats[0].shape[-1])
    score = np.concatenate(scores).reshape(-1)
    quality = np.concatenate(quals).reshape(-1)
    xs = np.concatenate(xs_all).reshape(-1, xs_all[0].shape[-1])
    groups = np.array(groups)
    image_ids = np.array(image_ids)

    print(f"\n{seen} images, {feat.shape[0]} predictions, "
          f"{feat.shape[1]}-d {args.feature} features, "
          f"{len(np.unique(groups))} sequences")
    print(f"predictions with LaneIoU > 0.5: {(quality > 0.5).mean():.1%}")

    if args.dump:
        np.savez_compressed(args.dump, feat=feat, score=score, quality=quality,
                            xs=xs, groups=groups, image_ids=image_ids)
        print(f"raw arrays -> {args.dump}")

    probe_pred, is_hold = train_probe(feat, quality, groups, device, epochs=args.epochs)

    s_hold, q_hold = score[is_hold], quality[is_hold]
    xs_hold, img_hold = xs[is_hold], image_ids[is_hold]
    label = q_hold > 0.5

    print(f"\nheld-out sequences: {len(np.unique(groups[is_hold]))}, "
          f"{is_hold.sum()} predictions\n")
    print(f"{'':22} {'Spearman':>10} {'AUROC>0.5':>11}")
    print("-" * 46)
    print(f"{'model score':22} {spearman(s_hold, q_hold):10.4f} {auroc(s_hold, label):11.4f}")
    print(f"{'probe on features':22} {spearman(probe_pred, q_hold):10.4f} "
          f"{auroc(probe_pred, label):11.4f}")

    model_cluster, n_pairs = cluster_spearman(
        s_hold, q_hold, xs_hold, img_hold, args.cluster_iou,
        head.lane_width, head.img_w, head.img_h, device)
    probe_cluster, _ = cluster_spearman(
        probe_pred, q_hold, xs_hold, img_hold, args.cluster_iou,
        head.lane_width, head.img_w, head.img_h, device)

    print(f"\nPairwise accuracy inside NMS clusters ({n_pairs} supervised pairs)")
    print("this is the ordering F1 depends on; 0.5 = coin flip")
    print(f"  model score        {model_cluster:.4f}")
    print(f"  probe on features  {probe_cluster:.4f}")

    gain = probe_cluster - model_cluster
    print()
    if not np.isfinite(gain):
        print("Too few competing pairs to judge; raise --max-images.")
    elif gain > 0.05:
        print(f"EPISTEMIC: the probe recovers {gain:+.3f} cluster pair accuracy from the")
        print("same features the head already has. A better scoring objective can reach it.")
    elif gain > 0.02:
        print(f"MIXED: {gain:+.3f}. Some headroom, but most of the oracle gap is not in")
        print("these features. Expect small gains from scoring work.")
    else:
        print(f"ALERT — ALEATORIC: {gain:+.3f}. An over-parameterized head with nothing to")
        print("do but this task cannot rank better than your score does. The oracle gap is")
        print("the dataset, not the architecture. This is a result worth reporting.")


if __name__ == "__main__":
    main()
