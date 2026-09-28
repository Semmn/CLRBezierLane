"""Pairwise ranking loss restricted to the candidates NMS compares.

Why this and not another quality target
---------------------------------------
F1 at a re-selected threshold depends only on the *order* of the scores: NMS
keeps the highest-scoring member of each duplicate cluster, and the threshold
is a cutoff on that same order. Any monotone rescaling of the score is
therefore worth exactly zero F1, which is why swapping cross-entropy for focal
loss moves the optimal threshold from ~0.9 to ~0.5 without changing F1, and why
regressing the absolute IoU value buys calibration the metric does not pay for.

The decision the score is actually used for — "which of these near-duplicates
is best?" — is never in the loss. This module puts it there, and only there:

    for every pair (i, j) of predictions whose mutual LaneIoU exceeds the NMS
    threshold, and whose qualities differ by more than a margin,
        L += softplus( -(s_i - s_j) / tau )   where q_i > q_j

Three properties follow, and they are the point:

* **Monotone-invariant.** The loss constrains differences, not levels, so the
  score distribution and your confidence threshold do not move. Nothing has to
  be re-tuned when it is switched on.
* **Weak.** It asks for an ordering inside a cluster of geometrically similar
  candidates, not for a calibrated value across the whole dataset. That is a
  far smaller hypothesis class than IoU regression, which is the usual reason
  ranking objectives generalize where quality regression does not.
* **Local.** Candidates inside one cluster differ by a few pixels of geometry,
  so the comparison is a local decision rather than a global one.

Interaction with the classification loss
----------------------------------------
Under **one-to-one** assignment the cluster holds one matched positive and
several unmatched duplicates that cross-entropy labels 0 identically. Ranking
then orders exactly the candidates CE is indifferent between: complementary,
no conflict.

Under **one-to-many** assignment the duplicates are all labelled 1, so CE pulls
them together while this loss pushes them apart. That conflict is real. Use
``positives_only=False`` (the default) so the loss also sees matched-vs-
unmatched pairs, keep ``loss_weight`` small, and watch ``rank_pair_acc``
against ``loss_cls``: if classification degrades while pair accuracy rises, the
two objectives are fighting and the score needs to be split into a
threshold head and an ordering head.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .lane_iou import pairwise_lane_iou


def cluster_rank_loss(
    scores,
    pred_xs,
    quality,
    valid=None,
    cluster_iou_thr: float = 0.35,
    margin: float = 0.05,
    tau: float = 0.5,
    weight_by_gap: bool = True,
    positives_only: bool = False,
    positive_mask=None,
    lane_width: float = 7.5 / 800,
    img_w: int = 800,
    img_h: int = 320,
    max_pairs: int = 20000,
):
    """Ranking loss over duplicate clusters, for one image.

    Args:
        scores: ``[K]`` lane score (a logit or a probability; only differences
            are used, so either works — keep it consistent).
        pred_xs: ``[K, R]`` predicted x per row, normalized by ``img_w - 1``.
        quality: ``[K]`` LaneIoU of each prediction against its best GT. Rows
            with no GT overlap should be 0.
        valid: optional ``[K]`` bool, predictions to consider at all.
        cluster_iou_thr: mutual LaneIoU above which two predictions are treated
            as duplicates, i.e. as competing in NMS. Match this to the NMS
            threshold actually used at inference.
        margin: minimum quality gap for a pair to be supervised. Pairs closer
            than this carry no reliable ordering and only add noise.
        tau: logit temperature. Smaller = sharper separation demanded.
        weight_by_gap: weight each pair by ``q_i - q_j``, so clear-cut pairs
            dominate and marginal ones barely contribute.
        positives_only: restrict to pairs where both are assigned positives.
        positive_mask: ``[K]`` bool, required when ``positives_only``.
        max_pairs: cap on supervised pairs per image (subsampled if exceeded).

    Returns:
        ``(loss, stats)`` — a scalar and a dict of diagnostics.
    """
    device = scores.device
    zero = scores.new_zeros(())
    stats = {"rank_pairs": zero, "rank_pair_acc": zero}

    if pred_xs.numel() == 0 or scores.numel() < 2:
        return zero, stats

    keep = torch.ones_like(scores, dtype=torch.bool) if valid is None else valid.bool()
    if positives_only:
        if positive_mask is None:
            raise ValueError("positives_only=True requires positive_mask")
        keep = keep & positive_mask.bool()
    if int(keep.sum()) < 2:
        return zero, stats

    s = scores[keep]
    q = quality[keep]
    xs = pred_xs[keep]

    # Who competes with whom: mutual LaneIoU above the NMS threshold. The
    # matrix is computed without gradient — cluster membership is a routing
    # decision, not something to learn through.
    with torch.no_grad():
        mutual = pairwise_lane_iou(xs, xs, lane_width, img_w, img_h)
        n = mutual.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=device)
        competes = (mutual > cluster_iou_thr) & ~eye
        gap = q[:, None] - q[None, :]
        # Keep the oriented pair (i beats j) once: gap > margin implies i > j.
        pairs = competes & (gap > margin)
        idx_i, idx_j = pairs.nonzero(as_tuple=True)

        if idx_i.numel() == 0:
            return zero, stats
        if idx_i.numel() > max_pairs:
            sel = torch.randperm(idx_i.numel(), device=device)[:max_pairs]
            idx_i, idx_j = idx_i[sel], idx_j[sel]
        gaps = gap[idx_i, idx_j]

    diff = s[idx_i] - s[idx_j]
    losses = F.softplus(-diff / tau)
    if weight_by_gap:
        weights = gaps / gaps.sum().clamp_min(1e-6)
        loss = (losses * weights).sum()
    else:
        loss = losses.mean()

    with torch.no_grad():
        stats["rank_pairs"] = torch.as_tensor(float(idx_i.numel()), device=device)
        stats["rank_pair_acc"] = (diff > 0).float().mean()
    return loss, stats


def batch_cluster_rank_loss(scores, pred_xs, quality, valid=None,
                            positive_mask=None, **kwargs):
    """``cluster_rank_loss`` over a batch: ``[B, K]`` / ``[B, K, R]``.

    Clusters never cross images, so the batch is just a loop; the per-image
    matrices are K x K with K = 35, which is negligible.
    """
    batch = scores.shape[0]
    total = scores.new_zeros(())
    counted = 0
    agg = {"rank_pairs": scores.new_zeros(()), "rank_pair_acc": scores.new_zeros(())}
    for b in range(batch):
        loss, stats = cluster_rank_loss(
            scores[b], pred_xs[b], quality[b],
            valid=None if valid is None else valid[b],
            positive_mask=None if positive_mask is None else positive_mask[b],
            **kwargs)
        if float(stats["rank_pairs"]) > 0:
            total = total + loss
            agg["rank_pairs"] = agg["rank_pairs"] + stats["rank_pairs"]
            agg["rank_pair_acc"] = agg["rank_pair_acc"] + stats["rank_pair_acc"]
            counted += 1
    if counted == 0:
        return total, agg
    agg["rank_pairs"] = agg["rank_pairs"] / counted
    agg["rank_pair_acc"] = agg["rank_pair_acc"] / counted
    return total / counted, agg
