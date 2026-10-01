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


def mean_row_distance(pred_xs, img_w: int):
    """Pairwise mean |dx| in pixels over rows where both lanes are valid.

    This is what CLRNet's lane NMS measures, so it is what cluster membership
    should use. LaneIoU cannot substitute: with a half-width of 7.5/800 two
    lanes more than 15 px apart have *zero* overlap, so no IoU threshold can
    reach an NMS distance of 50 px. An IoU rule therefore supervises only the
    innermost few pixels of each cluster and leaves the rest invisible.
    """
    valid = (pred_xs >= 0.0) & (pred_xs <= 1.0)
    both = valid[:, None, :] & valid[None, :, :]
    diff = (pred_xs[:, None, :] - pred_xs[None, :, :]).abs() * float(img_w - 1)
    count = both.sum(-1)
    dist = (diff * both).sum(-1) / count.clamp(min=1)
    return dist.masked_fill(count == 0, float("inf"))


def cluster_rank_loss(
    scores,
    pred_xs,
    quality,
    valid=None,
    cluster_mode: str = "distance",
    nms_thres: float = 50.0,
    cluster_iou_thr: float = 0.35,
    margin: float = 0.05,
    tau: float = 0.5,
    weight_by_gap: bool = True,
    weight_mode: str = "gap",
    decisive_thr: float = 0.5,
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
        cluster_mode: "distance" (default) treats two predictions as competing
            when their mean row distance is below ``nms_thres``, which is
            exactly CLRNet's lane NMS rule. "iou" uses ``cluster_iou_thr`` on
            LaneIoU instead, which can only ever select pairs closer than
            2 * lane_width (15 px at the default) and so misses most of the
            cluster.
        nms_thres: distance threshold in pixels at ``img_w``, for
            ``cluster_mode="distance"``. Set it to the ``test_cfg.nms_thres``
            you actually run at inference.
        cluster_iou_thr: LaneIoU threshold for ``cluster_mode="iou"``.
        margin: minimum quality gap for a pair to be supervised. Pairs closer
            than this carry no reliable ordering and only add noise.
        tau: logit temperature. Smaller = sharper separation demanded.
        weight_by_gap: weight each pair by ``q_i - q_j``, so clear-cut pairs
            dominate and marginal ones barely contribute.
        weight_mode: what the weighting should chase.
            "gap"      — by ``q_i - q_j`` (the original behaviour).
            "decisive" — only pairs that straddle ``decisive_thr``, i.e. where
                one member is a metric true positive and the other is not.
                Those are the only comparisons whose outcome can turn a TP into
                an FP; for a pair at (0.9, 0.7) NMS may keep either and F1 never
                notices. Gap weighting is doubly misaligned with F1 here: it
                spends weight on inconsequential pairs and under-weights the
                hard decisive ones like (0.55, 0.45).
            "boundary" — decisive pairs, weighted toward the threshold, so the
                (0.55, 0.45) cases dominate the (0.95, 0.05) ones the model
                already gets right.
        decisive_thr: the metric's IoU threshold, 0.5 for CULane.
        positives_only: restrict to pairs where both are assigned positives.
        positive_mask: ``[K]`` bool, required when ``positives_only``.
        max_pairs: cap on supervised pairs per image (subsampled if exceeded).

    Returns:
        ``(loss, stats)`` — a scalar and a dict of diagnostics.
    """
    device = scores.device
    zero = scores.new_zeros(())
    stats = {"rank_pairs": zero, "rank_pair_acc": zero, "rank_logit_gap": zero,
             "rank_decisive_frac": zero, "rank_pair_acc_dec": zero}

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
        if cluster_mode == "distance":
            proximity = mean_row_distance(xs, img_w)
            n = proximity.shape[0]
            eye = torch.eye(n, dtype=torch.bool, device=device)
            competes = (proximity < nms_thres) & ~eye
        elif cluster_mode == "iou":
            mutual = pairwise_lane_iou(xs, xs, lane_width, img_w, img_h)
            n = mutual.shape[0]
            eye = torch.eye(n, dtype=torch.bool, device=device)
            competes = (mutual > cluster_iou_thr) & ~eye
        else:
            raise ValueError("cluster_mode must be 'distance' or 'iou'")
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

    with torch.no_grad():
        q_hi, q_lo = q[idx_i], q[idx_j]
        decisive = (q_hi > decisive_thr) & (q_lo <= decisive_thr)
        if weight_mode == "decisive":
            raw_w = decisive.to(s.dtype)
        elif weight_mode == "boundary":
            # closeness of the pair to the threshold: 1 when both sit on it.
            near = 1.0 - (q_hi - q_lo).clamp(0.0, 1.0)
            raw_w = decisive.to(s.dtype) * near
        elif weight_mode == "gap":
            raw_w = gaps if weight_by_gap else torch.ones_like(gaps)
        else:
            raise ValueError("weight_mode must be 'gap', 'decisive' or 'boundary'")

    diff = s[idx_i] - s[idx_j]
    losses = F.softplus(-diff / tau)
    total_w = raw_w.sum()
    if float(total_w) <= 0:
        # no pair of consequence in this image
        return zero, stats
    loss = (losses * (raw_w / total_w.clamp_min(1e-6))).sum()

    with torch.no_grad():
        stats["rank_pairs"] = torch.as_tensor(float(idx_i.numel()), device=device)
        stats["rank_pair_acc"] = (diff > 0).float().mean()
        # The typical within-cluster score gap, in whatever space `scores` is.
        # tau should sit near this: far below it and every ordered pair
        # saturates to zero gradient, far above it and the loss stays in its
        # linear region and treats easy and hard pairs alike. Cross-entropy and
        # focal loss produce very different gaps, so tau does not transfer
        # between them.
        stats["rank_logit_gap"] = diff.abs().mean()
        stats["rank_decisive_frac"] = decisive.float().mean()
        if decisive.any():
            stats["rank_pair_acc_dec"] = (diff[decisive] > 0).float().mean()
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
    agg = {k: scores.new_zeros(()) for k in
           ("rank_pairs", "rank_pair_acc", "rank_logit_gap",
            "rank_decisive_frac", "rank_pair_acc_dec")}
    for b in range(batch):
        loss, stats = cluster_rank_loss(
            scores[b], pred_xs[b], quality[b],
            valid=None if valid is None else valid[b],
            positive_mask=None if positive_mask is None else positive_mask[b],
            **kwargs)
        if float(stats["rank_pairs"]) > 0:
            total = total + loss
            for key in agg:
                agg[key] = agg[key] + stats[key]
            counted += 1
    if counted == 0:
        return total, agg
    for key in agg:
        agg[key] = agg[key] / counted
    return total / counted, agg
