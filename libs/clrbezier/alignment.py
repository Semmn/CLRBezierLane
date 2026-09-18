"""Confidence-localization alignment: IoU-aware classification targets.

CLRerNet uses LaneIoU to decide *which* anchors are positive, but the
classification target stays binary, so an anchor at IoU 0.55 and one at IoU 0.95
are trained toward the same value. The score therefore cannot rank by metric
IoU, which is what the F1 threshold and NMS depend on (CLRerNet's oracle
experiment: perfect confidence alone takes 80.86 -> 98.47).

This module supplies:

    quality_targets   soft classification targets from the matched pairs'
                      LaneIoU ("iou") or the TOOD task-aligned score
                      t = s^alpha * u^beta, normalized per GT ("task_aligned").
    QualityFocalLoss  2-class softmax cross-entropy toward a soft target with
                      the quality-focal modulation |t - p|^beta.
    ignore_mask       unmatched anchors whose LaneIoU with some GT exceeds a
                      threshold. Your main branch (one-to-one) labels those 0
                      while the auxiliary branch (one-to-many) labels the same
                      anchors 1 in the same iteration; ignoring them removes
                      that contradiction without touching the matching.

Width: the confidence target should use the LaneIoU width that matches the
evaluation metric. CULane draws 30 px at 1640 wide, i.e. ~14.6 px at 800 wide,
so the narrow half-width 7.5/800 (``lane_iou_dynamic`` in the cost cache) is the
right quantity, not the 30/800 ranking width.
"""
import torch
import torch.nn.functional as F


def _amax_per_gt(values, cols, num_gt):
    """Max of `values` grouped by GT index `cols` -> [num_gt]."""
    out = values.new_zeros(num_gt)
    try:
        return out.scatter_reduce(0, cols, values, reduce="amax", include_self=False)
    except (TypeError, RuntimeError):  # older torch
        for g in range(num_gt):
            sel = values[cols == g]
            if sel.numel():
                out[g] = sel.max()
        return out


def quality_targets(iou, scores=None, mode="iou", alpha=1.0, beta=6.0, cols=None, num_gt=0,
                    eps=1e-6):
    """Soft classification targets for matched pairs.

    Args:
        iou: [P] LaneIoU of each matched pair (narrow width), detached.
        scores: [P] current foreground probabilities, detached (task_aligned only).
        mode: "iou" or "task_aligned".
        cols: [P] GT index of each pair; needed for per-GT normalization.
    Returns:
        [P] targets in [0, 1].
    """
    iou = iou.clamp(0.0, 1.0)
    if mode == "iou":
        return iou
    if mode != "task_aligned":
        raise ValueError(f"Unknown quality target mode {mode!r}")
    if scores is None:
        raise ValueError("task_aligned targets need the current scores")
    t = scores.clamp(eps, 1.0).pow(alpha) * iou.clamp_min(eps).pow(beta)
    if cols is None or num_gt <= 0:
        return t.clamp(0.0, 1.0)
    # TOOD normalization: rescale each GT's targets so the best one equals that
    # GT's best IoU. With one-to-one matching this is the identity.
    max_t = _amax_per_gt(t, cols, num_gt).clamp_min(eps)
    max_iou = _amax_per_gt(iou, cols, num_gt)
    return (t / max_t[cols] * max_iou[cols]).clamp(0.0, 1.0)


class QualityFocalLoss(torch.nn.Module):
    """Quality focal loss for 2-class softmax logits and soft targets.

    loss = -[t log p_fg + (1 - t) log p_bg] * |t - p_fg|^beta

    With t in {0, 1} this reduces to focal-modulated cross-entropy, so the
    hard-label behaviour is recovered when quality targets are disabled.
    """

    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, logits, targets):
        """logits: [N, 2]; targets: [N] in [0, 1] -> [N] loss."""
        logp = F.log_softmax(logits, dim=-1)
        p_fg = logp[:, 1].exp()
        ce = -(targets * logp[:, 1] + (1.0 - targets) * logp[:, 0])
        return ce * (targets - p_fg).abs().pow(self.beta)


def ignore_unmatched(iou_matrix, matched_rows, threshold):
    """Anchors to drop from the classification loss.

    Args:
        iou_matrix: [N, G] LaneIoU between every prediction and every GT.
        matched_rows: indices of assigned predictions (never ignored).
        threshold: ignore unmatched anchors whose best IoU exceeds this.
    Returns:
        [N] bool mask, True where the anchor should be skipped.
    """
    if threshold is None or threshold >= 1.0 or iou_matrix.numel() == 0:
        return torch.zeros(iou_matrix.shape[0], dtype=torch.bool, device=iou_matrix.device)
    mask = iou_matrix.max(dim=1).values > float(threshold)
    mask[matched_rows] = False
    return mask
