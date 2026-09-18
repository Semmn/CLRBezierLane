"""LaneIoU (CLRerNet) loss and pairwise cost.

Inputs follow the official CLRerNet geometry convention:
    pred   = pred_xs_normalized * (img_w - 1) / img_w
    target = target_xs_pixels / img_w
Valid GT rows: 0 <= target < 1.
"""
import torch


def _lane_half_widths(xs, lane_width, img_w, img_h, max_dx=None, detach=False):
    n_strips = xs.shape[-1] - 1
    dy = float(img_h) / float(n_strips) * 2.0
    src = xs.detach() if detach else xs
    dx = (src[..., 2:] - src[..., :-2]) * float(img_w)
    if max_dx is not None:
        # Official: only the GT width ignores huge jumps across invalid rows.
        dx = torch.where(dx.abs() > max_dx, torch.zeros_like(dx), dx)
    width = float(lane_width) * torch.sqrt(dx.pow(2) + dy ** 2) / dy
    return torch.cat([width[..., 0:1], width, width[..., -1:]], dim=-1)


class LaneIoULoss(torch.nn.Module):
    """Aligned LaneIoU loss, (1 - IoU) * loss_weight per lane."""

    def __init__(self, loss_weight=4.0, lane_width=7.5 / 800, img_w=800, img_h=320):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.lane_width = float(lane_width)
        self.img_w = int(img_w)
        self.img_h = int(img_h)

    def forward(self, pred, target):
        # Official behavior: predicted width uses the detached prediction.
        pred_w = _lane_half_widths(pred, self.lane_width, self.img_w, self.img_h, detach=True)
        tgt_w = _lane_half_widths(target, self.lane_width, self.img_w, self.img_h, max_dx=1e4)
        ovr = torch.min(pred + pred_w, target + tgt_w) - torch.max(pred - pred_w, target - tgt_w)
        union = torch.max(pred + pred_w, target + tgt_w) - torch.min(pred - pred_w, target - tgt_w)
        invalid = (target < 0) | (target >= 1.0)
        ovr = ovr.masked_fill(invalid, 0.0)
        union = union.masked_fill(invalid, 0.0)
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return (1.0 - iou) * self.loss_weight


@torch.no_grad()
def pairwise_lane_iou(pred, target, lane_width, img_w, img_h, start=None, end=None):
    """Pairwise LaneIoU similarity [Np, Nt].

    If start/end are given (row fractions from the bottom, [Np]), prediction
    rows outside [start, end) are invalid and pred-only / target-only rows add
    virtual union, as in CLRerNet ``LaneIoUCost(use_pred_start_end=True)``.
    """
    pred_w = _lane_half_widths(pred, lane_width, img_w, img_h, detach=True)
    tgt_w = _lane_half_widths(target, lane_width, img_w, img_h, max_dx=1e4)
    px1, px2 = (pred - pred_w)[:, None, :], (pred + pred_w)[:, None, :]
    tx1, tx2 = (target - tgt_w)[None], (target + tgt_w)[None]
    ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
    union = torch.max(px2, tx2) - torch.min(px1, tx1)

    num_pred, num_gt = pred.shape[0], target.shape[0]
    invalid_gt = ((target < 0) | (target >= 1.0))[None].expand(num_pred, -1, -1)
    if start is None or end is None:
        ovr = ovr.masked_fill(invalid_gt, 0.0)
        union = union.masked_fill(invalid_gt, 0.0)
    else:
        rows = pred.shape[-1]
        h = rows - 1
        yind = torch.arange(rows, device=pred.device).view(1, 1, rows)
        start_idx = (start * h).long().view(-1, 1, 1)
        end_idx = (end * h).long().view(-1, 1, 1)
        invalid_pred = ((pred < 0) | (pred >= 1.0))[:, None, :].expand(-1, num_gt, -1)
        invalid_pred = invalid_pred | (yind < start_idx) | (yind >= end_idx)
        invalid_any = invalid_pred | invalid_gt
        ovr = ovr.masked_fill(invalid_any, 0.0)
        union = union.masked_fill(invalid_any, 0.0)
        pred_only = invalid_any & ~invalid_pred
        gt_only = invalid_any & ~invalid_gt
        union = union + pred_only.float() * (2.0 * pred_w)[:, None, :]
        union = union + gt_only.float() * (2.0 * tgt_w)[None]
    return ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
