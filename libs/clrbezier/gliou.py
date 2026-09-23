import torch
import torch.nn as nn
import torch.nn.functional as F

def _calculate_lane_half_widths(lane_x, base_half_width,
                                img_h, img_w, angle_aware=True, detach_geometry=False, sanitize_large_dx=False,
                                max_dx_px=1.0e4):
    """
    Calculate the virtual half-width of a lane at every sampled row.
    Args:
        lane_x:
            Tensor [..., N]
            Normalized x coordinates.
        base_half_width:
            Scalar half-width in normalized x coordinates.
            Example for CULane: 7.5 / 800
        img_h, img_w:
            Network input geometry used by LaneIoU.
        angle_aware:
            False: constant half-width
            True: CLRerNet-style angle-aware half-width.
        detach_geometry:
            when True, width itself does not backpropagate through lane geometry.
            CLRerNet does this for predicted width.
        sanitize_large_dx:
            Useful for GT because invalid target values such as -1e5
            can otherwise produce enormous local slopes.
    Returns:
        Tensor with the same shape as lane_x.
    """
    if not angle_aware:
        return torch.ones_like(lane_x) * float(base_half_width)
    num_rows = lane_x.shape[-1]

    # With fewer than three rows there is no centered difference.
    if num_rows < 3:
        return torch.ones_like(lane_x) * float(base_half_width)
    width_source = (lane_x.detach() if detach_geometry else lane_x)

    # CLRerNet calculates the local direction using two-row spacing.
    n_strips = num_rows - 1

    dy = (float(img_h) / float(n_strips) * 2.0)
    dx = (width_source[..., 2:] - width_source[..., :-2]) * float(img_w)

    # Avoid NaN/Inf propagation in the width calculation.
    dx = torch.where(torch.isfinite(dx), dx, torch.zeros_like(dx))

    if sanitize_large_dx:
        dx =  torch.where(
            torch.abs(dx) > float(max_dx_px), torch.zeros_like(dx), dx
        )

    widths = (float(base_half_width) * torch.sqrt(dx.pow(2) + dy ** 2) / dy)

    # The centered difference does not produce widths for the first and last row.
    # Match CLRerNet and replicate the nearest width.
    widths = torch.cat(
        [
            widths[..., 0:1], widths, widths[..., -1:],
        ],
        dim=-1
    )
    return widths

def aligned_generalized_lane_iou(pred_x, target_x, lane_width, img_h, img_w, angle_aware_width=True, eps=1.0e-9):
    """
    Generalized Lane IoU for already-aligned prediction/GT pairs.

    Shapes:
        pred_x:
            [N, num_points]
        target_x:
            [N, num_points]
    Coordinates:
        normalized x in approximately [0, 1].
    Invalid GT rows:
        target_x < 0
        target_x >= 1
    
    Returns:
        gliou: [N]
        valid_lane_mask: [N]

    ----------------------------------------------------------------
    GLIoU idea
    ----------------------------------------------------------------
    For one horizontal slice:
        prediction segment:
            [pred_x - pred_width, pred_x + pred_width]
        GT segment:
            [target_x - target_width, target_x + target_width]
    overlap:
        min(right) - max(left)
    union:
        max(right) - min(left)

    When lanes are separated, overlap becomes negative.
    For equal half-width e, ADNet defines:
        gap = ReLU(union - 4e)

    Here we generalize that to possibly different angle-aware half-widths:
        gap = ReLU(union - 2 * (pred_width + target_width))

    If:     pred_width = target_width = e
    then:   2 * (e + e) = 4e
    exactly recovering the ADNet expression.

    Finally:
        GLIoU = (sum(overlap) - sum(gap)) / sum(union)
    """
    if pred_x.shape != target_x.shape:
        raise ValueError(
            "aligned_generalized_lane_iou requires "
            "pred_x and target_x to have identical shapes. "
            f"Got pred={tuple(pred_x.shape)}, "
            f"target={tuple(target_x.shape)}"
        )

    # ---------------------------------------------------------------
    # Virtual half-widths
    # ---------------------------------------------------------------
    pred_width = _calculate_lane_half_widths(lane_x=pred_x,
                                             base_half_width=lane_width,
                                             img_h=img_h,
                                             img_w=img_w,
                                             angle_aware=angle_aware_width,
                                             detach_geometry=True, # do not let the loss optimize geometry merely by inflaing/altering the width calculation.
                                             sanitize_large_dx=False, # Match CLRerNet behavior for prediction.
                                             )

    target_width = _calculate_lane_half_widths(lane_x=target_x, base_half_width=lane_width,
                                               img_h=img_h, img_w=img_w, angle_aware=angle_aware_width, detach_geometry=False,
                                               sanitize_large_dx=True)

    # -----------------------------------------------------------
    # Segment boundaries
    # -----------------------------------------------------------
    pred_left = pred_x - pred_width
    pred_right = pred_x + pred_width
    target_left = target_x - target_width
    target_right = target_x + target_width

    # -----------------------------------------------------------
    # Raw overlap and union
    # -----------------------------------------------------------
    overlap = (torch.minimum(pred_right, target_right) - torch.maximum(pred_left, target_left))
    union = (torch.maximum(pred_right, target_right) - torch.minimum(pred_left, target_left))

    # -----------------------------------------------------------
    # GLIoU gap penalty
    # -----------------------------------------------------------
    total_segment_width = (2.0 * (pred_width + target_width))
    gap = F.relu(union - total_segment_width)

    # --------------------------------------------------
    # GT validity
    # --------------------------------------------------
    valid = (torch.isfinite(target_x) & (target_x >= 0.0) & (target_x < 1.0))
    overlap = torch.where(valid, overlap, torch.zeros_like(overlap))
    union = torch.where(valid, union, torch.zeros_like(union))
    gap = torch.where(valid, gap, torch.zeros_like(gap))

    # ---------------------------------------------------
    # Lane-level GLIoU
    # ---------------------------------------------------
    numerator = (overlap.sum(dim=-1) - gap.sum(dim=-1))
    denominator = union.sum(dim=-1)

    gliou = (numerator / (denominator + float(eps)))
    valid_lane_mask = valid.any(dim=-1)

    return (gliou, valid_lane_mask)

class GeneralizedLaneIoULoss(nn.Module):
    """
    GLIoU regression loss.
        loss = 1 - GLIoU
    This keeps the same call pattern as the existing LaneIoULoss.
    """
    def __init__(self, loss_weight=1.0, lane_width=7.5/800, img_h=320, img_w=800, angle_aware_width=True, eps=1.0e-9):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.lane_width = float(lane_width)
        self.img_h = int(img_h)
        self.img_w = int(img_w)

        self.angle_aware_width = bool(angle_aware_width)
        self.eps = float(eps)

    def forward(self, pred, target, reduction="none"):
        """
        Args:
            pred: [N_positive, num_points]
            target: [N_positive, num_points]

        Returns:
            scalar loss
        """
        if pred.shape != target.shape:
            raise ValueError(
                "GeneralizedLaneIoULoss shape mismatch: "
                f"pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}"
            )
        if pred.shape[0] == 0:
            if reduction == "none":
                return pred.new_zeros((0, ))
            return pred.sum() * 0.0

        gliou, valid_lane_mask = aligned_generalized_lane_iou(pred_x=pred,
            target_x=target, lane_width=self.lane_width, img_h=self.img_h, img_w=self.img_w,
            angle_aware_width=self.angle_aware_width, eps=self.eps)

        # --------------------------------------------
        # Raw ADNet-style regression objective.
        # 
        # GLIoU approximately:
        #   (-2, 1]
        # Loss:
        #   [0, 3)
        loss = 1.0 - gliou
        loss = torch.nan_to_num(loss, nan=3.0, posinf=3.0, neginf=3.0)

        # ------------------------------------------------
        # Apply global IoU loss weight
        # ------------------------------------------------
        loss = loss * self.loss_weight

        # -----------------------------------------------
        # No valid lanes.
        # -----------------------------------------------
        if not bool(valid_lane_mask.any()):
            if reduction == "none":
                return torch.zeros_like(loss)
            return pred.sum() * 0.0

        if reduction == "none":
            # Keep [N] shape so that packed group IDs remain aligned.
            # In normal matched-positive training every GT lane should
            # have at least one valid x sample. The zeroing here is
            # mainly a safaty mechanism.
            return torch.where(valid_lane_mask, loss, torch.zeros_like(loss))
        
        valid_loss = loss[valid_lane_mask]
        if reduction == "mean":
            return valid_loss.mean()
        elif reduction == "sum":
            return valid_loss.sum()
        else:
            raise ValueError(f"Unsupported reduction={reduction!r}. "
                             "Expected 'none', 'mean', or 'sum'.")
        

def pairwise_generalized_lane_iou(pred_x, target_x, lane_width, img_h, img_w, angle_aware_width=True, use_pred_start_end=False,
                                   pred_start=None, pred_end=None, eps=1.0e-9):
    """
    Pairwise GLIoU matrix.
    Args:
        pred_x: [N_pred, num_points]
        target_x: [N_gt, num_points]
        pred_start: [N_pred], normalized row coordinate.
        pred_end: [N_pred], normalized row coordinate.

    Returns:
        gliou:  [N_pred, N_gt]
    """
    num_pred = pred_x.shape[0]
    num_gt = target_x.shape[0]

    if (num_pred == 0 or num_gt == 0):
        return pred_x.new_zeros((num_pred, num_gt))
    if pred_x.shape[-1] != target_x.shape[-1]:
        raise ValueError(
            "Pairwise GLIoU row-count mismatch: "
            f"pred={pred_x.shape[-1]}, "
            f"GT={target_x.shape[-1]}"
        )

    # Assignment itself is non-differentiable.
    pred_x = pred_x.detach()
    target_x = target_x.detach()

    # ----------------------------------------------
    # Widths
    # ----------------------------------------------
    pred_width = _calculate_lane_half_widths(lane_x=pred_x,
                                             base_half_width=lane_width,
                                             img_h=img_h, img_w=img_w,
                                             angle_aware=angle_aware_width)
    target_width = _calculate_lane_half_widths(lane_x=target_x,
                                               base_half_width=lane_width,
                                               img_h=img_h,
                                               img_w=img_w,
                                               angle_aware=angle_aware_width,
                                               detach_geometry=False,
                                               sanitize_large_dx=True)

    # -----------------------------------------------
    # Expand into prediction x GT pairs.
    # -----------------------------------------------
    pred_left = (pred_x - pred_width)
    pred_right =  (pred_x + pred_width)

    target_left = (target_x - target_width)
    target_right = (target_x + target_width)
    overlap = (torch.minimum(pred_right[:, None, :], target_right[None, :, :]) - torch.maximum(pred_left[:, None, :], target_left[None, :, :]))
    union = (torch.maximum(pred_right[:, None, :], target_right[None, :, :]) - torch.minimum(pred_left[:, None, :], target_left[None, :, :]))

    pred_width_pair = pred_width[:, None, :]
    target_width_pair = target_width[None, :, :]
    gap = F.relu(union - 2.0 * (pred_width_pair + target_width_pair))

    # ===============================================
    # Validity handling
    # ===============================================
    target_valid = torch.isfinite(target_x) & (target_x >= 0.0) & (target_x < 1.0)
    if not use_pred_start_end:
        # match the behavior of the regular LaneIoU cost:
        # GT validity determines which horizontal rows participate.
        valid = target_valid[None, :, :]
        overlap = torch.where(valid, overlap, torch.zeros_like(overlap)) 
        union = torch.where(valid, union, torch.zeros_like(union))

        gap = torch.where(valid, gap, torch.zeros_like(gap))

    else:
        if (pred_start is None or pred_end is None):
            raise ValueError("use_pred_start_end=True requires "
                             "pred_start and pred_end")

        num_rows = pred_x.shape[-1]
        n_strips = num_rows - 1
        pred_start = pred_start.detach()
        pred_end = pred_end.detach()

        pred_valid = torch.isfinite(pred_x) & (pred_x >= 0.0) & (pred_x < 1.0)
        row_indices = torch.arange(num_rows, device=pred_x.device, dtype=torch.long).view(1, num_rows)
        start_indices = (pred_start * float(n_strips)).long().clamp(min=0, max=n_strips).view(num_pred, 1)
        end_indices = (pred_end * float(n_strips)).long().clamp(min=0, max=num_rows).view(num_pred, 1)

        pred_valid = (pred_valid & (row_indices >= start_indices) & (row_indices < end_indices))

        pred_valid_pair = pred_valid[:, None, :]
        target_valid_pair = target_valid[None, :, :]
        both_valid = (pred_valid_pair & target_valid_pair)

        pred_only = (pred_valid_pair &~target_valid_pair)
        target_only = (~pred_valid_pair & target_valid_pair)

        # Overlap/gap only have geometric meaning where both
        # lanes exist.
        overlap = torch.where(both_valid, overlap, torch.zeros_like(overlap))
        gap = torch.where(both_valid, gap, torch.zeros_like(gap))
        new_union = torch.zeros_like(union)

        # Both lane exists.
        new_union = torch.where(both_valid, union, new_union)

        # Prediction exists but GT does not:
        # union is the full prediction segment width.
        new_union = torch.where(pred_only, 2.0 * pred_width_pair, new_union)
        # GT exists but prediction does not:
        # union is the full GT segment width.
        new_union = torch.where(target_only, 2.0 * target_width_pair, new_union)
        union = new_union

    # =========================================================
    # Aggregate horizontal slices
    # =========================================================
    numerator = overlap.sum(dim=-1) - gap.sum(dim=-1)
    denominator = union.sum(dim=-1)
    gliou = numerator / (denominator + float(eps))

    # Assignment is non-differentiable, so numerical clamping
    # here is safe and avoids weird cost from roundoff.
    gliou = torch.clamp(gliou, min=-2.0, max=1.0)
    return gliou

def pairwise_generalized_lane_iou_cost(
        pred_x, target_x, lane_width, img_h, img_w, angle_aware_width=True,
        use_pred_start_end=False, pred_start=None, pred_end=None, normalize_cost=True
):
    """
    Convert GLIoU similarity into a minimization cost.

        GLIoU: (-2, 1]
        1 - GLIoU: [0, 3)
        
    Optional normalization:
        (1 - GLIoU) / 3: [0, 1)
    """
    gliou = pairwise_generalized_lane_iou(pred_x=pred_x,
                                          target_x=target_x, lane_width=lane_width,
                                          img_h=img_h, img_w=img_w, angle_aware_width=angle_aware_width,
                                          use_pred_start_end=use_pred_start_end, pred_start=pred_start,
                                          pred_end=pred_end)
    cost = 1.0 - gliou
    if normalize_cost :
        cost = (cost / 3.0)

    return cost


