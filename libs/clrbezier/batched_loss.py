"""Whole-batch cost cache, assignment and per-image reductions for CLRBezierHead.

The per-image path (``build_cost_cache`` + ``assigner.assign`` + per-image losses
inside a Python loop over the batch) launches a few hundred small kernels per
image, per stage and per branch, plus a host sync for every Hungarian solve,
top-k selection and statistic. At batch 32 with the auxiliary branch that is
~75k tensor ops and several hundred syncs per iteration. This module computes
the same quantities for the whole batch at once:

* GTs are laid out flat ([J] lanes over the batch, ``GTLayout``). Pairwise costs
  are computed per GT against the predictions of its own image ([J, N, R]),
  so no work is spent on padding, then placed into a padded [B, N, Gmax] view
  for the assigners (padding = +inf cost, never assigned).
* Hungarian matching copies all cost matrices to the host once per stage and
  solves each image on the CPU, exactly as before.
* Top-k and SimOTA are vectorized over images and GTs.
* Per-image means are reduced with a one-hot sum (no index_add / scatter_add,
  which are non-deterministic on CUDA).

Every value is computed with the same elementwise operations as the per-image
path, so assignments and losses match it (up to float summation order). Exact
cost ties in Top-k / SimOTA go to the lower prediction index here; the loop's
torch.topk leaves that choice to the kernel.
"""
import torch
from scipy.optimize import linear_sum_assignment

from .gliou import _calculate_lane_half_widths
from .lane_iou import _lane_half_widths


class GTLayout:
    """Valid GT lanes of a batch, flattened: lane j belongs to image img[j] at slot[j]."""

    def __init__(self, valid_targets, device):
        self.counts = [int(t.shape[0]) for t in valid_targets]
        self.batch_size = len(self.counts)
        self.num = sum(self.counts)
        self.gmax = max(self.counts) if self.counts else 0
        self.offsets = [0]
        for c in self.counts[:-1]:
            self.offsets.append(self.offsets[-1] + c)
        img, slot = [], []
        for b, c in enumerate(self.counts):
            img += [b] * c
            slot += list(range(c))
        self.img = torch.tensor(img, dtype=torch.long, device=device)
        self.offsets_t = torch.tensor(self.offsets, dtype=torch.long, device=device)
        self.slot = torch.tensor(slot, dtype=torch.long, device=device)
        width = valid_targets[0].shape[1] if valid_targets else 0
        self.flat = (torch.cat(list(valid_targets), 0) if self.num
                     else torch.zeros((0, width), device=device))
        # [B, Gmax] valid-slot mask
        self.mask = torch.zeros((self.batch_size, self.gmax), dtype=torch.bool, device=device)
        if self.num:
            self.mask[self.img, self.slot] = True

    def flat_index(self, b, g):
        """Global lane index of (image b, slot g)."""
        return self.offsets_t[b] + g

    def pad(self, values_jn, fill):
        """[J, N] per-GT values -> [B, N, Gmax] with ``fill`` on padding."""
        num_pred = values_jn.shape[1]
        out = values_jn.new_full((self.batch_size, self.gmax, num_pred), fill)
        if self.num:
            out[self.img, self.slot] = values_jn
        return out.transpose(1, 2)


def _per_gt_lane_iou(pred, pred_w, target, tgt_w, start_idx=None, end_idx=None):
    """``pairwise_lane_iou`` for each GT against its own image's predictions.

    pred, pred_w: [J, N, R]; target, tgt_w: [J, R]; start_idx/end_idx: [J, N].
    Same elementwise operations as lane_iou.pairwise_lane_iou -> [J, N].
    """
    px1, px2 = pred - pred_w, pred + pred_w
    tx1, tx2 = (target - tgt_w)[:, None, :], (target + tgt_w)[:, None, :]
    ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
    union = torch.max(px2, tx2) - torch.min(px1, tx1)
    invalid_gt = ((target < 0) | (target >= 1.0))[:, None, :].expand_as(ovr)
    if start_idx is None:
        ovr = ovr.masked_fill(invalid_gt, 0.0)
        union = union.masked_fill(invalid_gt, 0.0)
    else:
        rows = pred.shape[-1]
        yind = torch.arange(rows, device=pred.device).view(1, 1, rows)
        invalid_pred = (pred < 0) | (pred >= 1.0)
        invalid_pred = (invalid_pred | (yind < start_idx[..., None])
                        | (yind >= end_idx[..., None]))
        invalid_any = invalid_pred | invalid_gt
        ovr = ovr.masked_fill(invalid_any, 0.0)
        union = union.masked_fill(invalid_any, 0.0)
        pred_only = invalid_any & ~invalid_pred
        gt_only = invalid_any & ~invalid_gt
        union = union + pred_only.float() * (2.0 * pred_w)
        union = union + gt_only.float() * (2.0 * tgt_w)[:, None, :]
    return ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)


def _per_gt_gliou(pred, target, lane_width, img_h, img_w, pred_start=None, pred_end=None,
                  eps=1.0e-9):
    """``gliou.pairwise_generalized_lane_iou`` for each GT against its own image's
    predictions. pred: [J, N, R]; target: [J, R]; pred_start/end: [J, N] -> [J, N]."""
    pred = pred.detach()
    target = target.detach()
    pred_width = _calculate_lane_half_widths(lane_x=pred, base_half_width=lane_width,
                                             img_h=img_h, img_w=img_w, angle_aware=True)
    target_width = _calculate_lane_half_widths(lane_x=target, base_half_width=lane_width,
                                               img_h=img_h, img_w=img_w, angle_aware=True,
                                               detach_geometry=False, sanitize_large_dx=True)
    pred_left, pred_right = pred - pred_width, pred + pred_width
    target_left = (target - target_width)[:, None, :]
    target_right = (target + target_width)[:, None, :]
    overlap = torch.minimum(pred_right, target_right) - torch.maximum(pred_left, target_left)
    union = torch.maximum(pred_right, target_right) - torch.minimum(pred_left, target_left)
    target_width_pair = target_width[:, None, :]
    gap = torch.relu(union - 2.0 * (pred_width + target_width_pair))
    target_valid = (torch.isfinite(target) & (target >= 0.0) & (target < 1.0))[:, None, :]
    if pred_start is None:
        overlap = torch.where(target_valid, overlap, torch.zeros_like(overlap))
        union = torch.where(target_valid, union, torch.zeros_like(union))
        gap = torch.where(target_valid, gap, torch.zeros_like(gap))
    else:
        num_rows = pred.shape[-1]
        n_strips = num_rows - 1
        pred_valid = torch.isfinite(pred) & (pred >= 0.0) & (pred < 1.0)
        row_indices = torch.arange(num_rows, device=pred.device, dtype=torch.long).view(1, 1, -1)
        start_indices = (pred_start.detach() * float(n_strips)).long().clamp(
            min=0, max=n_strips)[..., None]
        end_indices = (pred_end.detach() * float(n_strips)).long().clamp(
            min=0, max=num_rows)[..., None]
        pred_valid = pred_valid & (row_indices >= start_indices) & (row_indices < end_indices)
        both_valid = pred_valid & target_valid
        pred_only = pred_valid & ~target_valid
        target_only = ~pred_valid & target_valid
        overlap = torch.where(both_valid, overlap, torch.zeros_like(overlap))
        gap = torch.where(both_valid, gap, torch.zeros_like(gap))
        new_union = torch.zeros_like(union)
        new_union = torch.where(both_valid, union, new_union)
        new_union = torch.where(pred_only, 2.0 * pred_width, new_union)
        new_union = torch.where(target_only, 2.0 * target_width_pair, new_union)
        union = new_union
    numerator = overlap.sum(dim=-1) - gap.sum(dim=-1)
    denominator = union.sum(dim=-1)
    return torch.clamp(numerator / (denominator + float(eps)), min=-2.0, max=1.0)


@torch.no_grad()
def batched_cost_cache(preds, layout, img_w, img_h, lane_width, lane_width_cost, required,
                       iou_shape=None, cls_eps=1e-6, chunk=128, iou_kind="laneiou"):
    """``build_cost_cache`` for the whole batch.

    preds: [B, N, 6+R] (detached). Returns padded [B, N, Gmax] tensors under the
    same keys as build_cost_cache (cls_cost broadcast over GTs). GTs are
    processed in chunks of ``chunk`` lanes to bound the [J, N, R] temporaries.
    iou_kind: "laneiou" (lane_iou.pairwise_lane_iou) or "gliou" (the head's
    cost_iou_type="gliou" functions, with iou_shape as their (w, h)).
    """
    required = set(required)
    preds = preds.float()
    batch_size, num_pred = preds.shape[:2]
    gmax = layout.gmax
    iou_w, iou_h = iou_shape if iou_shape is not None else (img_w, img_h)
    cache = {}
    if "cls_cost" in required:
        fg = torch.softmax(preds[..., :2], dim=-1)[..., 1].clamp(cls_eps, 1.0 - cls_eps)
        cost = torch.nan_to_num(-torch.log(fg), nan=100.0, posinf=100.0, neginf=100.0)
        cache["cls_cost"] = cost[..., None].expand(batch_size, num_pred, gmax)
    keys = [k for k in ("point_cost", "lane_iou_dynamic", "iou_cost_assign") if k in required]
    if not keys:
        return cache
    pred_xs = preds[..., 6:]
    pred_geo = pred_xs * (float(img_w - 1) / float(img_w))
    pred_w_dyn = pred_w_cost = None
    gliou = iou_kind == "gliou"
    if "iou_cost_assign" in required and gliou:
        start = (1.0 - preds[..., 2]).clamp(0.0, 1.0)
        end = (start + preds[..., 5].clamp(0.0, 1.0)).clamp(0.0, 1.0)
    if "lane_iou_dynamic" in required and not gliou:
        pred_w_dyn = _lane_half_widths(pred_geo, lane_width, iou_w, iou_h, detach=True)
    if "iou_cost_assign" in required and not gliou:
        pred_w_cost = _lane_half_widths(pred_geo, lane_width_cost, iou_w, iou_h, detach=True)
        rows = pred_xs.shape[-1]
        start = (1.0 - preds[..., 2]).clamp(0.0, 1.0)
        end = (start + preds[..., 5].clamp(0.0, 1.0)).clamp(0.0, 1.0)
        start_idx_all = (start * (rows - 1)).long()
        end_idx_all = (end * (rows - 1)).long()
    if "point_cost" in required:
        p_oob_all = (pred_xs < 0.0) | (pred_xs > 1.0)
        pred_clamped = pred_xs.clamp(0.0, 1.0)

    target = layout.flat.float()
    out = {k: [] for k in keys}
    for lo in range(0, layout.num, chunk):
        hi = min(layout.num, lo + chunk)
        img = layout.img[lo:hi]
        tgt = target[lo:hi]
        if "point_cost" in required:
            target_xs_pt = tgt[:, 6:] / float(img_w - 1)
            t_valid = (target_xs_pt >= 0.0) & (target_xs_pt <= 1.0)
            diff = (pred_clamped[img] - target_xs_pt.clamp(0.0, 1.0)[:, None, :]).abs()
            diff = diff + 0.5 * p_oob_all[img].float()
            m = t_valid[:, None, :].float()
            out["point_cost"].append((diff * m).sum(-1) / m.sum(-1).clamp(min=1.0))
        target_geo = tgt[:, 6:] / float(img_w)
        p_geo = pred_geo[img]
        if "lane_iou_dynamic" in required and gliou:
            iou = _per_gt_gliou(p_geo, target_geo, lane_width, iou_h, iou_w)
            out["lane_iou_dynamic"].append(torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0))
        elif "lane_iou_dynamic" in required:
            tgt_w = _lane_half_widths(target_geo, lane_width, iou_w, iou_h, max_dx=1e4)
            iou = _per_gt_lane_iou(p_geo, pred_w_dyn[img], target_geo, tgt_w)
            out["lane_iou_dynamic"].append(torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0))
        if "iou_cost_assign" in required and gliou:
            iou = _per_gt_gliou(p_geo, target_geo, lane_width_cost, iou_h, iou_w,
                                start[img], end[img])
            out["iou_cost_assign"].append(
                1.0 - torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0))
        elif "iou_cost_assign" in required:
            tgt_w = _lane_half_widths(target_geo, lane_width_cost, iou_w, iou_h, max_dx=1e4)
            iou = _per_gt_lane_iou(p_geo, pred_w_cost[img], target_geo, tgt_w,
                                   start_idx_all[img], end_idx_all[img])
            out["iou_cost_assign"].append(
                1.0 - torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0))
    for k in keys:
        vals = torch.cat(out[k], 0) if out[k] else preds.new_zeros((0, num_pred))
        cache[k] = layout.pad(vals, 0.0)
    return cache


def _smallest(cost, k):
    """The k smallest values along dim 1 and their indices, ascending, exact ties
    broken by the lower index (stable sort), so the first d of them are exactly
    the d smallest for every d <= k, on CPU and CUDA alike."""
    values, idx = torch.sort(cost, dim=1, stable=True)
    return values[:, :k], idx[:, :k]


@torch.no_grad()
def batched_assign(assigner, cache, layout):
    """Run one assigner on every image. Returns (b, n, g) LongTensors of pairs,
    ordered by image then prediction, as the per-image path produces them."""
    # by name (subclasses included), not isinstance: robust to the module being
    # imported twice under different names
    names = {c.__name__ for c in type(assigner).__mro__}
    kind = next((k for k in ("HungarianLaneAssigner", "TopKLaneAssigner", "SimOTALaneAssigner")
                 if k in names), type(assigner).__name__)
    cost = assigner.total_cost(cache)                        # [B, N, G], padding finite
    dev = cost.device
    empty = torch.empty(0, dtype=torch.long, device=dev)
    if layout.num == 0:
        return empty, empty, empty
    num_pred = cost.shape[1]
    gmask = layout.mask[:, None, :]                          # [B, 1, G]
    cost_inf = cost.masked_fill(~gmask, float("inf"))

    if kind == "HungarianLaneAssigner":
        host = cost.cpu().numpy()                            # one transfer per stage
        bs, ns, gs = [], [], []
        for b, g in enumerate(layout.counts):
            if g == 0:
                continue
            r, c = linear_sum_assignment(host[b, :, :g])
            bs.append(torch.full((len(r),), b, dtype=torch.long))
            ns.append(torch.as_tensor(r, dtype=torch.long))
            gs.append(torch.as_tensor(c, dtype=torch.long))
        if not bs:
            return empty, empty, empty
        return torch.cat(bs).to(dev), torch.cat(ns).to(dev), torch.cat(gs).to(dev)

    if kind == "TopKLaneAssigner":
        k = min(assigner.topk, num_pred)
        # k cheapest predictions per GT; exact ties go to the lower prediction index
        topk_cost, topk_idx = _smallest(cost_inf, k)                              # [B, k, G]
        candidate = torch.full_like(cost_inf, float("inf"))
        bi = torch.arange(cost.shape[0], device=dev).view(-1, 1, 1).expand_as(topk_idx)
        gi = torch.arange(cost.shape[2], device=dev).view(1, 1, -1).expand_as(topk_idx)
        candidate[bi, topk_idx, gi] = topk_cost
        best_cost, best_gt = candidate.min(dim=2)            # [B, N]
        b, n = torch.nonzero(torch.isfinite(best_cost), as_tuple=True)
        return b, n, best_gt[b, n]

    if kind == "SimOTALaneAssigner":
        iou = cache["lane_iou_dynamic"].clamp(0.0, 1.0)
        topk_ious, _ = torch.topk(iou, k=min(assigner.candidate_topk, num_pred), dim=1)
        dynamic_ks = topk_ious.sum(1).int().clamp(min=assigner.min_dynamic_k, max=num_pred)
        kmax = min(num_pred, max(assigner.candidate_topk, assigner.min_dynamic_k))
        _, pos = _smallest(cost_inf, kmax)                                     # [B, kmax, G]
        take = (torch.arange(kmax, device=dev).view(1, -1, 1) < dynamic_ks[:, None, :]) & gmask
        matching = torch.zeros(cost.shape, dtype=torch.bool, device=dev)
        bi = torch.arange(cost.shape[0], device=dev).view(-1, 1, 1).expand_as(pos)
        gi = torch.arange(cost.shape[2], device=dev).view(1, 1, -1).expand_as(pos)
        matching[bi[take], pos[take], gi[take]] = True
        multi = matching.sum(2) > 1                          # [B, N]
        best = torch.argmin(cost_inf, dim=2)                 # first minimum, as before
        one_hot = torch.zeros_like(matching)
        bb, nn_ = torch.nonzero(multi, as_tuple=True)
        one_hot[bb, nn_, best[bb, nn_]] = True
        matching = torch.where(multi[..., None], one_hot, matching)
        b, n = torch.nonzero(matching.any(2), as_tuple=True)
        return b, n, torch.argmax(matching[b, n].long(), dim=1)

    raise TypeError(f"no batched assignment for {kind}")


@torch.no_grad()
def batched_keep_mask(gate, iou, gt_index, num_gt, thr):
    """``QualityGate.keep_mask`` for pairs of the whole batch (GT ids global).

    The best pair of each GT is the first pair, in pair order, whose IoU equals
    that GT's maximum, as the loop picks it.
    """
    keep = iou >= thr
    if gate.keep_best and iou.numel() > 0:
        best = torch.full((num_gt,), -1.0, device=iou.device, dtype=iou.dtype)
        best = best.scatter_reduce(0, gt_index, iou, reduce="amax", include_self=True)
        is_best = iou >= best[gt_index]
        order = torch.arange(iou.numel(), device=iou.device)
        big = iou.numel()
        first_idx = torch.full((num_gt,), big, device=iou.device, dtype=torch.long)
        first_idx = first_idx.scatter_reduce(0, gt_index, torch.where(is_best, order, big),
                                             reduce="amin", include_self=True)
        keep = keep | (is_best & (order == first_idx[gt_index]))
    return keep


def per_image_sum(values, img, batch_size):
    """Sum of ``values`` [P] per image -> [B] (deterministic one-hot reduction)."""
    if values.numel() == 0:
        return values.new_zeros(batch_size)
    onehot = img.view(1, -1) == torch.arange(batch_size, device=img.device).view(-1, 1)
    # where, not a 0/1 product: 0 * inf would turn one image's inf into NaN for all
    return torch.where(onehot, values.view(1, -1), values.new_zeros(())).sum(1)


def per_image_spearman(conf, quality, img, batch_size):
    """Sum over images (with > 2 pairs) of Spearman(conf, quality), and their count.

    Ranks are taken within each image (ties broken by pair order).
    """
    num = conf.numel()
    if num == 0:
        z = conf.new_zeros(())
        return z, z
    count = per_image_sum(torch.ones_like(conf), img, batch_size)
    idx = torch.arange(batch_size, device=img.device)
    start = (count.view(1, -1) * (idx.view(1, -1) < idx.view(-1, 1)).to(count.dtype)).sum(1)
    positions = torch.arange(num, device=img.device)

    def ranks(v):
        by_value = torch.sort(v, stable=True).indices
        order = by_value[torch.sort(img[by_value], stable=True).indices]
        pos = torch.empty_like(order)
        pos[order] = positions
        return pos.to(conf.dtype) - start[img]

    mean = (count - 1.0) * 0.5
    rc = ranks(conf) - mean[img]
    rq = ranks(quality) - mean[img]
    cov = per_image_sum(rc * rq, img, batch_size)
    den = (per_image_sum(rc * rc, img, batch_size).sqrt()
           * per_image_sum(rq * rq, img, batch_size).sqrt())
    use = (count > 2) & (den > 0)
    corr = torch.where(use, cov / den.clamp_min(1e-12), torch.zeros_like(cov))
    return corr.sum(), use.to(conf.dtype).sum()
