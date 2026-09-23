"""Assigners for CLRBezierHead (ports of the V11 cached assigners).

All assigners consume one shared per-image, per-stage cost cache so the
collaborative branch never recomputes pairwise LaneIoU.

Prediction layout: CLRerNet convention (see geometry.py). Targets: official
CLR target rows [.., start_y(image y), start_x, theta, length(count), xs(px)].
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .lane_iou import pairwise_lane_iou


@torch.no_grad()
def build_cost_cache(pred, target, img_w, img_h, lane_width, lane_width_cost,
                     required=("cls_cost", "point_cost", "iou_cost_assign", "lane_iou_dynamic"),
                     cls_eps=1e-6, iou_fns=None):
    """iou_fns: optional {"dynamic": fn(pred, target),
    "cost": fn(pred, target, start, end)} to replace LaneIoU (e.g. GLIoU)."""
    """pred: [N, 6+R]; target: [M, 6+R] (valid lanes only)."""
    required = set(required)
    cache = {}
    num_gt = target.shape[0]
    pred = pred.float()
    target = target.float()
    pred_xs = pred[:, 6:]
    target_xs_pt = target[:, 6:] / float(img_w - 1)
    pred_geo = pred_xs * (float(img_w - 1) / float(img_w))
    target_geo = target[:, 6:] / float(img_w)

    if "cls_cost" in required:
        fg = F.softmax(pred[:, :2], dim=-1)[:, 1].clamp(cls_eps, 1.0 - cls_eps)
        cost = torch.nan_to_num(-torch.log(fg), nan=100.0, posinf=100.0, neginf=100.0)
        cache["cls_cost"] = cost[:, None].expand(-1, num_gt)

    if "point_cost" in required:
        t_valid = (target_xs_pt >= 0.0) & (target_xs_pt <= 1.0)
        p_oob = (pred_xs < 0.0) | (pred_xs > 1.0)
        diff = (pred_xs.clamp(0.0, 1.0)[:, None, :] - target_xs_pt.clamp(0.0, 1.0)[None]).abs()
        diff = diff + 0.5 * p_oob[:, None, :].float()
        m = t_valid[None].float()
        cache["point_cost"] = (diff * m).sum(-1) / m.sum(-1).clamp(min=1.0)

    iou_fns = iou_fns or {}
    if "lane_iou_dynamic" in required:
        fn = iou_fns.get("dynamic")
        iou = (fn(pred_geo, target_geo) if fn is not None
               else pairwise_lane_iou(pred_geo, target_geo, lane_width, img_w, img_h))
        cache["lane_iou_dynamic"] = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)

    if "iou_cost_assign" in required:
        # Official start_y is image y; LaneIoU start/end are row fractions from the bottom.
        start = (1.0 - pred[:, 2]).clamp(0.0, 1.0)
        end = (start + pred[:, 5].clamp(0.0, 1.0)).clamp(0.0, 1.0)
        fn = iou_fns.get("cost")
        iou = (fn(pred_geo, target_geo, start, end) if fn is not None
               else pairwise_lane_iou(pred_geo, target_geo, lane_width_cost, img_w, img_h, start, end))
        cache["iou_cost_assign"] = 1.0 - torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
    return cache


class _CachedCostAssigner:
    required_cache_keys = frozenset({"cls_cost", "point_cost", "iou_cost_assign"})

    def __init__(self, cls_weight=1.0, point_weight=2.0, iou_weight=3.0):
        self.cls_weight = float(cls_weight)
        self.point_weight = float(point_weight)
        self.iou_weight = float(iou_weight)

    def total_cost(self, cache):
        cost = (self.cls_weight * cache["cls_cost"]
                + self.point_weight * cache["point_cost"]
                + self.iou_weight * cache["iou_cost_assign"])
        return torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=100.0)


class HungarianLaneAssigner(_CachedCostAssigner):
    """One-to-one assignment (V11 main branch)."""

    @torch.no_grad()
    def assign(self, cache):
        cost = self.total_cost(cache)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        dev = cost.device
        return (torch.as_tensor(rows, device=dev, dtype=torch.long),
                torch.as_tensor(cols, device=dev, dtype=torch.long))


class TopKLaneAssigner(_CachedCostAssigner):
    """Top-k per GT; a prediction picked by several GTs keeps the lowest cost."""

    def __init__(self, topk=4, **kwargs):
        super().__init__(**kwargs)
        self.topk = int(topk)

    @torch.no_grad()
    def assign(self, cache):
        cost = self.total_cost(cache)
        num_pred, num_gt = cost.shape
        dev = cost.device
        if num_pred == 0 or num_gt == 0:
            empty = torch.empty(0, dtype=torch.long, device=dev)
            return empty, empty
        k = min(self.topk, num_pred)
        topk_cost, topk_idx = torch.topk(cost, k=k, dim=0, largest=False)
        candidate = torch.full_like(cost, float("inf"))
        gt_idx = torch.arange(num_gt, device=dev).view(1, -1).expand(k, -1)
        candidate[topk_idx, gt_idx] = topk_cost
        best_cost, best_gt = candidate.min(dim=1)
        rows = torch.nonzero(torch.isfinite(best_cost), as_tuple=False).squeeze(1)
        return rows, best_gt[rows]


class SimOTALaneAssigner(_CachedCostAssigner):
    """Dynamic-k assignment; k from the narrow-width LaneIoU (V11 stage-2 aux)."""

    required_cache_keys = frozenset({"cls_cost", "point_cost", "iou_cost_assign", "lane_iou_dynamic"})

    def __init__(self, candidate_topk=10, min_dynamic_k=1, **kwargs):
        super().__init__(**kwargs)
        self.candidate_topk = int(candidate_topk)
        self.min_dynamic_k = int(min_dynamic_k)

    @torch.no_grad()
    def assign(self, cache):
        cost = self.total_cost(cache)
        iou = cache["lane_iou_dynamic"].clamp(0.0, 1.0)
        num_pred, num_gt = cost.shape
        dev = cost.device
        if num_pred == 0 or num_gt == 0:
            empty = torch.empty(0, dtype=torch.long, device=dev)
            return empty, empty
        topk_ious, _ = torch.topk(iou, k=min(self.candidate_topk, num_pred), dim=0)
        dynamic_ks = topk_ious.sum(0).int().clamp(min=self.min_dynamic_k, max=num_pred)
        matching = torch.zeros(cost.shape, dtype=torch.bool, device=dev)
        for g in range(num_gt):
            _, pos = torch.topk(cost[:, g], k=int(dynamic_ks[g].item()), largest=False)
            matching[pos, g] = True
        multi = matching.sum(1) > 1
        if bool(multi.any()):
            best = torch.argmin(cost[multi], dim=1)
            matching[multi] = False
            matching[torch.nonzero(multi, as_tuple=False).squeeze(1), best] = True
        rows = torch.nonzero(matching.any(1), as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return rows, torch.empty(0, dtype=torch.long, device=dev)
        return rows, torch.argmax(matching[rows].long(), dim=1)


_ASSIGNERS = {
    "HungarianLaneAssigner": HungarianLaneAssigner,
    "TopKLaneAssigner": TopKLaneAssigner,
    "SimOTALaneAssigner": SimOTALaneAssigner,
}


def build_lane_assigner(cfg):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in _ASSIGNERS:
        raise KeyError(f"Unknown lane assigner {kind!r}; available: {sorted(_ASSIGNERS)}")
    return _ASSIGNERS[kind](**cfg)
