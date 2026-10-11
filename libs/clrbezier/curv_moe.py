"""Curvature-routed mixture of expert classifiers (``bbox_head.curv_moe_cfg``).

A router reads each query's RoI feature and predicts the curvature bin of the
lane the query is detecting. The main branch then scores the query with the
classifier of that bin only (hard top-1 routing). With K experts and bin edges
[e_0, ..., e_{K-2}] on the curvature magnitude |c|:

    expert 0: |c| <= e_0              (straight)
    expert k: e_{k-1} < |c| <= e_k
    expert K-1: |c| > e_{K-2}

``tools/clrbezier/pretrain_curvature_router.py`` sets the edges from the
training set as [straight, 1 sigma, 2 sigma, ...] (sigma: std of the signed
curvature over all GT lanes), trains the router on a trained checkpoint, and
writes a checkpoint the MoE run loads with ``load_from``.

Curvature of a lane (``measure``), computed on its valid rows in the network
input frame (img_w x img_h px):
    "sagitta_px": signed largest perpendicular distance from the chord joining
        the lane's bottom and top points, in px. Positive bulges to the right.
        A straight lane is 0 whatever its slope.
    "curvature": 8 * sagitta / (chord^2 + 4 * sagitta^2), signed, in 1/px: the
        curvature of the circle through the two ends and the farthest point.
        Unlike the sagitta it does not grow with lane length.

Router input (``router_input``): "feature" is the RoI feature alone; "geometry"
describes the query's own stage-output lane (its signed curvature over its
visible rows, the bin that curvature falls in, its bottom y and its span);
"both" (default) concatenates the two. The RoI feature is pooled along the
query's own reference curve, so it shows what the lane looks like relative to
that curve, not how curved the curve itself is: a router on the feature alone
learns little beyond the class prior. The geometry carries the curvature
directly. With the geometry, ``rule_prior`` adds a fixed logit to the bin of
the query's own predicted curvature and the router's last layer starts at zero,
so the untrained router routes by that rule and training only overrides it
where the labels disagree (a plain router under the ~88% straight imbalance
tends to merge the narrow 2 px .. 1 sigma bin into the straight one).

Router labels: a query is labelled with the bin of the GT lane its stage
output overlaps most, if that LaneIoU (narrow width, GT rows) is at least
``label_iou_thr``; other queries carry no label. Only the main branch is
routed; the auxiliary one-to-many branch keeps the shared classifier.

The experts start as copies of the base classifier when a checkpoint without
them is loaded, so before any MoE training the head predicts exactly what the
base checkpoint predicted.
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lane_iou import _lane_half_widths

MEASURES = ("sagitta_px", "curvature")
EXPERT_TYPES = ("tower", "linear")
ROUTER_INPUTS = ("feature", "geometry", "both")


def geometry_dim(num_experts: int) -> int:
    """Width of the geometry router input: curvature, |curvature|, the bin one-hot,
    bottom y and span."""
    return int(num_experts) + 4


@torch.no_grad()
def visible_lane_px(xs, y_start, length, prior_ys, img_w, n_strips):
    """Row x in px of predicted lanes, invalid (-1) outside each lane's visible span.

    xs: [..., R] normalized x (x_px / (img_w - 1)); y_start / length: [...] image-y
    bottom and span as in ``brr_reference`` (same clamps). Returns (xs_px, y_bottom,
    span).
    """
    one_row = 1.0 / float(n_strips)
    y_bottom = y_start.float().clamp(one_row, 1.0)
    safe_len = torch.minimum(length.float().clamp_min(2.0 * one_row), y_bottom + one_row)
    span = (safe_len - one_row).clamp_min(0.0)
    y_top = (y_bottom - span).clamp(0.0, 1.0)
    ys = prior_ys.to(device=xs.device, dtype=torch.float32)
    eps = 0.25 * one_row
    vis = (ys <= y_bottom[..., None] + eps) & (ys >= y_top[..., None] - eps)
    px = torch.nan_to_num(xs.float() * float(img_w - 1), nan=-1.0, posinf=-1.0, neginf=-1.0)
    return torch.where(vis, px, torch.full_like(px, -1.0)), y_bottom, span


@torch.no_grad()
def lane_curvature(xs_px: torch.Tensor, prior_ys: torch.Tensor, img_w: int, img_h: int,
                   measure: str = "sagitta_px") -> torch.Tensor:
    """Signed curvature measure of each lane, ``xs_px`` [..., R] -> [...].

    ``xs_px``: x in px of the img_w-wide input at the rows ``prior_ys`` (image
    y fractions, row 0 at the bottom). Rows outside [0, img_w) are invalid.
    Lanes with fewer than 3 valid rows count as straight (0).
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")
    xs = xs_px.float()
    rows = xs.shape[-1]
    valid = (xs >= 0.0) & (xs < float(img_w))
    y = prior_ys.to(device=xs.device, dtype=xs.dtype) * float(img_h)
    y = y.expand_as(xs)
    idx = torch.arange(rows, device=xs.device).expand_as(xs)
    n = valid.sum(-1)
    first = torch.where(valid, idx, torch.full_like(idx, rows)).min(-1).values.clamp(max=rows - 1)
    last = torch.where(valid, idx, torch.full_like(idx, -1)).max(-1).values.clamp(min=0)
    xa, ya = xs.gather(-1, first[..., None]), y.gather(-1, first[..., None])
    xb, yb = xs.gather(-1, last[..., None]), y.gather(-1, last[..., None])
    dx, dy = xb - xa, yb - ya
    chord = torch.sqrt(dx * dx + dy * dy)
    xs_v = torch.where(valid, xs, xa)
    # signed distance from the chord; the sign flip makes "bulges right" positive
    dist = -((xs_v - xa) * dy - (y - ya) * dx) / chord.clamp_min(1e-6)
    dist = torch.where(valid, dist, torch.zeros_like(dist))
    pick = dist.abs().argmax(-1, keepdim=True)
    sag = dist.gather(-1, pick).squeeze(-1)
    chord = chord.squeeze(-1)
    ok = (n >= 3) & (chord > 1e-6)
    out = sag if measure == "sagitta_px" else 8.0 * sag / (chord.pow(2) + 4.0 * sag.pow(2)).clamp_min(1e-6)
    return torch.where(ok, out, torch.zeros_like(out))


def curvature_bins(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Bin of each |value|: 0 for |v| <= edges[0], ..., K-1 for |v| > edges[-1]."""
    return torch.bucketize(values.abs().float(), edges.to(values.device).float(), right=False)


def edges_from_sigma(values, straight: float, sigmas: Sequence[float] = (1.0, 2.0)):
    """[straight, s1 * sigma, s2 * sigma, ...] with sigma the std of the signed values."""
    v = torch.as_tensor(values, dtype=torch.float64)
    sigma = float(v.std(unbiased=False))
    edges = [float(straight)] + [float(s) * sigma for s in sigmas]
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"bin edges must increase, got {edges} (sigma {sigma:.4g}); "
                         "lower the straight threshold or use larger sigma multiples")
    return edges, sigma


@torch.no_grad()
def batched_lane_iou(pred_geo, gt_geo, lane_width, img_w, img_h):
    """LaneIoU over the GT rows, [B, Q, R] x [B, M, R] -> [B, Q, M].

    Same convention and values as ``pairwise_lane_iou`` without start/end
    (the assigners' ``lane_iou_dynamic``). Padded GT slots (all rows invalid)
    give 0.
    """
    pred_w = _lane_half_widths(pred_geo, lane_width, img_w, img_h, detach=True)
    tgt_w = _lane_half_widths(gt_geo, lane_width, img_w, img_h, max_dx=1e4)
    p1, p2 = (pred_geo - pred_w)[:, :, None], (pred_geo + pred_w)[:, :, None]
    t1, t2 = (gt_geo - tgt_w)[:, None], (gt_geo + tgt_w)[:, None]
    ovr = torch.min(p2, t2) - torch.max(p1, t1)
    union = torch.max(p2, t2) - torch.min(p1, t1)
    invalid = ((gt_geo < 0) | (gt_geo >= 1.0))[:, None]
    ovr = ovr.masked_fill(invalid, 0.0)
    union = union.masked_fill(invalid, 0.0)
    return torch.nan_to_num(ovr.sum(-1) / (union.sum(-1) + 1e-9), nan=0.0)


@torch.no_grad()
def router_labels(pred, gt_pad, gt_bins, gt_mask, lane_width, img_w, img_h, iou_shape, thr):
    """Router label of every query, [B, Q] (long, -1 = no label).

    pred: [B, Q, 6+R] stage output (normalized xs); gt_pad: [B, M, 6+R] GT
    (xs in px, padded slots invalid); gt_bins / gt_mask: [B, M].
    """
    B, Q = pred.shape[:2]
    if gt_pad.shape[1] == 0:
        return torch.full((B, Q), -1, dtype=torch.long, device=pred.device)
    iou_w, iou_h = iou_shape
    pred_geo = pred[..., 6:].float() * (float(img_w - 1) / float(img_w))
    gt_geo = gt_pad[..., 6:].float() / float(img_w)
    iou = batched_lane_iou(pred_geo, gt_geo, lane_width, iou_w, iou_h)
    iou = iou.masked_fill(~gt_mask[:, None, :], -1.0)
    best, j = iou.max(-1)
    labels = gt_bins.gather(1, j)
    return torch.where(best >= thr, labels, torch.full_like(labels, -1))


class _ClsExpert(nn.Module):
    """One expert classifier: optionally its own copy of the cls tower, then a Linear."""

    def __init__(self, tower: Optional[nn.ModuleList], out: nn.Linear):
        super().__init__()
        self.tower = tower
        self.out = out

    def forward(self, x):
        if self.tower is not None:
            for m in self.tower:
                x = m(x)
        return self.out(x)


class CurvatureMoEClassifier(nn.Module):
    """Router + K expert classifiers with hard top-1 routing.

    Args:
        cls_modules / cls_layers: the head's base classifier. Each expert is a
            deep copy ("tower": the MLP and the Linear; "linear": the Linear
            only, after the shared MLP).
        num_experts: K; must be len(bin_edges) + 1 when edges are given.
        bin_edges: K-1 increasing edges on |curvature|, or None to take them
            from the router checkpoint.
        measure: "sagitta_px" | "curvature" (see the module docstring).
        expert: "tower" | "linear".
        router_hidden: hidden width of the router MLP (0 = a single Linear).
        router_detach: the router reads detached features, so its loss never
            reaches the shared trunk.
        router_loss_weight: weight of the router cross-entropy during MoE
            training (0 freezes the router).
        freeze_router: keep the pretrained router fixed.
        label_iou_thr: LaneIoU a query's stage output needs with a GT lane to
            get that lane's bin as its router label.
        router_input: "feature" | "geometry" | "both" (see the module docstring).
        rule_prior: logit added to the query's own-curvature bin (None: 4.0 when
            the geometry is used, else 0; needs the geometry).
    """

    def __init__(self, cls_modules: nn.ModuleList, cls_layers: nn.Linear, in_dim: int,
                 num_experts: int = 4, bin_edges=None, measure: str = "sagitta_px",
                 expert: str = "tower", router_hidden: int = 64, router_detach: bool = True,
                 router_loss_weight: float = 1.0, freeze_router: bool = False,
                 label_iou_thr: float = 0.5, router_input: str = "both",
                 rule_prior: Optional[float] = None):
        super().__init__()
        if router_input not in ROUTER_INPUTS:
            raise ValueError(f"curv_moe_cfg.router_input must be one of {ROUTER_INPUTS}, "
                             f"got {router_input!r}")
        if measure not in MEASURES:
            raise ValueError(f"curv_moe_cfg.measure must be one of {MEASURES}, got {measure!r}")
        if expert not in EXPERT_TYPES:
            raise ValueError(f"curv_moe_cfg.expert must be one of {EXPERT_TYPES}, got {expert!r}")
        self.num_experts = int(num_experts)
        if self.num_experts < 1:
            raise ValueError("curv_moe_cfg.num_experts must be >= 1")
        self.edges_from_cfg = bin_edges is not None
        if self.edges_from_cfg:
            edges = [float(e) for e in bin_edges]
            if len(edges) != self.num_experts - 1:
                raise ValueError(f"curv_moe_cfg: {len(edges)} bin edges give {len(edges) + 1} "
                                 f"bins, but num_experts={self.num_experts}")
            if any(e < 0 for e in edges) or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
                raise ValueError(f"curv_moe_cfg.bin_edges must be >= 0 and increasing: {edges}")
            edge_t = torch.tensor(edges, dtype=torch.float32)
        else:
            edge_t = torch.full((self.num_experts - 1,), float("nan"))
        self.register_buffer("bin_edges", edge_t)
        # 1 once pretrain_curvature_router.py has trained the router
        self.register_buffer("router_trained", torch.zeros((), dtype=torch.long))
        self.measure = measure
        self.expert_type = expert
        self.router_detach = bool(router_detach)
        self.router_loss_weight = float(router_loss_weight)
        self.freeze_router = bool(freeze_router) or self.router_loss_weight == 0.0
        self.label_iou_thr = float(label_iou_thr)
        self.router_input = router_input
        self.use_feature = router_input in ("feature", "both")
        self.use_geometry = router_input in ("geometry", "both")
        if rule_prior is None:
            rule_prior = 4.0 if self.use_geometry else 0.0
        if rule_prior < 0 or (rule_prior > 0 and not self.use_geometry):
            raise ValueError("curv_moe_cfg.rule_prior must be >= 0 and needs router_input "
                             "'geometry' or 'both'")
        self.rule_prior = float(rule_prior)
        in_dim = ((int(in_dim) if self.use_feature else 0)
                  + (geometry_dim(self.num_experts) if self.use_geometry else 0))
        hidden = int(router_hidden)
        self.router = (nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                     nn.Linear(hidden, self.num_experts))
                       if hidden > 0 else nn.Linear(in_dim, self.num_experts))
        tower = cls_modules if expert == "tower" else None
        self.experts = nn.ModuleList([
            _ClsExpert(copy.deepcopy(tower) if tower is not None else None,
                       copy.deepcopy(cls_layers)) for _ in range(self.num_experts)])
        if self.freeze_router:
            self.router.requires_grad_(False)
        self.capture = False  # pretrain_curvature_router.py: keep router inputs

    def edges(self) -> torch.Tensor:
        if not bool(torch.isfinite(self.bin_edges).all()):
            raise ValueError(
                "curv_moe: curvature bin edges are unknown. Load a checkpoint written by "
                "tools/clrbezier/pretrain_curvature_router.py (load_from), or set "
                "curv_moe_cfg.bin_edges.")
        return self.bin_edges

    @torch.no_grad()
    def geometry(self, xs, y_start, length, prior_ys, img_w, img_h, n_strips):
        """Geometry router input of predicted lanes [..., geometry_dim] and the bin of
        each lane's own curvature [...] (the rule a router without learning would use).

        xs: [..., R] normalized row x of the stage output; y_start / length: its
        bottom and span (prediction slots 2 and 5).
        """
        px, y_bottom, span = visible_lane_px(xs, y_start, length, prior_ys, img_w, n_strips)
        c = lane_curvature(px, prior_ys, img_w, img_h, self.measure)
        edges = self.edges()
        own_bin = curvature_bins(c, edges)
        scale = float(edges[-1]) if edges.numel() and float(edges[-1]) > 0 else 1.0
        z = (c / scale).clamp(-8.0, 8.0)
        onehot = F.one_hot(own_bin, self.num_experts).float()
        feats = torch.cat([z[..., None], z.abs()[..., None], onehot,
                           y_bottom[..., None], span[..., None]], dim=-1)
        return feats, own_bin

    def router_x(self, roi_f: Optional[torch.Tensor], geo: Optional[torch.Tensor]):
        """The router's input [N, D] from the RoI feature [N, C] and the geometry
        [N, G] (either may be None when ``router_input`` does not use it)."""
        parts = []
        if self.use_feature:
            parts.append(roi_f.detach() if self.router_detach else roi_f)
        if self.use_geometry:
            parts.append(geo.detach())
        if len(parts) == 1:
            return parts[0]
        return torch.cat([p.float() for p in parts], dim=-1)

    def logits(self, x: torch.Tensor, own_bin: Optional[torch.Tensor] = None):
        """Router logits [N, K] (float) for router input x [N, D] and the bin of each
        query's own predicted curvature [N] (needed when rule_prior > 0)."""
        out = self.router(x).float()
        if self.rule_prior > 0:
            out = out + self.rule_prior * F.one_hot(own_bin.reshape(-1), self.num_experts).to(out)
        return out

    def route(self, x: torch.Tensor, own_bin: Optional[torch.Tensor] = None):
        """Router logits [N, K] (float) and the expert index [N]."""
        logits = self.logits(x, own_bin)
        return logits, logits.argmax(-1)

    def forward(self, x: torch.Tensor, cls_f: torch.Tensor,
                own_bin: Optional[torch.Tensor] = None):
        """x: router input [N, D] (``router_x``); cls_f: classifier input [N, C] (the
        shared tower's output when expert="linear"); own_bin: [N] bin of each
        query's own predicted curvature. Returns cls logits [N, 2], router logits
        [N, K] and the route [N]."""
        logits, route = self.route(x, own_bin)
        outs = torch.stack([e(cls_f) for e in self.experts], dim=1)       # [N, K, 2]
        cls = outs.gather(1, route.view(-1, 1, 1).expand(-1, 1, outs.shape[-1])).squeeze(1)
        return cls, logits, route

    def init_experts(self):
        """Classifier-style init for the expert output layers (the head re-inits
        every Linear first); the towers keep the head's Linear init."""
        for e in self.experts:
            for p in e.out.parameters():
                nn.init.normal_(p, mean=0.0, std=1.0e-3)
        if self.rule_prior > 0:
            # the untrained router routes by the own-curvature rule
            last = self.router[-1] if isinstance(self.router, nn.Sequential) else self.router
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def lane_bins(self, xs_px, prior_ys, img_w, img_h):
        values = lane_curvature(xs_px, prior_ys, img_w, img_h, self.measure)
        return curvature_bins(values, self.edges()), values

    def router_loss(self, router_logits, preds, gt_pad, gt_mask, prior_ys, img_w, img_h,
                    lane_width, iou_shape):
        """Mean router CE over the labelled queries of all stages, and diagnostics."""
        bins, _ = self.lane_bins(gt_pad[..., 6:], prior_ys, img_w, img_h)
        bins = torch.where(gt_mask, bins, torch.zeros_like(bins))
        total, count, stats = 0.0, 0, {}
        num_queries = 0
        for s, (logit, pred) in enumerate(zip(router_logits, preds)):
            lab = router_labels(pred.detach(), gt_pad, bins, gt_mask, lane_width, img_w, img_h,
                                iou_shape, self.label_iou_thr)
            flat = logit.reshape(-1, logit.shape[-1])
            total = total + F.cross_entropy(flat, lab.reshape(-1), ignore_index=-1,
                                            reduction="sum")
            keep = lab >= 0
            count = count + keep.sum()
            num_queries += lab.numel()
            if s == len(preds) - 1:
                route = logit.detach().argmax(-1)
                n_last = keep.sum().clamp_min(1)
                stats["curv_router_acc"] = ((route == lab) & keep).sum().float() / n_last
                for k in range(self.num_experts):
                    stats[f"curv_route_e{k}"] = (route == k).float().mean()
        loss = total / torch.as_tensor(count).clamp_min(1).to(dtype=torch.float32)
        stats["curv_label_frac"] = torch.as_tensor(count).float() / max(num_queries, 1)
        n_gt = gt_mask.sum().clamp_min(1).float()
        for k in range(self.num_experts):
            # share of this batch's GT lanes in each curvature bin
            stats[f"curv_gt_e{k}"] = ((bins == k) & gt_mask).sum().float() / n_gt
        return loss, stats
