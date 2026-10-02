"""Curve-aligned deformable ROI gather, rewritten.

Why it is rewritten rather than tuned
-------------------------------------
The vendored UnLanedet version had a symmetry bug that made the deformable
branch inert, which is the most likely explanation for the 64.20 -> 77.45 ->
still-below-baseline history:

``zero_init()`` zeroed ``offset_head.weight`` AND ``.bias``, so every one of the
``num_offsets - 1`` learned offsets was exactly zero and all samples for a
reference point coincided. With identical sampled features the softmax over
samples has **exactly zero** gradient (the Jacobian's columns sum to zero, and
``dL/dw_i`` is the same for every i), so ``weight_head`` never learned; and each
offset row received an identical gradient, so the offsets stayed tied to each
other forever. The branch was one bilinear sample paying for four, and
``tools/clrbezier/check_deform_init.py`` was *asserting* that condition. Both
claims are verified numerically in ``tools/clrbezier/test_deform_moe.py``.

``output_projection`` was zeroed too, which compounds it: on the first step the
gradient into the whole branch is zero, so nothing inside it can start moving.

What changed
------------
* **Offsets start on a fan.** ``offset_head.bias`` carries a fixed pattern of
  distinct, non-zero normal displacements -- alternating sign with growing
  magnitude, so the samples read "half a lane-width left, half right, a full
  width left, a full width right" along the curve normal. Deformable-DETR breaks
  the same symmetry the same way. ``init_mode="zero"`` restores the old
  behaviour for an A/B.
* **Nothing that needs gradient is zeroed.** The branch is kept quiet at init by
  a small ``residual_scale`` (0.1 by default) rather than a zero projection, so
  every parameter inside receives gradient from step 0.
* **Composed, not copied.** It subclasses the official ``ROIGather``, which
  removes ~120 duplicated lines and with them a silent divergence: the old
  ``f_key`` was conv->norm with no ReLU while the baseline's is ``conv_bn_relu``,
  so the deformable and plain gathers were not comparable.
* **Dead paths removed.** Multi-level sampling, ``shared_value_projection``,
  the B*S-batch query-to-query interaction (off in every config and quadratic in
  192 queries), and the unconditional metadata accumulation are gone. The curve
  frame is computed once instead of twice.
* **Optional MoE on the offsets**: ``moe_cfg`` routes *where to look* rather than
  how to read out, which is the placement the probe argues for when the pooled
  features turn out not to carry the curvature.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ROIGather
from .moe import MoELinear

# Config keys the vendored version accepted that no longer exist. Accepted and
# reported once rather than silently dropped, so an old config still runs and the
# person is told what stopped applying.
RETIRED_KEYS = (
    "use_conv_activation", "norm_type", "global_pool_size", "global_dropout",
    "shared_value_projection", "use_curve_point_query_interaction",
    "curve_query_interaction_stages", "curve_query_interaction_num_heads",
    "curve_query_interaction_ffn_dim", "curve_query_interaction_use_ffn",
    "deform_dropout",
)


def offset_fan(num_offsets: int, cap: float = 0.9) -> torch.Tensor:
    """Distinct, non-zero initial normal displacements, as tanh pre-activations.

    ``num_offsets`` counts every sample including index 0, which is pinned to the
    reference point itself so the branch still "starts on the curve". The
    remaining ``num_offsets - 1`` get alternating signs and growing magnitude:

        num_offsets=2 -> [+0.90]
        num_offsets=3 -> [+0.90, -0.90]
        num_offsets=4 -> [+0.45, -0.45, +0.90]
        num_offsets=5 -> [+0.45, -0.45, +0.90, -0.90]

    Returned in ``atanh`` space because the head applies ``tanh(raw) * max``, so
    a bias of ``atanh(0.45)`` yields a displacement of ``0.45 * max``. ``cap``
    keeps the magnitudes inside ``(-1, 1)`` where ``atanh`` is finite.
    """
    n = int(num_offsets) - 1
    if n <= 0:
        return torch.zeros(0)
    levels = (n + 1) // 2
    mags = [float(cap) * ((i // 2) + 1) / levels for i in range(n)]
    signs = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    pattern = torch.tensor([s * m for s, m in zip(signs, mags)], dtype=torch.float32)
    return torch.atanh(pattern.clamp(-float(cap), float(cap)))


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1,
                   eps: float = 1e-6) -> torch.Tensor:
    """Softmax over the valid entries, zero elsewhere.

    Rows with no valid entry return all zeros instead of a renormalized
    near-zero denominator, which the vendored version could produce.
    """
    neg = torch.finfo(logits.dtype).min / 4.0
    filled = logits.masked_fill(~mask, neg)
    weights = torch.softmax(filled, dim=dim) * mask.to(logits.dtype)
    denom = weights.sum(dim=dim, keepdim=True)
    return torch.where(denom > eps, weights / denom.clamp_min(eps),
                       torch.zeros_like(weights))


def build_clr_curve_reference_points(prior_xs: torch.Tensor, prior_ys: torch.Tensor,
                                     num_curve_samples: int
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sub-select S of the P CLR pooling rows as reference points.

    Signature and the ``(points, mask)`` return are unchanged from the vendored
    version, so the head's call site needs no edit.

    ``prior_xs`` is ``[B, K, P]`` normalized x, ``prior_ys`` is ``[P]``, ``[B, P]``
    or ``[B, K, P]``. Returns ``([B, K, S, 2], [B, K, S])``.
    """
    if prior_xs.ndim != 3:
        raise ValueError(f"prior_xs must be [B, K, P], got {tuple(prior_xs.shape)}")
    batch, num_queries, num_rows = prior_xs.shape
    steps = int(num_curve_samples)
    if not 2 <= steps <= num_rows:
        raise ValueError(f"num_curve_samples must be in [2, {num_rows}], got {steps}")

    xs = prior_xs.float()
    ys = prior_ys.float()
    if ys.ndim == 1:
        ys = ys.view(1, 1, -1).expand(batch, num_queries, -1)
    elif ys.ndim == 2:
        ys = ys.view(batch, 1, -1).expand(-1, num_queries, -1)
    elif ys.ndim != 3:
        raise ValueError(f"prior_ys must have 1, 2 or 3 dims, got {ys.ndim}")
    if ys.shape[-1] != num_rows:
        raise ValueError(f"prior_ys has {ys.shape[-1]} rows, prior_xs has {num_rows}")

    idx = torch.linspace(0, num_rows - 1, steps=steps, device=xs.device).round().long()
    idx = torch.unique_consecutive(idx)          # a duplicate row would give a
    if idx.numel() < steps:                      # zero-length tangent downstream
        idx = torch.arange(min(steps, num_rows), device=xs.device)
    sel_x = xs.index_select(-1, idx)
    sel_y = ys.index_select(-1, idx)
    points = torch.stack((sel_x, sel_y), dim=-1)                  # [B, K, S, 2]
    valid = (torch.isfinite(points).all(-1)
             & (points >= 0.0).all(-1) & (points <= 1.0).all(-1))
    return torch.nan_to_num(points, nan=0.0).clamp(0.0, 1.0), valid


class CurveDeformableSampler(nn.Module):
    """Attend over a fan of samples around each reference point, pool along the curve.

    Single feature level by design: every shipped config passed one, and the
    machinery that generalized over levels (a mean over per-level centre samples,
    an ``Lmax``-wide offset head plus ``index_select``) existed to support a path
    nothing took.
    """

    def __init__(self, in_channels: int, hidden_dim: int, num_curve_samples: int = 18,
                 num_offsets: int = 4, max_normal_offset: float = 2.0,
                 offset_mode: str = "normal", max_tangent_offset: float = 1.0,
                 residual_scale_init: float = 0.1, init_mode: str = "symmetry_broken",
                 offset_init_gain: float = 0.1, moe_cfg: Optional[dict] = None):
        super().__init__()
        if num_offsets < 2:
            raise ValueError("num_offsets must be >= 2 (index 0 is the reference point)")
        if offset_mode not in ("normal", "tangent_normal"):
            raise ValueError("offset_mode must be 'normal' or 'tangent_normal'")
        if init_mode not in ("symmetry_broken", "zero"):
            raise ValueError("init_mode must be 'symmetry_broken' or 'zero'")
        if init_mode == "zero" and moe_cfg:
            # The flag exists only to reproduce the old degeneracy for an A/B, and
            # that degeneracy cannot occur with experts: their weights differ, so
            # the offsets differ whatever the bias is. Zeroing expert weights to
            # force it would instead zero the router's gradient permanently.
            raise ValueError("init_mode='zero' is incompatible with moe_cfg; the "
                             "offset symmetry cannot be restored once the offsets "
                             "come from distinct experts")
        self.num_curve_samples = int(num_curve_samples)
        self.num_offsets = int(num_offsets)
        self.offset_mode = offset_mode
        self.offset_dim = 1 if offset_mode == "normal" else 2
        self.max_normal_offset = float(max_normal_offset)
        self.max_tangent_offset = float(max_tangent_offset)
        self.init_mode = init_mode
        self.offset_init_gain = float(offset_init_gain)

        self.value_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        # One projection on the concatenation rather than three summed Linear(D,D):
        # composing linear maps on a linear path buys no capacity, only parameters.
        self.point_proj = nn.Linear(hidden_dim * 2 + 6, hidden_dim)
        self.point_norm = nn.LayerNorm(hidden_dim)

        n_off = (self.num_offsets - 1) * self.offset_dim
        self.moe_cfg = dict(moe_cfg) if moe_cfg else None
        if self.moe_cfg:
            self.offset_head = MoELinear(hidden_dim, n_off, **self.moe_cfg)
        else:
            self.offset_head = nn.Linear(hidden_dim, n_off)
        self.weight_head = nn.Linear(hidden_dim, self.num_offsets)

        self.curve_dw = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.curve_pw = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.curve_norm = nn.LayerNorm(hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # Small, not zero: a zero output projection starves everything upstream
        # of it on the first step, which is half of what made the old branch inert.
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.register_buffer("fan", offset_fan(self.num_offsets), persistent=False)
        self.reset_offsets()

    def reset_offsets(self):
        """Set the offset head's starting fan. Called from ``__init__`` and from
        the gather's ``zero_init``, so a later blanket re-init cannot undo it."""
        pattern = self.fan.repeat_interleave(self.offset_dim)
        with torch.no_grad():
            if self.moe_cfg:
                weight, bias = self.offset_head.experts.weight, self.offset_head.experts.bias
                for e in range(weight.shape[0]):
                    # Re-initialize rather than scale in place, so a second call
                    # (the head invokes zero_init after its own re-init) lands on
                    # the same distribution instead of gain^2 of it.
                    nn.init.kaiming_uniform_(weight[e], a=math.sqrt(5))
                weight.mul_(self.offset_init_gain)
                if bias is not None:
                    bias.copy_(pattern.to(bias).expand_as(bias))
                return
            if self.init_mode == "zero":
                nn.init.zeros_(self.offset_head.weight)
                nn.init.zeros_(self.offset_head.bias)
                return
            nn.init.kaiming_uniform_(self.offset_head.weight, a=math.sqrt(5))
            self.offset_head.weight.mul_(self.offset_init_gain)
            self.offset_head.bias.copy_(pattern.to(self.offset_head.bias))

    @staticmethod
    def curve_frame(points: torch.Tensor, height: int, width: int):
        """Unit tangent and normal at each reference point, in pixel space.

        ``points`` is ``[B, K, S, 2]`` normalized; returns two ``[B, K, S, 2]``.
        The normalize epsilon is deliberately loose: ``F.normalize``'s Jacobian
        scales as ``1/||d||``, so a pair of coincident reference points would
        otherwise produce a gradient up to 1e6. ``build_clr_curve_reference_points``
        now de-duplicates its row index for the same reason.
        """
        scale = points.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        pix = points * scale
        diff = torch.zeros_like(pix)
        if pix.shape[-2] >= 3:
            diff[..., 1:-1, :] = pix[..., 2:, :] - pix[..., :-2, :]
        diff[..., 0, :] = pix[..., 1, :] - pix[..., 0, :]
        diff[..., -1, :] = pix[..., -1, :] - pix[..., -2, :]
        tangent = F.normalize(diff, dim=-1, eps=1e-3)
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
        return tangent, normal

    @staticmethod
    def sample(feature: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """``feature`` ``[B, D, H, W]``, ``grid`` ``[B, K, S, R, 2]`` normalized ->
        ``[B, K, S, R, D]``."""
        b, k, s, r, _ = grid.shape
        flat = (grid.reshape(b, k * s * r, 1, 2) * 2.0 - 1.0)
        out = F.grid_sample(feature, flat, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        return out.squeeze(-1).permute(0, 2, 1).reshape(b, k, s, r, -1)

    def forward(self, feature: torch.Tensor, query: torch.Tensor,
                reference_points: torch.Tensor, reference_valid_mask: torch.Tensor):
        """``feature`` ``[B, C, H, W]``, ``query`` ``[B, K, D]``,
        ``reference_points`` ``[B, K, S, 2]``, ``reference_valid_mask`` ``[B, K, S]``.

        Returns ``(context [B, K, D], stats dict)``.
        """
        b, k, s, _ = reference_points.shape
        height, width = feature.shape[-2:]
        stats = {}

        value = self.value_proj(feature)                                    # [B, D, H, W]
        centre = self.sample(value, reference_points.unsqueeze(3)).squeeze(3)  # [B,K,S,D]
        tangent, normal = self.curve_frame(reference_points, height, width)
        geometry = torch.cat((reference_points, tangent, normal), dim=-1)    # [B,K,S,6]

        point_query = self.point_norm(self.point_proj(torch.cat(
            (query.unsqueeze(2).expand(b, k, s, query.shape[-1]), centre, geometry), dim=-1)))

        if self.moe_cfg:
            raw, moe_stats = self.offset_head(point_query)
            stats.update(moe_stats)
        else:
            raw = self.offset_head(point_query)
        raw = raw.reshape(b, k, s, self.num_offsets - 1, self.offset_dim)

        pixel_scale = reference_points.new_tensor([max(width - 1, 1), max(height - 1, 1)])
        disp = normal[..., None, :] * (torch.tanh(raw[..., 0:1]) * self.max_normal_offset)
        if self.offset_mode == "tangent_normal":
            disp = disp + tangent[..., None, :] * (
                torch.tanh(raw[..., 1:2]) * self.max_tangent_offset)
        disp = torch.cat((torch.zeros_like(disp[..., :1, :]), disp), dim=-2)  # pin sample 0
        grid = reference_points[..., None, :] + disp / pixel_scale            # [B,K,S,R,2]

        inside = (torch.isfinite(grid).all(-1)
                  & (grid >= 0.0).all(-1) & (grid <= 1.0).all(-1))
        valid = reference_valid_mask[..., None] & inside                      # [B,K,S,R]
        sampled = self.sample(value, grid.clamp(0.0, 1.0))                    # [B,K,S,R,D]

        attn = masked_softmax(self.weight_head(point_query), valid, dim=-1)
        point_features = (sampled * attn[..., None]).sum(dim=3)               # [B,K,S,D]
        point_features = point_features * reference_valid_mask[..., None]

        # Depthwise-separable conv along the curve, residual.
        enc = self.curve_norm(point_features).reshape(b * k, s, -1).transpose(1, 2)
        enc = self.curve_pw(F.gelu(self.curve_dw(enc)))
        point_features = point_features + enc.transpose(1, 2).reshape(b, k, s, -1) \
            * reference_valid_mask[..., None]

        pool = masked_softmax(self.pool_score(point_features).squeeze(-1),
                              reference_valid_mask, dim=2)
        context = self.out_proj((point_features * pool[..., None]).sum(dim=2))

        with torch.no_grad():
            stats["deform_offset_px"] = disp[..., 1:, :].norm(dim=-1).mean()
            # The spread across offsets is the symmetry check: ~0 means the fan
            # collapsed and the branch is one bilinear sample again.
            spread = disp[..., 1:, 0]
            stats["deform_offset_spread"] = (spread.std(dim=-1).mean()
                                             if spread.shape[-1] >= 2
                                             else spread.new_zeros(()))
            stats["deform_attn_max"] = attn.amax(dim=-1).mean()
            stats["deform_residual_scale"] = self.residual_scale.detach()
        return context * self.residual_scale, stats


class CurveAlignedDeformableROIGather(ROIGather):
    """``ROIGather`` plus a curve-aligned deformable branch.

    Drop-in for the head: ``forward(roi_features, x, layer_index,
    reference_points=..., reference_valid_mask=...)`` returns ``[B, K, D]``.
    """

    def __init__(self, in_channels, num_priors, sample_points, fc_hidden_dim,
                 refine_layers, mid_channels=48,
                 use_deformable_curve_sampling=True, deformable_stages=None,
                 deform_num_curve_samples=18, deform_num_offsets=4,
                 deform_offset_mode="normal", deform_max_normal_offset=2.0,
                 deform_max_tangent_offset=1.0, deform_residual_scale=0.1,
                 deform_init_mode="symmetry_broken", deform_offset_init_gain=0.1,
                 deform_moe_cfg=None, **retired):
        super().__init__(in_channels, num_priors, sample_points, fc_hidden_dim,
                         refine_layers, mid_channels=mid_channels)
        ignored = sorted(k for k in retired if k in RETIRED_KEYS)
        unknown = sorted(k for k in retired if k not in RETIRED_KEYS)
        if unknown:
            raise TypeError(f"unexpected arguments: {unknown}")
        if ignored:
            print(f"[CurveAlignedDeformableROIGather] ignoring retired config keys "
                  f"{ignored}: this rewrite composes the official ROIGather, so its "
                  f"global-attention branch and norm/activation knobs come from there.")
        self.refine_layers = int(refine_layers)
        self.use_deformable = bool(use_deformable_curve_sampling)
        self.deformable_stages = set(range(self.refine_layers) if deformable_stages is None
                                     else [int(s) for s in deformable_stages])
        self.deform_curve_samples = int(deform_num_curve_samples)
        self.sampler = CurveDeformableSampler(
            in_channels, fc_hidden_dim, num_curve_samples=self.deform_curve_samples,
            num_offsets=deform_num_offsets, max_normal_offset=deform_max_normal_offset,
            offset_mode=deform_offset_mode, max_tangent_offset=deform_max_tangent_offset,
            residual_scale_init=deform_residual_scale, init_mode=deform_init_mode,
            offset_init_gain=deform_offset_init_gain, moe_cfg=deform_moe_cfg,
        ) if self.use_deformable else None
        self.last_stats: dict = {}

    def zero_init(self):
        """Kept for the head, which calls it after its own blanket re-init.

        It no longer zeroes the sampler: that is what made the branch inert. It
        re-applies the offset fan, so a ``trunc_normal_`` sweep over the head
        cannot flatten the pattern, and zeroes only the composed attention's
        output conv -- the one place a zero is correct, because it gates a
        residual whose inputs already receive gradient from the main path.
        """
        nn.init.zeros_(self.attention.W.weight)
        nn.init.zeros_(self.attention.W.bias)
        if self.sampler is not None:
            self.sampler.reset_offsets()

    def forward(self, roi_features, x, layer_index, reference_points=None,
                reference_valid_mask=None, **unused):
        roi = super().forward(roi_features, x, layer_index)          # [B, K, D]
        self.last_stats = {}
        if (self.sampler is None or int(layer_index) not in self.deformable_stages):
            return roi
        if reference_points is None:
            raise ValueError("reference_points is required on a deformable stage")
        if reference_valid_mask is None:
            reference_valid_mask = torch.ones(reference_points.shape[:3],
                                              dtype=torch.bool, device=roi.device)
        context, stats = self.sampler(x, roi, reference_points, reference_valid_mask)
        self.last_stats = stats
        return roi + context
