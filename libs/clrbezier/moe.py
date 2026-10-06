"""Mixture of experts for the CLRBezier head: dense-soft and top-k-hard gating.

Why soft before hard
--------------------
V-MoE's sparse routing is calibrated for a regime this model is nowhere near.
Its results come from JFT-300M with on the order of 10^5-10^6 tokens per step,
all of them informative. Here a step carries 192 queries x batch 32 = 6144
tokens, of which only the assigned positives supply a regression signal -- at
``main_num_pos`` around 8 that is roughly 256 informative tokens, and splitting
those top-1 across four experts leaves about 64 each. The importance/load
losses, the noise and Batch Prioritized Routing exist to correct a *bias* when
statistics average out; at 64 tokens per expert they are fighting variance.

Dense-soft gating removes every one of those knobs. It is fully differentiable,
cannot collapse (all experts always receive gradient), needs no capacity factor
and no balance loss, and it is cheap here because an expert is a small FC, not a
transformer FFN. Then the router's own entropy tells you whether to go sparse:
it falls toward 0 if specialization is real, and sits at ln(E) if there is
nothing to specialize -- one run instead of a hyperparameter search.

Never zero-initialize an expert
-------------------------------
The router's gradient is ``J_softmax @ (expert_outputs @ dL/dy)``. If the expert
outputs are identical -- which they are when the experts share weights, and when
they are all zeroed -- that product is **exactly zero** and the router never
learns, forever. Verified numerically alongside the deformable-sampler bug that
has the same shape (``curve_deformable_roi_gather`` zeroed ``offset_head``, so
every sample coincided, so the attention over samples had exactly zero gradient
and all offsets stayed tied).

So: the *router* is zero-initialized (a uniform gate, no shock at step 0) and
the *experts* keep ordinary diverse initialization. That ordering is what gives
the router a non-degenerate gradient from the first step. Note also that a gate
frozen at uniform is algebraically one averaged layer -- ``sum_e (1/E) W_e x ==
mean(W) x`` -- so the capacity only appears once the gate varies with the input.
``moe_entropy`` dropping below ln(E) is the evidence that it does.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_MODES = ("soft", "hard")


def cv_squared(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Squared coefficient of variation: 0 when uniform, E-1 when fully collapsed.

    Shazeer et al.'s balance penalty. Reported for both gating modes, but only
    added to the loss in hard mode -- soft gating cannot starve an expert.
    """
    if x.numel() <= 1:
        return x.new_zeros(())
    return x.float().var(unbiased=False) / (x.float().mean() ** 2 + eps)


class MoEGate(nn.Module):
    """Routes a feature to E experts.

    Args:
        dim: feature width the router reads.
        num_experts: E. 2-4 is the useful range here; past that the informative
            token count per expert drops below anything trainable.
        mode: "soft" for a dense softmax over all experts (recommended first),
            "hard" for top-k. Hard mode computes every expert and masks, so the
            arithmetic matches true sparse dispatch exactly while the speed win
            does not materialise -- which is the right trade while the expert is
            a small FC and you are measuring whether routing helps at all.
        top_k: experts kept in hard mode.
        temperature: logits are divided by this. Above 1 softens the gate, below
            1 sharpens it. Leave at 1 and read the entropy instead of tuning it.
        noise_std: Gaussian noise on the logits during training, hard mode only
            (Shazeer's exploration term). 0 disables it.
        balance_weight: weight on the cv_squared penalty returned in the stats.
            Only meaningful in hard mode.
    """

    def __init__(self, dim: int, num_experts: int, mode: str = "soft",
                 top_k: int = 1, temperature: float = 1.0, noise_std: float = 0.0,
                 balance_weight: float = 0.01):
        super().__init__()
        if mode not in GATE_MODES:
            raise ValueError(f"mode must be one of {GATE_MODES}, got {mode!r}")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_experts = int(num_experts)
        self.mode = mode
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.balance_weight = float(balance_weight)
        self.proj = nn.Linear(dim, self.num_experts, bias=False)
        self.zero_init()

    def zero_init(self):
        """Uniform gate at step 0. The experts are NOT touched -- see the module
        docstring: identical or zeroed experts give the router zero gradient."""
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """``x`` is ``[..., dim]``; returns gate weights ``[..., E]`` and stats."""
        logits = self.proj(x.float()) / self.temperature
        if self.mode == "hard" and self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        if self.mode == "soft" or self.top_k >= self.num_experts:
            weights = probs
        else:
            top_val, top_idx = logits.topk(self.top_k, dim=-1)
            if self.top_k == 1:
                # Switch Transformer: keep the chosen expert's own probability.
                # Renormalizing a single value gives a constant 1.0, which has
                # exactly zero gradient, so the router (and the load term built
                # from these weights) would never learn from the task.
                top_w = probs.gather(-1, top_idx)
            else:
                # Renormalize within the selected set, then scatter back. Gradient
                # reaches the router through these values; the argmax itself is
                # not differentiable, which is standard for sparse MoE.
                top_w = F.softmax(top_val, dim=-1)
            weights = torch.zeros_like(probs).scatter(-1, top_idx, top_w)

        stats = self._stats(probs, weights)
        return weights.to(x.dtype), stats

    @torch.no_grad()
    def _diagnostics(self, probs, weights):
        flat_p = probs.reshape(-1, self.num_experts)
        flat_w = weights.reshape(-1, self.num_experts)
        entropy = -(flat_p * (flat_p + 1e-12).log()).sum(-1).mean()
        return {
            # ln(E) means a uniform gate (nothing specialized); 0 means one-hot.
            "moe_entropy": entropy,
            "moe_entropy_frac": entropy / math.log(max(self.num_experts, 2)),
            "moe_max_gate": flat_w.max(-1).values.mean(),
            # Fraction of tokens each expert actually receives.
            "moe_load_cv2": cv_squared((flat_w > 0).float().mean(0)),
            "moe_importance_cv2": cv_squared(flat_p.mean(0)),
        }

    def _stats(self, probs, weights) -> dict:
        stats = dict(self._diagnostics(probs, weights))
        if self.mode == "hard" and self.balance_weight > 0:
            flat_p = probs.reshape(-1, self.num_experts)
            flat_w = weights.reshape(-1, self.num_experts)
            # Differentiable: importance through probs, load through the gate
            # values of the chosen experts (a soft surrogate for the count).
            importance = flat_p.mean(0)
            load = flat_w.mean(0)
            stats["moe_balance_loss"] = self.balance_weight * (
                cv_squared(importance) + cv_squared(load))
        return stats


class _ExpertStack(nn.Module):
    """E weight matrices applied in parallel, in one einsum.

    Each expert gets its own ordinary initialization. The spread is deliberate
    and load-bearing: identical experts make the router's gradient vanish.
    """

    def __init__(self, num_experts: int, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(num_experts, out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for e in range(self.weight.shape[0]):
            nn.init.kaiming_uniform_(self.weight[e], a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[-1]
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., in]`` -> ``[..., E, out]``."""
        # Align dtypes explicitly. Under AMP the activation arrives as fp16 while
        # the parameter stays fp32; einsum is on the autocast list so it would
        # usually be handled, but outside an autocast region (an eval pass, a
        # diagnostic script, a hook) mixed dtypes raise. The comparison makes this
        # a no-op in the common case.
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        out = torch.einsum("...i,eoi->...eo", x, weight)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    @torch.no_grad()
    def divergence(self) -> torch.Tensor:
        """Mean pairwise relative distance between experts.

        Post-training this is the cheapest answer to "did anything specialize?".
        Near 0 means the experts collapsed onto each other and the mixture is
        one layer wearing E hats.
        """
        w = self.weight.flatten(1).float()
        n = w.shape[0]
        if n < 2:
            return w.new_zeros(())
        d = torch.cdist(w, w) / (w.norm(dim=1, keepdim=True) + 1e-9)
        return d[~torch.eye(n, dtype=torch.bool, device=w.device)].mean()


class MoELinear(nn.Module):
    """Conditionally parameterized linear layer: ``y = sum_e g_e(x) W_e x``.

    Bilinear in the gate and the input, so it is strictly more expressive than
    any single ``Linear`` -- this is CondConv / Dynamic Convolution, not V-MoE.
    That lineage matters: those were built for efficient networks at MobileNet
    scale, which is the regime a lane-detection head actually lives in.

    The router reads ``x`` unless ``route_on`` is passed to ``forward``, which
    lets the gate see something more informative than the layer's own input
    (a previous stage's geometry, say).
    """

    def __init__(self, in_dim: int, out_dim: int, num_experts: int = 4,
                 gate_dim: Optional[int] = None, bias: bool = True, **gate_kwargs):
        super().__init__()
        self.experts = _ExpertStack(num_experts, in_dim, out_dim, bias=bias)
        self.gate = MoEGate(gate_dim or in_dim, num_experts, **gate_kwargs)
        self.num_experts = int(num_experts)

    def forward(self, x, route_on: Optional[torch.Tensor] = None):
        weights, stats = self.gate(x if route_on is None else route_on)
        out = (self.experts(x) * weights.unsqueeze(-1)).sum(dim=-2)
        stats["moe_expert_divergence"] = self.experts.divergence()
        return out, stats


class MoEFFN(nn.Module):
    """E parallel two-layer MLP experts -- the V-MoE placement, for token stages.

    Use this where there are real tokens to process (GSRC). On a leaf read-out
    ``MoELinear`` is the better-matched construction.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 num_experts: int = 4, dropout: float = 0.0,
                 act_layer=nn.GELU, out_dim: Optional[int] = None, **gate_kwargs):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        out_dim = out_dim or dim
        self.fc1 = _ExpertStack(num_experts, dim, hidden_dim)
        self.fc2 = _ExpertStack(num_experts, hidden_dim, out_dim)
        self.act = act_layer()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gate = MoEGate(dim, num_experts, **gate_kwargs)
        self.num_experts = int(num_experts)

    def forward(self, x, route_on: Optional[torch.Tensor] = None):
        weights, stats = self.gate(x if route_on is None else route_on)
        h = self.drop(self.act(self.fc1(x)))                      # [..., E, hidden]
        # fc2 holds one matrix per expert and h already carries the expert axis,
        # so contract per-expert rather than through _ExpertStack.forward.
        w2 = self.fc2.weight
        w2 = w2 if w2.dtype == h.dtype else w2.to(h.dtype)
        out = torch.einsum("...eh,eoh->...eo", h, w2)
        if self.fc2.bias is not None:
            out = out + self.fc2.bias.to(out.dtype)
        out = (out * weights.unsqueeze(-1)).sum(dim=-2)
        stats["moe_expert_divergence"] = self.fc1.divergence()
        return out, stats


def merge_moe_stats(target: dict, stats: dict, prefix: str = "") -> dict:
    """Accumulate stats from several MoE layers by mean, for the head's log dict.

    Keys are taken from whatever the gate reports rather than hardcoded, so a
    gate and a head from different versions cannot disagree about the key set --
    the same mistake that produced a KeyError on ``rank_logit_gap``.
    """
    for key, value in stats.items():
        name = f"{prefix}{key}" if prefix else key
        if name in target:
            target[name] = target[name] + value.detach()
            target[f"__n_{name}"] = target.get(f"__n_{name}", 1) + 1
        else:
            target[name] = value.detach()
    return target


def finalize_moe_stats(target: dict) -> dict:
    """Divide the accumulated sums by their counts and drop the bookkeeping."""
    counts = {k[4:]: v for k, v in target.items() if k.startswith("__n_")}
    out = {k: v for k, v in target.items() if not k.startswith("__n_")}
    for key, n in counts.items():
        if key in out:
            out[key] = out[key] / n
    return out
