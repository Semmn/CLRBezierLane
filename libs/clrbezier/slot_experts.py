"""Lane-slot experts for the per-query head (``bbox_head.slot_expert_cfg``).

Every query is routed, at every refinement stage, by the slot of the lane it
samples at that stage: the stage's input reference (the prior at stage 0, the
previous stage's output after that). The slot is the bin of the lane's chord
angle (or x_bottom, see curv_moe.py) between fixed edges, a geometric rule with
nothing to learn. Because the route is known before the stage runs, it picks
the regression as well as the score: the expert of a slot decides where the
lane goes next and how confident it is.

Expert parts (``parts``); each expert holds its own copy of these base layers:
    "embed": RoIGather's lane embedding, fc (sample_points * C -> fc_hidden_dim)
             + LayerNorm: the layer that reads the features sampled along the lane
    "cls":   classification tower + one-to-one output layer (+ the auxiliary
             one-to-many output layer when the auxiliary branch is routed)
    "reg":   regression tower + output layer (start, length, control points,
             row offsets)
Shared by all slots: the RoI pooling and its per-stage convs (BatchNorm), the
feature-map attention, the optional query attention / GSRC / lateral modules,
and the segmentation decoder.

Routing is hard top-1. Every expert runs on every query and each query keeps
its own expert's output, so all experts take part in every step (DDP with
find_unused_parameters=False) while each query's gradient reaches its own
expert only. The experts are small (one fc and two towers), so this costs
little.

The experts start as copies of the base layers: loading a checkpoint without
experts fills each with the base weights, so before any training the head
predicts exactly what the base checkpoint predicted.
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .curv_moe import MEASURES, lane_measure, measure_bins, visible_lane_px

PARTS = ("embed", "cls", "reg")
BRANCHES = ("main", "aux")

# expert parameter prefix -> base head parameter prefix (load and init copies)
_BASE_PREFIX = {
    "embed_fc.": "roi_gather.fc.",
    "embed_norm.": "roi_gather.fc_norm.",
    "cls_tower.": "cls_modules.",
    "cls_out.": "cls_layers.",
    "aux_cls_out.": "aux_cls_layers.",
    "reg_tower.": "reg_modules.",
    "reg_out.": "reg_layers.",
}


def base_name(expert_name: str) -> str:
    """Name of the base head parameter an expert parameter is a copy of."""
    for mine, base in _BASE_PREFIX.items():
        if expert_name.startswith(mine):
            return base + expert_name[len(mine):]
    raise KeyError(expert_name)


class _SlotExpert(nn.Module):
    def __init__(self, parts, embed_fc, embed_norm, cls_tower, cls_out, aux_cls_out,
                 reg_tower, reg_out):
        super().__init__()
        if "embed" in parts:
            self.embed_fc = copy.deepcopy(embed_fc)
            self.embed_norm = copy.deepcopy(embed_norm)
        if "cls" in parts:
            self.cls_tower = copy.deepcopy(cls_tower)
            self.cls_out = copy.deepcopy(cls_out)
            self.aux_cls_out = copy.deepcopy(aux_cls_out) if aux_cls_out is not None else None
        if "reg" in parts:
            self.reg_tower = copy.deepcopy(reg_tower)
            self.reg_out = copy.deepcopy(reg_out)

    def embed(self, x):
        return self.embed_norm(self.embed_fc(x))

    def cls(self, x, aux: bool):
        for m in self.cls_tower:
            x = m(x)
        head = self.aux_cls_out if (aux and self.aux_cls_out is not None) else self.cls_out
        return head(x)

    def reg(self, x):
        for m in self.reg_tower:
            x = m(x)
        return self.reg_out(x)


class LaneSlotExperts(nn.Module):
    """K per-slot copies of the per-query head layers, routed by the lane slot.

    Args:
        num_experts: K slots; ``bin_edges`` holds K-1 increasing edges.
        bin_edges: slot edges on the measure (degrees for "angle", px for
            "x_bottom"). tools/clrbezier/pretrain_curvature_router.py prints
            the quantile edges of the GT lanes for a slot config.
        measure: "angle" | "x_bottom" (or a curvature measure of curv_moe.py).
        parts: which layers are per slot (see the module docstring).
        branches: which branches are routed. "aux" routes the perturbed
            one-to-many queries by their own slot too, so the experts get the
            auxiliary supervision and the o2o / o2m heads keep sharing
            everything but the final cls layer, as in the base head.
    """

    def __init__(self, embed_fc, embed_norm, cls_tower, cls_out, aux_cls_out, reg_tower,
                 reg_out, num_experts: int = 4, bin_edges: Optional[Sequence[float]] = None,
                 measure: str = "angle", parts: Sequence[str] = PARTS,
                 branches: Sequence[str] = BRANCHES):
        super().__init__()
        self.num_experts = int(num_experts)
        if self.num_experts < 1:
            raise ValueError("slot_expert_cfg.num_experts must be >= 1")
        if measure not in MEASURES:
            raise ValueError(f"slot_expert_cfg.measure must be one of {MEASURES}, got {measure!r}")
        self.measure = measure
        self.parts = tuple(parts)
        bad = [p for p in self.parts if p not in PARTS]
        if bad or not self.parts:
            raise ValueError(f"slot_expert_cfg.parts must be a non-empty subset of {PARTS}, "
                             f"got {list(parts)}")
        self.branches = tuple(branches)
        bad = [b for b in self.branches if b not in BRANCHES]
        if bad or "main" not in self.branches:
            raise ValueError(f"slot_expert_cfg.branches must contain 'main' and only {BRANCHES}, "
                             f"got {list(branches)}")
        if bin_edges is None:
            if self.num_experts > 1:
                raise ValueError(
                    "slot_expert_cfg.bin_edges is required (K-1 increasing edges). For lane "
                    "slots, run tools/clrbezier/pretrain_curvature_router.py on a slot "
                    "curv_moe config and copy the 'bin edges' it prints.")
            bin_edges = []
        edges = torch.as_tensor([float(e) for e in bin_edges], dtype=torch.float32)
        if edges.numel() != self.num_experts - 1:
            raise ValueError(f"slot_expert_cfg: {self.num_experts} experts need "
                             f"{self.num_experts - 1} bin edges, got {edges.tolist()}")
        if edges.numel() > 1 and not bool((edges[1:] > edges[:-1]).all()):
            raise ValueError(f"slot_expert_cfg.bin_edges must increase, got {edges.tolist()}")
        self.register_buffer("bin_edges", edges)
        self.register_buffer("measure_code", torch.tensor(MEASURES.index(measure)))
        route_aux = "aux" in self.branches
        self.experts = nn.ModuleList([
            _SlotExpert(self.parts, embed_fc, embed_norm, cls_tower, cls_out,
                        aux_cls_out if route_aux else None, reg_tower, reg_out)
            for _ in range(self.num_experts)])

    def has(self, part: str) -> bool:
        return part in self.parts

    def copy_from_base(self, head: nn.Module):
        """Make every expert a copy of the head's base layers (after init)."""
        base = dict(head.named_parameters())
        with torch.no_grad():
            for expert in self.experts:
                for name, p in expert.named_parameters():
                    p.copy_(base[base_name(name)])

    @torch.no_grad()
    def route(self, xs, y_start, length, prior_ys, img_w, img_h, n_strips):
        """Slot [...] and measure [...] of lanes given as reference rows.

        xs: [..., R] normalized row x (x_px / (img_w - 1)); y_start / length:
        the lane's bottom and span as ``brr_reference`` writes them (slots 2 and 5).
        """
        px, _, _ = visible_lane_px(xs, y_start, length, prior_ys, img_w, n_strips)
        value = lane_measure(px, prior_ys, img_w, img_h, self.measure)
        if self.num_experts == 1:
            return torch.zeros(value.shape, dtype=torch.long, device=value.device), value
        return measure_bins(value, self.bin_edges, self.measure), value

    def _pick(self, outs, route):
        if self.num_experts == 1:
            return outs[0]
        stacked = torch.stack(outs, dim=1)                                   # [N, K, D]
        idx = route.reshape(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
        return stacked.gather(1, idx).squeeze(1)

    def embed(self, x, route):
        """Lane embedding before the ReLU, [N, S*C] -> [N, D]."""
        return self._pick([e.embed(x) for e in self.experts], route)

    def cls(self, x, route, aux: bool = False):
        return self._pick([e.cls(x, aux) for e in self.experts], route)

    def reg(self, x, route):
        return self._pick([e.reg(x) for e in self.experts], route)
