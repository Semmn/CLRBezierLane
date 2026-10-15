"""Converge, then decide: a decision stage on the cascade's converged lanes
(bbox_head.decider_cfg).

The refinement cascade answers "where are the lanes". Which converged copy *is*
the lane is decided here, once, after the geometry has settled, jointly over the
candidates:

    tokens   features re-pooled along each final lane from every feature level
             (no memory of the anchor or of earlier stages' RoI features)
    lane     geometry embedding + mean of its point tokens
    layers   x = LN(x + RelationSelfAttention(x))  nearest other candidates,
                                                   with this stage's own scores
                                                   from the layer before
             x = LN(x + CrossAttention(x, tokens)) the lane's own points
             x = LN(x + FFN(x))
    heads    one-to-one logit and a quality (LaneIoU) logit, after every layer

Why each part (CurveLanes stage traces, curvelanes_stage_trace_results.md):
- the cascade's own score is fixed at stage 0 ("early ownership"); re-pooling
  every RoI slot on the final lane removes the anchor's features;
- a per-prior score cannot pick one copy among near-identical copies (eval-label
  probes F1 <= 82.9; geometry-only labels: the head follows its own label at
  AUC 0.70), so the decision attends to the other candidates;
- the one-to-one target is matched on geometry first, with this stage's own
  score bounded to breaking ties (tie_cost.patch, cls_cost_mode="linear").

Geometry and the decision layers' scores enter the attention as constants.
Prior art: DETR / Deformable DETR (set prediction, deep supervision), Sparse
R-CNN, Relation Networks (learned duplicate removal), Polar R-CNN (lanes).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dedup import lane_rows
from .lane_decoder import LaneRelationAttention
from .modules import pool_prior_features


class DeciderLayer(nn.Module):
    def __init__(self, dim, num_heads, num_neighbors, n_strips, x_scale, ffn_dim,
                 dropout, dist_scale, max_dist):
        super().__init__()
        self.self_attn = LaneRelationAttention(dim, num_heads, num_neighbors, n_strips, x_scale,
                                               dist_scale, max_dist, dropout)
        self.norm_sa = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm_ca = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.norm_ffn = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, tokens, key_mask, lanes, scores):
        """x [B, Q, C]; tokens [B*Q, S, C]; key_mask [B*Q, S] (True = ignore)."""
        batch_size, num_q, dim = x.shape
        x = self.norm_sa(x + self.drop(self.self_attn(x, lanes, scores)))
        ca = self.cross_attn(x.reshape(batch_size * num_q, 1, dim), tokens, tokens,
                             key_padding_mask=key_mask, need_weights=False)[0]
        x = self.norm_ca(x + self.drop(ca.view(batch_size, num_q, dim)))
        return self.norm_ffn(x + self.drop(self.ffn(x)))


class LaneDecider(nn.Module):
    """Scores the cascade's final lanes jointly; see the module docstring."""

    def __init__(self, feat_channels, num_levels, num_points, n_strips, img_w, dim=128,
                 num_layers=2, num_heads=4, num_neighbors=16, ffn_dim=512, dropout=0.0,
                 dist_scale=50.0, max_dist=10.0, use_scores=True, prior_prob=0.01,
                 score_input=False):
        super().__init__()
        if dim % num_heads:
            raise ValueError("decider_cfg.num_heads must divide decider_cfg.dim")
        if num_layers < 1:
            raise ValueError("decider_cfg.num_layers must be >= 1")
        self.feat_channels = int(feat_channels)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.n_strips = int(n_strips)
        self.use_scores = bool(use_scores)
        self.score_input = bool(score_input)
        self.prior_prob = float(prior_prob)
        self.point_proj = nn.Linear(self.feat_channels * self.num_levels, dim)
        self.point_pos = nn.Parameter(torch.zeros(self.num_points, dim))
        self.span_embed = nn.Embedding(2, dim)   # sample row outside / inside the lane's rows
        # sample xs, in-span flags, start, length (+ the cascade's score)
        geo_dim = 2 * self.num_points + 2 + int(self.score_input)
        self.geo = nn.Sequential(nn.Linear(geo_dim, dim), nn.ReLU(inplace=True),
                                 nn.Linear(dim, dim))
        self.lane_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            DeciderLayer(dim, num_heads, num_neighbors, n_strips, float(img_w - 1), ffn_dim,
                         dropout, dist_scale, max_dist)
            for _ in range(int(num_layers))])
        self.cls_head = nn.Linear(dim, 1)       # shared by the layers (deep supervision)
        self.quality_head = nn.Linear(dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                m._reset_parameters()
        nn.init.trunc_normal_(self.point_pos, std=0.02)
        nn.init.trunc_normal_(self.span_embed.weight, std=0.02)
        # rare positives (one per lane among all candidates), as in focal-loss detectors
        bias = -math.log((1.0 - self.prior_prob) / self.prior_prob)
        nn.init.constant_(self.cls_head.bias, bias)

    @torch.no_grad()
    def sample_rows(self, lanes, sample_x_indices):
        """xs at the RoI sample rows (clamped to the image, in CLR sampling order) and
        whether each sample row lies inside the lane's own rows, both [B, Q, S]."""
        xs = lanes[..., 6:].float()
        idx = sample_x_indices.long()
        on_map = torch.nan_to_num(xs[..., idx], nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        start, end = lane_rows(lanes, self.n_strips)
        inside = (idx.view(1, 1, -1) >= start[..., None]) & (idx.view(1, 1, -1) <= end[..., None])
        # the same flip _refine applies before pool_prior_features
        return torch.flip(on_map, dims=[-1]), torch.flip(inside, dims=[-1])

    def pool(self, feats, prior_xs, prior_feat_ys):
        """Fresh features along each lane from every level: [B, Q, S, C * levels]."""
        batch_size, num_q, num_s = prior_xs.shape
        out = []
        for f in feats:
            p = pool_prior_features(f, prior_xs.to(f.dtype), prior_feat_ys, f.shape[1])
            out.append(p.view(batch_size, num_q, f.shape[1], num_s))
        return torch.cat(out, dim=2).permute(0, 1, 3, 2)

    def forward(self, feats, lanes, sample_x_indices, prior_feat_ys):
        """feats: the cascade's feature maps (any order, ``num_levels`` of them);
        lanes [B, Q, 6+R]: the final lanes (official layout, detached).
        Returns per layer the one-to-one logits [B, Q] and the quality logits [B, Q]."""
        if len(feats) != self.num_levels:
            raise ValueError(f"decider: {len(feats)} feature maps, built for {self.num_levels}")
        lanes = lanes.detach().float()
        batch_size, num_q = lanes.shape[:2]
        prior_xs, inside = self.sample_rows(lanes, sample_x_indices)
        pooled = self.pool(feats, prior_xs, prior_feat_ys).float()       # [B, Q, S, C*L]
        tok = (self.point_proj(pooled) + self.point_pos
               + self.span_embed(inside.long()))                          # [B, Q, S, D]
        dim = tok.shape[-1]
        geo = [prior_xs, inside.float(), lanes[..., 2:3].clamp(0.0, 1.0),
               lanes[..., 5:6].clamp(0.0, 1.0)]
        cascade_s = None
        if self.score_input:
            # the cascade's own score: which copy it already prefers (an identity the
            # re-pooled features and the geometry of converged copies do not carry)
            cascade_s = torch.softmax(lanes[..., :2], dim=-1)[..., 1]
            geo.append(cascade_s[..., None])
        geo_in = torch.cat(geo, dim=-1)
        w = inside.float()[..., None]
        # mean over the lane's own rows (all rows when it covers none of them)
        w = torch.where(w.sum(2, keepdim=True) > 0, w, torch.ones_like(w))
        x = self.lane_in(self.geo(geo_in) + (tok * w).sum(2) / w.sum(2))
        tokens = tok.reshape(batch_size * num_q, self.num_points, dim)
        key_mask = ~inside.reshape(batch_size * num_q, self.num_points)
        key_mask = key_mask & ~key_mask.all(-1, keepdim=True)   # never mask every key
        logits, quality, self_w = [], [], []
        scores = cascade_s   # score_input: every layer compares the cascade's scores
        for layer in self.layers:
            x = layer(x, tokens, key_mask, lanes, scores)
            z = self.cls_head(x).squeeze(-1)
            logits.append(z)
            quality.append(self.quality_head(x).squeeze(-1))
            self_w.append(layer.self_attn.last_self_weight)
            if self.use_scores and not self.score_input:
                scores = torch.sigmoid(z.detach().float())
        return dict(logits=logits, quality=quality, self_w=self_w)
