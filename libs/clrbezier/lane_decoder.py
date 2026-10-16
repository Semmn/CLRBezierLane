"""Sparse R-CNN-style lane decoder (bbox_head.decoder_cfg).

Every prior carries a query vector from stage to stage. At each refinement stage
the query is updated before the cls/reg towers read it:

    x = LN(q + roi)                         this stage's RoIGather feature
    x = LN(x + RelationSelfAttention(x))    the nearest other lanes and their scores
    x = LN(x + CrossAttention(x, points))   the lane's own sampled points
    x = LN(x + FFN(x))
    q = x                                   carried to the next stage

Without the self-attention, a query's score depends on its own RoI feature only.
Copies of one lane have nearly the same features, so a one-to-one classifier
cannot tell which copy to keep and NMS is needed.

The relation attention lets a query see the other lanes. Each query attends to
itself and to its ``num_neighbors`` nearest lanes by the CULane NMS distance
(mean |dx| over the rows both cover). That distance is measured on the lanes the
previous stage predicted, together with their scores; stage 0 uses the anchors
and no scores. Relation features enter the attention three ways: a per-head bias
on the logits, and offsets on the keys and the values. The value offsets let a
query learn not only that a neighbour overlaps it but whether that neighbour
scores higher. Geometry and scores are constants (no gradient).

The aux branch packs its perturbed groups into the batch dimension, one group per
row, so attention never crosses groups, as Group DETR requires. Each stage has its
own layer; the prior queries are shared by every group.

Prior art: Sparse R-CNN (Sun et al., CVPR 2021), DETR (Carion et al., 2020),
Relation Networks (Hu et al., 2018).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dedup import lane_rows

# per-neighbour relation features, see LaneRelationAttention.neighbours
REL_FEATURES = 11


@torch.no_grad()
def batched_pairwise_geometry(lanes, n_strips, x_scale, max_elems=1 << 24):
    """NMS geometry between the lanes of each row, lanes [B, Q, 6+R].

    Same quantities as dedup.pairwise_geometry, batched and chunked over B so the
    [chunk, Q, Q, R] difference tensor stays under ``max_elems`` elements.
    Returns mean |dx| (px), |dx| on the lowest and the highest common row, the
    number of common rows (all [B, Q, Q]), and the row extents (start, end) [B, Q].
    """
    batch_size, num_q = lanes.shape[:2]
    start, end = lane_rows(lanes, n_strips)
    xs = lanes[..., 6:].float() * x_scale
    num_rows = xs.shape[-1]
    r = torch.arange(num_rows, device=lanes.device).view(1, 1, 1, -1)
    chunk = max(1, int(max_elems) // max(1, num_q * num_q * num_rows))
    out = [], [], [], []
    for b0 in range(0, batch_size, chunk):
        s, e, x = start[b0:b0 + chunk], end[b0:b0 + chunk], xs[b0:b0 + chunk]
        lo = torch.maximum(s[:, :, None], s[:, None, :])
        hi = torch.minimum(e[:, :, None], e[:, None, :]).clamp(max=num_rows - 1)
        count = (hi - lo + 1).clamp(min=0)
        inside = (r >= lo[..., None]) & (r <= hi[..., None])
        diff = (x[:, :, None, :] - x[:, None, :, :]).abs()
        mean = (diff * inside).sum(-1) / count.clamp(min=1)
        d_bottom = diff.gather(-1, lo.clamp(0, num_rows - 1)[..., None]).squeeze(-1)
        d_top = diff.gather(-1, hi.clamp(0, num_rows - 1)[..., None]).squeeze(-1)
        for acc, t in zip(out, (mean, d_bottom, d_top, count)):
            acc.append(t)
    mean, d_bottom, d_top, count = (torch.cat(t, dim=0) for t in out)
    return mean, d_bottom, d_top, count, start, end


class LaneRelationAttention(nn.Module):
    """Self-attention of each query over itself and its nearest lanes."""

    def __init__(self, dim, num_heads, num_neighbors, n_strips, x_scale,
                 dist_scale=50.0, max_dist=10.0, dropout=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("decoder_cfg.num_heads must divide fc_hidden_dim")
        if num_neighbors < 1:
            raise ValueError("decoder_cfg.num_neighbors must be >= 1")
        self.num_heads = int(num_heads)
        self.num_neighbors = int(num_neighbors)
        self.n_strips = int(n_strips)
        self.x_scale = float(x_scale)
        self.dist_scale = float(dist_scale)   # NMS threshold (px): distances in its units
        self.max_dist = float(max_dist)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.rel = nn.Sequential(nn.Linear(REL_FEATURES, dim), nn.ReLU(inplace=True),
                                 nn.Linear(dim, dim))
        self.rel_k = nn.Linear(dim, dim, bias=False)
        self.rel_v = nn.Linear(dim, dim, bias=False)
        self.rel_bias = nn.Linear(dim, num_heads)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.last_self_weight = None
        self.rel_extra = None

    def add_extra_features(self, num):
        """Per-lane features (forward's extra [B, Q, num]) whose neighbour values and
        differences to the query's own join the relation features. Zero-initialised, so
        the attention starts exactly as without them."""
        dim = self.out.in_features
        self.rel_extra = nn.Linear(2 * int(num), dim)
        nn.init.zeros_(self.rel_extra.weight)
        nn.init.zeros_(self.rel_extra.bias)

    @torch.no_grad()
    def neighbours(self, lanes, scores):
        """Slot 0 is the query itself, slots 1..k its nearest lanes that share a row
        with it (fewer if fewer exist). Returns indices [B, Q, 1+k], validity and
        relation features [B, Q, 1+k, REL_FEATURES]."""
        batch_size, num_q = lanes.shape[:2]
        dev = lanes.device
        mean, d_bot, d_top, count, start, end = batched_pairwise_geometry(
            lanes, self.n_strips, self.x_scale)
        idx_q = torch.arange(num_q, device=dev)
        self_idx = idx_q.view(1, num_q, 1).expand(batch_size, num_q, 1)
        k = min(self.num_neighbors, num_q - 1)
        if k > 0:
            other = (count > 0) & (idx_q[:, None] != idx_q[None, :])
            dist = torch.where(other, mean, torch.full_like(mean, float("inf")))
            d, j = dist.topk(k, dim=-1, largest=False)
            nbr = torch.cat([self_idx, j], dim=-1)
            valid = torch.cat([torch.ones_like(self_idx, dtype=torch.bool), torch.isfinite(d)], -1)
        else:
            nbr, valid = self_idx, torch.ones_like(self_idx, dtype=torch.bool)
        n = float(self.n_strips)
        span = (end - start + 1).clamp(min=1).float()
        cnt = count.gather(2, nbr).float()
        span_j = span.gather(1, nbr.reshape(batch_size, -1)).view_as(cnt)
        start_j = start.gather(1, nbr.reshape(batch_size, -1)).view_as(cnt)
        if scores is None:
            s_i = torch.zeros((batch_size, num_q), device=dev)
            has_score = 0.0
        else:
            s_i = scores.float()
            has_score = 1.0
        s_j = s_i.gather(1, nbr.reshape(batch_size, -1)).view_as(cnt)
        is_self = torch.zeros_like(cnt)
        is_self[..., 0] = 1.0
        f = torch.stack([
            mean.gather(2, nbr) / self.dist_scale,
            d_bot.gather(2, nbr) / self.dist_scale,
            d_top.gather(2, nbr) / self.dist_scale,
            cnt / n,
            cnt / torch.minimum(span[..., None], span_j),
            (start_j - start[..., None]).float() / n,
            (span_j - span[..., None]) / n,
            s_j,
            s_j - s_i[..., None],
            torch.full_like(cnt, has_score),
            is_self,
        ], dim=-1)
        f[..., :3] = f[..., :3].clamp(max=self.max_dist)
        f = torch.where(valid[..., None], f, torch.zeros_like(f))
        return nbr, valid, f

    def forward(self, x, lanes, scores=None, extra=None):
        """x [B, Q, C]; lanes [B, Q, 6+R] (official layout); scores [B, Q] or None;
        extra [B, Q, E] or None (add_extra_features; detached here like the scores)."""
        batch_size, num_q, dim = x.shape
        heads, hd = self.num_heads, dim // self.num_heads
        nbr, valid, f = self.neighbours(lanes.detach().float(),
                                        None if scores is None else scores.detach())
        k1 = nbr.shape[-1]
        e = self.rel(f.to(x.dtype))                                     # [B, Q, 1+k, C]
        if extra is not None:
            if self.rel_extra is None:
                raise ValueError("LaneRelationAttention: extra features need add_extra_features()")
            ex = extra.detach().to(x.dtype)
            n_ex = ex.shape[-1]
            ex_j = torch.gather(ex, 1, nbr.reshape(batch_size, num_q * k1, 1).expand(-1, -1, n_ex))
            ex_j = ex_j.view(batch_size, num_q, k1, n_ex)
            ex_f = torch.cat([ex_j, ex_j - ex[:, :, None]], dim=-1)
            ex_f = torch.where(valid[..., None], ex_f, torch.zeros_like(ex_f))
            e = e + self.rel_extra(ex_f)
        flat = nbr.reshape(batch_size, num_q * k1, 1).expand(-1, -1, dim)

        def gather(t):
            return torch.gather(t, 1, flat).view(batch_size, num_q, k1, dim)

        q = self.q(x).view(batch_size, num_q, 1, heads, hd)
        key = (gather(self.k(x)) + self.rel_k(e)).view(batch_size, num_q, k1, heads, hd)
        val = (gather(self.v(x)) + self.rel_v(e)).view(batch_size, num_q, k1, heads, hd)
        logits = (q * key).sum(-1) / math.sqrt(hd) + self.rel_bias(e)   # [B, Q, 1+k, H]
        logits = logits.masked_fill(~valid[..., None], float("-inf"))
        attn = torch.softmax(logits, dim=2)
        with torch.no_grad():
            self.last_self_weight = attn[:, :, 0].mean()
        out = (self.drop(attn)[..., None] * val).sum(2).reshape(batch_size, num_q, dim)
        return self.out(out)


class PointCrossAttention(nn.Module):
    """Each query attends to the points its own lane was sampled at."""

    def __init__(self, dim, point_dim, num_points, num_heads, dropout=0.0):
        super().__init__()
        self.point_proj = nn.Linear(point_dim, dim)
        self.point_pos = nn.Parameter(torch.zeros(num_points, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, points):
        """x [B, Q, C]; points [B*Q, Cp, S, 1] (pool_prior_features)."""
        batch_size, num_q, dim = x.shape
        p = self.point_proj(points.squeeze(-1).transpose(1, 2).to(x.dtype))   # [B*Q, S, C]
        out = self.attn(x.reshape(batch_size * num_q, 1, dim), p + self.point_pos, p,
                        need_weights=False)[0]
        return out.view(batch_size, num_q, dim)


class LaneDecoderLayer(nn.Module):
    def __init__(self, dim, point_dim, num_points, n_strips, x_scale, num_heads=4,
                 num_neighbors=16, ffn_dim=256, dropout=0.0, cross_attn=True,
                 dist_scale=50.0, max_dist=10.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.self_attn = LaneRelationAttention(dim, num_heads, num_neighbors, n_strips, x_scale,
                                               dist_scale, max_dist, dropout)
        self.norm_sa = nn.LayerNorm(dim)
        self.cross_attn = (PointCrossAttention(dim, point_dim, num_points, num_heads, dropout)
                           if cross_attn else None)
        self.norm_ca = nn.LayerNorm(dim) if cross_attn else None
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.norm_ffn = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, roi, points, lanes, scores=None):
        x = self.norm_in(q + roi)
        x = self.norm_sa(x + self.drop(self.self_attn(x, lanes, scores)))
        if self.cross_attn is not None:
            x = self.norm_ca(x + self.drop(self.cross_attn(x, points)))
        return self.norm_ffn(x + self.drop(self.ffn(x)))


class LaneDecoder(nn.Module):
    """One LaneDecoderLayer per refinement stage, and the per-prior queries."""

    def __init__(self, num_priors, dim, point_dim, num_points, n_strips, img_w, num_stages,
                 num_heads=4, num_neighbors=16, ffn_dim=256, dropout=0.0, cross_attn=True,
                 dist_scale=50.0, max_dist=10.0, query_init_std=0.02):
        super().__init__()
        self.num_priors = int(num_priors)
        self.query_init_std = float(query_init_std)
        self.query = nn.Embedding(self.num_priors, dim)
        self.layers = nn.ModuleList([
            LaneDecoderLayer(dim, point_dim, num_points, n_strips, float(img_w - 1), num_heads,
                             num_neighbors, ffn_dim, dropout, cross_attn, dist_scale, max_dist)
            for _ in range(int(num_stages))])
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
            elif isinstance(m, PointCrossAttention):
                nn.init.trunc_normal_(m.point_pos, std=0.02)
        nn.init.trunc_normal_(self.query.weight, std=self.query_init_std)

    def init_queries(self, batch_size, num_q):
        if num_q != self.num_priors:
            raise ValueError(f"decoder_cfg: {num_q} queries per row, {self.num_priors} priors")
        return self.query.weight.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, stage, q, roi, points, lanes, scores=None):
        return self.layers[stage](q, roi, points, lanes, scores)

    def self_weights(self):
        """Mean attention weight each query puts on itself, per stage (last call)."""
        return [layer.self_attn.last_self_weight for layer in self.layers]
