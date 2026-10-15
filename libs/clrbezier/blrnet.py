"""BLRNet, the Bezier iterative lane detector (bbox_head type "BLRHead").

A lane detector without the cross-layer refinement cascade. YSY's design
(2026-10-15, blrnet_design_review.md):

  anchors     N learned anchors, each a content embedding and a free 2D cubic
              Bezier (control-point x free, y ordered bottom -> top, so the
              curve is monotone in y like every lane). No row representation.
  sampling    once: S points on the anchor curve, every FPN level at each point
              (optionally with learned per-point offsets, align_cfg).
  semantic    embedding + geometry encoding; relation self-attention over the
  branch      nearest anchors; cross-attention to the image tokens (coarsest
              level + 2D sine position) and to the anchor's own point tokens.
  geometry    the sampled points as one vector; relation self-attention;
  branch      cross-attention to the semantic set. Fused anchor feature.
  refinement  T steps of a GRU over the fused feature ("self-feedback"); each
              step updates the control points and the per-point lateral offsets
              of the output curve (the residual branch, with its own loss),
              optionally after re-sampling the finest level along the current
              curve. Deep supervision on every step.
  heads       o2m classification (per anchor) and, after one more relation
              self-attention over the converged lanes with the o2m scores as
              neighbour features, o2o classification: NMS-free at test.
  targets     geometry first: Hungarian (o2o) and SimOTA (o2m) with the score
              bounded to breaking ties (cls_cost_mode="linear"); LaneIoU on the
              rows, point L1 against the GT curve sampled at the same t,
              control-point smooth-L1 against the chord-length GT fit; focal
              classification. Optional feature alignment (align_cfg): anchors
              assigned to GT before any prediction (k nearest in mean |dx|),
              their per-point sampling offsets supervised by the lateral distance
              to the GT and/or their sampled features by the (stop-gradient)
              features sampled on the GT.

Output is the official CLRerNet layout [cls(2), start_y, start_x, theta, length,
xs(R)], so evaluation, the threshold sweep and the NMS are unchanged.

Prior art: Polar R-CNN (sample once, o2o head over interacting anchors), Sparse
R-CNN (content embeddings), BezierLaneNet / BezierFormer (Bezier lanes, sampled
point loss), DETR / Deformable DETR (set matching, iterative decoding).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from .assigners import build_cost_cache, build_lane_assigner
from .data_adapters import CLRTargetAdapter
from .dedup import lane_rows
from .geometry import bernstein_basis
from .head import StagePredictions, _OfficialHead, _smooth_l1
from .lane_decoder import LaneRelationAttention
from .lane_iou import LaneIoULoss
from .modules import SegDecoder
from .priors import clrernet_style_bezier_priors


# ------------------------------------------------------------------ geometry
def anchor_control_points(params):
    """params [..., 8] = [x0..x3, y0, d1..d3] -> control points [..., 4, 2]
    (x, y), bottom -> top, with y_k = y0 - sum_{i<=k} softplus(d_i)."""
    x = params[..., :4]
    y0 = params[..., 4:5]
    inc = F.softplus(params[..., 5:8])
    y = torch.cat([y0, y0 - torch.cumsum(inc, dim=-1)], dim=-1)
    return torch.stack([x, y], dim=-1)


def params_from_control_points(cps, min_inc=1e-4):
    """Inverse of anchor_control_points for control points with ordered y."""
    x, y = cps[..., 0], cps[..., 1]
    inc = (y[..., :-1] - y[..., 1:]).clamp_min(min_inc)
    d = torch.log(torch.expm1(inc))
    return torch.cat([x, y[..., :1], d], dim=-1)


def curve_points(cps, t):
    """cps [..., 4, 2], t [P] -> points [..., P, 2]."""
    basis = bernstein_basis(t.to(cps.dtype))                    # [P, 4]
    return torch.matmul(basis.expand(*cps.shape[:-2], -1, -1), cps)


def bezier_end_slopes(cps, eps=1e-4):
    """dx per unit of y towards the top at the two ends of cubic Bezier curves
    cps [..., 4, 2] (bottom -> top): the tangent directions 3(P1 - P0), 3(P3 - P2)."""
    x, y = cps[..., 0], cps[..., 1]
    s_bot = (x[..., 1] - x[..., 0]) / (y[..., 0] - y[..., 1]).clamp_min(eps)
    s_top = (x[..., 3] - x[..., 2]) / (y[..., 2] - y[..., 3]).clamp_min(eps)
    return torch.stack([s_bot, s_top], dim=-1)


def curve_to_rows(pts, prior_ys, n_strips, slope_max=3.0, end_slopes=None):
    """Row representation of a 2D polyline pts [..., P, 2] (bottom -> top).

    Returns xs [..., R] at prior_ys, start_y [...] and length [...] in the
    official convention. Rows inside the curve's y-span interpolate the polyline
    exactly (gradient to both ends of the segment). Rows outside continue the
    curve along a detached, clamped slope: end_slopes [..., 2] (bottom, top; the
    Bezier tangents, smooth) when given, else the end segments. Their gradient
    is a lateral translation of the end point only, so the extent is learned
    from the extent and control-point losses, not from far rows."""
    x, y = pts[..., 0], pts[..., 1]
    y = torch.cummin(y, dim=-1).values                           # monotone non-increasing
    lead, num_p = y.shape[:-1], y.shape[-1]
    yf, xf = y.reshape(-1, num_p), x.reshape(-1, num_p)
    a = (-yf).contiguous()
    rows = prior_ys.to(a.dtype)
    v = (-rows).view(1, -1).expand(a.shape[0], -1).contiguous()
    hi = torch.searchsorted(a, v).clamp(1, num_p - 1)
    lo = hi - 1
    x_lo, x_hi = xf.gather(1, lo), xf.gather(1, hi)
    y_lo, y_hi = yf.gather(1, lo), yf.gather(1, hi)
    yr = rows.view(1, -1)
    dy = (y_lo - y_hi).clamp_min(1e-4)
    inside = (yr <= y_lo) & (yr >= y_hi)
    x_in = x_lo + (y_lo - yr) / dy * (x_hi - x_lo)
    if end_slopes is None:
        slope = ((x_hi - x_lo) / dy).clamp(-slope_max, slope_max).detach()
    else:
        s = end_slopes.reshape(-1, 2).detach().clamp(-slope_max, slope_max)
        slope = torch.where(yr > y_lo, s[:, :1], s[:, 1:])
    x_out = x_lo + (y_lo - yr).detach() * slope
    xs = torch.where(inside, x_in, x_out).reshape(*lead, -1)
    y_bot = y[..., 0].clamp(0.0, 1.0)
    y_top = y[..., -1].clamp(0.0, 1.0)
    length = (y_bot - y_top).clamp_min(0.0) + 1.0 / float(n_strips)
    return xs, y_bot, length


@torch.no_grad()
def fit_gt_bezier(gt, prior_ys, img_w, img_h, ridge=1e-2):
    """Chord-length least-squares cubic Bezier fit of GT lanes [G, 6+R] (official
    rows, px). Returns control points [G, 4, 2] (normalized x, y; bottom -> top)
    and a validity mask (>= 2 rows). The ridge pulls toward the chord."""
    xs_px = gt[:, 6:].float()
    valid = (xs_px >= 0.0) & (xs_px < float(img_w))
    num_g, num_r = xs_px.shape
    x = xs_px / float(img_w - 1)
    y = prior_ys.to(x.dtype).view(1, -1).expand(num_g, -1)
    px, py = x * float(img_w - 1), y * float(img_h - 1)
    both = (valid[:, 1:] & valid[:, :-1]).float()
    seg = torch.sqrt((px[:, 1:] - px[:, :-1]) ** 2 + (py[:, 1:] - py[:, :-1]) ** 2) * both
    cum = torch.cat([seg.new_zeros(num_g, 1), torch.cumsum(seg, dim=1)], dim=1)
    first = torch.argmax(valid.int(), dim=1)
    last = num_r - 1 - torch.argmax(valid.flip(1).int(), dim=1)
    c0 = cum.gather(1, first[:, None])
    total = (cum.gather(1, last[:, None]) - c0).clamp_min(1e-6)
    t = ((cum - c0) / total).clamp(0.0, 1.0)
    basis = bernstein_basis(t)                                   # [G, R, 4]
    w = valid.float()[..., None]
    pts = torch.stack([x, y], dim=-1)                            # [G, R, 2]
    bw = (basis * w).transpose(1, 2)                             # [G, 4, R]
    eye = torch.eye(4, device=gt.device, dtype=x.dtype)
    p_first = pts.gather(1, first.view(-1, 1, 1).expand(-1, 1, 2))
    p_last = pts.gather(1, last.view(-1, 1, 1).expand(-1, 1, 2))
    frac = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], device=gt.device, dtype=x.dtype)
    c_lin = p_first + (p_last - p_first) * frac.view(1, 4, 1)
    cps = torch.linalg.solve(bw @ basis + ridge * eye, bw @ pts + ridge * c_lin)
    return cps, valid.sum(1) >= 2


@torch.no_grad()
def gt_x_at_y(gt, y_q, img_w, n_strips):
    """x (normalized) of GT lanes [G, 6+R] at the image y's y_q [G, S], and whether
    the GT covers that y."""
    xs_px = gt[:, 6:].float()
    valid = (xs_px >= 0.0) & (xs_px < float(img_w))
    r = (1.0 - y_q.float()) * float(n_strips)
    lo = r.floor().clamp(0, n_strips - 1).long()
    hi = lo + 1
    frac = (r - lo.float()).clamp(0.0, 1.0)
    x = (xs_px.gather(1, lo) * (1.0 - frac) + xs_px.gather(1, hi) * frac) / float(img_w - 1)
    ok = valid.gather(1, lo) & valid.gather(1, hi) & (r >= 0.0) & (r <= float(n_strips))
    return x, ok


def sample_points(feat, xy):
    """feat [B, C, H, W]; xy [B, N, S, 2] normalized (x, y) -> [B, N, S, C]."""
    grid = (xy * 2.0 - 1.0).to(feat.dtype)
    return F.grid_sample(feat, grid, align_corners=True).permute(0, 2, 3, 1)


def sine_position_2d(h, w, dim, device, dtype, temperature=10000.0):
    """Normalized 2D sine embedding [h*w, dim] (dim divisible by 4)."""
    y = torch.linspace(0.0, 1.0, h, device=device, dtype=torch.float32) * 2.0 * math.pi
    x = torch.linspace(0.0, 1.0, w, device=device, dtype=torch.float32) * 2.0 * math.pi
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    n = dim // 4
    freq = temperature ** (torch.arange(n, device=device, dtype=torch.float32) / n)
    py, px = yy.reshape(-1, 1) / freq, xx.reshape(-1, 1) / freq
    return torch.cat([py.sin(), py.cos(), px.sin(), px.cos()], dim=-1).to(dtype)


def focal_2cls(logits, target, alpha, gamma):
    """Per-sample softmax focal loss, logits [..., 2], target [...] in {0, 1}."""
    logp = F.log_softmax(logits.float(), dim=-1)
    logp_t = logp.gather(-1, target.long()[..., None]).squeeze(-1)
    return -alpha * (1.0 - logp_t.exp()).pow(gamma) * logp_t


# ------------------------------------------------------------------ modules
class _MLP(nn.Sequential):
    def __init__(self, dim_in, dim, dim_out=None):
        super().__init__(nn.Linear(dim_in, dim), nn.ReLU(inplace=True),
                         nn.Linear(dim, dim_out or dim))


class BLRBlock(nn.Module):
    """One semantic / geometry interaction block (design items 2.1, 2.2)."""

    def __init__(self, dim, num_heads, num_neighbors, n_strips, x_scale, ffn_dim, dropout,
                 dist_scale, max_dist):
        super().__init__()
        rel = dict(dim=dim, num_heads=num_heads, num_neighbors=num_neighbors, n_strips=n_strips,
                   x_scale=x_scale, dist_scale=dist_scale, max_dist=max_dist, dropout=dropout)
        self.sem_sa = LaneRelationAttention(**rel)
        self.sem_img = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.sem_pts = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.sem_ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
                                     nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.geo_sa = LaneRelationAttention(**rel)
        self.geo_sem = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.geo_ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
                                     nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(7)])
        self.drop = nn.Dropout(dropout)

    def forward(self, sem, geo, tok, img_tok, lanes):
        n = self.norms
        batch_size, num_q, dim = sem.shape
        sem = n[0](sem + self.drop(self.sem_sa(sem, lanes)))
        sem = n[1](sem + self.drop(self.sem_img(sem, img_tok, img_tok, need_weights=False)[0]))
        q = sem.reshape(batch_size * num_q, 1, dim)
        k = tok.reshape(batch_size * num_q, -1, dim)
        sem = n[2](sem + self.drop(self.sem_pts(q, k, k, need_weights=False)[0].view_as(sem)))
        sem = n[3](sem + self.drop(self.sem_ffn(sem)))
        geo = n[4](geo + self.drop(self.geo_sa(geo, lanes)))
        geo = n[5](geo + self.drop(self.geo_sem(geo, sem, sem, need_weights=False)[0]))
        geo = n[6](geo + self.drop(self.geo_ffn(geo)))
        return sem, geo


def ms_deform_sample(values, locs, weights):
    """Pure-PyTorch core of multi-scale deformable attention (Deformable DETR). values: per
    level [BH, C, H_l, W_l]; locs: per level [BH, Q, K, 2] normalized (x, y) with the
    align_corners=True convention of sample_points; weights [BH, Q, L*K] (softmaxed,
    level-major) -> [BH, Q, C]."""
    s = torch.cat([F.grid_sample(v, (g * 2.0 - 1.0).to(v.dtype), align_corners=True)
                   for v, g in zip(values, locs)], dim=-1)                       # [BH, C, Q, L*K]
    return torch.einsum("bcqk,bqk->bqc", s, weights.to(s.dtype))


def _deform_grid_bias(num_heads, num_levels, num_points):
    """Deformable DETR's offset initialization: head h looks in direction 2*pi*h/H, its
    k-th sample k+1 cells away (in units of the level's grid)."""
    th = torch.arange(num_heads, dtype=torch.float32) * (2.0 * math.pi / num_heads)
    d = torch.stack([th.cos(), th.sin()], dim=-1)
    d = d / d.abs().max(-1, keepdim=True).values                                  # [H, 2]
    k = torch.arange(1, num_points + 1, dtype=torch.float32)
    return (d[:, None, None, :] * k[None, None, :, None]).expand(num_heads, num_levels,
                                                                 num_points, 2).reshape(-1)


class CurveDeformAttention(nn.Module):
    """BezierFormer-style curve attention: a lane query samples the neck features at K
    learned offsets around each of its S reference points on the current curve, on the
    finest `num_levels` levels, with one softmax over all S*L*K samples per head. The
    offsets and weights are predicted per reference point from query + point embedding."""

    def __init__(self, dim, value_channels, num_levels, num_heads, num_points, x_scale):
        super().__init__()
        if value_channels % num_heads:
            raise ValueError("deform_heads must divide the neck channels")
        self.heads, self.levels, self.points = int(num_heads), int(num_levels), int(num_points)
        self.c_h = value_channels // self.heads
        self.x_scale = float(x_scale)
        self.value_proj = nn.ModuleList([nn.Conv2d(value_channels, value_channels, 1)
                                         for _ in range(self.levels)])
        self.offsets = nn.Linear(dim, self.heads * self.levels * self.points * 2)
        self.weights = nn.Linear(dim, self.heads * self.levels * self.points)
        self.out = nn.Linear(value_channels, dim)
        self.last_offset_px = None

    def reset_parameters(self):
        nn.init.zeros_(self.offsets.weight)
        with torch.no_grad():
            self.offsets.bias.copy_(_deform_grid_bias(self.heads, self.levels, self.points))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        for conv in self.value_proj:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q, pos, ref, feats):
        """q [B, N, D]; pos [S, D]; ref [B, N, S, 2] normalized; feats: the `levels` maps
        [B, C, H_l, W_l] (coarsest first) -> [B, N, D]."""
        batch_size, num_q, dim = q.shape
        S, H, L, K = ref.shape[2], self.heads, self.levels, self.points
        qs = q[:, :, None, :] + pos[None, None]                                    # [B, N, S, D]
        off = self.offsets(qs).float().view(batch_size, num_q, S, H, L, K, 2)
        w = self.weights(qs).float().view(batch_size, num_q, S, H, L, K)
        w = torch.softmax(w.permute(0, 3, 1, 4, 2, 5).reshape(batch_size * H, num_q, L * S * K), dim=-1)
        values, locs, px = [], [], []
        for l, f in enumerate(feats):
            h_l, w_l = f.shape[-2:]
            v = self.value_proj[l](f).view(batch_size, H, self.c_h, h_l, w_l)
            values.append(v.reshape(batch_size * H, self.c_h, h_l, w_l))
            cell = off.new_tensor([1.0 / max(w_l - 1, 1), 1.0 / max(h_l - 1, 1)])
            g = ref[:, :, :, None, None, :].float() + off[..., l, :, :] * cell          # [B, N, S, H, K, 2]
            locs.append(g.permute(0, 3, 1, 2, 4, 5).reshape(batch_size * H, num_q, S * K, 2))
            px.append(off[..., l, :, 0].detach().abs().mean() * cell[0] * self.x_scale)
        out = ms_deform_sample(values, locs, w)                                      # [BH, N, c_h]
        out = out.view(batch_size, H, num_q, self.c_h).permute(0, 2, 1, 3).reshape(batch_size, num_q, H * self.c_h)
        self.last_offset_px = torch.stack(px).mean()
        return self.out(out)


class StepBlock(nn.Module):
    """Option B: a decoder block inside one refinement step: relation self-attention over
    the current lanes, curve attention at the current geometry, FFN (its own weights)."""

    def __init__(self, dim, rel, deform, ffn_dim, dropout):
        super().__init__()
        self.sa = LaneRelationAttention(dim=dim, **rel)
        self.curve = CurveDeformAttention(dim, **deform)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.drop = nn.Dropout(dropout)

    def forward(self, h, lanes, pos, ref, feats):
        n = self.norms
        h = n[0](h + self.drop(self.sa(h, lanes)))
        h = n[1](h + self.drop(self.curve(h, pos, ref, feats)))
        return n[2](h + self.drop(self.ffn(h)))


class DeformableEncoderLayer(nn.Module):
    """One Deformable-DETR encoder layer over the tokens of several levels."""

    def __init__(self, dim, num_levels, num_heads, num_points, ffn_dim, dropout):
        super().__init__()
        if dim % num_heads:
            raise ValueError("encoder_cfg.dim must be divisible by encoder_cfg.num_heads")
        self.heads, self.levels, self.points = int(num_heads), int(num_levels), int(num_points)
        self.c_h = dim // self.heads
        self.value_proj = nn.Linear(dim, dim)
        self.offsets = nn.Linear(dim, self.heads * self.levels * self.points * 2)
        self.weights = nn.Linear(dim, self.heads * self.levels * self.points)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.zeros_(self.offsets.weight)
        with torch.no_grad():
            self.offsets.bias.copy_(_deform_grid_bias(self.heads, self.levels, self.points))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        for lin in (self.value_proj, self.out):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, pos, ref, shapes):
        """x [B, T, D] tokens of the levels (coarsest first); pos [T, D]; ref [T, 2]
        normalized token centres; shapes [(H_l, W_l)]."""
        batch_size, num_t, dim = x.shape
        H, L, K = self.heads, self.levels, self.points
        q = x + pos
        off = self.offsets(q).float().view(batch_size, num_t, H, L, K, 2)
        w = torch.softmax(self.weights(q).float().view(batch_size, num_t, H, L * K), dim=-1)
        w = w.permute(0, 2, 1, 3).reshape(batch_size * H, num_t, L * K)
        v = self.value_proj(x).view(batch_size, num_t, H, self.c_h)
        values, locs, start = [], [], 0
        for l, (h_l, w_l) in enumerate(shapes):
            n_l = h_l * w_l
            values.append(v[:, start:start + n_l].permute(0, 2, 3, 1).reshape(batch_size * H, self.c_h, h_l, w_l))
            start += n_l
            cell = off.new_tensor([1.0 / max(w_l - 1, 1), 1.0 / max(h_l - 1, 1)])
            g = ref[None, :, None, None, :].float() + off[:, :, :, l] * cell          # [B, T, H, K, 2]
            locs.append(g.permute(0, 2, 1, 3, 4).reshape(batch_size * H, num_t, K, 2))
        a = ms_deform_sample(values, locs, w).view(batch_size, H, num_t, self.c_h)
        a = a.permute(0, 2, 1, 3).reshape(batch_size, num_t, dim)
        x = self.norms[0](x + self.drop(self.out(a)))
        return self.norms[1](x + self.drop(self.ffn(x)))


class DeformableEncoder(nn.Module):
    """Option C: a Deformable-DETR encoder over the coarsest `levels` neck maps, between the
    neck and the head. Its output is added to the neck maps through a zero-initialized
    projection, so at init the head sees exactly the neck features."""

    def __init__(self, in_channels, dim, num_layers, levels, num_heads, num_points, ffn_dim, dropout):
        super().__init__()
        if dim % 4:
            raise ValueError("encoder_cfg.dim must be divisible by 4")
        self.levels, self.dim = int(levels), int(dim)
        self.in_proj = nn.ModuleList([nn.Conv2d(in_channels, dim, 1) for _ in range(self.levels)])
        self.level_embed = nn.Parameter(torch.zeros(self.levels, dim))
        self.layers = nn.ModuleList([DeformableEncoderLayer(dim, self.levels, num_heads, num_points,
                                                            ffn_dim, dropout) for _ in range(int(num_layers))])
        self.out_proj = nn.ModuleList([nn.Conv2d(dim, in_channels, 1) for _ in range(self.levels)])
        self.last_rel_delta = None

    def reset_parameters(self):
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        for conv in self.in_proj:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        for layer in self.layers:
            layer.reset_parameters()
        for conv in self.out_proj:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, feats):
        """feats: list coarsest first; the first `levels` maps are encoded -> new list."""
        enc = feats[:self.levels]
        batch_size, dev = enc[0].shape[0], enc[0].device
        toks, poss, refs, shapes = [], [], [], []
        for l, f in enumerate(enc):
            h_l, w_l = f.shape[-2:]
            toks.append(self.in_proj[l](f).flatten(2).transpose(1, 2).float())
            poss.append(sine_position_2d(h_l, w_l, self.dim, dev, torch.float32) + self.level_embed[l].float())
            ys, xs = torch.meshgrid(torch.linspace(0.0, 1.0, h_l, device=dev),
                                    torch.linspace(0.0, 1.0, w_l, device=dev), indexing="ij")
            refs.append(torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1))
            shapes.append((h_l, w_l))
        x, pos, ref = torch.cat(toks, dim=1), torch.cat(poss, dim=0), torch.cat(refs, dim=0)
        for layer in self.layers:
            x = layer(x, pos, ref, shapes)
        out, start, deltas = list(feats), 0, []
        for l, f in enumerate(enc):
            h_l, w_l = shapes[l]
            n_l = h_l * w_l
            d = self.out_proj[l](x[:, start:start + n_l].transpose(1, 2).reshape(batch_size, self.dim, h_l, w_l))
            start += n_l
            out[l] = f + d.to(f.dtype)
            deltas.append(d.detach().abs().mean() / f.detach().abs().mean().clamp_min(1e-6))
        self.last_rel_delta = torch.stack(deltas).mean()
        return out



@MODELS.register_module()
class BLRHead(_OfficialHead):
    """See the module docstring. Constructor arguments:

    num_points (72): output rows (and output curve samples); sample_points (36):
    feature samples per anchor; prior_feat_channels (64): neck channels;
    num_levels (3): neck levels used (the last ones, fine -> coarse).
    anchor_cfg : num_anchors (192), dim (256), num_blocks (2), num_heads (8),
                 num_neighbors (16), ffn_dim (1024), dropout (0.0), dist_scale (50 px),
                 point_dim (64, per-point width before the geometry vector)
    refine_cfg : num_iters (3), resample ("fine" | "none"), step_scale (1.0),
                 point_offsets (True: per-point lateral offsets of the output curve),
                 max_point_offset_px (40, tanh bound of those offsets),
                 proposal (False; True: image-conditioned initial curve, a tanh-bounded
                 update of the 8 anchor parameters predicted from the P5-conditioned query
                 before any sampling and supervised as step 0), proposal_max_px (64),
                 step_blocks (False; True: a decoder block with its own weights inside
                 every refinement step: relation self-attention over the current lanes,
                 curve attention = deformable attention over the anchor's own curve at the
                 current geometry, FFN), step_ffn_dim (1024), deform_heads (4, must divide
                 the neck channels), deform_points (4 offsets per curve point, head and
                 level), deform_levels (3: the finest levels it samples),
                 deform_curve_points (= sample_points: reference points on the curve)
    encoder_cfg: None (off) or dict: a Deformable-DETR encoder between the neck and the
                 head over the coarsest `levels` maps, added to them through a zero-
                 initialized projection: dim (128), num_layers (2), levels (= num_levels),
                 num_heads (4), num_points (4), ffn_dim (512), dropout (0.0)
    align_cfg  : None (off) or dict: offsets (True: learned per-point sampling
                 offsets), max_offset_px (24), topk (3), max_px (50), min_cover
                 (0.25), target ("offset" | "feature" | "both"), feature_loss
                 ("cos" | "l2" | "infonce"), tau (0.1), offset_weight (0.05, px
                 smooth-L1), feature_weight (1.0)
    assign_cfg : o2o (Hungarian cls 0.15 linear, iou 3), o2m (SimOTA topk 4,
                 cls 0.15 linear, iou 3) or None to drop the o2m head
    loss_cfg   : cls_loss_weight (2.0), focal_alpha (0.25), focal_gamma (2.0),
                 iou_loss_weight (4.0), point_loss_weight (0.05, px), cp_loss_weight
                 (0.05, px), point_target ("lateral" | "same_t"), extent_loss_weight (0.2,
                 rows: the curve's bottom and top against the GT's first and last
                 visible row), smooth_l1_beta (1.0), o2m_cls_weight (1.0),
                 o2m_reg_weight (0.5), lane_width, lane_width_cost, iou_eval_shape,
                 seg_loss_weight (1.0), seg_bg_weight (0.4), cp_fit_ridge (1e-2)
    test_score : None (o2o without NMS, o2m with NMS) | "o2o" | "o2m"
    """

    def __init__(self, num_points=72, sample_points=36, prior_feat_channels=64, num_levels=3,
                 img_w=800, img_h=320, seg_num_classes=5, anchor_cfg=None, refine_cfg=None,
                 align_cfg=None, assign_cfg=None, loss_cfg=None, target_adapter=None,
                 test_score=None, train_cfg=None, test_cfg=None, init_cfg=None, encoder_cfg=None,
                 **unused):
        BaseModule.__init__(self, init_cfg=init_cfg)
        if unused:
            print(f"[BLRHead] ignoring unused head arguments: {sorted(unused)}")
        a = dict(anchor_cfg or {})
        r = dict(refine_cfg or {})
        lc = dict(loss_cfg or {})
        asg = dict(assign_cfg or {})
        for name, cfg, known in (
                ("anchor_cfg", a, {"num_anchors", "dim", "num_blocks", "num_heads", "num_neighbors",
                                   "ffn_dim", "dropout", "dist_scale", "max_dist", "point_dim"}),
                ("refine_cfg", r, {"num_iters", "resample", "step_scale", "point_offsets",
                                   "max_point_offset_px", "proposal", "proposal_max_px", "step_blocks",
                                   "step_ffn_dim", "deform_heads", "deform_points", "deform_levels",
                                   "deform_curve_points"}),
                ("assign_cfg", asg, {"o2o", "o2m"}),
                ("loss_cfg", lc, {"cls_loss_weight", "focal_alpha", "focal_gamma", "iou_loss_weight",
                                  "point_loss_weight", "cp_loss_weight", "smooth_l1_beta",
                                  "o2m_cls_weight", "o2m_reg_weight", "lane_width", "lane_width_cost",
                                  "iou_eval_shape", "seg_loss_weight", "seg_bg_weight",
                                  "seg_ignore_label", "cp_fit_ridge", "point_target",
                                  "extent_loss_weight"})):
            unknown = set(cfg) - known
            if unknown:
                raise ValueError(f"BLRHead {name}: unknown keys {sorted(unknown)}")

        # ---- attributes the official decoding path reads
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.img_w, self.img_h = int(img_w), int(img_h)
        self.num_points = self.n_offsets = int(num_points)
        self.n_strips = self.n_offsets - 1
        self.sample_points = int(sample_points)
        self.num_levels = int(num_levels)
        self.prior_feat_channels = int(prior_feat_channels)
        self.register_buffer("prior_ys", torch.linspace(1, 0, steps=self.n_offsets), persistent=False)
        self.register_buffer("t_sample", torch.linspace(0, 1, steps=self.sample_points), persistent=False)
        self.register_buffer("t_out", torch.linspace(0, 1, steps=self.n_offsets), persistent=False)
        if test_score not in (None, "o2o", "o2m"):
            raise ValueError("BLRHead test_score must be None, 'o2o' or 'o2m'")
        self.test_score = test_score

        # ---- anchors
        self.num_anchors = int(a.get("num_anchors", 192))
        dim = self.dim = int(a.get("dim", 256))
        if dim % 4 or dim % int(a.get("num_heads", 8)):
            raise ValueError("anchor_cfg.dim must be divisible by 4 and by num_heads")
        heads = int(a.get("num_heads", 8))
        cps = clrernet_style_bezier_priors(self.num_anchors, self.img_w, self.img_h).flip(1)
        self.anchor_params = nn.Parameter(params_from_control_points(cps))     # [N, 8]
        self.anchor_embed = nn.Parameter(torch.zeros(self.num_anchors, dim))
        self.geo_enc = _MLP(10, dim)                      # control points + start, length
        self.img_proj = nn.Linear(self.prior_feat_channels, dim)
        self.sem_pre = nn.MultiheadAttention(dim, heads, dropout=float(a.get("dropout", 0.0)),
                                             batch_first=True)
        self.sem_pre_norm = nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])
        feat_in = self.prior_feat_channels * self.num_levels
        point_dim = int(a.get("point_dim", 64))
        self.point_proj = nn.Linear(feat_in, dim)
        self.point_pos = nn.Parameter(torch.zeros(self.sample_points, dim))
        self.point_norm = nn.LayerNorm(dim)
        self.geo_point = nn.Sequential(nn.Linear(feat_in, point_dim), nn.ReLU(inplace=True))
        self.geo_vec = nn.Sequential(nn.Linear(point_dim * self.sample_points, dim),
                                     nn.ReLU(inplace=True), nn.Linear(dim, dim), nn.LayerNorm(dim))
        rel = dict(num_heads=heads, num_neighbors=int(a.get("num_neighbors", 16)),
                   n_strips=self.n_strips, x_scale=float(self.img_w - 1),
                   ffn_dim=int(a.get("ffn_dim", 1024)), dropout=float(a.get("dropout", 0.0)),
                   dist_scale=float(a.get("dist_scale", 50.0)), max_dist=float(a.get("max_dist", 10.0)))
        self.blocks = nn.ModuleList([BLRBlock(dim, **rel) for _ in range(int(a.get("num_blocks", 2)))])
        self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))

        # ---- feature alignment / learned sampling offsets
        self.align = None
        if align_cfg is not None:
            al = dict(align_cfg)
            known = {"offsets", "max_offset_px", "topk", "max_px", "min_cover", "target",
                     "feature_loss", "tau", "offset_weight", "feature_weight"}
            if set(al) - known:
                raise ValueError(f"BLRHead align_cfg: unknown keys {sorted(set(al) - known)}")
            self.align = dict(offsets=bool(al.get("offsets", True)),
                              max_offset_px=float(al.get("max_offset_px", 24.0)),
                              topk=int(al.get("topk", 3)), max_px=float(al.get("max_px", 50.0)),
                              min_cover=float(al.get("min_cover", 0.25)),
                              target=str(al.get("target", "both")),
                              feature_loss=str(al.get("feature_loss", "cos")),
                              tau=float(al.get("tau", 0.1)),
                              offset_weight=float(al.get("offset_weight", 0.05)),
                              feature_weight=float(al.get("feature_weight", 1.0)))
            if self.align["target"] not in ("offset", "feature", "both"):
                raise ValueError("align_cfg.target must be 'offset', 'feature' or 'both'")
            if self.align["feature_loss"] not in ("cos", "l2", "infonce"):
                raise ValueError("align_cfg.feature_loss must be 'cos', 'l2' or 'infonce'")
            if self.align["target"] == "offset" and not self.align["offsets"]:
                raise ValueError("align_cfg.target='offset' needs align_cfg.offsets=True")
            self.offset_head = (nn.Linear(dim, 2 * self.sample_points) if self.align["offsets"]
                                else None)
        else:
            self.offset_head = None

        # ---- refinement loop
        self.num_iters = int(r.get("num_iters", 3))
        if self.num_iters < 1:
            raise ValueError("refine_cfg.num_iters must be >= 1")
        self.resample = str(r.get("resample", "fine"))
        if self.resample not in ("fine", "none"):
            raise ValueError("refine_cfg.resample must be 'fine' or 'none'")
        self.step_scale = float(r.get("step_scale", 1.0))
        self.point_offsets = bool(r.get("point_offsets", True))
        self.max_point_offset_px = float(r.get("max_point_offset_px", 40.0))
        self.max_point_offset = self.max_point_offset_px / float(self.img_w - 1)
        self.proposal_max_px = float(r.get("proposal_max_px", 64.0))
        self.proposal = nn.Linear(dim, 8) if bool(r.get("proposal", False)) else None
        self.delta_enc = _MLP(8, dim)
        self.resample_enc = (nn.Sequential(nn.Linear(self.prior_feat_channels * self.sample_points, dim),
                                           nn.ReLU(inplace=True), nn.Linear(dim, dim))
                             if self.resample == "fine" else None)
        gru_in = 2 * dim + (dim if self.resample_enc is not None else 0)
        self.gru = nn.GRUCell(gru_in, dim)
        n_out = 8 + (self.n_offsets if self.point_offsets else 0)
        self.update = nn.Linear(dim, n_out)        # in px (x, offsets) and rows-scale (y, increments)
        self.register_buffer("param_scale", torch.tensor([1.0 / (self.img_w - 1)] * 4
                                                         + [1.0 / (self.img_h - 1)] * 4), persistent=False)
        self.step_blocks = None
        if bool(r.get("step_blocks", False)):
            self.deform_levels = int(r.get("deform_levels", self.num_levels))
            if not 1 <= self.deform_levels <= self.num_levels:
                raise ValueError("refine_cfg.deform_levels must be between 1 and num_levels")
            deform = dict(value_channels=self.prior_feat_channels, num_levels=self.deform_levels,
                          num_heads=int(r.get("deform_heads", 4)), num_points=int(r.get("deform_points", 4)),
                          x_scale=float(self.img_w - 1))
            rel_sb = {k: v for k, v in rel.items() if k != "ffn_dim"}
            self.step_blocks = nn.ModuleList([StepBlock(dim, rel_sb, deform, int(r.get("step_ffn_dim", 1024)),
                                                        rel["dropout"]) for _ in range(self.num_iters)])
            self.deform_curve_points = int(r.get("deform_curve_points", self.sample_points))
            self.step_pos = nn.Parameter(torch.zeros(self.deform_curve_points, dim))
            self.register_buffer("t_deform", torch.linspace(0, 1, steps=self.deform_curve_points),
                                 persistent=False)

        # ---- heads (the o2m head only with an o2m assigner)
        self.o2m_assigner = None
        o2m = asg.get("o2m", dict(type="SimOTALaneAssigner", candidate_topk=4, min_dynamic_k=1,
                                  cls_weight=0.15, cls_cost_mode="linear", point_weight=0.0,
                                  iou_weight=3.0))
        self.o2m_assigner = build_lane_assigner(o2m) if o2m else None
        self.cls_o2m = nn.Linear(dim, 2) if self.o2m_assigner is not None else None
        self.final_sa = LaneRelationAttention(dim, heads, rel["num_neighbors"], self.n_strips,
                                              float(self.img_w - 1), rel["dist_scale"],
                                              rel["max_dist"], rel["dropout"])
        self.final_norm = nn.LayerNorm(dim)
        self.cls_o2o = nn.Linear(dim, 2)
        self.seg_decoder = SegDecoder(self.img_h, self.img_w,
                                      self.prior_feat_channels * self.num_levels, seg_num_classes)
        self.encoder = None
        if encoder_cfg is not None:
            ec = dict(encoder_cfg)
            known = {"dim", "num_layers", "levels", "num_heads", "num_points", "ffn_dim", "dropout"}
            if set(ec) - known:
                raise ValueError(f"BLRHead encoder_cfg: unknown keys {sorted(set(ec) - known)}")
            levels = int(ec.get("levels", self.num_levels))
            if not 1 <= levels <= self.num_levels:
                raise ValueError("encoder_cfg.levels must be between 1 and num_levels")
            self.encoder = DeformableEncoder(self.prior_feat_channels, int(ec.get("dim", 128)),
                                             int(ec.get("num_layers", 2)), levels, int(ec.get("num_heads", 4)),
                                             int(ec.get("num_points", 4)), int(ec.get("ffn_dim", 512)),
                                             float(ec.get("dropout", 0.0)))

        # ---- assignment and losses
        self.o2o_assigner = build_lane_assigner(asg.get("o2o") or dict(
            type="HungarianLaneAssigner", cls_weight=0.15, cls_cost_mode="linear",
            point_weight=0.0, iou_weight=3.0))
        self.cls_loss_weight = float(lc.get("cls_loss_weight", 2.0))
        self.focal_alpha = float(lc.get("focal_alpha", 0.25))
        self.focal_gamma = float(lc.get("focal_gamma", 2.0))
        self.point_loss_weight = float(lc.get("point_loss_weight", 0.05))
        self.point_target = str(lc.get("point_target", "lateral"))
        if self.point_target not in ("lateral", "same_t"):
            raise ValueError("loss_cfg.point_target must be 'lateral' or 'same_t'")
        self.cp_loss_weight = float(lc.get("cp_loss_weight", 0.05))
        self.extent_loss_weight = float(lc.get("extent_loss_weight", 0.2))
        self.smooth_l1_beta = float(lc.get("smooth_l1_beta", 1.0))
        self.o2m_cls_weight = float(lc.get("o2m_cls_weight", 1.0))
        self.o2m_reg_weight = float(lc.get("o2m_reg_weight", 0.5))
        self.lane_width = float(lc.get("lane_width", 7.5 / 800))
        self.lane_width_cost = float(lc.get("lane_width_cost", 30.0 / 800))
        shape = lc.get("iou_eval_shape", None)
        self.iou_h, self.iou_w = ((int(shape[0]), int(shape[1])) if shape else (self.img_h, self.img_w))
        self.iou_loss = LaneIoULoss(float(lc.get("iou_loss_weight", 4.0)), self.lane_width,
                                    self.iou_w, self.iou_h)
        self.cp_fit_ridge = float(lc.get("cp_fit_ridge", 1e-2))
        self.seg_loss_weight = float(lc.get("seg_loss_weight", 1.0))
        seg_weights = torch.ones(seg_num_classes)
        seg_weights[0] = float(lc.get("seg_bg_weight", 0.4))
        self.register_buffer("seg_class_weights", seg_weights, persistent=False)
        self.seg_ignore_label = int(lc.get("seg_ignore_label", 255))
        self.target_adapter = CLRTargetAdapter(n_offsets=self.n_offsets, img_w=self.img_w,
                                               img_h=self.img_h, **dict(target_adapter or {}))
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long))
        self.reset_parameters()

    # ------------------------------------------------------------------ init
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
        nn.init.normal_(self.anchor_embed, std=0.02)
        nn.init.trunc_normal_(self.point_pos, std=0.02)
        # the sampling offsets and the classifiers start small (the cascade's 1e-3
        # init); the geometry update (px) starts at a fraction of a pixel, so the
        # first prediction is the anchor itself
        for lin in [m for m in (self.cls_o2m, self.cls_o2o, self.offset_head, self.proposal)
                    if m is not None]:
            nn.init.normal_(lin.weight, std=1.0e-3)
            nn.init.zeros_(lin.bias)
        nn.init.normal_(self.update.weight, std=1.0e-2)
        nn.init.zeros_(self.update.bias)
        # deformable attention: offsets start on Deformable DETR's grid, weights uniform; the
        # encoder's output projection starts at zero (identity on the neck features)
        for m in self.modules():
            if isinstance(m, (CurveDeformAttention, DeformableEncoder)):
                m.reset_parameters()
        if self.step_blocks is not None:
            nn.init.trunc_normal_(self.step_pos, std=0.02)

    def init_weights(self):
        """The official head's init touches the anchor generator this head does
        not have; everything here is initialized in __init__."""
        return

    # ------------------------------------------------------------------ pieces
    def _select_features(self, x):
        if isinstance(x, dict):
            x = list(x.values())
        feats = list(x)[len(x) - self.num_levels:]
        feats.reverse()                                            # coarsest first
        areas = [f.shape[-2] * f.shape[-1] for f in feats]
        if any(p > q for p, q in zip(areas[:-1], areas[1:])):
            raise RuntimeError("BLRHead expects neck outputs ordered fine -> coarse")
        return feats

    def official_layout(self, logits, pts, end_slopes=None):
        """[B, N, 6+R] from the output points [B, N, P, 2] and logits [B, N, 2];
        end_slopes [B, N, 2] continue the lane beyond its ends (bezier_end_slopes)."""
        xs, start_y, length = curve_to_rows(pts, self.prior_ys, self.n_strips, end_slopes=end_slopes)
        row = ((1.0 - start_y) * self.n_strips).round().long().clamp(0, self.n_strips)
        start_x = xs.gather(-1, row[..., None]).squeeze(-1)
        with torch.no_grad():
            dx = (pts[..., -1, 0] - pts[..., 0, 0]) * float(self.img_w - 1)
            dy = (pts[..., 0, 1] - pts[..., -1, 1]).clamp_min(1e-6) * float(self.img_h - 1)
            theta = (torch.atan2(dy, dx) / math.pi).clamp(0.01, 0.99)
        return torch.cat([logits.to(pts.dtype), start_y[..., None], start_x[..., None],
                          theta[..., None], length[..., None], xs], dim=-1)

    def _geometry_code(self, cps, lanes):
        return torch.cat([cps.flatten(-2), lanes[..., 2:3], lanes[..., 5:6]], dim=-1).detach()

    def _forward(self, feats):
        batch_size = feats[0].shape[0]
        dev, dt = feats[0].device, torch.float32
        num_q, dim = self.num_anchors, self.dim
        zeros2 = feats[0].new_zeros((batch_size, num_q, 2), dtype=dt)

        # anchors: geometry, rows, semantic query
        params0 = self.anchor_params[None].expand(batch_size, -1, -1).float()
        cps0 = anchor_control_points(params0)                                   # [B, N, 4, 2]
        pts_s = curve_points(cps0, self.t_sample)                               # [B, N, S, 2]
        lanes0 = self.official_layout(zeros2, curve_points(cps0, self.t_out), bezier_end_slopes(cps0))
        sem = self.anchor_embed[None].expand(batch_size, -1, -1) + self.geo_enc(
            self._geometry_code(cps0, lanes0))
        top = feats[0]
        img_tok = self.img_proj(top.flatten(2).transpose(1, 2).float())
        img_tok = img_tok + sine_position_2d(top.shape[-2], top.shape[-1], dim, dev, img_tok.dtype)[None]
        sem = self.sem_pre_norm[0](sem)
        sem = self.sem_pre_norm[1](sem + self.sem_pre(sem, img_tok, img_tok, need_weights=False)[0])

        # proposal (optional): an image-conditioned initial curve before any sampling; the
        # update is bounded (px) and supervised as step 0. The sampling grid, the relation
        # neighbours and the alignment assignment then use the proposed curve.
        params_init, prop_px = params0, None
        if self.proposal is not None:
            prop_px = torch.tanh(self.proposal(sem).float() / self.proposal_max_px) * self.proposal_max_px
            params_init = params0 + prop_px * self.param_scale
            cps_init = anchor_control_points(params_init)
            pts_s = curve_points(cps_init, self.t_sample)
            lanes0 = self.official_layout(zeros2, curve_points(cps_init, self.t_out), bezier_end_slopes(cps_init))

        # sampling, once, every level (with learned per-point offsets if configured)
        offsets = None
        grid = pts_s.detach()
        if self.offset_head is not None:
            off = torch.tanh(self.offset_head(sem).view(batch_size, num_q, self.sample_points, 2))
            scale = off.new_tensor([self.align["max_offset_px"] / float(self.img_w - 1),
                                    self.align["max_offset_px"] / float(self.img_h - 1)])
            offsets = off * scale
            grid = grid + offsets
        raw = torch.cat([sample_points(f, grid).float() for f in feats], dim=-1)   # [B, N, S, L*C]
        tok = self.point_norm(self.point_proj(raw) + self.point_pos)
        geo = self.geo_vec(self.geo_point(raw).flatten(2))
        for block in self.blocks:
            sem, geo = block(sem, geo, tok, img_tok, lanes0)
        h = self.fuse(torch.cat([sem, geo], dim=-1))

        # iterative refinement (self-feedback)
        params = params_init
        delta_raw = None
        preds, cps_list, pts_list = [], [], []
        if self.proposal is not None:                           # step 0 = the proposal
            preds.append(lanes0)
            cps_list.append(cps_init)
            pts_list.append(curve_points(cps_init, self.t_out))
        fine = feats[-1]
        hf = h.reshape(batch_size * num_q, dim)
        cur_pts = pts_s.detach()
        cps_cur = cps_init if self.proposal is not None else cps0
        deform_feats = feats[len(feats) - self.deform_levels:] if self.step_blocks is not None else None
        step_off, step_w = [], []
        for i in range(self.num_iters):
            if self.step_blocks is not None:                     # option B: a decoder block per step
                lanes_cur = (preds[-1] if preds else lanes0).detach()
                ref = (cur_pts if self.deform_curve_points == self.sample_points
                       else curve_points(cps_cur, self.t_deform).detach())
                hb = self.step_blocks[i](hf.view(batch_size, num_q, dim), lanes_cur, self.step_pos, ref,
                                         deform_feats)
                hf = hb.reshape(batch_size * num_q, dim)
                step_off.append(self.step_blocks[i].curve.last_offset_px)
                step_w.append(self.step_blocks[i].sa.last_self_weight)
            inp = [hf, self.delta_enc((params - params0).detach().float()).reshape(-1, dim)]
            if self.resample_enc is not None:
                fine_feat = sample_points(fine, cur_pts).float().flatten(2)          # [B, N, S*C]
                inp.append(self.resample_enc(fine_feat).reshape(-1, dim))
            hf = self.gru(torch.cat(inp, dim=-1), hf)
            upd = self.update(hf).view(batch_size, num_q, -1).float() * self.step_scale   # px / rows
            params = params + upd[..., :8] * self.param_scale
            cps = anchor_control_points(params)
            cps_cur = cps
            pts = curve_points(cps, self.t_out)
            if self.point_offsets:
                d = upd[..., 8:]                                   # [B, N, P] lateral, px
                delta_raw = d if delta_raw is None else delta_raw + d
                delta_pts = torch.tanh(delta_raw / self.max_point_offset_px) * self.max_point_offset
                pts = torch.cat([pts[..., :1] + delta_pts[..., None], pts[..., 1:]], dim=-1)
            preds.append(self.official_layout(zeros2, pts, bezier_end_slopes(cps)))
            cps_list.append(cps)
            pts_list.append(pts)
            cur_pts = curve_points(cps, self.t_sample).detach()
        h = hf.view(batch_size, num_q, dim)

        # heads: o2m per anchor; o2o after the converged lanes have interacted
        cls_o2m, s_o2m = None, None
        if self.cls_o2m is not None:
            cls_o2m = self.cls_o2m(h).float()
            with torch.no_grad():
                s_o2m = torch.softmax(cls_o2m, dim=-1)[..., 1]
        h_fin = self.final_norm(h + self.final_sa(h, preds[-1], s_o2m))
        cls_o2o = self.cls_o2o(h_fin).float()
        return dict(preds=preds, cps=cps_list, pts=pts_list, cls_o2o=cls_o2o, cls_o2m=cls_o2m,
                    anchor_lanes=lanes0.detach(), anchor_pts=pts_s.detach(), raw_feats=raw,
                    offsets=offsets, feats=feats, h=h, prop_px=prop_px,
                    step_off_px=(torch.stack(step_off).mean() if step_off else None),
                    step_self_w=(torch.stack([w for w in step_w if w is not None]).mean()
                                 if step_w and all(w is not None for w in step_w) else None),
                    enc_rel_delta=(self.encoder.last_rel_delta if self.encoder is not None else None),
                    self_w=[b.sem_sa.last_self_weight for b in self.blocks]
                    + [self.final_sa.last_self_weight])

    # ------------------------------------------------------------------ forward
    def _encode(self, feats):
        return self.encoder(feats) if self.encoder is not None else feats

    def forward_train(self, feats):
        feats = self._encode(feats)
        out = self._forward(feats)
        size = feats[-1].shape[-2:]
        seg_in = torch.cat([F.interpolate(f, size=size, mode="bilinear", align_corners=False)
                            for f in feats], dim=1)
        out["seg"] = self.seg_decoder(seg_in)
        return out

    def forward_test(self, feats):
        feats = self._encode(feats)
        out = self._forward(feats)
        cfg = self.test_cfg or {}
        use_nms = bool(cfg.get("use_nms", True))
        source = self.test_score or ("o2m" if use_nms else "o2o")
        z = out["cls_o2o"] if source == "o2o" or out["cls_o2m"] is None else out["cls_o2m"]
        if not use_nms:
            topk = cfg.get("nms_topk")
            if topk is not None and 0 < int(topk) < z.shape[1]:
                fg = z[..., 1] - z[..., 0]
                keep = torch.zeros_like(fg, dtype=torch.bool)
                keep.scatter_(1, fg.topk(int(topk), dim=1).indices, True)
                z = torch.stack([z[..., 0], torch.where(keep, z[..., 1], z[..., 1] - 1e4)], dim=-1)
        preds = list(out["preds"])
        last = preds[-1].clone()
        last[..., :2] = z.to(last.dtype)
        preds[-1] = last
        return StagePredictions([self.to_official_pred_dict(p) for p in preds])

    @staticmethod
    def to_official_pred_dict(pred):
        return {"cls_logits": pred[..., :2], "anchor_params": pred[..., 2:5],
                "lengths": pred[..., 5:6], "xs": pred[..., 6:]}

    def forward(self, x, *args, **kwargs):
        feats = self._select_features(x)
        if self.training:
            return self.forward_train(feats)
        return self.forward_test(feats)

    def loss(self, x, batch_data_samples, *args, **kwargs):
        if self.training:
            self._train_step += 1
        feats = self._select_features(x)
        outs = self.forward_train(feats)
        device = feats[0].device
        lanes = self.target_adapter.extract_lanes(batch_data_samples, device)
        seg_gt = None
        if self.seg_loss_weight > 0 and "seg" in outs:
            seg_gt = self.target_adapter.extract_seg(batch_data_samples, device, outs["seg"].shape[-2:])
        return self.loss_by_outputs(outs, lanes, seg_gt)

    # ------------------------------------------------------------------ losses
    def _assign(self, assigner, pred, gt):
        cache = build_cost_cache(pred, gt, self.img_w, self.img_h, self.lane_width,
                                 self.lane_width_cost, set(assigner.required_cache_keys),
                                 iou_shape=(self.iou_w, self.iou_h))
        return assigner.assign(cache)

    def _geometry_terms(self, outs, b, rows, cols, gt, gt_cps, gt_ok, gt_pts):
        """Mean over the matched pairs of one image and over the iterations of the
        LaneIoU, point and control-point terms (unweighted point / cp, in px).

        point_target "lateral": each output point against the GT x at the point's
        own y (points outside the GT's visible span get none), the same quantity
        LaneIoU measures; "same_t": against the GT curve sampled at the same t
        (BezierLaneNet style), which also fixes the parameterization and fights
        the row losses while the extents differ."""
        scale = float(self.img_w - 1) / float(self.img_w)
        px = gt.new_tensor([float(self.img_w - 1), float(self.img_h - 1)])
        tgt_rows = gt[cols, 6:] / float(self.img_w)
        gt_valid = (gt[cols, 6:] >= 0.0) & (gt[cols, 6:] < float(self.img_w))
        first = torch.argmax(gt_valid.int(), dim=1)
        last = self.n_strips - torch.argmax(gt_valid.flip(1).int(), dim=1)
        y_ext = torch.stack([self.prior_ys[first], self.prior_ys[last]], dim=-1)   # [M, 2]
        iou = pt = cp = ext = gt.new_zeros(())
        for pred, cps, pts in zip(outs["preds"], outs["cps"], outs["pts"]):
            iou = iou + self.iou_loss(pred[b, rows, 6:] * scale, tgt_rows).mean()
            ends = torch.stack([cps[b, rows, 0, 1], cps[b, rows, 3, 1]], dim=-1)
            ext = ext + _smooth_l1((ends - y_ext) * float(self.n_strips), 0.0, self.smooth_l1_beta).mean()
            if self.point_target == "lateral":
                p_m = pts[b, rows]
                x_gt, on = gt_x_at_y(gt[cols], p_m[..., 1].detach(), self.img_w, self.n_strips)
                err = _smooth_l1((p_m[..., 0] - x_gt) * px[0], 0.0, self.smooth_l1_beta)
                pt = pt + (err * on).sum() / on.sum().clamp_min(1.0)
            else:
                pt = pt + _smooth_l1((pts[b, rows] - gt_pts[cols]) * px, 0.0, self.smooth_l1_beta).mean()
            cp_lane = _smooth_l1((cps[b, rows] - gt_cps[cols]) * px, 0.0,
                                 self.smooth_l1_beta).mean(dim=(-1, -2))
            ok = gt_ok[cols].float()
            cp = cp + (cp_lane * ok).sum() / ok.sum().clamp_min(1.0)
        n = float(len(outs["preds"]))
        return iou / n, pt / n, cp / n, ext / n

    def _align_terms(self, outs, b, gt):
        """Feature-alignment terms of one image (design item 3): anchors assigned to
        GT before any prediction (k nearest in mean |dx| over the GT rows the anchor
        covers), their sampling offsets against the lateral distance to the GT, and
        their sampled features against the features sampled on the GT (no grad)."""
        al = self.align
        lanes = outs["anchor_lanes"][b]                              # [N, 6+R]
        xs_a = lanes[:, 6:] * float(self.img_w - 1)
        gt_px = gt[:, 6:]
        valid_g = (gt_px >= 0.0) & (gt_px < float(self.img_w))     # [G, R]
        start, end = lane_rows(lanes, self.n_strips)
        r = torch.arange(self.n_offsets, device=gt.device).view(1, 1, -1)
        cover = (r >= start[:, None, None]) & (r <= end[:, None, None]) & valid_g[None]   # [N, G, R]
        cnt = cover.sum(-1).float()
        dist = ((xs_a[:, None, :] - gt_px[None]).abs() * cover).sum(-1) / cnt.clamp_min(1.0)
        ok = (cnt >= al["min_cover"] * valid_g.sum(-1).float()[None].clamp_min(1.0)) & (dist <= al["max_px"])
        dist = torch.where(ok, dist, torch.full_like(dist, float("inf")))
        k = min(al["topk"], dist.shape[0])
        d_k, n_k = dist.topk(k, dim=0, largest=False)                # [k, G]
        pair = torch.isfinite(d_k)
        if not bool(pair.any()):
            return None
        n_idx = n_k[pair]
        g_idx = torch.arange(gt.shape[0], device=gt.device)[None].expand(k, -1)[pair]
        y_q = outs["anchor_pts"][b, n_idx, :, 1]                      # [M, S]
        x_gt, on_gt = gt_x_at_y(gt[g_idx], y_q, self.img_w, self.n_strips)
        if not bool(on_gt.any()):
            return None
        terms = {}
        x_a = outs["anchor_pts"][b, n_idx, :, 0]
        if al["target"] in ("offset", "both") and outs["offsets"] is not None:
            off_x = outs["offsets"][b, n_idx, :, 0] * float(self.img_w - 1)
            tgt = ((x_gt - x_a) * float(self.img_w - 1)).detach()
            err = _smooth_l1(off_x, tgt, self.smooth_l1_beta)
            terms["offset"] = (err * on_gt).sum() / on_gt.sum().clamp_min(1.0)
        if al["target"] in ("feature", "both"):
            with torch.no_grad():
                xy = torch.stack([x_gt, y_q], dim=-1)[None]          # [1, M, S, 2]
                f_gt = torch.cat([sample_points(f[b:b + 1], xy).float() for f in outs["feats"]], dim=-1)[0]
            f_a = outs["raw_feats"][b, n_idx]                        # [M, S, LC]
            if al["feature_loss"] == "l2":
                err = ((f_a - f_gt) ** 2).mean(-1)
            elif al["feature_loss"] == "cos":
                err = 1.0 - F.cosine_similarity(f_a, f_gt, dim=-1, eps=1e-6)
            else:
                qn = F.normalize(f_a[on_gt], dim=-1)                 # [V, LC]
                kn = F.normalize(f_gt[on_gt], dim=-1)
                logits = qn @ kn.t() / al["tau"]
                err = F.cross_entropy(logits, torch.arange(qn.shape[0], device=gt.device),
                                      reduction="none")
                terms["feature"] = err.mean()
                return terms
            terms["feature"] = (err * on_gt).sum() / on_gt.sum().clamp_min(1.0)
        return terms

    def loss_by_outputs(self, outs, lanes, seg_gt=None):
        valid_targets = [t[t[:, 1] == 1] for t in lanes]
        batch_size, num_q = outs["cls_o2o"].shape[:2]
        dev = outs["cls_o2o"].device
        zero = outs["cls_o2o"].new_zeros(())
        tgt_o2o = torch.zeros((batch_size, num_q), dtype=torch.long, device=dev)
        tgt_o2m = torch.zeros_like(tgt_o2o)
        gt_counts = torch.ones(batch_size, device=dev)
        iou_sum = pt_sum = cp_sum = ext_sum = zero
        n_img = 0
        align_sum = {"offset": zero, "feature": zero}
        align_n = 0
        num_pos_o2o = num_pos_o2m = 0
        final = outs["preds"][-1]
        pred_o2o = torch.cat([outs["cls_o2o"].detach(), final[..., 2:].detach()], dim=-1)
        pred_o2m = (torch.cat([outs["cls_o2m"].detach(), final[..., 2:].detach()], dim=-1)
                    if self.o2m_assigner is not None else None)
        for b, gt in enumerate(valid_targets):
            if gt.shape[0] == 0:
                continue
            gt_counts[b] = float(gt.shape[0])
            gt_cps, gt_ok = fit_gt_bezier(gt, self.prior_ys, self.img_w, self.img_h, self.cp_fit_ridge)
            gt_pts = curve_points(gt_cps, self.t_out)
            rows, cols = self._assign(self.o2o_assigner, pred_o2o[b], gt)
            tgt_o2o[b, rows] = 1
            num_pos_o2o += int(rows.numel())
            n_img += 1
            if rows.numel():
                iou, pt, cp, ext = self._geometry_terms(outs, b, rows, cols, gt, gt_cps, gt_ok, gt_pts)
                iou_sum, pt_sum, cp_sum, ext_sum = iou_sum + iou, pt_sum + pt, cp_sum + cp, ext_sum + ext
            if self.o2m_assigner is not None:
                rows_m, cols_m = self._assign(self.o2m_assigner, pred_o2m[b], gt)
                tgt_o2m[b, rows_m] = 1
                num_pos_o2m += int(rows_m.numel())
                if rows_m.numel() and self.o2m_reg_weight > 0:
                    iou, pt, cp, ext = self._geometry_terms(outs, b, rows_m, cols_m, gt, gt_cps, gt_ok, gt_pts)
                    w = self.o2m_reg_weight
                    iou_sum, pt_sum = iou_sum + w * iou, pt_sum + w * pt
                    cp_sum, ext_sum = cp_sum + w * cp, ext_sum + w * ext
            if self.align is not None:
                terms = self._align_terms(outs, b, gt)
                if terms:
                    align_n += 1
                    for key, val in terms.items():
                        align_sum[key] = align_sum[key] + val
        norm = float(max(1, n_img))
        cls_o2o = (focal_2cls(outs["cls_o2o"], tgt_o2o, self.focal_alpha, self.focal_gamma).sum(-1)
                   / gt_counts).mean()
        losses = dict(loss_cls=cls_o2o * self.cls_loss_weight,
                      loss_iou=iou_sum / norm,
                      loss_point=pt_sum / norm * self.point_loss_weight,
                      loss_cp=cp_sum / norm * self.cp_loss_weight,
                      loss_extent=ext_sum / norm * self.extent_loss_weight)
        if self.o2m_assigner is not None:
            cls_o2m = (focal_2cls(outs["cls_o2m"], tgt_o2m, self.focal_alpha, self.focal_gamma).sum(-1)
                       / gt_counts).mean()
            losses["loss_cls_o2m"] = cls_o2m * self.cls_loss_weight * self.o2m_cls_weight
        if self.align is not None:
            an = float(max(1, align_n))
            if self.align["target"] in ("offset", "both") and outs["offsets"] is not None:
                losses["loss_align_offset"] = align_sum["offset"] / an * self.align["offset_weight"]
            if self.align["target"] in ("feature", "both"):
                losses["loss_align_feat"] = align_sum["feature"] / an * self.align["feature_weight"]
        if seg_gt is not None and "seg" in outs:
            losses["loss_seg"] = F.nll_loss(F.log_softmax(outs["seg"], dim=1), seg_gt,
                                            weight=self.seg_class_weights,
                                            ignore_index=self.seg_ignore_label) * self.seg_loss_weight
        # diagnostics (not summed by mmengine)
        with torch.no_grad():
            diag = dict(num_pos_o2o=torch.tensor(float(num_pos_o2o), device=dev),
                        num_pos_o2m=torch.tensor(float(num_pos_o2m), device=dev))
            for i, w in enumerate(outs["self_w"]):
                diag[f"self_w_{'final' if i == len(outs['self_w']) - 1 else i}"] = w
            if outs["offsets"] is not None:
                diag["offset_px"] = (outs["offsets"][..., 0].abs().mean() * float(self.img_w - 1))
            if outs.get("prop_px") is not None:
                diag["prop_px"] = outs["prop_px"][..., :4].abs().mean()
            for key in ("step_off_px", "step_self_w", "enc_rel_delta"):
                if outs.get(key) is not None:
                    diag[key] = outs[key].detach()
            pos = tgt_o2o.bool()
            if bool(pos.any()):
                diag["o2o_pos_prob"] = torch.softmax(outs["cls_o2o"], -1)[..., 1][pos].mean()
                diag["o2o_neg_prob"] = torch.softmax(outs["cls_o2o"], -1)[..., 1][~pos].mean()
            if self.point_offsets:
                last_pts = outs["pts"][-1]
                base = curve_points(outs["cps"][-1], self.t_out)
                diag["point_offset_px"] = ((last_pts - base)[..., 0].abs().mean() * float(self.img_w - 1))
        losses.update(diag)
        return losses
