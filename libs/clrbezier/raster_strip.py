"""Curve-aligned strips for render-and-compare refinement (BLRHead strip_cfg).

A strip belongs to one candidate lane: S rows along its curve (P_s, unit normal n_s) and U
lateral bins u_k = (-K..K) * bin_px, so cell (s, k) sits at P_s + u_k n_s (input pixels).
Image evidence is sampled into the strip once; lanes are rendered into it analytically as
one 1D Gaussian per row (never rasterized on an image grid and resampled, which loses the
sub-pixel position). A small CNN compares the two and predicts, per row, where the lane
is inside the strip (sub-bin soft-argmax), whether it is there at all, and a quality
vector for the classifier. Only the candidate's own raster changes between iterations.

Conventions: points in input pixels (x right, y down). Feature maps use the ResNet
cell-centre convention: cell i of a stride-s map is centred at input pixel s * i. (The
point sampler of the head uses (W-1)-normalized points with align_corners=True, which
reads the feature centred at x (W - s) / (W - 1), up to s - 1 px away.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def curve_normals(pts):
    """pts [..., S, 2] -> unit normals [..., S, 2] (the tangent rotated by +90 degrees)."""
    d = torch.cat([pts[..., 1:2, :] - pts[..., 0:1, :],
                   pts[..., 2:, :] - pts[..., :-2, :],
                   pts[..., -1:, :] - pts[..., -2:-1, :]], dim=-2)
    t = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.stack([-t[..., 1], t[..., 0]], dim=-1)


def strip_cells(pts, nrm, u_bins):
    """pts, nrm [..., S, 2]; u_bins [U] -> cell centres [..., S, U, 2]."""
    return pts[..., :, None, :] + u_bins[:, None] * nrm[..., :, None, :]


def sample_strip(feat, cells, img_w, img_h):
    """Bilinear read of feat [B, C, Hf, Wf] at cells [B, N, S, U, 2] (input px) with the
    cell-centre convention; the stride is img / map size. Returns [B, N, C, S, U]
    (zeros outside the map)."""
    b, n, s, u, _ = cells.shape
    hf, wf = feat.shape[-2:]
    fx = cells[..., 0] * (wf / float(img_w))
    fy = cells[..., 1] * (hf / float(img_h))
    grid = torch.stack([fx / max(wf - 1, 1) * 2 - 1, fy / max(hf - 1, 1) * 2 - 1], dim=-1)
    out = F.grid_sample(feat, grid.reshape(b, n * s, u, 2).to(feat.dtype), mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.reshape(b, -1, n, s, u).permute(0, 2, 1, 3, 4)


def render_rows(u_pos, u_bins, sigma):
    """A lane at lateral position u_pos [..., S] (px) as one Gaussian per row -> [..., S, U]."""
    return torch.exp(-0.5 * ((u_bins - u_pos[..., None]) / sigma) ** 2)


def normal_crossings(pts, nrm, poly, poly_valid, max_abs):
    """Where a polyline crosses the normal line of each strip row.

    pts, nrm [..., S, 2]; poly [..., G, 2] with vertex validity poly_valid [..., G] (a segment
    counts when both ends are valid). Returns the signed offset u [..., S] of the crossing
    nearest to the row centre and a mask [..., S] (a crossing exists and |u| <= max_abs)."""
    a = poly[..., :-1, :].unsqueeze(-3)                                  # [..., 1, G-1, 2]
    e = (poly[..., 1:, :] - poly[..., :-1, :]).unsqueeze(-3)
    seg_ok = (poly_valid[..., :-1] & poly_valid[..., 1:]).unsqueeze(-2)   # [..., 1, G-1]
    w = a - pts.unsqueeze(-2)                                            # [..., S, G-1, 2]
    n = nrm.unsqueeze(-2)
    den = n[..., 0] * e[..., 1] - n[..., 1] * e[..., 0]
    ok = (den.abs() > 1e-6) & seg_ok
    den = torch.where(ok, den, torch.ones_like(den))
    u = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / den
    v = (w[..., 0] * n[..., 1] - w[..., 1] * n[..., 0]) / den
    ok = ok & (v >= 0.0) & (v <= 1.0)
    u_abs = torch.where(ok, u.abs(), torch.full_like(u, float("inf")))
    best = u_abs.argmin(-1, keepdim=True)
    u_best = u.gather(-1, best).squeeze(-1)
    has = torch.isfinite(u_abs.gather(-1, best).squeeze(-1)) & (u_best.abs() <= max_abs)
    return torch.where(has, u_best, torch.zeros_like(u_best)), has


@torch.no_grad()
def nearest_lanes(pts, valid, num):
    """Indices [B, N, num] of the other candidates with the smallest mean distance between
    corresponding strip rows (rows valid in both)."""
    d = (pts[:, :, None] - pts[:, None]).norm(dim=-1)                   # [B, N, N, S]
    w = (valid[:, :, None] & valid[:, None]).to(d.dtype)
    d = (d * w).sum(-1) / w.sum(-1).clamp_min(1.0)
    d = d.masked_fill(w.sum(-1) == 0, float("inf"))
    eye = torch.eye(pts.shape[1], dtype=torch.bool, device=pts.device)
    return d.masked_fill(eye, float("inf")).topk(num, dim=-1, largest=False).indices


@torch.no_grad()
def render_neighbours(pts, nrm, valid, scores, nbr_idx, u_bins, sigma, poly=None, poly_valid=None):
    """Max over the neighbours j of score_j * (j's crossing of each row's normal, rendered as a
    Gaussian). pts, nrm [B, N, S, 2] the strip rows; valid [B, N, S]; scores [B, N]; nbr_idx
    [B, N, M]. poly / poly_valid [B, N, G(, 2)]: the candidates' current polylines when they
    differ from the strip rows (default: the rows themselves)."""
    if poly is None:
        poly, poly_valid = pts, valid
    reach = float(u_bins.abs().max()) + 3.0 * sigma
    bi = torch.arange(pts.shape[0], device=pts.device)[:, None]
    out = pts.new_zeros(pts.shape[:-1] + (u_bins.numel(),))
    for m in range(nbr_idx.shape[-1]):
        j = nbr_idx[..., m]
        u_j, has = normal_crossings(pts, nrm, poly[bi, j], poly_valid[bi, j], reach)
        r = render_rows(u_j, u_bins, sigma) * (scores[bi, j][..., None] * has)[..., None]
        out = torch.maximum(out, r)
    return out


class StripRefiner(nn.Module):
    """Render-and-compare CNN over [M, C, S, U] strips (M = B * N candidates)."""

    def __init__(self, in_ch, ctx_dim, dim=32, layers=3, iters=2, half_bins=8, bin_px=2.0,
                 sigma_bins=1.0):
        super().__init__()
        self.iters = int(iters)
        self.register_buffer("u_bins", torch.arange(-half_bins, half_bins + 1).float() * bin_px,
                             persistent=False)
        self.sigma = float(sigma_bins) * float(bin_px)
        groups = 4 if dim % 4 == 0 else 1
        self.inp = nn.Sequential(nn.Conv2d(in_ch + 1, dim, 1), nn.GroupNorm(groups, dim), nn.ReLU(inplace=True))
        self.ctx = nn.Linear(ctx_dim, dim)
        body = []
        for i in range(int(layers)):
            body += [nn.Conv2d(dim + (1 if i == 0 else 0), dim, 3, padding=1), nn.GroupNorm(groups, dim),
                     nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        self.lat = nn.Conv2d(dim, 1, 1)
        self.vis = nn.Linear(dim, 1)
        self.quality = nn.Linear(dim, dim)

    def embed(self, strip, nbr, ctx):
        """The strip input: evidence [M, C, S, U] and neighbour raster [M, S, U] projected, plus
        the candidate's context ctx [M, ctx_dim] -> [M, dim, S, U]."""
        return self.inp(torch.cat([strip, nbr[:, None].to(strip.dtype)], 1)) + self.ctx(ctx)[..., None, None]

    def step(self, x0, u_own, own_ok=None):
        """One pass: the own raster at u_own [M, S] (px; rows with own_ok False get none) ->
        dict(logits [M, S, U], u [M, S] px, vis [M, S]) and the features read at the
        predicted offset [M, dim, S]."""
        u_bins = self.u_bins.to(x0.dtype)
        own = render_rows(u_own.detach(), u_bins, self.sigma)
        if own_ok is not None:
            own = own * own_ok[..., None].to(own.dtype)
        h = self.body(torch.cat([x0, own[:, None]], 1))
        logits = self.lat(h)[:, 0]
        prob = logits.softmax(-1)
        u = (prob * u_bins).sum(-1)
        read = (h * prob[:, None]).sum(-1)                      # [M, dim, S]: features at the predicted offset
        return dict(logits=logits, u=u, vis=self.vis(read.transpose(1, 2))[..., 0]), read

    def pool(self, read, row_valid):
        """Quality vector [M, dim]: the read features averaged over the valid rows."""
        w = row_valid.to(read.dtype)[:, None]
        return self.quality((read * w).sum(-1) / w.sum(-1).clamp_min(1.0))

    def forward(self, strip, nbr, row_valid, ctx):
        """strip [M, C, S, U]; nbr [M, S, U]; row_valid [M, S]; ctx [M, ctx_dim].
        Returns one dict per iteration (logits [M, S, U], u [M, S] px, vis [M, S]) and the
        quality vector [M, dim] of the last iteration."""
        x0 = self.embed(strip, nbr, ctx)
        u_cur = strip.new_zeros(strip.shape[0], strip.shape[2])
        outs = []
        for _ in range(self.iters):
            out, read = self.step(x0, u_cur)
            u_cur = out["u"]
            outs.append(out)
        return outs, self.pool(read, row_valid)


class StripQuality(nn.Module):
    """Localization quality of a candidate from its strip (strip_cfg.quality): how well the
    candidate's own rendered position agrees with where the strip says the lane is.

    Per strip row: the top-k probabilities of the lateral distribution (how sharp the
    evidence is), the agreement sum_u p(u) g(u - pos) with the candidate's own raster g
    (a Gaussian of the strip's sigma at its output position pos; 1 when the distribution
    sits on it), and the visibility. Pooled over the candidate's rows (plain and
    visibility-weighted means, the share of usable rows) -> MLP -> one logit, trained
    towards the candidate's best LaneIoU with any GT lane (as GFLv2's distribution-guided
    quality is for boxes). The inputs are detached, so this loss trains only this MLP."""

    def __init__(self, topk=4, hidden=64):
        super().__init__()
        self.topk = int(topk)
        n_row = self.topk + 2
        self.mlp = nn.Sequential(nn.Linear(2 * n_row + 1, int(hidden)), nn.ReLU(inplace=True),
                                 nn.Linear(int(hidden), 1))

    def stats(self, logits, pos, vis, row_ok, u_bins, sigma):
        """logits [M, S, U]; pos [M, S] px (the candidate's output position in the strip);
        vis [M, S] logits; row_ok [M, S] bool -> [M, 2 * (topk + 2) + 1]."""
        p = logits.detach().float().softmax(-1)
        k = min(self.topk, p.shape[-1])
        top = p.topk(k, dim=-1).values
        if k < self.topk:
            top = F.pad(top, (0, self.topk - k))
        own = render_rows(pos.detach().float(), u_bins.float(), sigma)
        agree = (p * own).sum(-1, keepdim=True)
        v = torch.sigmoid(vis.detach().float())[..., None]
        f = torch.cat([top, agree, v], dim=-1)                                  # [M, S, topk + 2]
        w = row_ok.float()[..., None]
        mean = (f * w).sum(1) / w.sum(1).clamp_min(1.0)
        wv = w * v
        vmean = (f * wv).sum(1) / wv.sum(1).clamp_min(1e-6)
        return torch.cat([mean, vmean, w.mean(1)], dim=-1)

    def forward(self, logits, pos, vis, row_ok, u_bins, sigma):
        return self.mlp(self.stats(logits, pos, vis, row_ok, u_bins, sigma))[..., 0]
