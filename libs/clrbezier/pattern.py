"""Lanes as pose x pattern (clrbezierlane/pose_pattern_design.md; HBLRHead pattern_cfg).

A cubic Bezier lane (or a segment of one) is split into
- its pose: the two end points in px, bottom first, the top above the bottom (image y down);
- its pattern: the two inner control points in the curve's own chord frame, with the shift, tilt and length
  removed. The chord runs from (0, 0) (bottom) to (0, 1) (top); a point's first coordinate is along the
  normal (to the right of the chord, looking from bottom to top), the second along the chord. A pattern is
  4 numbers (a1, b1, a2, b2); a straight lane with uniform speed is (0, 1/3, 0, 2/3).
The same bend gives the same pattern wherever the lane is and however it is tilted. Patterns are predicted
as a softmax mixture of prototypes (k-means of training lanes, tools/clrbezier/build_pattern_codebook.py)
plus a residual.

Everything here is plain torch on px coordinates."""
import torch
import torch.nn.functional as F

from .geometry import bernstein_basis

CANON_MIN_LEN_PX = 1.0       # chord length floor of the chord frame (px)
STRAIGHT = (0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0)


def chord_frame(p0, p3, min_len=CANON_MIN_LEN_PX):
    """Chord frames of curves from p0 to p3 [..., 2] (px) -> (e, n, length): e the unit chord direction, n the
    unit normal (e turned by +90 degrees in image axes: right of an upward chord), length [..., 1] (at least
    min_len). A chord shorter than 1e-6 px points up."""
    v = p3 - p0
    norm = v.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, -1.0], device=v.device, dtype=v.dtype).expand_as(v)
    e = torch.where(norm > 1e-6, v / norm.clamp_min(1e-6), up)
    n = torch.stack([-e[..., 1], e[..., 0]], dim=-1)
    return e, n, norm.clamp_min(min_len)


def to_canon(c_px, min_len=CANON_MIN_LEN_PX):
    """Patterns of cubics c_px [..., 4, 2] (px) -> [..., 4] = (a1, b1, a2, b2)."""
    p0 = c_px[..., 0, :]
    e, n, length = chord_frame(p0, c_px[..., 3, :], min_len)
    rel = c_px[..., 1:3, :] - p0[..., None, :]                                  # [..., 2, 2]
    a = (rel * n[..., None, :]).sum(-1) / length
    b = (rel * e[..., None, :]).sum(-1) / length
    return torch.stack([a[..., 0], b[..., 0], a[..., 1], b[..., 1]], dim=-1)


def from_canon(p0, p3, pattern, min_len=CANON_MIN_LEN_PX):
    """Cubics [..., 4, 2] (px) from their poses p0, p3 [..., 2] and patterns [..., 4]. Inverse of to_canon."""
    e, n, length = chord_frame(p0, p3, min_len)
    ab = pattern.view(*pattern.shape[:-1], 2, 2)                                # (a, b) per inner point
    inner = p0[..., None, :] + length[..., None, :] * (ab[..., :1] * n[..., None, :] + ab[..., 1:] * e[..., None, :])
    return torch.cat([p0[..., None, :], inner, p3[..., None, :]], dim=-2)


def reverse_canon(pattern):
    """The pattern of the same cubic read from its other end."""
    a1, b1, a2, b2 = pattern.unbind(-1)
    return torch.stack([-a2, 1.0 - b2, -a1, 1.0 - b1], dim=-1)


def clamp_inner_monotone(c_px):
    """Cubics c_px [..., 4, 2] (px, bottom first, y0 >= y3) with their inner control points' y clamped so that
    y0 >= y1 >= y2 >= y3: monotone in y, end points kept."""
    y0, y1, y2, y3 = c_px[..., 1].unbind(-1)
    y1 = torch.minimum(torch.maximum(y1, y3), y0)
    y2 = torch.minimum(torch.maximum(y2, y3), y1)
    return torch.stack([c_px[..., 0], torch.stack([y0, y1, y2, y3], dim=-1)], dim=-1)


def _inv_softplus(x):
    return torch.where(x > 20.0, x, x + torch.log(-torch.expm1(-x)))


def pose_chain_step(j_px, d, eps):
    """y-monotone update of chains of joints j_px [..., S + 1, 2] (px, bottom first, y non-increasing) by
    d [..., S, 4], per piece s between joints s and s + 1: (x move of joint s, x move of joint s + 1, change of
    the softplus pre-activation of the y gap between them, y move of joint 0 (piece 0 only, the others unused)).
    Shared joints get the mean x move. Every gap stays positive (at least eps before the update)."""
    num_s = d.shape[-2]
    dx = j_px.new_zeros(j_px.shape[:-1])
    dx = dx + F.pad(d[..., 0], (0, 1)) + F.pad(d[..., 1], (1, 0))
    count = torch.full((num_s + 1,), 2.0, device=j_px.device, dtype=j_px.dtype)
    count[0] = count[-1] = 1.0
    y = j_px[..., 1]
    gap = (y[..., :-1] - y[..., 1:]).clamp_min(eps)
    gap = F.softplus(_inv_softplus(gap) + d[..., 2])
    y0 = y[..., :1] + d[..., 0, 3:4]
    return torch.stack([j_px[..., 0] + dx / count, torch.cat([y0, y0 - torch.cumsum(gap, dim=-1)], dim=-1)], dim=-1)


def assemble_pieces(pcs):
    """Pieces [..., S, 4, 2] whose ends meet -> piecewise control points [..., 3S + 1, 2]."""
    num_s = pcs.shape[-3]
    return torch.cat([pcs[..., 0, :, :]] + [pcs[..., s, 1:, :] for s in range(1, num_s)], dim=-2)


@torch.no_grad()
def fit_cubic_pinned(pts, ridge=1e-2):
    """Least-squares cubics through polylines pts [..., M, 2] sampled uniformly in arc length, with the end
    points pinned to the polyline's ends and a ridge toward the chord thirds -> [..., 4, 2]."""
    num_m = pts.shape[-2]
    t = torch.linspace(0.0, 1.0, num_m, device=pts.device, dtype=pts.dtype)
    basis = bernstein_basis(t)                                                    # [M, 4]
    p0, p3 = pts[..., 0, :], pts[..., -1, :]
    rhs = pts - basis[:, :1] * p0[..., None, :] - basis[:, 3:] * p3[..., None, :]   # [..., M, 2]
    a = basis[:, 1:3]                                                             # [M, 2]
    frac = torch.tensor([1.0 / 3.0, 2.0 / 3.0], device=pts.device, dtype=pts.dtype).view(2, 1)
    c_lin = p0[..., None, :] + (p3 - p0)[..., None, :] * frac                     # [..., 2, 2]
    lhs = a.T @ a + ridge * torch.eye(2, device=pts.device, dtype=pts.dtype)
    sol = torch.linalg.solve(lhs.expand(*pts.shape[:-2], 2, 2), a.T @ rhs + ridge * c_lin)
    return torch.cat([p0[..., None, :], sol, p3[..., None, :]], dim=-2)


def project_to_polyline(p, poly):
    """Nearest positions of points p [..., K, 2] on polylines poly [..., M, 2] (same leading dims), as a
    continuous vertex index in [0, M - 1] -> [..., K]."""
    s0, s1 = poly[..., :-1, :], poly[..., 1:, :]
    d = s1 - s0                                                                   # [..., M-1, 2]
    rel = p[..., :, None, :] - s0[..., None, :, :]                                # [..., K, M-1, 2]
    t = ((rel * d[..., None, :, :]).sum(-1) / (d * d).sum(-1).clamp_min(1e-12)[..., None, :]).clamp(0.0, 1.0)
    dist = (rel - t[..., None] * d[..., None, :, :]).norm(dim=-1)
    k = dist.argmin(dim=-1, keepdim=True)
    return k[..., 0].to(p.dtype) + t.gather(-1, k)[..., 0]


def sample_polyline(poly, u):
    """Points of polylines poly [..., M, 2] at continuous vertex indices u [..., K] -> [..., K, 2]."""
    num_m = poly.shape[-2]
    u = u.clamp(0.0, float(num_m - 1))
    i0 = u.floor().clamp(max=num_m - 2).long()
    f = (u - i0.to(u.dtype))[..., None]
    idx = i0[..., None].expand(*i0.shape, 2)
    p0 = poly.gather(-2, idx)
    p1 = poly.gather(-2, idx + 1)
    return p0 + f * (p1 - p0)


@torch.no_grad()
def segment_targets(joints_px, gt_px, num_points=16, min_len_px=4.0, ridge=1e-2):
    """The GT under each predicted segment: joints_px [P, S + 1, 2] (px) are projected onto their GT polylines
    gt_px [P, M, 2] (px, uniform in arc length); the GT between the projections of joints s and s + 1 is
    resampled (num_points) and fit with a pinned cubic -> (patterns [P, S, 4], chord lengths [P, S] px,
    valid [P, S]: GT piece at least min_len_px long)."""
    num_p, num_j = joints_px.shape[:2]
    num_s, num_m = num_j - 1, gt_px.shape[-2]
    u = project_to_polyline(joints_px, gt_px)                                     # [P, S+1]
    w = torch.linspace(0.0, 1.0, num_points, device=gt_px.device, dtype=gt_px.dtype)
    uu = u[:, :-1, None] + (u[:, 1:] - u[:, :-1])[..., None] * w                   # [P, S, m]
    sub = sample_polyline(gt_px[:, None].expand(num_p, num_s, num_m, 2), uu)      # [P, S, m, 2]
    fit = fit_cubic_pinned(sub, ridge)
    arc = (sub[..., 1:, :] - sub[..., :-1, :]).norm(dim=-1).sum(-1)
    chord = (fit[..., 3, :] - fit[..., 0, :]).norm(dim=-1).clamp_min(CANON_MIN_LEN_PX)
    return to_canon(fit), chord, arc >= min_len_px


def arc_pieces(gt_px, num_pieces, num_points=16, ridge=1e-2):
    """Polylines gt_px [..., M, 2] (px, uniform in arc length) cut into num_pieces equal arc pieces, each fit
    with a pinned cubic -> [..., S, 4, 2] (to_canon gives the segment patterns of a lane whose joints sit at its
    arc thirds; the codebook script)."""
    num_m = gt_px.shape[-2]
    edges = torch.linspace(0.0, float(num_m - 1), num_pieces + 1, device=gt_px.device, dtype=gt_px.dtype)
    w = torch.linspace(0.0, 1.0, num_points, device=gt_px.device, dtype=gt_px.dtype)
    uu = edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * w                  # [S, m]
    lead = gt_px.shape[:-2]
    sub = sample_polyline(gt_px[..., None, :, :].expand(*lead, num_pieces, num_m, 2), uu.expand(*lead, *uu.shape))
    return fit_cubic_pinned(sub, ridge)


# ------------------------------------------------------------------ prototypes
def mixture(logits, protos):
    """Softmax mixtures of prototypes protos [K, 4] by logits [..., K] -> [..., 4]."""
    return torch.softmax(logits, dim=-1) @ protos.to(logits.dtype)


def assign_logits(pattern, protos, sigma):
    """Soft-assignment logits of patterns [..., 4] to prototypes [K, 4] (Gaussian, width sigma) -> [..., K]."""
    d2 = ((pattern[..., None, :] - protos.to(pattern.dtype)) ** 2).sum(-1)
    return -d2 / (2.0 * float(sigma) ** 2)


def prior_logits(pattern, protos, sigma, floor=0.5):
    """Start logits of curves with patterns [..., 4]: the log of their soft assignment mixed with a uniform floor
    (share floor), so no prototype starts more than log(1 + (1 - floor) K / floor) above another -> [..., K]."""
    k = protos.shape[0]
    p = torch.softmax(assign_logits(pattern, protos, sigma), dim=-1)
    return torch.log((1.0 - floor) * p + floor / k)


def soft_ce(logits, target):
    """Cross-entropy of logits [..., K] against target distributions [..., K] -> [...]."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1)


@torch.no_grad()
def kmeans(x, k, iters=50, seed=0):
    """k-means (k-means++ seeding, Lloyd iterations) of x [N, D] -> (centers [k, D], assignment [N]). An empty
    cluster is re-seeded at the point farthest from its center."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    num = x.shape[0]
    centers = x[torch.randint(num, (1,), generator=gen)].clone()
    d2 = ((x - centers[0]) ** 2).sum(-1)
    for _ in range(1, k):
        p = d2 / d2.sum().clamp_min(1e-12)
        i = torch.multinomial(p.cpu(), 1, generator=gen).to(x.device)
        centers = torch.cat([centers, x[i]], dim=0)
        d2 = torch.minimum(d2, ((x - x[i]) ** 2).sum(-1))
    assign = torch.zeros(num, dtype=torch.long, device=x.device)
    for _ in range(iters):
        dist = torch.cdist(x, centers)
        assign = dist.argmin(-1)
        for j in range(k):
            m = assign == j
            if bool(m.any()):
                centers[j] = x[m].mean(0)
            else:
                far = dist.min(-1).values.argmax()
                centers[j] = x[far]
                assign[far] = j
    return centers, torch.cdist(x, centers).argmin(-1)


def proto_sigma(protos):
    """Soft-target width of a codebook: half the median distance from each prototype to its nearest other."""
    d = torch.cdist(protos, protos)
    d = d + torch.eye(len(protos), device=protos.device, dtype=protos.dtype) * 1e9
    return 0.5 * float(d.min(-1).values.median())
