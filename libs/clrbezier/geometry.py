"""Bezier Reference Refinement (BRR) geometry, CLRerNet convention.

Conventions (identical to the official CLRerNet head / get_lanes)
-----------------------------------------------------------------
Rows:
    Row index i = 0..n_strips, bottom -> top.
    prior_ys[i] = 1 - i / n_strips  (normalized image y, top = 0, bottom = 1).
Prediction tensor [..., 6 + n_offsets]:
    [0:2]  classification logits (background, lane)
    [2]    start_y  = normalized IMAGE y of the lane start (1 = bottom).
                      Start row index = round((1 - start_y) * n_strips).
    [3]    start_x  (normalized by img_w - 1)
    [4]    theta    (normalized by pi)
    [5]    length   (visible row count / n_strips)
    [6:]   x at every row, normalized by (img_w - 1), bottom -> top.
Official pred_dict view:
    anchor_params = pred[..., 2:5], lengths = pred[..., 5:6], xs = pred[..., 6:].

BRR state (per query)
---------------------
    y_start : visible-bottom image y (== official start_y)
    cp_x    : [P0x, P1x, P2x, P3x], cubic x(y) over FIXED global y = [0, 1/3, 2/3, 1]
              (P0 = image top, P3 = image bottom), normalized by (img_w - 1).
Length is not persistent; every stage predicts it fresh (as in CLRerNet).

Local (support-conditioned) control points are used only by the prior
initializer and the structured perturbation. They are [..., 4, 2] (x, y) in
normalized image coordinates, ordered top -> bottom, with P1/P2 y at exactly
1/3 and 2/3 of the visible support.
"""
import math

import torch

CP_FRACTIONS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def bernstein_basis(y):
    """Cubic Bernstein basis at global y. y: [..., R] -> [..., R, 4]."""
    omt = 1.0 - y
    return torch.stack(
        (omt.pow(3), 3.0 * omt.pow(2) * y, 3.0 * omt * y.pow(2), y.pow(3)), dim=-1
    )


CP_FRAMES = ("global", "anchored", "support")


def support_top(y_start, length, n_strips, min_span):
    """Top of the support frame for a lane starting at ``y_start``.

    ``length`` is the official visible length (row count / n_strips), so the
    visible span in image y is ``length - 1/n_strips``. The span is kept at
    least ``min_span`` (a short or collapsed length must not squeeze the four
    control points into a couple of rows) and at most ``y_start`` (the frame
    never reaches above the image top).
    """
    one_row = 1.0 / float(n_strips)
    span = (length - one_row).clamp_min(float(min_span))
    span = torch.minimum(span, y_start).clamp_min(1e-3)
    return y_start - span


def bezier_t(query_y, y_start=None, frame="global", eps=1e-3, y_top=None):
    """Map image y to the Bernstein parameter t.

    "global"   t = y. The four control points sit at image y = 0, 1/3, 2/3, 1,
               so the curve's domain is the whole image whatever the lane does.

    "support"  t = (y - y_top) / (y_start - y_top), y_top from the length
               (``support_top``). The control points sit at the thirds of the
               lane's own visible span, so start, length and shape are three
               separate things.

    "anchored" t = y / y_start. The control points sit at 0, y_start/3,
               2*y_start/3, y_start: the last one *is* the lane's start point
               and the first is pinned to the image top, which is CLRNet's
               anchor span. A lane entering the frame halfway up then covers
               its own domain completely instead of half of it, and the basis
               functions are evaluated where the data actually is.

    Rows below the start give t > 1. A cubic extrapolates violently there, and
    the official decoder does walk below the start for ``extend_bottom``, so
    those rows are handled by the linear continuation in ``eval_cubic``
    rather than by the cubic itself.
    """
    y = torch.as_tensor(query_y)
    if frame == "global":
        return y
    if frame not in CP_FRAMES:
        raise ValueError(f"cp_frame must be one of {CP_FRAMES}, got {frame!r}")
    if y_start is None:
        raise ValueError(f"the {frame} frame needs y_start")
    if frame == "support":
        if y_top is None:
            raise ValueError("the support frame needs y_top")
        top = y_top
    else:
        top = torch.zeros_like(y_start)
    denom = (y_start - top).clamp_min(eps)
    while denom.ndim < y.ndim:
        denom = denom.unsqueeze(-1)
        top = top.unsqueeze(-1)
    return (y - top) / denom


def eval_cubic(cp_x, query_y, y_start=None, frame="global", eps=1e-3, y_top=None):
    """x(y) in any frame, with a linear continuation outside the frame.

    ``y_top`` is needed (and only used) by the support frame.
    """
    y = torch.as_tensor(query_y, device=cp_x.device, dtype=cp_x.dtype)
    while y.ndim < cp_x.ndim:
        y = y.unsqueeze(0)
    if frame == "global":
        return eval_global_cubic(cp_x, y)

    t = bezier_t(y, y_start, frame=frame, eps=eps, y_top=y_top)
    t_in = t.clamp(0.0, 1.0)
    omt = 1.0 - t_in
    x = (omt.pow(3) * cp_x[..., 0:1]
         + 3.0 * omt.pow(2) * t_in * cp_x[..., 1:2]
         + 3.0 * omt * t_in.pow(2) * cp_x[..., 2:3]
         + t_in.pow(3) * cp_x[..., 3:4])
    # dx/dt at the clamped endpoint, for a C1 linear continuation outside [0, 1].
    slope = 3.0 * (omt.pow(2) * (cp_x[..., 1:2] - cp_x[..., 0:1])
                   + 2.0 * omt * t_in * (cp_x[..., 2:3] - cp_x[..., 1:2])
                   + t_in.pow(2) * (cp_x[..., 3:4] - cp_x[..., 2:3]))
    return x + slope * (t - t_in)


def eval_global_cubic(cp_x, query_y):
    """Evaluate x(y) of the global cubic.

    cp_x: [..., 4]; query_y: [R] (broadcast) or [..., R]. Returns [..., R].
    """
    y = torch.as_tensor(query_y, device=cp_x.device, dtype=cp_x.dtype)
    while y.ndim < cp_x.ndim:
        y = y.unsqueeze(0)
    omt = 1.0 - y
    return (
        omt.pow(3) * cp_x[..., 0:1]
        + 3.0 * omt.pow(2) * y * cp_x[..., 1:2]
        + 3.0 * omt * y.pow(2) * cp_x[..., 2:3]
        + y.pow(3) * cp_x[..., 3:4]
    )


def project_support_y(control_points, eps=1e-4):
    """Keep x and the two support endpoints; put P1/P2 y at 1/3 and 2/3.

    Port of V11 ``_project_control_points_to_support_conditioned_y``.
    control_points: [..., 4, 2] (top -> bottom).
    """
    x = control_points[..., 0]
    raw_top = torch.nan_to_num(control_points[..., 0, 1], nan=0.0, posinf=1.0, neginf=0.0)
    raw_bottom = torch.nan_to_num(control_points[..., 3, 1], nan=1.0, posinf=1.0, neginf=0.0)
    swap = (raw_top > raw_bottom).detach()
    y_top = torch.where(swap, raw_bottom, raw_top).clamp(0.0, 1.0)
    y_bottom = torch.where(swap, raw_top, raw_bottom).clamp(0.0, 1.0)
    too_short = ((y_bottom - y_top) < eps).detach()
    center = 0.5 * (y_top + y_bottom)
    fallback_top = (center - 0.5 * eps).clamp(0.0, 1.0 - eps)
    y_top = torch.where(too_short, fallback_top, y_top)
    y_bottom = torch.where(too_short, fallback_top + eps, y_bottom)
    fractions = control_points.new_tensor(CP_FRACTIONS)
    y = y_top.unsqueeze(-1) + (y_bottom - y_top).unsqueeze(-1) * fractions
    return torch.stack((x, y), dim=-1)


def globalize_local_cp(control_points, margin, eps=1e-4, cp_frame="global",
                       legacy_tangent=False):
    """Convert support-local cubic CPs to the fixed-global-y BRR state.

    Port of V11 ``_brr_globalize_control_points``. Exact for the
    support-local parameterization before the trust-region clamp.

    ``legacy_tangent=True`` reproduces the pre-2026-10-06 anchored result,
    which dropped the frame Jacobian (only matters for cp_frame="anchored";
    keep it for evaluating checkpoints trained before the fix).

    Not used for cp_frame="support": there the local CPs already are the state
    (see ``support_state_from_local``).

    Returns:
        cp_x:    [..., 4] global CPs clamped to [-margin, 1 + margin]
        y_start: [...]    visible-bottom image y (official start_y)
    """
    cp = project_support_y(control_points.float(), eps)
    x = cp[..., 0]
    y_top = cp[..., 0, 1]
    y_bottom = cp[..., 3, 1]
    span = (y_bottom - y_top).clamp_min(eps)

    def eval_local(t):
        omt = 1.0 - t
        return (
            omt.pow(3) * x[..., 0]
            + 3.0 * omt.pow(2) * t * x[..., 1]
            + 3.0 * omt * t.pow(2) * x[..., 2]
            + t.pow(3) * x[..., 3]
        )

    def deriv_local(t):
        omt = 1.0 - t
        return 3.0 * (
            omt.pow(2) * (x[..., 1] - x[..., 0])
            + 2.0 * omt * t * (x[..., 2] - x[..., 1])
            + t.pow(2) * (x[..., 3] - x[..., 2])
        )

    # The target frame spans [0, frame_bottom]: the whole image for "global",
    # the lane's own start point for "anchored" (where t_bottom is 1 by
    # construction, so the fourth control point IS the start point).
    frame_bottom = torch.ones_like(y_bottom) if cp_frame == "global" else y_bottom
    t_top = (0.0 - y_top) / span
    t_bottom = (frame_bottom - y_top) / span
    # Affine reparameterization u = t_top + (t_bottom - t_top) * t. Its
    # Jacobian du/dt = frame_bottom / span scales the end tangents; dropping it
    # is only exact for the global frame (frame_bottom = 1).
    du_dt = (1.0 / span) if legacy_tangent else (frame_bottom / span)
    q0 = eval_local(t_top)
    q1 = q0 + deriv_local(t_top) * du_dt / 3.0
    q3 = eval_local(t_bottom)
    q2 = q3 - deriv_local(t_bottom) * du_dt / 3.0
    cp_x = torch.stack((q0, q1, q2, q3), dim=-1)
    cp_x = torch.nan_to_num(cp_x, nan=0.5, posinf=1.0 + margin, neginf=-margin)
    cp_x = cp_x.clamp(-margin, 1.0 + margin)
    y_start = y_bottom.clamp(0.0, 1.0)
    return cp_x, y_start


def reparam_cp_to_frame(cp_x, ratio):
    """Express a cubic's control points in a frame whose end sits at ``ratio``.

    Both frames start at the image top, so the map between their Bernstein
    parameters is t_other = ratio * t_this, and de Casteljau subdivision at
    ``ratio`` gives the exact control points of the same curve. Valid for
    ratio > 1 too, where it is an extrapolation of the same cubic.

    Needed because with cp_frame="anchored" the GT control points are fitted in
    the GT's own frame (t = y / y_start_gt) while the predicted state lives in
    the predicted frame (t = y / y_start_pred). Comparing the two coefficient
    vectors directly penalizes a perfectly correct curve whenever the two
    start points differ — 50 px of phantom error for a 0.1 difference in
    y_start. This removes that.
    """
    r = ratio.unsqueeze(-1) if ratio.ndim == cp_x.ndim - 1 else ratio
    omr = 1.0 - r
    p0, p1, p2, p3 = (cp_x[..., i:i + 1] for i in range(4))
    q0 = p0
    q1 = omr * p0 + r * p1
    q2 = omr.pow(2) * p0 + 2.0 * r * omr * p1 + r.pow(2) * p2
    q3 = (omr.pow(3) * p0 + 3.0 * omr.pow(2) * r * p1
          + 3.0 * omr * r.pow(2) * p2 + r.pow(3) * p3)
    return torch.cat((q0, q1, q2, q3), dim=-1)


def transport_cp(cp_x, src_top, src_bottom, dst_top, dst_bottom, num_samples=16,
                 outside_weight=0.01, eps=1e-3):
    """Re-express a curve given on frame [src_top, src_bottom] in frame [dst_top, dst_bottom].

    The source curve is evaluated as ``eval_cubic`` draws it (cubic inside its
    frame, C1 linear continuation outside) and the destination cubic is a
    weighted least-squares fit to it:
      * ``num_samples`` points on the overlap of the two frames, weight 1;
      * ``num_samples`` points over the whole destination frame, weight
        ``outside_weight``.
    Where the destination lies inside the source the fit is exact (a cubic
    restricted to a sub-interval is a cubic). Where the destination is larger,
    the curve is kept (almost) exactly on the rows the source actually covers,
    and the small weight on the rest keeps the extension close to the drawn
    linear continuation instead of letting the cubic extrapolate freely. With
    no overlap at all it fits the linear continuation.

    Covers the anchored frame too (both tops = 0). Frame arguments are [...]
    and must be detached unless the caller wants the frame's gradient (the
    map is differentiable in all of its arguments). Runs in float32 with
    autocast disabled (float64 inputs stay float64): the 4x4 normal equations
    are not half-precision safe.
    """
    with torch.autocast(device_type=cp_x.device.type, enabled=False):
        dtype = torch.promote_types(cp_x.dtype, torch.float32)   # fp32, or fp64 if given
        cp = cp_x.to(dtype)
        st, sb, dt, db = (v.to(dtype) for v in (src_top, src_bottom, dst_top, dst_bottom))
        s = torch.linspace(0.0, 1.0, int(num_samples), device=cp.device, dtype=cp.dtype)
        lo = torch.maximum(st, dt)
        hi = torch.minimum(sb, db)
        overlap = (hi - lo).clamp_min(0.0)
        y_in = lo.unsqueeze(-1) + overlap.unsqueeze(-1) * s
        y_all = dt.unsqueeze(-1) + (db - dt).unsqueeze(-1) * s
        ys = torch.cat((y_in, y_all), dim=-1)
        w_in = (overlap > 1e-6).to(cp.dtype).unsqueeze(-1).expand_as(y_in)
        w = torch.cat((w_in, torch.full_like(y_all, float(outside_weight))), dim=-1)
        t = (ys - dt.unsqueeze(-1)) / (db - dt).clamp_min(eps).unsqueeze(-1)
        xs = eval_cubic(cp, ys, sb, "support", eps=eps, y_top=st)
        basis = bernstein_basis(t)                                  # [..., 2S, 4]
        btw = basis.transpose(-1, -2) * w.unsqueeze(-2)             # [..., 4, 2S]
        eye = torch.eye(4, device=cp.device, dtype=cp.dtype)
        lhs = btw @ basis + 1e-8 * eye
        return torch.linalg.solve(lhs, (btw @ xs.unsqueeze(-1))).squeeze(-1)


def support_state_from_local(control_points, margin, n_strips, min_span, eps=1e-4):
    """Initial support-frame state from support-local prior CPs.

    The prior's own visible span becomes the state's length, so for spans of
    at least ``min_span`` the conversion is the identity on x (the support
    frame *is* the local frame). Shorter priors are re-expressed on the
    ``min_span`` frame by ``transport_cp`` (exact: their straight/curved
    piece is continued linearly, as the evaluator does).

    Returns cp_x [..., 4], y_start [...], length [...], frame_top [...].
    """
    cp = project_support_y(control_points.float(), eps)
    x = cp[..., 0]
    y_top = cp[..., 0, 1]
    y_bottom = cp[..., 3, 1].clamp(1.0 / float(n_strips), 1.0)
    length = (y_bottom - y_top).clamp_min(0.0) + 1.0 / float(n_strips)
    frame_top = support_top(y_bottom, length, n_strips, min_span)
    cp_x = transport_cp(x, y_top, y_bottom, frame_top, y_bottom)
    cp_x = torch.nan_to_num(cp_x, nan=0.5, posinf=1.0 + margin, neginf=-margin)
    return cp_x.clamp(-margin, 1.0 + margin), y_bottom, length, frame_top


def update_framed_state(cp_x, y_start, length, frame_top, delta, n_strips, margin,
                        cp_frame, min_span=0.1, transport=True, length_mode="residual",
                        detach_frame=True):
    """BRR update for the start-conditioned frames ("anchored", "support").

    delta: [..., 6] = [d_start_y, length term, dP0x..dP3x].

    length_mode: "residual" (length = input length + delta[1], the support
        frame's default, since its frame is built from the length) or "fresh"
        (length = delta[1], CLRerNet's convention and the anchored default).
    transport: move the input control points into the new frame before
        adding dP (``transport_cp`` with the frame detached), so a change of
        start/length only moves the support and leaves the curve where it
        was. Without it the old coefficients are reinterpreted in the new
        frame and every support step also bends/stretches the curve.
    detach_frame: transport with detached frames. Must match how the
        reference is evaluated: with frame_grad the reference sees the live
        frame, and so must the transport, or autograd differentiates a
        stretched curve the forward pass never drew.

    Returns dict with the clamped state (cp_x, y_start, length, frame_top),
    the state the CP delta was added to (``base``) and the unclamped raw
    values (raw_y, raw_length, raw_delta_x = base + dP before the clamp).
    """
    one_row = 1.0 / float(n_strips)
    raw_y = y_start + delta[..., 0]
    raw_length = (length + delta[..., 1]) if length_mode == "residual" else delta[..., 1]
    new_y = raw_y.clamp(one_row, 1.0)
    # at least two rows, at most every row from the start up to the image top
    new_length = torch.minimum(raw_length.clamp_min(2.0 * one_row), new_y + one_row)
    if cp_frame == "support":
        new_top = support_top(new_y, new_length, n_strips, min_span)
    else:
        new_top = torch.zeros_like(new_y)
    if transport:
        frames = (frame_top, y_start, new_top, new_y)
        if detach_frame:
            frames = tuple(f.detach() for f in frames)
        base = transport_cp(cp_x, *frames)
    else:
        base = cp_x
    raw_x = base + delta[..., 2:6]
    new_x = raw_x.clamp(-margin, 1.0 + margin)
    return dict(cp_x=new_x, y_start=new_y, length=new_length, frame_top=new_top,
                base=base, raw_x=raw_x, raw_y=raw_y, raw_length=raw_length)


def update_brr_state(cp_x, y_start, delta, n_strips, margin):
    """Additive BRR update (V11 independent-CP residuals).

    delta: [..., 6] = [d_start_y, fresh_length, dP0x, dP1x, dP2x, dP3x].

    Official convention: y_start is image y, so it is clamped to
    [1/n_strips, 1] (V11 local start_y in [0, 1 - 1/n_strips]).

    Returns (new_cp_x, new_y_start, raw_cp_x, raw_y_start). The raw values are
    unclamped and used for direct supervision (as in V11); the clamped values
    feed the reference and the next stage.
    """
    one_row = 1.0 / float(n_strips)
    raw_y_start = y_start + delta[..., 0]
    raw_cp_x = cp_x + delta[..., 2:6]
    new_y_start = raw_y_start.clamp(one_row, 1.0)
    new_cp_x = raw_cp_x.clamp(-margin, 1.0 + margin)
    return new_cp_x, new_y_start, raw_cp_x, raw_y_start


def brr_reference(cp_x, y_start, length, prior_ys, sample_x_indices, img_w, img_h,
                  n_strips, cp_frame="global", frame_top=None, frame_bottom=None):
    """Decode BRR state into a CLRerNet-layout prediction tensor.

    Port of V11 ``_brr_reference_from_cp`` with official start_y.

    ``frame_bottom`` / ``frame_top`` are the frame the control points are
    evaluated in (default: ``y_start`` and, for "support", required top).
    Passing detached copies keeps the curve's gradient out of start/length.

    Returns:
        reference:        [..., 6 + R] (cls slots zero)
        reference_on_map: [..., S] x at RoI sampling rows, clamped to [0, 1]
    """
    n_rows = int(prior_ys.numel())
    ys = prior_ys.to(device=cp_x.device, dtype=cp_x.dtype)
    fb = y_start if frame_bottom is None else frame_bottom
    x_full = eval_cubic(cp_x, ys, fb, cp_frame, y_top=frame_top)
    length = torch.as_tensor(length, device=cp_x.device, dtype=cp_x.dtype)

    reference = cp_x.new_zeros(cp_x.shape[:-1] + (6 + n_rows,))
    reference[..., 2] = y_start
    reference[..., 5] = length
    reference[..., 6:] = x_full

    # start_x / theta are metadata only (NMS ignores theta).
    one_row = 1.0 / float(n_strips)
    y_bottom = y_start.clamp(one_row, 1.0)
    max_len = y_bottom + one_row
    safe_len = torch.minimum(length.clamp_min(2.0 * one_row), max_len)
    span = (safe_len - one_row).clamp_min(0.0)
    y_top = (y_bottom - span).clamp(0.0, 1.0)
    start_x = eval_cubic(cp_x, y_bottom.unsqueeze(-1), fb, cp_frame, y_top=frame_top).squeeze(-1)
    top_x = eval_cubic(cp_x, y_top.unsqueeze(-1), fb, cp_frame, y_top=frame_top).squeeze(-1)
    dx_pix = (top_x - start_x) * float(img_w - 1)
    dy_pix = span * float(img_h - 1)
    theta = torch.atan2(dy_pix.clamp_min(1.0e-6), dx_pix) / math.pi
    theta = torch.nan_to_num(theta, nan=0.5, posinf=0.99, neginf=0.01).clamp(0.01, 0.99)
    reference[..., 3] = start_x
    reference[..., 4] = theta

    reference_on_map = torch.nan_to_num(
        x_full[..., sample_x_indices], nan=0.5, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    return reference, reference_on_map


def fit_global_cubic_to_clr_rows(target_xs_px, prior_ys, img_w, ridge, margin,
                                 min_valid_points=2, cp_frame="global",
                                 n_strips=None, min_span=0.1, return_frame=False):
    """See ``_fit_global_cubic_to_clr_rows``. Runs in float32 with autocast off:
    under AMP the einsums would build the 4x4 normal equations in fp16, which
    moves the global-frame CP target by ~16 px on average (as in transport_cp)."""
    with torch.autocast(device_type=target_xs_px.device.type, enabled=False):
        return _fit_global_cubic_to_clr_rows(
            target_xs_px.float(), prior_ys.float(), img_w, ridge, margin,
            min_valid_points, cp_frame, n_strips, min_span, return_frame)


def _fit_global_cubic_to_clr_rows(target_xs_px, prior_ys, img_w, ridge, margin,
                                  min_valid_points=2, cp_frame="global",
                                  n_strips=None, min_span=0.1, return_frame=False):
    """Fit fixed-global-y cubic CPs to GT CLR rows (batched, differentiable-free).

    Same objective as UnLaneDet ``GenerateLaneLine._fit_brr_control_points_x``:
    ridge least squares on visible rows, regularized toward the best affine
    x(y) line. Computing it here from the official CLR target removes any
    dataset-pipeline change and guarantees the CP target is built from exactly
    the rows used by LaneIoU.

    Args:
        target_xs_px: [N, R] GT x in network-input pixels, bottom -> top.
            Valid rows satisfy 0 <= x < img_w (same rule as LaneIoU).
    Returns:
        cp_x:  [N, 4] normalized by (img_w - 1), clamped to the BRR margin.
        valid: [N] bool.
        frame: [N, 2] (frame_top, frame_bottom) in image y, only with
               ``return_frame=True``. For "support" it is the GT's own visible
               span (min_span-clamped exactly like the prediction's).
    """
    xs = target_xs_px.float()
    num, num_rows = xs.shape
    if n_strips is None:
        n_strips = num_rows - 1
    if num == 0:
        empty = (xs.new_zeros((0, 4)), torch.zeros((0,), dtype=torch.bool, device=xs.device))
        return empty + (xs.new_zeros((0, 2)),) if return_frame else empty
    y = prior_ys.to(device=xs.device, dtype=xs.dtype).view(1, num_rows).expand(num, num_rows)
    valid = torch.isfinite(xs) & (xs >= 0.0) & (xs < float(img_w))
    x = torch.where(valid, xs / float(max(1, img_w - 1)), torch.zeros_like(xs))
    mask = valid.to(xs.dtype)
    count = mask.sum(dim=-1)

    frame_top = xs.new_zeros((num,))
    frame_bottom = xs.new_ones((num,))
    if cp_frame in ("anchored", "support"):
        # The GT's own start point: the lowest row it occupies. Fitting in
        # t = y / y_start puts the control points on the same frame the head
        # predicts in, so the CP loss and the reference agree.
        row_y = prior_ys.to(device=xs.device, dtype=xs.dtype).view(1, num_rows)
        y_start_gt = (row_y * mask).max(dim=-1).values.clamp_min(1e-3)
        frame_bottom = y_start_gt
        if cp_frame == "support":
            # ... and its own top: the highest row it occupies.
            y_top_gt = torch.where(mask > 0, row_y.expand(num, num_rows),
                                   torch.full_like(xs, 2.0)).min(dim=-1).values
            y_top_gt = torch.minimum(y_top_gt, y_start_gt)
            length_gt = (y_start_gt - y_top_gt) + 1.0 / float(n_strips)
            frame_top = support_top(y_start_gt, length_gt, n_strips, min_span)
        y = ((y - frame_top.view(num, 1))
             / (frame_bottom - frame_top).clamp_min(1e-3).view(num, 1)).clamp(0.0, 1.0)

    basis = bernstein_basis(y)  # [N, R, 4]
    weighted = basis * mask.unsqueeze(-1)
    btb = torch.einsum("nri,nrj->nij", weighted, basis)
    btx = torch.einsum("nri,nr->ni", weighted, x)

    safe_count = count.clamp_min(1.0)
    sy = (mask * y).sum(-1)
    syy = (mask * y * y).sum(-1)
    sx = (mask * x).sum(-1)
    syx = (mask * y * x).sum(-1)
    det = safe_count * syy - sy * sy
    det_ok = det.abs() > 1.0e-10
    safe_det = torch.where(det_ok, det, torch.ones_like(det))
    a = torch.where(det_ok, (sx * syy - sy * syx) / safe_det, sx / safe_count)
    b = torch.where(det_ok, (safe_count * syx - sy * sx) / safe_det, torch.zeros_like(det))
    cp_y = xs.new_tensor(CP_FRACTIONS)
    affine = a.unsqueeze(-1) + b.unsqueeze(-1) * cp_y

    ridge = max(float(ridge), 1.0e-8)
    system = btb + ridge * torch.eye(4, device=xs.device, dtype=xs.dtype).unsqueeze(0)
    rhs = btx + ridge * affine
    cp_x = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)

    ok = (count >= float(min_valid_points)) & torch.isfinite(cp_x).all(-1)
    cp_x = torch.where(ok.unsqueeze(-1), cp_x, affine)
    cp_x = torch.nan_to_num(cp_x, nan=0.5, posinf=1.0 + margin, neginf=-margin)
    cp_x = cp_x.clamp(-margin, 1.0 + margin)
    if return_frame:
        return cp_x, ok, torch.stack((frame_top, frame_bottom), dim=-1)
    return cp_x, ok
