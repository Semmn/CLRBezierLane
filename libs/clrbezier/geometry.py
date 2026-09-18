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


def globalize_local_cp(control_points, margin, eps=1e-4):
    """Convert support-local cubic CPs to the fixed-global-y BRR state.

    Port of V11 ``_brr_globalize_control_points``. Exact for the
    support-local parameterization before the trust-region clamp.

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

    t_top = (0.0 - y_top) / span
    t_bottom = (1.0 - y_top) / span
    q0 = eval_local(t_top)
    q1 = q0 + deriv_local(t_top) / span / 3.0
    q3 = eval_local(t_bottom)
    q2 = q3 - deriv_local(t_bottom) / span / 3.0
    cp_x = torch.stack((q0, q1, q2, q3), dim=-1)
    cp_x = torch.nan_to_num(cp_x, nan=0.5, posinf=1.0 + margin, neginf=-margin)
    cp_x = cp_x.clamp(-margin, 1.0 + margin)
    y_start = y_bottom.clamp(0.0, 1.0)
    return cp_x, y_start


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


def brr_reference(cp_x, y_start, length, prior_ys, sample_x_indices, img_w, img_h, n_strips):
    """Decode BRR state into a CLRerNet-layout prediction tensor.

    Port of V11 ``_brr_reference_from_cp`` with official start_y.

    Returns:
        reference:        [..., 6 + R] (cls slots zero)
        reference_on_map: [..., S] x at RoI sampling rows, clamped to [0, 1]
    """
    n_rows = int(prior_ys.numel())
    ys = prior_ys.to(device=cp_x.device, dtype=cp_x.dtype)
    x_full = eval_global_cubic(cp_x, ys)
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
    start_x = eval_global_cubic(cp_x, y_bottom.unsqueeze(-1)).squeeze(-1)
    top_x = eval_global_cubic(cp_x, y_top.unsqueeze(-1)).squeeze(-1)
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


def fit_global_cubic_to_clr_rows(target_xs_px, prior_ys, img_w, ridge, margin, min_valid_points=2):
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
    """
    xs = target_xs_px.float()
    num, num_rows = xs.shape
    if num == 0:
        return xs.new_zeros((0, 4)), torch.zeros((0,), dtype=torch.bool, device=xs.device)
    y = prior_ys.to(device=xs.device, dtype=xs.dtype).view(1, num_rows).expand(num, num_rows)
    valid = torch.isfinite(xs) & (xs >= 0.0) & (xs < float(img_w))
    x = torch.where(valid, xs / float(max(1, img_w - 1)), torch.zeros_like(xs))
    mask = valid.to(xs.dtype)
    count = mask.sum(dim=-1)

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
    return cp_x.clamp(-margin, 1.0 + margin), ok
