"""Sparse Bezier priors and structured prior perturbation (V11 active path)."""
import math

import torch
import torch.nn as nn

from .geometry import inverse_sigmoid, project_support_y


# ---------------------------------------------------------------------------
# CLRNet/CLRerNet straight-anchor tiling -> collinear support-local cubic CPs
# ---------------------------------------------------------------------------
def clrnet_straight_prior_params(num_priors):
    """[K, 3] = [start_y_bottom_offset, start_x, theta] (CLRNet tiling).

    start_y here is the ORIGINAL CLRNet bottom-offset value; it is only used to
    construct image-space control points and never leaves this function.
    Port of UnLaneDet ``make_clrernet_straight_prior_parameters``.
    """
    num_priors = int(num_priors)
    params = torch.zeros((num_priors, 3), dtype=torch.float32)
    bottom = num_priors * 3 // 4
    left = num_priors // 8
    right_start = left + bottom

    side_levels = max(1, (left + 1) // 2)
    denom = left // 2 - 1
    if denom > 0:
        strip = 0.5 / float(denom)
        side_y = [(i // 2) * strip for i in range(left)]
    elif left > 0:
        levels = [0.0] if side_levels == 1 else torch.linspace(0.0, 0.5, side_levels).tolist()
        side_y = [levels[min(i // 2, side_levels - 1)] for i in range(left)]
    else:
        side_y = []

    for i in range(left):
        params[i] = torch.tensor([side_y[i], 0.0, 0.16 if i % 2 == 0 else 0.32])

    bottom_strip = 1.0 / float(max(1, bottom // 4 + 1))
    for i in range(left, right_start):
        params[i] = torch.tensor(
            [0.0, ((i - left) // 4 + 1) * bottom_strip, 0.2 * (i % 4 + 1)]
        )

    right_count = num_priors - right_start
    if right_count > 0:
        right_levels = max(1, (right_count + 1) // 2)
        if denom > 0:
            right_y = [(i // 2) * strip for i in range(right_count)]
        else:
            levels = [0.0] if right_levels == 1 else torch.linspace(0.0, 0.5, right_levels).tolist()
            right_y = [levels[min(i // 2, right_levels - 1)] for i in range(right_count)]
        for local_i, i in enumerate(range(right_start, num_priors)):
            params[i] = torch.tensor([right_y[local_i], 1.0, 0.68 if i % 2 == 0 else 0.84])
    return params


def clrernet_style_bezier_priors(num_priors, img_w, img_h, eps=1e-4,
                                 visible_only=True, min_support=1.0 / 71.0):
    """[K, 4, 2] support-local collinear cubic CPs, top -> bottom, image coords.

    Port of UnLaneDet ``make_clrernet_style_bezier_priors``.
    """
    fractions = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    priors = []
    for start_y, start_x, theta in clrnet_straight_prior_params(num_priors).tolist():
        max_support = max(0.0, 1.0 - start_y)
        if max_support < min_support:
            start_y = max(0.0, 1.0 - min_support)
            max_support = min_support
        tan_theta = math.tan(theta * math.pi + 1.0e-5)
        if abs(tan_theta) < 1.0e-8:
            tan_theta = 1.0e-8 if tan_theta >= 0.0 else -1.0e-8
        dx_du = img_h / ((img_w - 1.0) * tan_theta)
        support = max_support
        if visible_only and abs(dx_du) > 1.0e-12:
            boundary = (1.0 - start_x) / dx_du if dx_du > 0.0 else (0.0 - start_x) / dx_du
            if boundary >= 0.0:
                support = min(support, boundary)
        support = max(min_support, min(max_support, support))
        y_bottom = 1.0 - start_y
        y_top = max(0.0, y_bottom - support)
        ys = y_top + fractions * (y_bottom - y_top)
        xs = start_x + (y_bottom - ys) * dx_du
        priors.append(torch.stack([xs.clamp(eps, 1.0 - eps), ys.clamp(0.0, 1.0)], dim=-1))
    return torch.stack(priors, dim=0)


class BezierPriorBank(nn.Module):
    """Fixed prior logits + small learnable delta (V11: learnable_pos_prior=False,
    allow_small_delta=True, prior_delta_scale=0.1)."""

    def __init__(self, num_priors, img_w, img_h, delta_scale=0.1, eps=1e-4,
                 visible_only=True, min_support=1.0 / 71.0):
        super().__init__()
        self.num_priors = int(num_priors)
        self.delta_scale = float(delta_scale)
        self.eps = float(eps)
        cp = clrernet_style_bezier_priors(num_priors, img_w, img_h, eps, visible_only, min_support)
        cp = project_support_y(cp, eps)
        logits = inverse_sigmoid(cp.clamp(eps, 1.0 - eps).reshape(self.num_priors, 8), eps=1e-12)
        self.register_buffer("prior_logits", logits)
        self.prior_delta = nn.Parameter(torch.zeros_like(logits))

    def forward(self, batch_size):
        """Clean support-local CPs [B, K, 4, 2]."""
        logits = self.prior_logits + self.delta_scale * self.prior_delta
        cp = torch.sigmoid(logits.view(self.num_priors, 4, 2))
        cp = project_support_y(cp, self.eps)
        # V11 round-trips through clamp + inverse_sigmoid + sigmoid; the clamp
        # is the only numerical effect.
        cp = cp.clamp(self.eps, 1.0 - self.eps)
        return cp.unsqueeze(0).expand(batch_size, -1, -1, -1).float()


# ---------------------------------------------------------------------------
# Structured perturbation for the collaborative auxiliary branch
# ---------------------------------------------------------------------------
def make_beta_schedule(schedule, timesteps, beta_start=1e-4, beta_end=2e-2,
                       cosine_s=0.008, max_beta=0.999):
    schedule = schedule.lower()
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
        ac = torch.cos(((x / timesteps) + cosine_s) / (1.0 + cosine_s) * math.pi * 0.5) ** 2
        ac = ac / ac[0]
        return (1.0 - ac[1:] / ac[:-1]).clamp(1e-8, max_beta)
    raise ValueError(f"Unknown beta schedule {schedule!r}.")


class StructuredPriorPerturbation(nn.Module):
    """Port of V11 ``q_sample_structured_coeffs`` + ``apply_structured_prior_perturbation``
    in support_conditioned_prior_y mode."""

    def __init__(self, timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2,
                 noise_scale=1.0, coeff_dim=4, coeff_scale=1.0, use_tanh=True,
                 max_translate=0.06, max_slope=0.16, max_curve=0.08, max_y_shift=0.06,
                 clamp_x=True, clamp_y=True, eps=1e-4):
        super().__init__()
        betas = make_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_ab", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_omab", torch.sqrt(1.0 - alphas_cumprod), persistent=False)
        self.timesteps = int(timesteps)
        self.noise_scale = float(noise_scale)
        self.coeff_dim = int(coeff_dim)
        self.coeff_scale = float(coeff_scale)
        self.use_tanh = bool(use_tanh)
        self.max_translate = float(max_translate)
        self.max_slope = float(max_slope)
        self.max_curve = float(max_curve)
        self.max_y_shift = float(max_y_shift)
        self.clamp_x = bool(clamp_x)
        self.clamp_y = bool(clamp_y)
        self.eps = float(eps)

    @staticmethod
    def _normalize_basis(basis, eps=1e-6):
        return basis / basis.abs().amax(dim=-1, keepdim=True).clamp(min=eps)

    def sample_coeffs(self, shape_bk, t, device, dtype):
        """shape_bk = (B', K); t: [B'] long. Returns [B', K, C]."""
        noise = torch.randn(shape_bk + (self.coeff_dim,), device=device, dtype=dtype) * self.noise_scale
        t = t.long().clamp(0, self.timesteps - 1)
        sqrt_omab = self.sqrt_omab.gather(0, t).view(-1, 1, 1).to(dtype)
        # Clean coefficients are zero, so sqrt_ab * 0 drops out.
        return sqrt_omab * noise

    def perturb(self, control_points, coeffs):
        x = control_points[..., 0]
        y = control_points[..., 1]
        coeffs = coeffs * self.coeff_scale
        if self.use_tanh:
            coeffs = torch.tanh(coeffs)

        def c(idx):
            if coeffs.shape[-1] <= idx:
                return coeffs.new_zeros(coeffs.shape[:-1] + (1,))
            return coeffs[..., idx:idx + 1]

        y_centered = y - y.mean(dim=-1, keepdim=True)
        slope_basis = self._normalize_basis(y_centered)
        curve_basis = y_centered ** 2
        curve_basis = self._normalize_basis(curve_basis - curve_basis.mean(dim=-1, keepdim=True))
        x = x + (c(0) * self.max_translate
                 + c(1) * self.max_slope * slope_basis
                 + c(2) * self.max_curve * curve_basis)
        num_cp = control_points.shape[-2]
        y_basis = torch.linspace(1.0, 0.0, num_cp, dtype=control_points.dtype,
                                 device=control_points.device)
        y = y + c(3) * self.max_y_shift * y_basis
        if self.clamp_x:
            x = x.clamp(0.01, 0.99)
        if self.clamp_y:
            y = y.clamp(0.0, 1.0)
        return project_support_y(torch.stack((x, y), dim=-1), self.eps)
