"""Truncated anchored diffusion on the anchor control points (``bbox_head.diffusion_cfg``).

DiffusionDrive-style: the noise goes on the anchors (not on the GT), only up to a
truncated timestep, and the head learns to take a noisy anchor to the clean lane.
Independent of the auxiliary branch's ``perturb_cfg``: its own schedule, its own
noise, applied to the main branch.

State: each anchor's BRR state in the "global" / "anchored" frames, the 4
control-point x's (normalized, x_px / (img_w - 1)) and the start y (image y,
1 = bottom).

Forward process (training), t ~ U{0, ..., T}, T = ``trunc_t``:
    x_t = anchor + sigma(t) * eps,  eps ~ N(0, I)
    sigma(t) = sigma_max * sqrt((1 - abar_t) / (1 - abar_T))    schedule="ddpm"
    sigma(t) = sigma_max * t / T                                 schedule="linear"
abar_t is the DDPM cumulative product of (1 - beta) (``beta_schedule`` over
``timesteps``), so "ddpm" keeps the DDPM noise growth over the truncated range.
There is no sqrt(abar_t) shrink of the anchor: it would pull every anchor
towards the image centre, so the noise is centred on the anchor itself and
sigma_max (``sigma_x_px`` in x, ``sigma_y`` in start y) bounds how far a noisy
anchor strays.

Denoiser: one full cascade pass (all refinement stages) predicts the clean state
(x0) and the score. A sinusoidal timestep embedding scales and shifts every
stage's RoI feature; its last layer starts at zero, so at init the head ignores t.

Sampler (inference): DDIM with eta = 0 in the x0 parametrization, at
``infer_steps`` timesteps T = t_0 > t_1 > ... > t_{S-1} > 0:
    x_{t_{k+1}} = x0_hat + sigma(t_{k+1}) / sigma(t_k) * (x_{t_k} - x0_hat)
The last pass's predictions are the output. The first input is the clean anchor
(``infer_noise=False``, deterministic) or anchor + sigma(T) * eps with a fixed
seed. infer_steps=1 costs what the base head costs.

Training rollouts (``rollout_prob``): with that probability an iteration runs
the first k sampler steps without gradient (k uniform in 1..S-1) and trains the
next step on their DDIM output, so the later steps see at training time the
inputs they get at test time. Those steps run in eval mode, as at test time:
the stage jitter (brr_cfg.stage_jitter) and dropout act only on the trained
pass. 0 = DiffusionDrive (noisy anchors only).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .priors import make_beta_schedule

SCHEDULES = ("ddpm", "linear")


class AnchorDiffusion(nn.Module):
    def __init__(self, hidden_dim: int, img_w: int, timesteps: int = 1000,
                 beta_schedule: str = "linear", beta_start: float = 1e-4, beta_end: float = 2e-2,
                 trunc_t: int = 50, schedule: str = "ddpm", sigma_x_px: float = 16.0,
                 sigma_y: float = 0.02, infer_steps: int = 2, infer_noise: bool = False,
                 infer_seed: int = 0, rollout_prob: float = 0.5, time_cond: bool = True):
        super().__init__()
        self.timesteps = int(timesteps)
        self.trunc_t = int(trunc_t)
        if not 1 <= self.trunc_t <= self.timesteps:
            raise ValueError(f"diffusion_cfg.trunc_t must be in [1, timesteps={self.timesteps}]")
        if schedule not in SCHEDULES:
            raise ValueError(f"diffusion_cfg.schedule must be one of {SCHEDULES}, got {schedule!r}")
        self.schedule = schedule
        self.infer_steps = int(infer_steps)
        if not 1 <= self.infer_steps <= self.trunc_t:
            raise ValueError("diffusion_cfg.infer_steps must be in [1, trunc_t]")
        if sigma_x_px < 0 or sigma_y < 0:
            raise ValueError("diffusion_cfg.sigma_x_px / sigma_y must be >= 0")
        self.rollout_prob = float(rollout_prob)
        if not 0.0 <= self.rollout_prob <= 1.0:
            raise ValueError("diffusion_cfg.rollout_prob must be in [0, 1]")
        self.sigma_x_px = float(sigma_x_px)
        self.sigma_x = float(sigma_x_px) / float(img_w - 1)
        self.sigma_y = float(sigma_y)
        self.infer_noise = bool(infer_noise)
        self.infer_seed = int(infer_seed)
        if schedule == "ddpm":
            betas = make_beta_schedule(beta_schedule, self.timesteps, beta_start, beta_end)
            abar = torch.cat([torch.ones(1), torch.cumprod(1.0 - betas.double(), 0).float()])
            abar = abar[: self.trunc_t + 1]
            unit = torch.sqrt((1.0 - abar) / (1.0 - abar[-1]))
        else:
            unit = torch.arange(self.trunc_t + 1, dtype=torch.float32) / float(self.trunc_t)
        # sigma(t) / sigma_max for t = 0..T: 0 at t = 0, 1 at t = T
        self.register_buffer("unit_sigma", unit.float(), persistent=False)
        self.hidden_dim = int(hidden_dim)
        self.time_mlp = (nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
                                       nn.Linear(self.hidden_dim, 2 * self.hidden_dim))
                         if time_cond else None)
        self.zero_init()

    def zero_init(self):
        """The timestep modulation starts as the identity."""
        if self.time_mlp is not None:
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)

    # ---- schedule -------------------------------------------------------------
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """sigma(t) / sigma_max, t: long tensor in [0, T]."""
        return self.unit_sigma.to(t.device)[t.long().clamp(0, self.trunc_t)]

    def sampler_steps(self):
        """The sampler's timesteps [T, ..., t_{S-1}], decreasing, all > 0."""
        s, big_t = self.infer_steps, self.trunc_t
        return [int(round(big_t * (s - k) / s)) for k in range(s)]

    def sample_t(self, shape, device) -> torch.Tensor:
        return torch.randint(0, self.trunc_t + 1, shape, device=device)

    def perturb(self, cp_x, y_start, t, eps_x, eps_y, margin: float, y_min: float):
        """x_t = anchor + sigma(t) * eps. cp_x [..., 4], y_start / t [...]."""
        s = self.sigma(t).to(cp_x.dtype)
        x = cp_x + (s * self.sigma_x)[..., None] * eps_x
        y = y_start + s * self.sigma_y * eps_y
        return x.clamp(-margin, 1.0 + margin), y.clamp(y_min, 1.0)

    def ddim(self, x_t, y_t, x0, y0, t, t_next, margin: float, y_min: float):
        """One deterministic DDIM step from t to t_next with the predicted x0."""
        r = (self.sigma(t_next) / self.sigma(t).clamp_min(1e-12)).to(x_t.dtype)
        x = x0 + r[..., None] * (x_t - x0)
        y = y0 + r * (y_t - y0)
        return x.clamp(-margin, 1.0 + margin), y.clamp(y_min, 1.0)

    # ---- timestep conditioning ------------------------------------------------
    def embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device,
                                                            dtype=torch.float32) / half)
        args = t.float()[..., None] * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.hidden_dim:
            emb = torch.nn.functional.pad(emb, (0, self.hidden_dim - emb.shape[-1]))
        return emb

    def modulate(self, roi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """roi [B, Q, D] * (1 + scale(t)) + shift(t), t [B, Q]."""
        if self.time_mlp is None:
            return roi
        scale, shift = self.time_mlp(self.embed(t).to(roi.dtype)).chunk(2, dim=-1)
        return roi * (1.0 + scale) + shift
