"""Per-control-point preconditioning for the BRR update.

The persistent control points sit at fixed global y = [0, 1/3, 2/3, 1] and the
cubic Bernstein basis has global support, so a lane that occupies only part of
the image leaves its upper control points weakly supported by data. Measured on
the 72-row grid, for a lane visible over the bottom 30% of the image the
gradient reaching P0 is ~1.6% of the gradient reaching P3, and the condition
number of the visible-row design matrix is ~800 against ~19 for the equivalent
straight-line (affine) parameterization.

With one shared learning rate that means P0 and P1 train tens of times slower
than P2 and P3 on short lanes. This module rescales the predicted per-control-
point delta by the inverse column norm of the Bernstein design matrix over each
lane's *own* predicted visible span, normalized so a full-height lane is
unchanged.

This is a diagnostic as much as a fix. If it closes the IoU-loss gap against
the straight-line baseline, conditioning was the cause; if it does not, the
explanation is elsewhere and nothing else in the model has been disturbed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .geometry import bernstein_basis


class ControlPointPreconditioner(nn.Module):
    """Diagonal preconditioner on the four control-point residuals.

    Args:
        n_strips: number of row intervals (71 for 72 rows).
        max_gain: upper clamp on the per-control-point scale. Without it a
            barely-visible lane asks for a gain in the hundreds, which turns a
            slow control point into a divergent one. 8 keeps the correction
            meaningful for the spans that dominate CULane while bounding the
            step; raise it only if the diagnostics below stay flat.
        min_length: predicted lengths below this are treated as this, so an
            early-training length of ~0 cannot produce an unbounded gain.
        power: 1.0 fully equalizes the columns; 0.5 applies half the
            correction, which is the safer first run.
    """

    def __init__(self, n_strips: int = 71, max_gain: float = 8.0,
                 min_length: float = 0.1, power: float = 1.0,
                 apply_to: str = "both"):
        super().__init__()
        if apply_to not in ("both", "reference"):
            raise ValueError("apply_to must be 'both' or 'reference'")
        # "reference" scales only the curve the IoU loss sees, leaving the
        # BRR state supervision on the raw delta. That matters for short
        # lanes: fit_global_cubic_to_clr_rows regularizes toward the best
        # affine fit, so a short lane's target control points are mostly the
        # ridge prior rather than the data, and multiplying the step toward
        # them by up to max_gain amplifies a regularization artifact.
        self.apply_to = apply_to
        self.n_strips = int(n_strips)
        self.max_gain = float(max_gain)
        self.min_length = float(min_length)
        self.power = float(power)

        # prior_ys, bottom (y=1) -> top (y=0); B[i, k] = B_k(y_i)
        ys = torch.linspace(1.0, 0.0, steps=self.n_strips + 1)
        basis = bernstein_basis(ys)                       # [R, 4]
        self.register_buffer("row_y", ys, persistent=False)
        self.register_buffer("basis_sq", basis.pow(2), persistent=False)
        # Column norms over the whole image: the normalization reference.
        self.register_buffer("full_norm", basis.pow(2).sum(0).sqrt(), persistent=False)

    def gains(self, y_start: torch.Tensor, length: torch.Tensor,
              frame: str = "global") -> torch.Tensor:
        """Per-control-point scale, shape ``y_start.shape + (4,)``.

        ``y_start`` is image y of the lane start (1 = bottom) and ``length`` is
        the visible row count over ``n_strips``, i.e. the official convention.
        Both are detached: the preconditioner shapes the step, it is not itself
        a thing to learn through.
        """
        y_start = y_start.detach()
        length = length.detach().clamp(self.min_length, 1.0)

        rows = self.row_y.to(device=y_start.device, dtype=y_start.dtype)
        basis_sq = self.basis_sq.to(device=y_start.device, dtype=y_start.dtype)
        full = self.full_norm.to(device=y_start.device, dtype=y_start.dtype)

        # Visible band, expressed in the Bernstein parameter t.
        #   global   : t = y, so the band is [y_start - length, y_start].
        #   anchored : t = y / y_start, so the band is [1 - length/y_start, 1]
        #              and a lane reaching the image top covers t = [0, 1]
        #              entirely — no deficit at all, which is the point of the
        #              anchored frame.
        #   support  : the frame is the visible span itself, t = [0, 1], so
        #              every gain is 1 and the preconditioner is a no-op.
        if frame == "support":
            lo = torch.zeros_like(y_start)[..., None]
            hi = torch.ones_like(lo)
        elif frame == "anchored":
            coverage = (length / y_start.clamp_min(1e-3)).clamp(0.0, 1.0)
            lo = (1.0 - coverage)[..., None]
            hi = torch.ones_like(lo)
        else:
            top = (y_start - length).clamp(0.0, 1.0)
            lo = torch.minimum(top, y_start)[..., None]
            hi = torch.maximum(top, y_start)[..., None]
        mask = ((rows >= lo) & (rows <= hi)).to(y_start.dtype)   # [..., R]

        # Column norms restricted to the visible rows.
        norm = torch.sqrt(torch.matmul(mask, basis_sq).clamp_min(1e-12))  # [..., 4]
        gain = (full / norm).clamp(1.0, self.max_gain)
        if self.power != 1.0:
            gain = gain.pow(self.power)
        return gain

    def forward(self, delta_cp: torch.Tensor, y_start: torch.Tensor,
                length: torch.Tensor, frame: str = "global") -> torch.Tensor:
        return delta_cp * self.gains(y_start, length, frame)

    @torch.no_grad()
    def diagnostics(self, y_start: torch.Tensor, length: torch.Tensor,
                    scores: torch.Tensor = None, topk: int = 4,
                    frame: str = "global") -> dict:
        """Averaged over every query, and over the top-scoring few.

        The plain mean covers all anchors, most of which match nothing and
        carry meaningless predicted geometry, so it says more about the unused
        anchors than about the lanes. The ``_top`` figures average over the
        ``topk`` highest-scoring queries per image, which is roughly the set
        that survives NMS.
        """
        gain = self.gains(y_start, length, frame)
        out = {
            "precond_gain_p0": gain[..., 0].mean(),
            "precond_gain_p1": gain[..., 1].mean(),
            "precond_gain_p3": gain[..., 3].mean(),
            "precond_clipped": (gain >= self.max_gain - 1e-6).float().mean(),
        }
        if scores is not None and gain.ndim >= 2:
            k = min(int(topk), gain.shape[-2])
            idx = scores.topk(k, dim=-1).indices                      # [..., k]
            sel = torch.gather(gain, -2, idx[..., None].expand(*idx.shape, 4))
            out["precond_gain_p0_top"] = sel[..., 0].mean()
            out["precond_gain_p3_top"] = sel[..., 3].mean()
            out["precond_clipped_top"] = (sel >= self.max_gain - 1e-6).float().mean()
        return out
