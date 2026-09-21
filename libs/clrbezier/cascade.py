"""Cascade refinement utilities for CLRBezierHead.

1. ReferenceReprojector
   The next stage pools features along the Bezier reference, but a stage's
   prediction is reference + 72-row dense residual, and the residual is never
   propagated. Re-projection fits global cubic control points to the stage's
   full prediction (same ridge-to-affine objective used for GT control points)
   and hands that curve to the next stage, so the next stage samples along what
   was actually predicted.

   Cost: masked least squares with a fixed, precomputed Bernstein basis
   (one [R] x [R, 16] product and one 4x4 solve per lane), run under no_grad
   because the next-stage reference is detached anyway.

   Stability: blend lambda ramped from 0 over a warmup, a per-control-point
   trust region in pixels, and a fallback to the Bezier reference when the
   predicted lane has too few visible rows.

2. QualityGate
   Stage-increasing positive quality (Cascade R-CNN). After assignment, pairs
   whose LaneIoU is below the stage threshold are dropped from the positives.
   The IoU is the narrow LaneIoU (half-width 7.5/800 at 800 px), which
   approximates the CULane metric, so a threshold of 0.5 means "already a
   true positive by the metric". keep_best keeps each GT's best pair so no GT
   loses supervision while predictions are poor.
"""
import torch
import torch.nn as nn

from .geometry import CP_FRACTIONS, bernstein_basis, eval_global_cubic


# =============================================================================
# Re-projection
# =============================================================================
class ReferenceReprojector(nn.Module):
    """Fit fixed-global-y cubic CPs to predicted rows (batched, no grad)."""

    def __init__(self, prior_ys, img_w, n_strips, ridge=1e-2, margin=0.5,
                 min_rows=4, blend=1.0, warmup_iters=0, max_shift_px=None):
        super().__init__()
        ys = prior_ys.detach().float().clone()
        basis = bernstein_basis(ys)                               # [R, 4]
        self.register_buffer("ys", ys, persistent=False)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer(
            "basis_outer", (basis[:, :, None] * basis[:, None, :]).reshape(-1, 16),
            persistent=False)                                     # [R, 16]
        self.register_buffer("cp_y", torch.tensor(CP_FRACTIONS), persistent=False)
        self.img_w = int(img_w)
        self.n_strips = int(n_strips)
        self.ridge = max(float(ridge), 1e-8)
        self.margin = float(margin)
        self.min_rows = int(min_rows)
        self.blend = float(blend)
        self.warmup_iters = int(warmup_iters)
        self.max_shift = None if max_shift_px is None else float(max_shift_px) / float(img_w - 1)

    def blend_factor(self, step, training):
        if not training or self.warmup_iters <= 0:
            return self.blend
        return self.blend * min(1.0, float(step) / float(self.warmup_iters))

    @torch.no_grad()
    def fit(self, xs, mask):
        """xs: [N, R] normalized x; mask: [N, R] bool -> (cp [N, 4], ok [N])."""
        w = mask.to(xs.dtype)
        x = torch.where(mask, xs, torch.zeros_like(xs))
        count = w.sum(-1)
        btb = (w @ self.basis_outer.to(xs.dtype)).view(-1, 4, 4)
        btx = (w * x) @ self.basis.to(xs.dtype)

        ys = self.ys.to(xs.dtype)
        sy = w @ ys
        syy = w @ (ys * ys)
        sx = (w * x).sum(-1)
        syx = (w * x) @ ys
        safe_n = count.clamp_min(1.0)
        det = safe_n * syy - sy * sy
        det_ok = det.abs() > 1e-10
        safe_det = torch.where(det_ok, det, torch.ones_like(det))
        a = torch.where(det_ok, (sx * syy - sy * syx) / safe_det, sx / safe_n)
        b = torch.where(det_ok, (safe_n * syx - sy * sx) / safe_det, torch.zeros_like(det))
        affine = a[:, None] + b[:, None] * self.cp_y.to(xs.dtype)

        eye = torch.eye(4, device=xs.device, dtype=xs.dtype)
        cp = torch.linalg.solve(btb + self.ridge * eye, (btx + self.ridge * affine)[..., None])[..., 0]
        ok = (count >= float(self.min_rows)) & torch.isfinite(cp).all(-1)
        cp = torch.nan_to_num(cp, nan=0.5).clamp(-self.margin, 1.0 + self.margin)
        return cp, ok

    @torch.no_grad()
    def project(self, pred, bezier_cp, step=0, training=True):
        """Next-stage control points.

        Args:
            pred: [B, Q, 6 + R] stage prediction (official layout).
            bezier_cp: [B, Q, 4] Bezier-updated control points of this stage.
        Returns:
            (cp [B, Q, 4], stats dict)
        """
        lam = self.blend_factor(step, training)
        if lam <= 0.0:
            return bezier_cp, dict(ok_frac=0.0, shift_px=0.0, blend=lam)
        batch, num_q, _ = pred.shape
        xs = pred[..., 6:].reshape(batch * num_q, -1)
        rows = xs.shape[-1]
        n = float(self.n_strips)
        start = torch.round((1.0 - pred[..., 2]) * n).clamp(0, n).reshape(-1, 1)
        length = torch.round(pred[..., 5] * n).clamp(min=0).reshape(-1, 1)
        idx = torch.arange(rows, device=xs.device, dtype=xs.dtype).view(1, -1)
        mask = (idx >= start) & (idx < start + length) & (xs >= 0.0) & (xs < 1.0)
        refit, ok = self.fit(xs.float(), mask)
        refit = refit.view(batch, num_q, 4)
        ok = ok.view(batch, num_q, 1)

        target = bezier_cp + lam * (refit - bezier_cp)
        if self.max_shift is not None:
            target = bezier_cp + (target - bezier_cp).clamp(-self.max_shift, self.max_shift)
        out = torch.where(ok, target, bezier_cp).clamp(-self.margin, 1.0 + self.margin)
        shift = ((out - bezier_cp).abs().mean(-1) * float(self.img_w - 1))
        stats = dict(ok_frac=float(ok.float().mean()),
                     shift_px=float(shift[ok[..., 0]].mean()) if bool(ok.any()) else 0.0,
                     blend=lam)
        return out, stats


def sampling_xs(cp_x, prior_ys, sample_x_indices):
    """Reference x at the RoI sampling rows, clamped as in brr_reference."""
    ys = prior_ys[sample_x_indices].to(cp_x.dtype)
    xs = eval_global_cubic(cp_x, ys)
    return torch.nan_to_num(xs, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


# =============================================================================
# Stage-wise quality gate
# =============================================================================
class QualityGate:
    """Per-stage minimum LaneIoU for positives.

    Args:
        min_iou: list indexed by stage (missing stages -> 0).
        keep_best: always keep each GT's best pair.
        mode: "cls_and_reg" (Cascade R-CNN: gated pairs are negatives and get
            no regression) or "cls_only" (gated pairs still get regression).
        gate_on: "output" (IoU of the stage prediction, matching how the head
            assigns) or "input" (IoU of the reference the stage sampled from,
            the Cascade R-CNN definition; best combined with re-projection).
        warmup_iters: thresholds ramp linearly from 0.
    """

    def __init__(self, min_iou=(0.0, 0.3, 0.5), keep_best=True, mode="cls_and_reg",
                 gate_on="output", warmup_iters=0):
        if mode not in ("cls_and_reg", "cls_only"):
            raise ValueError(f"Unknown gate mode {mode!r}")
        if gate_on not in ("output", "input"):
            raise ValueError(f"Unknown gate_on {gate_on!r}")
        self.min_iou = [float(v) for v in min_iou]
        self.keep_best = bool(keep_best)
        self.mode = mode
        self.gate_on = gate_on
        self.warmup_iters = int(warmup_iters)

    def threshold(self, stage, step, training=True):
        thr = self.min_iou[stage] if stage < len(self.min_iou) else 0.0
        if training and self.warmup_iters > 0:
            thr *= min(1.0, float(step) / float(self.warmup_iters))
        return thr

    @torch.no_grad()
    def keep_mask(self, iou, cols, num_gt, thr):
        """iou, cols: [P] -> bool [P]."""
        keep = iou >= thr
        if self.keep_best and iou.numel() > 0:
            best = torch.full((num_gt,), -1.0, device=iou.device, dtype=iou.dtype)
            best = best.scatter_reduce(0, cols, iou, reduce="amax", include_self=True)
            is_best = iou >= best[cols]
            # one best per GT (first occurrence on ties)
            first = torch.zeros_like(is_best)
            seen = torch.zeros(num_gt, dtype=torch.bool, device=iou.device)
            order = torch.nonzero(is_best, as_tuple=False).squeeze(1)
            for i in order.tolist():
                g = int(cols[i])
                if not bool(seen[g]):
                    first[i] = True
                    seen[g] = True
            keep = keep | first
        return keep
