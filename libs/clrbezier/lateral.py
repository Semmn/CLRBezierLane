"""Lateral evidence pooling.

CLRNet-family heads pool features along the predicted curve only, so a query's
feature is collected at its own (possibly wrong) location. Nothing in it
indicates that the image evidence is stronger a few pixels to one side, which is
exactly what a confidence score has to know to correlate with metric IoU. This
is the most likely reason a quality/IoU regression branch improves early and
then saturates.

This module samples the same rows at several lateral offsets around the
reference curve:

    x - 2d, x - d, x, x + d, x + 2d      (normalized image-x offsets)

and encodes the resulting left/right evidence profile into a per-query vector.
A centred lane gives a symmetric profile, an offset lane an asymmetric one, so
the displacement becomes observable. The output is added to the classification
features (and optionally the regression features) through a zero-initialized
gate, so the model starts identical to the baseline.

Offsets are given in normalized image-x units. The CULane metric half-width is
7.5/800 at the network resolution, so offsets around +-7.5/800 and +-15/800
straddle the band the metric scores.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralEvidence(nn.Module):
    """Args:
        in_channels: feature-map channels (prior_feat_channels).
        dim: output width (fc_hidden_dim).
        sample_points: rows sampled per query (same as ROIGather).
        offsets: normalized lateral offsets; 0.0 is included automatically if
            absent, so the centre strip is always available for comparison.
        mid_channels: channels of the encoder conv.
        gate_init: LayerScale init; 0.0 keeps the baseline at iteration 0.
        apply_to: "cls" (default), "reg", or "both".
    """

    def __init__(self, in_channels=64, dim=64, sample_points=36,
                 offsets=(-15.0 / 800, -7.5 / 800, 7.5 / 800, 15.0 / 800),
                 mid_channels=32, gate_init=0.0, apply_to="cls"):
        super().__init__()
        if apply_to not in ("cls", "reg", "both"):
            raise ValueError(f"Unknown apply_to {apply_to!r}")
        offs = sorted(set(float(o) for o in offsets) | {0.0})
        self.register_buffer("offsets", torch.tensor(offs, dtype=torch.float32), persistent=False)
        self.num_offsets = len(offs)
        self.apply_to = apply_to
        self.sample_points = int(sample_points)

        # (rows x offsets) -> profile encoding. The conv sees the whole lateral
        # profile at once, so it can express "evidence is stronger on one side".
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(9, 1), padding=(4, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, self.num_offsets), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(mid_channels * self.sample_points, dim)
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.empty(dim))
        self.gate_init = float(gate_init)
        self.zero_init()

    def zero_init(self):
        nn.init.constant_(self.gamma, self.gate_init)

    def forward(self, feature_map, prior_xs, prior_feat_ys):
        """Args:
            feature_map: [B, C, H, W] current stage features.
            prior_xs: [B, K, S] reference x at the sampling rows, in the same
                order ROIGather uses (already flipped), normalized to [0, 1].
            prior_feat_ys: [S] sampling rows, normalized.
        Returns:
            [B, K, dim] evidence embedding.
        """
        batch, num_q, num_points = prior_xs.shape
        xs = prior_xs[..., None] + self.offsets.view(1, 1, 1, -1)  # [B, K, S, O]
        xs = xs.clamp(0.0, 1.0)
        ys = prior_feat_ys.view(1, 1, -1, 1).expand(batch, num_q, num_points, self.num_offsets)
        grid = torch.stack((xs * 2.0 - 1.0, ys.to(xs.dtype) * 2.0 - 1.0), dim=-1)
        grid = grid.reshape(batch, num_q * num_points, self.num_offsets, 2)

        feat = F.grid_sample(feature_map, grid, align_corners=True)  # [B, C, K*S, O]
        channels = feat.shape[1]
        feat = feat.view(batch, channels, num_q, num_points, self.num_offsets)
        feat = feat.permute(0, 2, 1, 3, 4).reshape(batch * num_q, channels,
                                                   num_points, self.num_offsets)
        out = self.encoder(feat).reshape(batch * num_q, -1)
        out = self.norm(self.fc(out)).view(batch, num_q, -1)
        return out

    def fuse(self, features, evidence):
        """Zero-gated residual into the tower input."""
        return features + self.gamma * evidence
