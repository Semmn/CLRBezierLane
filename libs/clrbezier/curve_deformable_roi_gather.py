import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- UnLaneDet layer shim -------------------------------------------------
# Replaces "from ...layers import Conv2d, get_norm" with a self-contained
# detectron2-style Conv2d(..., norm=...) so this module has no repo-internal
# dependency. Semantics are unchanged: conv -> norm (no activation).
class Conv2d(nn.Conv2d):
    def __init__(self, *args, norm=None, activation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def get_norm(norm, out_channels):
    if norm is None or norm == "":
        return None
    if callable(norm) and not isinstance(norm, str):
        return norm(out_channels)
    return {
        "BN": lambda c: nn.BatchNorm2d(c),
        "SyncBN": lambda c: nn.SyncBatchNorm(c),
        "GN": lambda c: nn.GroupNorm(32, c),
        "LN": lambda c: nn.GroupNorm(1, c),
    }[norm](out_channels)
# --------------------------------------------------------------------------

def _expand_level_values(
    value: Union[float, Sequence[float]],
    num_levels: int,
    name: str
) -> Tuple[float, ...]:
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(num_levels))

    value = tuple(float(v) for v in value)

    if len(value) != num_levels:
        raise ValueError(
            f"{name} must contain {num_levels} values, "
            f"got {len(value)}."
        )
    return value

def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    eps: float = 1.0e-6
) -> torch.Tensor:
    """
    Numerically safe masked softmax.

    If every element is invalid along 'dim', the returned probability is zero rather than NaN.
    """
    if logits.shape != mask.shape:
        raise ValueError(
            "logits and mask must have identical shapes, "
            f"got logits={tuple(logits.shape)}, "
            f"mask={tuple(mask.shape)}."
        )
    mask = mask.to(dtype=torch.bool)
    # -1e5 overflows to -inf in fp16 (AMP); an all-invalid row then yields NaN
    # from softmax. finfo.min / 4 stays finite in every supported dtype, so an
    # all-invalid row gives a uniform distribution that the mask below zeroes.
    mask_value = torch.finfo(logits.dtype).min / 4.0
    masked_logits = logits.masked_fill(~mask, mask_value)

    probabilities = F.softmax(masked_logits, dim=dim)
    probabilities = probabilities * mask.to(dtype=probabilities.dtype)

    denominator = probabilities.sum(dim=dim, keepdim=True)
    probabilities = torch.where(denominator > 0.0, probabilities / denominator.clamp(min=eps), torch.zeros_like(probabilities))

    return probabilities

class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25), align_corners=False):
        super().__init__()
        self.size = tuple(size)
        self.align_corners = bool(align_corners)

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=self.align_corners)

        return x.flatten(2)

class CurvePointQueryInteraction(nn.Module):
    """
    Query-to-query interaction at each curve sample position.

    Input:
        point_features:
            (B, K, S, D)
        geometry_features:
            (B, K, S, G)
        valid_mask:
            (B, K, S)
    
    Operation:
        For every curve sample index s, perform self-attention between the K lane queries:
            (B, K, S, D) -> (B*S, K, D) -> query-to-query attention -> (B, K, S, D)
    
    This preserves the spatial/curve dimension instead of first collapsing each lane into one vector.
    """
    def __init__(self, hidden_dim, geometry_dim=6, num_heads=4,
                 ffn_dim=128, dropout=0.0, use_ffn=True, residual_scale_init=1.0,
                 ffn_scale_init=1.0, zero_init_residual=True):
        super().__init__()

        hidden_dim = int(hidden_dim)
        geometry_dim = int(geometry_dim)
        num_heads = int(num_heads)
        ffn_dim = int(ffn_dim)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads, "
                f"got hidden_dim={hidden_dim}, "
                f"num_heads={num_heads}."
            )

        self.hidden_dim = hidden_dim
        self.geometry_dim = geometry_dim
        self.use_ffn = bool(use_ffn)

        self.content_norm = nn.LayerNorm(hidden_dim)
        self.geometry_norm = nn.LayerNorm(geometry_dim)

        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.query_key_norm = nn.LayerNorm(hidden_dim)

        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                                    dropout=float(dropout), batch_first=True)
        self.attention_dropout = nn.Dropout(float(dropout))

        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))

        if self.use_ffn:
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(ffn_dim, hidden_dim),
            )
            self.ffn_dropout = nn.Dropout(float(dropout))
            self.ffn_scale = nn.Parameter(torch.tensor(float(ffn_scale_init), dtype=torch.float32))
        else:
            self.ffn_norm = None
            self.ffn = None
            self.ffn_dropout = None
            self.register_parameter("ffn_scale", None)

        if zero_init_residual:
            nn.init.zeros_(self.self_attention.out_proj.weight)

            if self.self_attention.out_proj.bias is not None:
                nn.init.zeros_(self.self_attention.out_proj.bias)

            if self.use_ffn:
                nn.init.zeros_(self.ffn[-1].weight)
                nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, point_features, geometry_features, valid_mask):
        if point_features.ndim != 4:
            raise ValueError("point_features must have shape (B, K, S, D), "
                             f"got {tuple(point_features.shape)}.")
        if geometry_features.ndim != 4:
            raise ValueError("geometry_features must have shape (B, K, S, G), "
                             f"got {tuple(geometry_features.shape)}.")
        if valid_mask.ndim != 3:
            raise ValueError("valid_mask must have shape (B, K, S), "
                             f"got {tuple(valid_mask.shape)}.")
        if point_features.shape[:3] != geometry_features.shape[:3]:
            raise ValueError("point/geometry shape mismatch: "
                             f"point={tuple(point_features.shape)}, "
                             f"geometry={tuple(geometry_features.shape)}.")
        if point_features.shape[:3] != valid_mask.shape:
            raise ValueError(
                "Point/mask shape mismatch: "
                f"point={tuple(point_features.shape)}, "
                f"mask={tuple(valid_mask.shape)}."
            )

        batch_size, num_queries, num_samples, hidden_dim = point_features.shape
        content = self.content_norm(point_features)
        geometry_embedding = self.geometry_projection(self.geometry_norm(geometry_features))
        query_key_features = self.query_key_norm(content + geometry_embedding)

        # (B, K, S, D) -> (B, S, K, D) -> (B*S, K, D)
        query_key_features = query_key_features.permute(0, 2, 1, 3).reshape(batch_size * num_samples, num_queries, hidden_dim)
        value_features = content.permute(0, 2, 1, 3).reshape(batch_size * num_samples, num_queries, hidden_dim)

        # Mutihead Attention masks keys, not queries.
        key_padding_mask = (~valid_mask).permute(0, 2, 1).reshape(batch_size * num_samples, num_queries)

        # Prevent all-masked rows from producing NaNs.
        all_invalid = key_padding_mask.all(dim=1)

        safe_key_padding_mask = key_padding_mask.clone()

        if all_invalid.any():
            safe_key_padding_mask[all_invalid, 0] = False
        attention_output, _ = self.self_attention(query=query_key_features, key=query_key_features,
                                                  value=value_features, key_padding_mask=safe_key_padding_mask, need_weights=False)
        attention_output = attention_output.reshape(batch_size, num_samples, num_queries, hidden_dim).permute(0, 2, 1, 3).contiguous()

        # Invalid query positions must not reeive an update.
        attention_output = attention_output * valid_mask[..., None].to(dtype=attention_output.dtype)

        output = (point_features + self.residual_scale.to(dtype=point_features.dtype) * self.attention_dropout(attention_output))

        if self.use_ffn:
            ffn_output = self.ffn(self.ffn_norm(output))
            ffn_output = ffn_output * valid_mask[..., None].to(dtype=ffn_output.dtype)
            output = output + self.ffn_scale.to(dtype=output.dtype) * self.ffn_dropout(ffn_output)

        return output


class CurveAlignedDeformableSampler(nn.Module):
    """
    Curve-aligned deformable feature sampling.
    The module receives reference points from the current CLR lane representation.
    It therefore follows the stage-wise refined lane, rather than always using the original Bezier control points.

    For each reference point:
        1. Sample the feature at the current lane position.
        2. Combine the local feature, lane query, absolute coordinate, and curve orientation.
        3. Predict offsets in a tangent-normal coordinate frame.
        4. Sample nearby features.
        5. Predict attention weights over level/offset samples.
        6. Optionally perform query-to-query attention while preserving the curve sample dimension.
        7. Encode and pool the pointwise curve features.

    Inputs:
        features:
            List of feature maps. Each map has shape (B, C, H, W).
        feature_level_indices:
            Corresponding FPN/refinement-level indices.
        query:
            Lane-level query feature, shape (B, K, D).
        reference_points:
            Current lane reference points, shape (B, K, S, 2),
            normalized to [0, 1] in (x, y) order.
        reference_valid_mask:
            Optional mask with shape (B, K, S)

    Output:
        curve_context:
            Shape (B, K, D)
    """
    def __init__(self, in_channels, hidden_dim, max_feature_levels,
                 num_curve_samples=18, num_offsets=4, offset_mode="normal", max_normal_offset=2.0, max_tangent_offset=1.0,
                 align_corners=True, padding_mode="zeros", dropout=0.0, use_curve_encoder=True, 
                 use_point_query_interaction=False, query_interaction_num_heads=4, query_interaction_ffn_dim=128,
                 query_interaction_use_ffn=True, zero_init_output=True, shared_value_projection=False):
        super().__init__()

        in_channels = int(in_channels)
        hidden_dim = int(hidden_dim)
        max_feature_levels = int(max_feature_levels)
        num_curve_samples = int(num_curve_samples)
        num_offsets = int(num_offsets)

        if num_curve_samples < 2:
            raise ValueError("num_curve_samples must be at least 2.")
        if num_offsets < 2:
            raise ValueError("num_offsets must be at least 2 because the first "
                             "sample is reserved for the zero-offset center.")
        if offset_mode not in {"normal", "tangent_normal"}:
            raise ValueError("Offset_mode must be 'normal' or 'tangent_normal', "
                             f"got {offset_mode!r}.")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.max_feature_levels = max_feature_levels
        self.num_curve_samples = num_curve_samples
        self.num_offsets = num_offsets
        self.num_learned_offsets = num_offsets - 1

        self.offset_mode = offset_mode

        self.align_corners = bool(align_corners)
        self.padding_mode = str(padding_mode)

        self.max_normal_offset = _expand_level_values(max_normal_offset, max_feature_levels, "max_normal_offset")
        self.max_tangent_offset = _expand_level_values(max_tangent_offset, max_feature_levels, "max_tangent_offset")

        self.shared_value_projection = shared_value_projection
        if self.shared_value_projection:
            self.value_projection = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        else:
            self.value_projections = nn.ModuleList([
             nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0) for _ in range(max_feature_levels)
            ])

        # Build point-specific sampling queries.
        self.lane_query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.reference_feature_projection = nn.Linear(hidden_dim, hidden_dim)

        # x, y, tangent_x, tangent_y, normal_x, normal_y
        self.geometry_projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.point_query_norm = nn.LayerNorm(hidden_dim)
        offset_dimension = 1 if offset_mode == "normal" else 2

        self.offset_head = nn.Linear(hidden_dim, max_feature_levels * self.num_learned_offsets * offset_dimension)
        self.weight_head = nn.Linear(hidden_dim, max_feature_levels * num_offsets)

        # Begin exactly on the current curve.
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        # Initially assign equal weights to coincident samples.
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

        self.use_point_query_interaction = bool(use_point_query_interaction)

        if self.use_point_query_interaction:
            self.point_query_interaction = CurvePointQueryInteraction(hidden_dim=hidden_dim, geometry_dim=6, num_heads=query_interaction_num_heads,
                                                                      ffn_dim=query_interaction_ffn_dim, dropout=dropout,
                                                                      use_ffn=query_interaction_use_ffn, residual_scale_init=1.0, ffn_scale_init=1.0, zero_init_residual=True)

        else:
            self.point_query_interaction = None

        self.use_curve_encoder = bool(use_curve_encoder)

        if self.use_curve_encoder:
            self.curve_encoder_norm = nn.LayerNorm(hidden_dim)
            self.curve_depthwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
            self.curve_pointwise_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

            # Pointwise encoder starts as identity.
            nn.init.zeros_(self.curve_pointwise_conv.weight)
            nn.init.zeros_(self.curve_pointwise_conv.bias)
        else:
            self.curve_encoder_norm = None
            self.curve_depthwise_conv = None
            self.curve_pointwise_conv = None

        # Learn which curve sections are informative.
        self.point_pooling_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        # Uniform curve-point pooling at initialization.
        nn.init.zeros_(self.point_pooling_score[-1].weight)
        nn.init.zeros_(self.point_pooling_score[-1].bias)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(float(dropout))

        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _build_curve_frame(
        reference_points, height, width
    ):
        """
        Compute tangent and normal vectors in feature-map pixel space.

        Args:
            reference_points:
                (B, K, S, 2), normalized x/y.

        Returns:
            tangent:
                (B, K, S, 2), unit vector in pixel coordinates.
            normal:
                (B, K, S, 2), unit vector in pixel coordinates.
        """
        pixel_scale = reference_points.new_tensor([max(width-1, 1), max(height-1, 1)])

        points_pixel = reference_points * pixel_scale
        difference = torch.zeros_like(points_pixel)

        difference[:, :, 1:-1] = points_pixel[:, :, 2:] - points_pixel[:, :, :-2]
        difference[:, :, 0] = points_pixel[:, :, 1] - points_pixel[:, :, 0]
        difference[:, :, -1] = points_pixel[:, :, -1] - points_pixel[:, :, -2]

        tangent = F.normalize(difference, dim=-1, eps=1.0e-6)

        normal = torch.stack([
            -tangent[..., 1], tangent[..., 0],
        ], dim=-1)

        return tangent, normal

    def _sample_feature(
            self, feature, sample_points
    ):
        """
        Args:
            feature:
                (B, D, H, W)
            sample_points:
                (B, K, S, R, 2), normalized x/y.

        Returns:
            sampled:
                (B, K, S, R, D)
        """
        batch_size, channels, _, _ = feature.shape
        _, num_queries, num_samples, num_offsets, _ = sample_points.shape

        grid = sample_points * 2.0 - 1.0
        grid = grid.reshape(batch_size, num_queries * num_samples * num_offsets, 1, 2)
        grid = grid.to(device=feature.device, dtype=feature.dtype)

        sampled = F.grid_sample(feature, grid, mode='bilinear', padding_mode=self.padding_mode, align_corners=self.align_corners)
        sampled = sampled.squeeze(-1)
        sampled = sampled.reshape(batch_size, channels, num_queries, num_samples, num_offsets)

        sampled = sampled.permute(0, 2, 3, 4, 1).contiguous()
        return sampled

    def _encode_curve_points(self, point_features, valid_mask):
        if not self.use_curve_encoder:
            return point_features

        batch_size, num_queries, num_samples, hidden_dim = point_features.shape
        normalized = self.curve_encoder_norm(point_features)
        encoded = normalized.reshape(batch_size * num_queries, num_samples, hidden_dim).transpose(1, 2)

        encoded = self.curve_depthwise_conv(encoded)
        encoded = F.gelu(encoded)
        encoded = self.curve_pointwise_conv(encoded)

        encoded = encoded.transpose(1, 2).reshape(batch_size, num_queries, num_samples, hidden_dim)

        encoded = encoded * valid_mask[..., None].to(dtype=encoded.dtype)

        return point_features + encoded

    def forward(self, features: List[torch.Tensor], feature_level_indices: Sequence[int], query: torch.Tensor,
                reference_points: torch.Tensor, reference_valid_mask: Optional[torch.Tensor]=None, 
                apply_query_interaction: bool = True, return_metadata: bool = False):
        if query.ndim != 3:
            raise ValueError("query must have shape (B, K, D), "
                             f"got {tuple(query.shape)}.")

        if reference_points.ndim != 4:
            raise ValueError(
                "reference_points must have shape (B, K, S, 2), "
                f"got {tuple(reference_points.shape)}."
            )
        if reference_points.shape[-1] != 2:
            raise ValueError("reference_points must contain x/y coordinates.")

        if reference_points.shape[:2] != query.shape[:2]:
            raise ValueError("Query/reference shape mismatch: "
                             f"query={tuple(query.shape)}, "
                             f"reference={tuple(reference_points.shape)}.")
        if (reference_points.shape[2] != self.num_curve_samples):
            raise ValueError("Unexpected number of curve samples: "
                             f"expected {self.num_curve_samples}, "
                             f"got {reference_points.shape[2]}.")

        if len(features) != len(feature_level_indices):
            raise ValueError(
                "features and feature_level_indices must have identical lengths."
            )

        if len(features) == 0:
            raise ValueError("At least one deformable feature level is required.")

        level_indices = [int(level_index) for level_index in feature_level_indices]
        for level_index in level_indices:
            if not (0 <= level_index < self.max_feature_levels):
                raise ValueError("Feature-level index is outside the configured "
                                 f"range: {level_index}.")
        batch_size, num_queries, num_samples, _ = reference_points.shape
        reference_points = reference_points.to(device=query.device, dtype=query.dtype)
        finite_reference = torch.isfinite(reference_points).all(dim=-1)

        inside_reference = (reference_points[..., 0] >= 0.0) & (reference_points[..., 0] <= 1.0) & (reference_points[..., 1] >= 0.0) & (reference_points[..., 1] <= 1.0)
        automatically_valid = finite_reference & inside_reference

        if reference_valid_mask is None:
            reference_valid_mask = automatically_valid
        else:
            if (reference_valid_mask.shape != reference_points.shape[:3]):
                raise ValueError("reference_valid_mask must have shape "
                                 f"{reference_points.shape[:3]}, got {tuple(reference_valid_mask.shape)}.")
            reference_valid_mask = (reference_valid_mask.to(device=query.device, dtype=torch.bool) & automatically_valid)

        safe_reference_points = torch.nan_to_num(reference_points, nan=0.0, posinf=1.0, neginf=0.0)
        safe_reference_points = torch.where(reference_valid_mask[..., None], safe_reference_points.clamp(min=0.0, max=1.0),
                                            torch.zeros_like(safe_reference_points))

        # -----------------------------------------------------------------------
        # Project selected feature levels and sample the zero-offset
        # feature at the current lane reference points.
        # -----------------------------------------------------------------------
        projected_features = []
        center_features = []
        center_points = safe_reference_points.unsqueeze(3)

        for feature, level_index in zip(features, level_indices):
            if feature.ndim != 4:
                raise ValueError("Each feature must have shape (B, C, H, W), "
                                 f"got {tuple(feature.shape)}.")

            if self.shared_value_projection:
                projected_feature = self.value_projection(feature)
            else:
                projected_feature = self.value_projections[level_index](feature)
            projected_features.append(projected_feature)

            center_sample = self._sample_feature(projected_feature, center_points).squeeze(3)
            center_features.append(center_sample)

        # Current-only mode has one term. Multi-level mode uses their mean only to predict
        # offsets/weights; final aggregation remains learned.
        center_feature = torch.stack(center_features, dim=0).mean(dim=0)

        # Use the current/highest supplied feature level as the canonical coordinate frame for point-query geometry.
        canonical_height = features[-1].shape[-2]
        canonical_width = features[-1].shape[-1]

        canonical_tangent, canonical_normal = self._build_curve_frame(safe_reference_points, canonical_height, canonical_width)
        point_geometry = torch.cat([safe_reference_points, canonical_tangent, canonical_normal], dim=-1)

        point_query = self.lane_query_projection(query)[:, :, None, :] + self.reference_feature_projection(center_feature) + self.geometry_projection(point_geometry)
        point_query = self.point_query_norm(point_query)

        offset_dimension = (1 if self.offset_mode == "normal" else 2)

        raw_offsets = self.offset_head(point_query).reshape(batch_size, num_queries, num_samples, self.max_feature_levels,
                                                            self.num_learned_offsets, offset_dimension)

        weight_logits = self.weight_head(point_query).reshape(batch_size, num_queries, num_samples, self.max_feature_levels,
                                                              self.num_offsets)

        level_index_tensor = torch.as_tensor(level_indices, device=query.device, dtype=torch.long)

        raw_offsets = torch.index_select(raw_offsets, dim=3, index=level_index_tensor)
        weight_logits = torch.index_select(weight_logits, dim=3, index=level_index_tensor)

        sampled_features_by_level = []
        sample_validity_by_level = []
        sample_points_by_level = []
        displacement_by_level = []

        # ------------------------------------------------------------------------------
        # Build curve-relative samples independently for each feature map.
        # Offset magnitudes are expressed in that feature map's pixels.
        # ------------------------------------------------------------------------------
        for selected_position, (feature, projected_feature, level_index) in enumerate(zip(features, projected_features, level_indices)):
            height = feature.shape[-2]
            width = feature.shape[-1]

            tangent, normal = self._build_curve_frame(safe_reference_points, height, width)
            level_raw_offsets = raw_offsets[..., selected_position, :, :]
            normal_offset = torch.tanh(level_raw_offsets[..., 0]) * float(self.max_normal_offset[level_index])

            if self.offset_mode == "tangent_normal":
                tangent_offset = torch.tanh(level_raw_offsets[..., 1]) * float(self.max_tangent_offset[level_index])
            else:
                tangent_offset = torch.zeros_like(normal_offset)
            displacement_pixel = (normal[:, :, :, None, :] * normal_offset[..., None] + tangent[:, :, :, None, :] * tangent_offset[..., None])

            # Reserve offset index 0 for the exact current lane position.
            zero_displacement = torch.zeros((batch_size, num_queries, num_samples, 1, 2), device=query.device, dtype=query.dtype)
            displacement_pixel = torch.cat([zero_displacement, displacement_pixel], dim=3)
            pixel_scale = query.new_tensor([max(width - 1, 1), max(height - 1, 1)])

            displacement_normalized = displacement_pixel / pixel_scale
            sample_points = safe_reference_points[:, :, :, None, :] + displacement_normalized
            inside_samples = torch.isfinite(sample_points).all(dim=-1) & (sample_points[..., 0] >= 0.0) \
            & (sample_points[..., 0] <= 1.0) & (sample_points[..., 1] >= 0.0) & (sample_points[..., 1] <= 1.0)

            sample_validity = (reference_valid_mask[:, :, :, None] & inside_samples)
            sampled_feature = self._sample_feature(projected_feature, sample_points)

            sampled_features_by_level.append(sampled_feature)
            sample_validity_by_level.append(sample_validity)
            sample_points_by_level.append(sample_points)
            displacement_by_level.append(displacement_normalized)

        # (B, K, S, L, R, D)
        sampled_features = torch.stack(sampled_features_by_level, dim=3)
        # (B, K, S, L, R)
        sample_validity = torch.stack(sample_validity_by_level, dim=3)

        num_selected_levels = len(features)

        flattened_logits = weight_logits.reshape(batch_size, num_queries, num_samples, num_selected_levels * self.num_offsets)
        flattened_validity = sample_validity.reshape(batch_size, num_queries, num_samples, num_selected_levels * self.num_offsets)

        attention_weights = masked_softmax(flattened_logits, flattened_validity, dim=-1)
        attention_weights = attention_weights.reshape(batch_size, num_queries, num_samples, num_selected_levels, self.num_offsets)
        point_features = (sampled_features * attention_weights[..., None]).sum(dim=(3, 4))

        point_features = point_features * reference_valid_mask[..., None].to(dtype=point_features.dtype)

        # --------------------------------------------------------------------
        # Query-to-query interaction with curve sample dimension intact.
        # --------------------------------------------------------------------
        if (self.use_point_query_interaction and bool(apply_query_interaction)):
            point_features = self.point_query_interaction(point_features=point_features, 
                                                          geometry_features=point_geometry, valid_mask=reference_valid_mask)

        # ----------------------------------------------------------------------
        # Model continuity and local changes along each curve.
        # ----------------------------------------------------------------------
        point_features = self._encode_curve_points(point_features, reference_valid_mask)

        # -----------------------------------------------------------------------
        # Learned pooling across curve points.
        # -----------------------------------------------------------------------
        point_scores = self.point_pooling_score(point_features).squeeze(-1)
        point_weights = masked_softmax(point_scores, reference_valid_mask, dim=2)
        curve_context = (point_features * point_weights[..., None]).sum(dim=2)

        curve_context = self.output_projection(curve_context)
        curve_context = self.output_dropout(curve_context)

        if not return_metadata:
            return curve_context

        metadata = {
            "point_weights": point_weights,
            "sampling_attention_weights": attention_weights,
            "reference_valid_mask": reference_valid_mask,
            "sample_points_by_level": sample_points_by_level,
            "normalized_displacements_by_level": displacement_by_level,
            "feature_level_indices": level_indices,
        }

        return curve_context, metadata


class CurveAlignedDeformableROIGather(nn.Module):
    """
    Enhanced ROIGather with:
        1. Cross-layer prior-aligned feature aggregation.
        2. Dense query-to-feature global interaction.
        3. Curve-aligned deformable local sampling.
        4. Optional point-preserving query-to-query interaction.

    The returned feature is shared by classification and regression.
    """
    def __init__(
            self,
            in_channels,
            num_priors,
            sample_points,
            fc_hidden_dim,
            refine_layers,
            mid_channels=48,
            use_conv_activation=True,
            norm_type="BN",

            global_pool_size=(10, 25),
            global_dropout=0.1,

            use_deformable_curve_sampling=True,
            deformable_stages=None,

            deform_num_curve_samples = 18,
            deform_num_offsets = 4,
            deform_offset_mode="normal",
            deform_max_normal_offset=2.0,
            deform_max_tangent_offset=1.0,
            deform_dropout=0.0,
            shared_value_projection=False,

            use_curve_point_query_interaction=False,
            curve_query_interaction_stages=None,
            curve_query_interaction_num_heads = 4,
            curve_query_interaction_ffn_dim = 128,
            curve_query_interaction_use_ffn = True,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.num_priors = int(num_priors)
        self.sample_points = int(sample_points)
        self.fc_hidden_dim = int(fc_hidden_dim)
        self.refine_layers = int(refine_layers)

        self.global_dropout = float(global_dropout)

        use_bias = norm_type == ""

        # --------------------------------------------------------------------------------
        # original dense global interaction branch
        # --------------------------------------------------------------------------------
        self.f_key = Conv2d(in_channels=self.in_channels,
                            out_channels=self.fc_hidden_dim,
                            kernel_size=1, stride=1, padding=0, bias=use_bias,
                            norm=get_norm(norm=norm_type, out_channels=self.fc_hidden_dim))
        self.f_query = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_priors,
                out_channels=self.num_priors,
                kernel_size=1, stride=1, padding=0, groups=self.num_priors
            ), nn.ReLU(inplace=True)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.fc_hidden_dim,
                                 kernel_size=1, stride=1, padding=0)

        self.global_output_projection = nn.Conv1d(
            in_channels=self.num_priors, out_channels=self.num_priors, kernel_size=1, 
            stride=1, padding=0, groups=self.num_priors,
        )

        self.resize = FeatureResize(size=global_pool_size, align_corners=False)

        nn.init.zeros_(self.global_output_projection.weight)
        nn.init.zeros_(self.global_output_projection.bias)

        # -------------------------------------------------------------------------------------
        # Cross-layer prior-feature aggregation
        # -------------------------------------------------------------------------------------
        # CLRNet/CLRerNet ROIGather uses ConvModule = conv -> norm -> ReLU and
        # reduces every stage's prior features to mid_channels before the
        # concatenation that catconv consumes (mid_channels * (stage + 1)).
        mid_channels = int(mid_channels)
        self.mid_channels = mid_channels
        make_act = (lambda: nn.ReLU(inplace=True)) if use_conv_activation else (lambda: None)
        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()

        for stage in range(self.refine_layers):
            self.convs.append(
                Conv2d(self.in_channels, mid_channels, kernel_size=(9, 1),
                       padding=(4, 0), bias=False,
                       norm=get_norm(norm=norm_type, out_channels=mid_channels),
                       activation=make_act())
            )

            self.catconv.append(
                Conv2d(mid_channels * (stage + 1), self.in_channels, kernel_size=(9, 1),
                       padding=(4, 0), bias=False,
                       norm=get_norm(norm=norm_type, out_channels=self.in_channels),
                       activation=make_act())
            )

        self.fc = nn.Linear(self.sample_points * self.in_channels, self.fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(self.fc_hidden_dim)

        # -----------------------------------------------------------------------------------
        # Curve-aligned deformable branch
        # -----------------------------------------------------------------------------------
        self.use_deformable_curve_sampling = bool(use_deformable_curve_sampling)
        if deformable_stages is None:
            deformable_stages = tuple(range(self.refine_layers))

        self.deformable_stages = {int(stage) for stage in deformable_stages}

        if curve_query_interaction_stages is None:
            curve_query_interaction_stages = (tuple(range(self.refine_layers)) if use_curve_point_query_interaction else tuple())

        self.curve_query_interaction_stages = {
            int(stage) for stage in curve_query_interaction_stages
        }

        if self.use_deformable_curve_sampling:
            self.curve_deformable_sampler = CurveAlignedDeformableSampler(
                in_channels=self.in_channels,
                hidden_dim=self.fc_hidden_dim,
                max_feature_levels=self.refine_layers,
                num_curve_samples=deform_num_curve_samples,
                num_offsets=deform_num_offsets,
                offset_mode=deform_offset_mode,
                max_normal_offset=deform_max_normal_offset,
                max_tangent_offset=deform_max_tangent_offset,
                align_corners=True,
                padding_mode='zeros',
                dropout=deform_dropout,
                shared_value_projection=shared_value_projection,
                use_curve_encoder=True,
                use_point_query_interaction=use_curve_point_query_interaction,
                query_interaction_num_heads=curve_query_interaction_num_heads,
                query_interaction_ffn_dim = curve_query_interaction_ffn_dim,
                query_interaction_use_ffn = curve_query_interaction_use_ffn,
                zero_init_output=True,
            )
        else:
            self.curve_deformable_sampler = None

    def roi_fea(self, roi_features, layer_index):
        if len(roi_features) != layer_index + 1:
            raise ValueError("Expected one prior feature tensor for every stage up "
                             f"to layer_index={layer_index}, got "
                             f"{len(roi_features)} tensors.")
        transformed_features = []

        for stage_index, feature in enumerate(roi_features):
            transformed_feature = self.convs[stage_index](feature)
            transformed_features.append(transformed_feature)

        concatenated_feature = torch.cat(transformed_features, dim=1)
        concatenated_feature = self.catconv[layer_index](concatenated_feature)

        return concatenated_feature

    def _global_feature_interaction(self, roi, feature_map):
        """
        Original CLRNet-style dense query-to-feature interaction.

        Args:
            roi:
                (B, K, D)
            feature_map:
                (B, C, H, W)
        Returns:
            global_context:
                (B, K, D)
        """
        query = self.f_query(roi)
        key = self.f_key(feature_map)
        value = self.f_value(feature_map)

        key = self.resize(key)
        value = self.resize(value).permute(0, 2, 1)

        similarity = torch.matmul(query, key)
        similarity = (self.fc_hidden_dim ** -0.5) * similarity
        similarity = F.softmax(similarity, dim=-1)

        global_context = torch.matmul(similarity, value)
        global_context = self.global_output_projection(global_context)

        return global_context

    def forward(self, roi_features, x, layer_index, 
                reference_points=None,
                reference_valid_mask=None,

                deform_features=None,
                deform_feature_level_indices=None,

                return_deformable_metadata=False,
    ):
        """
        Args:
            roi_features:
                List of prior-aligned features from stages 0..layer_index.
                Each element:
                    (B*K, C, P, 1)
            x:
                Current refinement-stage feature map
                    (B, C, H, W)
            layer_index:
                Current refinement stage.

            reference_points:
                Current CLR lane points:
                    (B, K, S, 2)
            reference_valid_mask:
                Optional:
                    (B, K, S)
            deform_features:
                Feature maps used by deformable sampling.
                Efficient/default mode:
                    [x]
                Cumulative multi-level mode:
                    feature_maps[:layer_index + 1]
            
            deform_feature_level_indices:
                FPN/refinement indices corresponding to deform_features.
            return_deformable_metadata:
                Return offsets and attention weights for visualization.

        Returns:
            roi:
                shared lane representation:
                    (B, K, D)
        """
        layer_index = int(layer_index)

        if not(0 <= layer_index < self.refine_layers):
            raise ValueError(f"Invalid layer_index={layer_index}.")

        batch_size = x.shape[0]

        # ---------------------------------------------------------------------
        # Cross-layer prior-aligned feature aggregation
        # ---------------------------------------------------------------------
        roi = self.roi_fea(roi_features, layer_index)
        roi = roi.contiguous().reshape(batch_size * self.num_priors, -1)

        expected_flat_dimension = self.sample_points * self.in_channels

        if roi.shape[-1] != expected_flat_dimension:
            raise RuntimeError("Unexpected flattened ROI dimension: "
                               f"expected {expected_flat_dimension}, "
                               f"got {roi.shape[-1]}.")
        roi = self.fc(roi)
        roi = self.fc_norm(roi)
        roi = F.relu(roi, inplace=True)
        roi = roi.reshape(batch_size, self.num_priors, self.fc_hidden_dim)

        # -----------------------------------------------------------------------
        # Dense global query-to-feature interaction
        # -----------------------------------------------------------------------
        global_context = self._global_feature_interaction(roi=roi, feature_map=x)
        roi = (roi + F.dropout(global_context, p=self.global_dropout, training=self.training))

        deformable_metadata = None
        deformable_enabled = (self.use_deformable_curve_sampling and layer_index in self.deformable_stages)

        if deformable_enabled:
            if reference_points is None:
                raise ValueError(
                    "reference_points must be provided when "
                    "curve-aligned deformable sampling is enabled."
                )

            # Efficient default: only current stage feature.
            if deform_features is None:
                deform_features = [x]

            if deform_feature_level_indices is None:
                deform_feature_level_indices = [layer_index]
            apply_query_interaction = (layer_index in self.curve_query_interaction_stages)

            deformable_output = self.curve_deformable_sampler(
                features=deform_features, feature_level_indices=deform_feature_level_indices,
                query=roi, reference_points=reference_points, reference_valid_mask=reference_valid_mask,
                apply_query_interaction=apply_query_interaction, return_metadata=return_deformable_metadata
            )

            if return_deformable_metadata:
                deformable_context, deformable_metadata = deformable_output
            else:
                deformable_context = deformable_output

            # Shared local-global representation for cls and reg.
            roi = roi + deformable_context

        if return_deformable_metadata:
            metadata = {"deformable" : deformable_metadata}
            return roi, metadata

        return roi


def build_clr_curve_reference_points(prior_xs, prior_ys, num_curve_samples):
    """
    Args:
        prior_xs:
            (B, K, P), ordered consistently with pool_prior_features.
        prior_ys:
            (P, ), (B, P), or (B, K, P).
        num_curve_samples:
            Number of points used by deformable RoIGather.

    Returns:
        reference_points:
            (B, K, S, 2), normalized x/y.
        reference_valid_mask:
            (B, K, S).
    """
    if prior_xs.ndim != 3:
        raise ValueError(
            "prior_xs must have shape (B, K, P), "
            f"got {tuple(prior_xs.shape)}."
        )

    if not torch.is_floating_point(prior_xs):
        raise TypeError("prior_xs must be floating-point tensor, "
                        f"got dtype={prior_xs.dtype}.")

    # Double precision is unncessary for normalized lane geometry and can conflict with float32 network layers.
    if prior_xs.dtype == torch.float64:
        prior_xs = prior_xs.float()
    prior_ys = prior_ys.to(device=prior_xs.device, dtype=prior_xs.dtype)
    
    batch_size, num_priors, num_points = prior_xs.shape

    if prior_ys.ndim == 1:
        if prior_ys.shape[0] != num_points:
            raise ValueError("prior_ys length does not match prior_xs.")

        prior_ys = prior_ys.reshape(1, 1, num_points).expand(batch_size, num_priors, num_points)
    elif prior_ys.ndim == 2:
        if prior_ys.shape != (batch_size, num_points):
            raise ValueError("Two-dimensional prior_ys must have shape (B, P).")

        prior_ys = prior_ys[:, None, :].expand(batch_size, num_priors, num_points)
    elif prior_ys.ndim == 3:
        if prior_ys.shape != prior_xs.shape:
            raise ValueError("Three-dimensional prior_ys must have the same shape as prior_xs.")

    else:
        raise ValueError("prior_ys must have 1, 2, or 3 dimensions.")

    # Selecting existing CLR locations avoids interpolating invalid or out-of-image x-coordinate sentinels.
    sample_indices = torch.linspace(0, num_points - 1, steps=num_curve_samples, device=prior_xs.device).round().long()

    sampled_x = torch.index_select(prior_xs, dim=2, index=sample_indices)
    sampled_y = torch.index_select(prior_ys, dim=2, index=sample_indices)

    reference_points = torch.stack(
        [
            sampled_x,
            sampled_y,
        ],
        dim=-1
    )

    reference_valid_mask = torch.isfinite(reference_points).all(dim=-1) & (reference_points[..., 0] >= 0.0) \
    & (reference_points[..., 0] <= 1.0) & (reference_points[..., 1] >= 0.0) & (reference_points[..., 1] <= 1.0)
    reference_points = torch.nan_to_num(reference_points, nan=0.0, posinf=1.0, neginf=0.0)

    return (reference_points, reference_valid_mask)

