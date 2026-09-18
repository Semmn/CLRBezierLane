"""CLRerNet head building blocks (ports of the UnLaneDet V11 versions, which
were aligned with the official CLRerNet ROIGather)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, kernel_size, padding):
    """ConvModule(conv -> BN -> ReLU) equivalent, bias disabled."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super().__init__()
        self.size = (int(size[0]), int(size[1]))

    def forward(self, x):
        return F.interpolate(x, self.size, mode="nearest").flatten(2)


class AnchorVecFeatureMapAttention(nn.Module):
    def __init__(self, n_query, dim, resize_size=(10, 25)):
        super().__init__()
        self.n_query = int(n_query)
        self.dim = int(dim)
        self.resize = FeatureResize(resize_size)
        self.f_key = conv_bn_relu(dim, dim, 1, 0)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.n_query, self.n_query, 1, groups=self.n_query), nn.ReLU(inplace=True)
        )
        self.f_value = nn.Conv2d(dim, dim, 1)
        self.W = nn.Conv1d(self.n_query, self.n_query, 1, groups=self.n_query)
        nn.init.constant_(self.W.weight, 0.0)
        nn.init.constant_(self.W.bias, 0.0)

    def forward(self, roi, fmap):
        query = self.f_query(roi)
        key = self.resize(self.f_key(fmap))
        value = self.resize(self.f_value(fmap)).permute(0, 2, 1).contiguous()
        sim = F.softmax(torch.matmul(query, key) * (float(self.dim) ** -0.5), dim=-1)
        return self.W(torch.matmul(sim, value))


class ROIGather(nn.Module):
    """Cross-stage lane-feature aggregation + feature-map attention."""

    def __init__(self, in_channels, num_priors, sample_points, fc_hidden_dim,
                 refine_layers, mid_channels=48):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_priors = int(num_priors)
        self.sample_points = int(sample_points)
        self.fc_hidden_dim = int(fc_hidden_dim)
        self.attention = AnchorVecFeatureMapAttention(num_priors, in_channels)
        self.convs = nn.ModuleList(
            [conv_bn_relu(in_channels, mid_channels, (9, 1), (4, 0)) for _ in range(refine_layers)]
        )
        self.catconv = nn.ModuleList(
            [conv_bn_relu(mid_channels * (i + 1), in_channels, (9, 1), (4, 0))
             for i in range(refine_layers)]
        )
        self.fc = nn.Linear(sample_points * in_channels, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def forward(self, roi_features, fmap, layer_index):
        """roi_features: list of [B*N, C, S, 1] for stages 0..layer_index.
        fmap: current-stage feature map [B, C, H, W]. Returns [B, N, fc_hidden_dim]."""
        batch_size = fmap.shape[0]
        num_queries = roi_features[0].shape[0] // batch_size
        feats = [self.convs[i](f) for i, f in enumerate(roi_features)]
        roi = self.catconv[layer_index](torch.cat(feats, dim=1))
        roi = roi.contiguous().view(batch_size * num_queries, -1)
        roi = F.relu(self.fc_norm(self.fc(roi)))
        roi = roi.view(batch_size, num_queries, self.fc_hidden_dim)
        context = F.dropout(self.attention(roi, fmap), p=0.1, training=self.training)
        return roi + context


def linear_relu(dim):
    return [nn.Linear(dim, dim), nn.ReLU(inplace=True)]


class SegDecoder(nn.Module):
    """CLRNet PlainDecoder: dropout -> 1x1 conv -> upsample to input size."""

    def __init__(self, img_h, img_w, in_channels, num_classes):
        super().__init__()
        self.img_h, self.img_w = int(img_h), int(img_w)
        self.dropout = nn.Dropout2d(0.1)
        self.conv8 = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        x = self.conv8(self.dropout(x))
        return F.interpolate(x, size=[self.img_h, self.img_w], mode="bilinear", align_corners=False)


def pool_prior_features(feature_map, prior_xs, prior_feat_ys, num_channels):
    """Sample features along priors.

    feature_map: [B, C, H, W]; prior_xs: [B, N, S] in sampling order
    (already flipped, as in CLRNet); prior_feat_ys: [S].
    Returns [B*N, C, S, 1].
    """
    batch_size, num_priors, num_points = prior_xs.shape
    xs = prior_xs.view(batch_size, num_priors, num_points, 1)
    ys = prior_feat_ys.view(1, 1, num_points, 1).expand(batch_size, num_priors, num_points, 1)
    grid = torch.cat((xs * 2.0 - 1.0, ys.to(xs.dtype) * 2.0 - 1.0), dim=-1)
    feature = F.grid_sample(feature_map, grid, align_corners=True).permute(0, 2, 1, 3)
    return feature.reshape(batch_size * num_priors, num_channels, num_points, 1)
