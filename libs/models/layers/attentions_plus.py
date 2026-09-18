import math
import time
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from einops import einsum, rearrange, repeat
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
import logging


logger = logging.getLogger(__name__)

class DeformConv2d(nn.Module):
    def __init__(self, inc, ouc, kernel_size=3, padding=1, stride=1, groups=1, bias=False):
        super().__init__()
        
        if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            kernel_size = kernel_size
        elif isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.offset_dim = 2 * kernel_size[0] * kernel_size[1] * groups
        self.deform_conv = torchvision.ops.DeformConv2d(inc, ouc, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups, bias=bias)
        # offset must follow the stride of deformable conv
        self.offset_conv = nn.Conv2d(inc, self.offset_dim, kernel_size=3, padding=1, dilation=1, stride=stride, bias=True, groups=1) 

    def forward(self, x):
        x = self.deform_conv(x, self.offset_conv(x))
        return x


@MODELS.register_module()
class AnchorPlusROIGather(nn.Module):
    """
    CLRNet ROIGather module to process pooled features
    and make them interact with global information.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/roi_gather.py
    """

    def __init__(
        self,
        in_channels,
        num_priors,
        sample_points,
        fc_hidden_dim,
        refine_layers,
        mid_channels=48,
        cross_attention_weight=1.0,
    ):
        """
        Args:
            in_channels (int): Number of input feature channels.
            num_priors (int): Number of priors (anchors).
            sample_points (int): Number of pooling sample points (rows).
            fc_hidden_dim (int): FC middle channel number.
            refine_layers (int): The number of refine levels.
            mid_channels (int): The number of input channels to catconv.
            cross_attention_weight (float): Weight to fuse cross attention result.
        """
        super(AnchorPlusROIGather, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.cross_attention_weight = cross_attention_weight

        if self.cross_attention_weight > 0:
            self.attention = AnchorVecFeatureMapAttention(num_priors, in_channels, total_num_refines=refine_layers)

        # learnable layers
        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        # during refinement process, total number of applied convolutions are not 3 -> 4 > 5 (in case of total 3 levels of features being used for prediction)
        num_refine_anchors = [1, 2, 5] # in case of total number of refinement is 3
        for i in range(refine_layers):
            refinements = []
            for j in range(num_refine_anchors[i]):
                if j == 0:
                    refinements.append(
                        ConvModule(
                            in_channels,
                            mid_channels,
                            (9, 1),
                            padding=(4, 0),
                            bias=False,
                            norm_cfg=dict(type='BN'),
                        ))
                else:
                    refinements.append(
                        ConvModule(
                            mid_channels,
                            mid_channels,
                            (9, 1),
                            padding=(4, 0),
                            bias=False,
                            norm_cfg=dict(type='BN'),
                        ))
            
            self.convs.append(nn.Sequential(*refinements))

            self.catconv.append(
                ConvModule(
                    mid_channels * (i + 1),
                    in_channels,
                    (9, 1),
                    padding=(4, 0),
                    bias=False,
                    norm_cfg=dict(type='BN'),
                )
            )

        self.fc = nn.Linear(sample_points * in_channels, fc_hidden_dim, bias=False)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        """
        Args:
            x (List[torch.Tensor]): List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, C, Ns, 1).
            layer_index (int): Current refine layer index.
        Returns:
            cat_feat (torch.Tensor): Fused feature tensor, shape (B * Np, C, Ns, 1).
        B: batch size, Np: number of priors (anchors), Ns: number of sample points (rows).
        """
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = torch.cat(feats, dim=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward_roi(self, roi_features, layer_index, bs):
        """
        Gather the ROI (pooled) features of the multiple refine levels
        and fuse them into the output tensor.
        Args:
            roi_features (List[torch.Tensor]): List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, Cin, Ns, 1).
            layer_index (int): The current refine layer index.
            bs (int): Batchsize B.
        Returns:
            roi (torch.Tensor): Output features, shape (B, Np, Ch).
        B: batch size, Np: number of priors (anchors), Ns: number of sample points (rows),
        Cin: input channel number, Ch: hidden channel number.
        """
        # [B * Np, Cin, Ns, 1] * N -> [B * Np, Cin, Ns, 1]
        roi = self.roi_fea(roi_features, layer_index)
        # [B * Np, Cin, Ns, 1] -> [B * Np, Cin * Ns]
        roi = roi.contiguous().view(bs * self.num_priors, -1)

        # [B * Np, Cin * Ns] -> [B * Np, Ch]
        roi = F.relu(self.fc_norm(self.fc(roi)))
        # [B * Np, Ch] -> [B, Np, Ch]
        roi = roi.view(bs, self.num_priors, -1)
        return roi

    def forward(self, roi_features, fmap_pyramid, layer_index):
        """
        ROIGather forward function.
        Args:
            roi_features (List[torch.Tensor]): List of pooled feature tensors
                at the current and past refine layers.
                shape: (B * Np, Cin, Ns, 1).
            fmap_pyramid (List[torch.Tensor]): Multi-level feature pyramid.
                Each tensor has a shape (B, Cin, H_i, W_i) where i is the pyramid level.
            layer_index (int): The current refine layer index.
        Returns:
            roi (torch.Tensor): Output feature tensors, shape (B, Np, Ch).
        B: batch size, Np: number of priors (anchors), Ns: number of sample points (rows),
        Cin: input channel number, Ch: hidden channel number.
        """
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            fmap_pyramid: feature map pyramid
            layer_index: currently on which layer to refine
        Return:
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        fmap = fmap_pyramid[layer_index]
        roi = self.forward_roi(
            roi_features, layer_index, fmap.size(0)
        )  # [B, Np, Ch]

        if self.cross_attention_weight > 0:
            context = self.attention(roi, fmap, layer_index)
            roi = roi + self.cross_attention_weight * F.dropout(
                context, p=0.1, training=self.training
            )

        return roi


class AnchorVecFeatureMapAttention(nn.Module):
    def __init__(self, n_query, dim, total_num_refines):
        """
        Args:
            n_query (int): Number of queries (priors, anchors).
            dim (int): Key and Value dim.
        """
        super(AnchorVecFeatureMapAttention, self).__init__()
        
        # in case of incoporating 3 level of features.
        # default feature map shape: ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25])
        # instead of using bilinear interpolate for key, value resizing, we use deformable convolution for the size matching
        self.dim = dim
        self.conv_resizes = nn.ModuleList()
        for i in range(total_num_refines):
            conv_resize = nn.Sequential(
                DeformConv2d(inc=dim, ouc=dim, kernel_size=3, padding=1, stride=2**i, groups=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU())
            
            self.conv_resizes.append(conv_resize)
            
        # self.resize = FeatureResize()
        
        self.f_key = ConvModule(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type='BN'),
        )

        self.f_query = nn.Sequential(
            nn.Conv1d(
                in_channels=n_query,
                out_channels=n_query,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=n_query,
            ),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0
        )
        self.W = nn.Conv1d(
            in_channels=n_query,
            out_channels=n_query,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=n_query,
            bias=True
        )
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, roi, fmap, layer_index):
        """
        Forward function for cross attention.
        Args:
            roi (torch.Tensor): Features pooled by priors, shape (B, Np, C).
            fmap (torch.Tensor): Feature maps at the current refine level, shape (B, C, H, W).
            layer_index (int): current layer for refinement
        Returns:
            context (torch.Tensor): Output global context, shape (B, Np, C).
        B: batch size, Np: number of priors (anchors)
        """
        query = self.f_query(roi)  # [B, Np, C] ->  [B, Np, C]
        
        # resize the source of key, value first
        fmap = self.conv_resizes[layer_index](fmap) # apply deformable conv2d v2 (by torchvision)
        
        key = self.f_key(fmap)  # [B, C, H, W]
        key = key.flatten(2)
        # key = self.resize(key)  # [B, C, H'W']
        
        value = self.f_value(fmap)  # [B, C, H, W]
        value = value.flatten(2).permute(0, 2, 1)
        # value = self.resize(value)  # [B, C, H'W']
        # value = value.permute(0, 2, 1)  # [B, H'W', C]

        # attention: [B, Np, C] x [B, C, H'W'] -> [B, Np, H'W']
        sim_map = torch.matmul(query, key)
        sim_map = (self.dim**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)  # [B, Np, C]
        context = self.W(context)  # [B, Np, C]
        return context


class FeatureResize(nn.Module):
    """Resize the feature map by interpolation."""
    def __init__(self, size=(10, 25)):
        """
        Args:
            size (tuple): Target size (H', W').
        """
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        """
        Forward function.
        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W).
        Returns:
            out (torch.Tensor): Resized tensor with shape (B, C, H'W').
        """
        x = F.interpolate(x, self.size)
        return x.flatten(2)
