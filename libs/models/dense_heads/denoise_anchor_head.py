"""
Adapted from:
https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn.bricks.transformer import build_attention
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.registry import MODELS
from mmdet.registry import TASK_UTILS
from mmdet.structures import SampleList
from nms import nms
from torch import Tensor

from libs.models.dense_heads.seg_decoder import SegDecoder
from libs.models.dense_heads.segman_decoder import SegMANDecoder
from libs.utils.lane_utils import Lane

class FeatCLSAttn(nn.Module):
    def __init__(self, dim=64, kv_type='img', num_points=36):
        super(FeatCLSAttn, self).__init__()
        
        self.dim = dim
        self.patch_sz = 1
        self.kv_type = kv_type
        self.num_points = num_points
        
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = dim ** 0.5
        
        self.linear_o = nn.Linear(dim, dim)
        
        if kv_type=='anchor':
            self.anchor_linear = nn.Linear(in_features=dim * num_points, out_features=dim, bias=True)
    
    def forward(self, q, kv):
        """
            q: cls features (B, Nr, C)
            kv: feature map shape (B, C, H, W)
        """
        
        B, Nr, C = q.shape
        
        if self.kv_type == 'img':
            B, C, H, W = kv.shape
            kv = kv.reshape(B, C, H*W).permute(0, 2, 1) # (B, H*W, C)
        elif self.kv_type == 'anchor':
            B, Nr, C, Np = kv.shape
            # kv = torch.mean(kv, dim=-1) # (B, Nr, C)
            kv = self.anchor_linear(kv.reshape(B, Nr, -1)) # (B, Nr, C * Np) -> (B, Nr, C)
        elif self.kv_type == 'fc_feat':
            pass
        else:
            raise Exception("Wrong kv_type: choose one of this ['img', 'anchor', 'fc_feat']")
        
        qx = self.linear_q(q)
        kx = self.linear_k(kv)
        vx = self.linear_v(kv)
        
        attn = self.softmax((qx @ kx.permute(0, 2, 1))/self.scale) # (B, Nr, H*W)
        attn_o = attn @ vx # (B, Nr, C)
        
        attn_o = self.linear_o(attn_o)
        return attn_o # (B, Nr, C)

# noised anchor denoising layer!
# based on the transformer attention.
class DenoiseLayer(nn.Module):
    def __init__(self, num_layers, dim, num_head, dropout, ffn_dim, num_points, scale_range):
        super(DenoiseLayer, self).__init__()
        
        self.num_layers = num_layers
        self.dim = dim
        self.num_head = num_head
        self.dropout = dropout
        self.ffn_dim = ffn_dim
        self.num_points = num_points # number of points
        
        if scale_range > 1 and scale_range % 2 != 0:
            raise Exception("scale range must be in powers of 2...")
        if scale_range == 1 or scale_range>=8:
            self.scale_range = 0
        else:
            self.scale_range = int(math.log2(scale_range)) # scale range based on power of 2.
        
        # inputs tensor (B * Nr, C, Ns, 1) that will be 'Query'. kernel_size=(9, 1) and padding=(4,0)
        adap_layer = []
        for _ in range(2):
            # depth-wise convolution
            adap_layer.append(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(9, 1), padding=(4,0), bias=False, groups=dim))
            adap_layer.append(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 1), padding=0, bias=False, groups=1))
            adap_layer.append(nn.LayerNorm(normalized_shape=[dim, num_points, 1]))
            adap_layer.append(nn.ReLU())
        self.adap_layer = nn.Sequential(*adap_layer)
        
        self.self_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_head, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim),
                                 nn.ReLU(),
                                 nn.Linear(ffn_dim, dim))
        self.ffn2 = nn.Sequential(nn.Linear(dim, ffn_dim),
                                 nn.ReLU(),
                                 nn.Linear(ffn_dim, dim))
        
        self.downsampler = nn.PixelUnshuffle(downscale_factor=2)
        
        self.channel_adapter = nn.ModuleList()
        for scale in range(scale_range):
            self.channel_adapter.append(nn.Conv2d(in_channels=dim * (4**(scale+1)), out_channels=dim, kernel_size=(1, 1), stride=1, bias=True))
            
        self.q_channel_adapter = nn.Linear(in_features=num_points * dim, out_features=dim, bias=True)
        
        self.self_layernorm = nn.LayerNorm(normalized_shape=[dim])
        self.ffn_layernorm = nn.LayerNorm(normalized_shape=[dim])
        self.cross_layernorm = nn.LayerNorm(normalized_shape=[dim])
        self.ffn2_layernorm = nn.LayerNorm(normalized_shape=[dim])
        
    def forward(self, q, kv, pyramid_level):
        """
            q: (B * Nr, C, Ns, 1)
            kv: feature map (B, C, H, W)
            pyramid_level: level of input feature pyramid (1:top) - scale between feature map must be 2
        """
        q = self.adap_layer(q)
        q = q.reshape(-1, self.dim, self.num_points).permute(0, 2, 1) # (B*Nr, C, Ns) -> (B*Nr, Ns, C)
        qx = self.self_layernorm(q)
        aq, _ = self.self_attention(qx, qx, qx) # (B*Nr, Ns, C)
        q = q + aq
        qx = self.ffn_layernorm(q)
        fq = self.ffn(qx) # ffn -> (B*Nr, Ns, C)
        q = q + fq
        
        # tokenizer-kv
        B, C, H, W = kv.shape
        if pyramid_level>1:
            if pyramid_level < 4:
                for _ in range(pyramid_level-1):
                    kv = self.downsampler(kv)
                kv = self.channel_adapter[pyramid_level-2](kv)
            else:
                kv = F.interpolate(kv, size=[10, 25])
        kv = kv.reshape(B, C, -1).permute(0, 2, 1) # (B, C, H*W)
        
        # adjust the query dimension
        q = q.reshape(B, -1, self.num_points * self.dim) # (B, Nr, Ns * C)
        q = self.q_channel_adapter(q) # (B, Nr, Ns*C) -> (B, Nr, C)
        
        qx = self.cross_layernorm(q)
        aq, _ = self.cross_attention(qx, kv, kv) # (B, Nr, C)
        q = q + aq
        
        qx = self.ffn2_layernorm(q)
        fq = self.ffn2(qx) # (B, Nr, C)
        q = q + fq
        
        return q # (B, Nr, C)


@MODELS.register_module()
class CLRDenoiseAnchorHead(BaseDenseHead):
    def __init__(
        self,
        anchor_generator,
        img_w=800,
        img_h=320,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        attention=None,
        loss_cls=None,
        loss_bbox=None,
        loss_iou=None,
        loss_seg=None,
        loss_denoise=None,
        train_cfg=None,
        test_cfg=None,
        use_segman_decoder=False,
        segman_decoder_params=None,
        use_assign_regularizer=False,
        regularizer_ema_beta=0.01
    ):
        super(CLRDenoiseAnchorHead, self).__init__()
        self.anchor_generator = TASK_UTILS.build(anchor_generator)
        self.img_w = img_w
        self.img_h = img_h
        self.n_offsets = self.anchor_generator.num_offsets
        self.n_strips = self.n_offsets - 1
        self.strip_size = self.img_h / self.n_strips
        self.num_priors = attention.num_priors = self.anchor_generator.num_priors
        self.sample_points = attention.sample_points = sample_points
        self.refine_layers = attention.refine_layers = refine_layers
        self.fc_hidden_dim = attention.fc_hidden_dim = fc_hidden_dim
        self.prior_feat_channels = attention.in_channels = prior_feat_channels # number of feature channels of prior. (defaults to 64 as neck outputs 64 channels)
        self.attention = MODELS.build(attention)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_seg = MODELS.build(loss_seg) if loss_seg["loss_weight"] > 0 else None
        self.loss_iou = MODELS.build(loss_iou)
        self.loss_denoise = MODELS.build(loss_denoise)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_segman_decoder = use_segman_decoder # whether to use segman decoder for multi-task segmentation head
        self.use_assign_regularizer = use_assign_regularizer # whether to use the number of assignment as regularizer for assignment (by loss or matching cost)
        self.regularizer_ema_beta = regularizer_ema_beta # ema beta for assign regularizer. Only valid when use_assign_regularizer=True
        if self.use_segman_decoder:
            if segman_decoder_params is None:
                self.segman_decoder_params = {'embed_dim': 128,
                                              'feat_proj_dim': 192,
                                              'num_classes': 5,
                                              'dropout_ratio': 0.01,
                                              'channel_split': False,
                                              'interpolate_mode': 'bilinear',
                                              'use_rpb': False}
            else:
                self.segman_decoder_params = segman_decoder_params
        else:
            self.segman_decoder_params = None # if use_segman_decoder is False, segman_decoder_params is not used
            
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(train_cfg['assigner'])
        

        # Non-learnable parameters
        # when sample_points=72, n_strips=sample_points-1=71 this generates the indices [0, 1, ..., 71]
        self.register_buffer(
            name="sample_x_indices",
            tensor=(
                torch.linspace(0, 1, steps=self.sample_points, dtype=torch.float32)
                * self.n_strips
            ).long(),
        )
        # this generates the y coordinates starts from 1.0 to 0.0 (normalized coordinates starting from the bottom of the image and increases to the top)
        self.register_buffer(
            name="prior_feat_ys",
            tensor=torch.flip(
                (self.sample_x_indices.float() / self.n_strips), dims=[-1]
            ),
        )
        # this also creates the y coordinates for the prior points in the range of [1.0, 0.0]
        self.register_buffer(
            name="prior_ys",
            tensor=torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32),
        )
        
        # Exponential moving average of number of assignment by simOTA
        # initial value for ema_num_assign follows the start value of total number of assignment
        self.register_buffer(
            name="ema_num_assign",
            tensor=torch.tensor(2.0)
        )
        self.register_buffer(
            name="ema_beta",
            tensor=torch.tensor(0.01)
        )

        reg_modules = list()
        cls_modules = list()
        for i in range(num_fc):
            reg_modules += [
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(inplace=True),
            ]
            cls_modules += [
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(inplace=True),
            ]
            
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)
        self.reg_layers = nn.Linear(self.fc_hidden_dim, self.n_offsets + 4)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)
        self.attention = build_attention(attention)
        
        self.cross_fcls = FeatCLSAttn(dim=self.prior_feat_channels, kv_type='img')
        self.cross_acls = FeatCLSAttn(dim=self.prior_feat_channels, kv_type='anchor', num_points=self.sample_points)
        self.cross_ccls = FeatCLSAttn(dim=self.prior_feat_channels, kv_type='fc_feat')

        # Auxiliary head
        if self.loss_seg:
            if self.use_segman_decoder:
                self.seg_decoder = SegMANDecoder(
                    image_size=(img_h, img_w),
                    in_channels=prior_feat_channels,
                    embed_dim=segman_decoder_params['embed_dim'],
                    feat_proj_dim=segman_decoder_params['feat_proj_dim'],
                    num_classes=segman_decoder_params['num_classes'],
                    dropout_ratio=segman_decoder_params['dropout_ratio'],
                    channel_split=segman_decoder_params['channel_split'],
                    interpolate_mode=segman_decoder_params['interpolate_mode'],
                    use_rpb=segman_decoder_params['use_rpb']
                )
            else:
                self.seg_decoder = SegDecoder(
                    self.img_h,
                    self.img_w,
                    num_classes=5,
                    prior_feat_channels=self.prior_feat_channels,
                    refine_layers=self.refine_layers,
                )
                
        self.decoder = DenoiseLayer(num_layers=3, dim=self.prior_feat_channels, num_head=2, dropout=0.2,
                                    ffn_dim=self.prior_feat_channels*2, num_points=self.sample_points, scale_range=4)
        layers = []
        for _ in range(2):
            layers.append(nn.Sequential(nn.Linear(in_features=self.prior_feat_channels, out_features=self.prior_feat_channels, bias=True),
                                            nn.ReLU()))
        layers.append(nn.Linear(in_features=self.prior_feat_channels, out_features=self.n_offsets, bias=True))
        self.decoder_head = nn.Sequential(*layers)
        
        self.init_weights()

    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)

    def pool_prior_features(self, batch_features, prior_xs):
        """
        Pool features from the feature map along the prior points.
        Args:
            batch_features (torch.Tensor): Input feature maps, shape: (B, C, H, W)
            prior_xs (torch.Tensor):. Prior points, shape (B, Np, Ns)
                where Np is the number of priors and Ns is the number of sample points.
        Returns:
            feature (torch.Tensor): Pooled features with shape (B * Np, C, Ns, 1).
        """

        batch_size = batch_features.shape[0]

        # (batch, num_priors, num_points) -> (batch, num_priors, num_points, 1)
        prior_xs = prior_xs.view(batch_size, self.num_priors, -1, 1)
        # (num_points) -> (batch_size * num_priors, num_points) -> (batch, num_priors, num_points, 1)
        prior_ys = self.prior_feat_ys.repeat(batch_size * self.num_priors).view(
            batch_size, self.num_priors, -1, 1
        )

        # change the range of prior_xs and prior_ys to [-1, 1]
        prior_xs = prior_xs * 2.0 - 1.0
        prior_ys = prior_ys * 2.0 - 1.0
        # (batch, num_priors, num_points, 2) concatenated
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
        
        # torch.nn.functional.grid_sample() do grid-sampling the (B, C, H, W) feature map according to the (batch, num_priors, num_points, 2) grid
        # output tensors are permuted to (B, C, Ns, 2) (from (B, C, Np, Ns) -> (B, Np, C, Ns))
        feature = F.grid_sample(batch_features, grid, align_corners=True).permute(
            0, 2, 1, 3
        )
        
        # (B, Np, C, Ns) -> (B* Np, C, Ns, 1)
        feature = feature.reshape(
            batch_size * self.num_priors,
            self.prior_feat_channels,
            self.sample_points,
            1,
        )
        return feature # this functions samples the values from the feature map at the prior points and returns the sampled values (pooled features)

    def forward(self, x, **kwargs):
        """
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: Input features (list[Tensor]). Each tensor has a shape (B, C, H_i, W_i),
                where i is the pyramid level.
                Example of shapes: ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        Returns:
            pred_dict (List[dict]): List of prediction dicts each of which containins multiple lane predictions.
                cls_logits (torch.Tensor): 2-class logits with shape (B, Np, 2).
                anchor_params (torch.Tensor): anchor parameters with shape (B, Np, 3).
                lengths (torch.Tensor): lane lengths in row numbers with shape (B, Np, 1).
                xs (torch.Tensor): x coordinates of the lane points with shape (B, Np, Nr).

        B: batch size, Np: number of priors (anchors), Nr: num_points (rows).
        """
        batch_size = x[0].shape[0]
        feature_pyramid = list(x[len(x) - self.refine_layers :])
        feature_pyramid.reverse()
        # e.g. [1, 64, 10, 25], [1, 64, 20, 50] [1, 64, 40, 100]
        
        # creates the anchors from the anchor generator
        # anchors are the instance of weight of embedding layer which has shape of (num_priors, 3)
        # from self.prior_ys (normalized y coordinates starting from 1.0 to 0.0), and self.sample_x_indices (indices of sample points starting from 0 to 71)
        # and img_w, img_h are used to denormalize the coordinates of anchors
        
        # it creates the coordinates of anchors (Np, Nr) using the prior of y coordinates and anchor's start x, y coordinates and theta (angle) parameters
        # and sample_x_indices are used to sample the x coordinates of anchors.
        # sampled_xs is sampled anchor x sampled at sample_x_indices (but it samples all the points from the anchors -> Nr=Ns by defaults)
        full_xs, sampled_xs, noised_anchor = self.anchor_generator.generate_anchors(
            self.anchor_generator.prior_embeddings.weight, # (Np, 3) -> (start of y, start of x, theta)
            self.prior_ys, # (Nr, ) -> constrained spacing of y coordinates (=priors)
            self.sample_x_indices, # (Ns, ) -> indices of sample points to sample which points are sampled
            self.img_w, # original img_w to calculate the x coordinates of anchors (in denormalized coordinates)
            self.img_h, # original img_h to calculate the y coordinates of anchors (in denormalized coordinates)
            add_noise=True
        )

        anchor_params = self.anchor_generator.prior_embeddings.weight.clone().repeat(
            batch_size, 1, 1
        )  # [B, Np, 3]
        
        # (num_priors, num_samples(=num_points by default)) -> (batch, num_priors, num_samples(=num_points by default))
        priors_on_featmap = sampled_xs.repeat(batch_size, 1, 1)
        full_xs = full_xs.unsqueeze(0).repeat(batch_size, 1, 1)
        noised_anchor = noised_anchor.unsqueeze(0).repeat(batch_size, 1, 1) # (B, Num_Prior, Num_Samples)

        predictions_list = []
        denoised_tgt_list = []

        # iterative refine
        pooled_features_stages = []
        for stage in range(self.refine_layers):
            prior_xs = priors_on_featmap  # torch.flip(priors_on_featmap, dims=[2])  # [24, 192, 36]
            
            # 1. anchor ROI pooling
            # [B, C, H, W] X [B, Np, Ns] => [B * Np, C, Ns, 1]
            # output pooled features by given coordinates of prior points. x coordinates are generated from anchors and y coordinates are fixed which is initialized
            # as the self.prior_feat_ys (which is normalized y coordinates starting from 1.0 to 0.0 with 72 sample points linearly spaced)
            pooled_features = self.pool_prior_features(feature_pyramid[stage], prior_xs)
            pooled_feat_w_noise = self.pool_prior_features(feature_pyramid[stage], noised_anchor)
            
            # (N, Nr, C)
            decoded_feature = self.decoder(pooled_feat_w_noise, feature_pyramid[stage], pyramid_level=stage+1) # pyramid_level starts from 1
            # (N, Nr, Ns)
            pred_denoised = self.decoder_head(decoded_feature)
            
            pooled_features_stages.append(pooled_features)
            
            # 2. generate cls embedding
            cls_embed = self.anchor_generator.generate_cls_features() # (Nr, self.prior_feat_channels)
            cls_embed = cls_embed.unsqueeze(0).repeat(batch_size, 1, 1) # (B, Nr, self.prior_feat_channels)
            
            # 3. compute cross-attention between classification feature and feature map (top feature map)
            cls_embed = self.cross_fcls(cls_embed, feature_pyramid[0]) 
            
            # 4. compute cross-attention between clssification feature and anchor features
            cls_embed = self.cross_acls(cls_embed, pooled_features.reshape(batch_size, self.num_priors, self.prior_feat_channels, -1))

            # 5. ROI gather
            # pooled features [B * Np, C, Ns, 1] * stages
            # feature pyramid: [B, C, Hs, Ws] (s = 0, 1, 2)
            fc_features = self.attention(
                pooled_features_stages, feature_pyramid, stage
            )  # [B, Np, Ch], Ch: fc_hidden_dim
            fc_features = fc_features.view(self.num_priors, batch_size, -1).reshape(
                batch_size * self.num_priors, self.fc_hidden_dim
            )  # [B * Np, Ch]
            
            cls_embed = self.cross_ccls(cls_embed, fc_features.reshape(batch_size, self.num_priors, -1))
            cls_embed = cls_embed.reshape(batch_size * self.num_priors, -1)

            # 6. cls and reg heads
            cls_features = fc_features.clone() + cls_embed.clone()
            reg_features = fc_features.clone() + cls_embed.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features)
            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[1]
            )  # (B, Np, 2)

            reg = self.reg_layers(reg_features)
            reg = reg.reshape(batch_size, -1, reg.shape[1])  # (B, Np, 4 + Nr)
            
            denoised_tgt = {
                "target_anchor" : full_xs
            }

            # 7. reg processing
            anchor_params += reg[:, :, :3]  # y0, x0, theta
            updated_anchor_xs, _, updated_noise_anchor = self.anchor_generator.generate_anchors(
                anchor_params.view(-1, 3),
                self.prior_ys,
                self.sample_x_indices,
                self.img_w,
                self.img_h,
                add_noise=True
            )
            updated_anchor_xs = updated_anchor_xs.view(batch_size, self.num_priors, -1)
            full_xs = updated_anchor_xs # update the full x points
            updated_noise_anchor = updated_noise_anchor.view(batch_size, self.num_priors, -1)
            
            # mean of gaussian random noise will be updated.
            noised_anchor = (noised_anchor + updated_noise_anchor) / 2
            reg_xs = updated_anchor_xs + reg[..., 4:]

            pred_dict = {
                "cls_logits": cls_logits,
                "anchor_params": anchor_params,
                "lengths": reg[:, :, 3:4],
                "xs": reg_xs,
                "denoised": pred_denoised
            }
            
            predictions_list.append(pred_dict)
            denoised_tgt_list.append(denoised_tgt)
            

            if stage != self.refine_layers - 1:
                anchor_params = anchor_params.detach().clone()
                priors_on_featmap = updated_anchor_xs.detach().clone()[
                    ..., self.sample_x_indices
                ]

        return predictions_list, denoised_tgt_list

    def loss_by_feat(self, out_dict, batch_data_samples, denoising_target):
        """Loss calculation from the network output.

        Args:
            out_dict (dict[torch.Tensor]): Output dict from the network containing:
                predictions (List[dict]): 3-layer prediction dicts each of which contains:
                    cls_logits: shape (B, Np, 2), anchor_params: shape (B, Np, 3),
                    lengths: shape (B, Np, 1) and xs: shape (B, Np, Nr).
                seg (torch.Tensor): segmentation maps, shape (B, C, H, W).
                where
                B: batch size, Np: number of priors (anchors), Nr: number of rows,
                C: segmentation channels, H and W: the largest feature's spatial shape.
            batch_data_samples: (List[:obj:`DetDataSample`]): The data samples
                that include meta information.
            denoising_target: original anchors that gaussian noises are not added
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = len(batch_data_samples)
        device = out_dict["predictions"][0]["cls_logits"].device
        cls_loss = torch.tensor(0.0).to(device)
        reg_xytl_loss = torch.tensor(0.0).to(device)
        iou_loss = torch.tensor(0.0).to(device)
        denoised_loss = torch.tensor(0.0).to(device)
        num_assignment = torch.tensor(0.0).to(device)
        total_assignment = torch.tensor(0.0).to(device)

        for stage in range(self.refine_layers):
            
            denoised_loss = denoised_loss + self.loss_denoise(out_dict["predictions"][stage]['denoised'], denoising_target[stage]["target_anchor"])
            
            for b, img_meta in enumerate(batch_data_samples):
                pred_dict = {k: v[b] for k, v in out_dict["predictions"][stage].items()}
                cls_pred = pred_dict["cls_logits"]
                target = img_meta.lanes.clone().to(device)  # [n_lanes, 78]
                target = target[target[:, 1] == 1]
                cls_target = cls_pred.new_zeros(cls_pred.shape[0]).long()
                
                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_loss = cls_loss + self.loss_cls(cls_pred, cls_target).sum()
                    continue
                
                # assignment runs here#
                with torch.no_grad():
                    (matched_row_inds, matched_col_inds, num_assignment_, total_assignment_) = self.assigner.assign(
                        pred_dict, target.clone(), img_meta
                    )
                num_assignment = num_assignment + num_assignment_
                total_assignment = total_assignment + total_assignment_
                
                # classification targets
                cls_target[matched_row_inds] = 1
                cls_loss = (
                    cls_loss
                    + self.loss_cls(cls_pred, cls_target).sum() / target.shape[0]
                )

                # regression targets -> [start_y, start_x, theta]
                # (all transformed to absolute values), only on matched pairs
                reg_yxtl = torch.cat(
                    (pred_dict["anchor_params"], pred_dict["lengths"]), dim=1
                )
                reg_yxtl = reg_yxtl[matched_row_inds]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= self.img_w - 1
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()

                # regression targets -> S coordinates (all transformed to absolute values)
                pred_xs = pred_dict["xs"][matched_row_inds]
                target_xs = target[matched_col_inds, 6:].clone()

                # adjust target length by start point difference
                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        reg_yxtl[:, 0].round().long(), 0, self.n_strips
                    )  # ensure the predictions starts is valid
                    target_starts = (
                        (target[matched_col_inds, 2] * self.n_strips).round().long()
                    )
                    target_yxtl[:, -1] -= predictions_starts - target_starts

                # Loss calculation
                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180

                reg_xytl_loss = (
                    reg_xytl_loss + self.loss_bbox(reg_yxtl, target_yxtl).mean()
                )

                iou_loss = iou_loss + self.loss_iou(
                    pred_xs * (self.img_w - 1) / self.img_w, target_xs / self.img_w
                )
                
        num_assignment /= batch_size * self.refine_layers
        total_assignment /= batch_size * self.refine_layers
        
        cls_loss /= batch_size * self.refine_layers
        
        reg_xytl_loss /= batch_size * self.refine_layers
        iou_loss /= batch_size * self.refine_layers
        
        # because loss_denoise use 'none' reduction, we need to also divide the sum of loss by batch_size
        denoised_loss /= self.refine_layers

        loss_dict = {
            "loss_cls": cls_loss,
            "loss_reg_xytl": reg_xytl_loss,
            "loss_iou": iou_loss,
            "loss_denoise": denoised_loss,
            # add raw scalars for logging
            "num_assignment": num_assignment.detach().float(),
            "total_assignment": total_assignment.detach().float(),
        }

        # extra segmentation loss
        if self.loss_seg:
            tgt_masks = np.array([t.gt_masks[0] for t in batch_data_samples])
            tgt_masks = torch.tensor(tgt_masks).long().to(device)  # (B, H, W)
            loss_dict["loss_seg"] = self.loss_seg(out_dict["seg"], tgt_masks)

        return loss_dict

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Forward function for training mode.
        Args:
            x (list[Tensor]): Features from backbone.
            batch_data_samples (List[:obj:`DetDataSample`]): The data samples
                that include meta information.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        predictions, denoising_tgt = self(x)
        out_dict = {"predictions": predictions}
        if self.loss_seg:
            out_dict["seg"] = self.forward_seg(x)

        losses = self.loss_by_feat(out_dict, batch_data_samples, denoising_tgt)
        
        # update the ema_value
        self.ema_num_assign = self.ema_num_assign * (1-self.ema_beta) + losses["total_assignment"] * self.ema_beta
        
        if self.use_assign_regularizer:
            eta = 1e-8
            ema_weight = torch.log(self.ema_num_assign + eta)
            weights =  torch.clip(ema_weight * 0.9, 1.0, 2.0)

            loss_str = ["loss_cls", "loss_reg_xytl", "loss_iou", "loss_seg"]
            for l_str in loss_str:
                losses[l_str] *= weights
        
        return losses

    def forward_seg(self, x):
        """Forward function for training mode.
        Args:
            x (list[torch.tensor]): Features from backbone.
        Returns:
            torch.tensor: segmentation maps, shape (B, C, H, W), where
            B: batch size, C: segmentation channels, H and W: the largest feature's spatial shape.
        """
        if self.use_segman_decoder:
            # backbone feature output order: from bottom to top. (high resolution, low channels to low resolution, high channels)
            seg = self.seg_decoder(x)
        else:
            batch_features = list(x[len(x) - self.refine_layers :])
            batch_features.reverse()
            seg_features = torch.cat(
                [
                    F.interpolate(
                        feature,
                        size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for feature in batch_features
                ],
                dim=1,
            )
            seg = self.seg_decoder(seg_features)
        return seg

    def get_lanes(self, pred_dict, as_lanes=True, extend_bottom=True):
        """
        Convert model output to lane instances.
        Args:
            pred_dict (dict): prediction dict containing multiple lanes.
                cls_logits (torch.Tensor): 2-class logits with shape (B, Np, 2).
                anchor_params (torch.Tensor): anchor parameters with shape (B, Np, 3).
                lengths (torch.Tensor): lane lengths in row numbers with shape (B, Np, 1).
                xs (torch.Tensor): x coordinates of the lane points with shape (B, Np, Nr).
            as_lanes (bool): transform to the Lane instance for interpolation.
        Returns:
            pred (List[torch.Tensor]): List of lane tensors (shape: (N, 2))
                or `Lane` objects, where N is the number of rows.
            scores (torch.Tensor): Confidence scores of the lanes.

        B: batch size, Np: num_priors, Nr: num_points (rows).
        """
        softmax = nn.Softmax(dim=2)
        threshold = self.test_cfg.conf_threshold
        all_scores = softmax(pred_dict["cls_logits"])[:, :, 1]
        all_keep_inds = all_scores >= threshold  # [64, 192]
        out_preds = []
        out_scores = []
        for scores, xs, lengths, anchor_params, keep_inds in zip(
            all_scores, pred_dict["xs"], pred_dict["lengths"],
            pred_dict["anchor_params"], all_keep_inds):
            scores = scores[keep_inds]
            xs = xs[keep_inds]
            lengths = lengths[keep_inds]
            anchor_params = anchor_params[keep_inds]
            if xs.shape[0] == 0:
                out_preds.append([])
                out_scores.append([])
                continue

            if self.test_cfg.use_nms:
                nms_anchor_params = anchor_params[..., :2].detach().clone()
                nms_anchor_params[..., 0] = 1 - nms_anchor_params[..., 0]
                nms_predictions = torch.cat(
                    [
                        pred_dict["cls_logits"][0, keep_inds].detach().clone(),
                        nms_anchor_params[..., :2],
                        lengths.detach().clone() * self.n_strips,
                        xs.detach().clone() * (self.img_w - 1),
                    ],
                    dim=-1,
                )  # [N, 77]
                keep, num_to_keep, _ = nms(
                    nms_predictions,
                    scores,
                    overlap=self.test_cfg.nms_thres,
                    top_k=self.test_cfg.nms_topk,
                )
                keep = keep[:num_to_keep]
                xs = xs[keep]
                scores = scores[keep]
                lengths = lengths[keep]
                anchor_params = anchor_params[keep]

            lengths = torch.round(lengths * self.n_strips)
            pred = self.predictions_to_lanes(xs, anchor_params, lengths, scores, as_lanes, extend_bottom)
            out_preds.append(pred)
            out_scores.append(scores)

        return out_preds, out_scores

    def predictions_to_lanes(
        self, pred_xs, anchor_params, lengths, scores, as_lanes=True, extend_bottom=True
    ):
        """
        Convert predictions to the lane segment instances.
        Args:
            pred_xs (torch.Tensor): x coordinates of the lane points with shape (Nl, Nr).
            anchor_params (torch.Tensor): anchor parameters with shape (Nl, 3).
            lengths (torch.Tensor): lane lengths in row numbers with shape (Nl, 1).
            scores (torch.Tensor): confidence scores with shape (Nl,).
            as_lanes (bool): transform to the Lane instance for interpolation.
            extend_bottom (bool): if the prediction does not start at the bottom of the image,
                extend its prediction until the x is outside the image.
        Returns:
            lanes (List[torch.Tensor]): List of lane tensors (shape: (N, 2))
                or `Lane` objects, where N is the number of rows.

        B: batch size, Nl: number of lanes after NMS, Nr: num_points (rows).
        """
        prior_ys = self.prior_ys.double()
        lanes = []
        for lane_xs, lane_param, length, score in zip(
            pred_xs, anchor_params, lengths, scores
        ):
            start = min(
                max(0, int(round((1 - lane_param[0].item()) * self.n_strips))),
                self.n_strips,
            )
            length = int(round(length.item()))
            end = start + length - 1
            end = min(end, len(prior_ys) - 1)
            if extend_bottom:
                edge = (lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)
                start -= edge.flip(0).cumprod(dim=0).sum()
            lane_ys = prior_ys[start : end + 1]
            lane_xs = lane_xs[start : end + 1]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (
                lane_ys * (self.test_cfg.ori_img_h - self.test_cfg.cut_height)
                + self.test_cfg.cut_height
            ) / self.test_cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1
            ).squeeze(2)
            if as_lanes:
                lane = Lane(
                    points=points.cpu().numpy(),
                    metadata={
                        "start_x": lane_param[1],
                        "start_y": lane_param[0],
                        "conf": score,
                    },
                )
            else:
                lane = points
            lanes.append(lane)
        return lanes

    def predict(self, feats, data_samples, rescale=False):
        """Test function without test-time augmentation.
        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the FPN.
            data_samples (List[:obj:`DetDataSample`]): The data samples
                that include meta information.
            rescale (bool, optional): Whether to rescale the results.
        Returns:
            result_dict (dict): Inference result containing
                lanes (List[torch.Tensor]): List of lane tensors (shape: (N, 2))
                    or `Lane` objects, where N is the number of rows.
                scores (torch.Tensor): Confidence scores of the lanes.
        """
        pred_dict = self(feats)[-1]
        all_lanes, all_scores = self.get_lanes(
            pred_dict,
            as_lanes=self.test_cfg.as_lanes,
            extend_bottom=self.test_cfg.extend_bottom
            )
        result_dict = [{
            "lanes": lanes,
            "scores": scores,
            "metainfo": ds.metainfo,
        }
        for lanes, scores, ds in zip(all_lanes, all_scores, data_samples)]
        return result_dict
