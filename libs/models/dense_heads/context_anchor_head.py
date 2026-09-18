"""
Adapted from:
https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py
"""
from typing import Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class ContextExtractor(nn.Module):
    """
        ContextExtractor enlarges the receptive field of extraction. 
        It gives the learnable & more wide receptive field than one-pixel grid sampling.
        
        N: batch size
        C: number of channels
        H: height of image
        W: width of image
        Nr: number of anchors
        Np: number of points
        k: kernel size
        
        img: (N, C_in, H, W) image
        points: (N, Nr, Np, 2) normalized coordinate [-1, 1], order: (x, y)
        kernel: (C_out, C_in, k, k) or (N, Nr, Np, C_out, C_in, k, k) # if you apply different kernel by point, use the later form
        bias: (C_out, ) or (N, Nr, Np, C_out)
        return: (N, Nr, Np) # scala-convolution result (channel summation) for each point
    """
    
    def __init__(self, C_in, C_out, kr_size=3, stride=1, padding='same', align_corners=False):
        super().__init__()
        self.C_in, self.C_out, self.k = C_in, C_out, kr_size
        self.stride = stride
        self.padding = padding
        self.align_corners=align_corners
        
        
        self.kernel = nn.Parameter(torch.empty(C_out, C_in, kr_size, kr_size))    
        self.bias = nn.Parameter(torch.empty(C_out))
        
        # initialize as default (same as Conv2D kaiming)
        nn.init.kaiming_uniform_(self.kernel, a=0)
        nn.init.zeros_(self.bias)
        
        # precompute the k * k offsets (registered as buffer so it moves with .to(device))
        yy, xx = torch.meshgrid(
            torch.linspace(-(kr_size-1)/2, (kr_size-1)/2, steps=kr_size),
            torch.linspace(-(kr_size-1)/2, (kr_size-1)/2, steps=kr_size),
            indexing='ij'
        )
        
        # stride support
        self.register_buffer('xx', xx * stride)
        self.register_buffer('yy', yy * stride)
        
        
    def forward(self, img, points):
        """ 
        img: (N, C_in, H, W) image
        points: (N, Nr, Np, 2) normalized coordinate [-1, 1], order: (x, y)
        """
        N, C, H, W = img.shape
        N, Nr, Np, _ = points.shape
        # make grid (N, Nr, Np, 2) -> (N, Nr, Np, 1, 1, 2) (in x, y order, normalized coordinate [-1, 1])
        base = points.view(N, Nr, Np, 1, 1, 2)
        offsets = torch.stack([self.xx, self.yy], dim=-1) # (k, k, 2)
        grid = base + offsets # (N, Nr, Np, k, k, 2) - assume that already normalize coordinate
        
        # grid_sample inputs (N, H_out, W_out, 2) grid reshape
        grid = grid.view(N, Nr * Np * self.k, self.k, 2)
        # extract patch-samples for each channel: (N, C, Nr * Np * k, k)
        patches = F.grid_sample(img, grid, mode='bilinear', align_corners=self.align_corners)
        # (N, C, Nr, Np, k, k)
        patches = patches.view(N, C, Nr, Np, self.k, self.k)
        
        # kernel broadcast
        if self.kernel.dim() == 4: 
            # (N, C, Nr, Np, k, k) @ (C_out, Cin, k, k) -> (N, C_out, Nr, Np)
            out = torch.einsum('n c r p a b, o c a b -> n o r p', patches, self.kernel)
            out = out.permute(0, 2, 1, 3) # (N, Nr, C_out, Np)
            
            if self.bias is not None:
                out = out + self.bias.view(1, 1, -1, 1)
                
        else: 
            # dynamic per-point weights: (N, Nr, Np, C_out, C_in, k, k)
            # expand patches to match and contract: (N, Nr, Np, C_out)
            # Einsum dims: patches(n, c, nr, np, a, b), weight(n, nr, np, o, c, a, b) -> (n, o, nr, np)
            out = torch.einsum('n c r p a b, n r p o c a b -> n o r p', patches, self.kernel)
            
            if self.bias is not None:
                out = out + self.bias.permute(0, 1, 3, 2) # (N, Nr, C_out, Np)
        
        return out # (N, Nr, C_out, Np)


class AnchorKMEEncoder(nn.Module):
    """
        Kernel Mean Embedding (KME) with Random Fourier Features (RFF) - linear in number of anchors
        Treat the set of anchors as a distribution and embed it with a kernel mean.
        
        in_channels: number of input channels
        rff_dim: dimension of Random Fourier Features
        sigma: divide factor that is the scale of random gaussian distribution
        use_second_moment: whether to use second momentum
        out_dim: output dimension
        dropout: dropout rate
        num_points: number of points that consists the anchors (only used when learnable=True)
        learnable: whether to learn the anchor representation instead of mean-pooling. If false, then use the mean pooling to get the anchor representation
    """
    
    def __init__(self, in_channels, rff_dim, sigma, use_second_moment, out_dim, dropout, num_points, learnable=False):
        super().__init__()
        self.in_channels = in_channels
        self.rff_dim = rff_dim
        self.sigma = sigma
        self.use_second_moment = use_second_moment
        self.out_dim = out_dim
        self.dropout = dropout
        self.learnable = learnable
        
        # Random Fourier bases (fixed) for a Gaussian kernel on concated features (context + positional)
        self.register_buffer('W', torch.randn(in_channels + 2, rff_dim) / sigma)
        self.register_buffer('b', 2 * math.pi * torch.rand(rff_dim))
        self.post = nn.Sequential(
            nn.LayerNorm(rff_dim * (2 if use_second_moment else 1)),
            nn.Linear(rff_dim * (2 if use_second_moment else 1), out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
        if learnable:
            # we does not apply the depthwise convolution style here. But you can apply depthwise and see how it goes!
            # here, we did not reduce the output channels, because RFF kernel takes (in_channels + 2) as input dimension.
            self.anchor_encoder = nn.Conv2d(in_channels=in_channels+2, out_channels=in_channels+2, kernel_size=(1, num_points), stride=1, padding=0, dilation=1, bias=True)
            
        
    def rff(self, u):
        # u: (..., Dtot)
        proj = u @ self.W + self.b  # (..., R)
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(proj)
    
    def forward(self, x, pos, mask=None):
        """
            x: (B, Nr, C, Np) shape tensor
            pos: (B, Nr, Np, 2) grid tensor
            mask: (B, Nr) mask tensor
        """
        x = x.permute(0, 2, 1, 3) # (B, Nr, C, Np) -> (B, C, Nr, Np)
        pos = pos.permute(0, 3, 1, 2) # (B, Nr, Np, 2) -> (B, 2, Nr, Np)
            
        if self.learnable:
            x = torch.cat((x, pos), dim=1) # (B, Nr, C, Np), (B, C, Nr, Np) -> (B, C+2, Nr, Np)
            x = self.anchor_encoder(x).squeeze().permute(0, 2, 1) # (B, C+2, Nr, Np) -> (B, C+2, Nr, 1) -> (B, C+2, Nr) -> (B, Nr, C+2)
        else: 
            x_mean = torch.mean(x, dim=-1) # (B, C, Nr, Np) -> (B, C, Nr)
            pos_mean = torch.mean(pos, dim=-1) # (B, 2, Nr, Np) -> (B, 2, Nr)
            x = torch.cat((x_mean, pos_mean), dim=1).permute(0, 2, 1) # (B, C+2, Nr) -> (B, Nr, C+2)
            
        phi = self.rff(x) # (B, Nr, C+2) -> (B, Nr, rff_dim)
        if mask is not None:
            m = mask.float().unsqueeze(-1) # (B, Nr, 1)
            phi_sum = (phi * m).sum(dim=1) # (B, rff_dim)
            denominator = m.sum(dim=1).clamp_min(1.0)
            mu = phi_sum / denominator     # mean embedding
        else:
            mu = phi.mean(dim=1)
        
        if self.use_second_moment:
            # second moment in feature space (diagonal approximate): E[phi^2]
            if mask is not None:
                mu2 = ((phi ** 2) * m).sum(dim=1) / denominator
            else:
                mu2 = (phi ** 2).mean(dim=1)
            
            stats = torch.cat([mu, mu2], dim=-1)
        else:
            stats = mu
        
        dist_repr = self.post(stats) # (B, out_dim)
        
        # concatenate this dist_repr with your global head or fuse back into per-anchor heads.
        return dist_repr # Distribution representation
    


class IPAttnDistEncoder(nn.Module):
    """
        Inducing-Point (Learnable queries(tokens) that models the distribution of anchor via cross-attention with original anchor distribution)
        
        it is lightweight and simple for implementation. But what number of learnable anchors are optimal for the leraning?
        
        in_channels: number of input channels,
        dim: number of feature dimension for distribution encoder
        heads: number of heads
        lr_anchors: learnable anchors
        dropout: dropout rate in multi-head attention
        num_points: number of points (=Np)
    """
    def __init__(self, num_points, in_channels, dim, heads, lr_anchor, dropout=0.0):
        super().__init__()
        self.num_points = num_points
        self.in_channels = in_channels
        self.dim = dim
        self.heads=heads
        self.lr_points = lr_anchor
        self.dropout=dropout
        
        # convolution that creates the representative feature for anchors
        self.anchor_encoder = nn.Conv2d(in_channels=in_channels+2, out_channels=dim, kernel_size=(1, num_points), dilation=1, stride=1,
                                        padding=0, bias=True)
        
        # shared across the batch dimension (lr_points, dim) - initialized as normalized random gaussian distribution with 0.02 scale
        self.point_queries = nn.Parameter(torch.randn(lr_anchor, dim) * 0.02) 
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, bias=True, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(dim, heads, dropout=dropout, bias=True, batch_first=True)
        
        # layer-normalization (but considering most of the normalization layers are batch-norm this must be used carefully.)
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x, xpos):
        """
        x: (N, Nr, C, Np) tensor
        xpos: (X, Nr, Np, 2) grid tensor
        """
        B = x.size(0)
        
        x = x.permute(0, 2, 1, 3) # (N, C, Nr, Np)
        xpos = xpos.permute(0, 3, 1, 2) # (N, 2, Nr, Np)
        
        x = torch.concat((x, xpos), dim=1) # (N, C+2, Nr, Np)
        
        x = self.anchor_encoder(x).squeeze().permute(0, 2, 1) # (N, C+2, Nr, Np) -> (N, C, Nr) -> (N, Nr, C)
        point_queries = self.point_queries.unsqueeze(0).expand(B, -1, -1) # (lr_points, dim) -> (N, lr_points, dim) - (expand batch dimension)
        ip_attn, _ = self.cross_attn(point_queries, x, x) # (N, lr_points, dim) -> (N, lr_points, dim)
        x_attn, _ = self.cross_attn2(x, ip_attn, ip_attn) # (N, Nr, dim) -> (N, Nr, dim)
        
        return self.ln(x_attn)



class ConvDistEncoder(nn.Module):
    """
        Sparse local mixing over a fixed anchor grid - depthwise separable 1D/2D convolution
        Although the distribution of anchors changes dynamically, the order of anchors (input set) are fixed during training.
        here, the kernel size or dilation or deformable can be good choices.
        
        appply large-receptive-field convolution stack to the anchor feature - (B, C, Nr, Np)
        dla34 training hyperparameters: Nr=192, Np=36 (72 but sampled to 36)

        in_channels, hiddens, num_layers: see AnchorDwConvEncoder
    """
    
    class AnchorDwConvEncoder(nn.Module):
        """
            Convolutional Encoder which uses the multi-kernel size (ks=3, 7, 9) in depth-wise separable style.
            this will save the computational memory while using the large-kernel convolution
            
            Note that space between features are not the real distance of anchors (=pixels). Each Anchor will
            form some specific distribution but that does not mean that it has the high correlation with pixel-distance.
            
            Also Note that if the kernel size is too large, it can cause dilution of features as the space is the based on pixel-distance.
            See the Graph Convolutional Network (GNN).
            
            in_channels: input number of channels
            hiddens: hidden channels (often in_channels * 2 or in_channels * 4 (=128 or 256))
            out_channels: output number of channels
            num_layers: number of layers for each spatial convolution (default=2).
        """
        
        def create_convb(self, in_channels, out_channels, kernel_size, dilation, stride, padding, bias, groups):
            block = [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                          dilation=dilation, stride=stride, padding=padding, bias=bias, groups=groups),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
                
            ]
            cblock = nn.Sequential(*block)
            return cblock
            
        def __init__(self, in_channels, hiddens, out_channels, num_layers=2):
            super().__init__()
            self.in_channels = in_channels
            self.hiddens = hiddens
            self.out_channels = out_channels
            self.num_layers = num_layers
            
            # channels-mixer (pointwise convolution): in_chanenls -> hiddens (expansion)
            self.cmixer_exp = nn.Conv2d(in_channels, hiddens, kernel_size=1, stride=1, padding=0, bias=True)
            
            smixer_3l = []
            smixer_7l = []
            smixer_9l = []
            # spatial-mixer (depthwise convolution) - ks=3, 7, 9
            for _ in range(num_layers):
                smixer_3l.append(self.create_convb(in_channels=hiddens, out_channels=hiddens, kernel_size=3, dilation=1, stride=1, padding=1, bias=True, groups=hiddens))
                smixer_7l.append(self.create_convb(in_channels=hiddens, out_channels=hiddens, kernel_size=7, dilation=1, stride=1, padding=3, bias=True, groups=hiddens))
                smixer_9l.append(self.create_convb(in_channels=hiddens, out_channels=hiddens, kernel_size=9, dilation=1, stride=1, padding=4, bias=True, groups=hiddens))
            
            self.smixer_3 = nn.Sequential(*smixer_3l)
            self.smixer_7 = nn.Sequential(*smixer_7l)
            self.smixer_9 = nn.Sequential(*smixer_9l)
                
            # channels-mixer (pointwise convolution): hiddens -> in_channels (shrink)
            self.cmixer_shr = nn.Conv2d(hiddens, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            
        def forward(self, x):
            
            exp_x = self.cmixer_exp(x)
            mixer_x = self.smixer_3(exp_x) + self.smixer_7(exp_x) + self.smixer_9(exp_x)
            x = self.cmixer_shr(mixer_x)
            return x
                    
                
    def __init__(self, in_channels=2, hiddens=16, num_layers=3):
        super().__init__()
        self.in_channels=in_channels
        self.anchor_encoder = ConvDistEncoder.AnchorDwConvEncoder(in_channels=in_channels, hiddens=hiddens, out_channels=hiddens, num_layers=num_layers)
    
    def forward(self, xpos):
        """
            xpos: (B, Nr, Np, 2) shape tensor (it is the grid that shows the 2D location)
        """
        xpos = xpos.permute(0, 3, 1, 2) # (B, 2, Nr, Np)
        x = self.anchor_encoder(xpos) # (B, hidden, Nr, Np)
        x = torch.mean(x, dim=-1) # (B, hidden, Nr)
        x = x.permute(0, 2, 1) # (B, Nr, hidden)
        return x


class SATTDistEncoder(nn.Module):
    """
    self-attention for modeling the distribution of learned anchors.
    Naive implementation of distribution encoder. As the number of anchors = 192
    Simply applying the self-attention to describe the distribution requires too much computation
    More efficient (less quadratic) methods maybe the convolution or inducing-point attention (learnable point set to match the original distribution)

    dim: input model dimension
    num_points: number of points for each anchor
    
    =========================================================================
    x: input features for the convolution, (B, Nr, C_out, Np)
    x_pos: normalized grid [-1, 1] for positional information, (B, Nr, Np, 2)
    
    """
    def __init__(self, dim, num_points, attn_dropout):
        super().__init__()
        self.d_model = dim
        self.num_points = num_points
        self.attn_drop = attn_dropout
        
        self.feat_encoder = nn.Conv2d(in_channels=dim+2, out_channels=dim+2, kernel_size=(1, num_points),
                                      stride=1, padding=0, bias=True)
        
        
        self.q = nn.Linear(in_features=dim+2, out_features=dim, bias=True)
        self.k = nn.Linear(in_features=dim+2, out_features=dim, bias=True) 
        self.v = nn.Linear(in_features=dim+2, out_features=dim, bias=True)
        
        self.o = nn.Linear(in_features=dim, out_features=dim, bias=True)
    
    def forward(self, x, x_pos):
        
        # (B, Nr, Np, 2) -> (B, Nr, 2, Np)
        x_pos = x_pos.permute(0, 1, 3, 2)
        # (B, Nr, C_out+2, Np)
        x_feat = torch.concat((x, x_pos), dim=2)
        # (B, Nr, C_out+2, Np) -> (B, C_out+2, Nr, Np)
        x_feat = x_feat.permute(0, 2, 1, 3)
        
        
        # x_anchor_feat is the feature that represents the anchors
        # (B, C_out+2, Nr, Np) -> (B, C_out+2, Nr, 1) -> (B, C_out+2, Nr)
        x_anchor_feat = self.feat_encoder(x_feat).squeeze()
        
        # permute the order
        # (B, C_out+2, Nr) -> (B, Nr, C_out+2)
        x_anchor_feat = x_anchor_feat.permute(0, 2, 1)
        
        query = self.q(x_anchor_feat)
        key = self.k(x_anchor_feat)
        value = self.v(x_anchor_feat)
        
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.attn_drop, is_causal=False)
        
        # (B, Nr, C_out+2) -> (B, Nr, C_out)
        attn_o = self.o(attn)
        
        return attn_o

class AnchorFeatAttn(nn.Module):
    """
        Self-Attn for learning object (lane) relation.
        This gives the context modeling between anchor features.
        
        in_dim: input dimension of anchor feature
        num_heads: number of heads for attention
        dropout: dropout rate for self-attention
        compress_rt: compress ratio of anchors. this is used for saving computation.
        num_anchors: Number of anchors
        num_layers: number of attention layers
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout, 
                 compress_rt=2.0, num_anchors=192, num_points=36, num_layers=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.compress_rt = compress_rt
        self.num_anchors = num_anchors
        self.num_points = num_points
        self.num_layers = num_layers
        
        # (B, Nr, C)
        self.compressor = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=int(compress_rt), padding=1, dilation=1, bias=True)
        # bridge layer between convolution and attention
        self.bridge = nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.group_norm = nn.GroupNorm(compress_rt, num_channels=out_dim) # this input (B, C, *) shape tensor. permute before input.
        
        self.attn = nn.ModuleList()
        self.norm_attn = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.norm_fpn = nn.ModuleList()
        
        for _ in range(num_layers):
            self.attn.append(nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads, dropout=dropout, bias=True, batch_first=True))
            # self.norm_attn.append(nn.GroupNorm(num_groups=int(compress_rt), num_channels=out_dim))
            self.norm_attn.append(nn.LayerNorm(normalized_shape=out_dim))
            self.ffn.append(
                nn.Sequential(
                        nn.Linear(in_features=out_dim, out_features=hidden_dim, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True)
                    )
            )
            self.dropout = nn.Dropout(p=dropout)
            # self.norm_fpn.append(nn.GroupNorm(num_groups=int(compress_rt), num_channels=out_dim))
            self.norm_fpn.append(nn.LayerNorm(normalized_shape=out_dim))
            
        self.pos_embed = nn.Parameter(torch.randn(num_anchors//compress_rt, out_dim)) # (Nr//compress_rt, C_out)
        self.group_norm_out = nn.GroupNorm(compress_rt, num_channels=out_dim)
        self.bridge_out = nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        
        self.channel_adapter = nn.Linear(in_features=(in_dim+2) * self.num_points, out_features=in_dim, bias=True)
        
    def forward(self, x):
        """
            x: (B, Nr, C, Np) features
        """
        B = x.shape[0]
        # (B, Nr, C, Np) -> (B, Nr, C*Np) -> (B, Nr, C) -> (B, C, Nr)
        # anchor_feature = torch.mean(x, dim=-1).permute(0, 2, 1)
        x = x.reshape(B, self.num_anchors, -1)
        anchor_feature = self.channel_adapter(x)
        anchor_feature = anchor_feature.permute(0, 2, 1)

        # (B, C, Nr) -> (B, C, Nr//compress_rt)
        compressed = self.compressor(anchor_feature)
        compressed = self.group_norm(compressed)
        
        # (B, C, Nr/compress_rt)
        bridge = self.bridge(compressed)
        # (B, C, Nr//compress_rt) -> (B, Nr//compress_rt, C)
        x = bridge.permute(0, 2, 1) 
        # (B, Nr//compress_rt, C)
        for i in range(self.num_layers):
            # (B, N//comress_rt, C_out) + (N//compress_rt, C_out) -> (B, N//compress_rt, C_out)
            x = x + self.pos_embed.unsqueeze(0).repeat(B, 1, 1)
            norm_x = self.norm_attn[i](x)
            x = self.attn[i](norm_x, norm_x, norm_x)[0] + x
            norm_x = self.norm_fpn[i](x)
            x = self.dropout(self.ffn[i](norm_x)) + x
            
        # (B, C, Nr//compress_rt)
        x = x.permute(0, 2, 1)
        x = self.group_norm_out(x)
        out = self.bridge_out(x)
        
        # (B, C, Nr//compress_rt) -> (B, C, Nr) -> (B, Nr, C)
        out = out.repeat(1, 1, self.compress_rt).permute(0, 2, 1)
        
        return out


# context-enhanced anchor by custom context extractor instead of torch.grid_sample function
@MODELS.register_module()
class ContextAnchorHead(BaseDenseHead):
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
        loss_dpp=None, # dpp loss for the distribution diversity
        train_cfg=None,
        test_cfg=None,
        use_segman_decoder=False,
        segman_decoder_params=None,
        context_num_layers=3,
    ):
        super(ContextAnchorHead, self).__init__()
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
        self.loss_dpp = MODELS.build(loss_dpp) if loss_dpp["loss_weight"] > 0 else None
        self.loss_iou = MODELS.build(loss_iou)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_segman_decoder = use_segman_decoder # whether to use segman decoder for multi-task segmentation head
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
            
        self.context_num_layers = context_num_layers # number of refine layers for context anchors
            
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

        reg_modules = list()
        cls_modules = list()
        
        reg_modules += [nn.Linear(self.prior_feat_channels + self.prior_feat_channels, self.fc_hidden_dim),
                        nn.ReLU(inplace=True)]
        cls_modules += [nn.Linear(self.prior_feat_channels + self.prior_feat_channels, self.fc_hidden_dim),
                        nn.ReLU(inplace=True)]
        
        for _ in range(num_fc-1):
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
        
        # context extractor -> (N, Nr, C_out, Np)
        # self.context_anchor = ContextExtractor(C_in=self.prior_feat_channels, C_out=self.prior_feat_channels, kr_size=3, stride=1, padding='same', align_corners=False)
        self.context_encoder = AnchorFeatAttn(in_dim=self.prior_feat_channels, hidden_dim=self.prior_feat_channels * 2, out_dim=self.prior_feat_channels, num_heads=2,
                                              dropout=0.1, compress_rt=4, num_anchors=self.num_priors, num_points=self.sample_points, 
                                              num_layers=self.context_num_layers)
        
        
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
        
        # instead of using torch.grid_sample() function, apply the convolution based feature extracting
        # output tensors are permuted to (B, C, Ns, 2) (from (B, C, Nr, Np) -> (B, Nr, C, Np))
        feature = F.grid_sample(batch_features, grid, align_corners=True).permute(
           0, 2, 1, 3
        )
        
        # (B, Nr, C)
        context_feature = self.context_encoder(torch.cat((feature, grid.permute(0, 1, 3, 2)), dim=2))
        
        # (B, Np, C, Ns) -> (B* Np, C, Ns, 1)
        feature = feature.reshape(
            batch_size * self.num_priors,
            self.prior_feat_channels,
            self.sample_points,
            1,
        )
        
        # this functions samples the values from the feature map at the prior points and returns the sampled values (pooled features)
        return feature, context_feature

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
        _, sampled_xs = self.anchor_generator.generate_anchors(
            self.anchor_generator.prior_embeddings.weight, # (Np, 3) -> (start of y, start of x, theta)
            self.prior_ys, # (Nr, ) -> constrained spacing of y coordinates (=priors)
            self.sample_x_indices, # (Ns, ) -> indices of sample points to sample which points are sampled
            self.img_w, # original img_w to calculate the x coordinates of anchors (in denormalized coordinates)
            self.img_h, # original img_h to calculate the y coordinates of anchors (in denormalized coordinates)
        )

        anchor_params = self.anchor_generator.prior_embeddings.weight.clone().repeat(
            batch_size, 1, 1
        )  # [B, Np, 3]
        # (num_priors, num_samples(=num_points by default)) -> (batch, num_priors, num_samples(=num_points by default))
        priors_on_featmap = sampled_xs.repeat(batch_size, 1, 1)

        predictions_list = []

        # iterative refine
        pooled_features_stages = []
        for stage in range(self.refine_layers):
            prior_xs = priors_on_featmap  # torch.flip(priors_on_featmap, dims=[2])  # [24, 192, 36]
            
            # 1. anchor ROI pooling
            # [B, C, H, W] X [B, Nr, Np] => [B * Nr, C, Np, 1]
            # output pooled features by given coordinates of prior points. x coordinates are generated from anchors and y coordinates are fixed which is initialized
            # as the self.prior_feat_ys (which is normalized y coordinates starting from 1.0 to 0.0 with 72 sample points linearly spaced)
            pooled_features, extra_features = self.pool_prior_features(feature_pyramid[stage], prior_xs)
            pooled_features_stages.append(pooled_features)

            # 2. ROI gather
            # pooled features [B * Nr, C, Np, 1] * stages
            # feature pyramid: [B, C, Hs, Ws] (s = 0, 1, 2)
            fc_features_attn = self.attention(
                pooled_features_stages, feature_pyramid, stage
            )  # [B, Nr, Ch], Ch: fc_hidden_dim
            
            fc_features = fc_features_attn.view(self.num_priors, batch_size, -1).reshape(
                batch_size * self.num_priors, self.fc_hidden_dim
            )  # [B * Nr, Ch]
            
            # expand anchor dimension and repeat as much as number of anchors.
            # reshape the feature to concatenate with fc_features
            # [B, dim] -> [B, 1, dim] -> [B, Nr, dim] -> [B * Nr, dim] 
            # dist_features = dist_features.unsqueeze(1).repeat(1, self.num_priors, 1).reshape(batch_size * self.num_priors, -1)
            B, S, D = extra_features.shape
            fc_features = torch.concat((fc_features, extra_features.reshape(B * S, D)), dim=-1)
            

            # 3. cls and reg heads
            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
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

            # 4. reg processing
            anchor_params += reg[:, :, :3]  # y0, x0, theta
            updated_anchor_xs, _ = self.anchor_generator.generate_anchors(
                anchor_params.view(-1, 3),
                self.prior_ys,
                self.sample_x_indices,
                self.img_w,
                self.img_h,
            )
            updated_anchor_xs = updated_anchor_xs.view(batch_size, self.num_priors, -1)
            reg_xs = updated_anchor_xs + reg[..., 4:]

            pred_dict = {
                "cls_logits": cls_logits,
                "anchor_params": anchor_params,
                "lengths": reg[:, :, 3:4],
                "xs": reg_xs,
            }
            
            if self.loss_dpp:
                pred_dict["dpp_features"] = fc_features_attn.reshape(batch_size, self.num_priors, -1)
                # pred_dict["dpp_features"] = dist_features.reshape(batch_size, self.num_priors, -1)
                # pred_dict["dpp_features"] = fc_features.reshape(batch_size, self.num_priors, -1) # concatenate of fc_features and dist_features
                
            predictions_list.append(pred_dict)

            if stage != self.refine_layers - 1:
                anchor_params = anchor_params.detach().clone()
                priors_on_featmap = updated_anchor_xs.detach().clone()[
                    ..., self.sample_x_indices
                ]

        return predictions_list

    def loss_by_feat(self, out_dict, batch_data_samples):
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
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = len(batch_data_samples)
        device = out_dict["predictions"][0]["cls_logits"].device
        cls_loss = torch.tensor(0.0).to(device)
        reg_xytl_loss = torch.tensor(0.0).to(device)
        iou_loss = torch.tensor(0.0).to(device)
        
        if self.loss_dpp:
            dpp_loss = torch.tensor(0.0).to(device)
        
        num_assignment = torch.tensor(0.0).to(device)
        total_assignment = torch.tensor(0.0).to(device)

        for stage in range(self.refine_layers):
            
            if self.loss_dpp:
                dpp_loss = dpp_loss + self.loss_dpp(out_dict["predictions"][stage]["dpp_features"])
                
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

                ## here assignment runs => assignment runs 'per image'
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

        loss_dict = {
            "loss_cls": cls_loss,
            "loss_reg_xytl": reg_xytl_loss,
            "loss_iou": iou_loss,
            # add raw scalars for logging
            "num_assignment": num_assignment.detach().float(),
            "total_assignment": total_assignment.detach().float(),
        }
        
        # extra dpp loss
        if self.loss_dpp:
            dpp_loss /= self.refine_layers
            loss_dict["loss_dpp"] = dpp_loss

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
        predictions = self(x)
        out_dict = {"predictions": predictions}

        if self.loss_seg:
            out_dict["seg"] = self.forward_seg(x)
            
        losses = self.loss_by_feat(out_dict, batch_data_samples)
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
