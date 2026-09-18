import math
import time
from typing import Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM as DWConv
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table
from mmcv.cnn import ConvModule
from mmengine.runner import load_state_dict, load_checkpoint
from natten import NeighborhoodAttention2D, use_fused_na, use_gemm_na
from natten.functional import na2d, na2d_av, na2d_qk, natten2dav, natten2dqkrpb
# from timm.models.vision_transformer import _cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from natten.flops import qk_2d_rpb_flop, av_2d_flop, add_natten_handle

try:
    from csm_triton import CrossScanTriton, CrossMergeTriton
except:
    from .csm_triton import CrossScanTriton, CrossMergeTriton

import selective_scan_cuda_oflex

from mmdet.registry import MODELS
import logging

logger = logging.getLogger(__name__)

# use_gemm_na(True)
# use_fused_na(True)

def get_continuous_paths(N):
    # Note that N is always even since we use image resolution of 256, 512, 1024 with the SD VAE encoder
    paths_lr = []
    reverse_lr = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (N - 1, 0, -1, 1),
    ]:
        path = lr_tranverse(N, start_row, start_col, dir_row, dir_col)
        paths_lr.append(path)
        reverse_lr.append(reverse_permut(path))
    

    paths_tb = []
    reverse_tb = []
    for start_row, start_col, dir_row, dir_col in [
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        path =tb_tranverse(N, start_row, start_col, dir_row, dir_col)
        paths_tb.append(path)
        reverse_tb.append(reverse_permut(path))
    
    
    return paths_lr, paths_tb, reverse_lr, reverse_tb
    

def lr_tranverse(N,start_row=0, start_col=0, dir_row=1, dir_col=1):
    path = []
    for i in range(N):
        for j in range(N):
            # If the row number is even, move right; otherwise, move left
            col = j if i % 2 == 0 else N - 1 - j
            path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
    return path

def tb_tranverse(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
    path = []
    for j in range(N):
        for i in range(N):
            # If the column number is even, move down; otherwise, move up
            row = i if j % 2 == 0 else N - 1 - i
            path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
    return path

def reverse_permut(permutation):
    n = len(permutation)
    reverse = [0] * n
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse

def cross_scan_continuous(x, num_scans =4, split=False):
    B, C, W, H = x.size()
    N= W

    if split and C>1:
        C = int(C/num_scans)
        split_indexes = [C,C,C,C]
        x1, x2, x3, x4 = torch.split(x, split_indexes, dim=1)

    xs = x.new_empty((B, num_scans, C, H * W))
    
    paths_lr, paths_tb, reverse_lr, reverse_tb = get_continuous_paths(N)
    paths_lr = torch.tensor(paths_lr, device=x.device, dtype=torch.long)
    paths_tb = torch.tensor(paths_tb, device=x.device, dtype=torch.long)
    reverse_lr = torch.tensor(reverse_lr, device=x.device, dtype=torch.long)
    reverse_tb = torch.tensor(reverse_tb, device=x.device, dtype=torch.long)
    
    if split and C>1:
        xs[:, 0] = torch.index_select(x1.flatten(-2,-1), -1, paths_lr[0])
        xs[:, 1] = torch.index_select(x2.flatten(-2,-1), -1, paths_lr[1])
        xs[:, 2] = torch.index_select(x3.flatten(-2,-1), -1, paths_tb[0])
        xs[:, 3] = torch.index_select(x4.flatten(-2,-1), -1, paths_tb[1])
    
    else:
        for i in range(paths_lr.size(0)):
            xs[:, i] = torch.index_select(x.flatten(-2,-1), -1, paths_lr[i])
            
        for i in range(paths_tb.size(0)):
            xs[:, i+num_scans//2] = torch.index_select(x.flatten(-2,-1), -1, paths_tb[i])
    
    return xs, paths_lr, paths_tb, reverse_lr, reverse_tb
    
def cross_merge_continuous(ys, paths_lr, paths_tb, reverse_lr, reverse_tb, split=False):
    B, K, D, H, W = ys.shape
    L = W*H

    ys = ys.view(B, K, D, -1)
    ys = ys.permute(0,2,1,3) # B, D, K, L

    if split:
        B, D, K, L = ys.size()
        return ys.reshape(B,D*K, L)
    
    corresponding_scan_paths = torch.concat([reverse_lr,reverse_tb], dim=0).view(1,1,K,L)
    corresponding_scan_paths = corresponding_scan_paths.repeat(B,D,1,1)
    y = torch.gather(ys, -1, corresponding_scan_paths)
    y = torch.sum(y,dim=2) # B, D, L
    
    return y    


def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def selective_scan_flop_jit(inputs, outputs):

    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)

    return flops


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    
    def forward(self, slen):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        return (sin, cos)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5, enable_bias=True):
        super().__init__()
        
        self.dim = dim
        self.init_value = init_value
        self.enable_bias = enable_bias
          
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x
    
    def extra_repr(self) -> str:
        return '{dim}, init_value={init_value}, bias={enable_bias}'.format(**self.__dict__)
    

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels):
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


def toodd(size):
    size = to_2tuple(size)
    if size[0] % 2 == 1:
        pass
    else:
        size[0] = size[0] + 1 
    if size[1] % 2 == 1:
        pass
    else:
        size[1] = size[0] + 1
    return size


class VSSM(nn.Module):

    def __init__(
        self,
        d_model=96,
        d_state=1,
        expansion_ratio=1,
        dt_rank="auto",
        norm_layer=LayerNorm2d,
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        k_groups=4,
        **kwargs,    
    ):
        
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.expansion_ratio = expansion_ratio
        if self.expansion_ratio !=1:
            self.xproj = nn.Linear(d_model, d_inner)
            self.yproj = nn.Linear(d_inner, d_model)
        # # # out proj =======================================
        # self.out_norm = norm_layer(d_inner)
        
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0).view(-1, d_inner, 1))
        del self.x_proj
        
        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_groups, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_groups, merge=True) # (K * D)
        
        # self.factor1 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
        # self.factor2 = nn.Parameter(torch.ones(d_inner, 1, 1), requires_grad=True)
            
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=None, backnrows=None, ssoflex=False):
        return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    def _cross_scan(self, x):
        return CrossScanTriton.apply(x)
    
    def _cross_merge(self, x):
        return CrossMergeTriton.apply(x)
    
    
    def forward(self, x, to_dtype=False, force_fp32=False):

        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        
        # xs = torch.stack([x, x.flip([-1])], dim=1).reshape(B, -1, L)
        xs = self._cross_scan(x)
        if self.expansion_ratio!=1:
            xs = self.xproj(xs.permute(0,1,3,2).contiguous()).permute(0,1,3,2).contiguous()
        xs = xs.reshape(B, -1, L)
        x_dbl = F.conv1d(xs, self.x_proj_weight, bias=None, groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), dt_projs_weight.reshape(K * D, -1, 1), groups=K)
        
        dts = dts.contiguous().reshape(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().reshape(B, K, N, L)
        Cs = Cs.contiguous().reshape(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.reshape(-1).to(torch.float)
              
        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)
                  
        ys = self._selective_scan(xs, dts, As, Bs, 
                                  Cs, Ds, delta_bias,
                                  delta_softplus=True,
                                  ssoflex=True)
        
        # y = ys.reshape(B, K, -1, L)
        # yf = F.conv1d(y[:, 0, ...], weight=self.factor1, groups=D)
        # yb = F.conv1d(y[:, 1, ...].flip([-1]), weight=self.factor2, groups=D)
        # y = yf + yb
        y = self._cross_merge(ys.reshape(B, K, -1, H, W)).reshape(B, -1, H, W)
        
        if self.expansion_ratio!=1:
            y = self.yproj(y.permute(0,3,2,1).contiguous()).permute(0,3,2,1).contiguous()

        if to_dtype:
            y = y.to(x.dtype)
        
        return y

class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0,
        kernel_size=3
    ): 
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        padding = kernel_size // 2
        
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=kernel_size, padding=padding, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class VSSMBlock(nn.Module):

    def __init__(self,
                 embed_dim=64,
                 expansion_ratio=1,
                 channel_split=False,
                 drop_path=0, 
                 layerscale=False, 
                 layer_init_values=1e-6,
                 token_mixer=VSSM,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d):
        # retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(embed_dim)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.token_mixer = token_mixer(d_model=embed_dim,
                                    k_groups=4, 
                                    expansion_ratio=expansion_ratio,
                                    channel_split=channel_split,)
        self.norm2 = norm_layer(embed_dim)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.mlp = channel_mixer(embed_dim, embed_dim*4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x):
        x = x + self.cpe1(x)
        token_mix_feat = self.token_mixer(self.norm1(x))
        x = x + self.drop_path(self.layer_scale1(token_mix_feat))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
            
        return x
    

import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS

class RowColEnhance(nn.Module):
    def __init__(self, in_channels, proj_dim, feat_size, num_heads=4):
        super(RowColEnhance, self).__init__()
        self.in_channels = in_channels
        self.proj_dim = proj_dim
        self.feat_h = feat_size[0]
        self.feat_w = feat_size[1]
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads # projection dimension per head
        
        self.row_conv = nn.Conv2d(in_channels, proj_dim, kernel_size=(1, self.feat_w), stride=1, padding=0, bias=False)
        self.col_conv = nn.Conv2d(in_channels, proj_dim, kernel_size=(self.feat_h, 1), stride=1, padding=0, bias=False)
        
        self.conv_dw_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_dim, dilation=1)
        self.conv_pw_v = nn.Conv2d(in_channels, proj_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.conv_o = nn.Conv2d(proj_dim, proj_dim, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.linear_qk = nn.Linear(1, self.head_dim, bias=False) # linear projection of attention map to high-dimensional space
        self.linear_qk.apply(lambda x: nn.init.constant_(x.weight, 1.0)) # initialize to identity mapping
        
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).
        Returns:
            torch.Tensor: Enhanced feature map of shape (B, proj_dim, H, W).
        """
        # row-extracted feature (as Query)
        row_feat = self.row_conv(x) # (B, proj_dim, H, 1)
        row_feat = row_feat.reshape(-1, self.num_heads, self.head_dim, self.feat_h, 1) # (B, num_heads, head_dim, H, 1)
        row_feat = row_feat.squeeze(-1) # (B, num_heads, head_dim, H)
        row_feat = row_feat.permute(0, 1, 3, 2) # (B, num_heads, H, head_dim)
        
        # Column-wise enhancement # (as Key)
        col_feat = self.col_conv(x) # (B, proj_dim, 1, W)
        col_feat = col_feat.reshape(-1, self.num_heads, self.head_dim, 1, self.feat_w) # (B, num_heads, head_dim, 1, W)
        col_feat = col_feat.squeeze(-2) # (B, num_heads, head_dim, W)
        
        rc_attn = row_feat @ col_feat # (B, num_heads, H, W)
        rc_attn = F.softmax(rc_attn, dim=-1)
        # Linear projection of attention map
        rc_attn = self.linear_qk(rc_attn.unsqueeze(-1)) # (B, num_heads, H, W, self.head_dim)
        rc_attn = rc_attn.permute(0, 1, 4, 2, 3) # (B, num_heads, self.head_dim, H, W)
        rc_attn = rc_attn.reshape(-1, self.num_heads * self.head_dim, self.feat_h, self.feat_w) # (B, proj_dim, H, W)

        # Value projection
        value_feat = self.conv_dw_v(x) # (B, in_channels, H, W)
        value_feat = self.conv_pw_v(value_feat) # (B, proj_dim, H, W)
        
        # element-wise multiplication of attention map instead of matrix multiplication
        # This is equivalent to applying the grouped (=as num_heads) re-weighting to the value projected features
        enhanced_feat = value_feat * rc_attn
        enhanced_feat = self.conv_o(enhanced_feat) # (B, proj_dim, H, W)
        
        return enhanced_feat
    
class RowColEnhanceModule(nn.Module):
    def __init__(self, downscale, upscale, in_channels, proj_dim, feat_size):
        super(RowColEnhanceModule, self).__init__()
        self.downscale = downscale # pixel unshuffle downscale factor
        self.upscale = upscale
        self.rowcol_attn = RowColEnhance(in_channels, proj_dim, feat_size)
        self.feat_size = feat_size
        if downscale > 1:
            self.downscaler = nn.Sequential(
                nn.PixelUnshuffle(downscale),
                nn.Conv2d(in_channels=in_channels * int(downscale**2), out_channels=in_channels, kernel_size=1, stride=1, padding=0)
            )
        if upscale > 1:
            self.upscaler = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels * int(self.upscale ** 2), kernel_size=1, stride=1, padding=0),
                nn.PixelShuffle(self.upscale)
            )
        
    
    def forward(self, x):
        x = self.downscaler(x) if self.downscale > 1 else x
        x = self.rowcol_attn(x)
        
        x = self.upscaler(x) if self.upscale > 1 else x
        
        return x

@MODELS.register_module()
class SegMANDecoderFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, feat_sizes, upsample_t=(40, 100), dsample_t=(10, 25)):
        """
        Feature pyramid network for CLRerNet.
        Row-Col Enhanced Feature Pyramid network.
        Args:
            in_channels (List[int]): Channel number list.
            out_channels (int): Number of output feature map channels.
            num_outs (int): Number of output feature map levels.
        """
        super(SegMANDecoderFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.feat_sizes = feat_sizes # list of (H, W) for each level ex) (80, 200) -> (40, 100) -> (20, 50) -> (10, 25)
        self.upsample_t = upsample_t # upsampling-target
        self.dsample_t = dsample_t   # downsampling-target
        self.msfp_dim = out_channels

        self.backbone_end_level = self.num_ins
        self.start_level = 0
        
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            
        self.aggregate_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, self.msfp_dim, kernel_size=1, stride=1, padding=0, bias=False),
            LayerNorm2d(self.msfp_dim), #nn.GroupNorm(4, self.msfp_dim),
            nn.ReLU()
        )
        
        # row-col enhancement (row-col attention)
        self.feat_rowcol_attn = RowColEnhance(self.msfp_dim, self.msfp_dim, feat_sizes[0])
        self.feat_rowcol_ffn = FFN(embed_dim=self.msfp_dim, ffn_dim=self.msfp_dim * 2)
        
        self.feat_vssm = VSSMBlock(embed_dim=self.msfp_dim, expansion_ratio=1,
                                   channel_split=False, drop_path=0, layerscale=False,
                                   layer_init_values=1e-6,
                                   token_mixer=VSSM, channel_mixer=FFN,
                                   norm_layer=LayerNorm2d)
        
        self.feat_dsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(kernel_size=(3+2*i), stride=(2**i), padding=(i+1), in_channels=self.msfp_dim, out_channels=self.out_channels, bias=False),
                LayerNorm2d(self.out_channels), #nn.GroupNorm(4, self.out_channels),
                nn.ReLU()
            ) for i in range(3)
        ])
        
        # multi-scale feature processing
        self.msfp_dsample_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(kernel_size=3, stride=2, padding=1, in_channels=self.msfp_dim, out_channels=self.msfp_dim * 2, bias=False),
                LayerNorm2d(self.msfp_dim * 2), #nn.GroupNorm(4, self.msfp_dim * 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(kernel_size=5, stride=4, padding=2, in_channels=self.msfp_dim, out_channels=self.msfp_dim * 4, bias=False),
                LayerNorm2d(self.msfp_dim * 4), #nn.GroupNorm(4, self.msfp_dim * 4),
                nn.ReLU()    
            )])
        
        self.msfp_pixel_unshuffle = nn.ModuleList([
            nn.PixelUnshuffle(4), # 4x downsample
            nn.PixelUnshuffle(2) # 2x downsample
        ])
        self.msfp_lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.msfp_dim * 4 * (2**(2-i)), out_channels=self.msfp_dim, kernel_size=1, stride=1, padding=0, bias=False),
                LayerNorm2d(self.msfp_dim), #nn.GroupNorm(4, self.msfp_dim),
                nn.ReLU()
            ) for i in range(3)
        ])
        
        # row-col enhancement (row-col attention)
        self.msfp_rowcol_attn = RowColEnhance(self.msfp_dim * 3, self.msfp_dim * 3, feat_sizes[-1])
        self.msfp_rowcol_ffn = FFN(embed_dim=self.msfp_dim * 3, ffn_dim=self.msfp_dim * 3 * 2)
        
        # after residual connection, concatenate the feature maps through channel-dim
        # then apply SS2D module + FFN
        self.msfp_vssm = VSSMBlock(embed_dim=self.msfp_dim * 3, expansion_ratio=1,
                                   channel_split=False, drop_path=0, layerscale=False,
                                   layer_init_values=1e-6,
                                   token_mixer=VSSM, channel_mixer=FFN,
                                   norm_layer=LayerNorm2d)
        
        self.msfp_out_proj = nn.ModuleList([
            nn.Conv2d(in_channels=self.msfp_dim, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=self.msfp_dim, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=self.msfp_dim, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0)
        ])
        
        self.rowcol_block = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        RowColEnhanceModule(
                            downscale=1,
                            upscale=1,
                            in_channels=self.out_channels,
                            proj_dim=self.out_channels,
                            feat_size=feat_sizes[i]
                        ),
                        FFN(
                            embed_dim=self.out_channels,
                            ffn_dim=self.out_channels * 4,
                            kernel_size=1 + 4 * (2-i),
                        )
                    ]
                ) for i in range(3)
            ] 
        )
    
    def forward_lateral(self, inputs):
        if isinstance(inputs, tuple):
            inputs = list(inputs)

        assert len(inputs) >= len(self.in_channels)  # 4 > 3

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals [1, 128, 40, 100], [1, 256, 20, 50], [1, 512, 10, 25]
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        return laterals
    
    def forward_msfp(self, feats):
        
        # multi-scale feature processing
        msfp_down_l1 = self.msfp_dsample_conv[1](feats) # 4x downsample
        msfp_feat_l1 = self.msfp_lateral_convs[2](msfp_down_l1) # 4C -> C
        
        msfp_down_l2 = self.msfp_dsample_conv[0](feats) # 2x downsample
        msfp_feat_l2 = self.msfp_pixel_unshuffle[1](msfp_down_l2) # 2x downsample
        msfp_feat_l2 = self.msfp_lateral_convs[1](msfp_feat_l2) # 8C -> C
        
        msfp_feat_l3 = self.msfp_pixel_unshuffle[0](feats) # 4x downsample
        msfp_feat_l3 = self.msfp_lateral_convs[0](msfp_feat_l3) # 16C -> C
        
        msfp_feat = torch.cat([msfp_feat_l1, msfp_feat_l2, msfp_feat_l3], dim=1) # (B, C*3, H, W)
        
        # apply row-col attention + FFN
        rowcol_enhanced = self.msfp_rowcol_attn(msfp_feat)
        msfp_feat = rowcol_enhanced + msfp_feat # residual connection
        
        msfp_feat_ffn = self.msfp_rowcol_ffn(msfp_feat)
        msfp_feat = msfp_feat + msfp_feat_ffn # residual connection
        
        # apply VSSM module
        msfp_feat = self.msfp_vssm(msfp_feat) # (B, C*3, H, W)
        
        splited_tensor = torch.split(msfp_feat, msfp_feat.shape[1]//3, dim=1) # split tensor to [C, C, C] channel feature map
        
        # upsample to the feat_size[0] ex) (40, 100)
        out_l1 = F.interpolate(splited_tensor[0], size=self.feat_sizes[2], mode='bilinear', align_corners=False) # (10, 25)
        out_l2 = F.interpolate(splited_tensor[1], size=self.feat_sizes[1], mode='bilinear', align_corners=False) # (20, 50)
        out_l3 = F.interpolate(splited_tensor[2], size=self.feat_sizes[0], mode='bilinear', align_corners=False) # (40, 100)
        
        # project to output channels
        out_l1 = self.msfp_out_proj[0](out_l1) # (B, C, 10, 25)
        out_l2 = self.msfp_out_proj[1](out_l2) # (B, C, 20, 50)
        out_l3 = self.msfp_out_proj[2](out_l3) # (B, C, 40, 100)
        
        return out_l3, out_l2, out_l1
        
        
    def forward(self, inputs):
        """
        Args:
            inputs (List[torch.Tensor]): Input feature maps.
              Example of shapes:
                ([1, 64, 80, 200], [1, 128, 40, 100], [1, 256, 20, 50], [1, 512, 10, 25]).
        Returns:
            outputs (Tuple[torch.Tensor]): Output feature maps.
              The number of feature map levels and channels correspond to
               `num_outs` and `out_channels` respectively.
              Example of shapes:
                ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        """
        
        laterals = self.forward_lateral(inputs)      
        
        feats = [
            F.interpolate(
                laterals[i],
                size=(self.feat_sizes[0]),
                mode='bilinear',
                align_corners=False
            ) for i in range(len(laterals))
        ]
        
        concat_feat = torch.cat(feats, dim=1) # (B, C*3, H, W)
        aggregated_feat = self.aggregate_conv(concat_feat)
        
        msfp_feat = self.forward_msfp(aggregated_feat)
        
        # row-col attention + FFN
        rowcol_enhanced = self.feat_rowcol_attn(aggregated_feat)
        aggregated_feat = rowcol_enhanced + aggregated_feat # residual connection
        rowcol_enhanced_ffn = self.feat_rowcol_ffn(aggregated_feat)
        aggregated_feat = aggregated_feat + rowcol_enhanced_ffn # residual connection
        
        # apply VSSM block to the concatenated features (VSSM= SS2D + FFN)
        vssm_feat = self.feat_vssm(aggregated_feat)
        
        # apply downsample to the concatenated features
        vssm_feats = [
            self.feat_dsample[i](vssm_feat) for i in range(len(self.feat_dsample))
        ]
                
        # add multi-scale feature processing results to each level
        for i in range(len(feats)):
            # add msfp_feat, vssm_feats to laterals (Rol-Col Attn + SSM enhanced high-resolution path + MSPE Low-resolution path)
            laterals[i] = laterals[i] + msfp_feat[i] + vssm_feats[i]
        
        # apply row-col attention + FFN
        for i in range(len(feats)):
            rowcol_attn = self.rowcol_block[i][0](laterals[i]) # apply row-col attention
            laterals[i] = laterals[i] + rowcol_attn # residual connection
            rowcol_ffn = self.rowcol_block[i][1](laterals[i]) # apply FFN
            laterals[i] = laterals[i] + rowcol_ffn # residual connection
            
        return tuple(laterals)  