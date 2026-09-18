# globally structured (LASS) row-column attention to extract the global feature from the feature map and feed it to the head.
# different from the rc_former design, added modules are not located in the FPN path.

# structured means row-column wise spatial prior in attention
# global means the global feature extractor like ViT style attention or deformable attention or LASS (SSM + attention)

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM as DWConv
from einops import einsum, rearrange, repeat
from mmcv.cnn import ConvModule
from natten.functional import na2d, na2d_av, na2d_qk
from timm.models.layers import DropPath, to_2tuple

try:
    from csm_triton import CrossScanTriton, CrossMergeTriton
except:
    from .csm_triton import CrossScanTriton, CrossMergeTriton

import selective_scan_cuda_oflex

from mmdet.registry import MODELS
import logging

logger = logging.getLogger(__name__)


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
    

# channel mixer
class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.ReLU,
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

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels):
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)


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


# Layer normalization for 2D feature maps
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
        ssm_split=False,
        **kwargs,    
    ):
        
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(expansion_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.expansion_ratio = expansion_ratio
        self.ssm_split = ssm_split
        if self.ssm_split:
            d_inner = int(d_inner/4)
            self.yproj = nn.Linear(d_model,d_model)

        if self.expansion_ratio != 1.0 :
            self.proj = nn.Linear(d_model,d_inner)
            self.yproj = nn.Linear(d_inner,d_model)


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
        if not self.ssm_split:
            return CrossScanTriton.apply(x), None, None, None, None
        else:
            return cross_scan_continuous(x, split=True)
    
    def _cross_merge(self, x, paths_lr=None, paths_tb=None, reverse_lr=None, reverse_tb=None):
        if not self.ssm_split:
            return CrossMergeTriton.apply(x)
        else:
            return cross_merge_continuous(x, paths_lr, paths_tb, reverse_lr, reverse_tb, split=True)
    
    
    def forward(self, x, to_dtype=False, force_fp32=False):
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        
        if self.expansion_ratio != 1.0:
            x = self.proj(x.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
            xs, paths_lr, paths_tb, reverse_lr, reverse_tb = self._cross_scan(x) # size b, 4, embed_dim, L
            xs = xs.reshape(B, -1, L).contiguous()
        else:
            xs, paths_lr, paths_tb, reverse_lr, reverse_tb = self._cross_scan(x)
            xs = xs.reshape(B, -1, L).contiguous()
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
        
        y = self._cross_merge(ys.reshape(B, K, -1, H, W).contiguous(), paths_lr, paths_tb, reverse_lr, reverse_tb).reshape(B, -1, L).contiguous()

        if self.ssm_split:
            y = self.yproj(y.permute(0,2,1)).permute(0,2,1) # mix channel

        if self.expansion_ratio != 1.0:
            y = self.yproj(y.permute(0,2,1)).permute(0,2,1)
        
        if to_dtype:
            y = y.to(x.dtype)
        
        return y
     

class Attention(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 window_size, 
                 window_dilation, 
                 global_mode=False, 
                 image_size=None, 
                 use_rpb=False, 
                 sr_ratio=1,
                 fused_na=True,
                 ssm_ratio=1,
                 ssm_split=False):
        
        super().__init__()
        window_size = to_2tuple(window_size)
        window_dilation = to_2tuple(window_dilation)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.window_dilation = window_dilation
        self.global_mode = global_mode
        self.sr_ratio = sr_ratio
        self.image_size = image_size
        self.fused_na = fused_na
        
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1)
        self.lepe = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        if not global_mode:
            self.dwconv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
                nn.BatchNorm2d(embed_dim), # change BatchNorm2d to LayerNorm2d
                # LayerNorm2d(embed_dim)
            )
            self.ssm = VSSM(d_model=embed_dim, expansion_ratio=ssm_ratio, ssm_split=ssm_split)
            self.norm = LayerNorm2d(embed_dim)
            
        if use_rpb:
            rpb_list = [nn.Parameter(torch.empty(num_heads, (2 * window_size[0] - 1), (2 * window_size[1] - 1)), requires_grad=True)]
            if global_mode: 
                rpb_list.append(nn.Parameter(torch.empty(num_heads, image_size[0]*image_size[0], image_size[1]*image_size[1]), requires_grad=True))
            self.rpb = nn.ParameterList(rpb_list)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.qkv.weight, gain=2**-2.5)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_normal_(self.proj.weight, gain=2**-2.5)
        nn.init.zeros_(self.proj.bias)
        if hasattr(self, 'rpb'):
            for item in self.rpb:
                nn.init.zeros_(item) # which better? nn.init.trunc_normal_(item, std=0.02)
    
    
    def forward(self, x, pos_enc):
        
        B, C, H, W = x.shape
        
        qkv = self.qkv(x)
        lepe = self.lepe(qkv[:, -C:, ...])
        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b n h w c', m=3, n=self.num_heads)
        
        sin, cos = pos_enc
        q = theta_shift(q, sin, cos) * self.scale
        k = theta_shift(k, sin, cos)
        
        if hasattr(self, 'rpb'):
            rpb = self.rpb[0]
        else:
            rpb = None
    
        if self.fused_na:
            q = rearrange(q, 'b n h w c -> b h w n c')
            k = rearrange(k, 'b n h w c -> b h w n c')
            v = rearrange(v, 'b n h w c -> b h w n c')

            x = na2d(q, k, v, kernel_size=toodd(self.window_size), dilation=self.window_dilation, scale=float(q.size(-1)**0.5))
            q = rearrange(q, 'b h w n c -> b n h w c')
            k = rearrange(k, 'b h w n c -> b n h w c')
            x = rearrange(x, 'b h w n c -> b n h w c')

        else:
            attn = na2d_qk(q, k, kernel_size=toodd(self.window_size), dilation=self.window_dilation, rpb=rpb)
            attn = torch.softmax(attn, dim=-1) # b, h, h, w, k^2
            x = na2d_av(attn, v, kernel_size=toodd(self.window_size), dilation=self.window_dilation)
        
        if not self.global_mode:
            
            q = rearrange(q, 'b n h w c -> b n c h w').contiguous()
            k = rearrange(k, 'b n h w c -> b n c h w').contiguous()
            v = rearrange(x, 'b n h w c -> b n c h w').contiguous()
            
            v_r = v.flatten(1, 2)
            v = self.dwconv(v_r)
            v = F.silu(v)
            v = self.ssm(v)
            
            v = self.norm(v.reshape(B, -1, H, W).contiguous())
        
            x = v + v_r

        else:
            
            q = rearrange(q, 'b n h w c -> b n (h w) c')
            k = rearrange(k, 'b n h w c -> b n (h w) c')
            v = rearrange(x, 'b n h w c -> b n (h w) c')
            
            attn = einsum(q, k, 'b n l c, b n m c -> b n l m')
            
            if hasattr(self, 'rpb'):
                if attn.size(-1) != self.rpb[-1].size(1) or x.size(-2) != self.rpb[-1].size(2):
                    attn = attn + F.interpolate(self.rpb[-1].unsqueeze(0), size=attn.shape[2:], mode='bicubic', align_corners=False)
                else:
                    attn = attn + self.rpb[-1]
                
            attn = torch.softmax(attn, dim=-1)  
            x = einsum(attn, v, 'b n l m, b n m c -> b n c l').reshape(B, -1, H, W).contiguous()
        
        x = x + lepe
        x = self.proj(x)
        
        return x


# From SegMAN Encoder
class Block(nn.Module):

    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 num_heads=2, 
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=256, 
                 drop_path=0, 
                 layerscale=False, 
                 layer_init_values=1e-6,
                 token_mixer=Attention,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d,
                 fused_na=False,
                 ssm_ratio=1.0,
                 ssm_split=False):
        # retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim

        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.token_mixer = token_mixer(embed_dim, num_heads, window_size, window_dilation, global_mode, image_size, use_rpb, sr_ratio,
        ssm_ratio=ssm_ratio,ssm_split=ssm_split, fused_na=fused_na)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = channel_mixer(embed_dim, ffn_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x, pos_enc):
        
        x = x + self.cpe1(x)
        x = x + self.drop_path(self.layer_scale1(self.token_mixer(self.norm1(x), pos_enc)))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))  
            
        return x
    

# From SegMAN Encoder
class BasicLayer_Norm(nn.Module):
    
    def __init__(self,
                 image_size=None,
                 embed_dim=64, 
                 depth=4, 
                 num_heads=4,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=96, 
                 drop_path=0,
                 layerscale=False, 
                 layer_init_values=1e-6,
                 norm_layer=LayerNorm2d,
                 use_checkpoint=0,
            ):

        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # self.RoPE = RoPE(embed_dim, num_heads)

        self.rope = RoPE(embed_dim, num_heads)
        self.norm = norm_layer(embed_dim)

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(embed_dim=embed_dim,
                          num_heads=num_heads,
                          window_size=window_size,
                          window_dilation=window_dilation,
                          global_mode=global_mode,
                          ffn_dim=ffn_dim,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layerscale=layerscale,
                          layer_init_values=layer_init_values,
                          norm_layer=norm_layer,
                          image_size=image_size,
                          use_rpb=use_rpb,
                          sr_ratio=sr_ratio,
            )
            self.blocks.append(block)

    def forward(self, x):
        pos_enc = self.rope((x.shape[2:]))
        for i, blk in enumerate(self.blocks):
            if i < self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(blk, x, pos_enc, use_reentrant=False)
            else:
                x = blk(x, pos_enc)
        return self.norm(x)



# Row-Column Attention Module (token mixer)
class RowColumnAttention(nn.Module):
    """
        in_channels: number of input feature channels
        proj_dim: projected dimension for row/column tokens
        feat_size: (H, W) size of the input feature map
    """
    def __init__(self, in_channels, proj_dim, feat_size):
        super(RowColumnAttention, self).__init__()
        self.in_channels = in_channels
        self.proj_dim = proj_dim
        self.scale = proj_dim**0.5 # scaling factor for attention map

        self.feat_h = feat_size[0]
        self.feat_w = feat_size[1]
        
        self.row_qkv = nn.Conv2d(in_channels, proj_dim * 3, kernel_size=(1, self.feat_w), stride=1, padding=(0, 0), bias=True)
        self.col_qkv = nn.Conv2d(in_channels, proj_dim * 3, kernel_size=(self.feat_h, 1), stride=1, padding=(0, 0), bias=True)

        self.crow_qkv = nn.Linear(proj_dim, proj_dim * 3, bias=True)
        self.ccol_qkv = nn.Linear(proj_dim, proj_dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        
        self.fuse_and_refine = nn.Sequential(
            nn.Conv2d(proj_dim * 2, proj_dim * 4, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(proj_dim * 4, in_channels, kernel_size=1, stride=1)
        )
    def attention(self, q, k, v):
        # q,k,v: (B, Seq_len, C)    
        qk_attn = q @ k.permute(0, 2, 1)
        attn_map = self.softmax(qk_attn / self.scale)
        attn = attn_map @ v
        return attn
    
    def forward(self, x):
        # x: (B, C, H, W)
        row_t = self.row_qkv(x).squeeze().permute(0, 2, 1)            # (B, 3*proj_dim, H) -> (B, H, 3*proj_dim)
        col_t = self.col_qkv(x).squeeze().permute(0, 2, 1)            # (B, 3*proj_dim, W) -> (B, W, 3*proj_dim)
        row_tq, row_tk, row_tv = torch.chunk(row_t, chunks=3, dim=-1) # each: (B, H, proj_dim)
        col_tq, col_tk, col_tv = torch.chunk(col_t, chunks=3, dim=-1) # each: (B, W, proj_dim)

        # self-attention for row, column tokens
        row_t = self.attention(row_tq, row_tk, row_tv)
        col_t = self.attention(col_tq, col_tk, col_tv)
        
        crow_t = self.crow_qkv(row_t)
        ccol_t = self.ccol_qkv(col_t)
        crow_tq, crow_tk, crow_tv = torch.chunk(crow_t, chunks=3, dim=-1) # each: (B, H, proj_dim)
        ccol_tq, ccol_tk, ccol_tv = torch.chunk(ccol_t, chunks=3, dim=-1)

        # cross attention for mixture of row, column tokens
        r2c_t = self.attention(crow_tq, ccol_tk, ccol_tv) # (B, H, C)
        c2r_t = self.attention(ccol_tq, crow_tk, crow_tv) # (B, W, C)
        
        # reconstruct the spatial attention map
        r_spatial = r2c_t.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, self.feat_w)  # (B, C, H, 1) -> (B, C, H, W)
        c_spatial = c2r_t.permute(0, 2, 1).unsqueeze(-2).expand(-1, -1, self.feat_h, -1)  # (B, C, 1, W) -> (B, C, H, W)

        rc_spatial = torch.cat([r_spatial, c_spatial], dim=1) # (B, 2C, H, W)
        out = self.fuse_and_refine(rc_spatial) # (B, in_channels, H, W)
                
        return out

# attention block (token mixer + channel mixer)
class RowColumnBlock(nn.Module):
    def __init__(self, in_channels, proj_dim, feat_size, attn_drop, ffn_dim, ffn_act, ffn_drop):
        super(RowColumnBlock, self).__init__()
        self.in_channels = in_channels
        self.proj_dim = proj_dim
        self.feat_size = feat_size
        self.attn_drop = attn_drop
        self.ffn_dim = ffn_dim
        self.ffn_act = ffn_act
        self.ffn_drop = ffn_drop
        
        self.rc_attn = RowColumnAttention(in_channels=self.in_channels, 
                                                  proj_dim=self.proj_dim, feat_size=feat_size)
        self.ffn = FFN(embed_dim=proj_dim, ffn_dim=ffn_dim, 
                       act_layer=ffn_act, kernel_size=3)
        
        self.layer_norm1 = LayerNorm2d(proj_dim)
        self.layer_norm2 = LayerNorm2d(proj_dim)
        self.attn_dropout = nn.Dropout2d(attn_drop) if attn_drop > 0 else nn.Identity()
        self.ffn_dropout = nn.Dropout2d(ffn_drop) if ffn_drop > 0 else nn.Identity()
         
    def forward(self, x):
        x_norm = self.layer_norm1(x)
        x = x_norm + self.attn_dropout(self.rc_attn(x_norm))

        x_norm = self.layer_norm2(x)
        x = x_norm + self.ffn_dropout(self.ffn(x_norm))
        
        return x

# stack of row-column attention blocks
class RowColAttnLayer(nn.Module):
    def __init__(self, in_channels, proj_dim, feat_size, attn_drop, ffn_dim, ffn_act, ffn_drop, num_layers):
        super(RowColAttnLayer, self).__init__()
        self.num_layers = num_layers
    
        self.attns = nn.ModuleList()
        for _ in range(num_layers):
            self.attns.append(RowColumnBlock(in_channels, proj_dim, feat_size,
                                              attn_drop, ffn_dim, ffn_act, ffn_drop))
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.attns[i](x)
        return x

@MODELS.register_module()
class GSRCNeck(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, proj_dim, feat_size, ffn_dim, ffn_drop, ffn_kers, num_layers):
        """
        Feature pyramid network for CLRerNet with extra global feature extractor.
        The module use the same FPN design with default clrernet. The difference with default FPN is that
        row-column attention and LASS (ssm + attention) is used to further provide the global features into head.
        
        Args:
            in_channels (List[int]): Channel number list.
            out_channels (int): Number of output feature map channels.
            num_outs (int): Number of output feature map levels.
            proj_dim (int): Projection dimension for input of row-column attention.
            feat_size (tuple): feature map size for bottom or top pyramid level.
            ffn_dim (int): feed-forward dimension for the FFN block in Row-Column Attention block.
            ffn_drop (float): float for dropout in FFN module.
            ffn_kers (int): kernel size for feed-forward block.
            num_layers (int): number of layers for row-column attention blocks.
        """
        super(GSRCNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.proj_dim = proj_dim
        self.feat_size = feat_size
        self.ffn_dim = ffn_dim
        self.ffn_drop = ffn_drop
        self.ffn_kers = ffn_kers
        self.num_layers = num_layers

        self.backbone_end_level = self.num_ins
        self.start_level = 0
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # default fpn design (same as clrernet fpn)
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
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        # global feature extractor (row-column attention layer: stack of row-column attention blocks)
        global_structure_encoder = []
        global_structure_encoder.append(nn.Conv2d(in_channels=in_channels[-1], out_channels=proj_dim, kernel_size=1, bias=True, stride=1, padding=0))
        global_structure_encoder.append(RowColAttnLayer(in_channels=proj_dim, proj_dim=proj_dim, feat_size=feat_size, attn_drop=0.1, ffn_dim=ffn_dim,
                                                ffn_act=nn.ReLU, ffn_drop=ffn_drop, num_layers=num_layers))
        self.global_structure_encoder = nn.Sequential(*global_structure_encoder)
        
        global_feature_encoder = []
        global_feature_encoder.append(nn.Conv2d(in_channels=in_channels[-1], out_channels=proj_dim, kernel_size=1, bias=True, stride=1, padding=0))
        global_feature_encoder.append(BasicLayer_Norm(
                embed_dim=proj_dim, depth=num_layers, num_heads=2, window_size=7, window_dilation=1, 
                global_mode=False, use_rpb=False, sr_ratio=1, ffn_dim=ffn_dim, drop_path=0.0, layerscale=True,
                layer_init_values=1e-6, norm_layer=LayerNorm2d, use_checkpoint=0)) 
        self.global_feature_encoder = nn.Sequential(*global_feature_encoder)
        
        self.attention = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, dropout=0.0, bias=True, batch_first=True)
    
    def _init_base(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if getattr(m, 'weight', None) is not None:
                nn.init.constant_(m.weight, 1.0)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0.0)

    def init_weights(self):
        self.apply(self._init_base)
    
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
        if isinstance(inputs, tuple):
            inputs = list(inputs)

        assert len(inputs) >= len(self.in_channels)  # 4 > 3

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest'
            )

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        
        # output same feature map size as given feat_size.
        # outputs (B, 512, 10, 25). this will be used as feature map of attention layer in head.
        structured_attn = self.global_structure_encoder(inputs[-1])
        context_attn = self.global_feature_encoder(inputs[-1])
        
        B, C, H, W = structured_attn.shape
        
        structured_attn = structured_attn.reshape(B, C, -1).permute(0, 2, 1)
        context_attn = context_attn.reshape(B, C, -1).permute(0, 2, 1)
        
        gsrc = self.attention(structured_attn, context_attn, context_attn)
        
        gsrc = gsrc[0].permute(0, 2, 1).reshape(B, C, H, W)
        
        outs.append(gsrc)
        
        return tuple(outs)




