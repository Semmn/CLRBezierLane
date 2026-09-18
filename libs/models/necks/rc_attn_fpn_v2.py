import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
import torchvision

# channel mixer
class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.ReLU,
        kernel_size=3,
    ): 
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        padding = kernel_size // 2
        
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=kernel_size, padding=padding, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.fc2(x)
        return x

# Layer normalization for 2D feature maps
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    
    def forward(self, x):           # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)   # x: (B, H, W, C)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)   # x: (B, C, H, W)
        return x.contiguous()


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
class RowColAttnFPNV2(nn.Module):
    def __init__(self, in_channels, fpn_channels, rc_layers, feat_size=[(40, 100), (20, 50), (10, 25)]):
        """
        V2 version of feature pyramid network for CLRerNet. Additional feature map output supported.
        It must be additional feature map output must be appended at the end of main output list.
        Args:
            in_channels (List[int]): Channel number list. (from shallow to deeper)
            fpn_channels (int): Number of channels for FPN feature maps.
            rc_layers (int): Number of row-column attention layers per feature level.
            feat_size (List[Tuple[int]]): Feature map sizes for each input level. default values are for (800x320) input image.
        """
        super(RowColAttnFPNV2, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.backbone_end_level = len(in_channels)
        self.fpn_channels = fpn_channels
        self.rc_layers = rc_layers
        self.feat_size = feat_size

        self.start_level = 0
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.rc_attns = RowColAttnLayer(in_channels=fpn_channels, proj_dim=fpn_channels, feat_size=self.feat_size[-1], 
                            attn_drop=0.1, ffn_dim=fpn_channels*4, ffn_act=nn.ReLU, ffn_drop=0.1, num_layers=self.rc_layers)

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                fpn_channels,
                1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )
            fpn_conv = ConvModule(
                fpn_channels,
                fpn_channels,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

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
               `num_outs` and `fpn_channels` respectively.
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

        # apply row-column attention at the end of top level feature map
        rc_feature = self.rc_attns(laterals[-1])
        outs.append(rc_feature)
        return tuple(outs)


