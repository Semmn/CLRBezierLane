import torch.nn as nn
import torchvision
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
import torchvision

# token-mixer
class RowColEnhance(nn.Module):
    def __init__(self, in_channels, proj_dim, feat_size, num_heads=4):
        super(RowColEnhance, self).__init__()
        self.in_channels = in_channels
        self.proj_dim = proj_dim
        self.feat_h = feat_size[0]
        self.feat_w = feat_size[1]
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads # projection dimension per head
        
        self.row_conv = nn.Conv2d(in_channels, proj_dim, kernel_size=(1, self.feat_w), stride=1, padding=0, bias=True)
        self.col_conv = nn.Conv2d(in_channels, proj_dim, kernel_size=(self.feat_h, 1), stride=1, padding=0, bias=True)
        
        self.conv_dw_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=proj_dim, dilation=1)
        self.conv_pw_v = nn.Conv2d(in_channels, proj_dim, kernel_size=1, stride=1, padding=0, bias=True)
        
        # self.conv_o = nn.Conv2d(proj_dim, proj_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_o = torchvision.ops.DeformConv2d(proj_dim, proj_dim, kernel_size=3, padding=1, groups=1, bias=True)
        
        # offset mask shape: (batch_size, 2 * offset_groups * kernel_height * kernel_width, out_h, out_w)
        # in real implementation, we don't need to directly set the offset mask. instead we use small convolution
        # for the projecting offset mask.
        offset_groups=1
        kernel_height = 3
        kernel_width = 3
        offset_dims = 2 * offset_groups * kernel_height * kernel_width 
        self.offset_conv = nn.Conv2d(proj_dim, offset_dims, kernel_size=(kernel_height, kernel_width), padding=1, groups=offset_groups, bias=True) # for creating offset 
        
        self.linear_qk = nn.Linear(1, self.head_dim, bias=True) # linear projection of attention map to high-dimensional space
        self.linear_qk.apply(lambda x: nn.init.constant_(x.weight, 1.0)) # initialize to identity mapping
        self.linear_qk.apply(lambda x: nn.init.constant_(x.bias, 0.0)) # intialize to identity mapping (set bias as 0)
        
    
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
        
        offset = self.offset_conv(enhanced_feat) # (B, 18, H, W) for 3x3 kernel
        enhanced_feat = self.conv_o(enhanced_feat, offset) # (B, proj_dim, H, W)
        
        return enhanced_feat

# token-mixer with spatial ratio adjustment
class RowColEnhanceModule(nn.Module):
    def __init__(self, downscale, upscale, in_channels, proj_dim, feat_size, num_heads):
        super(RowColEnhanceModule, self).__init__()
        self.downscale = downscale # pixel unshuffle downscale factor
        self.upscale = upscale
        self.num_heads = num_heads
        self.rowcol_attn = RowColEnhance(in_channels, proj_dim, feat_size, num_heads=self.num_heads)
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
    

# @MODELS.register_module()
# class RowColumnAttnFPN(nn.Module):
#     def __init__(self, in_channels, out_channels, num_outs, num_blocks, feat_size=[(40, 100), (20, 50), (10, 25)]):
#         """
#         Feature pyramid network for CLRerNet.
#         Args:
#             in_channels (List[int]): Channel number list.
#             out_channels (int): Number of output feature map channels.
#             num_outs (int): Number of output feature map levels.
#             num_rc (int): number of row-col attention layers
#         """
#         super(RowColumnAttnFPN, self).__init__()
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
#         self.num_blocks = num_blocks
#         self.feat_size = feat_size

#         self.backbone_end_level = self.num_ins
#         self.start_level = 0
        
#         self.rowcol_layers = nn.ModuleList()
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()

#         for i in range(self.start_level, self.backbone_end_level):
#             attns = RowColAttnLayer(in_channels=out_channels, proj_dim=out_channels, feat_size=feat_size[i], ffn_dim=out_channels * 2,
#                                     ffn_drop=0.0, ffn_kers=3, num_layers=num_blocks)
            
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=None,
#                 norm_cfg=None,
#                 act_cfg=None,
#                 inplace=False,
#             )
            
#             fpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=None,
#                 norm_cfg=None,
#                 act_cfg=None,
#                 inplace=False,
#             )
            
#             self.rowcol_layers.append(attns)
#             self.lateral_convs.append(l_conv)
#             self.fpn_convs.append(fpn_conv)

#     def forward(self, inputs):
#         """
#         Args:
#             inputs (List[torch.Tensor]): Input feature maps.
#               Example of shapes:
#                 ([1, 64, 80, 200], [1, 128, 40, 100], [1, 256, 20, 50], [1, 512, 10, 25]).
#         Returns:
#             outputs (Tuple[torch.Tensor]): Output feature maps.
#               The number of feature map levels and channels correspond to
#                `num_outs` and `out_channels` respectively.
#               Example of shapes:
#                 ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
#         """
#         if isinstance(inputs, tuple):
#             inputs = list(inputs)

#         assert len(inputs) >= len(self.in_channels)  # 4 > 3
        
#         # remove the lowest level feature if more levels are provided
#         if len(inputs) > len(self.in_channels):
#             for _ in range(len(inputs) - len(self.in_channels)):
#                 del inputs[0]
                
#         # build laterals
#         laterals = []
#         used_backbone_levels = len(self.lateral_convs) # 3
#         for i in range(used_backbone_levels):
#             laterals.append(self.lateral_convs[i](inputs[i]))
        
#         # apply row-column attentions
#         attn_features = []
#         spatial_map = None # from top-level to low level, initialized with None
#         for i in range(used_backbone_levels - 1, -1, -1):    
#             attn_feature, spatial_map = self.rowcol_layers[i](laterals[i], spatial_map)
#             attn_features.append(attn_feature)
        
#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(0, used_backbone_levels - 1):
#             prev_shape = attn_features[i + 1].shape[2:]
#             attn_features[i + 1] += F.interpolate(
#                 attn_features[i], size=prev_shape, mode='nearest'
#             )
        
#         # convolution for post-processing
#         outs = []
#         for i in range(used_backbone_levels - 1, -1, -1):
#             outs.append(self.fpn_convs[i](attn_features[i]))
        
#         return tuple(outs)
    
    
# Row-Column Attention -> lateral -> FPN convolution
# def forward(self, inputs):
#     """
#     Args:
#         inputs (List[torch.Tensor]): Input feature maps.
#           Example of shapes:
#             ([1, 64, 80, 200], [1, 128, 40, 100], [1, 256, 20, 50], [1, 512, 10, 25]).
#     Returns:
#         outputs (Tuple[torch.Tensor]): Output feature maps.
#           The number of feature map levels and channels correspond to
#            `num_outs` and `out_channels` respectively.
#           Example of shapes:
#             ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
#     """
#     if isinstance(inputs, tuple):
#         inputs = list(inputs)

#     assert len(inputs) >= len(self.in_channels)  # 4 > 3

#     if len(inputs) > len(self.in_channels):
#         for _ in range(len(inputs) - len(self.in_channels)):
#             del inputs[0]

#     attn_features = []
#     spatial_map = None # from top-level to low level, initialized with None
    
#     # apply row-column attentions
#     used_backbone_levels = len(self.lateral_convs)
#     for i in range(used_backbone_levels - 1, -1, -1):    
#         attn_feature, spatial_map = self.rowcol_layers[i](inputs[i], spatial_map)
#         attn_features.append(attn_feature)

#     # build laterals
#     laterals = []
#     for i in range(used_backbone_levels):
#         laterals.append(self.lateral_convs[i](attn_features[used_backbone_levels-1-i]))
    
#     # build top-down path
#     used_backbone_levels = len(laterals)
#     for i in range(used_backbone_levels - 1, 0, -1):
#         prev_shape = laterals[i - 1].shape[2:]
#         laterals[i - 1] += F.interpolate(
#             laterals[i], size=prev_shape, mode='nearest'
#         )
    
#     # convolution for post-processing
#     outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
    
#     return tuple(outs)