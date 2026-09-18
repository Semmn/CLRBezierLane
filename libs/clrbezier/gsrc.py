"""GSRC (globally structured row-column context) for CLRBezierHead.

Ported from ``libs/models/necks/gsrc_fpn.py`` (GSRCNeck) into the head, so the
official CLRerNetFPN path stays untouched: only the head's per-query features
receive global context.

Pipeline
--------
    coarsest FPN level  ->  structure branch : row-column attention blocks
                        ->  context branch   : transformer (or SegMAN) blocks
                        ->  cross-attention (structure queries, context keys)
                        ->  tokens [B, H*W, C]

    ROIGather output [B, K, C] --cross-attention--> context per query
                               --zero-init gate--> added back to [B, K, C]

Both injection paths are zero-initialized, so a GSRC-enabled model starts
numerically identical to the model without GSRC.

Differences from the neck version:
    * ``squeeze(-1)`` / ``squeeze(-2)`` instead of ``squeeze()`` (the latter
      drops the batch dimension when batch size is 1, i.e. at inference);
    * the structure branch is self-contained here, so natten / selective-scan
      are only needed when ``context_branch="segman"``;
    * residual + LayerNorm around the structure/context cross-attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvFFN(nn.Module):
    """FFN from gsrc_fpn.py (1x1 -> act -> depthwise residual -> 1x1)."""

    def __init__(self, embed_dim, ffn_dim, act_layer=nn.ReLU, dropout=0.0, kernel_size=3):
        super().__init__()
        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.act_layer(self.fc1(x))
        x = x + self.dwconv(x)
        x = self.drop(x)
        return self.drop(self.fc2(x))


class RowColumnAttention(nn.Module):
    """Row/column token attention (gsrc_fpn.py), fixed for batch size 1."""

    def __init__(self, in_channels, proj_dim, feat_size):
        super().__init__()
        self.proj_dim = proj_dim
        self.scale = proj_dim ** 0.5
        self.feat_h, self.feat_w = int(feat_size[0]), int(feat_size[1])
        self.row_qkv = nn.Conv2d(in_channels, proj_dim * 3, kernel_size=(1, self.feat_w))
        self.col_qkv = nn.Conv2d(in_channels, proj_dim * 3, kernel_size=(self.feat_h, 1))
        self.crow_qkv = nn.Linear(proj_dim, proj_dim * 3, bias=True)
        self.ccol_qkv = nn.Linear(proj_dim, proj_dim * 3, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.fuse_and_refine = nn.Sequential(
            nn.Conv2d(proj_dim * 2, proj_dim * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(proj_dim * 4, in_channels, kernel_size=1),
        )

    def attention(self, q, k, v):
        return self.softmax((q @ k.permute(0, 2, 1)) / self.scale) @ v

    def forward(self, x):
        if x.shape[-2:] != (self.feat_h, self.feat_w):
            raise RuntimeError(
                f"GSRC row-column attention is built for feat_size "
                f"({self.feat_h}, {self.feat_w}) but got {tuple(x.shape[-2:])}. "
                "Set gsrc_cfg.feat_size to the coarsest FPN level size.")
        row_t = self.row_qkv(x).squeeze(-1).permute(0, 2, 1)  # [B, H, 3P]
        col_t = self.col_qkv(x).squeeze(-2).permute(0, 2, 1)  # [B, W, 3P]
        row_tq, row_tk, row_tv = torch.chunk(row_t, 3, dim=-1)
        col_tq, col_tk, col_tv = torch.chunk(col_t, 3, dim=-1)
        row_t = self.attention(row_tq, row_tk, row_tv)
        col_t = self.attention(col_tq, col_tk, col_tv)

        crow_tq, crow_tk, crow_tv = torch.chunk(self.crow_qkv(row_t), 3, dim=-1)
        ccol_tq, ccol_tk, ccol_tv = torch.chunk(self.ccol_qkv(col_t), 3, dim=-1)
        r2c_t = self.attention(crow_tq, ccol_tk, ccol_tv)
        c2r_t = self.attention(ccol_tq, crow_tk, crow_tv)

        r_spatial = r2c_t.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, self.feat_w)
        c_spatial = c2r_t.permute(0, 2, 1).unsqueeze(-2).expand(-1, -1, self.feat_h, -1)
        return self.fuse_and_refine(torch.cat([r_spatial, c_spatial], dim=1))


class RowColumnBlock(nn.Module):
    def __init__(self, dim, feat_size, ffn_dim, attn_drop=0.1, ffn_drop=0.25):
        super().__init__()
        self.rc_attn = RowColumnAttention(dim, dim, feat_size)
        self.ffn = ConvFFN(dim, ffn_dim)
        self.layer_norm1 = LayerNorm2d(dim)
        self.layer_norm2 = LayerNorm2d(dim)
        self.attn_dropout = nn.Dropout2d(attn_drop) if attn_drop > 0 else nn.Identity()
        self.ffn_dropout = nn.Dropout2d(ffn_drop) if ffn_drop > 0 else nn.Identity()

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        x = x_norm + self.attn_dropout(self.rc_attn(x_norm))
        x_norm = self.layer_norm2(x)
        return x_norm + self.ffn_dropout(self.ffn(x_norm))


class TokenTransformerBlock(nn.Module):
    """Pre-norm self-attention + MLP over flattened tokens."""

    def __init__(self, dim, num_heads=4, ffn_dim=256, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(ffn_dim, dim))

    def forward(self, tokens):
        x = self.norm1(tokens)
        tokens = tokens + self.attn(x, x, x, need_weights=False)[0]
        return tokens + self.mlp(self.norm2(tokens))


class GSRCContext(nn.Module):
    """Build global context tokens from one feature map.

    Args:
        in_channels: channels of the source map (coarsest FPN level: 64).
        proj_dim: token width (match fc_hidden_dim to keep injection cheap).
        feat_size: (H, W) of the source map; 800x320 input -> (10, 25).
        num_layers: blocks per branch.
        ffn_dim / ffn_drop: FFN width and dropout of the structure branch.
        context_branch: "transformer" (default, dependency-free),
            "segman" (BasicLayer_Norm from libs/models/necks/gsrc_fpn.py; needs
            natten and selective_scan_cuda_oflex), or "none" (structure only).
        num_heads: heads of the structure-context cross-attention.
    """

    def __init__(self, in_channels=64, proj_dim=64, feat_size=(10, 25), num_layers=2,
                 ffn_dim=256, ffn_drop=0.25, attn_drop=0.1, context_branch="transformer",
                 num_heads=4, use_pos_embed=True, segman_cfg=None):
        super().__init__()
        self.feat_size = (int(feat_size[0]), int(feat_size[1]))
        self.proj_dim = int(proj_dim)
        self.context_branch = context_branch
        num_tokens = self.feat_size[0] * self.feat_size[1]

        self.struct_proj = nn.Conv2d(in_channels, proj_dim, kernel_size=1)
        self.struct_layer = nn.Sequential(*[
            RowColumnBlock(proj_dim, self.feat_size, ffn_dim, attn_drop, ffn_drop)
            for _ in range(num_layers)])

        if context_branch == "none":
            self.ctx_proj = None
        elif context_branch == "transformer":
            self.ctx_proj = nn.Conv2d(in_channels, proj_dim, kernel_size=1)
            self.ctx_layer = nn.ModuleList([
                TokenTransformerBlock(proj_dim, num_heads, ffn_dim) for _ in range(num_layers)])
        elif context_branch == "segman":
            from libs.models.necks.gsrc_fpn import BasicLayer_Norm  # noqa: WPS433
            from libs.models.necks.gsrc_fpn import LayerNorm2d as GsrcLayerNorm2d
            cfg = dict(embed_dim=proj_dim, depth=num_layers, num_heads=2, window_size=7,
                       window_dilation=1, global_mode=False, use_rpb=False, sr_ratio=1,
                       ffn_dim=ffn_dim, drop_path=0.0, layerscale=True,
                       layer_init_values=1e-6, norm_layer=GsrcLayerNorm2d, use_checkpoint=0)
            cfg.update(segman_cfg or {})
            self.ctx_proj = nn.Conv2d(in_channels, proj_dim, kernel_size=1)
            self.ctx_layer = BasicLayer_Norm(**cfg)
        else:
            raise ValueError(f"Unknown context_branch {context_branch!r}")

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, proj_dim)) if use_pos_embed else None
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.ctx_proj is not None:
            self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads, batch_first=True)
            self.cross_norm_q = nn.LayerNorm(proj_dim)
            self.cross_norm_kv = nn.LayerNorm(proj_dim)
        self.out_norm = nn.LayerNorm(proj_dim)

    @staticmethod
    def _flatten(x):
        return x.flatten(2).permute(0, 2, 1).contiguous()  # [B, C, H, W] -> [B, L, C]

    def forward(self, feat):
        tokens = self._flatten(self.struct_layer(self.struct_proj(feat)))
        if self.ctx_proj is not None:
            ctx = self.ctx_proj(feat)
            if self.context_branch == "transformer":
                ctx = self._flatten(ctx)
                if self.pos_embed is not None:
                    ctx = ctx + self.pos_embed
                for block in self.ctx_layer:
                    ctx = block(ctx)
            else:
                ctx = self._flatten(self.ctx_layer(ctx))
            query = self.cross_norm_q(tokens if self.pos_embed is None else tokens + self.pos_embed)
            key = self.cross_norm_kv(ctx)
            tokens = tokens + self.cross_attn(query, key, ctx, need_weights=False)[0]
        elif self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        return self.out_norm(tokens)


class GSRCInjection(nn.Module):
    """Inject global tokens into the per-query features of one refinement stage.

    Both branches are zero-initialized (``fuse="gate"``: zero LayerScale;
    ``fuse="concat"``: identity for the query half, zero for the context half),
    so the stage output equals the ROIGather output before training.

    Args:
        dim: query/token width (fc_hidden_dim).
        num_heads: cross-attention heads.
        fuse: "gate" (default) or "concat".
        gate_init: initial LayerScale value; 0.0 reproduces the baseline exactly.
        query_self_attention: also let the K queries attend to each other
            (zero-gated). Useful for the one-to-one branch, where nothing else
            lets duplicate anchors suppress each other.
    """

    def __init__(self, dim=64, num_heads=4, fuse="gate", dropout=0.0,
                 query_self_attention=False, gate_init=0.0):
        super().__init__()
        if fuse not in ("gate", "concat"):
            raise ValueError(f"Unknown fuse {fuse!r}; use 'gate' or 'concat'")
        self.fuse = fuse
        self.query_self_attention = bool(query_self_attention)
        # gate_init > 0 (e.g. 1e-2) makes GSRC active from the first iteration;
        # 0.0 keeps the baseline exactly and lets the gate open on its own (ReZero).
        self.gate_init = float(gate_init)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        if fuse == "gate":
            self.gamma = nn.Parameter(torch.zeros(dim))
        else:
            self.concat_fc = nn.Linear(2 * dim, dim)
        if self.query_self_attention:
            self.norm_self = nn.LayerNorm(dim)
            self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.gamma_self = nn.Parameter(torch.zeros(dim))
        self.zero_init()

    def zero_init(self):
        """(Re-)apply the identity initialization; call after any global init."""
        if self.fuse == "gate":
            nn.init.constant_(self.gamma, self.gate_init)
        else:
            with torch.no_grad():
                dim = self.concat_fc.out_features
                self.concat_fc.weight.zero_()
                self.concat_fc.weight[:, :dim].copy_(torch.eye(dim))
                self.concat_fc.bias.zero_()
        if self.query_self_attention:
            nn.init.constant_(self.gamma_self, self.gate_init)

    def forward(self, queries, tokens):
        """queries: [B, K, C]; tokens: [B, L, C] -> [B, K, C]."""
        if tokens.shape[0] != queries.shape[0]:
            raise RuntimeError(
                f"GSRC tokens batch {tokens.shape[0]} != query batch {queries.shape[0]}; "
                "auxiliary groups must repeat the tokens.")
        if self.query_self_attention:
            q = self.norm_self(queries)
            queries = queries + self.gamma_self * self.self_attn(q, q, q, need_weights=False)[0]
        context = self.cross_attn(self.norm_q(queries), self.norm_kv(tokens), tokens,
                                  need_weights=False)[0]
        if self.fuse == "gate":
            return queries + self.gamma * context
        return F.relu(self.concat_fc(torch.cat([queries, context], dim=-1)))


class GSRCModule(nn.Module):
    """Context encoder + one injection module per refinement stage."""

    def __init__(self, in_channels=64, dim=64, refine_layers=3, stages=(0, 1, 2),
                 context_cfg=None, injection_cfg=None):
        super().__init__()
        context_cfg = dict(context_cfg or {})
        context_cfg.setdefault("in_channels", in_channels)
        context_cfg.setdefault("proj_dim", dim)
        self.context = GSRCContext(**context_cfg)
        self.stages = sorted(int(s) for s in stages)
        injection_cfg = dict(injection_cfg or {})
        injection_cfg.setdefault("dim", dim)
        self.injections = nn.ModuleDict(
            {str(s): GSRCInjection(**injection_cfg) for s in self.stages})

    def zero_init(self):
        for module in self.injections.values():
            module.zero_init()

    def tokens(self, feat):
        return self.context(feat)

    def inject(self, stage, queries, tokens):
        if stage not in self.stages or tokens is None:
            return queries
        return self.injections[str(stage)](queries, tokens)
