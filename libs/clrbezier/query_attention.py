"""Masked query-to-query attention with anchor-geometry positional bias.

Used after ROIGather (and after GSRC injection, when enabled), so anchors can
see each other before the cls/reg towers.

Masking
-------
The auxiliary branch packs M perturbed groups into one tensor of M*K queries.
Attention is restricted to a query's own group (block-diagonal mask), as in the
hybrid-branch designs of H-DETR / Co-DETR: groups must not exchange
information, otherwise the one-to-many branch leaks duplicates into the
one-to-one branch's feature space. The main branch is a single group, so the
mask is a no-op there; the same module and weights serve both.

Positional bias
---------------
Queries are anchors, so "position" is the Bezier state [y_start, P0x..P3x]:
    * content bias: an MLP embedding of a query's own state added to q and k;
    * pairwise bias: an MLP over relative geometry (difference of y_start and
      of the control points, plus |dx| statistics) producing one scalar per
      head, added to the attention logits. This lets a head implement "suppress
      anchors whose curve is close to mine" without having to infer geometry
      from appearance features.

The output is a zero-gated residual (``gate_init=0.0``), so adding this module
does not change the model at initialization.
"""
import torch
import torch.nn as nn


class AnchorStateEmbedding(nn.Module):
    """Embed the per-query BRR state [y_start, P0x..P3x] -> dim."""

    def __init__(self, dim, state_dim=5, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, state):
        return self.norm(self.mlp(state))


class PairwiseGeometryBias(nn.Module):
    """Relative-geometry attention bias, one scalar per head."""

    def __init__(self, num_heads, hidden=32, state_dim=5):
        super().__init__()
        # features: state difference (state_dim), |mean dx|, max |dx|, y_start gap
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + 3, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, num_heads))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, state):
        """state: [B, Q, S] -> bias [B, heads, Q, Q]."""
        diff = state[:, :, None, :] - state[:, None, :, :]
        cp_diff = diff[..., 1:]
        feats = torch.cat([
            diff,
            cp_diff.mean(-1, keepdim=True).abs(),
            cp_diff.abs().amax(-1, keepdim=True),
            diff[..., :1].abs(),
        ], dim=-1)
        return self.mlp(feats).permute(0, 3, 1, 2).contiguous()


class MaskedQuerySelfAttention(nn.Module):
    """Group-masked self-attention over anchor queries.

    Args:
        dim: query width (fc_hidden_dim).
        num_heads: attention heads.
        use_state_embedding: add the anchor-state embedding to q and k.
        use_pairwise_bias: add the relative-geometry bias to the logits.
        gate_init: LayerScale init; 0.0 keeps the model identical at start.
        detach_state: use the geometry as a constant (no gradient through the
            positional bias into the Bezier state). Recommended: the bias should
            describe the anchors, not become another path that moves them.
    """

    def __init__(self, dim=64, num_heads=4, dropout=0.0, use_state_embedding=True,
                 use_pairwise_bias=True, state_dim=5, bias_hidden=32, gate_init=0.0,
                 detach_state=True):
        super().__init__()
        self.num_heads = int(num_heads)
        self.detach_state = bool(detach_state)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.state_embed = AnchorStateEmbedding(dim, state_dim, dim) if use_state_embedding else None
        self.pair_bias = PairwiseGeometryBias(num_heads, bias_hidden, state_dim) if use_pairwise_bias else None
        self.gamma = nn.Parameter(torch.empty(dim))
        self.gate_init = float(gate_init)
        self.zero_init()

    def zero_init(self):
        nn.init.constant_(self.gamma, self.gate_init)
        if self.pair_bias is not None:
            nn.init.zeros_(self.pair_bias.mlp[-1].weight)
            nn.init.zeros_(self.pair_bias.mlp[-1].bias)

    @staticmethod
    def group_mask(num_queries, num_groups, device):
        """Block-diagonal mask [Q, Q]; True where attention is forbidden."""
        if num_groups <= 1:
            return None
        per_group = num_queries // num_groups
        ids = torch.arange(num_queries, device=device) // per_group
        return ids[:, None] != ids[None, :]

    def forward(self, queries, state, num_groups=1):
        """queries: [B, Q, C]; state: [B, Q, S]; -> [B, Q, C]."""
        batch, num_q, _ = queries.shape
        if self.detach_state:
            state = state.detach()
        x = self.norm(queries)
        q = k = x
        if self.state_embed is not None:
            pos = self.state_embed(state)
            q = q + pos
            k = k + pos

        attn_mask = None
        block = self.group_mask(num_q, num_groups, queries.device)
        if self.pair_bias is not None:
            bias = self.pair_bias(state)  # [B, H, Q, Q]
            if block is not None:
                bias = bias.masked_fill(block[None, None], float("-inf"))
            attn_mask = bias.reshape(batch * self.num_heads, num_q, num_q)
        elif block is not None:
            attn_mask = block

        out = self.attn(q, k, x, attn_mask=attn_mask, need_weights=False)[0]
        return queries + self.gamma * out
