"""BezierFormerNet: a replica of BezierFormer (Dong et al., arXiv 2404.16304, 2D lanes) as the paper describes it
(clrbezierlane/bf_replica/bf_replica_design.md). The paper has no code; what it leaves open is set to Deformable
DETR's defaults and listed in the design note.

    BezierFormerNet   detector: backbone -> head, no neck. The 2x input of dataset_curvelanes_clrbezier_2x*.py is
                      pooled 2 x 2 to 800 x 320 (as FovealLaneNet's context view); an 800 x 320 input is used as is.
    BezierFormerHead  head, in order:
1. tokens: C3-C5 through a 1 x 1 conv + GroupNorm to dim, plus a stride-64 level (3 x 3 stride-2 conv on C5 +
   GroupNorm); a level embedding and a sine embedding of each token's centre. 4 scales (Table VI).
2. encoder: enc_cfg.num_layers multi-scale deformable self-attention layers (FovealLaneHead's encoder layer, one view).
3. queries: E_0 learnable [N, dim]; C_0 = MLP(X) (Sec. III.B): an MLP of the encoded tokens averaged per scale, whose
   last layer starts at zero weights and a bias that spreads N straight lines over the image (a fan from the bottom
   edge towards the upper middle). Control points are free 2D (normalized x / (w - 1), y / (h - 1)); no y order.
4. decoder, dec_cfg.num_layers identical layers (Sec. III.C, Fig. 2). Layer i:
   - FV sampling: ref_points points of C_{i-1} at t = 0 .. 1; position encoding PE = MLP(concat of the points' sine
     embeddings) (Eq. 3);
   - self-attention over the queries, PE added to queries and keys (Eq. 4);
   - Bezier curve attention (Eq. 5): the encoded map of every scale is read at each reference point (f_{k,l}); two
     MLPs of f_{k,l} give attn_samples offsets (in that scale's cells) and weights per head; one softmax per head over
     all reference points, scales and samples; the values are the encoded tokens. Offsets and weights come from the
     reference-point features only, not from the query;
   - FFN; a classification head on E_i and an FFN for dC_i: C_i = C_{i-1} (detached, as Deformable DETR's box
     refinement) + dC_i. Heads are per layer; layer 0 (C_0, with E_0's score) is supervised too (Eq. 10).
5. loss of every layer i = 0 .. num_layers (Sec. III.E): sigmoid focal classification; L_reg = L_loc + L_len +
   L_endpoint with L_loc = 1 - (CIoU(pred -> GT) + CIoU(GT -> pred)) / 2, CIoU(A -> B) = mean over A's n_dis samples
   of (2e - d) / (2e + d), d = the distance to the nearest of B's n_dis samples (Eq. 6-7; px, e = width_px);
   L_len = |len(pred) / len(GT) - 1| (Eq. 8); L_endpoint = mean L2 distance of the first and the last points,
   normalized coordinates, index-paired (Eq. 9; GT bottom -> top). Assignment: SimOTA with top-k = 1 on
   cost = L_reg + L_cls per pair, as YOLOX's dynamic_k_matching (each GT lane takes its cheapest query; a query
   several lanes took goes to the lane with the lowest cost for it, so a lane can stay unmatched). Each layer is
   matched on its own outputs.
6. test (test_cfg.output): "rows" (default): the last layer's curve, bottom end first, through the official row
   layout (72 rows, blrnet.curve_to_rows) and CLRerHead's get_lanes / predictions_to_lanes, the same path and tools
   as every other run (the error breakdown and the threshold sweep read it); "points": out_points samples of the curve
   inside the image as Lane points (BezierFormer's dense output; y made increasing for Lane's spline, so a curve that
   turns back in y is cut there).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS
from mmengine.model import BaseModule

from libs.utils.lane_utils import Lane

from .blrnet import _deform_grid_bias, bezier_end_slopes, curve_points, curve_to_rows
from .data_adapters import _lookup
from .foveal_lane import CTX, CrossViewEncoderLayer, ViewFrame, _Map, _sample_maps, sine_embed
from .hblrnet import bottom_first, gt_polyline
from .head import StagePredictions


def _mlp(dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers[:-1])


def polyline_length(pts):
    """[..., P, 2] -> [...]."""
    return (pts[..., 1:, :] - pts[..., :-1, :]).norm(dim=-1).sum(-1)


def chamfer_iou(a, b, e, pairwise=False):
    """Symmetric Chamfer IoU (Eq. 6-7) of point sets a [..., P, 2] and b [..., Q, 2] (px): the mean of CIoU(a -> b) and
    CIoU(b -> a). pairwise: a [B, N, P, 2], b [B, G, Q, 2] -> [B, N, G]; else paired -> [...]."""
    if pairwise:
        a, b = a[:, :, None], b[:, None]
    diff = a[..., :, None, :] - b[..., None, :, :]
    d = (diff * diff).sum(-1).add(1e-12).sqrt()                                     # [..., P, Q]
    da, db = d.min(-1).values, d.min(-2).values
    two_e = 2.0 * float(e)
    return 0.5 * (((two_e - da) / (two_e + da)).mean(-1) + ((two_e - db) / (two_e + db)).mean(-1))


def simota_top1(cost):
    """SimOTA with top-k 1 on cost [N, G], as YOLOX's dynamic_k_matching with k = 1: each GT takes its cheapest query;
    a query several GTs took goes to the GT with the lowest cost for it (over all GTs). Returns the matched GT index of
    each query [N] (-1: none)."""
    num_n, num_g = cost.shape
    if num_g == 0 or num_n == 0:
        return torch.full((num_n,), -1, dtype=torch.long, device=cost.device)
    mat = torch.zeros_like(cost)
    mat[cost.argmin(0), torch.arange(num_g, device=cost.device)] = 1.0
    multi = mat.sum(1) > 1
    if bool(multi.any()):
        mat[multi] = 0.0
        mat[multi, cost[multi].argmin(1)] = 1.0
    return torch.where(mat.sum(1) > 0, mat.argmax(1), torch.full_like(mat[:, 0], -1, dtype=torch.long))


def focal_terms(logit, alpha, gamma):
    """Sigmoid focal loss of each logit as a positive and as a negative."""
    p = logit.sigmoid()
    pos = -alpha * (1 - p).pow(gamma) * F.logsigmoid(logit)
    neg = -(1 - alpha) * p.pow(gamma) * F.logsigmoid(-logit)
    return pos, neg


# ------------------------------------------------------------------ detector
@MODELS.register_module()
class BezierFormerNet(SingleStageDetector):
    """backbone -> BezierFormerHead (module docstring); no neck."""

    def __init__(self, backbone, bbox_head, neck=None, train_cfg=None, test_cfg=None, data_preprocessor=None,
                 init_cfg=None):
        if neck is not None:
            raise ValueError("BezierFormerNet has no neck: the head's encoder reads the backbone levels")
        super().__init__(backbone=backbone, neck=None, bbox_head=bbox_head, train_cfg=train_cfg, test_cfg=test_cfg,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    def extract_feat(self, batch_inputs, batch_data_samples=None):
        head = self.bbox_head
        size = tuple(batch_inputs.shape[-2:])
        if size == (2 * head.img_h, 2 * head.img_w):
            batch_inputs = F.avg_pool2d(batch_inputs, 2)
        elif size != (head.img_h, head.img_w):
            raise ValueError(f"BezierFormerNet expects {head.img_w} x {head.img_h} or twice that, got "
                             f"{size[1]} x {size[0]}")
        return tuple(self.backbone(batch_inputs))

    def loss(self, batch_inputs, batch_data_samples):
        return self.bbox_head.loss(self.extract_feat(batch_inputs), batch_data_samples)

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        for sample in batch_data_samples:
            sample.metainfo["batch_input_shape"] = tuple(batch_inputs.size()[-2:])
        return self.bbox_head.predict(self.extract_feat(batch_inputs), batch_data_samples)

    def _forward(self, batch_inputs, batch_data_samples=None):
        return self.bbox_head.forward(self.extract_feat(batch_inputs))


# ------------------------------------------------------------------ decoder
class BezierCurveAttention(nn.Module):
    """Eq. 5: each query reads every map at num_samples offsets (in the map's cells) around each of its reference
    points; offsets and weights are two MLPs of the encoded feature at the reference point on that map; one softmax
    per head over reference points x maps x samples."""

    def __init__(self, dim, num_maps, num_heads, num_samples):
        super().__init__()
        if dim % num_heads:
            raise ValueError("BezierFormerHead dec_cfg.attn_heads must divide dim")
        self.heads, self.maps, self.samples = int(num_heads), int(num_maps), int(num_samples)
        self.c_h = dim // self.heads
        self.value_proj = nn.Linear(dim, dim)
        self.off_mlp = _mlp([dim, dim, self.heads * self.samples * 2])
        self.w_mlp = _mlp([dim, dim, self.heads * self.samples])
        self.out = nn.Linear(dim, dim)
        self.last_offset_px = None

    def reset_parameters(self):
        for mlp in (self.off_mlp, self.w_mlp):
            nn.init.xavier_uniform_(mlp[0].weight)
            nn.init.zeros_(mlp[0].bias)
            nn.init.zeros_(mlp[2].weight)
        with torch.no_grad():
            self.off_mlp[2].bias.copy_(_deform_grid_bias(self.heads, 1, self.samples))
        nn.init.zeros_(self.w_mlp[2].bias)
        for lin in (self.value_proj, self.out):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, ref, tokens, maps, frame):
        """ref [B, N, K, 2] (px); tokens [B, T, D] -> [B, N, D]."""
        batch_size, num_q, num_k = ref.shape[:3]
        dim = tokens.shape[-1]
        H, L, S = self.heads, self.maps, self.samples
        ref = ref.float()
        feats = []
        for mp in maps:
            v = tokens[:, mp.start:mp.start + mp.n].transpose(1, 2).reshape(batch_size, dim, mp.h, mp.w)
            g = frame.grid(ref, mp.view).reshape(batch_size, num_q * num_k, 1, 2)
            f = F.grid_sample(v, g.to(v.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
            feats.append(f[..., 0].transpose(1, 2).reshape(batch_size, num_q, num_k, dim))
        f = torch.stack(feats, dim=3)                                                  # [B, N, K, L, D]
        off = self.off_mlp(f).float().view(batch_size, num_q, num_k, L, H, S, 2)
        logit = self.w_mlp(f).float().view(batch_size, num_q, num_k, L, H, S)
        w = torch.softmax(logit.permute(0, 1, 4, 2, 3, 5).reshape(batch_size, num_q, H, num_k * L * S), dim=-1)
        w_m = w.view(batch_size, num_q, H, num_k, L, S).permute(0, 1, 3, 2, 4, 5)       # [B, N, K, H, L, S]
        w_m = w_m.reshape(batch_size, num_q * num_k, H, L, S)
        ref_q = ref[:, :, :, None, None, :]
        grids = [frame.sample_grid(ref_q, off[:, :, :, m], mp).reshape(batch_size, num_q * num_k, H, S, 2)
                 for m, mp in enumerate(maps)]
        a = _sample_maps(self.value_proj(tokens).view(batch_size, tokens.shape[1], H, self.c_h), maps, grids, w_m, H)
        a = a.view(batch_size, num_q, num_k, dim).sum(2)                               # the softmax spans K
        with torch.no_grad():
            self.last_offset_px = off[:, :, :, 0].abs().mean() * maps[0].cell
        return self.out(a.to(tokens.dtype))


class BezierFormerLayer(nn.Module):
    """Self-attention (PE on queries and keys) -> Bezier curve attention -> FFN, post-norm (Deformable DETR)."""

    def __init__(self, dim, sa_heads, num_maps, attn_heads, num_samples, ffn_dim, dropout):
        super().__init__()
        self.sa = nn.MultiheadAttention(dim, sa_heads, dropout=dropout, batch_first=True)
        self.curve = BezierCurveAttention(dim, num_maps, attn_heads, num_samples)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        self.sa._reset_parameters()
        self.curve.reset_parameters()
        for lin in (self.ffn[0], self.ffn[3]):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for n in self.norms:
            nn.init.ones_(n.weight)
            nn.init.zeros_(n.bias)

    def forward(self, e, pe, ref, tokens, maps, frame):
        q = e + pe
        e = self.norms[0](e + self.drop(self.sa(q, q, e, need_weights=False)[0]))
        e = self.norms[1](e + self.drop(self.curve(ref, tokens, maps, frame)))
        return self.norms[2](e + self.drop(self.ffn(e)))


# ------------------------------------------------------------------ head
@MODELS.register_module()
class BezierFormerHead(BaseModule):
    """The replica head (module docstring). Config, all keys optional (defaults in brackets):
    in_channels ([128, 256, 512], backbone C3-C5), num_queries (16), dim (128, R18 in Table VI), img_w / img_h
    (800 / 320), num_points (72, rows of the official layout).
    enc_cfg: num_layers (6), num_heads (8), num_points (4), ffn_dim (1024), dropout (0.1) (Deformable DETR's).
    dec_cfg: num_layers (6), sa_heads (8), ref_points (5), attn_heads (8), attn_samples (5), ffn_dim (1024),
    dropout (0.1), init_hidden (512, C_0's MLP width).
    loss_cfg: n_dis (200), width_px (10.0, e), cls_weight (1.0), loc_weight (1.0), len_weight (1.0), end_weight (1.0),
    focal_alpha (0.25), focal_gamma (2.0), min_gt_len_px (8.0, shorter GT lanes are dropped).
    test_cfg: CLRerHead's (conf_threshold, use_nms, nms_thres, nms_topk, as_lanes, extend_bottom, ori_img_w / h,
    cut_height) plus output ("rows" | "points") and out_points (100, curve samples for "points")."""

    def __init__(self, in_channels=(128, 256, 512), num_queries=16, dim=128, img_w=800, img_h=320, num_points=72,
                 enc_cfg=None, dec_cfg=None, loss_cfg=None, train_cfg=None, test_cfg=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        ec, dc, lc = (dict(c or {}) for c in (enc_cfg, dec_cfg, loss_cfg))
        for name, cfg, known in (
                ("enc_cfg", ec, {"num_layers", "num_heads", "num_points", "ffn_dim", "dropout"}),
                ("dec_cfg", dc, {"num_layers", "sa_heads", "ref_points", "attn_heads", "attn_samples", "ffn_dim",
                                 "dropout", "init_hidden"}),
                ("loss_cfg", lc, {"n_dis", "width_px", "cls_weight", "loc_weight", "len_weight", "end_weight",
                                  "focal_alpha", "focal_gamma", "min_gt_len_px"})):
            if set(cfg) - known:
                raise ValueError(f"BezierFormerHead {name}: unknown keys {sorted(set(cfg) - known)}")
        if dim % 32:
            raise ValueError("BezierFormerHead dim must be a multiple of 32 (GroupNorm)")
        self.num_queries, self.dim = int(num_queries), int(dim)
        self.img_w, self.img_h = int(img_w), int(img_h)
        self.train_cfg, self.test_cfg = train_cfg, test_cfg
        self.n_offsets = int(num_points)
        self.n_strips = self.n_offsets - 1
        self.register_buffer("prior_ys", torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("px_scale", torch.tensor([float(img_w - 1), float(img_h - 1)]), persistent=False)
        num_maps = len(in_channels) + 1

        # tokens and encoder
        self.in_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, dim, 1), nn.GroupNorm(32, dim)) for c in in_channels])
        self.in_proj.append(nn.Sequential(nn.Conv2d(in_channels[-1], dim, 3, stride=2, padding=1),
                                          nn.GroupNorm(32, dim)))
        self.level_embed = nn.Parameter(torch.zeros(num_maps, dim))
        self.enc = nn.ModuleList([CrossViewEncoderLayer(dim, num_maps, int(ec.get("num_heads", 8)),
                                                        int(ec.get("num_points", 4)), int(ec.get("ffn_dim", 1024)),
                                                        float(ec.get("dropout", 0.1)))
                                  for _ in range(int(ec.get("num_layers", 6)))])

        # queries
        self.num_layers = int(dc.get("num_layers", 6))
        self.ref_points = int(dc.get("ref_points", 5))
        drop = float(dc.get("dropout", 0.1))
        hid = int(dc.get("init_hidden", 512))
        self.query_embed = nn.Parameter(torch.zeros(self.num_queries, dim))                 # E_0
        self.init_mlp = _mlp([num_maps * dim, hid, hid, self.num_queries * 8])           # C_0 = MLP(X)
        self.register_buffer("t_ref", torch.linspace(0.0, 1.0, self.ref_points), persistent=False)
        self.pe_mlp = _mlp([self.ref_points * dim, dim, dim])                            # Eq. 3
        self.layers = nn.ModuleList([
            BezierFormerLayer(dim, int(dc.get("sa_heads", 8)), num_maps, int(dc.get("attn_heads", 8)),
                              int(dc.get("attn_samples", 5)), int(dc.get("ffn_dim", 1024)), drop)
            for _ in range(self.num_layers)])
        self.cls_heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.num_layers + 1)])
        self.reg_heads = nn.ModuleList([_mlp([dim, dim, dim, 8]) for _ in range(self.num_layers)])

        # loss
        self.n_dis = int(lc.get("n_dis", 200))
        self.width_px = float(lc.get("width_px", 10.0))
        self.w_cls, self.w_loc = float(lc.get("cls_weight", 1.0)), float(lc.get("loc_weight", 1.0))
        self.w_len, self.w_end = float(lc.get("len_weight", 1.0)), float(lc.get("end_weight", 1.0))
        self.alpha, self.gamma = float(lc.get("focal_alpha", 0.25)), float(lc.get("focal_gamma", 2.0))
        self.min_gt_len_px = float(lc.get("min_gt_len_px", 8.0))
        self.register_buffer("t_dis", torch.linspace(0.0, 1.0, self.n_dis), persistent=False)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.in_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        nn.init.normal_(self.query_embed)                                                # nn.Embedding's default
        for layer in list(self.enc) + list(self.layers):
            layer.reset_parameters()
        for mlp in (self.init_mlp, self.pe_mlp):
            for lin in mlp:
                if isinstance(lin, nn.Linear):
                    nn.init.xavier_uniform_(lin.weight)
                    nn.init.zeros_(lin.bias)
        nn.init.zeros_(self.init_mlp[-1].weight)
        with torch.no_grad():
            self.init_mlp[-1].bias.copy_(self.fan_lines(self.num_queries).reshape(-1))
        prior = -math.log((1 - 0.01) / 0.01)
        for lin in self.cls_heads:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, prior)
        for mlp in self.reg_heads:
            for lin in mlp:
                if isinstance(lin, nn.Linear):
                    nn.init.xavier_uniform_(lin.weight)
                    nn.init.zeros_(lin.bias)
            nn.init.zeros_(mlp[-1].weight)

    @staticmethod
    def fan_lines(num):
        """C_0 at initialization: num straight lines (normalized control points [num, 4, 2], bottom first) from the
        bottom edge, x spread over [-0.25, 1.25], towards the upper middle (0.5 + 0.3 (x_b - 0.5), 0.35)."""
        xb = torch.linspace(-0.25, 1.25, num)
        bottom = torch.stack([xb, torch.ones(num)], -1)
        top = torch.stack([0.5 + 0.3 * (xb - 0.5), torch.full((num,), 0.35)], -1)
        t = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])[None, :, None]
        return bottom[:, None] + t * (top - bottom)[:, None]

    def _px(self, xy):
        return xy * self.px_scale.to(xy.dtype)

    # ------------------------------------------------------------------ forward
    def _tokens(self, feats):
        """Encoded tokens [B, T, D], the maps and the frame."""
        proj = [self.in_proj[lvl](f) for lvl, f in enumerate(feats)] + [self.in_proj[-1](feats[-1])]
        batch_size, dev = proj[0].shape[0], proj[0].device
        frame = ViewFrame(self.img_w, self.img_h, 0, 0, None)
        maps, toks, poss, refs, start = [], [], [], [], 0
        for lvl, f in enumerate(proj):
            h, w = f.shape[-2:]
            cell = (self.img_w / w, self.img_h / h)
            ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                                    torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
            ref = torch.stack([(xs.reshape(-1) + 0.5) * cell[0], (ys.reshape(-1) + 0.5) * cell[1]], dim=-1)
            square = abs(cell[0] - cell[1]) <= 1e-6
            maps.append(_Map(CTX, lvl, h, w, start, cell[0] if square else cell))
            start += h * w
            toks.append(f.flatten(2).transpose(1, 2))
            refs.append(ref[None].expand(batch_size, -1, -1))
            poss.append(sine_embed(ref / ref.new_tensor([self.img_w, self.img_h]), self.dim) + self.level_embed[lvl])
        x = torch.cat(toks, dim=1).float()
        pos = torch.cat(poss, dim=0)[None].expand(batch_size, -1, -1)
        ref = torch.cat(refs, dim=1)
        for layer in self.enc:
            x = layer(x, pos, ref, maps, frame, None)
        return x, maps, frame

    def _forward(self, feats):
        tokens, maps, frame = self._tokens(feats)
        batch_size = tokens.shape[0]
        pooled = torch.cat([tokens[:, mp.start:mp.start + mp.n].mean(1) for mp in maps], dim=-1)
        c = self.init_mlp(pooled).float().view(batch_size, self.num_queries, 4, 2)          # C_0
        e = self.query_embed[None].expand(batch_size, -1, -1).to(tokens.dtype)              # E_0
        logits, cps = [self.cls_heads[0](e).squeeze(-1).float()], [c]
        offs = []
        for i, layer in enumerate(self.layers):
            c_in = c.detach()
            pts = curve_points(c_in, self.t_ref)                                              # [B, N, K, 2]
            pe = self.pe_mlp(sine_embed(pts, self.dim).flatten(-2).to(e.dtype))
            e = layer(e, pe, self._px(pts), tokens, maps, frame)
            logits.append(self.cls_heads[i + 1](e).squeeze(-1).float())
            c = c_in + self.reg_heads[i](e).float().view(batch_size, self.num_queries, 4, 2)
            cps.append(c)
            offs.append(layer.curve.last_offset_px)
        return dict(logits=logits, cps=cps, step_off_px=torch.stack(offs).mean() if offs else None)

    def forward(self, feats, *args, **kwargs):
        out = self._forward(feats)
        if self.training:
            return out
        return StagePredictions([self.to_official_pred_dict(self._official(out["logits"][-1], out["cps"][-1]))])

    # ------------------------------------------------------------------ losses
    def extract_gts(self, batch_data_samples, device):
        """The augmented annotation polylines (``gt_points``) -> per image [G, n_dis, 2] px, bottom -> top, uniform
        in arc length (at most num_queries lanes per image)."""
        out = []
        for sample in batch_data_samples:
            value, _ = _lookup(sample, ["gt_points"])
            if value is None:
                raise KeyError("BezierFormerHead needs 'gt_points' in the train meta_keys (PackCLRNetInputs)")
            lanes = [gt_polyline(lane, self.img_w, self.img_h, self.n_dis, self.min_gt_len_px) for lane in value]
            lanes = [ln for ln in lanes if ln is not None][:self.num_queries]
            g = (torch.as_tensor(np.stack(lanes), device=device) if lanes
                 else torch.zeros((0, self.n_dis, 2), device=device))
            out.append(self._px(g.float()))
        return out

    def _reg_terms(self, p_px, g_px, pairwise):
        """L_loc, L_len, L_endpoint of curves p_px [.., n_dis, 2] against GT g_px (px; pairwise as chamfer_iou)."""
        ciou = chamfer_iou(p_px, g_px, self.width_px, pairwise)
        lp, lg = polyline_length(p_px), polyline_length(g_px).clamp_min(1e-6)
        pn, gn = p_px / self.px_scale, g_px / self.px_scale
        if pairwise:
            lp, lg = lp[:, :, None], lg[:, None]
            d0 = (pn[:, :, None, 0] - gn[:, None, :, 0]).norm(dim=-1)
            d1 = (pn[:, :, None, -1] - gn[:, None, :, -1]).norm(dim=-1)
        else:
            d0, d1 = (pn[..., 0, :] - gn[..., 0, :]).norm(dim=-1), (pn[..., -1, :] - gn[..., -1, :]).norm(dim=-1)
        return 1.0 - ciou, (lp / lg - 1.0).abs(), 0.5 * (d0 + d1), ciou

    def layer_loss(self, logit, cps, gts):
        """One layer's losses (logit [B, N], control points [B, N, 4, 2]) and its matching."""
        batch_size, num_q = logit.shape
        pts = self._px(curve_points(cps, self.t_dis))                                      # [B, N, n_dis, 2]
        pos, neg = focal_terms(logit, self.alpha, self.gamma)
        target = torch.zeros_like(logit, dtype=torch.bool)
        pairs_q, pairs_g, covered = [], [], 0
        with torch.no_grad():
            for b in range(batch_size):
                g = gts[b]
                if g.shape[0] == 0:
                    continue
                loc, ln, end, _ = self._reg_terms(pts[b:b + 1].float(), g[None], pairwise=True)
                cost = (self.w_loc * loc + self.w_len * ln + self.w_end * end)[0] \
                    + self.w_cls * (pos[b] - neg[b])[:, None]
                match = simota_top1(cost)
                q = torch.nonzero(match >= 0, as_tuple=False).squeeze(1)
                target[b, q] = True
                pairs_q.append((b, q))
                pairs_g.append(match[q])
                covered += int(match[q].unique().numel())
        num_pos = float(max(1, int(target.sum())))
        loss_cls = torch.where(target, pos, neg).sum() / num_pos * self.w_cls
        zero = cps.sum() * 0.0
        out = dict(loss_cls=loss_cls, loss_loc=zero, loss_len=zero, loss_end=zero)
        if pairs_q:
            p = torch.cat([pts[b, q] for b, q in pairs_q])
            g = torch.cat([gts[b][gi] for (b, _), gi in zip(pairs_q, pairs_g)])
            loc, ln, end, ciou = self._reg_terms(p.float(), g, pairwise=False)
            out.update(loss_loc=zero + self.w_loc * loc.sum() / num_pos,
                       loss_len=zero + self.w_len * ln.sum() / num_pos,
                       loss_end=zero + self.w_end * end.sum() / num_pos)
            out["ciou"] = ciou.detach().mean()
        else:
            out["ciou"] = zero.detach()
        n_gt = sum(int(g.shape[0]) for g in gts)
        out["match_frac"] = logit.new_tensor(covered / max(1, n_gt))
        with torch.no_grad():
            prob = logit.sigmoid()
            out["pos_prob"] = prob[target].mean() if bool(target.any()) else prob.new_tensor(0.0)
            out["neg_prob"] = prob[~target].mean() if bool((~target).any()) else prob.new_tensor(0.0)
        return out

    def loss(self, feats, batch_data_samples, *args, **kwargs):
        out = self._forward(feats)
        gts = self.extract_gts(batch_data_samples, out["cps"][0].device)
        losses = {}
        last = len(out["cps"]) - 1
        for i, (logit, cps) in enumerate(zip(out["logits"], out["cps"])):
            terms = self.layer_loss(logit, cps, gts)
            if i == last:
                losses.update(terms)
            else:
                losses.update({f"d{i}.{k}": v for k, v in terms.items() if k.startswith("loss") or k == "ciou"})
        losses["num_gt"] = out["cps"][0].new_tensor(sum(int(g.shape[0]) for g in gts) / max(1, len(gts)))
        if out["step_off_px"] is not None:
            losses["step_off_px"] = out["step_off_px"]
        return losses

    # ------------------------------------------------------------------ test
    @staticmethod
    def to_official_pred_dict(pred):
        return {"cls_logits": pred[..., :2], "anchor_params": pred[..., 2:5],
                "lengths": pred[..., 5:6], "xs": pred[..., 6:]}

    def _official(self, logit, cps):
        """[B, N, 6 + R] official rows of the curves (bottom end first), as BLRHead.official_layout; the 2-class
        logits (0, z) give softmax scores equal to the sigmoid of z."""
        pts = curve_points(cps, torch.linspace(0.0, 1.0, 73, device=cps.device, dtype=cps.dtype))
        pts, c = bottom_first(pts, cps)
        xs, start_y, length = curve_to_rows(pts, self.prior_ys, self.n_strips, end_slopes=bezier_end_slopes(c))
        row = ((1.0 - start_y) * self.n_strips).round().long().clamp(0, self.n_strips)
        start_x = xs.gather(-1, row[..., None]).squeeze(-1)
        with torch.no_grad():
            dx = (pts[..., -1, 0] - pts[..., 0, 0]) * float(self.img_w - 1)
            dy = (pts[..., 0, 1] - pts[..., -1, 1]).clamp_min(1e-6) * float(self.img_h - 1)
            theta = (torch.atan2(dy, dx) / math.pi).clamp(0.01, 0.99)
        logits = torch.stack([torch.zeros_like(logit), logit], dim=-1)
        return torch.cat([logits.to(pts.dtype), start_y[..., None], start_x[..., None], theta[..., None],
                          length[..., None], xs], dim=-1)

    def predict(self, feats, batch_data_samples, rescale=False):
        cfg = self.test_cfg
        output = cfg.get("output", "rows") if hasattr(cfg, "get") else "rows"
        if output == "rows":
            lanes, scores = self.get_lanes(self(feats)[-1], as_lanes=cfg.as_lanes, extend_bottom=cfg.extend_bottom)
        elif output == "points":
            lanes, scores = self.point_lanes(self._forward(feats))
        else:
            raise ValueError("BezierFormerHead test_cfg.output must be 'rows' or 'points'")
        return [{"lanes": ln, "scores": sc, "metainfo": ds.metainfo}
                for ln, sc, ds in zip(lanes, scores, batch_data_samples)]

    def point_lanes(self, out):
        """test_cfg.output "points": the last layer's curves as Lane points (out_points samples inside the image,
        normalized, y increasing), scores >= conf_threshold, at most nms_topk per image."""
        cfg = self.test_cfg
        n = int(cfg.get("out_points", 100))
        thr, topk = float(cfg.conf_threshold), int(cfg.get("nms_topk") or self.num_queries)
        score = out["logits"][-1].sigmoid()
        pts = curve_points(out["cps"][-1], torch.linspace(0.0, 1.0, n, device=score.device)).float()
        all_lanes, all_scores = [], []
        for b in range(score.shape[0]):
            keep = torch.nonzero(score[b] >= thr, as_tuple=False).squeeze(1)
            keep = keep[torch.argsort(score[b, keep], descending=True)][:topk]
            lanes, scores = [], []
            for q in keep.tolist():
                p = pts[b, q].cpu().numpy().astype(np.float64)
                p = p[(p[:, 0] >= 0) & (p[:, 0] < 1) & (p[:, 1] >= 0) & (p[:, 1] <= 1)]
                if len(p) < 2:
                    continue
                p = p[np.argsort(p[:, 1], kind="stable")]
                p = p[np.concatenate([[True], np.diff(p[:, 1]) > 1e-6])]
                if len(p) < 2:
                    continue
                s = float(score[b, q])
                lanes.append(Lane(points=p, metadata={"conf": s}) if cfg.as_lanes else p)
                scores.append(s)
            all_lanes.append(lanes)
            all_scores.append(scores)
        return all_lanes, all_scores

    def get_lanes(self, pred_dict, as_lanes=True, extend_bottom=True):
        """CLRerHead.get_lanes (libs/models/dense_heads/clrernet_head.py)."""
        softmax = nn.Softmax(dim=2)
        threshold = self.test_cfg.conf_threshold
        all_scores = softmax(pred_dict["cls_logits"])[:, :, 1]
        all_keep_inds = all_scores >= threshold
        out_preds, out_scores = [], []
        for scores, xs, lengths, anchor_params, keep_inds in zip(
                all_scores, pred_dict["xs"], pred_dict["lengths"], pred_dict["anchor_params"], all_keep_inds):
            scores, xs, lengths, anchor_params = scores[keep_inds], xs[keep_inds], lengths[keep_inds], \
                anchor_params[keep_inds]
            if xs.shape[0] == 0:
                out_preds.append([])
                out_scores.append([])
                continue
            if self.test_cfg.use_nms:
                from nms import nms
                nms_anchor_params = anchor_params[..., :2].detach().clone()
                nms_anchor_params[..., 0] = 1 - nms_anchor_params[..., 0]
                nms_predictions = torch.cat([pred_dict["cls_logits"][0, keep_inds].detach().clone(),
                                             nms_anchor_params[..., :2], lengths.detach().clone() * self.n_strips,
                                             xs.detach().clone() * (self.img_w - 1)], dim=-1)
                keep, num_to_keep, _ = nms(nms_predictions, scores, overlap=self.test_cfg.nms_thres,
                                           top_k=self.test_cfg.nms_topk)
                keep = keep[:num_to_keep]
                xs, scores, lengths, anchor_params = xs[keep], scores[keep], lengths[keep], anchor_params[keep]
            lengths = torch.round(lengths * self.n_strips)
            out_preds.append(self.predictions_to_lanes(xs, anchor_params, lengths, scores, as_lanes, extend_bottom))
            out_scores.append(scores)
        return out_preds, out_scores

    def predictions_to_lanes(self, pred_xs, anchor_params, lengths, scores, as_lanes=True, extend_bottom=True):
        """CLRerHead.predictions_to_lanes (libs/models/dense_heads/clrernet_head.py)."""
        prior_ys = self.prior_ys.double()
        lanes = []
        for lane_xs, lane_param, length, score in zip(pred_xs, anchor_params, lengths, scores):
            start = min(max(0, int(round((1 - lane_param[0].item()) * self.n_strips))), self.n_strips)
            length = int(round(length.item()))
            end = min(start + length - 1, len(prior_ys) - 1)
            if extend_bottom:
                edge = (lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)
                start -= edge.flip(0).cumprod(dim=0).sum()
            lane_ys = prior_ys[start:end + 1]
            lane_xs = lane_xs[start:end + 1].flip(0).double()
            lane_ys = lane_ys.flip(0)
            lane_ys = (lane_ys * (self.test_cfg.ori_img_h - self.test_cfg.cut_height)
                       + self.test_cfg.cut_height) / self.test_cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            if as_lanes:
                lanes.append(Lane(points=points.cpu().numpy(),
                                  metadata={"start_x": lane_param[1], "start_y": lane_param[0], "conf": score}))
            else:
                lanes.append(points)
        return lanes
