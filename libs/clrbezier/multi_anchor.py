"""MultiAnchorFormerHead: the BezierFormer replica (bezierformer.py) whose queries carry each lane in three anchored
forms (docs/multi_anchor/multi_anchor_former.md). Everything outside the lane representation is the replica's: backbone,
4-scale deformable encoder, E_0 / C_0 = MLP(X) queries, the 6 decoder layers (self-attention + Bezier curve attention
at 5 reference points + FFN), per-layer focal classification, the per-layer loss (Chamfer IoU + length + endpoint),
SimOTA top-1 (or Hungarian) matching on the paper's cost, and the test path (72 official rows or dense points).

Forms (anchor_cfg.forms), each with its own C_0 MLP (zero last weights; bias = the replica's fan of straight lines in
that form's parameters, so all forms start as the same 16 lines) and its own per-layer update FFN on E_i, refined as
the replica refines C: state_i = state_{i-1} (detached) + FFN_i(E_i).
- "B" Bezier (BezierFormer): 4 free 2D control points (normalized), no y order. Exactly the replica's C_0 MLP and
  update FFNs (the replica's own modules).
- "S" line anchor (CLRNet): start point (x_s, y_s) at the near end, angle theta_s of the line to the image x axis (as
  CLRNet's theta, (0, pi), the line runs up from the start), the far end's row y_e, and x offsets dx_j at `rows` rows
  spaced evenly from y_s to y_e: x_j = x_s + (y_s - y_j) cot(theta_s) + dx_j.
- "V" polar anchor (Polar R-CNN, arXiv 2411.01499, Eq. A2 / 7): a line about one FIXED global pole c (anchor_cfg.pole_px,
  set near the dataset's static vanishing point; tools/clrbezier/static_pole.py measures it), in the normal form
  cos(theta_v) (x - c_x) - sin(theta_v) (y - c_y) = r (image axes, y down; Polar R-CNN's y-up form with y -> -y),
  theta_v in (-pi/2, pi/2), r signed (normalized by img_w - 1), plus the same near / far rows y_s, y_e and x offsets
  dx_j: x_j = c_x + (r + sin(theta_v) (y_j - c_y)) / cos(theta_v) + dx_j.
  S and V are Polar R-CNN's / CLRNet's "straight anchor + per-row x offsets", with two changes the replica's losses
  need: the rows are spaced over the lane's own extent (fixed size, differentiable ends) instead of over the whole
  image, and there are 8 of them, not 72. CLRNet / Polar R-CNN supervise every row's x (smooth L1 per row); the
  replica's losses (Chamfer IoU, length, endpoint) have no row correspondence, so 36 free offsets went unconstrained
  and jagged and slowed every form (CPU overfit: S+B 3/5 with 36 rows, 5/5 with 8, S's own Chamfer IoU 0.16 -> 0.74;
  36 offsets on fixed image rows: 3/5). 8 rows give S / V 12 numbers per lane against B's 8.

Fusion: every form's polyline resampled at fuse_points arc-length fractions u (0 = near end, 1 = far end);
p(u) = sum_k w_k(u) p_k(u), w = softmax over the forms of a learned profile (knots per form, linear in u, shared by
all lanes and layers; zeros at the start = equal weights). The fused curve is the lane: reference points and position
encoding of the next layer, matching, the replica's loss, and the output.
Loss: + form_loss_weight x the replica's regression terms (Chamfer IoU, length, endpoint) of each form on the fused
curve's pairs, every layer (loss_form_loc / _len / _end).
anchor_cfg.output ("fused" | a form): the test output's curve (one form alone; the scores stay the model's).
Logs: form_w_<k>_near / _mid / _far (the profile at u = 0, 0.5, 1), form_ciou_<k> (last layer's pairs)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

from libs.utils.lane_utils import Lane

from .bezierformer import ASSIGNERS, BezierFormerHead, _mlp, focal_terms
from .blrnet import bezier_end_slopes, curve_points, curve_to_rows
from .foveal_lane import arc_resample, polyline_at, sine_embed
from .hblrnet import bottom_first
from .head import StagePredictions

FORMS = ("S", "B", "V")
THETA_S_RANGE = (0.02 * math.pi, 0.98 * math.pi)      # S: CLRNet's angle, kept off the horizontal
THETA_V_LIMIT = 0.48 * math.pi                         # V: |normal angle| < pi/2 (a horizontal line has no x(y))


def _ends(pts):
    """Pseudo control points [..., 4, 2] of polylines pts [..., P, 2]: their end segments give the end tangents."""
    return torch.stack([pts[..., 0, :], pts[..., 1, :], pts[..., -2, :], pts[..., -1, :]], dim=-2)


@MODELS.register_module()
class MultiAnchorFormerHead(BezierFormerHead):
    """The replica head with anchor_cfg (module docstring): forms (("S", "B", "V"), must include "B"), rows (8, x
    offsets of S and V; see the module docstring), knots (8), form_loss_weight (0.5), pole_px ((399.5, 22.8): the replica's fan of initial
    lines meets there; set it to the measured static vanishing point), fuse_points (73), output ("fused")."""

    def __init__(self, anchor_cfg=None, **kwargs):
        ac = dict(anchor_cfg or {})
        known = {"forms", "rows", "knots", "form_loss_weight", "pole_px", "fuse_points", "output"}
        if set(ac) - known:
            raise ValueError(f"MultiAnchorFormerHead anchor_cfg: unknown keys {sorted(set(ac) - known)}")
        super().__init__(**kwargs)
        self.forms = tuple(str(k) for k in ac.get("forms", FORMS))
        self.rows = int(ac.get("rows", 8))
        self.knots = int(ac.get("knots", 8))
        self.form_loss_weight = float(ac.get("form_loss_weight", 0.5))
        self.fuse_points = int(ac.get("fuse_points", 73))
        self.form_output = str(ac.get("output", "fused"))
        pole = tuple(float(v) for v in ac.get("pole_px", (399.5, 22.8)))
        if (not self.forms or len(set(self.forms)) != len(self.forms) or set(self.forms) - set(FORMS)
                or "B" not in self.forms or self.rows < 2 or self.knots < 2 or self.form_loss_weight < 0
                or self.fuse_points < 4 or len(pole) != 2 or self.form_output not in ("fused",) + self.forms):
            raise ValueError("MultiAnchorFormerHead anchor_cfg: forms distinct of 'S', 'B', 'V' incl. 'B'; rows, "
                             "knots >= 2; form_loss_weight >= 0; fuse_points >= 4; pole_px (x, y); output 'fused' or "
                             "a form")
        self.register_buffer("pole", torch.tensor(pole), persistent=False)
        self.register_buffer("t_rows", torch.linspace(0.0, 1.0, self.rows), persistent=False)
        self.register_buffer("t_dense", torch.linspace(0.0, 1.0, 2 * self.fuse_points - 1), persistent=False)
        num_maps = len(self.in_proj)
        hid = self.init_mlp[0].out_features
        self.form_init, self.form_reg = nn.ModuleDict(), nn.ModuleDict()
        for k in self.forms:
            if k == "B":                                # the replica's own C_0 MLP and update FFNs
                continue
            n = 4 + self.rows
            init = _mlp([num_maps * self.dim, hid, hid, self.num_queries * n])
            regs = nn.ModuleList([_mlp([self.dim, self.dim, self.dim, n]) for _ in range(self.num_layers)])
            for mlp in [init] + list(regs):
                for lin in mlp:
                    if isinstance(lin, nn.Linear):
                        nn.init.xavier_uniform_(lin.weight)
                        nn.init.zeros_(lin.bias)
                nn.init.zeros_(mlp[-1].weight)
            with torch.no_grad():
                init[-1].bias.copy_(self.fan_params(k).reshape(-1))
            self.form_init[k], self.form_reg[k] = init, regs
        self.form_profile = nn.Parameter(torch.zeros(len(self.forms), self.knots))

    # ------------------------------------------------------------------ forms
    def fan_params(self, k):
        """The replica's fan of initial straight lines (fan_lines) in form k's parameters [N, 4 + rows]."""
        c = self.fan_lines(self.num_queries)                                              # [N, 4, 2] normalized
        w1, h1 = float(self.img_w - 1), float(self.img_h - 1)
        bot, top = c[:, 0], c[:, 3]
        dx, dy = (top[:, 0] - bot[:, 0]) * w1, (top[:, 1] - bot[:, 1]) * h1                 # px, dy < 0 (upwards)
        zeros = torch.zeros(c.shape[0], self.rows)
        if k == "S":
            theta = torch.atan2(-dy, dx)                                                    # (0, pi)
            return torch.cat([torch.stack([bot[:, 0], bot[:, 1], theta, top[:, 1]], -1), zeros], -1)
        theta = torch.atan(dx / dy)                                     # cos(t) dx = sin(t) dy on the line
        cx, cy = float(self.pole[0]), float(self.pole[1])
        r = (torch.cos(theta) * (bot[:, 0] * w1 - cx) - torch.sin(theta) * (bot[:, 1] * h1 - cy)) / w1
        return torch.cat([torch.stack([theta, r, bot[:, 1], top[:, 1]], -1), zeros], -1)

    def form_points(self, k, z):
        """Form k's polyline (normalized, near end first): B [B, N, 4, 2] control points -> 2 fuse_points - 1 curve
        samples; S / V [B, N, 4 + rows] -> rows points."""
        if k == "B":
            return curve_points(z, self.t_dense)
        w1, h1 = float(self.img_w - 1), float(self.img_h - 1)
        y0, y1 = (z[..., 1], z[..., 3]) if k == "S" else (z[..., 2], z[..., 3])
        y = (y0[..., None] + (y1 - y0)[..., None] * self.t_rows) * h1                       # [B, N, R] px
        dx = z[..., 4:] * w1
        if k == "S":
            th = z[..., 2].clamp(*THETA_S_RANGE)
            x = z[..., 0, None] * w1 + (y0[..., None] * h1 - y) * (torch.cos(th) / torch.sin(th))[..., None] + dx
        else:
            th = z[..., 0].clamp(-THETA_V_LIMIT, THETA_V_LIMIT)
            cx, cy = self.pole[0], self.pole[1]
            x = cx + (z[..., 1, None] * w1 + torch.sin(th)[..., None] * (y - cy)) / torch.cos(th)[..., None] + dx
        return torch.stack([x / w1, y / h1], dim=-1)

    def _weights(self, dtype):
        a = F.interpolate(self.form_profile[None].float(), size=self.fuse_points, mode="linear", align_corners=True)[0]
        return torch.softmax(a, dim=0).to(dtype)                                            # [K, fuse_points]

    def fuse(self, states):
        """-> (per-form polylines resampled at fuse_points arc-length fractions [list of B, N, P, 2], fused)."""
        curves = [arc_resample(self.form_points(k, states[k]), self.fuse_points) for k in self.forms]
        w = self._weights(curves[0].dtype)
        return curves, sum(w[i][None, None, :, None] * c for i, c in enumerate(curves))

    # ------------------------------------------------------------------ forward
    def _forward(self, feats):
        tokens, maps, frame = self._tokens(feats)
        batch_size = tokens.shape[0]
        pooled = torch.cat([tokens[:, mp.start:mp.start + mp.n].mean(1) for mp in maps], dim=-1)
        states = {}
        for k in self.forms:
            if k == "B":
                states[k] = self.init_mlp(pooled).float().view(batch_size, self.num_queries, 4, 2)          # C_0
            else:
                states[k] = self.form_init[k](pooled).float().view(batch_size, self.num_queries, -1)
        e = self.query_embed[None].expand(batch_size, -1, -1).to(tokens.dtype)              # E_0
        form_curves, fused = self.fuse(states)
        logits, lanes, forms_out, offs = [self.cls_heads[0](e).squeeze(-1).float()], [fused], [form_curves], []
        for i, layer in enumerate(self.layers):
            z_in = {k: v.detach() for k, v in states.items()}
            pts = polyline_at(fused.detach(), self.t_ref)                                      # [B, N, K, 2]
            pe = self.pe_mlp(sine_embed(pts, self.dim).flatten(-2).to(e.dtype))
            e = layer(e, pe, self._px(pts), tokens, maps, frame)
            logits.append(self.cls_heads[i + 1](e).squeeze(-1).float())
            for k in self.forms:
                if k == "B":
                    states[k] = z_in[k] + self.reg_heads[i](e).float().view(batch_size, self.num_queries, 4, 2)
                else:
                    states[k] = z_in[k] + self.form_reg[k][i](e).float()
            form_curves, fused = self.fuse(states)
            lanes.append(fused)
            forms_out.append(form_curves)
            offs.append(layer.curve.last_offset_px)
        return dict(logits=logits, lanes=lanes, form_curves=forms_out, cps=None,
                    step_off_px=torch.stack(offs).mean() if offs else None)

    def _output_lanes(self, out):
        if self.form_output == "fused":
            return out["lanes"][-1]
        return out["form_curves"][-1][self.forms.index(self.form_output)]

    def forward(self, feats, *args, **kwargs):
        out = self._forward(feats)
        if self.training:
            return out
        return StagePredictions([self.to_official_pred_dict(self._official_pts(out["logits"][-1],
                                                                                self._output_lanes(out)))])

    # ------------------------------------------------------------------ losses
    def layer_loss_lanes(self, logit, lanes, form_curves, gts):
        """BezierFormerHead.layer_loss on polylines lanes [B, N, P, 2] (normalized), plus the form terms on the same
        pairs."""
        batch_size, num_q = logit.shape
        pts = self._px(arc_resample(lanes, self.n_dis))                                     # [B, N, n_dis, 2]
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
                    + self.w_cls * (pos[b] if self.cls_cost == "focal" else pos[b] - neg[b])[:, None]
                match = ASSIGNERS[self.assign](cost)
                q = torch.nonzero(match >= 0, as_tuple=False).squeeze(1)
                target[b, q] = True
                pairs_q.append((b, q))
                pairs_g.append(match[q])
                covered += int(match[q].unique().numel())
        num_pos = float(max(1, int(target.sum())))
        loss_cls = torch.where(target, pos, neg).sum() / num_pos * self.w_cls
        zero = lanes.sum() * 0.0
        out = dict(loss_cls=loss_cls, loss_loc=zero, loss_len=zero, loss_end=zero,
                   loss_form_loc=zero, loss_form_len=zero, loss_form_end=zero)
        logs = {f"form_ciou_{k}": zero.detach() for k in self.forms}
        if pairs_q:
            g = torch.cat([gts[b][gi] for (b, _), gi in zip(pairs_q, pairs_g)])
            p = torch.cat([pts[b, q] for b, q in pairs_q])
            loc, ln, end, ciou = self._reg_terms(p.float(), g, pairwise=False)
            out.update(loss_loc=zero + self.w_loc * loc.sum() / num_pos,
                       loss_len=zero + self.w_len * ln.sum() / num_pos,
                       loss_end=zero + self.w_end * end.sum() / num_pos)
            out["ciou"] = ciou.detach().mean()
            fl = fn = fe = zero
            for k, c in zip(self.forms, form_curves):
                fp = self._px(arc_resample(c, self.n_dis))
                fp = torch.cat([fp[b, q] for b, q in pairs_q])
                loc, ln, end, ciou = self._reg_terms(fp.float(), g, pairwise=False)
                fl, fn, fe = fl + loc.sum(), fn + ln.sum(), fe + end.sum()
                logs[f"form_ciou_{k}"] = ciou.detach().mean()
            wf = self.form_loss_weight / num_pos
            out.update(loss_form_loc=zero + wf * self.w_loc * fl, loss_form_len=zero + wf * self.w_len * fn,
                       loss_form_end=zero + wf * self.w_end * fe)
        else:
            out["ciou"] = zero.detach()
        out.update(logs)
        n_gt = sum(int(g.shape[0]) for g in gts)
        out["match_frac"] = logit.new_tensor(covered / max(1, n_gt))
        with torch.no_grad():
            prob = logit.sigmoid()
            out["pos_prob"] = prob[target].mean() if bool(target.any()) else prob.new_tensor(0.0)
            out["neg_prob"] = prob[~target].mean() if bool((~target).any()) else prob.new_tensor(0.0)
        return out

    def loss(self, feats, batch_data_samples, *args, **kwargs):
        out = self._forward(feats)
        gts = self.extract_gts(batch_data_samples, out["lanes"][0].device)
        losses = {}
        last = len(out["lanes"]) - 1
        for i, (logit, lanes, fc) in enumerate(zip(out["logits"], out["lanes"], out["form_curves"])):
            terms = self.layer_loss_lanes(logit, lanes, fc, gts)
            if i == last:
                losses.update(terms)
            else:
                losses.update({f"d{i}.{k}": v for k, v in terms.items() if k.startswith("loss") or k == "ciou"})
        losses["num_gt"] = out["lanes"][0].new_tensor(sum(int(g.shape[0]) for g in gts) / max(1, len(gts)))
        if out["step_off_px"] is not None:
            losses["step_off_px"] = out["step_off_px"]
        with torch.no_grad():
            w = self._weights(torch.float32)
            for i, k in enumerate(self.forms):
                for name, j in (("near", 0), ("mid", (w.shape[1] - 1) // 2), ("far", w.shape[1] - 1)):
                    losses[f"form_w_{k}_{name}"] = w[i, j]
        return losses

    # ------------------------------------------------------------------ test
    def _official_pts(self, logit, pts):
        """BezierFormerHead._official for polylines pts [B, N, P, 2] (normalized): their end segments give the end
        tangents that continue the lane beyond its ends."""
        pts, c = bottom_first(pts, _ends(pts))
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

    def point_lanes(self, out):
        """BezierFormerHead.point_lanes on the output polylines (out_points arc-length samples)."""
        cfg = self.test_cfg
        n = int(cfg.get("out_points", 100))
        thr, topk = float(cfg.conf_threshold), int(cfg.get("nms_topk") or self.num_queries)
        score = out["logits"][-1].sigmoid()
        pts = arc_resample(self._output_lanes(out), n).float()
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
