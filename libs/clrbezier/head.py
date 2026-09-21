"""CLRBezierHead: sparse Bezier-anchor CLR head for the official CLRerNet (mmdet 3.x).

Ported from UnLaneDet ``CLRDiffBezierHeadV11`` using only the modules active in
``resnet_v11_brr_collab_perturb_simota_lanewidth.py``:

    * 35 CLRerNet-style priors as support-local cubic Bezier CPs (+ small delta)
    * Bezier Reference Refinement (BRR): persistent [start_y, P0x..P3x],
      fresh length and fresh 72-row dense residual at every stage
    * main branch: Hungarian (one-to-one)
    * collaborative auxiliary branch: structured perturbation, M groups,
      TopK (stages 0/1) and SimOTA (stage 2)
    * losses: CE classification, LaneIoU, BRR support + CP, segmentation

Evaluation reuses the official CLRerHead decoding. In eval mode ``forward``
returns the official per-stage prediction dicts
(``cls_logits``, ``anchor_params``, ``lengths``, ``xs``), so NMS, lane decoding,
coordinate restoration, and result packing are the unmodified official code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from .assigners import build_cost_cache, build_lane_assigner
from .data_adapters import CLRTargetAdapter
from .alignment import QualityFocalLoss, ignore_unmatched, quality_targets
from .cascade import QualityGate, ReferenceReprojector, sampling_xs
from .geometry import eval_global_cubic
from .gsrc import GSRCModule
from .lateral import LateralEvidence
from .query_attention import MaskedQuerySelfAttention
from .geometry import brr_reference, fit_global_cubic_to_clr_rows, globalize_local_cp, update_brr_state
from .lane_iou import LaneIoULoss
from .modules import ROIGather, SegDecoder, linear_relu, pool_prior_features
from .priors import BezierPriorBank, StructuredPriorPerturbation


def _resolve_official_head():
    """Subclass the official CLRerHead so its predict/decoding path is inherited."""
    try:
        import libs.models  # noqa: F401  (registers the official CLRerNet modules)
    except ImportError:
        pass
    official = MODELS.get("CLRerHead")
    return official if official is not None else BaseModule


_OfficialHead = _resolve_official_head()


class StagePredictions(list):
    """Per-stage official prediction dicts.

    Behaves as a list (``outs[-1]``) and also as the final-stage dict
    (``outs["xs"]``), so it matches either calling style in the official
    predict path.
    """

    def __getitem__(self, key):
        if isinstance(key, str):
            return list.__getitem__(self, -1)[key]
        return list.__getitem__(self, key)

    def keys(self):
        return list.__getitem__(self, -1).keys()

    def values(self):
        return list.__getitem__(self, -1).values()

    def items(self):
        return list.__getitem__(self, -1).items()

    def get(self, key, default=None):
        return list.__getitem__(self, -1).get(key, default)


def _smooth_l1(pred, target, beta):
    diff = (pred - target).abs()
    if beta < 1e-6:
        return diff
    return torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


@MODELS.register_module()
class CLRBezierHead(_OfficialHead):
    def __init__(
        self,
        num_points=72,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_priors=35,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        img_w=800,
        img_h=320,
        roi_mid_channels=48,
        seg_num_classes=5,
        prior_cfg=None,
        brr_cfg=None,
        main_assigner=None,
        main_stage_assigners=None,
        main_quality_gate=None,
        reproject_cfg=None,
        aux_cfg=None,
        perturb_cfg=None,
        loss_cfg=None,
        gsrc_cfg=None,
        query_attn_cfg=None,
        lateral_cfg=None,
        aux_cls_head=False,
        target_adapter=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        **unused,
    ):
        # Skip the official CLRerHead.__init__ (it builds the 192-prior anchor
        # generator and straight-line regressor). Initialize the mmengine base only.
        BaseModule.__init__(self, init_cfg=init_cfg)
        if unused:
            print(f"[CLRBezierHead] ignoring unused head arguments: {sorted(unused)}")

        prior_cfg = dict(prior_cfg or {})
        brr_cfg = dict(brr_cfg or {})
        aux_given = aux_cfg is not None  # aux_cfg=None disables the collaborative branch
        aux_cfg = dict(aux_cfg or {})
        perturb_cfg = dict(perturb_cfg or {})
        loss_cfg = dict(loss_cfg or {})

        # ---- attributes read by the official decoding path -------------------
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.num_points = int(num_points)
        self.n_offsets = int(num_points)
        self.n_strips = int(num_points) - 1
        self.num_priors = int(num_priors)
        self.refine_layers = int(refine_layers)
        self.sample_points = int(sample_points)
        self.fc_hidden_dim = int(fc_hidden_dim)
        self.prior_feat_channels = int(prior_feat_channels)

        sample_idx = (torch.linspace(0, 1, steps=self.sample_points) * self.n_strips).long()
        self.register_buffer("sample_x_indices", sample_idx, persistent=False)
        self.register_buffer("sample_x_indexs", sample_idx.clone(), persistent=False)
        self.register_buffer(
            "prior_feat_ys", torch.flip(1 - sample_idx.float() / self.n_strips, dims=[-1]),
            persistent=False)
        self.register_buffer(
            "prior_ys", torch.linspace(1, 0, steps=self.n_offsets), persistent=False)

        # ---- priors / BRR -----------------------------------------------------
        self.eps = float(prior_cfg.get("eps", 1e-4))
        self.prior_bank = BezierPriorBank(
            self.num_priors, self.img_w, self.img_h,
            delta_scale=prior_cfg.get("delta_scale", 0.1), eps=self.eps,
            visible_only=prior_cfg.get("visible_only", True),
            min_support=prior_cfg.get("min_support", 1.0 / self.n_strips))
        self.cp_x_margin = float(brr_cfg.get("cp_x_margin", 0.5))

        # ---- network ----------------------------------------------------------
        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors, self.sample_points,
                                    self.fc_hidden_dim, self.refine_layers, roi_mid_channels)
        cls_modules, reg_modules = [], []
        for _ in range(num_fc):
            cls_modules += linear_relu(self.fc_hidden_dim)
            reg_modules += linear_relu(self.fc_hidden_dim)
        self.cls_modules = nn.ModuleList(cls_modules)
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)
        # [d_start_y, fresh_length, dP0x..dP3x, dense_x[R]]
        self.reg_layers = nn.Linear(self.fc_hidden_dim, 6 + self.n_offsets)
        self.seg_decoder = SegDecoder(self.img_h, self.img_w,
                                      self.prior_feat_channels * self.refine_layers, seg_num_classes)

        # ---- separate auxiliary classifier ------------------------------------
        # Only the final layer is separate: the auxiliary gradient still reaches
        # the shared towers, ROIGather and the backbone, which is where the
        # one-to-many supervision does its work.
        self.aux_cls_layers = nn.Linear(self.fc_hidden_dim, 2) if aux_cls_head else None

        # ---- lateral evidence -------------------------------------------------
        self.lateral = None
        if lateral_cfg:
            lateral_cfg = dict(lateral_cfg)
            self.lateral_stages = [int(v) for v in lateral_cfg.pop("stages", [0, 1, 2])]
            lateral_cfg.setdefault("in_channels", self.prior_feat_channels)
            lateral_cfg.setdefault("dim", self.fc_hidden_dim)
            lateral_cfg.setdefault("sample_points", self.sample_points)
            self.lateral = nn.ModuleDict(
                {str(i): LateralEvidence(**lateral_cfg) for i in self.lateral_stages})

        # ---- assignment -------------------------------------------------------
        self.main_assigner = build_lane_assigner(main_assigner or dict(
            type="HungarianLaneAssigner", cls_weight=1.0, point_weight=2.0, iou_weight=3.0))

        # Stage-wise main assigners (default: main_assigner for every stage).
        self.main_stage_assigners = {
            int(k): build_lane_assigner(v) for k, v in dict(main_stage_assigners or {}).items()}
        # Stage-increasing positive quality (Cascade R-CNN), per branch.
        self.main_gate = QualityGate(**dict(main_quality_gate)) if main_quality_gate else None
        aux_gate_cfg = aux_cfg.get("quality_gate") if aux_given else None
        self.aux_gate = QualityGate(**dict(aux_gate_cfg)) if aux_gate_cfg else None
        self.aux_enabled = aux_given and bool(aux_cfg.get("enabled", True))
        self.aux_num_groups = int(aux_cfg.get("num_groups", 3))
        self.aux_stages = [int(s) for s in aux_cfg.get("stages", [0, 1, 2])]
        self.aux_assigners = [build_lane_assigner(a) for a in aux_cfg.get("assigners", [
            dict(type="TopKLaneAssigner", topk=4, cls_weight=0.0, point_weight=2.0, iou_weight=3.0),
            dict(type="SimOTALaneAssigner", candidate_topk=10, min_dynamic_k=1,
                 cls_weight=0.25, point_weight=1.0, iou_weight=3.0),
        ])]
        self.aux_assigner_weights = [float(w) for w in aux_cfg.get(
            "assigner_weights", [1.0] * len(self.aux_assigners))]
        stage_ids = aux_cfg.get("stage_assigner_ids", {0: [0], 1: [0], 2: [1]})
        self.aux_stage_assigner_ids = {int(k): [int(i) for i in v] for k, v in dict(stage_ids).items()}
        self.aux_cls_loss_weight = float(aux_cfg.get("cls_loss_weight", 0.5))
        self.aux_reg_loss_weight = float(aux_cfg.get("reg_loss_weight", 0.5))
        self.aux_noise_t = int(aux_cfg.get("noise_t", 50))
        self.aux_random_t = bool(aux_cfg.get("random_t", True))
        self.aux_apply_brr = bool(aux_cfg.get("apply_brr_loss", True))
        self.perturbation = StructuredPriorPerturbation(eps=self.eps, **perturb_cfg)

        # ---- reference re-projection -----------------------------------------
        self.reproject = None
        self.reproject_stages = []
        if reproject_cfg:
            reproject_cfg = dict(reproject_cfg)
            self.reproject_stages = [int(v) for v in reproject_cfg.pop("stages", [0, 1])]
            reproject_cfg.setdefault("margin", self.cp_x_margin)
            self.reproject = ReferenceReprojector(self.prior_ys, self.img_w, self.n_strips,
                                                  **reproject_cfg)
        # training-step counter for warmups (saved with checkpoints for resume)
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long))
        self._step_cache = 0  # python mirror, refreshed once per training iteration
        self._need_input_xs = any(
            g is not None and g.gate_on == "input" for g in (self.main_gate, self.aux_gate))

        # ---- global context (GSRC) --------------------------------------------
        # Zero-initialized injection: with gsrc_cfg set, the model still starts
        # numerically identical to the model without it.
        self.gsrc = None
        if gsrc_cfg:
            gsrc_cfg = dict(gsrc_cfg)
            gsrc_cfg.setdefault("in_channels", self.prior_feat_channels)
            gsrc_cfg.setdefault("dim", self.fc_hidden_dim)
            gsrc_cfg.setdefault("refine_layers", self.refine_layers)
            self.gsrc_source = gsrc_cfg.pop("source", "coarsest")
            if self.gsrc_source != "coarsest":
                raise ValueError(
                    "gsrc_cfg.source='backbone' needs a detector that passes backbone "
                    "features to the head; only 'coarsest' (coarsest FPN level) is supported.")
            self.gsrc = GSRCModule(**gsrc_cfg)

        # ---- query-to-query attention -----------------------------------------
        # Group-masked (auxiliary groups never mix), anchor-geometry positional
        # bias, zero-gated: identical to the baseline at initialization.
        self.query_attn = None
        if query_attn_cfg:
            query_attn_cfg = dict(query_attn_cfg)
            self.query_attn_stages = [int(s) for s in query_attn_cfg.pop("stages", [0, 1, 2])]
            query_attn_cfg.setdefault("dim", self.fc_hidden_dim)
            query_attn_cfg.setdefault("state_dim", 5)  # [y_start, P0x..P3x]
            share = bool(query_attn_cfg.pop("share_across_stages", False))
            if share:
                shared = MaskedQuerySelfAttention(**query_attn_cfg)
                self.query_attn = nn.ModuleDict({str(i): shared for i in self.query_attn_stages})
            else:
                self.query_attn = nn.ModuleDict(
                    {str(i): MaskedQuerySelfAttention(**query_attn_cfg)
                     for i in self.query_attn_stages})

        # ---- losses -----------------------------------------------------------
        self.cls_loss_weight = float(loss_cfg.get("cls_loss_weight", 2.0))
        self.use_focal = bool(loss_cfg.get("use_focal", False))
        self.focal_alpha = float(loss_cfg.get("focal_alpha", 0.25))
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.cls_bg_weight = float(loss_cfg.get("cls_bg_weight", 0.4))
        # Confidence-localization alignment (see alignment.py).
        self.cls_target_mode = loss_cfg.get("cls_target_mode", "hard")  # hard | iou | task_aligned
        if self.cls_target_mode not in ("hard", "iou", "task_aligned"):
            raise ValueError(f"Unknown cls_target_mode {self.cls_target_mode!r}")
        # the auxiliary branch may use a different target mode (its fixed top-k
        # assignment includes weak pairs, which soft targets down-weight for free)
        self.aux_cls_target_mode = loss_cfg.get("aux_cls_target_mode", self.cls_target_mode)
        if self.aux_cls_target_mode not in ("hard", "iou", "task_aligned"):
            raise ValueError(f"Unknown aux_cls_target_mode {self.aux_cls_target_mode!r}")
        self.task_align_alpha = float(loss_cfg.get("task_align_alpha", 1.0))
        self.task_align_beta = float(loss_cfg.get("task_align_beta", 6.0))
        self.ignore_iou_thr = loss_cfg.get("ignore_iou_thr", None)
        self.qfl = QualityFocalLoss(float(loss_cfg.get("qfl_beta", 2.0)))
        self.lane_width = float(loss_cfg.get("lane_width", 7.5 / 800))
        self.lane_width_cost = float(loss_cfg.get("lane_width_cost", 30.0 / 800))
        self.iou_loss = LaneIoULoss(loss_cfg.get("iou_loss_weight", 4.0), self.lane_width,
                                    self.img_w, self.img_h)
        # aligned narrow LaneIoU for the quality gate (1 - loss with weight 1)
        self.gate_iou_fn = LaneIoULoss(1.0, self.lane_width, self.img_w, self.img_h)
        self.seg_loss_weight = float(loss_cfg.get("seg_loss_weight", 1.0))
        seg_weights = torch.ones(seg_num_classes)
        seg_weights[0] = float(loss_cfg.get("seg_bg_weight", 0.4))
        self.register_buffer("seg_class_weights", seg_weights, persistent=False)
        self.seg_ignore_label = int(loss_cfg.get("seg_ignore_label", 255))
        self.brr_support_weight = float(loss_cfg.get("brr_support_loss_weight", 0.2))
        self.brr_cp_weight = float(loss_cfg.get("brr_cp_loss_weight", 0.05))
        self.brr_support_beta = float(loss_cfg.get("brr_support_smooth_l1_beta", 1.0))
        self.brr_cp_beta = float(loss_cfg.get("brr_cp_smooth_l1_beta", 1.0))
        self.brr_component_weights = (float(loss_cfg.get("start_y_component_weight", 1.0)),
                                      float(loss_cfg.get("length_component_weight", 1.0)))
        self.brr_cp_fit_ridge = float(loss_cfg.get("brr_cp_fit_ridge", 1e-2))
        self.brr_loss_stages = [int(s) for s in loss_cfg.get("brr_loss_stages", [0, 1, 2])]

        adapter_cfg = dict(target_adapter or {})
        self.target_adapter = CLRTargetAdapter(
            n_offsets=self.n_offsets, img_w=self.img_w, img_h=self.img_h, **adapter_cfg)

        self._init_clrernet_weights()

    # ======================================================================
    # Initialization (official CLRerNet policy, as in V11)
    # ======================================================================
    def _init_clrernet_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        for p in list(self.cls_layers.parameters()) + list(self.reg_layers.parameters()):
            nn.init.normal_(p, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.roi_gather.attention.W.weight, 0.0)
        nn.init.constant_(self.roi_gather.attention.W.bias, 0.0)
        if self.gsrc is not None:
            # restore the identity initialization after the global re-init above
            self.gsrc.zero_init()
        if self.query_attn is not None:
            for module in self.query_attn.values():
                module.zero_init()
        if self.lateral is not None:
            for module in self.lateral.values():
                module.zero_init()
        if self.aux_cls_layers is not None:
            for p in self.aux_cls_layers.parameters():
                nn.init.normal_(p, mean=0.0, std=1.0e-3)

    def init_weights(self):
        # Do not call the official head's init_weights (it touches the anchor
        # generator we do not build). Backbone/neck init is unaffected.
        self._init_clrernet_weights()

    # ======================================================================
    # Features
    # ======================================================================
    def _select_features(self, x):
        if isinstance(x, dict):
            x = list(x.values())
        feats = list(x)[len(x) - self.refine_layers:]
        feats.reverse()  # coarsest first, as in CLRNet/CLRerNet
        areas = [f.shape[-2] * f.shape[-1] for f in feats]
        if any(a > b for a, b in zip(areas[:-1], areas[1:])):
            raise RuntimeError(
                f"CLRBezierHead expects neck outputs ordered fine -> coarse (reversed to "
                f"coarse -> fine inside the head). Got spatial sizes "
                f"{[tuple(f.shape[-2:]) for f in feats]} after reversal.")
        return feats

    # ======================================================================
    # BRR refinement
    # ======================================================================
    def _refine(self, feats, local_cp, context_tokens=None, num_groups=1, branch="main"):
        """Returns (preds, states, extras); extras holds input_xs and re-projection stats."""
        batch_size, num_q = local_cp.shape[:2]
        one_row = 1.0 / float(self.n_strips)
        geo = dict(prior_ys=self.prior_ys, sample_x_indices=self.sample_x_indices,
                   img_w=self.img_w, img_h=self.img_h, n_strips=self.n_strips)

        cp_x, y_start = globalize_local_cp(local_cp, self.cp_x_margin, self.eps)
        _, on_map = brr_reference(cp_x, y_start, y_start + one_row, **geo)

        pooled_stages, preds, states = [], [], []
        input_xs, reproj_stats = [], []
        step = self._step_cache
        for stage in range(self.refine_layers):
            if self._need_input_xs:
                # full-row x of the reference this stage samples from (gate_on="input")
                input_xs.append(eval_global_cubic(cp_x.detach(), self.prior_ys.to(cp_x.dtype)))
            prior_xs = torch.flip(on_map, dims=[2])
            pooled = pool_prior_features(feats[stage], prior_xs, self.prior_feat_ys,
                                         self.prior_feat_channels)
            pooled_stages.append(pooled)
            roi = self.roi_gather(pooled_stages, feats[stage], stage)
            if self.gsrc is not None:
                roi = self.gsrc.inject(stage, roi, context_tokens)
            if self.query_attn is not None and stage in self.query_attn_stages:
                # geometry of the anchors these RoI features were sampled from
                anchor_state = torch.cat([y_start.unsqueeze(-1), cp_x], dim=-1)
                roi = self.query_attn[str(stage)](roi, anchor_state, num_groups)
            cls_roi = reg_roi = roi
            if self.lateral is not None and stage in self.lateral_stages:
                evidence = self.lateral[str(stage)](feats[stage], prior_xs, self.prior_feat_ys)
                module = self.lateral[str(stage)]
                if module.apply_to in ("cls", "both"):
                    cls_roi = module.fuse(cls_roi, evidence)
                if module.apply_to in ("reg", "both"):
                    reg_roi = module.fuse(reg_roi, evidence)
            cls_f = cls_roi.reshape(batch_size * num_q, self.fc_hidden_dim)
            reg_f = reg_roi.reshape(batch_size * num_q, self.fc_hidden_dim)
            for m in self.cls_modules:
                cls_f = m(cls_f)
            for m in self.reg_modules:
                reg_f = m(reg_f)
            cls_head = (self.aux_cls_layers if (branch == "aux" and self.aux_cls_layers is not None)
                        else self.cls_layers)
            cls_logits = cls_head(cls_f).view(batch_size, num_q, 2).float()
            reg = self.reg_layers(reg_f).view(batch_size, num_q, -1).float()

            new_x, new_y, _, _ = update_brr_state(cp_x, y_start, reg[..., :6],
                                                  self.n_strips, self.cp_x_margin)
            length = reg[..., 1]
            ref, ref_on_map = brr_reference(new_x, new_y, length, **geo)
            pred = torch.cat([cls_logits, ref[..., 2:6], ref[..., 6:] + reg[..., 6:]], dim=-1)

            # Direct-supervision state: detached input + raw (unclamped) delta.
            raw_y = y_start.detach() + reg[..., 0]
            raw_x = cp_x.detach() + reg[..., 2:6]
            states.append(torch.cat([raw_y.unsqueeze(-1), length.unsqueeze(-1), raw_x], dim=-1))
            preds.append(pred)

            if stage != self.refine_layers - 1:
                next_x, next_map = new_x, ref_on_map
                if self.reproject is not None and stage in self.reproject_stages:
                    next_x, stats = self.reproject.project(pred.detach(), new_x.detach(),
                                                           step, self.training)
                    next_map = sampling_xs(next_x, self.prior_ys, self.sample_x_indices)
                    reproj_stats.append(stats)
                cp_x, y_start, on_map = next_x.detach(), new_y.detach(), next_map.detach()
        return preds, states, dict(input_xs=input_xs or None, reproj=reproj_stats)

    def gsrc_tokens(self, feats):
        """Global context tokens from the coarsest FPN level ([B, H*W, C])."""
        return None if self.gsrc is None else self.gsrc.tokens(feats[0])

    def forward_train(self, feats):
        batch_size = feats[-1].shape[0]
        clean_cp = self.prior_bank(batch_size)
        tokens = self.gsrc_tokens(feats)
        main_preds, main_states, main_extra = self._refine(feats, clean_cp, tokens)
        out = dict(main_preds=main_preds, main_states=main_states,
                   main_input_xs=main_extra["input_xs"], reproj=main_extra["reproj"])

        if self.aux_enabled and self.aux_num_groups > 0:
            m = self.aux_num_groups
            k = self.num_priors
            cp_rep = clean_cp.unsqueeze(1).expand(batch_size, m, k, 4, 2).reshape(batch_size * m, k, 4, 2)
            max_t = min(self.aux_noise_t, self.perturbation.timesteps - 1)
            if self.aux_random_t:
                t = torch.randint(0, max_t + 1, (batch_size * m,), device=clean_cp.device)
            else:
                t = torch.full((batch_size * m,), max_t, device=clean_cp.device, dtype=torch.long)
            coeffs = self.perturbation.sample_coeffs((batch_size * m, k), t, clean_cp.device, clean_cp.dtype)
            aux_cp = self.perturbation.perturb(cp_rep, coeffs)
            aux_feats = [f.repeat_interleave(m, dim=0) for f in feats]
            # tokens are global: computed once, shared by every perturbed group
            aux_tokens = None if tokens is None else tokens.repeat_interleave(m, dim=0)
            aux_preds, aux_states, aux_extra = self._refine(aux_feats, aux_cp, aux_tokens,
                                                            num_groups=m, branch="aux")
            if aux_extra["input_xs"] is not None:
                out["aux_input_xs"] = [x.view(batch_size, m * k, -1) for x in aux_extra["input_xs"]]
            out["aux_preds"] = [p.view(batch_size, m * k, -1) for p in aux_preds]
            out["aux_states"] = [s.view(batch_size, m * k, -1) for s in aux_states]

        size = feats[-1].shape[-2:]
        seg_in = torch.cat([F.interpolate(f, size=size, mode="bilinear", align_corners=False)
                            for f in feats], dim=1)
        out["seg"] = self.seg_decoder(seg_in)
        return out

    def forward_test(self, feats):
        preds, _, _ = self._refine(feats, self.prior_bank(feats[-1].shape[0]),
                                self.gsrc_tokens(feats))
        return StagePredictions([self.to_official_pred_dict(p) for p in preds])

    @staticmethod
    def to_official_pred_dict(pred):
        return {
            "cls_logits": pred[..., :2],
            "anchor_params": pred[..., 2:5],
            "lengths": pred[..., 5:6],
            "xs": pred[..., 6:],
        }

    def forward(self, x, *args, **kwargs):
        feats = self._select_features(x)
        if self.training:
            return self.forward_train(feats)
        return self.forward_test(feats)

    # ======================================================================
    # Losses
    # ======================================================================
    def loss(self, x, batch_data_samples, *args, **kwargs):
        if self.training:
            self._train_step += 1
            self._step_cache = int(self._train_step)  # one host sync per iteration
        feats = self._select_features(x)
        outs = self.forward_train(feats)
        device = feats[0].device
        lanes = self.target_adapter.extract_lanes(batch_data_samples, device)
        seg_gt = None
        if self.seg_loss_weight > 0:
            seg_gt = self.target_adapter.extract_seg(batch_data_samples, device, outs["seg"].shape[-2:])
        return self.loss_by_outputs(outs, lanes, seg_gt)

    def _aux_pairs(self, stage):
        return [(self.aux_assigners[i], self.aux_assigner_weights[i])
                for i in self.aux_stage_assigner_ids.get(stage, [])]

    def _main_pairs(self, stage):
        return [(self.main_stage_assigners.get(stage, self.main_assigner), 1.0)]

    def loss_by_outputs(self, outs, lanes, seg_gt=None):
        valid_targets = [t[t[:, 1] == 1] for t in lanes]
        gt_cps = [fit_global_cubic_to_clr_rows(t[:, 6:], self.prior_ys, self.img_w,
                                               self.brr_cp_fit_ridge, self.cp_x_margin)
                  for t in valid_targets]

        main = self._branch_losses(outs["main_preds"], outs["main_states"], valid_targets, gt_cps,
                                   self._main_pairs, list(range(self.refine_layers)), apply_brr=True,
                                   gate=self.main_gate, input_xs=outs.get("main_input_xs"))
        cls, iou, sup, cp = main["cls"], main["iou"], main["support"], main["cp"]
        diag = {"main_iou": main["iou"].detach(), "main_num_pos": main["num_pos"],
                "main_conf_iou_l1": main["conf_iou_l1"],
                "main_conf_iou_rank": main["conf_iou_rank"]}
        for st, count in enumerate(main["pos_stage"]):
            diag[f"main_pos_s{st}"] = count
        if outs.get("reproj"):
            dev = main["num_pos"].device
            diag["reproj_ok_frac"] = torch.tensor(
                sum(r["ok_frac"] for r in outs["reproj"]) / len(outs["reproj"]), device=dev)
            diag["reproj_shift_px"] = torch.tensor(
                sum(r["shift_px"] for r in outs["reproj"]) / len(outs["reproj"]), device=dev)
            diag["reproj_blend"] = torch.tensor(outs["reproj"][0]["blend"], device=dev)

        if "aux_preds" in outs:
            aux = self._branch_losses(outs["aux_preds"], outs["aux_states"], valid_targets, gt_cps,
                                      self._aux_pairs, self.aux_stages, apply_brr=self.aux_apply_brr,
                                      cls_target_mode=self.aux_cls_target_mode,
                                      gate=self.aux_gate, input_xs=outs.get("aux_input_xs"))
            cls = cls + self.aux_cls_loss_weight * aux["cls"]
            iou = iou + self.aux_reg_loss_weight * aux["iou"]
            sup = sup + self.aux_reg_loss_weight * aux["support"]
            cp = cp + self.aux_reg_loss_weight * aux["cp"]
            diag.update(aux_iou=aux["iou"].detach(), aux_num_pos=aux["num_pos"],
                        aux_conf_iou_l1=aux["conf_iou_l1"],
                        aux_conf_iou_rank=aux["conf_iou_rank"])

        losses = dict(
            loss_cls=cls * self.cls_loss_weight,
            loss_iou=iou,  # LaneIoULoss already applies iou_loss_weight
            loss_brr_support=sup * self.brr_support_weight,
            loss_brr_cp=cp * self.brr_cp_weight,
        )
        if seg_gt is not None:
            seg_loss = F.nll_loss(F.log_softmax(outs["seg"], dim=1), seg_gt,
                                  weight=self.seg_class_weights, ignore_index=self.seg_ignore_label)
            losses["loss_seg"] = seg_loss * self.seg_loss_weight
        # Keys without "loss" are logged but not summed by mmengine.
        losses.update(diag)
        return losses

    def _cls_loss(self, logits, cls_targets, gt_counts, quality=None, ignore=None):
        """logits [B,N,2]; cls_targets [A,B,N] -> per-assigner-image loss [A,B]."""
        num_a, batch_size, num_q = cls_targets.shape
        flat = logits.unsqueeze(0).expand(num_a, -1, -1, -1).reshape(-1, 2)
        tgt = cls_targets.reshape(-1)
        if quality is not None:
            raw = self.qfl(flat, quality.reshape(-1)).view(num_a, batch_size, num_q)
            w = torch.where(cls_targets == 0, raw.new_tensor(self.cls_bg_weight),
                            raw.new_tensor(1.0))
            if ignore is not None:
                w = w * (~ignore).to(raw.dtype)
            return (raw * w).sum(-1) / w.sum(-1).clamp_min(1.0)
        if self.use_focal:
            logp = F.log_softmax(flat, dim=-1)
            logp_t = logp.gather(1, tgt.view(-1, 1)).squeeze(1)
            raw = -self.focal_alpha * (1.0 - logp_t.exp()).pow(self.focal_gamma) * logp_t
            raw = raw.view(num_a, batch_size, num_q)
            return raw.sum(-1) / gt_counts.view(1, -1).clamp_min(1.0)
        raw = F.cross_entropy(flat, tgt, reduction="none").view(num_a, batch_size, num_q)
        w = torch.where(cls_targets == 0, raw.new_tensor(self.cls_bg_weight), raw.new_tensor(1.0))
        if ignore is not None:
            w = w * (~ignore).to(raw.dtype)
        return (raw * w).sum(-1) / w.sum(-1).clamp_min(1.0)

    def _brr_losses(self, state, target, gt_cp, gt_cp_ok):
        n = float(self.n_strips)
        pred_start = (1.0 - state[:, 0]) * n
        pred_len = state[:, 1] * n
        tgt_start = (1.0 - target[:, 2]) * n
        tgt_len = target[:, 5]
        with torch.no_grad():
            p_round = pred_start.detach().round().clamp(0, n)
            t_round = tgt_start.round().clamp(0, n)
            adj_len = tgt_len - (p_round - t_round)
        sup = _smooth_l1(torch.stack([pred_start, pred_len], -1),
                         torch.stack([tgt_start, adj_len], -1), self.brr_support_beta)
        sup = (sup * sup.new_tensor(self.brr_component_weights)).mean(-1).mean()

        scale = float(max(1, self.img_w - 1))
        cp_elem = _smooth_l1(state[:, 2:6] * scale, gt_cp * scale, self.brr_cp_beta).mean(-1)
        ok = gt_cp_ok.float()
        cp = (cp_elem * ok).sum() / ok.sum().clamp_min(1.0)
        return sup, cp

    def _branch_losses(self, preds_stages, states_stages, valid_targets, gt_cps, pairs_fn,
                       stages, apply_brr, cls_target_mode=None, gate=None, input_xs=None):
        device = preds_stages[0].device
        batch_size = preds_stages[0].shape[0]
        zero = preds_stages[0].new_zeros(())
        active = [s for s in stages if len(pairs_fn(s)) > 0]
        normalizer = float(max(1, batch_size * len(active)))
        gt_counts = torch.tensor([float(t.shape[0]) for t in valid_targets], device=device)
        scale = float(self.img_w - 1) / float(self.img_w)

        cls_sum, iou_sum, sup_sum, cp_sum = zero, zero, zero, zero
        num_pos = 0
        align_err, align_cnt = 0.0, 0  # mean |confidence - LaneIoU| over matched pairs
        rank_sum, rank_cnt = 0.0, 0     # Spearman(confidence, LaneIoU): what F1 actually needs
        pos_stage = [0] * self.refine_layers  # classification positives per stage (after gating)
        step = self._step_cache
        for stage in active:
            pairs = pairs_fn(stage)
            preds = preds_stages[stage]
            states = states_stages[stage]
            required = set()
            for assigner, _ in pairs:
                required |= set(assigner.required_cache_keys)
            mode = cls_target_mode or self.cls_target_mode
            use_quality = mode != "hard"
            use_ignore = self.ignore_iou_thr is not None
            gate_thr = gate.threshold(stage, step, self.training) if gate is not None else 0.0
            if use_quality or use_ignore or gate_thr > 0.0:
                # narrow-width LaneIoU: the quantity the CULane metric measures
                required.add("lane_iou_dynamic")
            cls_targets = torch.zeros((len(pairs), batch_size, preds.shape[1]),
                                      dtype=torch.long, device=device)
            quality = torch.zeros(cls_targets.shape, dtype=preds.dtype, device=device) \
                if use_quality else None
            ignore = torch.zeros(cls_targets.shape, dtype=torch.bool, device=device) \
                if use_ignore else None
            use_brr = apply_brr and stage in self.brr_loss_stages
            for b in range(batch_size):
                target = valid_targets[b]
                if target.shape[0] == 0:
                    continue
                cache = build_cost_cache(preds[b].detach(), target, self.img_w, self.img_h,
                                         self.lane_width, self.lane_width_cost, required)
                for a, (assigner, weight) in enumerate(pairs):
                    rows, cols = assigner.assign(cache)
                    reg_rows, reg_cols = rows, cols
                    if gate_thr > 0.0 and rows.numel() > 0:
                        if gate.gate_on == "output":
                            gate_iou = cache["lane_iou_dynamic"][rows, cols]
                        else:
                            with torch.no_grad():
                                gate_iou = 1.0 - self.gate_iou_fn(
                                    input_xs[stage][b, rows] * scale,
                                    target[cols, 6:] / float(self.img_w))
                        keep = gate.keep_mask(gate_iou, cols, int(target.shape[0]), gate_thr)
                        rows, cols = rows[keep], cols[keep]
                        if gate.mode == "cls_and_reg":
                            reg_rows, reg_cols = rows, cols
                    if use_ignore:
                        ignore[a, b] = ignore_unmatched(
                            cache["lane_iou_dynamic"], rows, self.ignore_iou_thr)
                    pos_stage[stage] += int(rows.numel())
                    if rows.numel() > 0:
                        cls_targets[a, b, rows] = 1
                        num_pos += int(rows.numel())
                        if "lane_iou_dynamic" in cache:
                            with torch.no_grad():
                                conf = F.softmax(preds[b, rows, :2].detach(), dim=-1)[:, 1]
                                pair_q = cache["lane_iou_dynamic"][rows, cols]
                                align_err += float((conf - pair_q).abs().sum())
                                align_cnt += int(rows.numel())
                                if rows.numel() > 2:
                                    rc = conf.argsort().argsort().float()
                                    rq = pair_q.argsort().argsort().float()
                                    rc = rc - rc.mean()
                                    rq = rq - rq.mean()
                                    denom = rc.norm() * rq.norm()
                                    if float(denom) > 0:
                                        rank_sum += float((rc * rq).sum() / denom)
                                        rank_cnt += 1
                        if use_quality:
                            pair_iou = cache["lane_iou_dynamic"][rows, cols]
                            pair_scores = None
                            if mode == "task_aligned":
                                pair_scores = F.softmax(
                                    preds[b, rows, :2].detach(), dim=-1)[:, 1]
                            quality[a, b, rows] = quality_targets(
                                pair_iou, pair_scores, mode,
                                self.task_align_alpha, self.task_align_beta,
                                cols, int(target.shape[0]))
                    if reg_rows.numel() == 0:
                        continue
                    per_lane = self.iou_loss(preds[b, reg_rows, 6:] * scale,
                                             target[reg_cols, 6:] / float(self.img_w))
                    iou_sum = iou_sum + weight * per_lane.mean()
                    if use_brr:
                        sup, cp = self._brr_losses(states[b, reg_rows], target[reg_cols],
                                                   gt_cps[b][0][reg_cols], gt_cps[b][1][reg_cols])
                        sup_sum = sup_sum + weight * sup
                        cp_sum = cp_sum + weight * cp
            weights = preds.new_tensor([w for _, w in pairs]).view(-1, 1)
            cls_sum = cls_sum + (self._cls_loss(preds[..., :2], cls_targets, gt_counts,
                                                quality, ignore) * weights).sum()

        return dict(cls=cls_sum / normalizer, iou=iou_sum / normalizer,
                    support=sup_sum / normalizer, cp=cp_sum / normalizer,
                    num_pos=torch.tensor(float(num_pos), device=device),
                    conf_iou_l1=torch.tensor(align_err / max(1, align_cnt), device=device),
                    conf_iou_rank=torch.tensor(rank_sum / max(1, rank_cnt), device=device),
                    pos_stage=[torch.tensor(float(c), device=device) for c in pos_stage])
