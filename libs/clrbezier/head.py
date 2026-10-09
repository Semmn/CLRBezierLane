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
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from .assigners import build_cost_cache, build_lane_assigner
from .batched_loss import (GTLayout, batched_assign, batched_cost_cache, batched_keep_mask,
                           per_image_spearman, per_image_sum)
from .data_adapters import CLRTargetAdapter
from .alignment import QualityFocalLoss, ignore_unmatched, quality_targets
from .cascade import QualityGate, ReferenceReprojector, sampling_xs
from .geometry import eval_cubic, eval_global_cubic
from .gsrc import GSRCModule
from .lateral import LateralEvidence
from .query_attention import MaskedQuerySelfAttention
from .geometry import (brr_reference, fit_global_cubic_to_clr_rows, globalize_local_cp,
                       reparam_cp_to_frame, support_state_from_local, support_top,
                       transport_cp, update_brr_state, update_framed_state)
from .lane_iou import LaneIoULoss, pairwise_lane_iou
from .preconditioner import ControlPointPreconditioner
from .ranking import batch_cluster_rank_loss
from .curve_deformable_roi_gather import (CurveAlignedDeformableROIGather,
                                            build_clr_curve_reference_points)
from .gliou import GeneralizedLaneIoULoss, pairwise_generalized_lane_iou
from .modules import ROIGather, SegDecoder, linear_relu, pool_prior_features
from .moe import MoEGate
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
        roi_gather_cfg=None,
        look_forward_twice=False,
        cp_frame="global",
        cp_precond_cfg=None,
        rank_loss_cfg=None,
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
        # Where the four control points live.
        #   "global"   : image y = 0, 1/3, 2/3, 1 — the curve's domain is the
        #                whole image however short the lane is.
        #   "anchored" : image y = 0, y_start/3, 2*y_start/3, y_start — the last
        #                control point IS the lane's start point and the first is
        #                pinned to the image top, matching CLRNet's anchor span.
        #   "support"  : image y = the thirds of [y_top, y_start], y_top from the
        #                predicted length — start, length and shape are separate.
        self.cp_frame = str(cp_frame)
        if self.cp_frame not in ("global", "anchored", "support"):
            raise ValueError("cp_frame must be 'global', 'anchored' or 'support'")
        if self.cp_frame != "global" and reproject_cfg:
            # ReferenceReprojector refits control points on the fixed global
            # basis; it would silently write global-frame points into an
            # anchored/support state.
            raise ValueError(f"reproject_cfg is not supported with cp_frame='{self.cp_frame}'")
        # Anchored priors were built without the frame Jacobian before
        # 2026-10-06 (globalize_local_cp). True reproduces that for evaluating
        # checkpoints trained with the old code.
        self.legacy_anchored_prior = bool(brr_cfg.get("legacy_anchored_prior", False))
        # Support frame (and opt-in for anchored), see update_framed_state:
        #   cp_transport   : keep the curve when start/length move, so dP is a
        #                    pure shape correction (default on for "support").
        #   length_mode    : "residual" (support default) or "fresh" (CLRerNet).
        #   min_span       : smallest frame span in image y (support only).
        #   frame_grad     : let the curve's losses (LaneIoU) reach start/length
        #                    through the frame. Off by default for "support":
        #                    start/length then learn only from the support
        #                    loss, as in the global frame.
        is_support = self.cp_frame == "support"
        self.cp_transport = bool(brr_cfg.get("cp_transport", is_support))
        self.length_mode = str(brr_cfg.get("length_mode", "residual" if is_support else "fresh"))
        if self.length_mode not in ("residual", "fresh"):
            raise ValueError("brr_cfg.length_mode must be 'residual' or 'fresh'")
        self.support_min_span = float(brr_cfg.get("min_span", 0.1))
        self.frame_grad = bool(brr_cfg.get("frame_grad", not is_support))
        #   continuation_grad : "full" (default) or "endpoint". Rows outside the
        #                    frame are drawn by a linear continuation; with
        #                    "endpoint" its slope is detached, so far-away rows
        #                    (a short frame against a long GT) no longer push
        #                    the control points with weights that grow with the
        #                    distance. Values are unchanged; anchored/support only.
        self.continuation_grad = str(brr_cfg.get("continuation_grad", "full"))
        if self.continuation_grad not in ("full", "endpoint"):
            raise ValueError("brr_cfg.continuation_grad must be 'full' or 'endpoint'")
        self.slope_grad = self.continuation_grad == "full"
        # The framed update path (update_framed_state) is used for "support",
        # and for "anchored" only when one of its options is switched on, so
        # existing anchored configs keep their exact behaviour.
        self.framed_update = is_support or (
            self.cp_frame == "anchored"
            and (self.cp_transport or self.length_mode != "fresh" or not self.frame_grad))
        if self.cp_frame == "global" and any(
                k in brr_cfg for k in ("cp_transport", "length_mode", "frame_grad", "min_span",
                                       "continuation_grad")):
            raise ValueError("brr_cfg cp_transport / length_mode / frame_grad / min_span / "
                             "continuation_grad "
                             "apply to the anchored and support frames only")
        if self.cp_frame == "anchored" and "min_span" in brr_cfg:
            raise ValueError("brr_cfg.min_span applies to cp_frame='support' only")
        #   stage_jitter   : None (default, off) or a dict. Training only: the
        #                    detached state a stage hands to the next one is
        #                    randomly shifted, tilted about its start point,
        #                    bent, and its start_y moved, before the next stage
        #                    samples features along it. Each later stage then
        #                    learns to refine from a spread of inputs instead
        #                    of exactly its predecessor's output. Length is
        #                    predicted fresh at every stage and is not touched.
        #                    Keys (defaults):
        #                      branches    ["aux"]  "main" and/or "aux"
        #                      after_stages [0, 1]  handoffs to jitter (the
        #                                           stage that produced it)
        #                      translate 0.01, slope 0.02, curve 0.01,
        #                      y_shift 0.02: half-widths of uniform draws, in
        #                      normalized image units (x: / (img_w - 1),
        #                      y: / img_h). slope is the x change per unit of
        #                      image height; curve is the bow at the middle of
        #                      the control-point frame.
        #                      prob 1.0: fraction of queries jittered.
        #                    stage_jitter=True or dict() switches it on with
        #                    these defaults.
        self.stage_jitter = self._parse_stage_jitter(brr_cfg.get("stage_jitter"))

        # ---- network ----------------------------------------------------------
        rg_cfg = dict(roi_gather_cfg or {})
        rg_type = rg_cfg.pop("type", "ROIGather")
        rg_common = dict(in_channels=self.prior_feat_channels, num_priors=self.num_priors,
                         sample_points=self.sample_points, fc_hidden_dim=self.fc_hidden_dim,
                         refine_layers=self.refine_layers)
        self.roi_gather_deformable = rg_type == "CurveAlignedDeformableROIGather"
        if rg_type == "ROIGather":
            self.roi_gather = ROIGather(mid_channels=roi_mid_channels, **rg_common)
        elif self.roi_gather_deformable:
            rg_cfg.setdefault("mid_channels", roi_mid_channels)
            self.deform_curve_samples = int(rg_cfg.get("deform_num_curve_samples", 18))
            self.zero_init_deformable = bool(rg_cfg.pop("zero_init_outputs", True))
            # With look_forward_twice the stage reference keeps its graph, so the
            # deformable sampling grid (18 x 4 grid_sample points per query, plus
            # the tangent/normal frame built from the same points) would send a
            # feature-gradient back into the previous stage's control points on
            # top of the 36 pooling points. V11 never had that path: it ran
            # deformable sampling without LFT. Detaching keeps the branch's
            # behaviour identical and removes the extra path.
            self.deform_detach_reference = bool(rg_cfg.pop("detach_reference", True))
            self.roi_gather = CurveAlignedDeformableROIGather(**rg_common, **rg_cfg)
        else:
            raise ValueError(f"Unknown roi_gather type {rg_type!r}")
        # look_forward_twice: False, True / "dino", or "chain".
        # "dino" (what True means): each stage's prediction is built on the
        # previous stage's *live* update, which itself starts from a detached
        # input, so stage i's delta gets its own loss plus stage i+1's loss and
        # nothing further (DINO, dino_layers.py). Pooling, q2q, lateral, the
        # preconditioner and the state supervision use detached values.
        # "chain" is the earlier behaviour: the whole state, including the
        # RoI pooling grid, keeps its graph across all stages. The forward pass
        # is identical in all three modes; only the gradients differ.
        if look_forward_twice is True:
            look_forward_twice = "dino"
        if look_forward_twice not in (False, None, "dino", "chain"):
            raise ValueError("look_forward_twice must be False, True, 'dino' or 'chain'")
        self.lft_mode = look_forward_twice or None
        self.look_forward_twice = self.lft_mode is not None
        if self.stage_jitter is not None and self.lft_mode is not None:
            # LFT routes the next stage's loss through the un-jittered update;
            # combining the two would make that gradient refer to a state the
            # next stage never sampled.
            raise ValueError("brr_cfg.stage_jitter cannot be combined with look_forward_twice")
        # Control-point preconditioning (off unless configured).
        self.cp_precond = (ControlPointPreconditioner(n_strips=self.n_strips,
                                                      **dict(cp_precond_cfg))
                           if cp_precond_cfg else None)
        # Pairwise ranking inside NMS duplicate clusters (off unless configured).
        rank_cfg = dict(rank_loss_cfg or {})
        self.rank_loss_enabled = bool(rank_loss_cfg) and rank_cfg.pop('enabled', True)
        self.rank_loss_weight = float(rank_cfg.pop('loss_weight', 1.0))
        self.rank_loss_stages = set(rank_cfg.pop('stages', [self.refine_layers - 1]))
        # "logit" (default) compares the binary logit; "prob" compares the
        # softmax probability, which is what the first version did and should
        # not be used. In probability space the difference is bounded by 1, so
        # with tau=0.5 the loss cannot fall below softplus(-2) = 0.127 and sits
        # near 0.32 whatever the model does; worse, d(prob)/d(logit) = p(1-p)
        # vanishes exactly for the confident duplicates that make up the hard
        # pairs, so the gradient dies where the ordering is wrong.
        self.rank_score_space = rank_cfg.pop('score_space', 'logit')
        if self.rank_score_space not in ('logit', 'prob'):
            raise ValueError("rank_loss_cfg.score_space must be 'logit' or 'prob'")
        # Default the cluster distance to the NMS threshold this model will
        # actually run with, so the loss supervises the comparisons NMS makes.
        if self.rank_loss_enabled and rank_cfg.get("cluster_mode", "distance") == "distance":
            rank_cfg.setdefault("nms_thres", float(
                (test_cfg or {}).get("nms_thres", 50.0)))
        # Opt-in: score each prediction only on its own predicted rows
        # [start, start + length), as the metric does. Default off (all GT rows,
        # as the QFL / gate qualities do), so existing rank runs are unchanged.
        self.rank_quality_extent = bool(rank_cfg.pop("quality_use_extent", False))
        if rank_cfg.get("positives_only", False):
            # the head has no assignment at this point to build positive_mask from
            raise ValueError("rank_loss_cfg.positives_only is not supported by CLRBezierHead")
        self.rank_loss_kwargs = rank_cfg
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
        # BatchNorm running statistics. The aux pass is a separate forward call,
        # so it normalizes with its own batch statistics either way; the switch
        # only decides whether it also updates the running mean/var that
        # evaluation uses. "shared" (old behaviour): main and aux both update
        # them, so eval normalizes with a ~47/53 main/aux blend. "main": the aux
        # pass leaves them untouched, so eval uses main-branch statistics only.
        self.aux_bn_stats = str(aux_cfg.get("bn_stats", "shared"))
        if self.aux_bn_stats not in ("shared", "main"):
            raise ValueError(f"Unknown aux_cfg bn_stats {self.aux_bn_stats!r}")
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
            # Frame of the control points the pairwise geometry compares.
            # "global" (default): every query's curve on the image frame [0, 1],
            # so two queries drawing the same curve get the same control points.
            # "native": the earlier input, the per-query anchored frame
            # [0, y_start] (support CPs were moved onto it), where the same
            # curve with starts 1.0 vs 0.9 differs by ~47 px in control points.
            self.q2q_geometry_frame = query_attn_cfg.pop("geometry_frame", "global")
            if self.q2q_geometry_frame not in ("global", "native"):
                raise ValueError("query_attn_cfg.geometry_frame must be 'global' or 'native'")
            # Where q2q runs inside a refinement stage:
            #   "pre_q2g"     : on the pooled lane features (after ROIGather's
            #                   fc + LayerNorm), before they attend to the
            #                   feature map, so q2g is queried with mixed queries;
            #   "post_q2g"    : right after the feature-map attention's residual,
            #                   before the deformable branch and GSRC;
            #   "post_gather" : after the whole gather, deformable branch and
            #                   GSRC (the original position; the default, so
            #                   existing checkpoints are unchanged).
            # Without a deformable branch or GSRC, "post_q2g" == "post_gather".
            self.q2q_position = query_attn_cfg.pop("position", "post_gather")
            if self.q2q_position not in ("pre_q2g", "post_q2g", "post_gather"):
                raise ValueError("query_attn_cfg.position must be 'pre_q2g', 'post_q2g' "
                                 "or 'post_gather'")
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
        # Image shape (h, w) the LaneIoU tilt-dependent width is computed in.
        # Official CLRerNet uses the metric crop, eval_shape=(320, 1640) on
        # CULane; None keeps the network shape (img_h, img_w), which is what
        # every existing config was trained with. Coordinates stay normalized
        # either way; only the slanted-lane width changes.
        eval_shape = loss_cfg.get("iou_eval_shape", None)
        self.iou_h, self.iou_w = ((int(eval_shape[0]), int(eval_shape[1])) if eval_shape
                                  else (self.img_h, self.img_w))
        self.iou_loss_type = loss_cfg.get("iou_loss_type", "laneiou")  # laneiou | gliou
        if self.iou_loss_type == "gliou":
            self.iou_loss = GeneralizedLaneIoULoss(
                loss_weight=float(loss_cfg.get("iou_loss_weight", 4.0)),
                lane_width=self.lane_width, img_h=self.iou_h, img_w=self.iou_w)
        elif self.iou_loss_type == "laneiou":
            self.iou_loss = LaneIoULoss(loss_cfg.get("iou_loss_weight", 4.0), self.lane_width,
                                        self.iou_w, self.iou_h)
        else:
            raise ValueError(f"Unknown iou_loss_type {self.iou_loss_type!r}")
        # assignment costs may use GLIoU independently of the loss
        self.cost_iou_type = loss_cfg.get("cost_iou_type", "laneiou")
        if self.cost_iou_type == "gliou":
            self.iou_fns = dict(
                dynamic=lambda p, t: pairwise_generalized_lane_iou(
                    p, t, self.lane_width, self.iou_h, self.iou_w),
                cost=lambda p, t, s, e: pairwise_generalized_lane_iou(
                    p, t, self.lane_width_cost, self.iou_h, self.iou_w,
                    use_pred_start_end=True, pred_start=s, pred_end=e))
        elif self.cost_iou_type == "laneiou":
            self.iou_fns = None
        else:
            raise ValueError(f"Unknown cost_iou_type {self.cost_iou_type!r}")
        # aligned narrow LaneIoU for the quality gate (1 - loss with weight 1)
        self.gate_iou_fn = LaneIoULoss(1.0, self.lane_width, self.iou_w, self.iou_h)
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
        # The length target is shifted by the rounded start-row error (CLRNet's
        # adjusted length). A prediction that starts far below the GT's top end
        # gets a target <= 0, which in the support frame collapses the frame to
        # min_span. A number (2.0 recommended) clamps the target at that many
        # rows; None keeps the old target.
        v = loss_cfg.get("brr_length_target_min", None)
        self.brr_length_target_min = None if v is None else float(v)
        self.brr_cp_fit_ridge = float(loss_cfg.get("brr_cp_fit_ridge", 1e-2))
        # True: GT control points that would leave [-cp_x_margin, 1 + cp_x_margin]
        # are refitted under that bound (closest representable curve) instead of
        # being clamped one by one. False keeps the old target.
        self.brr_cp_fit_box = bool(loss_cfg.get("brr_cp_fit_box", False))
        # Where the control-point loss compares prediction and GT:
        #   "pred"   (default) GT control points moved into each prediction's frame;
        #   "global" prediction moved onto the fixed global frame [0, 1], GT fitted
        #            there once (a fixed target per lane);
        #   "rows"   the predicted Bezier curve against the GT x on the GT's visible
        #            rows, in pixels (no GT fit, no frame, no margin clamp).
        self.brr_cp_loss_space = str(loss_cfg.get("brr_cp_loss_space", "pred"))
        if self.brr_cp_loss_space not in ("pred", "global", "rows"):
            raise ValueError("loss_cfg.brr_cp_loss_space must be 'pred', 'global' or 'rows', "
                             f"got {self.brr_cp_loss_space!r}")
        self.brr_loss_stages = [int(s) for s in loss_cfg.get("brr_loss_stages", [0, 1, 2])]
        # Compute the cost cache, assignment and losses for the whole batch at
        # once (batched_loss.py) instead of looping over images. Same values;
        # False keeps the per-image loop.
        self.batched_loss = bool(loss_cfg.get("batched_loss", True))

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
        if not self.roi_gather_deformable:
            nn.init.constant_(self.roi_gather.attention.W.weight, 0.0)
            nn.init.constant_(self.roi_gather.attention.W.bias, 0.0)
        elif self.zero_init_deformable:
            # The loop above re-initializes every Linear/Conv the head owns,
            # which destroys the deformable module's own zero-inits: not only
            # the two output gates, but offset_head, weight_head,
            # curve_pointwise_conv and point_pooling_score, i.e. everything
            # that makes the branch start by sampling on the current curve.
            # (V11 avoided this by running the official init immediately after
            # CLRHead.__init__, before the research modules were built.)
            self.roi_gather.zero_init()
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
        # MoE routers (deformable offsets, GSRC) start uniform; the Linear sweep
        # above had given them trunc_normal weights.
        for m in self.modules():
            if isinstance(m, MoEGate):
                m.zero_init()

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
                   img_w=self.img_w, img_h=self.img_h, n_strips=self.n_strips,
                   cp_frame=self.cp_frame, slope_grad=self.slope_grad)

        if self.cp_frame == "support":
            cp_x, y_start, length, frame_top = support_state_from_local(
                local_cp, self.cp_x_margin, self.n_strips, self.support_min_span, self.eps)
        else:
            cp_x, y_start = globalize_local_cp(local_cp, self.cp_x_margin, self.eps,
                                               cp_frame=self.cp_frame,
                                               legacy_tangent=self.legacy_anchored_prior)
            # full height from the start; also the anchored frame's own extent
            length = y_start + one_row
            frame_top = torch.zeros_like(y_start)
        _, on_map = brr_reference(cp_x, y_start, length, **geo, frame_top=frame_top)

        pooled_stages, preds, states = [], [], []
        input_xs, reproj_stats, precond_stats, moe_stats = [], [], [], []
        step = self._step_cache
        # lft_mode == "dino": the previous stage's update, live w.r.t. that
        # stage's delta only (same values as the detached cp_x/y_start/...).
        live = None
        for stage in range(self.refine_layers):
            if self._need_input_xs:
                # full-row x of the reference this stage samples from (gate_on="input")
                input_xs.append(eval_cubic(cp_x.detach(), self.prior_ys.to(cp_x.dtype),
                                           y_start.detach(), self.cp_frame,
                                           y_top=frame_top.detach()))
            prior_xs = torch.flip(on_map, dims=[2])
            pooled = pool_prior_features(feats[stage], prior_xs, self.prior_feat_ys,
                                         self.prior_feat_channels)
            pooled_stages.append(pooled)
            query_mixer = None
            if self.query_attn is not None and stage in self.query_attn_stages:
                # geometry of the anchors these RoI features were sampled from
                anchor_cp = cp_x
                if self.cp_frame != "global" and self.q2q_geometry_frame == "global":
                    # The pairwise geometry bias compares control points across
                    # queries, so give it one frame shared by all of them.
                    anchor_cp = transport_cp(cp_x, frame_top.detach(), y_start.detach(),
                                             torch.zeros_like(y_start),
                                             torch.ones_like(y_start))
                elif self.cp_frame == "support":
                    anchor_cp = transport_cp(cp_x, frame_top.detach(), y_start.detach(),
                                             torch.zeros_like(y_start), y_start.detach())
                anchor_state = torch.cat([y_start.unsqueeze(-1), anchor_cp], dim=-1)
                # The auxiliary groups are packed into the batch dimension
                # (forward_train repeat_interleaves the features), so each batch
                # row already holds exactly one group. A group mask over the K
                # queries would cut that one group into num_groups arbitrary
                # blocks, so attention always runs unmasked.
                q2q_module = self.query_attn[str(stage)]

                def query_mixer(queries, _m=q2q_module, _s=anchor_state):
                    return _m(queries, _s, 1)
            in_gather = query_mixer is not None and self.q2q_position != "post_gather"
            gather_kw = (dict(query_mixer=query_mixer, mixer_position=self.q2q_position)
                         if in_gather else {})
            if self.roi_gather_deformable:
                deform_xs = prior_xs.detach() if self.deform_detach_reference else prior_xs
                ref_pts, ref_mask = build_clr_curve_reference_points(
                    deform_xs, self.prior_feat_ys, self.deform_curve_samples)
                roi = self.roi_gather(pooled_stages, feats[stage], stage,
                                      reference_points=ref_pts, reference_valid_mask=ref_mask,
                                      **gather_kw)
            else:
                roi = self.roi_gather(pooled_stages, feats[stage], stage, **gather_kw)
            gather_stats = getattr(self.roi_gather, "last_stats", None)
            if gather_stats:
                moe_stats.append(gather_stats)
            if self.gsrc is not None:
                roi = self.gsrc.inject(stage, roi, context_tokens)
            if query_mixer is not None and not in_gather:
                roi = query_mixer(roi)
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

            reg_ref = reg
            if self.cp_precond is not None:
                # Equalize the per-control-point step over the lane's own
                # visible span. With apply_to="reference" only the curve the
                # IoU loss sees is rescaled; the BRR state supervision keeps
                # the raw delta.
                gains = self.cp_precond.gains(y_start, self._precond_length(length, reg),
                                              frame=self.cp_frame)
                reg_ref = torch.cat(
                    [reg[..., :2], reg[..., 2:6] * gains, reg[..., 6:]], dim=-1)
                if self.cp_precond.apply_to == "both":
                    reg = reg_ref
                precond_stats.append(self.cp_precond.diagnostics(
                    y_start, self._precond_length(length, reg),
                    scores=cls_logits[..., 1] - cls_logits[..., 0], frame=self.cp_frame))

            base_x, base_y, base_len, base_top = (
                (cp_x, y_start, length, frame_top) if live is None
                else (live["cp_x"], live["y_start"], live["length"], live["frame_top"]))
            dino_next = self.lft_mode == "dino" and stage != self.refine_layers - 1
            if self.framed_update:
                framed_kw = dict(n_strips=self.n_strips, margin=self.cp_x_margin,
                                 cp_frame=self.cp_frame, min_span=self.support_min_span,
                                 transport=self.cp_transport, length_mode=self.length_mode,
                                 detach_frame=not self.frame_grad)
                upd = update_framed_state(base_x, base_y, base_len, base_top,
                                          reg_ref[..., :6], **framed_kw)
                new_x, new_y, new_top = upd["cp_x"], upd["y_start"], upd["frame_top"]
                new_len = upd["length"]
                if dino_next:
                    nxt = update_framed_state(cp_x.detach(), y_start.detach(), length.detach(),
                                              frame_top.detach(), reg_ref[..., :6], **framed_kw)
                    live = dict(cp_x=nxt["cp_x"], y_start=nxt["y_start"],
                                length=nxt["length"], frame_top=nxt["frame_top"])
                if self.frame_grad:
                    ref, ref_on_map = brr_reference(new_x, new_y, new_len, **geo,
                                                    frame_top=new_top)
                else:
                    ref, ref_on_map = brr_reference(new_x, new_y, new_len, **geo,
                                                    frame_top=new_top.detach(),
                                                    frame_bottom=new_y.detach())
                # Direct-supervision state: detached input + raw delta, plus the
                # (detached) frame the control points are expressed in.
                raw_x = upd["base"].detach() + reg[..., 2:6]
                raw_y = y_start.detach() + reg[..., 0]
                raw_len = (length.detach() + reg[..., 1]) if self.length_mode == "residual" \
                    else reg_ref[..., 1]
                states.append(torch.cat([raw_y.unsqueeze(-1), raw_len.unsqueeze(-1), raw_x,
                                         new_top.detach().unsqueeze(-1),
                                         new_y.detach().unsqueeze(-1)], dim=-1))
            else:
                new_x, new_y, _, _ = update_brr_state(base_x, base_y, reg_ref[..., :6],
                                                      self.n_strips, self.cp_x_margin)
                new_len = reg_ref[..., 1]
                new_top = frame_top
                if dino_next:
                    nx, ny, _, _ = update_brr_state(cp_x.detach(), y_start.detach(),
                                                    reg_ref[..., :6], self.n_strips,
                                                    self.cp_x_margin)
                    live = dict(cp_x=nx, y_start=ny, length=new_len,
                                frame_top=frame_top.detach())
                ref, ref_on_map = brr_reference(new_x, new_y, new_len, **geo)
                # Direct-supervision state: detached input + raw (unclamped) delta.
                raw_y = y_start.detach() + reg[..., 0]
                raw_x = cp_x.detach() + reg[..., 2:6]
                states.append(torch.cat([raw_y.unsqueeze(-1), new_len.unsqueeze(-1), raw_x],
                                        dim=-1))
            pred = torch.cat([cls_logits, ref[..., 2:6], ref[..., 6:] + reg[..., 6:]], dim=-1)
            preds.append(pred)

            if stage != self.refine_layers - 1:
                next_x, next_map = new_x, ref_on_map
                if self.reproject is not None and stage in self.reproject_stages:
                    projected, stats = self.reproject.project(pred.detach(), new_x.detach(),
                                                              step, self.training)
                    # keep the gradient path through new_x when LFT is on: the
                    # re-projection correction itself is a constant
                    next_x = new_x + (projected - new_x.detach()) if self.lft_mode == "chain" \
                        else projected
                    if live is not None:
                        live["cp_x"] = live["cp_x"] + (projected - live["cp_x"].detach())
                    next_map = sampling_xs(next_x, self.prior_ys, self.sample_x_indices)
                    reproj_stats.append(stats)
                if self.lft_mode == "chain":
                    # Earlier behaviour: the whole next-stage reference keeps the
                    # graph, back through every stage and the RoI pooling grid.
                    cp_x, y_start, on_map = next_x, new_y, next_map
                    length, frame_top = new_len, new_top
                else:
                    cp_x, y_start, on_map = next_x.detach(), new_y.detach(), next_map.detach()
                    length, frame_top = new_len.detach(), new_top.detach()
                if self._jitter_active(branch, stage):
                    cp_x, y_start = self._jitter_state(cp_x, y_start)
                    _, on_map = brr_reference(cp_x, y_start, length, **geo, frame_top=frame_top)
        return preds, states, dict(input_xs=input_xs or None, reproj=reproj_stats,
                                   precond=precond_stats, moe=moe_stats)

    _JITTER_DEFAULTS = dict(branches=("aux",), after_stages=(0, 1), translate=0.01,
                            slope=0.02, curve=0.01, y_shift=0.02, prob=1.0)

    def _parse_stage_jitter(self, cfg):
        """Validated brr_cfg.stage_jitter, or None when it is off.

        None / False: off. True or a dict (an empty dict included): on, with
        the defaults for any key not given.
        """
        if cfg is None or cfg is False:
            return None
        cfg = {} if cfg is True else dict(cfg)
        unknown = set(cfg) - set(self._JITTER_DEFAULTS)
        if unknown:
            raise ValueError(f"brr_cfg.stage_jitter: unknown keys {sorted(unknown)}")
        if self.framed_update:
            # The framed state carries its own frame (top/bottom) and transports
            # the curve when start moves; a start jitter there is not a plain
            # shift of y_start.
            raise ValueError("brr_cfg.stage_jitter supports the global and anchored frames "
                             "(without the framed-update options) only")
        out = dict(self._JITTER_DEFAULTS, **cfg)
        branches = (out["branches"],) if isinstance(out["branches"], str) else out["branches"]
        out["branches"] = frozenset(str(b) for b in branches)
        if not out["branches"] or not out["branches"] <= {"main", "aux"}:
            raise ValueError("brr_cfg.stage_jitter.branches must be 'main' and/or 'aux'")
        out["after_stages"] = frozenset(int(s) for s in out["after_stages"])
        if not out["after_stages"] or not all(
                0 <= s < self.refine_layers - 1 for s in out["after_stages"]):
            raise ValueError("brr_cfg.stage_jitter.after_stages must name stages "
                             f"0..{self.refine_layers - 2} (the last stage hands nothing on)")
        for key in ("translate", "slope", "curve", "y_shift"):
            out[key] = float(out[key])
            if out[key] < 0.0:
                raise ValueError(f"brr_cfg.stage_jitter.{key} must be >= 0")
        out["prob"] = float(out["prob"])
        if not 0.0 <= out["prob"] <= 1.0:
            raise ValueError("brr_cfg.stage_jitter.prob must be in [0, 1]")
        return out

    def _jitter_active(self, branch, stage):
        return (self.training and self.stage_jitter is not None
                and branch in self.stage_jitter["branches"]
                and stage in self.stage_jitter["after_stages"])

    @torch.no_grad()
    def _jitter_state(self, cp_x, y_start):
        """Structured random jitter of a detached BRR state (brr_cfg.stage_jitter).

        cp_x [..., 4] (top -> bottom), y_start [...]. A linear change of x in y
        is added exactly by adding it at the control points' own y (they are
        equally spaced in the curve parameter), so translate and slope move the
        drawn curve by exactly a + b * (y_start - y); curve adds a bow of the
        drawn size at the middle of the control-point frame.
        """
        cfg = self.stage_jitter
        shape = y_start.shape

        def draw(half_width):
            return (torch.rand(shape, device=cp_x.device, dtype=cp_x.dtype) * 2.0 - 1.0) * half_width

        one_row = 1.0 / float(self.n_strips)
        frac = cp_x.new_tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        bottom = torch.ones_like(y_start) if self.cp_frame == "global" else y_start
        cp_y = bottom.unsqueeze(-1) * frac
        start = y_start.clamp(one_row, 1.0).unsqueeze(-1)
        # Bernstein weights of P1 + P2 sum to 3t(1 - t) = 3/4 at t = 1/2.
        bow = cp_x.new_tensor([0.0, 4.0 / 3.0, 4.0 / 3.0, 0.0])
        dx = (draw(cfg["translate"]).unsqueeze(-1)
              + draw(cfg["slope"]).unsqueeze(-1) * (start - cp_y)
              + draw(cfg["curve"]).unsqueeze(-1) * bow)
        dy = draw(cfg["y_shift"])
        if cfg["prob"] < 1.0:
            keep = (torch.rand(shape, device=cp_x.device) < cfg["prob"]).to(cp_x.dtype)
            dx = dx * keep.unsqueeze(-1)
            dy = dy * keep
        new_x = (cp_x + dx).clamp(-self.cp_x_margin, 1.0 + self.cp_x_margin)
        new_y = (y_start + dy).clamp(one_row, 1.0)
        return new_x, new_y

    def _precond_length(self, length, reg):
        """The length the preconditioner sees: the stage's predicted length."""
        if self.framed_update and self.length_mode == "residual":
            return length + reg[..., 1]
        return reg[..., 1]

    def gsrc_tokens(self, feats):
        """Global context tokens from the coarsest FPN level ([B, H*W, C])."""
        return None if self.gsrc is None else self.gsrc.tokens(feats[0])

    @contextlib.contextmanager
    def _frozen_bn_stats(self, enabled):
        """Run a training-mode pass that normalizes with batch statistics but
        leaves every BatchNorm running mean/var (and batch counter) untouched."""
        layers = [m for m in self.modules()
                  if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
        if not enabled or not layers:
            yield
            return
        for m in layers:
            m.track_running_stats = False
        try:
            yield
        finally:
            for m in layers:
                m.track_running_stats = True

    def forward_train(self, feats):
        batch_size = feats[-1].shape[0]
        clean_cp = self.prior_bank(batch_size)
        tokens = self.gsrc_tokens(feats)
        main_preds, main_states, main_extra = self._refine(feats, clean_cp, tokens)
        out = dict(main_preds=main_preds, main_states=main_states,
                   main_input_xs=main_extra["input_xs"], reproj=main_extra["reproj"],
                   precond=main_extra["precond"], moe=main_extra.get("moe", []))
        if self.gsrc is not None and getattr(self.gsrc, "last_stats", None):
            out["moe_gsrc"] = dict(self.gsrc.last_stats)

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
            with self._frozen_bn_stats(self.aux_bn_stats == "main"):
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
        layout = GTLayout(valid_targets, outs["main_preds"][0].device)
        fit_frame = "global" if self.brr_cp_loss_space == "global" else self.cp_frame
        # Every lane is fitted independently, so one call covers the batch.
        gt_flat = fit_global_cubic_to_clr_rows(layout.flat[:, 6:], self.prior_ys, self.img_w,
                                               self.brr_cp_fit_ridge, self.cp_x_margin,
                                               cp_frame=fit_frame, n_strips=self.n_strips,
                                               min_span=self.support_min_span,
                                               return_frame=True, box=self.brr_cp_fit_box)
        gt_cps = list(zip(*(torch.split(x, layout.counts) for x in gt_flat)))

        main = self._branch_losses(outs["main_preds"], outs["main_states"], valid_targets, gt_cps,
                                   self._main_pairs, list(range(self.refine_layers)), apply_brr=True,
                                   gate=self.main_gate, input_xs=outs.get("main_input_xs"),
                                   layout=layout, gt_flat=gt_flat)
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
        if outs.get("precond"):
            # Averaged over stages. Watch precond_clipped: if it sits near 1.0
            # the max_gain clamp is doing all the work and the correction is
            # saturated rather than adaptive.
            for key in outs["precond"][0]:
                diag[key] = torch.stack([st[key] for st in outs["precond"]]).mean()

        # MoE router diagnostics. Keys come from whatever the gate reports rather
        # than a hardcoded list, so a gate and a head from different versions
        # cannot disagree about the key set.
        moe_balance = None
        moe_sources = list(outs.get("moe", []))
        if outs.get("moe_gsrc"):
            moe_sources.append({f"gsrc_{k}": v for k, v in outs["moe_gsrc"].items()})
        if moe_sources:
            from .moe import finalize_moe_stats, merge_moe_stats
            acc = {}
            for st in moe_sources:
                merge_moe_stats(acc, {k: v for k, v in st.items()
                                      if not k.endswith("balance_loss")})
            diag.update(finalize_moe_stats(acc))
            balances = [v for st in moe_sources for k, v in st.items()
                        if k.endswith("balance_loss")]
            if balances:
                moe_balance = torch.stack([b.reshape(()) for b in balances]).mean()

        if "aux_preds" in outs:
            aux = self._branch_losses(outs["aux_preds"], outs["aux_states"], valid_targets, gt_cps,
                                      self._aux_pairs, self.aux_stages, apply_brr=self.aux_apply_brr,
                                      cls_target_mode=self.aux_cls_target_mode,
                                      gate=self.aux_gate, input_xs=outs.get("aux_input_xs"),
                                      layout=layout, gt_flat=gt_flat)
            cls = cls + self.aux_cls_loss_weight * aux["cls"]
            iou = iou + self.aux_reg_loss_weight * aux["iou"]
            sup = sup + self.aux_reg_loss_weight * aux["support"]
            cp = cp + self.aux_reg_loss_weight * aux["cp"]
            diag.update(aux_iou=aux["iou"].detach(), aux_num_pos=aux["num_pos"],
                        aux_conf_iou_l1=aux["conf_iou_l1"],
                        aux_conf_iou_rank=aux["conf_iou_rank"])

        if self.rank_loss_enabled:
            rank, rank_stats = self._rank_loss(outs["main_preds"], valid_targets)
            diag.update(rank_stats)
        else:
            rank = None

        losses = dict(
            loss_cls=cls * self.cls_loss_weight,
            loss_iou=iou,  # LaneIoULoss already applies iou_loss_weight
            loss_brr_support=sup * self.brr_support_weight,
            loss_brr_cp=cp * self.brr_cp_weight,
        )
        if rank is not None:
            losses["loss_rank"] = rank * self.rank_loss_weight
        if moe_balance is not None:
            # Hard gating only. cv_squared(importance) + cv_squared(load), already
            # scaled by the gate's balance_weight. Watch it against loss_cls: a
            # balance term that dominates is steering the model to spread tokens
            # rather than to detect lanes.
            losses["loss_moe_balance"] = moe_balance
        if seg_gt is not None:
            seg_loss = F.nll_loss(F.log_softmax(outs["seg"], dim=1), seg_gt,
                                  weight=self.seg_class_weights, ignore_index=self.seg_ignore_label)
            losses["loss_seg"] = seg_loss * self.seg_loss_weight
        # Keys without "loss" are logged but not summed by mmengine.
        losses.update(diag)
        return losses

    def _rank_loss(self, preds, valid_targets):
        """Pairwise ranking over the candidates NMS will compare.

        Quality is the narrow LaneIoU (the width the CULane metric
        approximates) against the best GT, computed without gradient: this loss
        supervises the score, never the geometry.
        """
        device = preds[0].device
        total = preds[0].new_zeros(())
        # Aggregate whatever diagnostics ranking.py reports rather than a
        # hardcoded list, so a head and a ranking module from different
        # versions cannot disagree about the key set.
        stats = {}
        counted = 0
        empty = None
        for stage in sorted(self.rank_loss_stages):
            if stage >= len(preds):
                continue
            pred = preds[stage]
            logits = pred[..., :2].float()
            score = (logits[..., 1] - logits[..., 0] if self.rank_score_space == "logit"
                     else F.softmax(logits, dim=-1)[..., 1])
            pred_xs = pred[..., 6:].float()
            with torch.no_grad():
                quality = torch.zeros_like(score)
                for b, target in enumerate(valid_targets):
                    if target.numel() == 0 or target.shape[0] == 0:
                        continue
                    geo_p = pred_xs[b] * (float(self.img_w - 1) / float(self.img_w))
                    geo_t = target[:, 6:] / float(self.img_w)
                    extent = {}
                    if self.rank_quality_extent:
                        start = (1.0 - pred[b, :, 2].float()).clamp(0.0, 1.0)
                        extent = dict(start=start,
                                      end=(start + pred[b, :, 5].float().clamp(0.0, 1.0))
                                      .clamp(0.0, 1.0))
                    iou = pairwise_lane_iou(geo_p, geo_t, self.lane_width,
                                            self.iou_w, self.iou_h, **extent)
                    quality[b] = torch.nan_to_num(iou, nan=0.0).max(dim=1).values
            loss, agg = batch_cluster_rank_loss(
                score, pred_xs, quality,
                lane_width=self.lane_width, img_w=self.img_w, img_h=self.img_h,
                **self.rank_loss_kwargs)
            if float(agg["rank_pairs"]) <= 0:
                # no supervised pair at this stage: do not dilute the others
                empty = {k: torch.zeros_like(v) for k, v in agg.items()}
                continue
            total = total + loss
            for key, value in agg.items():
                stats[key] = stats.get(key, torch.zeros_like(value)) + value
            counted += 1
        if counted:
            total = total / counted
            stats = {k: v / counted for k, v in stats.items()}
        elif empty is not None:
            stats = empty  # keep the logged key set stable
        return total, stats

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

    def _brr_losses(self, state, target, gt_cp, gt_cp_ok, gt_frame=None):
        """(support loss, CP loss) averaged over the given lanes."""
        sup, cp, ok = self._brr_lane_terms(state, target, gt_cp, gt_cp_ok, gt_frame)
        return sup.mean(), (cp * ok).sum() / ok.sum().clamp_min(1.0)

    def _brr_lane_terms(self, state, target, gt_cp, gt_cp_ok, gt_frame=None):
        """Per-lane BRR terms: support [P], CP [P] and the CP term's weight [P]."""
        n = float(self.n_strips)
        pred_start = (1.0 - state[:, 0]) * n
        pred_len = state[:, 1] * n
        tgt_start = (1.0 - target[:, 2]) * n
        tgt_len = target[:, 5]
        with torch.no_grad():
            p_round = pred_start.detach().round().clamp(0, n)
            t_round = tgt_start.round().clamp(0, n)
            adj_len = tgt_len - (p_round - t_round)
            if self.brr_length_target_min is not None:
                adj_len = adj_len.clamp_min(self.brr_length_target_min)
        sup = _smooth_l1(torch.stack([pred_start, pred_len], -1),
                         torch.stack([tgt_start, adj_len], -1), self.brr_support_beta)
        sup = (sup * sup.new_tensor(self.brr_component_weights)).mean(-1)

        scale = float(max(1, self.img_w - 1))
        if self.brr_cp_loss_space != "pred":
            cp_lane, ok = self._cp_lane_absolute(state, target, gt_cp, gt_cp_ok)
            return sup, cp_lane, ok
        if self.framed_update:
            # The GT control points were fitted on the GT's own frame
            # (gt_frame = [top, start]); the state's control points live on the
            # frame stored with it (state[:, 6:8]). Move the GT curve into the
            # prediction's frame (cubic inside its own support, linear
            # continuation outside, the same curve eval_cubic draws) before
            # comparing, and clamp to the range the state is kept in.
            with torch.no_grad():
                gt_cp = transport_cp(gt_cp, gt_frame[:, 0], gt_frame[:, 1],
                                     state[:, 6], state[:, 7]).clamp(
                    -self.cp_x_margin, 1.0 + self.cp_x_margin)
        elif self.cp_frame == "anchored":
            # The GT control points were fitted in the GT's own frame
            # (t = y / y_start_gt); the state lives in the predicted frame
            # (t = y / y_start_pred). Move the target into the predicted frame
            # before comparing, or a correct curve is charged for the
            # difference between the two start points.
            with torch.no_grad():
                ratio = (state[:, 0].detach().clamp(1e-3, 1.0)
                         / target[:, 2].clamp_min(1e-3)).clamp(0.2, 5.0)
                # For ratio > 1 the subdivision extrapolates the GT cubic past its
                # own support; on curved lanes that leaves the range the state is
                # clamped to (update_brr_state), giving a target no state can reach.
                # A no-op for ratio <= 1 (the result stays in the GT's convex hull).
                gt_cp = reparam_cp_to_frame(gt_cp, ratio).clamp(
                    -self.cp_x_margin, 1.0 + self.cp_x_margin)
        cp_elem = _smooth_l1(state[:, 2:6] * scale, gt_cp * scale, self.brr_cp_beta).mean(-1)
        return sup, cp_elem, gt_cp_ok.float()

    def _state_frame(self, state):
        """(top, bottom) of the image-y frame the state's raw control points live in.

        Framed update: stored with the state (columns 6:8). Legacy anchored: [0, y_s]
        with y_s clamped as the reference clamps it. Global: [0, 1]. Constants.
        """
        if self.framed_update:
            return state[:, 6].detach(), state[:, 7].detach()
        if self.cp_frame == "anchored":
            bottom = state[:, 0].detach().clamp(1.0 / float(self.n_strips), 1.0)
            return torch.zeros_like(bottom), bottom
        ones = state.new_ones(state.shape[0])
        return torch.zeros_like(ones), ones

    def _cp_loss_absolute(self, state, target, gt_cp, gt_cp_ok):
        """Control-point loss in a prediction-independent space (brr_cp_loss_space)."""
        cp, ok = self._cp_lane_absolute(state, target, gt_cp, gt_cp_ok)
        return (cp * ok).sum() / ok.sum().clamp_min(1.0)

    def _cp_lane_absolute(self, state, target, gt_cp, gt_cp_ok):
        """Per-lane CP term and its weight for brr_cp_loss_space 'global' / 'rows'."""
        scale = float(max(1, self.img_w - 1))
        pred_cp = state[:, 2:6]
        top, bottom = self._state_frame(state)
        if self.brr_cp_loss_space == "global":
            # The predicted curve, as drawn (cubic in its frame, linear continuation
            # outside), re-expressed on [0, 1]. transport_cp is linear in the
            # control points, so the gradient reaches them; the frames are constants.
            if self.cp_frame != "global":
                pred_cp = transport_cp(pred_cp, top, bottom,
                                       torch.zeros_like(top), torch.ones_like(bottom))
            cp_elem = _smooth_l1(pred_cp * scale, gt_cp * scale, self.brr_cp_beta).mean(-1)
            return cp_elem, gt_cp_ok.float()

        # "rows": the predicted Bezier curve (without the per-row residuals) against
        # the GT x on every row LaneIoU counts, in pixels.
        xs_gt = target[:, 6:]
        valid = torch.isfinite(xs_gt) & (xs_gt >= 0.0) & (xs_gt < float(self.img_w))
        ys = self.prior_ys.to(pred_cp.dtype)
        if self.cp_frame == "global":
            x = eval_global_cubic(pred_cp, ys)
        else:
            x = eval_cubic(pred_cp, ys, bottom, "support", y_top=top, slope_grad=self.slope_grad)
        row_err = _smooth_l1(x * scale, torch.where(valid, xs_gt, torch.zeros_like(xs_gt)),
                             self.brr_cp_beta)
        row_err = torch.where(valid, row_err, torch.zeros_like(row_err))
        count = valid.sum(-1)
        per_lane = row_err.sum(-1) / count.clamp_min(1)
        return per_lane, (count >= 2).float()

    def _branch_losses(self, preds_stages, states_stages, valid_targets, gt_cps, pairs_fn,
                       stages, apply_brr, cls_target_mode=None, gate=None, input_xs=None,
                       layout=None, gt_flat=None):
        if self.batched_loss and layout is not None:
            return self._branch_losses_batched(preds_stages, states_stages, layout, gt_flat,
                                               pairs_fn, stages, apply_brr, cls_target_mode,
                                               gate, input_xs)
        return self._branch_losses_loop(preds_stages, states_stages, valid_targets, gt_cps,
                                        pairs_fn, stages, apply_brr, cls_target_mode, gate,
                                        input_xs)

    def _branch_losses_batched(self, preds_stages, states_stages, layout, gt_flat, pairs_fn,
                               stages, apply_brr, cls_target_mode=None, gate=None, input_xs=None):
        """_branch_losses_loop for the whole batch at once.

        Per-image means are kept: each image's matched pairs are averaged, then
        the images are summed, exactly as the loop does.
        """
        device = preds_stages[0].device
        batch_size = preds_stages[0].shape[0]
        zero = preds_stages[0].new_zeros(())
        active = [s for s in stages if len(pairs_fn(s)) > 0]
        normalizer = float(max(1, batch_size * len(active)))
        gt_counts = torch.tensor([float(c) for c in layout.counts], device=device)
        scale = float(self.img_w - 1) / float(self.img_w)
        gt_cp_all, gt_ok_all, gt_frame_all = gt_flat

        cls_sum, iou_sum, sup_sum, cp_sum = zero, zero, zero, zero
        num_pos = 0
        align_err, align_cnt = zero, 0
        rank_sum, rank_cnt = zero, zero
        pos_stage = [0] * self.refine_layers
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
                required.add("lane_iou_dynamic")
            cls_targets = torch.zeros((len(pairs), batch_size, preds.shape[1]),
                                      dtype=torch.long, device=device)
            quality = torch.zeros(cls_targets.shape, dtype=preds.dtype, device=device) \
                if use_quality else None
            ignore = torch.zeros(cls_targets.shape, dtype=torch.bool, device=device) \
                if use_ignore else None
            use_brr = apply_brr and stage in self.brr_loss_stages
            cache = batched_cost_cache(preds.detach(), layout, self.img_w, self.img_h,
                                       self.lane_width, self.lane_width_cost, required,
                                       iou_shape=(self.iou_w, self.iou_h),
                                       iou_kind=self.cost_iou_type) if layout.num else {}
            for a, (assigner, weight) in enumerate(pairs):
                if layout.num == 0:
                    continue
                b, n, g = batched_assign(assigner, cache, layout)
                rb, rn, rg = b, n, g
                if gate_thr > 0.0 and b.numel() > 0:
                    if gate.gate_on == "output":
                        gate_iou = cache["lane_iou_dynamic"][b, n, g]
                    else:
                        with torch.no_grad():
                            gate_iou = 1.0 - self.gate_iou_fn(
                                input_xs[stage][b, n] * scale,
                                layout.flat[layout.flat_index(b, g), 6:] / float(self.img_w))
                    keep = batched_keep_mask(gate, gate_iou, layout.flat_index(b, g),
                                             layout.num, gate_thr)
                    b, n, g = b[keep], n[keep], g[keep]
                    if gate.mode == "cls_and_reg":
                        rb, rn, rg = b, n, g
                if use_ignore:
                    thr = self.ignore_iou_thr
                    if thr is not None and float(thr) < 1.0:
                        best = cache["lane_iou_dynamic"].masked_fill(
                            ~layout.mask[:, None, :], float("-inf")).max(dim=2).values
                        mask = best > float(thr)
                        mask[b, n] = False
                        ignore[a] = mask
                count = int(b.numel())
                pos_stage[stage] += count
                num_pos += count
                if count > 0:
                    cls_targets[a, b, n] = 1
                    j = layout.flat_index(b, g)
                if count > 0 and "lane_iou_dynamic" in cache:
                    with torch.no_grad():
                        conf = F.softmax(preds[b, n, :2].detach(), dim=-1)[:, 1]
                        pair_q = cache["lane_iou_dynamic"][b, n, g]
                        align_err = align_err + (conf - pair_q).abs().sum()
                        align_cnt += count
                        r_sum, r_cnt = per_image_spearman(conf, pair_q, b, batch_size)
                        rank_sum = rank_sum + r_sum
                        rank_cnt = rank_cnt + r_cnt
                if count > 0 and use_quality:
                    pair_iou = cache["lane_iou_dynamic"][b, n, g]
                    pair_scores = None
                    if mode == "task_aligned":
                        pair_scores = F.softmax(preds[b, n, :2].detach(), dim=-1)[:, 1]
                    quality[a, b, n] = quality_targets(
                        pair_iou, pair_scores, mode, self.task_align_alpha,
                        self.task_align_beta, j, layout.num)
                if rb.numel() == 0:
                    continue
                rj = layout.flat_index(rb, rg)
                target = layout.flat[rj]
                per_lane = self.iou_loss(preds[rb, rn, 6:] * scale,
                                         target[:, 6:] / float(self.img_w))
                lanes_per_img = per_image_sum(torch.ones_like(per_lane), rb, batch_size)
                denom = lanes_per_img.clamp_min(1.0)
                iou_sum = iou_sum + weight * (per_image_sum(per_lane, rb, batch_size) / denom).sum()
                if use_brr:
                    sup, cp, ok = self._brr_lane_terms(states[rb, rn], target, gt_cp_all[rj],
                                                       gt_ok_all[rj], gt_frame_all[rj])
                    sup_img = per_image_sum(sup, rb, batch_size) / denom
                    cp_img = (per_image_sum(cp * ok, rb, batch_size)
                              / per_image_sum(ok, rb, batch_size).clamp_min(1.0))
                    sup_sum = sup_sum + weight * sup_img.sum()
                    cp_sum = cp_sum + weight * cp_img.sum()
            weights = preds.new_tensor([w for _, w in pairs]).view(-1, 1)
            cls_sum = cls_sum + (self._cls_loss(preds[..., :2], cls_targets, gt_counts,
                                                quality, ignore) * weights).sum()

        return dict(cls=cls_sum / normalizer, iou=iou_sum / normalizer,
                    support=sup_sum / normalizer, cp=cp_sum / normalizer,
                    num_pos=torch.tensor(float(num_pos), device=device),
                    conf_iou_l1=(align_err / max(1, align_cnt)).detach(),
                    conf_iou_rank=(rank_sum / rank_cnt.clamp_min(1.0)).detach(),
                    pos_stage=[torch.tensor(float(c), device=device) for c in pos_stage])

    def _branch_losses_loop(self, preds_stages, states_stages, valid_targets, gt_cps, pairs_fn,
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
                                         self.lane_width, self.lane_width_cost, required,
                                         iou_fns=self.iou_fns,
                                         iou_shape=(self.iou_w, self.iou_h))
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
                                                   gt_cps[b][0][reg_cols], gt_cps[b][1][reg_cols],
                                                   gt_cps[b][2][reg_cols])
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
