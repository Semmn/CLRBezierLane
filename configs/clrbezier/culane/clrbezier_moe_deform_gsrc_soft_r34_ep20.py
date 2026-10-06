# Soft MoE on the GSRC token-block FFN -- the one placement in this head that
# matches V-MoE's own: the FFN of a block over real tokens (250 at (10, 25)),
# rather than a leaf read-out.
#
# This is the upstream alternative to routing the sampling offsets. Which one is
# right is decided by tools/clrbezier/probe_bow.py, not by preference:
#
#   features beat geometry  ->  the pooled features already carry the curvature
#                               and the read-out is not using it. A head-side or
#                               offset-side mixture is the targeted fix.
#   features add nothing    ->  the information never survived ROI sampling.
#                               Routing has to sit upstream -- here, or on the
#                               offsets, where it changes WHAT gets encoded.
#
# Note the token count is favourable here in a way it is not in the head: 250
# tokens x batch, all of them informative, against ~256 assigned positives per
# step on the query side. This is the one spot where the sparse-MoE literature's
# assumptions are not wildly violated.
_base_ = [
    "dataset_culane_clrernet.py",
    "../../_base_/default_runtime.py"
]

default_scope = 'mmdet'

# Lists are replaced (not merged) by mmengine: copy the official import list
# from configs/clrernet/culane/clrernet_culane_dla34.py and append libs.clrbezier.
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
        "libs.clrbezier",
    ],
    allow_failed_imports=False,
)
cfg_name = "clrbezier_moe_deform_gsrc_soft_r34_ep20.py"

img_w, img_h, num_points = 800, 320, 72
model = dict(
    type="CLRerNet",
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type="mmdet.ResNet",
        depth=34,
        num_stages=4,
        out_indices=(1, 2, 3),  # C3, C4, C5 -> [128, 256, 512] channels
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet34"),
    ),
    neck=dict(
        type="CLRerNetFPN",
        in_channels=[128, 256, 512], #DLA34
        out_channels=64,
        num_outs=3,
    ),
    bbox_head=dict(
        _delete_=True,
        type="CLRBezierHead",
        num_points=num_points,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_priors=35, # Number of priors
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        img_w=img_w,
        img_h=img_h,
        roi_mid_channels=48,
        seg_num_classes=5,
        # separate final classification layer for the auxiliary branch;
        # the shared towers still receive its gradient
        aux_cls_head=True,
        lateral_cfg=None,
        look_forward_twice=False,
        roi_gather_cfg=dict(
            type="CurveAlignedDeformableROIGather",
            deform_num_curve_samples=18,
            deform_num_offsets=4,
            deform_offset_mode="normal",
            deform_max_normal_offset=2.0,
            deform_init_mode="symmetry_broken",   # "zero" reproduces the old bug
            deform_residual_scale=0.1,            # small, NOT zero
            deform_offset_init_gain=0.1,
            detach_reference=True,
            deform_moe_cfg=dict(
                num_experts=4,      # 2-4 is the useful range: with ~8 positives per
                                    # image only ~256 tokens per step carry a
                                    # regression signal, so E=8 leaves ~32 each.
                mode="soft",
                temperature=1.0,    # leave it; read the entropy instead of tuning this
            ),
        # Watch, in this order:
        #   moe_entropy            ln(4)=1.386 at init. The whole experiment.
        #   moe_expert_divergence  > 0 at init and should stay > 0. If it decays toward 0
        #                          the experts collapsed onto each other.
        #   deform_offset_spread   the offset fan. ~0 means the symmetry broke down and
        #                          the deformable branch is one bilinear sample again.
        #   moe_load_cv2           0 is even. Informational in soft mode (no expert can
        #                          starve), diagnostic in hard mode.
        ),
        prior_cfg=dict(delta_scale=0.1, visible_only=True, min_support=1.0 / 71.0, eps=1e-4),
        # legacy_anchored_prior=True keeps the anchored side priors exactly as
        # they were trained before the 2026-10-06 globalize fix, so this config
        # reproduces its reported results. Drop it for new runs.
        brr_cfg=dict(cp_x_margin=0.5, legacy_anchored_prior=True),
        main_assigner=dict(type="HungarianLaneAssigner", cls_weight=1.0, point_weight=2.0, iou_weight=3.0),
        # per-stage override; stages not listed use main_assigner
        main_stage_assigners=None,
        main_quality_gate=None,
        reproject_cfg=None,
        cp_frame="anchored", # defaults to "global"
        cp_precond_cfg=None,
        rank_loss_cfg=dict(
            enabled=False,
            loss_weight=1.0,         # lower than the one-to-one default: the
                                     # CE conflict is live under one-to-many
            stages=[0,1,2],
            cluster_mode="distance", # CLRNet's lane NMS merges by mean |dx|, not
                                     # by IoU; "iou" can only reach 2*lane_width
                                     # (15 px) and misses most of each cluster
            nms_thres=50,            # defaults to model.test_cfg.nms_thres
            margin=0.05,
            score_space="logit",  # NOT "prob": in probability space the
                                  # loss floors at softplus(-1/tau) and the
                                  # gradient dies for confident duplicates
            tau=1.0,              # logit units now, so 0.5 is sharper than
                                  # it was on probabilities
            weight_mode='boundary',
            decisive_thr=0.5,

            weight_by_gap=True,
            positives_only=False,
        ),
        aux_cfg=dict(
            enabled=True, # Disable Auxiliary branch
            num_groups=3,
            stages=[0, 1, 2],
            assigners=[
                dict(type="TopKLaneAssigner", topk=4,
                        cls_weight=0.0, point_weight=2.0, iou_weight=3.0),
                dict(type="SimOTALaneAssigner", candidate_topk=10, min_dynamic_k=1,
                        cls_weight=0.25, point_weight=1.0, iou_weight=3.0),
            ],
            assigner_weights=[1.0, 1.0],
            stage_assigner_ids={"0": [0], "1": [0], "2": [1]},
            cls_loss_weight=0.5,
            reg_loss_weight=0.5,
            noise_t=50,
            random_t=True,
            apply_brr_loss=True,
        ),
        perturb_cfg=dict(
            timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2,
            noise_scale=1.0, coeff_dim=4, coeff_scale=1.0, use_tanh=True,
            max_translate=0.06, max_slope=0.16, max_curve=0.08, max_y_shift=0.06,
            clamp_x=True, clamp_y=True,
        ),
        loss_cfg=dict(
            use_focal=False,
            iou_loss_type="laneiou",     # regression loss = 1 - GLIoU
            cost_iou_type="laneiou",     # "laneiou" or "gliou"
            cls_bg_weight=0.4,
            iou_loss_weight=4.0,
            lane_width=7.5 / 800,        # half-width, paper w_lane = 15/800
            lane_width_cost=30.0 / 800,  # half-width, paper w_lane = 60/800
            seg_loss_weight=1.0,
            seg_bg_weight=0.4,
            brr_support_loss_weight=0.2,
            brr_cp_loss_weight=0.1,
            brr_support_smooth_l1_beta=1.0,
            brr_cp_smooth_l1_beta=1.0,
            brr_cp_fit_ridge=1e-2,
            brr_loss_stages=[0, 1, 2],

            # "hard" (baseline) | "iou" | "task_aligned"
            cls_target_mode="hard",
            aux_cls_target_mode="hard",   # "hard" keeps the auxiliary branch binary
            qfl_beta=2.0,
            # task_aligned only: t = s^alpha * u^beta, normalized per GT
            task_align_alpha=1.0,
            task_align_beta=6.0,
            # None disables; 0.7-0.8 is a reasonable starting band
            ignore_iou_thr=None,
            # NOTE: quality focal loss is smaller in scale than the weighted CE.
            # Retune cls_loss_weight after the first run (try 2.0 -> 4.0).
            cls_loss_weight=2.0,
        ),
        target_adapter=dict(
            lane_keys=None,             # pin after running the probe, e.g. ["lanes"]
            seg_keys=None,              # e.g. ["gt_masks"]
            start_y_convention="auto",  # verified against x validity, locked on batch 1
            length_unit="auto",
        ),
        gsrc_cfg=dict(
            source="coarsest",   # coarsest FPN level (P5); 800x320 input -> (10, 25)
            stages=[0, 1, 2],    # try [2] to inject only at the final stage
            context_cfg=dict(
                feat_size=(10, 25),
                num_layers=2,
                ffn_dim=256,
                ffn_drop=0.1,
                attn_drop=0.1,
                context_branch="transformer",  # "segman" (needs natten + selective_scan) | "none"
                moe_cfg=dict(
                    num_experts=4, # moe_cfg has no effect on "segman"
                    mode="soft",
                ),
                # Same reading as the deformable version: moe_entropy below ln(4) means the gate
                # specialized, flat at ln(4) means it did not. moe_expert_divergence must stay
                # above 0 or the experts have collapsed onto each other.
                num_heads=4,
                use_pos_embed=True,
                # The structure branch's residual is fixed here. The ported
                # RowColumnBlock added norm(x) instead of x, so it was not a
                # residual block at all: it could not represent the identity, it
                # destroyed the activation scale at every block, and the gradient
                # had to pass through LayerNorm's rank-deficient Jacobian with no
                # highway back. Set struct_residual="legacy" to reproduce it.
                struct_residual="pre_norm",
                struct_layer_scale=1e-2,   # small, not 0: at 0 the branch is dead
            ),
            injection_cfg=dict(
                # gamma starts at 0, so the context branch -- and the router inside it
                # -- receives NO gradient on step 1 and only begins learning once the
                # gate opens. Harmless for a long run, wasteful for a router you want
                # to watch sharpen. 1e-2 starts it live while staying near-identity.
                num_heads=4,
                fuse="gate",               # "concat" reproduces the v2-head style fusion
                gate_init=1e-2,             # 1e-2 to make GSRC active from iteration 0
                query_self_attention=True,  # separate ablation for the one-to-one branch
            ),
        ),
        query_attn_cfg=None
    ),
    test_cfg=dict(
        # Default CLRerNet uses conf_threshold=0.41
        conf_threshold=0.85,
        use_nms=True,
        as_lanes=True,
        extend_bottom=True,
        nms_thres=50,
        nms_topk=4,
        ori_img_w=1640,
        ori_img_h=590,
        cut_height=270,
    ),
)

# Number of epochs
total_epochs = 20 # Single GPU Training setup with batch size 32
checkpoint_config = dict(interval=total_epochs)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Batch size (Number of batch size per GPU)
train_dataloader=dict(batch_size=32) # Batch Size 32
randomness = dict(seed=0, deterministic=True)
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type="AdamW", lr=6e-4))
param_scheduler = [
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, T_max=total_epochs, end=total_epochs, by_epoch=True, convert_to_iter_based=True),
]
log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)