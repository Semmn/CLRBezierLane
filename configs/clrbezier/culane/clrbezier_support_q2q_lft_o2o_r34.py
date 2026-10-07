# Support-conditioned Bezier frame (cp_frame="support"), otherwise identical to
# clrbezier_anchored_o2m_r34.py. The four control points sit at the thirds of
# the lane's own visible span [start_y - span(length), start_y], so the state is
# (start_y, length, P0x..P3x) and start, length and shape are separate things.
#
# brr_cfg knobs (defaults shown) and what to try if training is unstable:
#   cp_transport=True  : a start/length step moves only the support; the curve
#                        stays where it was and dP is a pure shape correction.
#                        False = reinterpret the old coefficients in the new
#                        frame (the coupled version; ablation).
#   length_mode="residual": length is persistent state (input + delta), since
#                        the frame is built from it. "fresh" = CLRerNet's
#                        per-stage absolute length (starts near 0: the frame
#                        collapses to min_span at init; ablation only).
#   min_span=0.1       : smallest frame span in image y (~7 rows). Short lanes
#                        are fitted on this span instead of on 2-3 rows.
#   frame_grad=False   : LaneIoU does not reach start/length through the curve;
#                        they learn from loss_brr_support only, as in "global".
#                        True = the fully coupled gradient (ablation).
# Pairwise ranking inside NMS duplicate clusters.
#
# F1 at a re-selected threshold depends only on the order of the scores, so a
# monotone rescaling is worth zero F1 — which is why CE (optimal at ~0.9) and
# focal (optimal at ~0.5) give the same F1 and leave the misalignment intact.
# The comparison NMS actually makes, "which of these near-duplicates is best",
# appears in no loss. This adds it, and only it.
#
# The loss constrains score *differences* within a cluster, so the score
# distribution does not move and the confidence threshold does not need
# re-tuning when it is switched on.
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
cfg_name = "clrbezier_support_q2q_lft_o2o_r34.py"

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
        look_forward_twice=True,
        prior_cfg=dict(delta_scale=0.1, visible_only=True, min_support=1.0 / 71.0, eps=1e-4),
        brr_cfg=dict(cp_x_margin=0.5, cp_transport=True, length_mode="residual",
                     min_span=0.1, frame_grad=False),
        main_assigner=dict(type="HungarianLaneAssigner", cls_weight=1.0, point_weight=2.0, iou_weight=3.0),
        # per-stage override; stages not listed use main_assigner
        main_stage_assigners=None,
        main_quality_gate=None,
        reproject_cfg=None,
        cp_frame="support", # "global" | "anchored" | "support"
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
            iou_eval_shape=(320, 1640),
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
        gsrc_cfg=None,
        query_attn_cfg=dict(
            stages=[0, 1, 2],
            position='pre_q2g', # 'post_q2g', 'pre_q2g', 'post_gather'
            num_heads=2,
            use_state_embedding=True,
            use_pairwise_bias=True,
            gate_init=0.0,
            detach_state=True,
        ),
    ),
    test_cfg=dict(
        # Default CLRerNet uses conf_threshold=0.41
        conf_threshold=0.80,
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
total_epochs = 36
checkpoint_config = dict(interval=total_epochs)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Batch size (Number of batch size per GPU)
train_dataloader=dict(batch_size=32)
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

# Watch rank_pair_acc against loss_cls.
#
# Under one-to-one assignment the cluster holds one matched positive and
# several unmatched duplicates that CE labels 0 identically, so this loss
# orders exactly what CE is indifferent to: complementary.
#
# Under one-to-many the duplicates are all labelled 1, so CE pulls them
# together while this pushes them apart. If loss_cls degrades while
# rank_pair_acc rises, that conflict is live: lower loss_weight, or split the
# score into a thresholding head and an ordering head.
