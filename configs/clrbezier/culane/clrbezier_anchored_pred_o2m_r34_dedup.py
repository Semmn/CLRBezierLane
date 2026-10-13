# The O2M jitter config with a learned NMS head (bbox_head.dedup_cfg, dedup.py)
# trained jointly from scratch. Its inputs are detached, so the o2m head trains
# as in the base config, and the checkpoint evaluates both ways:
#   dedup scores, no NMS (this config's test_cfg), and
#   o2m scores + NMS (the baseline):
#     --cfg-options model.bbox_head.dedup_cfg.use_at_test=False model.test_cfg.use_nms=True
# The cheaper first test is the _dedup_ft config on an existing O2M checkpoint.
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
cfg_name = "clrbezier_anchored_pred_o2m_r34_dedup.py"

img_w, img_h, num_points = 800, 320, 72

_o2m = dict(type="SimOTALaneAssigner", candidate_topk=4, min_dynamic_k=1,
            cls_weight=1.0, point_weight=0.0, iou_weight=3.0)
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
        num_priors=192, # Number of priors
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        img_w=img_w,
        img_h=img_h,
        roi_mid_channels=48,
        seg_num_classes=5,
        # separate final classification layer for the auxiliary branch;
        # the shared towers still receive its gradient
        aux_cls_head=False,
        lateral_cfg=None,
        look_forward_twice=False,
        prior_cfg=dict(param='sigmoid', delta_scale=0.1, visible_only=True, min_support=1.0 / 71.0, eps=1e-4),
        brr_cfg=dict(cp_x_margin=0.1, cp_transport=True, length_mode="fresh", frame_grad=False,
            stage_jitter=False
        ), # cp_transport, length_mode, min_span, frame_grad apply to cp_frame="anchored"/"support" only (rejected for "global").
        main_assigner=_o2m,
        # per-stage override; stages not listed use main_assigner
        main_stage_assigners={"0": _o2m, "1": _o2m, "2": _o2m},
        main_quality_gate=dict(
            # narrow LaneIoU (~CULane metric IoU); 0.5 = already a metric TP
            min_iou=[0.0, 0.3, 0.5],
            keep_best=True,        # every GT keeps its best pair
            mode="cls_and_reg",    # "cls_only": gated pairs still get regression
            gate_on="output",      # "input": Cascade R-CNN definition (pair with re-projection)
            warmup_iters=1500,     # thresholds ramp from 0
        ),
        reproject_cfg=None,
        cp_frame="anchored", # "global" | "anchored" | "support"
        cp_precond_cfg=None,
        slot_expert_cfg=None,
        dedup_cfg=dict(train_mode="joint", target="o2m_top1", use_at_test=True),
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
            enabled=False, # Disable Auxiliary branch
            num_groups=3, # No additional groups
            stages=[0, 1, 2],
            assigners=[
                dict(type="SimOTALaneAssigner", candidate_topk=4, min_dynamic_k=1,
                            cls_weight=1.0, point_weight=0.0, iou_weight=3.0),
            ],
            assigner_weights=[1.0],
            stage_assigner_ids={"0": [0], "1": [0], "2": [0]},
            cls_loss_weight=0.5,
            reg_loss_weight=0.5,
            noise_t=50,
            random_t=False,
            apply_brr_loss=True,
            bn_stats="main",
        ),
        # Perturbation off, auxiliary branch on: zero noise AND no clamps, so the
        # aux branch gets exactly the clean priors (with clamp_x=True alone, the
        # x clamp to [0.01, 0.99] would still move the off-image control points
        # of about half the priors). The aux regression then equals the main
        # one; aux adds one-to-many (TopK/SimOTA) supervision on the same
        # predictions, with its own cls layer (aux_cls_head=True).
        # noise_t / random_t in aux_cfg have no effect in this setting.
        perturb_cfg=dict(
            timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2,
            noise_scale=1.0, coeff_dim=4, coeff_scale=1.0, use_tanh=True,
            max_translate=0.06, max_slope=0.16, max_curve=0.08, max_y_shift=0.06,
            clamp_x=True, clamp_y=True,
        ),
        loss_cfg=dict(
            use_focal=True,
            cls_loss_weight=2.0,
            # NOTE: quality focal loss is smaller in scale than the weighted CE.
            # Retune cls_loss_weight after the first run (try 2.0 -> 4.0).
            iou_loss_type="laneiou",     # regression loss = 1 - GLIoU
            cost_iou_type="laneiou",     # "laneiou" or "gliou"
            cls_bg_weight=0.4,
            iou_loss_weight=4.0,
            iou_eval_shape=(320, 800),
            lane_width=7.5 / 800,        # half-width, paper w_lane = 15/800
            lane_width_cost=30.0 / 800,  # half-width, paper w_lane = 60/800
            seg_loss_weight=1.0,
            seg_bg_weight=0.4,
            brr_cp_loss_space="pred", # "rows" | "pred (default)" | "global"
            brr_support_loss_weight=0.2,
            brr_cp_loss_weight=0.1,
            brr_support_smooth_l1_beta=1.0,
            brr_cp_smooth_l1_beta=1.0,
            brr_cp_fit_ridge=1e-2, # ignored in the "rows" mode
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
        ),
        target_adapter=dict(
            lane_keys=None,             # pin after running the probe, e.g. ["lanes"]
            seg_keys=None,              # e.g. ["gt_masks"]
            start_y_convention="auto",  # verified against x validity, locked on batch 1
            length_unit="auto",
        ),
        gsrc_cfg=None,
        query_attn_cfg=None,
    ),
    test_cfg=dict(
        # Default CLRerNet uses conf_threshold=0.41
        conf_threshold=0.40,
        use_nms=False,
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

load_from = None   # from scratch (the backbone still starts from ImageNet)
