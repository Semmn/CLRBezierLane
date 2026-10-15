# Global Bezier frame (cp_frame="global"), otherwise identical to
# clrbezier_support_rows_o2o_p192_r34.py (same data, aux, perturbation, losses,
# lr 3e-4, warm-up, clip, spike hook), so the two runs differ only in the frame.
#
# Why: in run1 of the support config every GPU's loss jumped at the same
# iteration (e2/368, e2/497-512, e2/555-564) on disjoint batches, led by
# loss_brr_support and followed by loss_brr_cp and loss_iou, with cls and seg
# unchanged. In the support frame the curve is drawn inside the predicted
# start/length span, so a start/length drift reshapes every lane. In the global
# frame the control points sit at fixed y = 0, 1/3, 2/3, 1: start/length only
# decide which rows are visible, so the shape cannot follow them. The two
# spike_head_fixes switches have nothing to act on here (no continuation
# outside the frame), and the support-only brr_cfg keys are rejected for
# "global", so brr_cfg keeps only cp_x_margin.
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
    "dataset_curvelanes_clrbezier.py",
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
cfg_name = "clrbezier_global_pred_o2m_r34_tie_cost.py"

img_w, img_h, num_points = 800, 320, 72
_o2m_tie = dict(type="SimOTALaneAssigner", candidate_topk=4, min_dynamic_k=1,
                cls_weight=0.15, cls_cost_mode="linear", point_weight=0.0, iou_weight=3.0)  # delta = 0.05

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
        in_channels=[128, 256, 512],  # ResNet-34 C3-C5
        out_channels=64,
        num_outs=3,
    ),
    bbox_head=dict(
        _delete_=True,
        type="CLRBezierHead",
        num_points=num_points,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_priors=192, # Number of priors (official CLRerNet / CurveLanes; was 35)
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        img_w=img_w,
        img_h=img_h,
        roi_mid_channels=48,
        seg_num_classes=2,
        # separate final classification layer for the auxiliary branch;
        # the shared towers still receive its gradient
        aux_cls_head=False,
        lateral_cfg=None,
        look_forward_twice=False,
        prior_cfg=dict(param='sigmoid', delta_scale=0.1, visible_only=True, min_support=1.0 / 71.0, eps=1e-4),
        # cp_transport, length_mode, min_span, frame_grad and continuation_grad
        # exist only for cp_frame="anchored"/"support" (rejected for "global").
        brr_cfg=dict(cp_x_margin=1.0),
        main_assigner=_o2m_tie,
        # per-stage override; stages not listed use main_assigner
        main_stage_assigners={"0": _o2m_tie, "1": _o2m_tie, "2": _o2m_tie},
        main_quality_gate=dict(
            # narrow LaneIoU (~CULane metric IoU); 0.5 = already a metric TP
            min_iou=[0.0, 0.3, 0.5],
            keep_best=True,        # every GT keeps its best pair
            mode="cls_and_reg",    # "cls_only": gated pairs still get regression
            gate_on="output",      # "input": Cascade R-CNN definition (pair with re-projection)
            warmup_iters=1500,     # thresholds ramp from 0
        ),
        reproject_cfg=None,
        cp_frame="global", # "global" | "anchored" | "support"
        cp_precond_cfg=None,
        rank_loss_cfg=dict(
            enabled=False,
            loss_weight=1.0,         # lower than the one-to-one default: the
                                     # CE conflict is live under one-to-many
            stages=[0,1,2],
            cluster_mode="distance", # CLRNet's lane NMS merges by mean |dx|, not
                                     # by IoU; "iou" can only reach 2*lane_width
                                     # (15 px) and misses most of each cluster
            nms_thres=15,            # match model.test_cfg.nms_thres (CurveLanes)
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
            bn_stats="main",
        ),
        perturb_cfg=dict(
            timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2,
            noise_scale=1.0, coeff_dim=4, coeff_scale=1.0, use_tanh=True,
            max_translate=0.06, max_slope=0.16, max_curve=0.08, max_y_shift=0.06,
            clamp_x=True, clamp_y=True,
        ),
        loss_cfg=dict(
            use_focal=True,
            iou_loss_type="laneiou",     # regression loss = 1 - GLIoU
            cost_iou_type="laneiou",     # "laneiou" or "gliou"
            # NOTE: quality focal loss is smaller in scale than the weighted CE.
            # Retune cls_loss_weight after the first run (try 2.0 -> 4.0).
            cls_loss_weight=2.0,
            cls_bg_weight=0.4,
            iou_loss_weight=4.0,
            # CULane's (320, 1640) does not apply here. Official CLRerNet uses a
            # per-image eval_shape = (crop_h / ori_h * 224, 224); for the
            # 2560x1440 images that is (800 / 1440 * 224, 224) = (124, 224).
            iou_eval_shape=(124, 224),
            lane_width=2.5 / 224,        # half-width in the 224-wide eval space (official CurveLanes)
            lane_width_cost=10.0 / 224,  # official CurveLanes cost width
            seg_loss_weight=2.0,
            seg_bg_weight=0.4,
            brr_cp_loss_space="pred", # "rows" | "pred (default)" | "global"
            brr_support_loss_weight=0.2,
            brr_cp_loss_weight=0.1,
            brr_support_smooth_l1_beta=1.0,
            brr_cp_smooth_l1_beta=1.0,
            brr_cp_fit_ridge=1e-2,
            brr_loss_stages=[0, 1, 2],
            # clamp CLRNet's adjusted length target at 2 rows (kept so this run
            # differs from the support run only in the frame; in the global
            # frame it only changes the length target, not the shape)
            brr_length_target_min=2.0,
            brr_cp_fit_box=True,

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
            lane_keys=["lanes"],
            seg_keys=["gt_masks"],
            start_y_convention="image_y",  # official PackCLRNetInputs: y0 = 1 - n_out / n_strips
            length_unit="count",
        ),
        gsrc_cfg=None,
        query_attn_cfg=None,
    ),
    test_cfg=dict(
        # CurveLanes values (official CLRerNet curvelanes branch uses 0.42 with
        # focal loss; this CE-trained head needs its own sweep on valid).
        conf_threshold=0.40,
        use_nms=True,
        as_lanes=True,
        extend_bottom=True,
        nms_thres=15,
        nms_topk=16,   # CurveLanes images have up to 16 lanes
        # Lanes are decoded normalized within the crop; CurvelanesMetric maps
        # them back with each image's crop_offset. CULane's 1640/590/270 here
        # would squeeze every lane into the bottom 54% of the crop.
        ori_img_w=1,
        ori_img_h=1,
        cut_height=0,
    ),
)

# Number of epochs
total_epochs = 36
checkpoint_config = dict(interval=total_epochs)
# Validation on valid/valid.txt every val_interval epochs (official CurveLanes: 3).
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Batch size (Number of batch size per GPU)
train_dataloader=dict(batch_size=32)
randomness = dict(seed=0, deterministic=True)
# Gradient clipping (global L2 norm). mmengine logs grad_norm once clip_grad is set:
# if clipping fires on most iterations, raise max_norm to ~2-3x the typical value.
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type="AdamW", lr=6e-4))
                    #  ,clip_grad=dict(max_norm=35, norm_type=2))
# Records every iteration whose loss or grad_norm exceeds 2x its running median
# (with the batch's image names) in work_dir/loss_spikes_rank*.jsonl.
custom_hooks = [dict(type="LossSpikeHook", factor=2.0)]
# grad_norm is averaged over the 50-iteration log window; also log its maximum.
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True,
    custom_cfg=[dict(data_src='grad_norm', method_name='max', window_size=50,
                     log_name='grad_norm_max')])
param_scheduler = [
    # dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=1500),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, T_max=total_epochs,
         end=total_epochs, by_epoch=True, convert_to_iter_based=True),
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
