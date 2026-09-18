# CLRBezierLane-R34 on tusimple. Dataset, schedule, test_cfg and evaluation are
# inherited from the CLRerNet-R34 baseline; only the head is replaced.
# Head settings mirror configs/clrbezier/culane/clrbezier_collab_perturb_r34.py.
_base_ = [
    "dataset_tusimple_clrernet.py",
    "../../_base_/default_runtime.py"
]
default_scope = "mmdet"
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
        "libs.lanedata",
        "libs.clrbezier",
    ],
    allow_failed_imports=False,
)
cfg_name = "clrbezier_tusimple_r34.py"

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
        seg_num_classes=7,
        prior_cfg=dict(delta_scale=0.1, visible_only=True, min_support=1.0 / 71.0, eps=1e-4),
        brr_cfg=dict(cp_x_margin=0.5),
        main_assigner=dict(
            type="HungarianLaneAssigner", cls_weight=1.0, point_weight=2.0, iou_weight=3.0),
        aux_cfg=dict(
            enabled=True,
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
            cls_loss_weight=2.0,
            use_focal=False,
            cls_bg_weight=0.4,
            iou_loss_weight=4.0,
            # CULane CLRerNet widths (same 800x320 input; CLRNet has no LaneIoU setting)
            lane_width=7.5 / 800,
            lane_width_cost=30.0 / 800,
            seg_loss_weight=1.0,
            seg_bg_weight=0.4,
            brr_support_loss_weight=0.2,
            brr_cp_loss_weight=0.05,
            brr_support_smooth_l1_beta=1.0,
            brr_cp_smooth_l1_beta=1.0,
            brr_cp_fit_ridge=1e-2,
            brr_loss_stages=[0, 1, 2],
        ),
        target_adapter=dict(
            lane_keys=["lanes"],
            seg_keys=["gt_masks"],
            start_y_convention="image_y",  # official PackCLRNetInputs: y0 = 1 - n_out / n_strips
            length_unit="count",
        ),
    ),
    test_cfg=dict(
        conf_threshold=0.40,  # CLRNet TuSimple (select by cross-validation for reporting)
        nms_thres=50,
        nms_topk=5,
        ori_img_w=1280,
        ori_img_h=720,
        cut_height=160,
    ),
)

# schedule: CLRNet R34 TuSimple (70 epochs, batch 32, AdamW 0.8e-3, cosine)
total_epochs = 168 # (36/15) * 70
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
train_dataloader = dict(batch_size=32)  # per GPU
randomness = dict(seed=0, deterministic=True)
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=0.8e-3))
param_scheduler = [
    dict(type="CosineAnnealingLR", eta_min=0, begin=0, T_max=total_epochs, end=total_epochs,
         by_epoch=True, convert_to_iter_based=True),
]
log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)