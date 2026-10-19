# Lane-aware foveation v1, run 2 (lane_foveation_design.md): clrernet_curvelanes_r34_4x6.py with the foveated
# dataset file (augment at 1600 x 640, then a fixed warp into the same 800 x 320 input that magnifies the far field;
# predictions mapped back before the metric). Model, schedule, batch: unchanged, so the pair isolates the warp.
# Needs lane_foveation.patch (libs/datasets/foveation.py).
# The baseline's notes follow.
# CLRerNet-R34 on CurveLanes, the baseline of the foveation wave (lane_foveation_design.md, run 1).
# clrernet_curvelanes_r34.py made standalone and fitted to 4 GPUs:
#   - model written out in full (base_clrernet_r34.py + the curvelane-branch overrides of clrernet_curvelanes_r34.py);
#   - batch 6 per GPU x 4 GPUs = 24, the curvelane branch's batch (24 on one GPU), same lr 6e-4; BN is synced
#     across the 4 GPUs (sync_bn) so its statistics also see 24 images, as on the single GPU;
#   - custom_imports without "libs.lanedata" (that module is libs.datasets in this tree).
# Schedule as the curvelane branch: 15 epochs, AdamW 6e-4, cosine to 0, no EMA.
# Pair: clrernet_curvelanes_r34_4x6_fovea.py (the same with the foveated dataset file).

_base_ = [
    "dataset_curvelanes_clrernet_fovea.py",
    "../../_base_/default_runtime.py",
]

default_scope = "mmdet"

custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
        "libs.datasets.foveation",
    ],
    allow_failed_imports=False,
)
cfg_name = "clrernet_curvelanes_r34_4x6_fovea.py"

model = dict(
    type="CLRerNet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
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
        in_channels=[128, 256, 512],
        out_channels=64,
        num_outs=3,
    ),
    bbox_head=dict(
        type="CLRerHead",
        anchor_generator=dict(
            type="CLRerNetAnchorGenerator",
            num_priors=192,
            num_points=72,
        ),
        img_w=800,
        img_h=320,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        attention=dict(type="ROIGather"),
        loss_cls=dict(type="KorniaFocalLoss", alpha=0.25, gamma=2, loss_weight=2.0),
        loss_bbox=dict(type="SmoothL1Loss", reduction="none", loss_weight=0.2),
        loss_iou=dict(type="LaneIoULoss", lane_width=2.5 / 224, loss_weight=4.0),  # curvelane branch
        loss_seg=dict(
            type="CLRNetSegLoss",
            loss_weight=2.0,  # curvelane branch
            num_classes=2,  # 1 lane + background
            ignore_label=255,
            bg_weight=0.4,
        ),
        use_segman_decoder=False,
        segman_decoder_params=None,
        use_assign_regularizer=False,
        regularizer_ema_beta=0.01,
    ),
    train_cfg=dict(
        assigner=dict(
            type="DynamicTopkAssigner",
            max_topk=4,
            min_topk=1,
            cost_combination=1,
            cls_cost=dict(type="FocalCost", weight=1.0),
            reg_cost=dict(type="DistanceCost", weight=0.0),
            iou_dynamick=dict(type="LaneIoUCost", lane_width=2.5 / 224,
                              use_pred_start_end=False, use_giou=True),
            iou_cost=dict(type="LaneIoUCost", lane_width=10 / 224,
                          use_pred_start_end=True, use_giou=True),
        )
    ),
    test_cfg=dict(
        conf_threshold=0.42,  # curvelane branch; sweep_conf_eval.py picks it on val
        use_nms=True,
        as_lanes=True,
        extend_bottom=True,
        nms_thres=15,
        nms_topk=16,
        # Lanes are decoded in crop-normalized y; FoveatedCurvelanesMetric unwarps them, then restores each
        # image's original coordinates from metainfo["crop_offset"].
        ori_img_w=1,
        ori_img_h=1,
        cut_height=0,
    ),
)

# schedule: curvelane branch (15 epochs, batch 24, AdamW 6e-4, cosine)
total_epochs = 15
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
train_dataloader = dict(batch_size=6)  # per GPU; x 4 GPUs = 24
sync_bn = "torch"  # BN over the 4 GPUs' 24 images, as the single-GPU batch of 24
randomness = dict(seed=0, deterministic=True)
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=6e-4))
param_scheduler = [
    dict(type="CosineAnnealingLR", eta_min=0, begin=0, T_max=total_epochs, end=total_epochs,
         by_epoch=True, convert_to_iter_based=True),
]
