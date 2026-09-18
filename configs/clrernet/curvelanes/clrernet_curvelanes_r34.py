# CLRerNet-R34 on CurveLanes. Model = configs/clrernet/base_clrernet_r34.py;
# dataset-dependent settings from the CLRerNet features/curvelane branch
# (configs/clrernet/curvelanes/clrernet_curvelanes_dla34.py).
_base_ = [
    "../base_clrernet_r34.py",
    "dataset_curvelanes_clrernet.py",
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
        "libs.lanedata",
    ],
    allow_failed_imports=False,
)
cfg_name = "clrernet_curvelanes_r34.py"

model = dict(
    bbox_head=dict(
        loss_iou=dict(type="LaneIoULoss", lane_width=2.5 / 224, loss_weight=4.0),
        loss_seg=dict(loss_weight=2.0, num_classes=2),  # 1 lane + background
    ),
    train_cfg=dict(
        assigner=dict(
            iou_dynamick=dict(type="LaneIoUCost", lane_width=2.5 / 224,
                              use_pred_start_end=False, use_giou=True),
            iou_cost=dict(type="LaneIoUCost", lane_width=10 / 224,
                          use_pred_start_end=True, use_giou=True),
        )
    ),
    test_cfg=dict(
        conf_threshold=0.42,  # curvelane branch
        nms_thres=15,
        nms_topk=16,
        # Lanes are decoded in crop-normalized y; CurvelanesMetric restores each
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
train_dataloader = dict(batch_size=24)  # per GPU
randomness = dict(seed=0, deterministic=True)
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=6e-4))
param_scheduler = [
    dict(type="CosineAnnealingLR", eta_min=0, begin=0, T_max=total_epochs, end=total_epochs,
         by_epoch=True, convert_to_iter_based=True),
]
