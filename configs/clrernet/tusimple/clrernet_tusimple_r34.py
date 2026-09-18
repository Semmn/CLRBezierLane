# CLRerNet-R34 on TuSimple. Model = configs/clrernet/base_clrernet_r34.py;
# dataset-dependent settings from CLRNet configs/clrnet/clr_resnet34_tusimple.py.
_base_ = [
    "../base_clrernet_r34.py",
    "dataset_tusimple_clrernet.py",
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
cfg_name = "clrernet_tusimple_r34.py"

model = dict(
    bbox_head=dict(
        loss_seg=dict(num_classes=7),  # CLRNet TuSimple: 6 lanes + background
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
total_epochs = 70
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
