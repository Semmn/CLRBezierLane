# CLRerNet-R34 on LLAMAS. Model = configs/clrernet/base_clrernet_r34.py;
# dataset-dependent settings from CLRNet configs/clrnet/clr_resnet18_llamas.py
# (CLRNet has no R34 LLAMAS config; R18 is the closest ResNet setting).
_base_ = [
    "../base_clrernet_r34.py",
    "dataset_llamas_clrernet.py",
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
cfg_name = "clrernet_llamas_r34.py"

model = dict(
    bbox_head=dict(
        loss_seg=dict(num_classes=5),  # CLRNet LLAMAS: 4 lanes + background
    ),

    test_cfg=dict(
        conf_threshold=0.45,  # CLRNet LLAMAS R18 (select by cross-validation for reporting)
        nms_thres=60,         # CLRNet LLAMAS R18 (DLA34 uses 50)
        nms_topk=4,
        ori_img_w=1276,
        ori_img_h=717,
        cut_height=300,
    ),
)

# schedule: CLRNet R18 LLAMAS (20 epochs, batch 64, AdamW 0.6e-3, cosine)
total_epochs = 20
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
train_dataloader = dict(batch_size=64)  # per GPU
randomness = dict(seed=0, deterministic=True)
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=0.6e-3))
param_scheduler = [
    dict(type="CosineAnnealingLR", eta_min=0, begin=0, T_max=total_epochs, end=total_epochs,
         by_epoch=True, convert_to_iter_based=True),
]
