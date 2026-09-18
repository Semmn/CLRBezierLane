# CurveLanes data settings for the official CLRerNet (mmdet 3.x).
# Translated from the CLRerNet features/curvelane branch
# (configs/clrernet/curvelanes/dataset_curvelanes_clrernet.py, mmdet 2.x):
#   augmentation list verbatim, cut_unsorted for training, max_lanes=16,
#   train on train/train_seg.txt (masks from tools/clrbezier/make_curvelanes_seg.py),
#   evaluate on valid/valid.txt with vega LaneMetricCore (224x224, width 5).
# Each image is cropped inside CurvelanesDataset by its resolution; the crop is
# carried in metainfo (crop_offset, crop_shape) and undone by CurvelanesMetric.
dataset_type = "CurvelanesDataset"
data_root = "/exhdd/seungyu/dataset/LaneDataset/CurveLanes"
img_scale = (800, 320)
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
compose_cfg = dict(bboxes=False, keypoints=True, masks=True)
max_lanes = 16

# data pipeline settings (verbatim from the curvelane branch)
train_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RGBShift", r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
            dict(
                type="HueSaturationValue",
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0,
            ),
        ],
        p=0.7,
    ),
    dict(type="JpegCompression", quality_lower=85, quality_upper=95, p=0.2),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=1.0),
            dict(type="MedianBlur", blur_limit=3, p=1.0),
        ],
        p=0.2,
    ),
    dict(type="RandomBrightness", limit=0.2, p=0.6),
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6,
    ),
    dict(
        type="RandomResizedCrop",
        height=img_scale[1],
        width=img_scale[0],
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type="LaneAlbumentation", pipelines=train_al_pipeline, cut_unsorted=True),
    dict(
        type="PackCLRNetInputs",
        max_lanes=max_lanes,
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "eval_shape",
            "img_shape",
            "gt_points",
            "gt_masks",
            "lanes",
        ],
    ),
]

val_pipeline = [
    dict(type="albumentation", pipelines=val_al_pipeline),
    dict(
        type="PackCLRNetInputs",
        max_lanes=max_lanes,
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "crop_shape",
            "crop_offset",
        ],
    ),
]

train_dataloader = dict(
    batch_size=24,  # curvelane branch: samples_per_gpu=24 (single GPU)
    num_workers=8,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + "/train/",
        data_list=data_root + "/train/train_seg.txt",
        diff_thr=0,
        pipeline=train_pipeline,
        test_mode=False,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root + "/valid/",
        data_list=data_root + "/valid/valid.txt",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="CurvelanesMetric", eval_width=224, eval_height=224,
                     iou_thresh=0.5, lane_width=5)
test_evaluator = val_evaluator
