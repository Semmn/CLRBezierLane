# LLAMAS data settings for the official CLRerNet (mmdet 3.x).
# Geometry, lane count and splits follow CLRNet configs/clrnet/clr_resnet18_llamas.py:
#   ori 1276x717, cut_height=300, max_lanes=4, train on train, evaluate on valid
#   (test labels are not public; split="test" only writes submission files).
# Seg masks are drawn in memory exactly as CLRNet LLAMAS.load_annotations does
# (class i+1, thickness 15) instead of being written next to the labels.
# Augmentation: CLRerNet albumentations translation of the CLRNet list, LLAMAS crop.
dataset_type = "LlamasDataset"
data_root = "/exhdd/seungyu/dataset/LaneDataset/LLAMAS"
crop_bbox = [0, 300, 1276, 717]
img_scale = (800, 320)
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
compose_cfg = dict(bboxes=False, keypoints=True, masks=True)
max_lanes = 4


# data pipeline settings
train_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(
        type="Crop",
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    dict(type="HorizontalFlip", p=0.5),
    dict(type="ChannelShuffle", p=0.1),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=0.04,
        contrast_limit=0.15,
        p=0.6,
    ),
    dict(
        type="HueSaturationValue",
        hue_shift_limit=(-10, 10),
        sat_shift_limit=(-10, 10),
        val_shift_limit=(-10, 10),
        p=0.7,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="MotionBlur", blur_limit=5, p=1.0),
            dict(type="MedianBlur", blur_limit=5, p=1.0),
        ],
        p=0.2,
    ),
    dict(
        type="IAAAffine",
        scale=(0.8, 1.2),
        rotate=(-10.0, 10.0),  # this sometimes breaks lane sorting
        translate_percent=0.1,
        p=0.7,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(
        type="Crop",
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type="albumentation", pipelines=train_al_pipeline),
    #dict(type="Normalize", **img_norm_cfg),
    dict(
        type="PackCLRNetInputs",
        max_lanes=max_lanes,
        #keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "ori_shape",
            "img_shape",
            "gt_points",
            "gt_masks",
            "lanes",
        ],
    ),
]

val_pipeline = [
    dict(type="albumentation", pipelines=val_al_pipeline),
    #dict(type="Normalize", **img_norm_cfg),
    dict(
        type="PackCLRNetInputs",
        max_lanes=max_lanes,
        #keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
        ],
    ),
]



train_dataloader = dict(
    batch_size=64,  # CLRNet R18 LLAMAS
    num_workers=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="train",
        pipeline=train_pipeline,
        test_mode=False,
    ),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="val",
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="LLAMASMetric", data_root=data_root, split="val")
test_evaluator = val_evaluator
