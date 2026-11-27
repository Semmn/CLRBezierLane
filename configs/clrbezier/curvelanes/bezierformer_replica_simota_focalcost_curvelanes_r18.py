# BezierFormer replica on the STANDARD CurveLanes pipeline (YSY 2026-11-27): bezierformer_replica_simota_focalcost_flip_
# curvelanes_r18.py with two changes only. (1) Data: dataset_curvelanes_clrbezier.py = CondLaneNet's / CLRerNet's
# CurveLanes pipeline as published (augmentation at 800 x 320, no horizontal flip) instead of the 2x + flip file
# inherited from the foveal network (augmentation at 1600 x 640, pooled 2 x 2 in BezierFormerNet; blur / JPEG ~half
# strength at the network's resolution). (2) 16 images on 1 GPU (GPU 3) instead of 4 x 4 GPUs.
# Pair for multi_anchor_former_curvelanes_r18.py. Notes of the flip config follow.
# BezierFormer replica, SimOTA with the paper's cost (YSY 2026-10-25): the replica (bezierformer_replica_flip_curvelanes_r18.py)
# with one change, loss_cfg.cls_cost = "focal": the matching cost's L_cls is the focal loss the query would take as
# this lane's positive, as the paper's cost = L_reg + L_cls. The first replica run used Deformable DETR's matching term
# (focal positive - focal negative) instead; with SimOTA top-1 that term makes a confident query up to ~1.4 cheaper
# for every lane, more than ~100 px of distance, so neighbour lanes lost their queries (match_frac 0.73 -> 0.39 over
# training, recall 24.6 at 0.5). SimOTA top-1 itself is kept, as the paper.
# Pair: bezierformer_replica_flip_curvelanes_r18.py; sibling: bezierformer_replica_hungarian_flip_curvelanes_r18.py.
# In the log: match_frac should stay near 1 (a lane can still lose its query when two lanes share their nearest one).
# Needs bf_replica.patch, bf_hungarian.patch, then bf_clscost.patch.

_base_ = [
    "dataset_curvelanes_clrbezier.py",
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
        "libs.clrbezier",
        "libs.datasets.foveation",    # the 2x pipeline's ToContextFrame
    ],
    allow_failed_imports=False,
)
cfg_name = "bezierformer_replica_simota_focalcost_curvelanes_r18.py"

img_w, img_h, num_points = 800, 320, 72
model = dict(
    type="BezierFormerNet",           # libs/clrbezier/bezierformer.py: pools the 2x input to 800 x 320, no neck
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type="mmdet.ResNet",
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),  # C3, C4, C5 -> [128, 256, 512] channels
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet18"),
    ),
    bbox_head=dict(
        type="BezierFormerHead",
        in_channels=[128, 256, 512],  # backbone C3-C5
        num_queries=16,               # N_query, CurveLanes (Table VI)
        dim=128,                      # dimension of E_i, R18 (Table VI)
        img_w=img_w,                  # input 800 x 320 (Table VI)
        img_h=img_h,
        num_points=num_points,        # rows of the official test layout (72)
        enc_cfg=dict(
            num_layers=6,             # not in the paper: Deformable DETR's (6)
            num_heads=8,              # (8)
            num_points=4,             # samples per head and scale (4)
            ffn_dim=1024,             # (1024)
            dropout=0.1,              # (0.1)
        ),
        dec_cfg=dict(
            num_layers=6,             # decoder layers (Table VI) (6)
            sa_heads=8,               # self-attention heads, not in the paper (8)
            ref_points=5,             # N_ref, at t = 0, 1/4, 1/2, 3/4, 1 (Table VI) (5)
            attn_heads=8,             # curve-attention heads, not in the paper (8)
            attn_samples=5,           # N_sample per reference point, head and scale (Table VI) (5)
            ffn_dim=1024,             # not in the paper (1024)
            dropout=0.1,              # not in the paper (0.1)
            init_hidden=512,          # width of C_0's MLP, not in the paper (512)
        ),
        loss_cfg=dict(
            n_dis=200,                # N_dis, points per curve in the Chamfer IoU (Table VI) (200)
            width_px=10.0,            # e, px of the 800 x 320 input (Table VI) (10)
            cls_weight=1.0,           # Eq. 10 sums the terms unweighted (1.0)
            loc_weight=1.0,           # (1.0)
            len_weight=1.0,           # (1.0)
            end_weight=1.0,           # (1.0)
            focal_alpha=0.25,         # (0.25)
            focal_gamma=2.0,          # (2.0)
            min_gt_len_px=8.0,        # shorter GT lanes (inside the image) are dropped, as our heads (8)
            assign="simota",          # SimOTA top-1, as the paper ("simota")
            cls_cost="focal",         # the cost's L_cls = focal positive term, as the paper ("focal_diff" = first run)
        ),
    ),
    test_cfg=dict(
        conf_threshold=0.50,
        use_nms=False,
        as_lanes=True,
        extend_bottom=True,           # as flip: the same test path
        nms_thres=15,
        nms_topk=16,                  # NMS-free: at most this many lanes per image
        ori_img_w=1,
        ori_img_h=1,
        cut_height=0,
        output="rows",                # "rows": the official 72-row layout (flip's path); "points": dense curve points
        out_points=100,               # curve samples for output "points" (100)
    ),
)

# flip's schedule (BezierFormer's Table VI): 24 epochs, AdamW 1e-4 (backbone 0.1x), poly decay, clip 35.
total_epochs = 24
checkpoint_config = dict(interval=total_epochs)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# 16 images on 1 GPU per step, one step per update: 16 per update, as the flip config's 4 x 4 GPUs. (Out of memory:
# batch_size 8 with accumulative_counts 2; BN then sees 8.)
train_dataloader = dict(batch_size=16)
sync_bn = "torch"  # unused on 1 GPU (mmengine reverts it to BN over the 16 images)
randomness = dict(seed=0, deterministic=True)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=1,
)
custom_hooks = [dict(type="LossSpikeHook", factor=2.0)]
log_processor = dict(
    type="LogProcessor", window_size=50, by_epoch=True,
    custom_cfg=[dict(data_src="grad_norm", method_name="max", window_size=50,
                     log_name="grad_norm_max")])
# Poly decay per iteration: lr = (base - 1e-5) * (1 - t / T) ** 2 + 1e-5, no warmup (eta_min absolute for every group).
param_scheduler = [
    dict(type="PolyLR", power=2.0, eta_min=1e-5, begin=0, end=total_epochs,
         by_epoch=True, convert_to_iter_based=True),
]
log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
