# BezierFormer replica (YSY 2026-10-23): BezierFormer (arXiv 2404.16304) as its paper describes it, with every addition
# of ours removed at once, to tell whether its 89.06 comes from anything the paper describes
# (clrbezierlane/bf_replica/bf_replica_design.md). Pair: foveal_lane_off_bfstyle_o2o_flip_curvelanes_r34.py (flip).
# The data (2x pipeline with flip, pooled to 800 x 320), the schedule (24 epochs, 16 per update, AdamW 1e-4, backbone
# 0.1x, poly to 1e-5, clip 35, sync BN) and the evaluation (rows, extend_bottom, CurvelanesMetric 224 x 224, width 5)
# are flip's; only the network and its losses differ.
#   - R18, dim 128, 4 scales (C3-C5 + a stride-64 level), a 6-layer deformable encoder (Deformable DETR's defaults:
#     8 heads, 4 points, FFN 1024, dropout 0.1; the paper gives no encoder settings).
#   - 16 queries: E_0 learnable; C_0 = MLP(X) of the encoded scales, averaged (no heatmap proposals). Free 2D cubic
#     control points (no y order).
#   - 6 decoder layers: self-attention with the position encoding of 5 curve points, Bezier curve attention (5
#     reference points x 4 scales x 5 samples; offsets and weights from the feature at each reference point), FFN;
#     per-layer classification head and control-point offset FFN (C_i = C_{i-1} detached + dC_i).
#   - Loss on every layer including C_0: sigmoid focal; Chamfer IoU (200 points, e = 10 px) + length ratio + endpoint
#     L2; SimOTA top-1 on cost = loss. No proposal, segmentation or vanishing-point losses, no extra scoring
#     self-attention.
# In the log: loss_cls / loss_loc / loss_len / loss_end of the last layer, d0.* (C_0) to d5.* of the earlier ones,
# ciou (mean Chamfer IoU of the matched pairs), match_frac (GT lanes that got a query; SimOTA top-1 leaves a lane
# unmatched when its cheapest query went to another lane), pos_prob / neg_prob, num_gt, step_off_px.
# Eval-only: --cfg-options model.test_cfg.output=points gives BezierFormer's dense curve output (samples inside the
# image) instead of the 72 rows. Needs bf_replica.patch (after interval.patch; a new file, no other head changes).

_base_ = [
    "dataset_curvelanes_clrbezier_2x_flip.py",
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
cfg_name = "bezierformer_replica_flip_curvelanes_r18.py"

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

# 4 images per GPU per step, one step per update: 16 per update over 4 GPUs.
train_dataloader = dict(batch_size=4)
sync_bn = "torch"  # BN over the 4 GPUs' 16 images
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
