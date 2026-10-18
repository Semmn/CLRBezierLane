# HBLRNet run C: run B (hblrnet_a16_2d_hier_curvelanes_r34_bgrecipe.py) with one change, the schedule
# (hier_cfg.schedule "chain"): the levels alternate, lane layer, segment layer, lane layer, ... (3 + 3,
# the same 6 layers). The curve = the exact split of the lane cubic + a segment residual kept in the lane
# frame, so a lane layer moves the whole curve and the residual moves with it. The levels talk in a chain:
# before each lane layer (but the first) and before scoring the lane token reads its 3 segment tokens
# (low -> high, keys carry each segment's place and residual); before each segment layer each segment
# reads its lane token and its siblings (high -> low). Both start as the identity (zero-initialized).
# Needs blrnet_hier.patch, blrnet_hier_chain.patch (and blrnet_o2m_ensure_one.patch).
# In the log: seg_move_px (mean residual update per segment layer, px), seg_res_px (final residual, px).

_base_ = [
    "dataset_curvelanes_clrbezier.py",
    "../../_base_/default_runtime.py",
]

default_scope = "mmdet"

# Lists are replaced (not merged) by mmengine: the official import list plus libs.clrbezier.
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
cfg_name = "hblrnet_a16_2d_hier_curvelanes_r34_bgrecipe.py"

img_w, img_h, num_points = 800, 320, 72
model = dict(
    type="CLRerNet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
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
        type="HBLRHead",             # libs/clrbezier/hblrnet.py (blrnet_hier.patch)
        num_points=num_points,        # output rows (72)
        sample_points=36,             # feature samples per anchor along its curve (36)
        prior_feat_channels=64,       # neck channels (64)
        num_levels=3,                 # neck levels sampled, the last ones, fine -> coarse (3)
        img_w=img_w,
        img_h=img_h,
        seg_num_classes=2,            # CurveLanes masks: background / lane (head default 5)
        anchor_cfg=dict(
            num_anchors=16,           # learned anchors (48 in the stepblocks run); CLRerNet tiling at
                                      #   init: 2 left, 12 bottom, 2 right (192)
            dim=256,                  # anchor feature width (256)
            num_blocks=2,             # semantic/geometry interaction blocks, each one
                                      #   semantic: relation self-attn, cross-attn to image tokens,
                                      #             cross-attn to the anchor's own points, FFN
                                      #   geometry: relation self-attn, cross-attn to the semantic set, FFN
                                      # about 2.6 M parameters per block at dim 256 / ffn 1024 (2)
            num_heads=8,              # attention heads, must divide dim (8)
            num_neighbors=16,         # anchors attended to in the relation self-attention (16)
            ffn_dim=1024,             # FFN width in the blocks (1024)
            dropout=0.0,              # attention / FFN dropout (0.0)
            dist_scale=15.0,          # relation features in units of the CurveLanes NMS threshold, px (50)
            max_dist=10.0,            # relation distance features are capped at max_dist * dist_scale (10)
            point_dim=64,             # per-point width before the geometry vector (64)
        ),
        refine_cfg=dict(
            num_iters=3,              # lane-level refinement steps, deep supervision on each
            resample="fine",          # "fine": re-sample P3 along the current curve at every step;
                                        # "none": feature space only ("fine")
            step_scale=1.0,           # multiplies the predicted updates (1.0)
            point_offsets=False,      # no row residual (HBLRHead requires False)
            step_blocks=True,         # option B: a decoder block with its own weights inside every refinement
                                        # step: relation self-attention over the current lanes, curve attention
                                        # (deformable attention over the anchor's own curve at the current
                                        # geometry), FFN (needs blrnet_stepblocks_encoder.patch) (False)
            step_ffn_dim=1024,        # FFN width of those blocks (1024)
            deform_heads=4,           # heads of the curve attention, must divide the neck channels 64 (4)
            deform_points=4,          # learned offsets per curve point, head and level (4)
            deform_levels=3,          # the finest levels it samples, 3 = all (num_levels)
            deform_curve_points=36,   # reference points on the curve (sample_points)
        ),
        # 2D geometry (all defaults spelled out)
        curve_cfg=dict(
            gt_points=48,             # arc-length samples of each GT polyline (gt_points meta key)
            out_per_segment=24,       # output samples per third of the lane: 3 * 24 + 1 points (24)
            rel_points=12,            # points per lane in the 2D relation attention (12)
            cost_points=25,           # points per lane in the matching cost (25)
            end_loss_weight=0.05,     # px smooth-L1 on the start and end points, every step (0.05)
            min_gt_len_px=8.0,        # shorter GT lanes (inside the image) are dropped (8)
        ),
        hier_cfg=dict(
            segments=3,               # segments per lane, split at t = 1/3, 2/3 (3)
            seg_iters=3,              # segment layers; "chain" needs = refine_cfg.num_iters (3)
            seg_deform_points=12,     # curve-attention reference points per segment (12)
            seg_sample_points=12,     # points re-sampled per segment for the GRU input (12)
            seg_neighbors=16,         # segments attended to in the relation attention (16)
            seg_rel_points=8,         # points per segment in the relation features (8)
            schedule="chain",         # alternate lane / segment layers with the level exchange ("sequential")
        ),

        align_cfg=None,               # the alignment offsets are off: their anchor-GT assignment is row-based
        assign_cfg=dict(
            # one-to-one, on the last refinement step, geometry first: the score term is linear
            # and weighted 0.15 = 3 * 0.05, so it only reorders copies within 0.05 LaneIoU
            o2o=dict(type="HungarianLaneAssigner", cls_weight=1.0, cls_cost_mode="log",
                     point_weight=0.0, iou_weight=3.0),
            # one-to-many for the o2m head (feature learning, with-NMS comparison); None drops it
            o2m=dict(type="SimOTALaneAssigner", candidate_topk=3, min_dynamic_k=1, cls_weight=1.0,
                     cls_cost_mode="log", point_weight=0.0, iou_weight=3.0,
                     ensure_one=True),        # every lane keeps >= 1 o2m positive (False)
        ),
        loss_cfg=dict(
            cls_loss_weight=2.0,      # softmax focal, summed over anchors / GT lanes per image (2.0)
            focal_alpha=0.25,         # (0.25)
            focal_gamma=2.0,          # (2.0)
            iou_loss_weight=4.0,      # 1 - symmetric 2D LaneIoU (lane_iou_2d), every step (4.0)
            point_loss_weight=0.05,   # unused by HBLRHead
            cp_loss_weight=0.05,      # px smooth-L1 on the 4 control points against the arc-length LS fit
                                      # of the GT polyline, every lane step (0.05)
            extent_loss_weight=0.2,   # unused by HBLRHead (curve_cfg.end_loss_weight instead)
            smooth_l1_beta=1.0,       # (1.0)
            o2m_cls_weight=1.0,       # o2m focal loss, relative to cls_loss_weight (1.0)
            o2m_reg_weight=0.5,       # the o2m positives' geometry terms, relative to the o2o ones (0.5)
            # LaneIoU geometry of the official CurveLanes evaluation (224 wide, 124 high for the
            # 2560x1440 images); set it to the baseline run's value if that differs.
            iou_eval_shape=(124, 224),
            lane_width=2.5 / 224,     # 2D LaneIoU half-width in the 224-wide eval space: loss and SimOTA k
            lane_width_cost=10.0 / 224,  # 2D LaneIoU half-width of the matching cost
            seg_loss_weight=2.0,      # (1.0)
            seg_bg_weight=0.4,        # (0.4)
            seg_ignore_label=255,     # (255)
            cp_fit_ridge=1e-2,        # ridge of the GT control-point fit, toward the chord (1e-2)
        ),
        target_adapter=dict(
            lane_keys=["lanes"],
            seg_keys=["gt_masks"],
            start_y_convention="image_y",  # official PackCLRNetInputs: y0 = 1 - n_out / n_strips
            length_unit="count",
        ),
        # Transformer encoder over the FPN levels before the anchors read them
        # (blrnet_stepblocks_encoder.patch). None turns it off.
        encoder_cfg=dict(
            dim=128,                  # token width (128)
            num_layers=2,             # encoder layers (2)
            levels=3,                 # coarsest levels encoded, 3 = all 5250 tokens at 320x800 (num_levels)
            num_heads=4,              # (4)
            num_points=4,             # learned offsets per token, head and level (4)
            ffn_dim=512,              # (512)
            dropout=0.0,              # (0.0)
        ),
        test_score=None,              # None: o2o scores without NMS, o2m scores with NMS;
                                      # "o2o" | "o2m" forces one (None)
    ),
    test_cfg=dict(
        conf_threshold=0.50,
        use_nms=False,
        as_lanes=True,
        extend_bottom=True,
        nms_thres=15,
        nms_topk=16,                  # NMS-free: at most this many lanes per image
        # Lanes are decoded normalized within the crop; CurvelanesMetric maps them back with
        # each image's crop_offset.
        ori_img_w=1,
        ori_img_h=1,
        cut_height=0,
    ),
)

# Schedule: epochs and batch as before; optimizer and LR decay from BezierFormer's Table VI.
total_epochs = 36
checkpoint_config = dict(interval=total_epochs)
# Validation on valid/valid.txt every val_interval epochs (official CurveLanes: 3).
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Batch size per GPU. The point tokens [32, 192, 36, 256] are the largest tensors of the
# head; if memory is short, lower anchor_cfg.dim to 128 or sample_points to 24 first.
train_dataloader = dict(batch_size=32)
randomness = dict(seed=0, deterministic=True)
# Gradient clipping (global L2 norm). The CPU smoke had norms of 15-30 with this head;
# if clipping fires on most iterations, raise max_norm to 2-3x the typical value.
# AdamW, lr 3e-4, weight decay 1e-4; the backbone at 0.1 x = 3e-5. With paramwise_cfg every
# parameter gets its own group and the logged "lr" is the first group's, a backbone weight;
# "base_lr" is the head's.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=1e-4),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
    # clip_grad=dict(max_norm=35, norm_type=2),
)
# Records every iteration whose loss or grad_norm exceeds 2x its running median
# (with the batch's image names) in work_dir/loss_spikes_rank*.jsonl.
custom_hooks = [dict(type="LossSpikeHook", factor=2.0)]
# grad_norm is averaged over the 50-iteration log window; also log its maximum.
log_processor = dict(
    type="LogProcessor", window_size=50, by_epoch=True,
    custom_cfg=[dict(data_src="grad_norm", method_name="max", window_size=50,
                     log_name="grad_norm_max")])
# Poly decay per iteration: lr = (base - 1e-5) * (1 - t / T) ** 2 + 1e-5, no warmup; eta_min is
# absolute for every group. Head: 3e-4 -> 1e-5; backbone: 3e-5 -> 1e-5.
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
