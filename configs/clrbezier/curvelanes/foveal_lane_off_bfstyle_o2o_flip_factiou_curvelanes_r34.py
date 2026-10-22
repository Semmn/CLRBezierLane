# Idea B, extent and lateral position judged apart (clrbezierlane/extent_shape/extent_shape_design.md):
# foveal_lane_off_bfstyle_o2o_flip_curvelanes_r34.py with curve_cfg.factorized_iou. Matching cost of every step:
# 1 - IoU_lat x IoU_ext at the metric's width (lane_width_cost 2.5 / 224), where IoU_lat is the 2D LaneIoU over the
# part both lanes share and IoU_ext the 1D IoU of their extents along the GT lane; the endpoint / length term is off
# (point_weight 0). Loss of every step's pairs: (1 - IoU_lat) + (1 - IoU_ext) in place of 1 - LaneIoU, and no
# endpoint smooth-L1 (end_loss_weight 0; the extent term moves the ends along the lane). Why: the current cost prefers
# the metric's miss in 14-16% of hit / miss pairs of copies (a cut far end vs a sideways shift); this cost in 0-0.6%
# (extent_shape/cost_vs_metric_output.txt). Logs o2o_lat_iou / o2o_ext_iou. Pair:
# foveal_lane_off_bfstyle_o2o_flip_curvelanes_r34.py (YSY 2026-10-22: replaces the costw + flip run, whose one change,
# lane_width_cost 2.5 / 224, this config also makes). Needs factorized_iou.patch (after point_cond.patch; hblrnet.py
# only). Validation, the network and its compute are the pair's. Notes of the pair follow.
# Horizontal flip (BezierFormer's Table VI, CurveLanes): foveal_lane_off_bfstyle_o2o_curvelanes_r34.py with the data
# file dataset_curvelanes_clrbezier_2x_flip.py, which adds HorizontalFlip(p = 0.5) to the training augmentation.
# The CurveLanes augmentation we inherited from CondLaneNet / the CLRerNet curvelane branch has no flip. Pair:
# foveal_lane_off_bfstyle_o2o_curvelanes_r34.py (the difference is the training pipeline only; validation and the
# network are the same). No code change: the patches of the pair (up to o2o_only.patch). Notes of the pair follow.
# o2o only (YSY 2026-10-21): foveal_lane_off_bfstyle_curvelanes_r34.py with assign_cfg.o2m None. The 16 queries
# get only the Hungarian (one-to-one) labels at every decoder step, so no query is pushed towards "positive" as a
# 2nd / 3rd copy of a lane while the o2o loss pushes it towards "negative". Gone: SimOTA labels, cls_o2m, the o2m
# geometry terms (0.5), and the o2m score in the scoring self-attention's keys. The proposal heatmap loss on the
# encoder map stays (dense, several cells per lane: a one-to-many signal on the encoder, not on the queries).
# This is BezierFormer's / Deformable DETR's supervision. Pair: foveal_lane_off_bfstyle_curvelanes_r34.py.
# Needs o2o_only.patch (after polar_feat.patch; without it the unused score projection fails DDP).
# off_bfstyle notes follow.
# FovealLaneNet off with BezierFormer's decoder layout, on BezierFormer's schedule (foveal_lane_off_bfsched): the
# pair for this file is foveal_lane_off_bfsched_curvelanes_r34.py, and the difference is the decoder only.
#   - 6 lane steps, no segment steps (hier_cfg None): the hierarchy gave no gain.
#   - Every step is supervised as a Deformable DETR decoder layer (dec_cfg.layer_loss): each step scores its own
#     queries with the shared o2o / o2m heads, is matched on its own curves and scores, and gets the focal and
#     LaneIoU / endpoint losses (logged d0.loss_* ... d4.loss_*; the last step keeps the plain names).
#   - 5 reference points per lane instead of 36 (t = 0, 1/4, 1/2, 3/4, 1), 8 samples per point, head and map
#     instead of 4 (BezierFormer: 5 x 5).
#   - 4 scales: a stride-64 level from C5 (3 x 3 stride-2 conv + GroupNorm, Deformable DETR's extra level), 65 more
#     tokens; the encoder stays at 3 layers.
# Needs bfstyle_decoder.patch (after polar_view.patch). Notes of foveal_lane_off_bfsched follow.
# FovealLaneNet off (run 1) on BezierFormer's training schedule: 24 epochs at 16 images per update (their batch 16),
# about 150k updates on CurveLanes train instead of run 1's ~28k (36 epochs at 128 per update). Everything else is
# foveal_lane_off_curvelanes_r34.py, so this file vs run 1 isolates the schedule.
#   - 4 images per GPU x 4 GPUs, no accumulation; BN synced over the 4 GPUs (sync_bn) so it still sees 16 images.
#   - lr 1e-4, BezierFormer's own (its Table VI, as in the bgrecipe configs; run 1 uses 3e-4 at 128), backbone 0.1x,
#     poly decay to 1e-5, clip 35, no warmup, as run 1.
#   - Not changed: matching and classification stay on the last decoder step (BezierFormer matches every layer).
# Run 1's notes follow.
# FovealLaneNet, run 1 of the foveal wave: foveal_lane_window_curvelanes_r34.py (run 2) with fovea_cfg.mode "off".
# The same network on the context view C alone: one backbone pass, an encoder over C's 5250 tokens, proposals,
# decoder and losses unchanged (no window, no F). Run 1 vs run 2 is the clean pair for the fovea; run 1 vs A-mono
# answers the redesign (dynamic anchors, full self-attention, no GRU / FPN / anchor embedding, no CP loss) together.
# The vanishing-column head and its loss stay on (as run 2), so the only difference is the second view.
# Run 2's notes follow.
# FovealLaneNet, run 2 of the foveal wave (the method): the two-view network of clrbezierlane/foveal_lane_network_design.md.
#   - The pipeline augments at 2x (dataset_curvelanes_clrbezier_2x.py); the network pools that image to the 800 x 320
#     context view C and crops the fovea F from it: a 400 x 80 window of the context frame (rows 0-80, columns centred
#     on the predicted vanishing column) at 2x, no resampling. GT, predictions, losses and the metric stay in C's frame.
#   - One R34 on both views (F normalized with the running BatchNorm statistics, which only C updates); no FPN; a
#     3-layer cross-view deformable encoder over C's and F's tokens (F masked outside the window).
#   - Dynamic anchors: a centre heatmap + ends + inner control points per view on the stride-8 encoder map; lanes
#     centred in the window are F's targets; the top 16 peaks of both views are the queries (their own encoder token +
#     the encoding of their curve + one shared token). No static anchors, no anchor embedding.
#   - Decoder: 3 lane steps + 3 segment steps (hier, sequential, as B-mono), each full self-attention with a curve
#     position encoding, dual-view curve attention at the current curve (both views, one softmax), FFN, monotone step.
#     No GRU. Scoring: o2m per query, o2o after one more self-attention whose keys carry the o2m score.
#   - Losses: A-mono's matching and per-step LaneIoU + endpoint losses, no control-point loss; proposal heatmap +
#     endpoints + LaneIoU; vanishing column; seg auxiliary on C.
# Pair: run 1 (foveal_lane_off_curvelanes_r34.py) is this file with fovea_cfg.mode "off"; run 3
# (foveal_lane_full_curvelanes_r34.py) uses the whole crop as the window (the upper bound).
# Batch: 16 per GPU x 2 accumulated steps = A-mono's 32 per GPU per update (the encoder's activations at 32 per GPU
# are about 45 GiB; at 16 about 22 GiB, measured on CPU by foveal_lane_tests/foveal_memory.py), lr and schedule as
# A-mono. Needs lane_foveation.patch and foveal_lane.patch.
# In the log: vp_err_px, vp_gt_frac, fovea_share (F's share of the last lane step's curve attention), prop_recall,
# prop_recall_fov (lanes centred in the window), prop_fovea_frac, win_lane_frac, cov0 / iou0 (step-0 coverage of
# the proposals), and HBLRHead's usual fields.

_base_ = [
    "dataset_curvelanes_clrbezier_2x_flip.py",
    "../../_base_/default_runtime.py",
]

default_scope = "mmdet"

# Lists are replaced (not merged) by mmengine: the official import list plus libs.clrbezier and the foveation module.
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
        "libs.clrbezier",
        "libs.datasets.foveation",
    ],
    allow_failed_imports=False,
)
cfg_name = "foveal_lane_off_bfstyle_o2o_flip_factiou_curvelanes_r34.py"

img_w, img_h, num_points = 800, 320, 72
model = dict(
    type="FovealLaneNet",            # libs/clrbezier/foveal_lane.py: two views of the 2x input, no neck
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
    bbox_head=dict(
        type="FovealLaneHead",
        in_channels=[128, 256, 512],  # backbone C3-C5
        num_queries=16,               # lane queries per image, from the proposal peaks (16)
        dim=256,                      # token / query width, a multiple of 32 (256)
        num_points=num_points,        # output rows of the official layout (72)
        img_w=img_w,                  # the context frame (800 x 320)
        img_h=img_h,
        seg_num_classes=2,            # CurveLanes masks: background / lane
        fovea_cfg=dict(
            mode="off",               # "window" (the method) | "off" (C only) | "full" (window = the whole crop)
            window_w=400,             # window in context px; F is 2x: 800 x 160 (2x each a multiple of 32) (400)
            window_h=80,              # (80)
            window_top=0,             # first window row: the top quarter of the crop, the far band (0)
            p_gt=0.5,                 # training: share of windows placed on the GT vanishing column (else the
                                      # detached prediction; test: the prediction) (0.5)
            jitter_px=40.0,           # uniform jitter of that GT column, context px (40)
            bn="context",             # F normalized with the running BatchNorm statistics (only C updates them), so
                                      # training and test treat F alike; "batch": training-mode BN on F too
        ),
        vp_cfg=dict(
            channels=128,             # conv width of the vanishing-column head on C5 (128)
            loss_weight=1.0,          # smooth-L1 on the column error / (img_w / 2), images with a GT point (1.0)
            beta=0.05,                # (0.05)
        ),
        enc_cfg=dict(
            num_layers=3,             # cross-view deformable encoder layers (3)
            num_heads=8,              # (8)
            num_points=4,             # samples per token, head and map (one map per level and view) (4)
            ffn_dim=1024,             # (1024)
            dropout=0.0,              # (0.0)
            checkpoint=False,         # recompute each layer in the backward pass (memory for time) (False)
            extra_level=True,         # a fourth level, stride 64 from C5 (3 x 3 stride-2 conv + GN) (False)
        ),
        prop_cfg=dict(
            channels=64,              # conv width of the proposal heads (64)
            sigma_cells=2.0,          # heatmap Gaussian, in cells of the view (2.0)
            radius_cells=2,           # cells around each GT centre that regress its lane (2)
            scale_px=100.0,           # end offsets per unit, context px (F's at half: its own px) (100)
            hm_weight=1.0,            # CenterNet focal loss (1.0)
            end_weight=0.05,          # px smooth-L1 of the bottom and top ends (0.05)
            iou_weight=1.0,           # 1 - 2D LaneIoU of the decoded proposal (1.0)
        ),
        dec_cfg=dict(
            num_iters=6,              # lane steps (6 without hier)
            self_attn="full",         # self-attention over the queries with a curve position encoding; "relation":
                                      # HBLRHead's relation attention (num_neighbors, dist_scale, max_dist)
            num_heads=8,              # (8)
            ffn_dim=1024,             # (1024)
            dropout=0.0,              # (0.0)
            curve_points=5,           # curve-attention reference points per lane, t = 0 .. 1 (36)
            deform_heads=8,           # curve-attention heads, must divide dim (8)
            deform_points=8,          # samples per reference point, head and map (4)
            step_scale=1.0,           # multiplies the predicted updates (1.0)
            cp_loss=False,            # control-point loss on the lane steps (loss_cfg.cp_loss_weight) (False)
            num_neighbors=16,         # relation attention only (16)
            dist_scale=15.0,          # relation attention only (15)
            max_dist=10.0,            # relation attention only (10)
            layer_loss=True,          # every step scored, matched and supervised (Deformable DETR) (False)
        ),
        # 2D geometry, as A-mono
        curve_cfg=dict(
            gt_points=48,             # arc-length samples of each GT polyline (48)
            out_per_segment=24,       # output samples per third of the lane: 3 * 24 + 1 points (24)
            rel_points=12,            # points per lane in the self-attention position encoding (12)
            cost_points=25,           # points per lane in the matching cost (25)
            end_loss_weight=0.0,      # B: no endpoint smooth-L1, the extent IoU moves the ends (pair: 0.05)
            min_gt_len_px=8.0,        # shorter GT lanes (inside the image) are dropped (8)
            orientation_free=True,    # end losses against the better GT direction; output bottom -> top
            frame="image",            # updates in the image axes (required)
            y_order="monotone",       # control points monotone in y by construction (required)
            match_end_len=True,       # matching point term = endpoint distance + length ratio (off: point_weight 0)
            factorized_iou=True,      # B: cost 1 - IoU_lat x IoU_ext, loss (1 - IoU_lat) + w (1 - IoU_ext) (False)
            extent_iou_weight=1.0,    # w of the extent term in the loss (1.0)
        ),
        hier_cfg=None,               # no segment steps
        assign_cfg=dict(
            o2o=dict(type="HungarianLaneAssigner", cls_weight=1.0, cls_cost_mode="log",
                     point_weight=0.0, iou_weight=3.0),   # B: no endpoint + length term (pair: 3.0)
            o2m=None,                 # o2o only: no SimOTA labels, no cls_o2m, no o2m geometry terms
        ),
        loss_cfg=dict(
            cls_loss_weight=2.0,      # softmax focal, summed over queries / GT lanes per image (2.0)
            focal_alpha=0.25,         # (0.25)
            focal_gamma=2.0,          # (2.0)
            iou_loss_weight=4.0,      # 1 - symmetric 2D LaneIoU, each step on its own matches (4.0)
            point_loss_weight=0.05,   # unused
            cp_loss_weight=0.05,      # only with dec_cfg.cp_loss
            extent_loss_weight=0.2,   # unused (curve_cfg.end_loss_weight instead)
            smooth_l1_beta=1.0,       # (1.0)
            o2m_cls_weight=1.0,       # unused (assign_cfg.o2m None)
            o2m_reg_weight=0.5,       # unused (assign_cfg.o2m None)
            iou_eval_shape=(124, 224),  # LaneIoU geometry of the official CurveLanes evaluation
            lane_width=2.5 / 224,     # 2D LaneIoU half-width in the 224-wide eval space: loss and SimOTA k
            lane_width_cost=2.5 / 224,   # B: the cost at the metric's width (pair: 10 / 224)
            seg_loss_weight=2.0,      # seg auxiliary on C's stride-8 encoder map
            seg_bg_weight=0.4,        # (0.4)
            seg_ignore_label=255,     # (255)
            cp_fit_ridge=1e-2,        # ridge of the GT control-point fit (cp_loss only)
        ),
        target_adapter=dict(
            lane_keys=["lanes"],
            seg_keys=["gt_masks"],
            start_y_convention="image_y",
            length_unit="count",
        ),
        test_score=None,              # None: o2o scores without NMS
    ),
    test_cfg=dict(
        conf_threshold=0.50,
        use_nms=False,
        as_lanes=True,
        extend_bottom=True,
        nms_thres=15,
        nms_topk=16,                  # NMS-free: at most this many lanes per image
        # Lanes are decoded normalized within the crop (the context frame); CurvelanesMetric maps them back with
        # each image's crop_offset.
        ori_img_w=1,
        ori_img_h=1,
        cut_height=0,
    ),
)

# BezierFormer's schedule: 24 epochs, AdamW 1e-4 (backbone 0.1x), poly decay, clip 35.
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
# Poly decay per iteration, as A-mono: lr = (base - 1e-5) * (1 - t / T) ** 2 + 1e-5, no warmup (eta_min absolute for
# every group).
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
