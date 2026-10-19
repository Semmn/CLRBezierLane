# HBLRNet run A'': run A' (hblrnet_a16_2d_pat_curvelanes_r34_bgrecipe.py) with image-inferred anchors (YSY 2026-10-19
# 04:55Z; clrbezierlane/dynamic_anchor_design.md):
#   - no static anchors and no anchor embedding: a light conv head on P3 (all head levels upsampled and concatenated)
#     predicts per cell a lane-centre heatmap, a pose (bottom and top points) and a pattern (logits over the same
#     lane codebook + a residual);
#   - the 16 strongest 3 x 3 peaks are the queries; their decoded curves (detached) are the step-0 curves, their
#     logits the step-0 pattern logits; a query's content = its curve's encoding + its cell's feature + one shared
#     token. Everything after step 0 is A'.
#   - targets: each GT lane's centre is its arc-length midpoint; heatmap = Gaussian at its cell (CenterNet focal);
#     the cells within 2 of a centre get the pose, 2D LaneIoU and pattern losses of their decoded curve.
# BEFORE TRAINING, once (the same codebook as A'; skip if A' already built it):
#     python tools/clrbezier/build_pattern_codebook.py configs/clrbezier/curvelanes/hblrnet_a16_2d_pat_centre_curvelanes_r34_bgrecipe.py
# Needs blrnet_hier_centre.patch (after blrnet_hier_pattern_cover.patch).
# In the log: loss_centre_hm, loss_centre_end, loss_centre_iou, loss_centre_pat, loss_centre_pat_ce, centre_recall
# (GT lanes with a selected peak within 2 cells of their centre), cov0 / iou0 (GT lanes whose best step-0 curve has
# 2D LaneIoU >= 0.5, and the mean best IoU; A' logs them too with this patch).
# Reads against A': the same run but for where the step-0 curves and the query content come from.
# HBLRNet run A': run A-mono (hblrnet_a16_2d_mono_curvelanes_r34_bgrecipe.py) with lanes as pose x pattern (YSY chose 2026-10-18 17:34Z;
# clrbezierlane/pose_pattern_design.md):
#   - a lane cubic is its pose (bottom and top points, image axes, the top above the bottom by construction) and
#     its pattern (the two inner control points in the lane's own chord frame: shift, tilt and length removed);
#   - the pattern is a softmax mixture of 32 prototypes (k-means of training lanes) plus a residual; each lane step
#     moves the pose, adds to the prototype logits and to the residual;
#   - losses: the control-point loss is replaced by a pattern smooth-L1 (px) and a soft cross-entropy on the
#     prototype logits; LaneIoU, endpoint and the matching cost as in A-mono.
# BEFORE TRAINING, once (writes pattern_cfg.codebook and a report next to it):
#     python tools/clrbezier/build_pattern_codebook.py configs/clrbezier/curvelanes/hblrnet_a16_2d_pat_curvelanes_r34_bgrecipe.py
# Needs blrnet_hier_pattern.patch (after blrnet_hier_mono.patch).
# In the log: loss_pat, loss_pat_ce, pat_acc (o2o lanes whose top prototype is their GT's), pat_ent (entropy of
# the mixture weights).
# Reads against A-mono: the same run but for the representation and its two losses.
# HBLRNet run A-mono: run A-of (hblrnet_a16_2d_of_curvelanes_r34_bgrecipe.py) with the geometry YSY asked for (2026-10-18 16:55Z):
#   - curve_cfg.y_order="monotone": every curve is monotone in y by construction (each control point's y is the
#     one below minus a softplus increment, x free; no rows). The segments stay one monotone chain with shared
#     joints and G1 handles. Lanes run bottom -> top like the GT, so they cannot reverse or fold.
#   - curve_cfg.frame="image": the lane and segment levels update in the same global frame (image axes / the
#     monotone parameters), not the lane's own tangent / normal.
#   - curve_cfg.match_end_len=True with point_weight 3.0 (o2o and o2m): the matching cost adds the endpoint
#     distance and the length ratio, as in BezierFormer, at the weight of the LaneIoU term.
#   - kept from A-of: orientation_free (only matters for lanes straddling horizontal now) and clip_grad 35.
# Needs blrnet_hier_mono.patch (after blrnet_hier_orient.patch).
# HBLRNet run A-of: run A (hblrnet_a16_2d_curvelanes_r34_bgrecipe.py) with the fix for the loss episodes:
#   - curve_cfg.orientation_free=True: a lane has no direction. The endpoint and control-point losses take
#     the better of the two GT directions per pair (MapTR's permutation-equivalent target), the relation
#     features read each lane from its lower end, and the test output turns each lane bottom -> top before
#     the rows. Before, a near-horizontal lane's "start" flipped with a 2 degree slope or the rotation
#     augmentation, while the LaneIoU loss and the matching ignore direction.
#   - gradient clipping on (max_norm 35), as a safety net; grad_norm and grad_norm_max are logged.
# Needs blrnet_hier.patch, blrnet_hier_orient.patch (and blrnet_o2m_ensure_one.patch).
# In the log: end_rev_frac (share of o2o pairs scored against the reversed GT, last layer).
# Reads against run A: the loss should have no episodes; compare F1 on the swept final checkpoints.
# Run A's own notes follow.
# HBLRNet run A: 2D Bezier lanes, lane level only (hier_bezier_design.md). From the 16-anchor run
# (blrnet_stepblocks_a16_curvelanes_r34_bgrecipe.py), with no row representation in training:
#   - head HBLRHead: 4 free 2D control points per lane, updated in the lane's tangent/normal frame;
#     symmetric 2D LaneIoU for the loss and the matching cost; 2D endpoint loss instead of start/length
#     in rows; 2D relation features; GT = the augmented annotation polylines (gt_points);
#   - no row residual (point_offsets False);
#   - 6 lane steps (3 before), the same step count as run B;
#   - alignment offsets off (their assignment is row-based).
# Optimizer, schedule, batch, anchors, o2m (ensure_one), encoder, curve attention: as a16.
# Needs blrnet_hier.patch (and blrnet_o2m_ensure_one.patch, like a16).
# Reads: A vs a16 = rows -> 2D (with 6 steps, no residual, no alignment); B vs A = the hierarchy.

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
cfg_name = "hblrnet_a16_2d_pat_centre_curvelanes_r34_bgrecipe.py"

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
        type="HBLRHead",             # libs/clrbezier/hblrnet.py (blrnet_hier.patch ... blrnet_hier_pattern.patch)
        num_points=num_points,        # output rows (72)
        sample_points=36,             # feature samples per anchor along its curve (36)
        prior_feat_channels=64,       # neck channels (64)
        num_levels=3,                 # neck levels sampled, the last ones, fine -> coarse (3)
        img_w=img_w,
        img_h=img_h,
        seg_num_classes=2,            # CurveLanes masks: background / lane (head default 5)
        anchor_cfg=dict(
            num_anchors=16,           # queries: the 16 strongest centre peaks (centre_cfg); A': learned anchors, CLRerNet tiling at
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
            num_iters=6,              # lane-level refinement steps, deep supervision on each
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
            orientation_free=True,    # end / cp losses against the better GT direction; output bottom -> top (False)
            frame="image",            # updates in the image axes at both levels ("lane": tangent / normal)
            y_order="monotone",       # control points monotone in y by construction (None, "penalty")
            match_end_len=True,       # matching point term = endpoint distance + length ratio (False)
        ),
        hier_cfg=None,                # run A: lane level only

        # lanes as pose x pattern (YSY, 2026-10-18; clrbezierlane/pose_pattern_design.md); None turns it off
        pattern_cfg=dict(
            codebook="work_dirs/pattern_codebook/curvelanes_k32.pt",  # from tools/clrbezier/build_pattern_codebook.py
                                      #   (run it once before training; the checkpoint keeps the prototypes)
            lane_protos=32,           # lane prototypes in the codebook (32)
            seg_protos=32,            # segment prototypes in the codebook (32)
            ce_weight=0.5,            # soft cross-entropy of the prototype logits, every output (0.5)
            l1_weight=0.05,           # px smooth-L1 on the pattern (difference x GT chord length), every output;
                                      #   replaces the control-point loss (0.05, as cp_loss_weight)
            seg_target_points=16,     # GT points re-sampled under each segment for its target (16)
            min_seg_len_px=4.0,       # GT pieces shorter than this get no segment pattern loss (4)
            tau_scale=1.0,            # multiplies the codebook's soft-target width (1.0)
        ),
        centre_cfg=dict(              # image-inferred anchors (lane-centre proposals); None = A'
            channels=64,              # conv width of the proposal head (64)
            sigma_cells=2.0,          # heatmap Gaussian around each centre cell, in P3 cells (2.0)
            radius_cells=2,           # cells within this of a centre get the pose / IoU / pattern losses (2)
            scale_px=100.0,           # pose offsets per unit of the head's output, px (100)
            hm_weight=1.0,            # CenterNet focal loss of the heatmap (1.0)
            iou_weight=1.0,           # 2D LaneIoU of each regression cell's decoded curve (1.0)
            end_weight=0.05,          # px smooth-L1 of its bottom and top points (0.05, as end_loss_weight)
        ),
        align_cfg=None,               # the alignment offsets are off: their anchor-GT assignment is row-based
        assign_cfg=dict(
            # one-to-one, on the last refinement step, geometry first: the score term is linear
            # and weighted 0.15 = 3 * 0.05, so it only reorders copies within 0.05 LaneIoU
            o2o=dict(type="HungarianLaneAssigner", cls_weight=1.0, cls_cost_mode="log",
                     point_weight=3.0, iou_weight=3.0),   # point term: endpoint + length (match_end_len)
            # one-to-many for the o2m head (feature learning, with-NMS comparison); None drops it
            o2m=dict(type="SimOTALaneAssigner", candidate_topk=3, min_dynamic_k=1, cls_weight=1.0,
                     cls_cost_mode="log", point_weight=3.0, iou_weight=3.0,
                     ensure_one=True),        # every lane keeps >= 1 o2m positive (False)
        ),
        loss_cfg=dict(
            cls_loss_weight=2.0,      # softmax focal, summed over anchors / GT lanes per image (2.0)
            focal_alpha=0.25,         # (0.25)
            focal_gamma=2.0,          # (2.0)
            iou_loss_weight=4.0,      # 1 - symmetric 2D LaneIoU (lane_iou_2d), every step (4.0)
            point_loss_weight=0.05,   # unused by HBLRHead
            cp_loss_weight=0.05,      # unused with pattern_cfg (pattern_cfg.l1_weight instead)
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
    clip_grad=dict(max_norm=35, norm_type=2),
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
