# BLRNet on CurveLanes with the strip refiner (render-and-compare in curve-aligned strips;
# blrnet_strip.patch, after blrnet_o2m_branch.patch; design and checks in
# raster_strip/raster_strip_design.md). Standalone: nothing is inherited except the dataset
# file and the default runtime, and every setting the head reads is written out below, with
# the head's default in the trailing comment.
#
# The control is YSY's 48-anchor run: blrnet_o2m_branch_curvelanes_r34.py with
# o2m_branch_cfg=None (o2m labels on the 48 main anchors). Differences from it:
#   backbone.out_indices    (1, 2, 3) -> (0, 1, 2, 3): C2 too (stride 4, 64 channels)
#   neck.in_channels        [128, 256, 512] -> [64, 128, 256, 512], num_outs 3 -> 4: adds P2;
#                           P3-P5 are computed exactly as before and the head's num_levels=3
#                           still takes them (the last three) for everything else
#   strip_cfg               dict(...) below: after the last GRU step, every candidate's output
#                           points move along their normals by a per-row correction read from a
#                           strip of P2 + P3 (2 px lateral bins, +-16 px) that also holds the
#                           rasters of its own curve and of its 4 nearest candidates; the
#                           strip's quality vector reaches the o2o head
#   o2m_branch_cfg          None (the control's setting)
#
# The head is BLRHead (libs/clrbezier/blrnet.py), not CLRBezierHead, so the cascade's
# keys do not exist here. Where a cascade setting has a counterpart:
#   main_assigner                 -> assign_cfg.o2o   (Hungarian, on the last refinement step)
#   aux_cfg assigners (o2m)       -> assign_cfg.o2m   (SimOTA; None drops the o2m head)
#   aux_cls_head                  -> the o2m head exists whenever assign_cfg.o2m is set
#   num_priors / prior_cfg        -> anchor_cfg.num_anchors (learned anchors, CLRerNet tiling at init)
#   refine_layers (3 stages)      -> refine_cfg.num_iters (GRU steps) and anchor_cfg.num_blocks
#                                    (semantic/geometry interaction blocks, run once before the loop)
#   roi_gather_cfg / sample_points-> sample_points + align_cfg.offsets (one sampling, all levels)
#   brr_cfg / cp_frame            -> none: control points are free 2D (x free, y ordered)
#   loss_cfg.brr_cp_loss_weight   -> loss_cfg.cp_loss_weight
#   start / length loss           -> loss_cfg.extent_loss_weight (rows, bottom and top of the curve)
#   residual (row offsets)        -> refine_cfg.point_offsets (lateral, bounded, with its own loss)
#   use_focal / cls_bg_weight     -> focal always (focal_alpha, focal_gamma); no bg weight
#   dedup_cfg, decoder_cfg, decider_cfg, query_attn_cfg, diffusion_cfg, rank_loss_cfg -> none
# An unknown key inside anchor_cfg / refine_cfg / align_cfg / loss_cfg raises at build
# time; an unknown top-level head argument is printed as ignored, so check the first
# log lines after changing a key.
#
# Watch in the log:
#   loss_strip                   lateral CE + L1 + visibility on the matched candidates' strips; it does
#                                not go to 0: about 2.6 is its floor (the entropy of the soft per-row
#                                target, summed over the 2 passes with weights 0.8 and 1.0)
#   loss_strip_iou / _point      LaneIoU and point loss of the corrected output (one step's weight)
#   strip_corr_px                mean |correction| over valid strip rows (bound 16 px; at init 0)
#   strip_q_w                    mean |weight| of the quality projection into the o2o head (0 at init;
#                                stays near 0 if the o2o head does not use the strip)
#   num_pos_o2o / num_pos_o2m    positives per batch (o2o = number of GT lanes)
#   o2o_pos_prob / o2o_neg_prob  mean o2o score of positives / negatives
#   self_w_0, self_w_1, self_w_final  weight an anchor keeps on itself in the relation attention
#   offset_px, point_offset_px   mean |x| of the sampling offsets / the GRU's output point offsets
#
# Evaluate NMS-free (this config) and with NMS (--cfg-options model.test_cfg.use_nms=True);
# sweep the threshold from 0.05 for both. Compare with the 48-anchor control (F1@50 and
# F1@75), then run curvelanes_stage_trace.py / curvelanes_score_diag.py on both checkpoints
# (score vs on-lane Spearman, dismissed lanes outscored by a worse copy).
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
cfg_name = "blrnet_strip_curvelanes_r34.py"

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
        out_indices=(0, 1, 2, 3),  # C2-C5 -> [64, 128, 256, 512] channels (C2 for the strip)
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet34"),
    ),
    neck=dict(
        type="CLRerNetFPN",
        in_channels=[64, 128, 256, 512],  # ResNet-34 C2-C5
        out_channels=64,
        num_outs=4,                   # P2-P5; the head reads P3-P5 (num_levels=3) and the strip P2 + P3
    ),
    bbox_head=dict(
        type="BLRHead",
        num_points=num_points,        # output rows (72)
        sample_points=36,             # feature samples per anchor along its curve (36)
        prior_feat_channels=64,       # neck channels (64)
        num_levels=3,                 # neck levels sampled, the last ones, fine -> coarse (3)
        img_w=img_w,
        img_h=img_h,
        seg_num_classes=2,            # CurveLanes masks: background / lane (head default 5)
        anchor_cfg=dict(
            num_anchors=48,           # anchors; CLRerNet tiling at init (192)
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
            num_iters=3,              # GRU refinement steps, deep supervision on each (3)
            resample="fine",          # "fine": re-sample P3 along the current curve at every step;
                                      # "none": feature space only ("fine")
            step_scale=1.0,           # multiplies the predicted updates (1.0)
            point_offsets=True,       # per-point lateral offsets of the output curve, the residual
                                      # branch, with its own point loss (True)
            max_point_offset_px=40.0, # tanh bound of those offsets, px (40)
        ),
        # Feature alignment (design item 3). None turns it off; then the sampling has no
        # learned offsets and no alignment losses.
        align_cfg=dict(
            offsets=True,             # learned per-point sampling offsets (True)
            max_offset_px=24.0,       # their tanh bound, px (24)
            topk=3,                   # anchors assigned per GT before any prediction, nearest in
                                      # mean |dx| over the rows they cover (3)
            max_px=50.0,              # farther anchors are not assigned (50)
            min_cover=0.25,           # an anchor must cover this fraction of the GT's rows (0.25)
            target="both",            # "offset": sampling offsets -> lateral distance to the GT;
                                      # "feature": sampled features -> GT-sampled features, stop-grad;
                                      # "both" ("both")
            feature_loss="cos",       # "cos" | "l2" | "infonce" ("cos")
            tau=0.1,                  # InfoNCE temperature (0.1)
            offset_weight=0.05,       # px smooth-L1 (0.05)
            feature_weight=1.0,       # (1.0)
        ),
        assign_cfg=dict(
            # one-to-one, on the last refinement step, geometry first: the score term is linear
            # and weighted 0.15 = 3 * 0.05, so it only reorders copies within 0.05 LaneIoU
            o2o=dict(type="HungarianLaneAssigner", cls_weight=0.15, cls_cost_mode="linear",
                     point_weight=0.0, iou_weight=3.0),
            # one-to-many, geometry only (YSY, 2026-10-15)
            o2m=dict(type="SimOTALaneAssigner", candidate_topk=3, min_dynamic_k=1, cls_weight=0.0,
                     cls_cost_mode="linear", point_weight=0.0, iou_weight=3.0),
        ),
        loss_cfg=dict(
            cls_loss_weight=2.0,      # softmax focal, summed over anchors / GT lanes per image (2.0)
            focal_alpha=0.25,         # (0.25)
            focal_gamma=2.0,          # (2.0)
            iou_loss_weight=4.0,      # LaneIoU on the 72 rows, every step (4.0)
            point_loss_weight=0.05,   # px smooth-L1 on the 72 output points, every step (0.05)
            point_target="lateral",   # "lateral": against the GT x at each point's own y;
                                      # "same_t": against the GT curve sampled at the same t ("lateral")
            cp_loss_weight=0.05,      # px smooth-L1 on the 4 control points against the chord-length
                                      # ridge LS fit of the GT, every step (0.05)
            extent_loss_weight=0.2,   # rows smooth-L1 on the curve's bottom and top against the GT's
                                      # first and last visible row; the start/length loss (0.2)
            smooth_l1_beta=1.0,       # (1.0)
            o2m_cls_weight=1.0,       # o2m focal loss, relative to cls_loss_weight (1.0)
            o2m_reg_weight=0.5,       # the o2m positives' geometry and strip terms, relative to the o2o
                                      # ones (0.5)
            # LaneIoU geometry of the official CurveLanes evaluation (224 wide, 124 high for the
            # 2560x1440 images); set it to the baseline run's value if that differs.
            iou_eval_shape=(124, 224),
            lane_width=2.5 / 224,     # LaneIoU half-width in the 224-wide eval space (7.5/800)
            lane_width_cost=10.0 / 224,  # cost half-width (30/800)
            seg_loss_weight=2.0,      # (1.0)
            seg_bg_weight=0.4,        # (0.4)
            seg_ignore_label=255,     # (255)
            cp_fit_ridge=1e-2,        # ridge of the GT control-point fit, toward the chord (1e-2)
            batched_loss=True,        # the loss for the whole batch at once (needs blrnet_batched_loss.patch);
                                      # False: the per-image loop, same values, ~2x slower step (True)
        ),
        target_adapter=dict(
            lane_keys=["lanes"],
            seg_keys=["gt_masks"],
            start_y_convention="image_y",  # official PackCLRNetInputs: y0 = 1 - n_out / n_strips
            length_unit="count",
        ),
        test_score=None,              # None: o2o scores without NMS, o2m scores with NMS;
                                      # "o2o" | "o2m" forces one (None). Only the o2o head reads the
                                      # strip's quality vector; the output lanes are the corrected
                                      # ones either way.
        # The o2m labels on a separate set of anchors (blrnet_o2m_branch.patch); None = on the
        # anchors themselves, as in the control. (None)
        o2m_branch_cfg=None,
        # True: the o2o head reads the anchor state without gradient (Polar R-CNN's
        # stop-gradient); needs o2m_branch_cfg=None. (False)
        o2o_stop_grad=False,
        # Render-and-compare refinement after the last GRU step (blrnet_strip.patch,
        # libs/clrbezier/raster_strip.py). None turns it off (the control). (None)
        strip_cfg=dict(
            levels=(0, 1),            # neck outputs sampled into the strip, fine -> coarse: P2 + P3 ((0, 1))
            half_bins=8,              # lateral bins per side: 17 bins (8)
            bin_px=2.0,               # bin spacing along the normal, input px: +-16 px reach (2.0)
            sigma_bins=1.0,           # Gaussian width of the rasters, in bins (1.0)
            iters=2,                  # passes; only the own raster moves between them (2)
            dim=32,                   # strip CNN width (32)
            layers=3,                 # 3x3 conv layers (GroupNorm, ReLU) per pass (3)
            num_neighbours=4,         # nearest candidates rendered into the strip, o2m-score weighted (4)
            row_step=2,               # strip rows = every 2nd of the 72 output points (36 rows) (2)
            gamma=0.8,                # loss weight ratio of an earlier pass to the next (0.8)
            loss_weight=1.0,          # lateral CE + L1 + visibility loss on the matched candidates (1.0)
            ce_weight=1.0,            # soft-label CE on the 17 bins (target: Gaussian at the GT crossing) (1.0)
            l1_weight=0.5,            # px smooth-L1 on the soft-argmax correction (0.5)
            vis_weight=0.5,           # BCE: does the GT cross this row inside the strip (0.5)
            step_weight=1.0,          # LaneIoU and point losses of the corrected output, as one more
                                      # refinement step (1.0)
        ),
    ),
    test_cfg=dict(
        conf_threshold=0.3,
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

# Schedule: the cascade's CurveLanes recipe.
total_epochs = 36
checkpoint_config = dict(interval=total_epochs)
# Validation on valid/valid.txt every val_interval epochs (official CurveLanes: 3).
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Batch size per GPU. The strip adds P2 (80 x 200 x 64) and its CNN activations
# ([32 x 48, 32, 36, 17] per layer and pass); if memory is short, lower strip_cfg.dim to 16
# or row_step to 3 first.
train_dataloader = dict(batch_size=32)
randomness = dict(seed=0, deterministic=True)
# Gradient clipping (global L2 norm). The CPU smoke had norms of 15-30 with this head;
# if clipping fires on most iterations, raise max_norm to 2-3x the typical value.
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=3e-4),
                     clip_grad=dict(max_norm=35, norm_type=2))
# Records every iteration whose loss or grad_norm exceeds 2x its running median
# (with the batch's image names) in work_dir/loss_spikes_rank*.jsonl.
custom_hooks = [dict(type="LossSpikeHook", factor=2.0)]
# grad_norm is averaged over the 50-iteration log window; also log its maximum.
log_processor = dict(
    type="LogProcessor", window_size=50, by_epoch=True,
    custom_cfg=[dict(data_src="grad_norm", method_name="max", window_size=50,
                     log_name="grad_norm_max")])
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=1500),
    dict(type="CosineAnnealingLR", eta_min=0, begin=0, T_max=total_epochs,
         end=total_epochs, by_epoch=True, convert_to_iter_based=True),
]
log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
