# BLRNet on CurveLanes with BezierFormer's training recipe, the transformer encoder and a
# stride-4 fused map for the reads that need position (Mask2Former's pixel-decoder pattern;
# blrnet_fine4.patch, after blrnet_strip_steps.patch). Standalone: nothing is inherited except
# the dataset file and the default runtime, and every setting the head reads is written out
# below, with the head's default in the trailing comment.
#
# The control is blrnet_curvelanes_r34_bgrecipe.py. Differences from it:
#   backbone.out_indices    (1, 2, 3) -> (0, 1, 2, 3): C2 too (stride 4, 64 channels)
#   neck.in_channels        [128, 256, 512] -> [64, 128, 256, 512], num_outs 3 -> 4: adds P2;
#                           P3-P5 are computed exactly as before and the head's num_levels=3
#                           still takes them (the last three) for the anchors, the encoder and
#                           the image tokens
#   fine_cfg                dict(seg=True): FineMap = 3x3 conv + GroupNorm + ReLU on P2 plus the
#                           encoded P3 upsampled (64 channels, stride 4). The per-step
#                           re-sampling along the current curve reads it instead of P3, and the
#                           segmentation decoder runs at stride 4 on P3-P5 upsampled plus it.
# Everything else (optimizer, schedule, assigners, encoder, anchors) is identical to the control.
# Cost at 320x800 (blrnet_tests/neck_cost_output.txt, fine4_output.txt): +2.5 GFLOPs per image
# (ResNet-34 is 37.4); stored activations at batch 32 about +1.0 GiB (P2 in the FPN 0.24, head
# step 0.77 including the stride-4 segmentation input). FineMap has 37k parameters.
#
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
#   num_pos_o2o / num_pos_o2m    positives per batch (o2o = number of GT lanes)
#   o2o_pos_prob / o2o_neg_prob  mean o2o score of positives / negatives
#   self_w_0, self_w_1, self_w_final  weight an anchor keeps on itself in the relation
#                                attention (1/17 = even split; ~1 = anchors ignore each other)
#   offset_px, point_offset_px   mean |x| of the sampling offsets / output point offsets
#                                (bounds 24 px / 40 px)
#
# Evaluate NMS-free (this config) and with NMS on the o2m scores
# (--cfg-options model.test_cfg.use_nms=True); sweep the threshold from 0.05 for both.
# Comparison: the cascade's 84.59 (o2m, NMS) / 84.23 (o2o, NMS-free) on valid.
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
cfg_name = "blrnet_fine4_curvelanes_r34_bgrecipe.py"

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
        out_indices=(0, 1, 2, 3),  # C2-C5 -> [64, 128, 256, 512] channels (C2 for the stride-4 map)
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
        num_outs=4,                   # P2-P5; the head reads P3-P5 (num_levels=3), fine_cfg reads P2
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
            num_anchors=48,          # learned anchors; CLRerNet's CurveLanes prior tiling at init (192)
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
            resample="fine",          # "fine": re-sample the finest map (here the fine_cfg stride-4
                                      # map, otherwise P3) along the current curve at every step;
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
            max_offset_px=48.0,       # their tanh bound, px (24)
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
            detach_feats=True,
            exclusive=True
        ),
        assign_cfg=dict(
            # one-to-one, on the last refinement step, geometry first: the score term is linear
            # and weighted 0.15 = 3 * 0.05, so it only reorders copies within 0.05 LaneIoU
            o2o=dict(type="HungarianLaneAssigner", cls_weight=1.0, cls_cost_mode="log",
                     point_weight=0.0, iou_weight=3.0),
            # one-to-many for the o2m head (feature learning, with-NMS comparison); None drops it
            o2m=dict(type="SimOTALaneAssigner", candidate_topk=3, min_dynamic_k=1, cls_weight=1.0,
                     cls_cost_mode="log", point_weight=0.0, iou_weight=3.0),
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
            o2m_reg_weight=0.5,       # the o2m positives' geometry terms, relative to the o2o ones (0.5)
            # LaneIoU geometry of the official CurveLanes evaluation (224 wide, 124 high for the
            # 2560x1440 images); set it to the baseline run's value if that differs.
            iou_eval_shape=(124, 224),
            lane_width=2.5 / 224,     # LaneIoU half-width in the 224-wide eval space (7.5/800)
            lane_width_cost=10.0 / 224,  # cost half-width (30/800)
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
        # Stride-4 map from P2 + the encoded P3 for the per-step re-sampling and the
        # segmentation decoder (blrnet_fine4.patch). None turns it off (the control). (None)
        fine_cfg=dict(
            seg=True,                 # segmentation decoder at stride 4 with the map as an input (True)
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
# AdamW, lr 1e-4, weight decay 1e-4; the backbone at 0.1 x = 1e-5. With paramwise_cfg every
# parameter gets its own group and the logged "lr" is the first group's, a backbone weight
# (1e-5 throughout); the head's lr follows the poly curve below.
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
# Poly decay per iteration: lr = (base - 1e-5) * (1 - t / T) ** 2 + 1e-5, no warmup.
# Head: 1e-4 -> 1e-5 (6.1e-5 after 9 epochs, 3.3e-5 after 18, 1.6e-5 after 27);
# backbone: base 1e-5 = the minimum, so constant.
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
