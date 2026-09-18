model = dict(
    type="CLRerNet",
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='SegMANEncoder_t',
        pretrained='/work/CLRerNet/pretrained/SegMAN_Encoder_t.pth.tar',
        style='pytorch'),
    neck=dict(
        type="SSMFPN",
        in_channels=[64, 144, 192],  # SegMAN Encoder
        out_channels=64, # (must be divisible by num_attn_heads * 4)
        num_outs=3,
        attn_num_heads=8,
        attn_window_size=7,
        attn_ssm_dtrank="auto",
        attn_window_dilation=1,
        attn_ssm_dstate=16,
        attn_ssm_ratio=2,
        attn_ssm_drop=0.1,
        attn_ssm_split=False,
        use_sa=False # use self-attention or cross-attention
    ),
    bbox_head=dict(
        type="CLRerHead",
        anchor_generator=dict(
            type="CLRerNetAnchorGenerator",
            num_priors=192,
            num_points=72,
        ),
        img_w=800,
        img_h=320,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        attention=dict(type="ROIGather"),
        loss_cls=dict(type="KorniaFocalLoss", alpha=0.25, gamma=2, loss_weight=2.0),
        loss_bbox=dict(type="SmoothL1Loss", reduction="none", loss_weight=0.2),
        loss_iou=dict(
            type="LaneIoULoss",
            lane_width=7.5 / 800,
            loss_weight=4.0,
        ),
        loss_seg=dict(
            type="CLRNetSegLoss",
            loss_weight=1.0,
            num_classes=5,  # 4 lanes + 1 background
            ignore_label=255,
            bg_weight=0.4,
        ),
        use_segman_decoder=True, # use SegMAN decoder for multi-task segmentation learning
        segman_decoder_params={
            'embed_dim': 128,
            'feat_proj_dim': 192,
            'num_classes': 5,
            'dropout_ratio': 0.01,
            'channel_split': False,
            'interpolate_mode': 'bilinear',
            'use_rpb': False
        }
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="DynamicTopkAssigner",
            max_topk=9,
            min_topk=1,
            cost_combination=1, # 0: CLRNet cost, 1: CLRerNet cost, 2: CLRerNet cost with classification
            assignment_type=2, # 0: dynamic k assign, 1: atss assign, 2: atss assign2 (candidates are selected based on the cost matrix)
            use_dynamick_alt=True,
            cls_cost=dict(type="FocalCost", weight=1.0),
            reg_cost=dict(type="DistanceCost", weight=0.0),
            iou_dynamick=dict(
                type="LaneIoUCost",
                lane_width=7.5 / 800,
                use_pred_start_end=False,
                use_giou=True,
            ),
            iou_cost=dict(
                type="LaneIoUCost",
                lane_width=30 / 800,
                use_pred_start_end=True,
                use_giou=True,
            ),
        )
    ),
    test_cfg=dict(
        # conf threshold is obtained from cross-validation
        # of the train set. The following value is
        # for CLRerNet w/ DLA34 & EMA model.
        conf_threshold=0.41,
        use_nms=True,
        as_lanes=True,
        extend_bottom=True,
        nms_thres=50,
        nms_topk=4,
        ori_img_w=1640,
        ori_img_h=590,
        cut_height=270,
    ),
)
