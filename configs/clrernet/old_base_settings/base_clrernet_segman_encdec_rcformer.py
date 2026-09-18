model = dict(
    type="CLRerNet",
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='SegMANEncoder_s',
        pretrained='/work/CLRerNet/pretrained/SegMAN_Encoder_s.pth.tar',
        style='pytorch'),
    neck=dict(
        type="RCFormerFPN",
        in_channels=[144, 288, 512],  # SegMAN Encoder
        out_channels=144,
        num_outs=3,
        num_rc=6, # number of row-column attention block layers
        feat_size=[(40, 100), (20, 50), (10, 25)], # feature map sizes
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
        prior_feat_channels=144,
        fc_hidden_dim=144,
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
            max_topk=4,
            min_topk=1,
            cost_combination=1,
            assignment_type=0,
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
