# Converge, then decide: screening run (decider.patch, after tie_cost.patch).
#
# The decision stage (bbox_head.decider_cfg, libs/clrbezier/decider.py) trains alone
# on the finished o2m checkpoint (F1 84.59). The backbone, the neck and the cascade
# are frozen and run as at test time, so the lanes are exactly that checkpoint's; only
# who gets the score changes. The decider re-pools features along each final lane,
# attends to the nearest other candidates, and is trained one-to-one on the final
# lanes, geometry first, with its own score bounded to breaking ties (0.05 LaneIoU).
#
# One checkpoint evaluates both ways (sweep the threshold from 0.05 for each):
#   decider scores, no NMS (this config's test_cfg)            <- the claim
#   decider scores + NMS:   --cfg-options model.test_cfg.use_nms=True
#   the o2m baseline:       --cfg-options model.bbox_head.decider_cfg.use_at_test=False model.test_cfg.use_nms=True
#
# load_from must be a checkpoint of _base_ (the config it was trained with).
_base_ = ["./clrbezier_global_pred_o2m_r34.py"]
cfg_name = "clrbezier_global_pred_o2m_r34_decider_ft.py"

model = dict(
    # frozen and in eval mode (BN statistics untouched)
    backbone=dict(frozen_stages=4, norm_eval=True),
    bbox_head=dict(
        decider_cfg=dict(
            train_mode="decider_only",
            # defaults, spelled out: Hungarian on the final lanes, LaneIoU first,
            # the decider's own score reorders copies within 0.05 LaneIoU only
            assigner=dict(type="HungarianLaneAssigner", cls_weight=0.15,
                          cls_cost_mode="linear", point_weight=0.0, iou_weight=3.0),
            loss="focal", loss_weight=2.0, quality_weight=1.0,
            dim=128, num_layers=2, num_neighbors=16,
            use_at_test=True,
        ),
    ),
    # focal-trained one-to-one scores sit lower than the o2m ones (sweep from 0.05)
    test_cfg=dict(use_nms=False, conf_threshold=0.3),
)

total_epochs = 6
checkpoint_config = dict(interval=total_epochs)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=2)
default_hooks = dict(checkpoint=dict(save_begin=1))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, T_max=total_epochs, end=total_epochs,
         by_epoch=True, convert_to_iter_based=True),
]

load_from = "work_dirs/clrbezier/curvelanes/clrbezier_global_pred_o2m_r34/run1/epoch_36.pth"
