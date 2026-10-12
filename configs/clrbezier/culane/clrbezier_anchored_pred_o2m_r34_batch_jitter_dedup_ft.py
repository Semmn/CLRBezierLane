# Learned NMS (bbox_head.dedup_cfg, dedup.py) trained on top of a finished O2M
# checkpoint. Only the small dedup head trains: the backbone and the rest of the
# head are frozen and run as at test time, so the detector stays exactly the
# loaded checkpoint. One checkpoint then evaluates both ways:
#   dedup scores, no NMS (this config's test_cfg), and
#   o2m scores + NMS (the baseline):
#     --cfg-options model.bbox_head.dedup_cfg.use_at_test=False model.test_cfg.use_nms=True
# Without NMS the dedup head keeps the top test_cfg.nms_topk lanes per image, the
# same cap NMS applies.
#
# load_from must be a checkpoint of _base_ (the config it was trained with).
_base_ = ["./clrbezier_anchored_pred_o2m_r34_batch_jitter.py"]
cfg_name = "clrbezier_anchored_pred_o2m_r34_batch_jitter_dedup_ft.py"

model = dict(
    # frozen and in eval mode (BN statistics untouched)
    backbone=dict(frozen_stages=4, norm_eval=True),
    bbox_head=dict(
        dedup_cfg=dict(
            train_mode="dedup_only",
            target="o2m_top1",   # per GT, the o2m positive with the lowest cost
            use_at_test=True,
        ),
    ),
    test_cfg=dict(use_nms=False),
)

total_epochs = 3
checkpoint_config = dict(interval=total_epochs)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
default_hooks = dict(checkpoint=dict(save_begin=1))
param_scheduler = [
    dict(type='CosineAnnealingLR', eta_min=0, begin=0, T_max=total_epochs, end=total_epochs,
         by_epoch=True, convert_to_iter_based=True),
]

load_from = "work_dirs/clrbezier/culane/clrbezier_anchored_pred_o2m_r34_batch_jitter/run1/epoch_25.pth"
