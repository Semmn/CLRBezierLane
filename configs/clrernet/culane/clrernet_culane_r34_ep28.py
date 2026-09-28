# CLRerNet-R34 baseline on a 28-epoch cosine schedule.
#
# Your validation curve for the baseline improves to ~28 epochs at conf 0.49,
# while the 36-epoch test result (80.38 F1) came out below the paper's 15-epoch
# R34 number (80.76 +- 0.13). Two readings are consistent with that, and this
# config exists to separate them:
#
#   1. the extra epochs help but the 36-epoch cosine anneals too slowly, in
#      which case a completed 28-epoch cosine beats both, or
#   2. CULane validation stops tracking test beyond ~15 epochs, in which case
#      this lands near 80.4 as well and the paper schedule (15) is the one to
#      report against.
#
# Reading 2 would be the same val/test divergence you already saw with IoU
# target learning, so it is worth one run before you trust any epoch count
# picked on validation.
#
# Note on the existing ep36 config: `model = dict(test_cfg=...)` followed by a
# second `model = dict(bbox_head=...)` rebinds `model`, so the conf_threshold
# 0.41 in the first line never reaches the merged config and the base 0.50
# applies. Merge those into one dict (as below) when you rerun it.
_base_ = ["clrernet_culane_r34.py"]

cfg_name = "clrernet_culane_r34_ep28.py"

total_epochs = 28
checkpoint_config = dict(interval=total_epochs)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(
        type="CosineAnnealingLR",
        eta_min=0,
        begin=0,
        T_max=total_epochs,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_begin=max(1, total_epochs - 10),
        max_keep_ckpts=11,
    ),
)

model = dict(test_cfg=dict(conf_threshold=0.49))
