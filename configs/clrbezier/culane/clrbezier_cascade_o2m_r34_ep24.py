# CLRBezier cascade + one-to-many main branch on a 24-epoch cosine schedule.
#
# Same rationale as clrbezier_collab_perturb_r34_ep24.py: your validation curve
# for this variant also improves to ~24 epochs and plateaus at conf 0.40.
#
# One thing to watch when shortening: the quality gate's and the re-projector's
# `warmup_iters` are counted in iterations, not epochs. At batch 32 on CULane
# with redundant-frame filtering (55,698 frames) one epoch is ~1,740 iterations,
# so 36 -> 24 epochs turns a 5,000-iteration warmup from 8% of training into
# 12%. Scale them by 2/3 if you want the same fraction of the schedule.
_base_ = ["clrbezier_cascade_o2m_r34.py"]

cfg_name = "clrbezier_cascade_o2m_r34_ep24.py"

total_epochs = 24
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

model = dict(test_cfg=dict(conf_threshold=0.55))
