# CLRerNet-R34 baseline (paper Table 2: CLRerNet† Res34, 80.76 ± 0.13).
#
# Only the backbone differs from the official DLA34 config. Neck, head,
# data pipeline, schedule, and evaluation are inherited unchanged.
#
# Before a full run, verify with tools/clrbezier/probe_official.py that the
# official neck accepts mmdet ResNet outputs (a tuple of C3, C4, C5). If the
# official DLANet returns a different container, adapt only this backbone
# block (e.g. out_indices=(0, 1, 2, 3)).
_base_ = [
    "../base_clrernet_r34.py",
    "dataset_culane_clrernet.py",
    "../../_base_/default_runtime.py",
]

default_scope = 'mmdet'

# custom imports
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
    ],
    allow_failed_imports=False,
)

cfg_name = "clrernet_culane_r34.py"
model = dict(test_cfg=dict(conf_threshold=0.39))

total_epochs = 15
checkpoint_config = dict(interval=total_epochs)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_dataloader=dict(
    batch_size=24
 ) # Number of batch size PER GPU

# seed
randomness = dict(seed=0, deterministic=True)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type="AdamW", lr=6e-4),
)

# learning rate policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        begin=0,
        T_max=total_epochs,
        end=total_epochs,
        by_epoch=True,
        convert_to_iter_based=True
    ),
]

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)