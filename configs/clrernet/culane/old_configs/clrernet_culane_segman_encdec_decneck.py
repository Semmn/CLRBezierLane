# inherit from runtime, base configs.
_base_ = [
    "../base_clrernet_segman_encdec_decneck.py",
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

cfg_name = "clrernet_culane_segman_encdec_decneck.py"

model = dict(test_cfg=dict(conf_threshold=0.41))

total_epochs = 15
checkpoint_config = dict(interval=total_epochs)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_dataloader=dict(
    batch_size=24
 ) # single GPU setting

# seed
randomness = dict(seed=0, deterministic=True)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-4), # instead of 6e-4 (learning rate is not halfved)
    clip_grad=dict(max_norm=1.0, norm_type=2) # gradient clipping is added to the model training
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
