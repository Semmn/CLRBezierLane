# Lane-slot experts for the per-query head (bbox_head.slot_expert_cfg,
# libs/clrbezier/slot_experts.py) on top of the config the base checkpoint was
# trained with.
#
# Every query is routed, at every stage, by the slot of the lane it samples at
# that stage (the prior at stage 0, the previous stage's output after that): the
# bin of the lane's chord angle. That slot's expert has its own lane embedding
# (RoIGather fc + LayerNorm), classification tower + outputs and regression
# tower + output, so it decides where the lane goes as well as its score. Both
# branches are routed, so the o2o and o2m heads still share everything but the
# final cls layer, as in the base head. No router to train: the slot is geometry.
#
# Fine-tune from the base checkpoint (load_from below). Every expert starts as a
# copy of the base layers, so epoch 0 = the base model.
# Control: clrbezier_anchored_pred_o2o_r34_batch_jitter_slotexperts_k1.py (one
# expert, same code path and schedule).
#
# _base_ must be the config the base checkpoint was trained with.
_base_ = ["./clrbezier_anchored_pred_o2o_r34_batch_jitter.py"]
cfg_name = "clrbezier_anchored_pred_o2o_r34_batch_jitter_slotexperts.py"

model = dict(
    bbox_head=dict(
        slot_expert_cfg=dict(
            num_experts=4,          # outer left | ego left | ego right | outer right
            measure="angle",        # chord angle in degrees, 90 = vertical
            # quartiles of the CULane train GT lane angle (router script log,
            # 2026-10-11): each slot gets a quarter of the lanes
            bin_edges=[43.881586, 89.597412, 136.069443],
            parts=["embed", "cls", "reg"],  # any non-empty subset
            branches=["main", "aux"],       # ["main"]: aux keeps the base layers
        ),
    ),
)

load_from = "work_dirs/clrbezier/culane/clrbezier_anchored_pred_o2o_r34_batch_jitter/run1/epoch_15.pth"

# fine-tune schedule (as the curvmoe configs)
max_epochs = 5
train_cfg = dict(max_epochs=max_epochs, val_interval=max_epochs)
optim_wrapper = dict(optimizer=dict(lr=1e-4))
param_scheduler = [
    dict(type="CosineAnnealingLR", eta_min=0, begin=0, T_max=max_epochs, end=max_epochs,
         by_epoch=True, convert_to_iter_based=True),
]
