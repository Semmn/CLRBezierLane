# The O2M jitter config with the linear prior update (prior_cfg.param="linear"):
# cp = prior_cp + delta_scale * delta instead of
# sigmoid(prior_logits + delta_scale * delta). Every control point moves at the
# same rate, and the points on the image border (frozen by the sigmoid) can move.
#
# delta_scale 1.0 gives CLRerNet's prior step (about lr per AdamW step). The
# base uses 0.2, so this differs from it in two ways; a sigmoid run at
# delta_scale 1.0 on the same base isolates the sigmoid.
#
# From scratch only: a checkpoint trained with the sigmoid refuses to load.
_base_ = ["./clrbezier_anchored_pred_o2m_r34_batch_jitter.py"]
cfg_name = "clrbezier_anchored_pred_o2m_r34_batch_jitter_linprior.py"

model = dict(
    bbox_head=dict(
        prior_cfg=dict(param="linear", delta_scale=1.0),
    ),
)

load_from = None   # from scratch (the backbone still starts from ImageNet)
