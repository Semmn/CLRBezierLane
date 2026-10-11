# clrbezier_anchored_pred_o2m_r34_batch_jitter_diffusion.py with the cascade
# quality gate on the main (SimOTA) branch. The gate runs after the assigner:
# at stages 1 and 2 a pair whose stage output has a narrow LaneIoU below
# min_iou becomes a negative (no regression either, "cls_and_reg"); every GT
# keeps its best pair. Stage 0 stays ungated (0.0), so queries that start from
# a heavily noised anchor still get their stage-0 regression.
#
# This is a second change against clrbezier_anchored_pred_o2m_r34_batch_jitter.py:
# to credit the diffusion alone, the base needs the same gate.
_base_ = ["./clrbezier_anchored_pred_o2m_r34_batch_jitter_diffusion.py"]
cfg_name = "clrbezier_anchored_pred_o2m_r34_batch_jitter_diffusion_gate.py"

model = dict(
    bbox_head=dict(
        main_quality_gate=dict(
            _delete_=True,         # the base sets main_quality_gate=None
            # narrow LaneIoU (~CULane metric IoU); 0.5 = already a metric TP
            min_iou=[0.0, 0.3, 0.5],
            keep_best=True,        # every GT keeps its best pair
            mode="cls_and_reg",    # "cls_only": gated pairs still get regression
            gate_on="output",      # "input": Cascade R-CNN definition (pair with re-projection)
            warmup_iters=1500,     # thresholds ramp from 0
        ),
    ),
)
