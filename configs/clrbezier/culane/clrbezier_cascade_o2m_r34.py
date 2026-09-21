# CLRBezierLane-R34, one-to-many main branch with stage-increasing positive
# quality (Cascade R-CNN) and reference re-projection.
#
# Main branch: SimOTA dynamic-k at every stage (close to CLRerNet's dynamic-k:
# candidate_topk=4, no point cost), with a per-stage minimum LaneIoU for
# positives. The collaborative branch is unchanged.
#
# Variants from this file:
#   gate only          -> reproject_cfg=None
#   re-projection only -> main_quality_gate=None
_base_ = ["clrbezier_collab_perturb_r34.py"]

cfg_name = "clrbezier_cascade_o2m_r34.py"

_o2m = dict(type="SimOTALaneAssigner", candidate_topk=4, min_dynamic_k=1,
            cls_weight=1.0, point_weight=0.0, iou_weight=3.0)

model = dict(
    bbox_head=dict(
        main_assigner=_o2m,
        # per-stage override; stages not listed use main_assigner
        main_stage_assigners={"0": _o2m, "1": _o2m, "2": _o2m},
        main_quality_gate=dict(
            # narrow LaneIoU (~CULane metric IoU); 0.5 = already a metric TP
            min_iou=[0.0, 0.3, 0.5],
            keep_best=True,        # every GT keeps its best pair
            mode="cls_and_reg",    # "cls_only": gated pairs still get regression
            gate_on="output",      # "input": Cascade R-CNN definition (pair with re-projection)
            warmup_iters=3000,     # thresholds ramp from 0
        ),
        reproject_cfg=dict(
            stages=[0, 1],         # stages whose output becomes the next reference
            ridge=1e-2,            # same ridge as the GT control-point fit
            min_rows=4,            # fewer visible predicted rows -> keep Bezier reference
            blend=1.0,             # final weight of the refit
            warmup_iters=3000,     # blend ramps from 0
            max_shift_px=20.0,     # per-control-point trust region (None disables)
        ),
    ),
)
