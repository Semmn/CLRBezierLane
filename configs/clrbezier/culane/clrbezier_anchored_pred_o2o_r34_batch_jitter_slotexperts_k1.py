# Control for ..._slotexperts.py: the same fine-tune with ONE expert, so every
# query runs the same (base-initialized) layers. It separates what the slot
# experts add from what the five extra epochs add.
_base_ = ["./clrbezier_anchored_pred_o2o_r34_batch_jitter_slotexperts.py"]
cfg_name = "clrbezier_anchored_pred_o2o_r34_batch_jitter_slotexperts_k1.py"

model = dict(
    bbox_head=dict(
        slot_expert_cfg=dict(num_experts=1, bin_edges=[]),
    ),
)
