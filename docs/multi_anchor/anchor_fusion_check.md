# Multi-anchor premise test (anchor_fusion_check.py)

Question (multi_anchor_analysis.md, sec. 3): do two differently anchored detectors fail on different lanes, so that
the near field of one plus the far field of the other would place lanes the first one misses?

Eval-only. No libs/ change, no training.

## Install

Copy `anchor_fusion_check.py` to `tools/clrbezier/`, next to `curvelanes_error_breakdown.py` and `repr_ceiling.py`.
The tool reads their dumps and reuses their metric code.

## Run (on YSY's server)

Step 1: one inference dump per checkpoint (GPU, one pass each over CurveLanes valid). Skip any dump you already have.

```
cd ~/CLRBezierLane
W=work_dirs/clrbezier/curvelanes
python tools/clrbezier/curvelanes_error_breakdown.py configs/clrbezier/curvelanes/foveal_lane_off_bfstyle_o2o_pointcond_curvelanes_r34.py \
    $W/foveal_lane_off_bfstyle_o2o_pointcond_curvelanes_r34/run1/epoch_24.pth \
    --save-dump work_dirs/err/pointcond.pkl --out-dir work_dirs/err/pointcond
python tools/clrbezier/curvelanes_error_breakdown.py configs/clrbezier/curvelanes/clrbezier_global_rows_o2o_p192_r34.py \
    $W/clrbezier_global_rows_o2o_p192_r34/run1/epoch_36.pth \
    --save-dump work_dirs/err/global_rows.pkl --out-dir work_dirs/err/global_rows
python tools/clrbezier/curvelanes_error_breakdown.py configs/clrbezier/curvelanes/blrnet_proposal_curvelanes_r34.py \
    $W/blrnet_proposal_curvelanes_r34/run1/epoch_36.pth \
    --save-dump work_dirs/err/blrnet_proposal.pkl --out-dir work_dirs/err/blrnet_proposal
```

`clrbezier_global_rows_o2o_p192_r34.py` is not in the GitHub repo's configs/. If it is missing on the server too,
use the copy mmengine saved in its work_dir (`$W/clrbezier_global_rows_o2o_p192_r34/run1/*.py`).

Take each model's threshold from its `summary.txt`: the starred row (best F1@50) in the column of its config's mode
(without NMS for the o2o configs, with NMS otherwise).

Step 2: the check (CPU, a few minutes). Base = the Bezier query model (its lanes, scores and extent are kept);
donor = the model whose near field is borrowed.

```
python tools/clrbezier/anchor_fusion_check.py \
    --base-config configs/clrbezier/curvelanes/foveal_lane_off_bfstyle_o2o_pointcond_curvelanes_r34.py \
    --base-dump work_dirs/err/pointcond.pkl --base-thr T_POINTCOND \
    --donor-config configs/clrbezier/curvelanes/clrbezier_global_rows_o2o_p192_r34.py \
    --donor-dump work_dirs/err/global_rows.pkl --donor-thr T_GLOBAL_ROWS \
    --out-dir work_dirs/anchor_fusion/pointcond_from_global_rows
```

Run it again with the BLRNet dump as donor (`--out-dir work_dirs/anchor_fusion/pointcond_from_blrnet`).

Send me both `summary.txt` files. `gt_lanes.csv` is only needed if the verdict is "in between".

## What it reports

- Rows `base` / `donor` / `fused` / `GT-gated oracle` / `union recall` (P, R, F1@50, F1@75).
  - `base` must equal the starred row of the base breakdown's sweep. If it doesn't, stop and tell me.
  - `fused` is realizable at test time: a fixed rule, no GT, no score.
  - `GT-gated oracle` fuses a lane only where that raises its best IoU with a GT lane. It uses the GT, so it is
    close to an upper bound for fusion, though not a strict one under the Hungarian.
- The header line counts paired lanes, paired lanes whose donor stops short of the near region (the donor's end
  offset is held there), and donor lanes shared by two base lanes.
- GT lanes split by base outcome (hit / near-only / other miss), with the donor's hit rate and the fused hit rate on
  each group.
- Signed near-third and far-third offsets of both models from the GT lane, with their correlation.
- VERDICT line, using the rule fixed before the run:
  - fused - base >= +1.0 F1@50 and the donor places >= 30% of near-only lanes: complementary, so design the joint
    model.
  - fused - base < +0.5 F1@50, or near-third r > 0.8 on the base's near-only lanes: same failure, so drop
    multi-anchoring.
  - If both fire, the gain rule wins and the line notes the high r.
  - Near-only means IoU in (0.2, 0.5], the 2026-10-23 near-miss definition.

Caveat: the two donors are older and weaker models, so a null result is partly a statement about them.

## Tests

`tests/test_anchor_fusion_check.py TREE`: 21/21 (`tests/test_anchor_fusion_check_output.txt`). They cover the
geometry (self-fusion is a no-op, the shift profile, point order, flat lanes, partial donors), one synthetic scene
end to end (a base near miss fixed by the donor, base counts equal to an independent Hungarian), and main() from two
dumps (1 worker == 2 workers).
