# Visualization Tools
# MODE=png (paper figure)
# MODE=csv (csv file)
# MODE=table (best / plateau table)
# MODE=dash (interactive page)

# Visualize through Figures (PNG)
CONFIG_NAME="clrbezier_rank_r34"
TRAIN_DIR="run1"
MODEL_NAME="clrbezier"
DATASET="culane"
LOG_DIR="work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$TRAIN_DIR"
MODE=dash
python3 -m lanevis $MODE $LOG_DIR -o $LOG_DIR/$MODE

CONFIG_NAME="clrbezier_lft_deform_gliou_r34"
TRAIN_DIR="run1"
MODEL_NAME="clrbezier"
DATASET="culane"
LOG_DIR="work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$TRAIN_DIR"
MODE=dash
python3 -m lanevis $MODE $LOG_DIR -o $LOG_DIR/$MODE


# Curvature probing tools
# --mode: ablate, model, gt
CUDA_VISIBLE_DEVICES=2 python tools/clrbezier/test_curvature.py --mode ablate \
    --data-root /exhdd/seungyu/dataset/LaneDataset/CULane \
    --data-list /exhdd/seungyu/dataset/LaneDataset/CULane/list/test.txt \
    --config /exhdd/seungyu/CLRBezierLane/configs/clrbezier/culane/clrbezier_anchored_o2m_r34.py \
    --checkpoint /exhdd/seungyu/CLRBezierLane/work_dirs/clrbezier/culane/clrbezier_anchored_o2m_r34/run1/epoch_36.pth \
    --jobs 8 --expect-f1 80.56

CUDA_VISIBLE_DEVICES=2 python tools/clrbezier/test_curvature.py --mode ablate \
    --data-root /exhdd/seungyu/dataset/LaneDataset/CULane \
    --data-list /exhdd/seungyu/dataset/LaneDataset/CULane/list/test.txt \
    --config /exhdd/seungyu/CLRBezierLane/configs/clrernet/culane/clrernet_culane_r34.py \
    --checkpoint /exhdd/seungyu/CLRBezierLane/work_dirs/clrernet/culane/clrernet_culane_r34_e36/run1/epoch_36.pth \
    --jobs 8 --expect-f1 80.38

# Confidence score eval sweep tools
python tools/clrbezier/sweep_conf_eval.py configs/clrbezier/culane/clrbezier_anchored_o2m_r34.py CKPT \
    --range 0.80 0.90 \
    --step 0.02 \
    --out ./work_dirs/clrbezier/culane/clrbezier_anchored_o2m_r34/threshold_sweep.txt