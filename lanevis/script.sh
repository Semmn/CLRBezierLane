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