# Revisited at the 2026.08.25
# All the results are tested under the confidence score of 0.41 in case of DLA34 backbone network.
# Training # 1 (Anchor 35, DLA34, epochs 15)
mkdir -p /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/run1
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 nohup bash tools/dist_train.sh /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35.py 4 --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/run1 > /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/run1/train.log 2>&1 < /dev/null &
tail -f /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/run1/train.log

# Training # 2 (Anchor 35, DLA34, epochs 36)
mkdir -p /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/run1
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 nohup bash tools/dist_train.sh /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35_ep36.py 4 --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/run1 > /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/run1/train.log 2>&1 < /dev/null &
tail -f /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/run1/train.log

# Training # 3 (Reproduce - Anchor 192, DLA34, epochs 15)
mkdir -p /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run1
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 nohup bash tools/dist_train.sh /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_reproduce.py 4 --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run1 > /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run1/train.log 2>&1 < /dev/null &
tail -f /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run1/train.log


# Evaluations # 1 (Anchor 35, DLA34, epochs 15)
CUDA_VISIBLE_DEVICES=0 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/run1/epoch_15.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/test1 2>&1 < /dev/null &

# Evaluations # 2 (Anchor 35, DLA34, epochs 36)
CUDA_VISIBLE_DEVICES=0 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35_ep36.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/run1/epoch_36.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/test1 2>&1 < /dev/null &
# (Anchor 35, DLA34, epochs 36, Conf 0.45)
CUDA_VISIBLE_DEVICES=1 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35_ep36.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/run1/epoch_36.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/test1 2>&1 < /dev/null &

# Evaluations # 3 (Reproduce - Anchor 192, DLA34, epochs 15, confidence score: 0.41)
CUDA_VISIBLE_DEVICES=2 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_reproduce.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run1/epoch_15.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/test1 2>&1 < /dev/null &


# Revisited at the 2026.09.07 
# Evaluation (Reproduce - Anchor 192, DLA34, epochs 15, confidence score: 0.43)
CUDA_VISIBLE_DEVICES=4 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_reproduce.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run1/epoch_15.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/test2 2>&1 < /dev/null &

# Training (Reproduce - Anchor 192, DLA34, epochs 15, batch size=24: exactly the same settings even for the seed)
mkdir -p /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run2
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 nohup bash tools/dist_train.sh /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_reproduce.py 1 --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run2 > /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run2/train.log 2>&1 < /dev/null &
tail -f /work/CLRerNet/work_dirs/clrernet_culane_dla34_reproduce/run2/train.log
# Evaluation (Anchors=35, DLA34, epochs 15, confidence score: 0.43)
CUDA_VISIBLE_DEVICES=0 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/run1/epoch_15.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/test2 2>&1 < /dev/null &
# Evaluation (Anchors=35, DLA34, epochs 15, confidence score: 0.45)
CUDA_VISIBLE_DEVICES=1 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/run1/epoch_15.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/test3 2>&1 < /dev/null &
# Evlauation (Anchors=35, DLA34, epochs 15, confidence score: 0.47)
CUDA_VISIBLE_DEVICES=0 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/run1/epoch_15.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35/test3 2>&1 < /dev/null &

# Evaluation (Anchors=35, DLA34, epochs 35, confidence score: 0.47)
CUDA_VISIBLE_DEVICES=1 nohup python3 tools/test.py /work/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_anchor35_ep36.py /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/run1/epoch_36.pth --work-dir /work/CLRerNet/work_dirs/clrernet_culane_dla34_anchor35_ep36/test1 2>&1 < /dev/null &

# ================================================================================================
# Visited at the 2026.09.17
# Now the main development starts with this repository.
# Proving the official repository.
python tools/probe_official.py \
    configs/clrbezier/culane/clrbezier_culane_r34.py \
    --baseline-config configs/clrernet/culane/clrernet_culane_r34.py

# Start Training for ported model
CONFIG_NAME="clrbezier_collab_perturb_r34"
MODEL_NAME="clrbezier"
PORT=25000
DATASET=culane
mkdir -p /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=$PORT nohup bash tools/dist_train.sh \
    /work/CLRerNet/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    4 \
    --work-dir /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/ \
    > /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log \
    2>&1 < /dev/null &
tail -f /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log

# Start training for CLRerNet with resnet34 and number of anchors=35 (epoch 36)
CONFIG_NAME="clrernet_culane_r34_a35"
MODEL_NAME="clrernet"
PORT=25001
DATASET=culane
mkdir -p /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=$PORT nohup bash tools/dist_train.sh \
    /work/CLRerNet/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    4 \
    --work-dir /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/ \
    > /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log \
    2>&1 < /dev/null &
tail -f /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log

# Test the ported model
CONFIG_NAME="clrbezier_collab_perturb_r34"
MODEL_NAME="clrbezier"
TRAIN_EXP_NAME=run1
EVAL_EXP_NAME=test1
DATASET=culane
mkdir -p /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME
CUDA_VISIBLE_DEVICES=4 nohup python3 tools/test.py /work/CLRerNet/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$TRAIN_EXP_NAME/epoch_36.pth \
    --work-dir /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME \
    > /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME/test.log \
    2>&1 < /dev/null &

# Test the CLRerNet (a=35, e=36)
CONFIG_NAME="clrernet_culane_r34_a35"
MODEL_NAME="clrernet"
TRAIN_EXP_NAME=run1
EVAL_EXP_NAME=test1
DATASET=culane
mkdir -p /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME
CUDA_VISIBLE_DEVICES=1 nohup python3 tools/test.py /work/CLRerNet/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$TRAIN_EXP_NAME/epoch_36.pth \
    --work-dir /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME \
    > /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME/test.log \
    2>&1 < /dev/null &


# ================================================================================================
# Visited at the 2026.09.18
# Porting Different Modules
CONFIG_NAME="clrbezier_collab_perturb_focal_r34"
MODEL_NAME="clrbezier"
PORT=25000
DATASET=culane
mkdir -p /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=$PORT nohup bash tools/dist_train.sh \
    /work/CLRerNet/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    4 \
    --work-dir /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/ \
    > /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log \
    2>&1 < /dev/null &
tail -f /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log


CONFIG_NAME="clrbezier_collab_perturb_focal_gsrc_r34"
MODEL_NAME="clrbezier"
PORT=25001
DATASET=culane
mkdir -p /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=$PORT nohup bash tools/dist_train.sh \
    /work/CLRerNet/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    4 \
    --work-dir /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/ \
    > /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log \
    2>&1 < /dev/null &
tail -f /work/CLRerNet/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log


# ================================================================================================
# Visited at the 2026.09.19
# Now Training on the Conda Environment instead of docker
# lane evidence pooler & IoU as target
CONFIG_NAME="clrbezier_collab_perturb_r34_evidence"
MODEL_NAME="clrbezier"
PORT=25000
DATASET=culane
mkdir -p /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=$PORT nohup bash tools/dist_train.sh \
    /exhdd/seungyu/CLRBezierLane/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    4 \
    --work-dir /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/ \
    > /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log \
    2>&1 < /dev/null &
tail -f /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log


CONFIG_NAME="clrbezier_collab_perturb_r34_lateral"
MODEL_NAME="clrbezier"
PORT=25001
DATASET=culane
mkdir -p /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=$PORT nohup bash tools/dist_train.sh \
    /exhdd/seungyu/CLRBezierLane/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    4 \
    --work-dir /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/ \
    > /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log \
    2>&1 < /dev/null &
tail -f /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log


# ================================================================================================
# Visited at the 2026.09.20
# Test Lateral Evidence only model and Lateral Evidence + Query-to-Query Attention + IoU model
# Test the confidence score range of 0.40~0.50

# Test Default model design with Focal Loss adapted
CONFIG_NAME="clrbezier_collab_perturb_r34_evidence"
MODEL_NAME="clrbezier"
TRAIN_EXP_NAME=run1
EVAL_EXP_NAME=test1
DATASET=culane
mkdir -p /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=2 nohup python3 tools/test.py /exhdd/seungyu/CLRBezierLane/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$TRAIN_EXP_NAME/epoch_36.pth \
    --work-dir /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME \
    > /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME/test.log \
    2>&1 < /dev/null &

# Test GSRC adapted model design with Focal Loss adapted
CONFIG_NAME="clrbezier_collab_perturb_r34_lateral"
MODEL_NAME="clrbezier"
TRAIN_EXP_NAME=run1
EVAL_EXP_NAME=test1
DATASET=culane
mkdir -p /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=3 nohup python3 tools/test.py /exhdd/seungyu/CLRBezierLane/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$TRAIN_EXP_NAME/epoch_36.pth \
    --work-dir /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME \
    > /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME/test.log \
    2>&1 < /dev/null &

# ================================================================================================
# Visited at the 2026.09.21
# Test No NMS Result Behavior (confidence score range of 0.70 ~ 0.90)
CONFIG_NAME="clrbezier_collab_perturb_r34"
MODEL_NAME="clrbezier"
TRAIN_EXP_NAME=run1
EVAL_EXP_NAME=test1
DATASET=culane
mkdir -p /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0 nohup python3 tools/test.py /exhdd/seungyu/CLRBezierLane/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$TRAIN_EXP_NAME/epoch_36.pth \
    --work-dir /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME \
    > /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/$EVAL_EXP_NAME/test.log \
    2>&1 < /dev/null &

CONFIG_NAME="clrbezier_r34_principled"
MODEL_NAME="clrbezier"
PORT=25001
DATASET=culane
mkdir -p /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=$PORT nohup bash tools/dist_train.sh \
    /exhdd/seungyu/CLRBezierLane/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    4 \
    --work-dir /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/ \
    > /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log \
    2>&1 < /dev/null &
tail -f /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log

CONFIG_NAME="clrbezier_reproject_r34"
MODEL_NAME="clrbezier"
PORT=25000
DATASET=culane
mkdir -p /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=$PORT nohup bash tools/dist_train.sh \
    /exhdd/seungyu/CLRBezierLane/configs/$MODEL_NAME/$DATASET/$CONFIG_NAME.py \
    4 \
    --work-dir /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/ \
    > /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log \
    2>&1 < /dev/null &
tail -f /exhdd/seungyu/CLRBezierLane/work_dirs/$MODEL_NAME/$DATASET/$CONFIG_NAME/run1/train.log
