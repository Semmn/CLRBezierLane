# CLRBezierLane Environment Setup

This guide describes the recommended Conda environment for **CLRBezierLane**.

## Environment

- Python 3.10
- PyTorch 2.1.2 + CUDA 12.1
- torchvision 0.16.2
- NumPy 1.26.4
- MMCV 2.1.0
- MMEngine 0.10.3
- MMDetection 3.3.0
- NATTEN 0.17.3
- SegMAN / VMamba `selective_scan`

> **Note**
> CLRBezierLane uses **MMDetection 3.3 / MMCV 2.x** as its main framework.
> Do not install SegMAN's original `mmcv-full`, `mmcls`, or MMSegmentation
> environment on top of this environment.

---

## 1. Create the Conda environment

```bash
conda create -n clrbezierlane python=3.10 -y
conda activate clrbezierlane

python -m pip install --upgrade pip wheel
```

---

## 2. Install PyTorch

Install PyTorch 2.1.2 with CUDA 12.1:

```bash
python -m pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 3. Pin NumPy and setuptools

CLRBezierLane uses NumPy 1.x because PyTorch 2.1.2 and several compiled
extensions are not compatible with NumPy 2.x in this environment.

`albumentations==0.4.6` also requires a setuptools version that still provides
`pkg_resources`.

```bash
python -m pip install --force-reinstall \
    numpy==1.26.4 \
    setuptools==80.9.0
```

---

## 4. Install the OpenMMLab stack

Install OpenMIM:

```bash
python -m pip install -U openmim
```

Then install the CLRerNet-compatible versions:

```bash
mim install "mmengine==0.10.3"
mim install "mmcv==2.1.0"
python -m pip install "mmdet==3.3.0"
```

Verify the compiled MMCV operators:

```bash
python - <<'PY'
import mmcv
import mmengine
import mmdet
from mmcv.ops import nms

print("MMCV:", mmcv.__version__)
print("MMEngine:", mmengine.__version__)
print("MMDetection:", mmdet.__version__)
print("MMCV ops: OK")
PY
```

Expected versions:

```text
MMCV        2.1.0
MMEngine    0.10.3
MMDetection 3.3.0
```

---

## 5. Install Python requirements

From the CLRBezierLane repository root:

```bash
cd ~/CLRBezierLane
```

Install the remaining dependencies:

```bash
python -m pip install \
    --no-build-isolation \
    -r requirements.txt
```

`--no-build-isolation` is required because `albumentations==0.4.6` is an old
source package whose build process expects `pkg_resources`.

The repository `requirements.txt` contains the CLRerNet dependencies and the
non-conflicting SegMAN dependencies. Framework-specific SegMAN packages such as
`mmcv-full` and `mmcls` are intentionally excluded.

---

## 6. Install NATTEN

Install the NATTEN wheel built for PyTorch 2.1.x and CUDA 12.1:

```bash
python -m pip install --no-deps \
    "natten==0.17.3+torch210cu121" \
    -f https://whl.natten.org/old
```

Verify:

```bash
python - <<'PY'
import natten
print("NATTEN:", natten.__version__)
PY
```

### Fallback

If the prebuilt wheel is unavailable, NATTEN can be built locally:

```bash
python -m pip install \
    --no-build-isolation \
    --no-deps \
    natten==0.17.3
```

---

## 7. Install Selective Scan

CLRBezierLane includes the required SegMAN / VMamba Selective Scan source under:

```text
kernels/selective_scan/
```

Make sure the CUDA compiler is available:

```bash
nvcc --version
```

Then build and install the extension:

```bash
cd ~/CLRBezierLane

python -m pip install \
    -v \
    --no-build-isolation \
    --no-deps \
    ./kernels/selective_scan
```

`--no-build-isolation` allows the build script to use the installed PyTorch
environment, while `--no-deps` prevents it from modifying the existing
PyTorch/CUDA package versions.

---


## 8. Build Lane NMS

CLRerNet uses a custom lane NMS extension located at:

```text
libs/models/layers/nms/
```

Build and install it from the repository root:

```bash
cd ~/CLRBezierLane

python -m pip install \
    -v \
    --no-build-isolation \
    --no-deps \
    ./libs/models/layers/nms
```

This is the Conda equivalent of the original Docker installation:

```bash
python libs/models/layers/nms/setup.py install
```

Using `pip install` is preferred because it keeps the extension installation
managed by the active Python environment.

If the pip installation fails, use the original CLRerNet fallback:

```bash
cd ~/CLRBezierLane/libs/models/layers/nms
python setup.py install
```


## 9. Verify the environment

Run:

```bash
python - <<'PY'
import numpy
import torch
import torchvision
import cv2
import albumentations
import mmcv
import mmengine
import mmdet
import natten
import einops
import timm

from mmcv.ops import nms

print("NumPy          :", numpy.__version__)
print("PyTorch        :", torch.__version__)
print("torchvision    :", torchvision.__version__)
print("CUDA           :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())
print("OpenCV         :", cv2.__version__)
print("Albumentations :", albumentations.__version__)
print("MMCV           :", mmcv.__version__)
print("MMEngine       :", mmengine.__version__)
print("MMDetection    :", mmdet.__version__)
print("NATTEN         :", natten.__version__)
print("einops         :", einops.__version__)
print("timm           :", timm.__version__)
print("MMCV ops       : OK")
PY
```

If the copied Selective Scan implementation exposes
`selective_scan_cuda_oflex`, verify it separately:

```bash
python - <<'PY'
import selective_scan_cuda_oflex
print("Selective Scan: OK")
PY
```

Finally, check for dependency conflicts:

```bash
python -m pip check
```

The core versions should remain:

```text
Python          3.10
NumPy           1.26.4
PyTorch         2.1.2+cu121
torchvision     0.16.2+cu121
MMCV            2.1.0
MMEngine        0.10.3
MMDetection     3.3.0
NATTEN          0.17.3
```

## 10. Install LaneMetricCore for Curvelane evaluation
Install noah-vega==1.8.5 with no dependencies.
```bash
pip install --no-deps noah-vega==1.8.5
```

## 11. CurveLanes dataset setup
1. Images need .lines.txt annotations next to them (train/images/*.lines.txt), as the curvelane branch reads. If your copy only has labels/*.lines.json, run curvelanes_json_to_txt.py.
```bash
python3 tools/clrbezier/curvelanes_json_to_txt.py your_curvelane_dataset_root
```

2. To write segmentation mask and file list (train/train_seg.txt) run below command
```bash
python3 tools/clrbezier/make_curvelanes_seg.py configs/clrernet/curvelanes/clrernet_curvelanes_r34.py
```