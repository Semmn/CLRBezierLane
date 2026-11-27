import sys
for k in [k for k in sys.modules if k.split(".")[0] in ("mmdet", "mmengine", "mmcv")]:
    del sys.modules[k]
