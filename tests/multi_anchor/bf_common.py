"""The bfstyle head's settings (heads_bfstyle.json, written by test_bfstyle_configs.py from the delivered config) and
builders, as polar_common.py."""
import contextlib
import copy
import io
import json
import os

import fnet_common as C

HERE = os.path.dirname(os.path.abspath(__file__))


def bf_kwargs(**over):
    with open(os.path.join(HERE, "heads_bfstyle.json")) as f:
        h = json.load(f)
    kw, test_cfg = copy.deepcopy(h["bbox_head"]), h["test_cfg"]
    kw.pop("type")
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(kw.get(k), dict):
            kw[k] = dict(kw[k], **v)
        else:
            kw[k] = v
    return kw, test_cfg


def build_net(fl, seed=0, width=1.0, **over):
    import torch
    kw, test_cfg = bf_kwargs(**over)
    kw["type"] = "FovealLaneHead"
    kw["in_channels"] = [int(c * width) for c in kw["in_channels"]]
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return fl.FovealLaneNet(backbone=dict(type="mmdet.ResNet", depth=34, out_indices=(1, 2, 3), width=width),
                                bbox_head=kw, test_cfg=C.Cfg(test_cfg))
