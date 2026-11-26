"""Shared pieces of the polar-view CPU tests: the polar head's settings (the delivered config, written to
heads_polar.json by test_polar_configs.py, or the window settings of fnet_common.py with POLAR_FC below) and
builders."""
import contextlib
import copy
import io
import json
import os

import fnet_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
POLAR_FC = dict(mode="polar", polar_w=800, polar_h=256, theta_deg=[-10.0, 190.0], r0_px=24.0, supersample=[2, 4],
                p_gt=0.5, jitter_px=20.0, jitter_y_px=10.0, bn="context")


def polar_kwargs(**over):
    path = os.path.join(HERE, "heads_polar.json")
    if os.path.exists(path):
        with open(path) as f:
            h = json.load(f)
        kw, test_cfg = copy.deepcopy(h["bbox_head"]), h["test_cfg"]
        kw.pop("type")
    else:
        kw, test_cfg = C.head_kwargs("window")
        kw["fovea_cfg"] = dict(POLAR_FC)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(kw.get(k), dict):
            kw[k] = dict(kw[k], **v)
        else:
            kw[k] = v
    return kw, test_cfg


def build_head(fl, seed=0, **over):
    import torch
    kw, test_cfg = polar_kwargs(**over)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return fl.FovealLaneHead(test_cfg=C.Cfg(test_cfg), **kw)


def build_net(fl, seed=0, width=1.0, **over):
    import torch
    kw, test_cfg = polar_kwargs(**over)
    kw["type"] = "FovealLaneHead"
    kw["in_channels"] = [int(c * width) for c in kw["in_channels"]]
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return fl.FovealLaneNet(backbone=dict(type="mmdet.ResNet", depth=34, out_indices=(1, 2, 3), width=width),
                                bbox_head=kw, test_cfg=C.Cfg(test_cfg))


def ray_lanes(vp, bottoms, r_top=25.0, n=30, w=800.0, h=320.0):
    """Straight lanes from the bottom points towards the vanishing point vp (context px), stopping r_top px short of
    it, cut to the crop; bottom -> top flat lists."""
    import numpy as np
    out = []
    for xb, yb in bottoms:
        d = np.array([vp[0] - xb, vp[1] - yb])
        L = np.hypot(*d)
        s = np.linspace(0.0, 1.0 - r_top / L, 400)
        p = np.array([xb, yb])[None] + s[:, None] * d[None]
        p = p[(p[:, 0] >= 0) & (p[:, 0] <= w - 1) & (p[:, 1] >= 0) & (p[:, 1] <= h - 1)]
        idx = np.linspace(0, len(p) - 1, n).round().astype(int)
        out.append(p[idx].reshape(-1).tolist())
    return out
