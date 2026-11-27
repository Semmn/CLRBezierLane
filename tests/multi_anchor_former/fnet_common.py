"""Shared pieces of the FovealLaneNet CPU tests: loading the patched tree with the stubs, head / detector kwargs
(the delivered window config, read by test_foveal_configs.py into heads.json, or the built-in copy below), synthetic
CurveLanes-like samples in the context frame."""
import contextlib
import copy
import io
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tree):
    sys.path[:0] = [os.path.join(HERE, "stubs"), tree]
    sys.modules["libs.models"] = types.ModuleType("libs.models")      # the official head is not needed here
    import libs.clrbezier.foveal_lane as fl  # noqa: E402
    return fl


W, H = 800, 320
CURVE = dict(gt_points=48, out_per_segment=24, rel_points=12, cost_points=25, end_loss_weight=0.05, min_gt_len_px=8.0,
             orientation_free=True, frame="image", y_order="monotone", match_end_len=True)
HEAD = dict(
    in_channels=[128, 256, 512], num_queries=16, dim=256, num_points=72, img_w=W, img_h=H, seg_num_classes=2,
    fovea_cfg=dict(mode="window", window_w=400, window_h=80, window_top=0, p_gt=0.5, jitter_px=40.0, bn="context"),
    vp_cfg=dict(channels=128, loss_weight=1.0, beta=0.05),
    enc_cfg=dict(num_layers=3, num_heads=8, num_points=4, ffn_dim=1024, dropout=0.0, checkpoint=False),
    prop_cfg=dict(channels=64, sigma_cells=2.0, radius_cells=2, scale_px=100.0, hm_weight=1.0, end_weight=0.05,
                  iou_weight=1.0),
    dec_cfg=dict(num_iters=3, self_attn="full", num_heads=8, ffn_dim=1024, dropout=0.0, curve_points=36,
                 deform_heads=8, deform_points=4, step_scale=1.0, cp_loss=False, num_neighbors=16, dist_scale=15.0,
                 max_dist=10.0),
    curve_cfg=CURVE,
    hier_cfg=dict(segments=3, seg_iters=3, seg_deform_points=12, seg_sample_points=12, seg_neighbors=16,
                  seg_rel_points=8, schedule="sequential"),
    assign_cfg=dict(o2o=dict(type="HungarianLaneAssigner", cls_weight=1.0, cls_cost_mode="log", point_weight=3.0,
                             iou_weight=3.0),
                    o2m=dict(type="SimOTALaneAssigner", candidate_topk=3, min_dynamic_k=1, cls_weight=1.0,
                             cls_cost_mode="log", point_weight=3.0, iou_weight=3.0, ensure_one=True)),
    loss_cfg=dict(cls_loss_weight=2.0, focal_alpha=0.25, focal_gamma=2.0, iou_loss_weight=4.0, point_loss_weight=0.05,
                  cp_loss_weight=0.05, extent_loss_weight=0.2, smooth_l1_beta=1.0, o2m_cls_weight=1.0,
                  o2m_reg_weight=0.5, iou_eval_shape=(124, 224), lane_width=2.5 / 224, lane_width_cost=10.0 / 224,
                  seg_loss_weight=2.0, seg_bg_weight=0.4, seg_ignore_label=255, cp_fit_ridge=1e-2),
    target_adapter=dict(lane_keys=["lanes"], seg_keys=["gt_masks"], start_y_convention="image_y", length_unit="count"),
    test_score=None)
TEST_CFG = dict(conf_threshold=0.5, use_nms=False, as_lanes=True, extend_bottom=True, nms_thres=15, nms_topk=16,
                ori_img_w=1, ori_img_h=1, cut_height=0)


def head_kwargs(mode="window", **over):
    path = os.path.join(HERE, "heads.json")
    if os.path.exists(path):                              # written by test_foveal_configs.py from the configs
        with open(path) as f:
            heads = json.load(f)
        kw = copy.deepcopy(heads[mode]["bbox_head"])
        kw.pop("type")
        test_cfg = heads[mode]["test_cfg"]
    else:
        kw = copy.deepcopy(HEAD)
        kw["fovea_cfg"]["mode"] = mode
        test_cfg = dict(TEST_CFG)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(kw.get(k), dict):
            kw[k] = dict(kw[k], **v)
        else:
            kw[k] = v
    return kw, test_cfg


class Cfg(dict):
    """test_cfg with attribute access (mmengine ConfigDict)."""
    __getattr__ = dict.get


def build_head(fl, mode="window", seed=0, **over):
    import torch
    kw, test_cfg = head_kwargs(mode, **over)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return fl.FovealLaneHead(test_cfg=Cfg(test_cfg), **kw)


def build_net(fl, mode="window", seed=0, width=1.0, **over):
    import torch
    kw, test_cfg = head_kwargs(mode, **over)
    kw["type"] = "FovealLaneHead"
    kw["in_channels"] = [int(c * width) for c in kw["in_channels"]]
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return fl.FovealLaneNet(backbone=dict(type="mmdet.ResNet", depth=34, out_indices=(1, 2, 3), width=width),
                                bbox_head=kw, test_cfg=Cfg(test_cfg))


class Sample:
    def __init__(self, pts, mask=None):
        self.metainfo = {"gt_points": pts}
        if mask is not None:
            self.metainfo["gt_masks"] = mask


def synth_lanes(seed, n_img, vp_x=None):
    """Per image 2-6 lanes converging toward a vanishing point near the top (x = vp_x or random), slightly curved,
    context px, bottom -> top, as flat lists."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_img):
        vx = rng.uniform(250, 550) if vp_x is None else vp_x
        vy = rng.uniform(-10, 30)
        lanes = []
        for _ in range(int(rng.integers(2, 7))):
            xb = rng.uniform(-200, W + 200)
            y_top = vy + rng.uniform(5, 60)
            ys = np.linspace(H + 5, y_top, int(rng.integers(10, 30)))
            t = (H + 5 - ys) / (H + 5 - vy)
            xs = xb + (vx - xb) * t + rng.normal(0, 40) * t * (1 - t)
            lanes.append(np.stack([xs, ys], 1).reshape(-1).tolist())
        out.append(lanes)
    return out
