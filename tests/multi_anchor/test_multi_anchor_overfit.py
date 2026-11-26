"""Overfit check of the multi-anchor head (foveal_lane_multi_anchor_curvelanes_r34.py) on 2 synthetic images, CPU, as
point_cond/tests/test_pointcond_overfit.py: the small head (dim 64, 1 encoder layer, quarter-width ResNet-34), AdamW
on the same batch. The lanes are drawn into the image;
each image has a flat side lane (slant > 4, the kind detectors lose) and a short far lane next to the vanishing point.

    python test_multi_anchor_overfit.py TREE [STEPS]

Passes if the total loss falls below 35% of its start, every step's LaneIoU loss and form loss falls (d0 .. d4 and the
last), and on the training images (test path, o2o scores) every GT lane has a prediction with score > 0.5 and 2D
LaneIoU >= 0.5 while at most one extra lane scores > 0.5 per image: for the fused output and for each form alone
(S, B, V: the same network, multi_anchor_cfg.output)."""
import sys
import time

import cv2
import numpy as np

import fnet_common as C
import copy
import json
import os

fl = C.load(sys.argv[1])
import torch  # noqa: E402
from libs.clrbezier.hblrnet import gt_polyline, lane_iou_2d  # noqa: E402

STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
torch.set_num_threads(4)
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({info})" if info else ""), flush=True)


def images(seed=0):
    rng = np.random.default_rng(seed)
    imgs, samples = [], []
    for b, (vx, vy) in enumerate(((360.0, 30.0), (450.0, 45.0))):
        lanes = []
        for xb in (60.0, 330.0, 620.0):
            ys = np.linspace(325, vy + 15, 25)
            t = (325 - ys) / (325 - vy)
            lanes.append(np.stack([xb + (vx - xb) * t + 30 * (b - 0.5) * t * (1 - t), ys], 1))
        s = np.linspace(0.0, 1.0, 20)[:, None]
        lanes.append(np.array([[799.0, 120.0]]) + s * np.array([[vx + 50 - 799.0, vy + 8 - 120.0]]))   # flat side
        lanes.append(np.stack([np.linspace(vx - 30, vx - 8, 10), np.linspace(vy + 45, vy + 10, 10)], 1))  # far, short
        img = (rng.random((640, 1600, 3)) * 40).astype(np.uint8)
        mask = np.zeros((320, 800), np.uint8)
        for p in lanes:
            cv2.polylines(img, [np.round(p * 2 * 16).astype(np.int32)], False, (255, 255, 255), 5, cv2.LINE_AA, 4)
            cv2.polylines(mask, [np.round(p * 16).astype(np.int32)], False, 1, 3, cv2.LINE_8, 4)
        imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        samples.append(C.Sample([p.reshape(-1).tolist() for p in lanes], mask.astype(np.int64)))
    return torch.stack(imgs), samples


x2, samples = images()
slants = [abs(p[0, 0] - p[-1, 0]) / abs(p[0, 1] - p[-1, 1])
          for s in samples for p in (np.asarray(s.metainfo["gt_points"][3]).reshape(-1, 2),)]
print(f"flat side lanes: slant {[round(v, 1) for v in slants]}")
hj = {}
exec(open(sys.argv[1] + "/configs/clrbezier/curvelanes/foveal_lane_multi_anchor_curvelanes_r34.py").read(), hj)
hj = dict(bbox_head=hj["model"]["bbox_head"], test_cfg=hj["model"]["test_cfg"])
kw = copy.deepcopy(hj["bbox_head"])
kw.update(dim=64, in_channels=[32, 64, 128])
kw["enc_cfg"] = dict(kw["enc_cfg"], num_layers=1, num_heads=4, ffn_dim=256)
kw["dec_cfg"] = dict(kw["dec_cfg"], num_heads=4, deform_heads=4, ffn_dim=256)
kw["prop_cfg"] = dict(kw["prop_cfg"], channels=32)
kw["vp_cfg"] = dict(kw["vp_cfg"], channels=32)
torch.manual_seed(0)
net = fl.FovealLaneNet(backbone=dict(type="mmdet.ResNet", depth=34, out_indices=(1, 2, 3), width=0.25), bbox_head=kw,
                       test_cfg=C.Cfg(hj["test_cfg"]))
opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": 3e-4},
                         {"params": net.bbox_head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
net.train()
t0 = time.time()
hist = []
for it in range(STEPS):
    L = net.loss(x2, samples)
    tot = sum(v for k, v in L.items() if "loss" in k)                   # as mmengine (d0.loss_* too)
    opt.zero_grad()
    tot.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 35.0)
    opt.step()
    hist.append(float(tot.detach()))
    if it == 0:
        start = {k: float(v.detach()) for k, v in L.items() if k.endswith("loss_iou") or k.endswith("loss_form_iou")}
    if it % 50 == 0 or it == STEPS - 1:
        keys = ("loss_cls", "loss_iou", "loss_form_iou", "d0.loss_iou", "d4.loss_iou", "form_iou_S", "form_iou_B",
                "form_iou_V", "form_w_S_near", "form_w_B_near", "form_w_V_near", "form_w_V_far", "loss_vp",
                "vp_err_px", "vp_err_y_px", "prop_recall")
        print(f"it {it:4d}  total {float(tot):.3f}  " + "  ".join(f"{k} {float(L[k]):.3f}" for k in keys if k in L)
              + f"  ({time.time() - t0:.0f} s)", flush=True)
first, last = np.mean(hist[:5]), np.mean(hist[-10:])
check("total loss falls below 35% of its start", last < 0.35 * first, f"{first:.2f} -> {last:.2f}")
end = {k: float(L[k]) for k in start}
check("every step's LaneIoU and form loss falls below half its start (d0 .. d4, last)", all(end[k] < 0.5 * start[k] for k in start),
      ", ".join(f"{k} {start[k]:.2f} -> {end[k]:.2f}" for k in sorted(start)))

net.eval()
head = net.bbox_head
for output in ("fused", "S", "B", "V"):
  head.form_output = output
  with torch.no_grad():
    xf = net.extract_feat(x2)
    out = head._forward(xf)
    score = torch.softmax(out["cls_o2o"], -1)[..., 1]
    curves = out["curves"][-1]
  ok_all, extra, names = True, [], ["left", "centre", "right", "flat side", "far short"]
  print(f"output {output}:")
  for b, s in enumerate(samples):
      gts = [torch.tensor(gt_polyline(ln, head.img_w, head.img_h, 48)) for ln in s.metainfo["gt_points"]]
      keep = score[b] > 0.5
      matched, found = set(), []
      for g in gts:
          iou = lane_iou_2d(curves[b][keep] * head.eval_scale, g[None].expand(int(keep.sum()), -1, -1) * head.eval_scale,
                            head.lane_width * head.iou_w) if keep.any() else torch.zeros(0)
          hit = torch.nonzero(iou >= 0.5)[:, 0].tolist()
          ok_all &= bool(hit)
          found.append(f"{float(iou.max()) if iou.numel() else 0:.2f}")
          matched |= set(hit[:1])
      extra.append(int(keep.sum()) - len(matched))
      print(f"image {b}: {int(keep.sum())} lanes > 0.5; best LaneIoU per GT lane "
            + ", ".join(f"{n} {v}" for n, v in zip(names, found)))
  check(f"test path, {output}: every GT lane found (score > 0.5, LaneIoU >= 0.5), incl. the flat side and the far lane",
        ok_all)
  check(f"test path, {output}: at most one extra lane per image", max(extra) <= 1, str(extra))
print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
