"""Overfit check of the multi-anchor BezierFormer (multi_anchor_former_curvelanes_r18.py) on the 2 synthetic images of
bf_replica's overfit test (clrbezierlane/bf_replica/tests/test_bf_replica_overfit.py, the replica with cls_cost focal:
5/5 there), CPU: the same small head (dim 64, 1 encoder layer, 6 decoder layers, quarter-width ResNet-18), AdamW on
the same batch, seed and steps.

    python test_multi_anchor_former_overfit.py TREE [STEPS [SEED]]

Passes (as the replica's) if the total loss falls below 35% of its start, every layer's L_loc falls below half its
start, every GT lane has a query at the end (match_frac 1), and on the training images (test path) every GT lane has
a prediction with score > 0.5 and LaneIoU >= 0.5 while at most one extra lane scores > 0.5 per image, for the fused
output. Each form alone (S, B, V) is reported the same way (not a pass criterion)."""
import copy
import json
import os
import sys
import time

import cv2
import numpy as np

import fnet_common as C

fl = C.load(sys.argv[1])
import torch  # noqa: E402
import libs.clrbezier.bezierformer as bf  # noqa: E402
import libs.clrbezier.multi_anchor  # noqa: E402,F401
from libs.clrbezier.hblrnet import bottom_first, gt_polyline, lane_iou_2d  # noqa: E402

STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
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
        for p in lanes:
            cv2.polylines(img, [np.round(p * 2 * 16).astype(np.int32)], False, (255, 255, 255), 5, cv2.LINE_AA, 4)
        imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        samples.append(C.Sample([p.reshape(-1).tolist() for p in lanes]))
    return torch.stack(imgs), samples


x2, samples = images()
hj = {}
exec(open(os.path.join(sys.argv[1], "configs/clrbezier/curvelanes/multi_anchor_former_curvelanes_r18.py")).read(), hj)
hj = dict(bbox_head=hj["model"]["bbox_head"], test_cfg=hj["model"]["test_cfg"])
kw = copy.deepcopy(hj["bbox_head"])
kw.pop("type")
kw.update(dim=64, in_channels=[32, 64, 128])
kw["enc_cfg"] = dict(kw["enc_cfg"], num_layers=1, num_heads=4, ffn_dim=256, dropout=0.0)
kw["dec_cfg"] = dict(kw["dec_cfg"], sa_heads=4, attn_heads=4, ffn_dim=256, dropout=0.0, init_hidden=256)
if os.environ.get("FORMS"):                                           # e.g. FORMS=B or FORMS=S,B (diagnosis)
    kw["anchor_cfg"] = dict(kw["anchor_cfg"], forms=tuple(os.environ["FORMS"].split(",")))
    print("forms:", kw["anchor_cfg"]["forms"])
if os.environ.get("ROWS"):                                            # diagnosis: number of S / V offset rows
    kw["anchor_cfg"] = dict(kw["anchor_cfg"], rows=int(os.environ["ROWS"]))
torch.manual_seed(SEED)
net = bf.BezierFormerNet(backbone=dict(type="mmdet.ResNet", depth=18, out_indices=(1, 2, 3), width=0.25),
                         bbox_head=dict(type="MultiAnchorFormerHead", **kw), test_cfg=C.Cfg(hj["test_cfg"]))
# the metric's LaneIoU geometry, from the flip head (as the other overfit tests read it)
with open(os.path.join(HERE, "heads_flip.json")) as f:
    fk = json.load(f)["bbox_head"]
fk.pop("type")
fk.update(dim=64, in_channels=[32, 64, 128])
ref_head = fl.FovealLaneHead(test_cfg=C.Cfg(hj["test_cfg"]), **fk)
opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": 3e-4},
                         {"params": net.bbox_head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
net.train()
t0 = time.time()
hist = []
for it in range(STEPS):
    L = net.loss(x2, samples)
    tot = sum(v for k, v in L.items() if "loss" in k)                   # as mmengine (d*.loss_* too)
    opt.zero_grad()
    tot.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 35.0)
    opt.step()
    hist.append(float(tot.detach()))
    if it == 0:
        start = {k: float(v.detach()) for k, v in L.items() if k.endswith("loss_loc")}
    if it % 50 == 0 or it == STEPS - 1:
        keys = ("loss_cls", "loss_loc", "loss_form_loc", "d0.loss_loc", "ciou", "form_ciou_S", "form_ciou_B",
                "form_ciou_V", "match_frac", "pos_prob", "neg_prob", "form_w_S_near", "form_w_B_near",
                "form_w_V_near", "form_w_V_far")
        print(f"it {it:4d}  total {float(tot):.3f}  " + "  ".join(f"{k} {float(L[k]):.3f}" for k in keys if k in L)
              + f"  ({time.time() - t0:.0f} s)", flush=True)
first, last = np.mean(hist[:5]), np.mean(hist[-10:])
check("total loss falls below 35% of its start", last < 0.35 * first, f"{first:.2f} -> {last:.2f}")
end = {k: float(L[k]) for k in start}
check("every layer's L_loc falls below half its start (d0 = C_0 .. d5, last)",
      all(end[k] < 0.5 * start[k] for k in start),
      ", ".join(f"{k} {start[k]:.2f} -> {end[k]:.2f}" for k in sorted(start)))
check("every GT lane has a query at the end (match_frac 1)", float(L["match_frac"]) > 0.999,
      f"match_frac {float(L['match_frac']):.2f}, ciou {float(L['ciou']):.3f}")

net.eval()
head = net.bbox_head
for output in ("fused",) + head.forms:
    head.form_output = output
    with torch.no_grad():
        out = head._forward(net.extract_feat(x2))
        score = out["logits"][-1].sigmoid()
        curves = bottom_first(head._output_lanes(out))
    ok_all, extra, names = True, [], ["left", "centre", "right", "flat side", "far short"]
    print(f"output {output}:")
    for b, s in enumerate(samples):
        gts = [torch.tensor(gt_polyline(ln, head.img_w, head.img_h, 48)) for ln in s.metainfo["gt_points"]]
        keep = score[b] > 0.5
        matched, found = set(), []
        for g in gts:
            iou = (lane_iou_2d(curves[b][keep] * ref_head.eval_scale, g[None].expand(int(keep.sum()), -1, -1)
                               * ref_head.eval_scale, ref_head.lane_width * ref_head.iou_w)
                   if keep.any() else torch.zeros(0))
            hit = torch.nonzero(iou >= 0.5)[:, 0].tolist()
            ok_all &= bool(hit)
            found.append(f"{float(iou.max()) if iou.numel() else 0:.2f}")
            matched |= set(hit[:1])
        extra.append(int(keep.sum()) - len(matched))
        print(f"image {b}: {int(keep.sum())} lanes > 0.5; best LaneIoU per GT lane "
              + ", ".join(f"{n} {v}" for n, v in zip(names, found)))
    if output == "fused":
        check("test path (fused): every GT lane found (score > 0.5, LaneIoU >= 0.5), incl. the flat side and the far "
              "lane", ok_all)
        check("test path (fused): at most one extra lane per image", max(extra) <= 1, str(extra))
    else:
        print(f"  {output} alone (reported): every lane found {ok_all}, extra {extra}")
head.form_output = "fused"
print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
