"""CPU tests of MultiAnchorFormerHead (libs/clrbezier/multi_anchor.py) and multi_anchor_former_curvelanes_r18.py.

    python test_multi_anchor_former.py TREE

1. config: = the replica pair (bezierformer_replica_simota_focalcost_curvelanes_r18.py, standard CurveLanes pipeline)
   but the head type, anchor_cfg, cfg_name; the pair = the flip replica config but the data file, cfg_name and the
   batch (16 on 1 GPU)
2. the replica inside: at the same seed every replica parameter is the replica's (B = its own C_0 MLP and update FFNs)
3. geometry: S = CLRNet's line (start + angle) + x offsets; V = Polar R-CNN's normal form about the pole + x offsets;
   at the start all three forms are the replica's fan of lines; with the default pole every fan line passes it (r = 0)
4. fusion: equal at the start; a profile on one form gives that form; output "S" / "V" puts that form on the test path
5. refinement as the replica: each layer's state = the previous one detached + its FFN, so the last layer's loss does
   not reach the earlier update FFNs of any form
6. training (widths 0.25 and 1.0): finite losses, every parameter a gradient; test path (rows and points) finite
7. form loss: loss_form_* scale with form_loss_weight (0 -> 0); with forms ("B",) they are the plain terms x weight
8. config errors
"""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fnet_common as C  # noqa: E402

TREE = sys.argv[1]
fl = C.load(TREE)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import libs.clrbezier.bezierformer as bf  # noqa: E402
import libs.clrbezier.multi_anchor as ma  # noqa: E402

torch.set_num_threads(4)
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({info})" if info else ""), flush=True)


def read_cfg(name):
    ns = {}
    exec(open(os.path.join(TREE, "configs/clrbezier/curvelanes", name)).read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [path]


def build(head, test_cfg, width=0.25, seed=0, **over):
    kw = copy.deepcopy(head)
    for k, v in over.items():
        kw[k] = dict(kw[k], **v) if isinstance(v, dict) and isinstance(kw.get(k), dict) else v
    kw["in_channels"] = [int(c * width) for c in kw["in_channels"]]
    torch.manual_seed(seed)
    return bf.BezierFormerNet(backbone=dict(type="mmdet.ResNet", depth=18, out_indices=(1, 2, 3), width=width),
                              bbox_head=kw, test_cfg=C.Cfg(test_cfg))


def batch(seed=5):
    xin = torch.rand(2, 3, 320, 800, generator=torch.Generator().manual_seed(seed))   # the standard pipeline's input
    return xin, [C.Sample(p) for p in C.synth_lanes(0, 2, vp_x=380.0)]


def run(net, xin, smp, seed=3, keys=None):
    net.train()
    torch.manual_seed(seed)
    L = net.loss(xin, smp)
    sum(v for k, v in L.items() if "loss" in k and (keys is None or k in keys)).backward()
    return L


print("1. config")
new = read_cfg("multi_anchor_former_curvelanes_r18.py")
pair = read_cfg("bezierformer_replica_simota_focalcost_curvelanes_r18.py")
flip = read_cfg("bezierformer_replica_simota_focalcost_flip_curvelanes_r18.py")
d = sorted(diff(pair, new))
check("= the replica pair but the head type, anchor_cfg, cfg_name; both on the standard data file, 16 on 1 GPU",
      d == ["cfg_name", "model.bbox_head.anchor_cfg", "model.bbox_head.type"]
      and new["model"]["bbox_head"]["type"] == "MultiAnchorFormerHead" and new["train_dataloader"]["batch_size"] == 16
      and new["_base_"][0] == "dataset_curvelanes_clrbezier.py", str(d))
d2 = sorted(diff(flip, pair))
check("the pair = the flip replica config but the data file (_base_), cfg_name and train_dataloader.batch_size",
      d2 == ["_base_", "cfg_name", "train_dataloader.batch_size"], str(d2))
HEAD, TCFG = new["model"]["bbox_head"], new["model"]["test_cfg"]
PAIR_HEAD = pair["model"]["bbox_head"]

print("2. the replica inside")
rep, net = build(PAIR_HEAD, TCFG), build(HEAD, TCFG)
s0, s1 = rep.state_dict(), net.state_dict()
extra = sorted(k for k in s1 if k not in s0)
check("same seed: every replica parameter equal; the rest = S / V's C_0 MLPs and update FFNs and the profile",
      all(k in s1 and torch.equal(s0[k], s1[k]) for k in s0)
      and all(k.startswith(("bbox_head.form_init.", "bbox_head.form_reg.", "bbox_head.form_profile")) for k in extra),
      f"{len(s0)} replica tensors, {len(extra)} new")

print("3. geometry")
head = net.bbox_head
w1, h1 = head.img_w - 1.0, head.img_h - 1.0
z = torch.tensor([[[0.4, 0.95, 1.2, 0.3] + [0.0] * head.rows]])
p = head.form_points("S", z)[0, 0] * torch.tensor([w1, h1])
d_ = F.normalize(p[-1] - p[0], dim=-1)
check("a. S: starts at (x_s, y_s), runs at angle theta_s to the x axis (y up), ends on row y_e",
      torch.allclose(p[0], torch.tensor([0.4 * w1, 0.95 * h1]), atol=1e-3)
      and abs(float(torch.atan2(-d_[1], d_[0])) - 1.2) < 1e-4 and abs(float(p[-1, 1]) - 0.3 * h1) < 1e-3)
zv = torch.tensor([[[0.3, 0.05, 0.95, 0.3] + [0.0] * head.rows]])
pv = head.form_points("V", zv)[0, 0] * torch.tensor([w1, h1])
cx, cy = head.pole.tolist()
res = torch.cos(torch.tensor(0.3)) * (pv[:, 0] - cx) - torch.sin(torch.tensor(0.3)) * (pv[:, 1] - cy) - 0.05 * w1
check("b. V: every point on cos(theta)(x - c_x) - sin(theta)(y - c_y) = r about the pole, rows y_s .. y_e",
      float(res.abs().max()) < 1e-2 and abs(float(pv[0, 1]) - 0.95 * h1) < 1e-3, f"max residual {float(res.abs().max()):.1e} px")
zo = zv.clone()
zo[..., 4:] = torch.linspace(0, 0.01, head.rows)
dxo = (head.form_points("V", zo) - head.form_points("V", zv))[0, 0] * torch.tensor([w1, h1])
check("c. x offsets move x only, by dx_j x (img_w - 1)",
      float(dxo[:, 1].abs().max()) == 0.0 and torch.allclose(dxo[:, 0], zo[0, 0, 4:] * w1, atol=1e-3))
xin, smp = batch()
net.eval()
with torch.no_grad():
    out = head._forward(net.extract_feat(xin))
f0 = out["form_curves"][0]
fan = build(HEAD, TCFG, anchor_cfg=dict(pole_px=(399.5, 22.8))).bbox_head.fan_params("V")   # the code's default pole
check("d. at the start all three forms are the replica's fan of lines (config pole); with the default pole every fan "
      "line passes it",
      max(float((f0[i] - f0[1]).abs().max()) for i in range(3)) * w1 < 1e-3 and float(fan[:, 1].abs().max()) < 1e-4,
      f"form difference {max(float((f0[i] - f0[1]).abs().max()) for i in range(3)) * w1:.1e} px, |r| max "
      f"{float(fan[:, 1].abs().max()) * w1:.1e} px")

print("4. fusion")
check("a. the profile starts equal (1/3)", torch.allclose(head._weights(torch.float32), torch.full((3, 73), 1 / 3.0)))
with torch.no_grad():
    head.form_profile[head.forms.index("V")] = 30.0
    out_v = head._forward(net.extract_feat(xin))
    head.form_profile.zero_()
check("b. a profile on V gives V's curve", torch.allclose(out_v["lanes"][-1], out_v["form_curves"][-1][2], atol=1e-6))
torch.manual_seed(1)
for k in ("S", "V"):
    for m in head.form_reg[k]:
        torch.nn.init.normal_(m[-1].weight, std=0.01)
outs = {}
with torch.no_grad():
    for o in ("fused", "S", "V"):
        head.form_output = o
        outs[o] = (head(net.extract_feat(xin))[-1], head._forward(net.extract_feat(xin)))
head.form_output = "fused"
check("c. output 'S' / 'V': that form's rows on the test path, same scores; differs from the fused output",
      all(torch.equal(outs[o][0]["cls_logits"], outs["fused"][0]["cls_logits"]) for o in ("S", "V"))
      and not torch.equal(outs["S"][0]["xs"], outs["fused"][0]["xs"])
      and not torch.equal(outs["V"][0]["xs"], outs["S"][0]["xs"]))

print("5. refinement as the replica")
net = build(HEAD, TCFG)
L = run(net, xin, smp, keys={"loss_loc", "loss_len", "loss_end", "loss_form_loc", "loss_form_len", "loss_form_end"})
h = net.bbox_head
early = [h.reg_heads[0]] + [h.form_reg[k][0] for k in ("S", "V")]
late = [h.reg_heads[-1]] + [h.form_reg[k][-1] for k in ("S", "V")]
g_early = [max(float(p_.grad.abs().max()) if p_.grad is not None else 0.0 for p_ in m.parameters()) for m in early]
g_late = [max(float(p_.grad.abs().max()) if p_.grad is not None else 0.0 for p_ in m.parameters()) for m in late]
check("the last layer's regression losses reach the last update FFNs of B, S, V and none of the first",
      all(g == 0.0 for g in g_early) and all(g > 0 for g in g_late), f"first {g_early}, last {g_late}")

print("6. training")
for width in (0.25, 1.0):
    net = build(HEAD, TCFG, width=width)
    L = run(net, xin, smp)
    none = [n_ for n_, p_ in net.named_parameters() if p_.grad is None]
    fin = all(bool(torch.isfinite(v).all()) for v in L.values())
    net.eval()
    with torch.no_grad():
        rows = net.bbox_head(net.extract_feat(xin))[-1]
        pts = net.bbox_head.point_lanes(net.bbox_head._forward(net.extract_feat(xin)))
    check(f"width {width}: finite losses, every parameter a gradient; rows and points test paths finite",
          fin and not none and all(bool(torch.isfinite(v).all()) for v in rows.values()) and len(pts[0]) == 2,
          f"{len(L)} losses / logs, no grad {none[:3]}")

print("7. form loss")
vals = {}
for wgt in (0.0, 0.5, 1.0):
    net = build(HEAD, TCFG, anchor_cfg=dict(form_loss_weight=wgt))
    net.train()
    torch.manual_seed(3)
    L = net.loss(xin, smp)
    vals[wgt] = tuple(float(L[k]) for k in ("loss_form_loc", "loss_form_len", "loss_form_end"))
check("loss_form_* scale with form_loss_weight (0 -> 0)",
      vals[0.0] == (0.0, 0.0, 0.0) and all(abs(a - 2 * b) <= 1e-4 * max(a, 1e-9) for a, b in zip(vals[1.0], vals[0.5]))
      and vals[1.0][0] > 0, str(vals))
net = build(HEAD, TCFG, anchor_cfg=dict(forms=("B",), form_loss_weight=1.0))
net.train()
torch.manual_seed(3)
L = net.loss(xin, smp)
check("forms ('B',): the fused lane is B, so loss_form_* = loss_loc / _len / _end",
      all(abs(float(L[f"loss_form_{t}"]) - float(L[f"loss_{t}"])) < 1e-5 * max(float(L[f"loss_{t}"]), 1e-9)
          for t in ("loc", "len", "end")))

print("8. config errors")
bad = [dict(anchor_cfg=dict(foo=1)), dict(anchor_cfg=dict(forms=("S", "V"))), dict(anchor_cfg=dict(forms=("B", "X"))),
       dict(anchor_cfg=dict(output="S", forms=("B", "V"))), dict(anchor_cfg=dict(rows=1)),
       dict(anchor_cfg=dict(pole_px=(1.0,)))]
errs = []
for over in bad:
    try:
        build(HEAD, TCFG, **over)
        errs.append(False)
    except ValueError:
        errs.append(True)
check("bad anchor_cfg raises ValueError", all(errs), str(errs))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
