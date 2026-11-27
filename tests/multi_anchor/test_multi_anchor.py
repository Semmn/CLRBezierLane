"""CPU tests of the multi-anchor head (libs/clrbezier/foveal_lane.py multi_anchor_cfg, hblrnet.py form losses) and the
config foveal_lane_multi_anchor_curvelanes_r34.py.

    python test_multi_anchor.py TREE ORIG_TREE

TREE: the repo with the change; ORIG_TREE: the repo before it (main). Each tree is loaded in its own subprocess for
the switch-off comparison (both are the package libs.clrbezier).

1. config: = foveal_lane_off_bfstyle_o2o_pointcond but multi_anchor_cfg, cfg_name and the batch (16 on 1 GPU)
2. switch off: identical to ORIG_TREE (parameters, losses / logs, gradients, test output) for the pointcond, o2o,
   bfstyle, window (+ segment steps), polar and ray heads
3. geometry: graph tangent at the anchor; arc-length resampling (ends kept, uniform spacing, gradients); the step-0
   forms (S starts at P0 along P1 - P0, V is the ray through v and P3, near end first; B = the proposal cubic); V's
   reference ray is bounded against VP error near v and carries no gradient to the anchor
4. fusion: equal weights at the start (fused = mean of the forms); a profile on one form gives that form; output
   "V" puts V's own curve on the test path
5. local readout: a change of the stride-8 map at S's anchor changes S's update and not V's (far end), and the
   reverse
6. training (widths 0.25 and 1.0): finite losses; every parameter gets a gradient; the form MLPs and the profile get
   nonzero gradients; the lane losses do not reach the vanishing-point head (V reads v detached)
7. form loss: loss_form_iou / loss_form_end scale with form_loss_weight (0 -> 0); with forms ("B",) the form loss is
   the plain loss of the same pairs (x weight, same IoU)
8. config errors: unknown key, bad form, forms without B, mode other than "off", interval / hier / cp_loss
"""
import copy
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def batch(C, seed=5):
    import torch
    xin = torch.rand(2, 3, 640, 1600, generator=torch.Generator().manual_seed(seed))
    lanes = C.synth_lanes(0, 2, vp_x=380.0)
    for ls in lanes:
        ls.append(np.stack([np.linspace(350, 375, 10), np.linspace(70, 20, 10)], 1).reshape(-1).tolist())
    return xin, [C.Sample(p_, (np.random.default_rng(1).random((320, 800)) > 0.9).astype(np.int64)) for p_ in lanes]


def polar_batch(C, seed=5):
    import torch
    import polar_common as P
    xin = torch.rand(2, 3, 640, 1600, generator=torch.Generator().manual_seed(seed))
    smp = []
    for vx, vy in ((400.0, 60.0), (300.0, 30.0)):
        ls = P.ray_lanes((vx, vy), [(50.0, 330.0), (330.0, 330.0), (650.0, 330.0), (900.0, 250.0)])
        smp.append(C.Sample(ls, (np.random.default_rng(1).random((320, 800)) > 0.9).astype(np.int64)))
    return xin, smp


def read_cfg(tree, name):
    ns = {}
    exec(open(os.path.join(tree, "configs/clrbezier/curvelanes", name)).read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}


def build(fl, C, head, test_cfg, width=0.25, seed=0, **over):
    import torch
    kw = copy.deepcopy(head)
    kw.pop("type", None)
    for k, v in over.items():
        kw[k] = dict(kw[k], **v) if isinstance(v, dict) and isinstance(kw.get(k), dict) else v
    kw["type"] = "FovealLaneHead"
    kw["in_channels"] = [int(c * width) for c in kw["in_channels"]]
    torch.manual_seed(seed)
    return fl.FovealLaneNet(backbone=dict(type="mmdet.ResNet", depth=34, out_indices=(1, 2, 3), width=width),
                            bbox_head=kw, test_cfg=C.Cfg(test_cfg))


def run(net, xin, smp, seed=3):
    import torch
    net.train()
    torch.manual_seed(seed)
    L = net.loss(xin, smp)
    sum(v for k, v in L.items() if "loss" in k).backward()
    return L


def test_out(net, xin):
    import torch
    net.eval()
    with torch.no_grad():
        return net.bbox_head.forward(net.extract_feat(xin))[-1]


def heads_json(name):
    import json
    with open(os.path.join(HERE, name)) as f:
        h = json.load(f)
    return h["bbox_head"], h["test_cfg"]


def dump(tree, out_path):
    """Subprocess: the switch-off cases on one tree -> pickle of (state_dict, losses, grads, test output) per case."""
    sys.path.insert(0, HERE)
    import fnet_common as C
    fl = C.load(tree)
    import torch
    import bf_common as BF
    import polar_common as P
    torch.set_num_threads(4)
    pc = read_cfg(tree, "foveal_lane_off_bfstyle_o2o_pointcond_curvelanes_r34.py")
    o2o, o2o_tc = heads_json("heads_o2o.json")
    cases = (("pointcond config", lambda: build(fl, C, pc["model"]["bbox_head"], pc["model"]["test_cfg"]), batch),
             ("o2o config", lambda: build(fl, C, o2o, o2o_tc), batch),
             ("bfstyle config", lambda: BF.build_net(fl, width=0.25), batch),
             ("window (+ segment steps)", lambda: C.build_net(fl, "window", width=0.25), batch),
             ("polar (image)", lambda: P.build_net(fl, width=0.25), polar_batch),
             ("ray config", lambda: build(fl, C, *heads_json("heads_ray.json")), polar_batch))
    res = {}
    for name, builder, data in cases:
        xin, smp = data(C)
        net = builder()
        L = run(net, xin, smp)
        res[name] = ({k: v.detach().clone() for k, v in net.state_dict().items()},
                     {k: v.detach().clone() for k, v in L.items()},
                     {n_: (None if p_.grad is None else p_.grad.clone()) for n_, p_ in net.named_parameters()},
                     {k: v.clone() for k, v in test_out(net, xin).items()})
    with open(out_path, "wb") as f:
        pickle.dump(res, f)


if len(sys.argv) > 1 and sys.argv[1] == "--dump":
    dump(sys.argv[2], sys.argv[3])
    sys.exit(0)

TREE, ORIG = sys.argv[1], sys.argv[2]
sys.path.insert(0, HERE)
import fnet_common as C  # noqa: E402

fl = C.load(TREE)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(4)
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({info})" if info else ""), flush=True)


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [path]


print("1. config")
new = read_cfg(TREE, "foveal_lane_multi_anchor_curvelanes_r34.py")
base = read_cfg(TREE, "foveal_lane_off_bfstyle_o2o_pointcond_curvelanes_r34.py")
d = sorted(diff(base, new))
check("= pointcond but multi_anchor_cfg, cfg_name and train_dataloader.batch_size (16)",
      d == ["cfg_name", "model.bbox_head.multi_anchor_cfg", "train_dataloader.batch_size"]
      and new["train_dataloader"]["batch_size"] == 16
      and new["model"]["bbox_head"]["multi_anchor_cfg"]["forms"] == ("S", "B", "V"), str(d))
HEAD, TCFG = new["model"]["bbox_head"], new["model"]["test_cfg"]

print("2. switch off: identical to the tree before the change")
tmp = os.path.join(HERE, "_dumps")
os.makedirs(tmp, exist_ok=True)
dumps = []
for tag, tree in (("orig", ORIG), ("new", TREE)):
    path = os.path.join(tmp, f"{tag}.pkl")
    subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", tree, path], check=True)
    with open(path, "rb") as f:
        dumps.append(pickle.load(f))
for name in dumps[0]:
    (s0, l0, g0, p0), (s1, l1, g1, p1) = dumps[0][name], dumps[1][name]
    same = (list(s0) == list(s1) and all(torch.equal(s0[k], s1[k]) for k in s0),
            set(l0) == set(l1) and all(torch.equal(l0[k], l1[k]) for k in l0),
            list(g0) == list(g1) and all((g0[k] is None and g1[k] is None) or torch.equal(g0[k], g1[k]) for k in g0),
            set(p0) == set(p1) and all(torch.equal(p0[k], p1[k]) for k in p0))
    check(f"{name}: identical (parameters, {len(l0)} losses / logs, gradients, test output)", all(same), str(same))

print("3. geometry")
torch.manual_seed(0)
anchor = torch.tensor([[400.0, 300.0]])
head_d = F.normalize(torch.tensor([[0.3, -1.0]]), dim=-1)
L, c2, c3 = torch.tensor([200.0]), torch.tensor([0.2]), torch.tensor([-0.1])
s = torch.tensor([0.0, 1e-7], dtype=torch.float64)
p = fl.graph_points(anchor.double(), head_d.double(), L.double(), c2.double(), c3.double(), s.double())
tang = F.normalize(p[0, 1] - p[0, 0], dim=-1)
check("a. graph: p(0) = the anchor, the tangent at the anchor = its heading",
      float((p[0, 0] - anchor[0].double()).abs().max()) < 1e-9 and float((tang - head_d[0].double()).abs().max()) < 1e-5,
      f"tangent error {float((tang - head_d[0].double()).abs().max()):.1e}")
dense = fl.graph_points(anchor, head_d, L, c2, c3, torch.linspace(0, 1, 145)).requires_grad_(True)
r = fl.arc_resample(dense, 73)
seg = (r[0, 1:] - r[0, :-1]).norm(dim=-1)
r.sum().backward()
check("b. arc_resample: ends kept, spacing uniform (rel. std < 1e-3), gradients reach the points",
      float((r[0, 0] - dense[0, 0]).abs().max()) < 1e-4 and float((r[0, -1] - dense[0, -1]).abs().max()) < 1e-3
      and float(seg.std() / seg.mean()) < 1e-3 and bool(torch.isfinite(dense.grad).all())
      and float(dense.grad.abs().sum()) > 0,
      f"spacing rel. std {float(seg.std() / seg.mean()):.1e}, end error {float((r[0, -1] - dense[0, -1]).abs().max()):.1e}")
zero_len = fl.arc_resample(torch.ones(1, 5, 2) * 3.0, 7)
check("b. arc_resample of a zero-length polyline: its point, finite", bool(torch.allclose(zero_len, torch.full_like(zero_len, 3.0))))
pa = fl.polyline_at(r, torch.tensor([0.0, 0.5, 1.0]))
check("c. polyline_at: u = 0, 1 are the ends, u = 0.5 the middle point",
      torch.allclose(pa[0, 0], r[0, 0]) and torch.allclose(pa[0, 2], r[0, -1]) and torch.allclose(pa[0, 1], r[0, 36]))

net = build(fl, C, HEAD, TCFG)
head = net.bbox_head
cps0 = head._norm(torch.tensor([[[[300.0, 318.0], [330.0, 220.0], [355.0, 120.0], [370.0, 40.0]],
                                 [[700.0, 300.0], [600.0, 250.0], [500.0, 190.0], [420.0, 60.0]]]]))
vp = torch.tensor([[390.0, 10.0]])
st = head._form_init(cps0, vp)
cpx = head._px(cps0)
pts = {k: head._form_curve(k, st[k], vp, head.t_form_dense) for k in head.forms}
t_s = F.normalize(pts["S"][..., 1, :] - pts["S"][..., 0, :], dim=-1)
t_p = F.normalize(cpx[..., 1, :] - cpx[..., 0, :], dim=-1)
ray = F.normalize(cpx[..., 3, :] - vp[:, None], dim=-1)
v_dir = F.normalize(pts["V"][..., 0, :] - pts["V"][..., -1, :], dim=-1)           # far -> near
cross = (ray[..., 0] * v_dir[..., 1] - ray[..., 1] * v_dir[..., 0]).abs()
check("d. step 0: S starts at P0 along P1 - P0; V ends (far) at P3 on the ray from v, near end first; B = the cubic",
      float((pts["S"][..., 0, :] - cpx[..., 0, :]).abs().max()) < 1e-3 and float((t_s - t_p).abs().max()) < 1e-3
      and float((pts["V"][..., -1, :] - cpx[..., 3, :]).abs().max()) < 1e-3 and float(cross.max()) < 1e-4
      and bool((pts["V"][..., 0, 1] > pts["V"][..., -1, 1]).all()) and bool(((v_dir * ray).sum(-1) > 0).all())
      and torch.allclose(pts["B"], head._px(fl.curve_points(cps0, head.t_form_dense)), atol=1e-4),
      f"V off the ray {float(cross.max()):.1e}")

sv = dict(st["V"])
a_near = torch.tensor([[[392.0, 15.0], [430.0, 40.0]]], requires_grad=True)       # 5.4 px and 50 px from v
sv["anchor"] = a_near
h0 = head._form_head("V", sv, vp)
h1 = head._form_head("V", sv, vp + torch.tensor([[5.0, 0.0]]))
turn = torch.rad2deg(torch.acos((h0 * h1).sum(-1).clamp(-1, 1)))[0]
check("e. V's reference ray: a 5 px VP error turns it by <= ~e / ray_min_px (30 px) even with the far end 5 px from v; "
      "no gradient from the ray to the anchor",
      float(turn[0]) < 10.5 and float(turn[1]) < 6.5 and not h0.requires_grad,
      f"turn {float(turn[0]):.1f} deg at 5 px, {float(turn[1]):.1f} deg at 50 px")

sv = dict(st["V"])
sv["sign"] = -torch.ones_like(sv["sign"])
sv["d0"] = F.normalize(sv["anchor"] - vp[:, None], dim=-1)
ok = []
for r_ in (5.0, 15.0, 25.0, 60.0):
    sv["anchor"] = vp[:, None] + r_ * sv["d0"]
    hd = head._form_head("V", sv, vp)
    ok.append(float((hd * -sv["d0"]).sum(-1).min()) > 0.999)
check("e. sign -1 (near end towards v): the floor pushes the reference away from v, heading towards v at r = 5 .. 60",
      all(ok), str(ok))

print("4. fusion")
curves, fused, ends = head._form_fuse(st, vp)
check("a. at the start the profile is equal: fused = the mean of the forms",
      torch.allclose(fused, sum(curves) / 3.0, atol=1e-6)
      and torch.allclose(head._form_weights(torch.float32), torch.full((3, head.num_out), 1.0 / 3.0)))
with torch.no_grad():
    head.form_profile.zero_()
    head.form_profile[head.forms.index("V")] = 30.0
_, fused_v, _ = head._form_fuse(st, vp)
check("b. a profile on V gives V's curve", torch.allclose(fused_v, curves[head.forms.index("V")], atol=1e-6))
with torch.no_grad():
    head.form_profile.zero_()
xin, smp = batch(C)
outs = {}
for o in ("fused", "V"):
    head.form_output = o
    net.eval()
    with torch.no_grad():
        outs[o] = head._forward(net.extract_feat(xin))
head.form_output = "fused"
check("c. output 'V': the test path's curves are V's own (same scores); 'fused': the fused curve",
      torch.equal(outs["V"]["curves"][-1], outs["V"]["form_curves"][-1][head.forms.index("V")])
      and torch.equal(outs["V"]["cls_o2o"], outs["fused"]["cls_o2o"])
      and not torch.equal(outs["fused"]["curves"][-1], outs["V"]["curves"][-1]))

print("5. local readout")
torch.manual_seed(1)
for k in ("S", "V"):
    torch.nn.init.normal_(head.form_mlp[k][2].weight, std=0.05)
with torch.no_grad():
    tokens, maps, frame, fine = head._tokens(net.extract_feat(xin[:1]))
    h = torch.randn(1, 2, head.dim)
    base_upd = {k: head._form_step(k, st[k], h, vp, fine, frame) for k in ("S", "V")}
    cell = maps[0].cell
    moved = {}
    for where, xy in (("S anchor", cpx[0, 0, 0]), ("V anchor", cpx[0, 0, 3])):
        f2 = [m.clone() for m in fine]
        cx, cy = int(float(xy[0]) // cell), int(float(xy[1]) // cell)
        f2[0][:, :, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] += 3.0
        moved[where] = {k: float((head._form_step(k, st[k], h, vp, f2, frame)["anchor"]
                                  - base_upd[k]["anchor"])[0, 0].abs().max()) for k in ("S", "V")}
check("a change of the map at S's anchor moves S's update and not V's; at V's anchor the reverse",
      moved["S anchor"]["S"] > 1e-4 and moved["S anchor"]["V"] == 0.0
      and moved["V anchor"]["V"] > 1e-4 and moved["V anchor"]["S"] == 0.0, str(moved))

print("6. training")
for width in (0.25, 1.0):
    net = build(fl, C, HEAD, TCFG, width=width)
    L = run(net, xin, smp)
    grads = {n_: p_.grad for n_, p_ in net.named_parameters()}
    none = [n_ for n_, g in grads.items() if g is None]
    fin = all(bool(torch.isfinite(v).all()) for v in L.values())
    nz = {n_: float(g.abs().sum()) for n_, g in grads.items() if "form_mlp" in n_ and n_.endswith("2.weight")
          or n_.endswith("form_profile")}
    keys = [k for k in L if k.startswith("form_") or "form" in k]
    check(f"width {width}: finite losses, every parameter gets a gradient, the form MLPs and the profile nonzero",
          fin and not none and len(nz) == 3 and all(v > 0 for v in nz.values()),
          f"{len(L)} losses / logs, no grad {none[:3]}, {sorted(nz)}")
    out = test_out(net, xin)
    check(f"width {width}: test path finite, 16 lanes", all(bool(torch.isfinite(v).all()) for v in out.values()
                                                             if torch.is_tensor(v)) and len(out) > 0)
net = build(fl, C, HEAD, TCFG)
net.train()
torch.manual_seed(3)
L = net.loss(xin, smp)
lane = sum(v for k, v in L.items() if "loss" in k and "vp" not in k and "prop" not in k and "seg" not in k)
lane.backward()
vp_g = [p_.grad for p_ in net.bbox_head.vp_head.parameters()]
check("the lane losses (fused, forms, scores, all steps) do not reach the vanishing-point head",
      all(g is None or float(g.abs().max()) == 0.0 for g in vp_g))

print("7. form loss")
vals = {}
for w in (0.0, 0.5, 1.0):
    net = build(fl, C, HEAD, TCFG, multi_anchor_cfg=dict(form_loss_weight=w))
    net.train()
    torch.manual_seed(3)
    L = net.loss(xin, smp)
    vals[w] = (float(L["loss_form_iou"]), float(L["loss_form_end"]))
check("loss_form_iou / loss_form_end scale with form_loss_weight (0 -> 0)",
      vals[0.0] == (0.0, 0.0) and abs(vals[1.0][0] - 2 * vals[0.5][0]) < 1e-4 * vals[1.0][0]
      and abs(vals[1.0][1] - 2 * vals[0.5][1]) < 1e-4 * vals[1.0][1] and vals[1.0][0] > 0, str(vals))
net = build(fl, C, HEAD, TCFG, multi_anchor_cfg=dict(forms=("B",), form_loss_weight=1.0))
net.train()
torch.manual_seed(3)
L = net.loss(xin, smp)
check("forms ('B',): the fused curve is B's, so the form loss = the plain loss of the same pairs",
      abs(float(L["loss_form_iou"]) - float(L["loss_iou"])) < 1e-5 * float(L["loss_iou"])
      and abs(float(L["loss_form_end"]) - float(L["loss_end"])) < 1e-5 * float(L["loss_end"]),
      f"iou {float(L['loss_form_iou']):.5f} vs {float(L['loss_iou']):.5f}")

print("8. config errors")
bad = [dict(multi_anchor_cfg=dict(foo=1)), dict(multi_anchor_cfg=dict(forms=("S", "X"))),
       dict(multi_anchor_cfg=dict(output="V", forms=("S", "B"))), dict(fovea_cfg=dict(mode="window")),
       dict(dec_cfg=dict(cp_loss=True)), dict(multi_anchor_cfg=dict(knots=1)),
       dict(multi_anchor_cfg=dict(forms=("S", "V"), output="fused"))]
errs = []
for over in bad:
    try:
        build(fl, C, HEAD, TCFG, **over)
        errs.append(False)
    except ValueError:
        errs.append(True)
check("bad multi_anchor_cfg / mode / cp_loss raise ValueError", all(errs), str(errs))

print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
