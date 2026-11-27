import sys, copy, numpy as np
sys.path.insert(0, ".")
import fnet_common as C
fl = C.load(sys.argv[1])
import torch
import libs.clrbezier.multi_anchor as ma
import libs.clrbezier.bezierformer as bf
ns = {}; exec(open(sys.argv[1] + "/configs/clrbezier/curvelanes/multi_anchor_former_curvelanes_r18.py").read(), ns)
kw = copy.deepcopy(ns["model"]["bbox_head"]); w = 0.25
kw["in_channels"] = [int(c * w) for c in kw["in_channels"]]
torch.manual_seed(0)
net = bf.BezierFormerNet(backbone=dict(type="mmdet.ResNet", depth=18, out_indices=(1, 2, 3), width=w), bbox_head=kw, test_cfg=C.Cfg(ns["model"]["test_cfg"]))
xin = torch.rand(2, 3, 640, 1600)
smp = [C.Sample(p) for p in C.synth_lanes(0, 2, vp_x=380.0)]
net.train()
L = net.loss(xin, smp)
print({k: round(float(v), 3) for k, v in L.items() if not k.startswith("d")})
sum(v for k, v in L.items() if "loss" in k).backward()
print("no grad:", [n for n, p in net.named_parameters() if p.grad is None])
net.eval()
with torch.no_grad(): o = net.bbox_head.forward(net.extract_feat(xin))
print(type(o))
h = net.bbox_head
with torch.no_grad():
    out = h._forward(net.extract_feat(xin))
    f = out["form_curves"][0]
    print("init forms equal:", [float((f[i] - f[1]).abs().max()) for i in range(3)])
