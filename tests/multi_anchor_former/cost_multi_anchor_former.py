"""GMACs, CPU time and parameters, batch 1, eval: the replica pair vs multi_anchor_former (configs in TREE).
GMACs from torch's FlopCounterMode (convolutions and matrix products). Run: python cost_multi_anchor_former.py TREE"""
import copy, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fnet_common as C  # noqa: E402
fl = C.load(sys.argv[1])
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
import libs.clrbezier.bezierformer as bf  # noqa: E402
import libs.clrbezier.multi_anchor  # noqa: E402,F401
torch.set_num_threads(4)
res = {}
x = torch.randn(1, 3, 640, 1600)
for name, file in (("replica (pair)", "bezierformer_replica_simota_focalcost_flip_curvelanes_r18.py"),
                   ("multi_anchor_former", "multi_anchor_former_curvelanes_r18.py")):
    ns = {}
    exec(open(os.path.join(sys.argv[1], "configs/clrbezier/curvelanes", file)).read(), ns)
    torch.manual_seed(0)
    net = bf.BezierFormerNet(backbone=dict(type="mmdet.ResNet", depth=18, out_indices=(1, 2, 3)),
                             bbox_head=copy.deepcopy(ns["model"]["bbox_head"]), test_cfg=C.Cfg(ns["model"]["test_cfg"])).eval()
    with FlopCounterMode(display=False) as fc:                     # outside no_grad (the module tracker's hooks)
        net._forward(x)
    g = fc.get_total_flops() / 2e9
    with torch.no_grad():
        for _ in range(2):
            net._forward(x)
        t0 = time.perf_counter()
        for _ in range(10):
            net._forward(x)
        ms = 100 * (time.perf_counter() - t0)
    res[name] = dict(gmacs=round(g, 3), cpu_ms=round(ms, 1), params_M=round(sum(p.numel() for p in net.parameters()) / 1e6, 3))
    print(name, json.dumps(res[name]), flush=True)
a, b = res["replica (pair)"], res["multi_anchor_former"]
res["delta"] = {k: round(b[k] - a[k], 3) for k in a}
print("delta", json.dumps(res["delta"]))
json.dump(res, open(os.path.join(HERE, "cost_multi_anchor_former_output.json"), "w"), indent=1)
