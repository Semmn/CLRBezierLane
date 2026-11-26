"""GMACs, CPU time and parameters, batch 1, eval: foveal_lane_off_bfstyle_o2o_pointcond (the pair) vs
foveal_lane_multi_anchor, heads from the configs in TREE. GMACs from torch's FlopCounterMode (convolutions and matrix
products; grid_sample and the elementwise resampling are not counted). Run: python cost_multi_anchor.py TREE"""
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fnet_common as C  # noqa: E402

fl = C.load(sys.argv[1])
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

torch.set_num_threads(4)


def cfg(name):
    ns = {}
    exec(open(os.path.join(sys.argv[1], "configs/clrbezier/curvelanes", name)).read(), ns)
    return ns["model"]["bbox_head"], ns["model"]["test_cfg"]


def gmacs(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 2e9


x = torch.randn(1, 3, 640, 1600)
res = {}
for name, file in (("pointcond (pair)", "foveal_lane_off_bfstyle_o2o_pointcond_curvelanes_r34.py"),
                   ("multi_anchor", "foveal_lane_multi_anchor_curvelanes_r34.py")):
    head, tc = cfg(file)
    torch.manual_seed(0)
    net = fl.FovealLaneNet(backbone=dict(type="mmdet.ResNet", depth=34, out_indices=(1, 2, 3)),
                           bbox_head=copy.deepcopy(head), test_cfg=C.Cfg(tc)).eval()
    with torch.no_grad():
        total = gmacs(lambda: net._forward(x))
        for _ in range(2):
            net._forward(x)
        n = 10
        t0 = time.perf_counter()
        for _ in range(n):
            net._forward(x)
        ms = 1000 * (time.perf_counter() - t0) / n
    res[name] = dict(gmacs=round(total, 3), cpu_ms=round(ms, 1),
                     params_M=round(sum(p.numel() for p in net.parameters()) / 1e6, 3))
    print(name, json.dumps(res[name]), flush=True)
a, b = res["pointcond (pair)"], res["multi_anchor"]
res["delta"] = dict(gmacs=round(b["gmacs"] - a["gmacs"], 3), cpu_ms=round(b["cpu_ms"] - a["cpu_ms"], 1),
                    params_M=round(b["params_M"] - a["params_M"], 3))
print("delta", json.dumps(res["delta"]))
json.dump(res, open(os.path.join(HERE, "cost_multi_anchor_output.json"), "w"), indent=1)
