"""What is the probe's ranking advantage actually worth in F1?

The probe result says the RoI features carry ordering information the
classifier does not use — on the one-to-many model, +0.05 pairwise accuracy on
the decisive pairs (one member above IoU 0.5, the other below). That bounds what
a ranking loss can buy, but it does not convert to F1 on its own: a decisive
pair only matters when getting it wrong actually flips the NMS winner for that
cluster, and clusters contain many pairs.

This converts it directly. It runs one inference pass, and decodes it twice:

* once with the model's own ``cls_logits`` — the baseline,
* once with those logits replaced by the probe's score, everything else
  identical, so NMS ranks by the probe and the threshold cuts on it.

The difference between the two best-threshold F1 values is the F1 a *perfect*
ranking objective could deliver from these features. If it is a point or more,
the ranking loss is worth building out. If it is a tenth, it is not, and that
number belongs in the paper as the reason.

The probe is fitted on the training split, so using it to score the test split
is not leakage — but it *is* an upper bound rather than an achievable result,
because the probe optimizes the ranking directly and in isolation.

    python tools/clrbezier/probe_information.py CONFIG CKPT --save-probe probe.pth
    python tools/clrbezier/rescore_eval.py CONFIG CKPT probe.pth
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.registry import MODELS

from libs.datasets.metrics.culane_metric import CULaneMetric

from probe_information import Probe          # same directory
from fast_sweep import lane_scores, run_sweep


def logit(p, eps=1e-6):
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("probe", help="the .pth written by --save-probe")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes for the rasterization pass")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--rgb-masks", action="store_true",
                    help="rasterize onto the official 3-channel canvas instead "
                         "of 1 channel; 2.3x slower, provably the same IoU")
    ap.add_argument("--dump-threshold", type=float, default=0.10)
    ap.add_argument("--grid", type=float, nargs=3, default=[0.10, 0.96, 0.02],
                    metavar=("START", "STOP", "STEP"))
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg.model.setdefault("test_cfg", {})
    cfg.model["test_cfg"]["conf_threshold"] = args.dump_threshold
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(device).eval()
    head = model.bbox_head

    bundle = torch.load(args.probe, map_location="cpu")
    probe = Probe(bundle["in_dim"]).to(device).eval()
    probe.load_state_dict(bundle["state_dict"])
    mean = bundle["mean"].to(device)
    std = bundle["std"].to(device)
    feature_key = bundle.get("feature", "roi")
    print(f"probe: {bundle['in_dim']}-d {feature_key} features, "
          f"objective={bundle.get('objective', '?')}")

    captured = {}

    def grab(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()
        return hook

    handles = [head.cls_modules[0].register_forward_pre_hook(grab("roi")),
               head.cls_layers.register_forward_pre_hook(grab("tower"))]

    loader_cfg = copy.deepcopy(
        cfg.test_dataloader if args.split == "test" else cfg.val_dataloader)
    loader = Runner.build_dataloader(loader_cfg)
    evaluator_cfg = cfg.test_evaluator if args.split == "test" else cfg.val_evaluator
    data_root, data_list = evaluator_cfg["data_root"], evaluator_cfg["data_list"]
    categories_dir = os.path.join(data_root, "list/test_split/")

    base_dump, probe_dump = [], []
    seen = 0
    with torch.no_grad():
        for data in loader:
            data = model.data_preprocessor(data, False)
            pyramid = model.extract_feat(data["inputs"])
            outs = head(pyramid)
            pred_dict = outs[-1] if isinstance(outs, list) else outs
            samples = data["data_samples"]
            batch, num_q = pred_dict["cls_logits"].shape[:2]

            lanes, scores = head.get_lanes(
                pred_dict, as_lanes=head.test_cfg.as_lanes,
                extend_bottom=head.test_cfg.extend_bottom)
            for i, sample in enumerate(samples):
                base_dump.append((sample.metainfo["sub_img_name"], lanes[i],
                                  lane_scores(lanes[i], scores[i])))

            # Same predictions, same NMS, same decode — only the score changes.
            feature = captured[feature_key].view(batch * num_q, -1).float()
            value = probe((feature - mean.to(feature)) / std.to(feature))
            prob = torch.sigmoid(value).view(batch, num_q)
            swapped = dict(pred_dict)
            zeros = torch.zeros_like(prob)
            swapped["cls_logits"] = torch.stack([zeros, logit(prob)], dim=-1)

            lanes, scores = head.get_lanes(
                swapped, as_lanes=head.test_cfg.as_lanes,
                extend_bottom=head.test_cfg.extend_bottom)
            for i, sample in enumerate(samples):
                probe_dump.append((sample.metainfo["sub_img_name"], lanes[i],
                                   lane_scores(lanes[i], scores[i])))

            seen += batch
            if seen % 2000 < batch:
                print(f"  {seen} images", flush=True)
            if args.max_images is not None and seen >= args.max_images:
                break
    for handle in handles:
        handle.remove()

    metric = CULaneMetric(data_root=data_root, data_list=data_list)
    start, stop, step = args.grid
    thresholds = [t for t in np.round(np.arange(start, stop + 1e-9, step), 4)
                  if t >= args.dump_threshold]

    rows, best = run_sweep({"baseline": base_dump, "probe": probe_dump},
                           thresholds, data_root, data_list, categories_dir,
                           metric, jobs=args.jobs, verify=not args.no_verify,
                           logger=MMLogger.get_current_instance(),
                           rgb_masks=args.rgb_masks)

    print(f"\n{'conf':>6} {'baseline F1':>12} {'probe-scored F1':>16}")
    print("-" * 38)
    for i, threshold in enumerate(thresholds):
        print(f"{threshold:6.2f} {rows['baseline'][i]['F1'] * 100:12.2f} "
              f"{rows['probe'][i]['F1'] * 100:16.2f}")

    gain = (best["probe"][1] - best["baseline"][1]) * 100
    print(f"\nbaseline     best F1 {best['baseline'][1] * 100:.2f} at conf {best['baseline'][0]:.2f}")
    print(f"probe-scored best F1 {best['probe'][1] * 100:.2f} at conf {best['probe'][0]:.2f}")
    print(f"\nF1 available from perfect re-scoring of these features: {gain:+.2f}")
    if gain >= 1.0:
        print("Worth building the ranking objective out — and this is the number to")
        print("quote as its ceiling.")
    elif gain >= 0.3:
        print("Modest. A ranking loss recovers some fraction of this, not all of it.")
    else:
        print("Not worth it. The ordering information exists but does not survive")
        print("NMS into the metric — which is itself the result worth reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
