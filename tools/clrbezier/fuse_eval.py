"""Fuse duplicate lanes instead of selecting one of them.

The ceiling problem
-------------------
A ranking loss and a "collapse the cluster" loss have the *same* ceiling. If
ranking were perfect, NMS would keep each cluster's best member; if collapse
were perfect, every member would equal the best one and NMS could keep any.
Both end at F1(best member per cluster), which is exactly what
``rescore_eval.py`` measures. So neither idea can beat that number.

Fusion can, and for a reason selection cannot share: averaging cancels
independent error. Simulating a cluster of K near-duplicates with independent
lateral error and a noisy confidence (sigma 6 px, ranker noise 3):

      K   NMS keeps top   score-weighted fuse   perfect selection
      2       0.679              0.728                0.717
      3       0.725              0.780                0.786
      4       0.746              0.811                0.819
      6       0.775              0.847                0.871

Fusion sits far above what selection actually achieves, and at small K it
passes even perfect selection. This is Weighted Boxes Fusion's argument
(Solovyev et al.), moved to lanes.

Nothing here is trained. It runs on a checkpoint you already have, so it costs
one inference pass and tells you whether the duplicates your model is currently
throwing away carry usable information.

One inference pass, and one rasterization pass per decode mode: the threshold
sweep runs off cached IoU matrices (``fast_sweep.py``). The first version of
this script called the official evaluator once per threshold per mode -- 88 full
CULane evaluations, each announcing itself -- which is why it looked like it was
evaluating in a loop forever.

    python tools/clrbezier/fuse_eval.py CONFIG CKPT --power 2.0
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.registry import MODELS

from mmengine.logging import MMLogger

from libs.datasets.metrics.culane_metric import CULaneMetric
from libs.clrbezier.ranking import mean_row_distance

from fast_sweep import run_sweep


def cluster_and_fuse(xs, scores, anchor_params, lengths, img_w, nms_thres,
                     topk, power, mode):
    """Greedy clustering by score, then fuse or select within each cluster.

    The cluster's extent (start_y, theta, length) is taken from its
    highest-scoring member: averaging extents across members that cover
    different row ranges is not meaningful. Only the lateral positions are
    fused, per row, over the members valid at that row.
    """
    order = torch.argsort(scores, descending=True)
    distance = mean_row_distance(xs, img_w)
    taken = torch.zeros(xs.shape[0], dtype=torch.bool, device=xs.device)

    keep_xs, keep_scores, keep_params, keep_len = [], [], [], []
    for idx in order.tolist():
        if taken[idx] or len(keep_xs) >= topk:
            continue
        members = (~taken) & (distance[idx] < nms_thres)
        members[idx] = True
        taken |= members
        sel = members.nonzero(as_tuple=True)[0]

        if mode == "nms" or sel.numel() == 1:
            keep_xs.append(xs[idx])
        else:
            w = scores[sel].clamp_min(1e-6).pow(power)[:, None]
            valid = ((xs[sel] >= 0.0) & (xs[sel] <= 1.0)).to(xs.dtype)
            wv = w * valid
            denom = wv.sum(0)
            fused = torch.where(denom > 0, (wv * xs[sel]).sum(0) / denom.clamp_min(1e-6),
                                xs[idx])
            keep_xs.append(fused)
        keep_scores.append(scores[idx])
        keep_params.append(anchor_params[idx])
        keep_len.append(lengths[idx])

    if not keep_xs:
        empty = xs.new_zeros((0, xs.shape[1]))
        return empty, xs.new_zeros((0,)), xs.new_zeros((0, 3)), xs.new_zeros((0, 1))
    return (torch.stack(keep_xs), torch.stack(keep_scores),
            torch.stack(keep_params), torch.stack(keep_len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--dump-threshold", type=float, default=0.10)
    ap.add_argument("--grid", type=float, nargs=3, default=[0.10, 0.96, 0.02],
                    metavar=("START", "STOP", "STEP"))
    ap.add_argument("--power", type=float, default=2.0,
                    help="fusion weight = score ** power; higher trusts the "
                         "top member more, power -> inf reproduces NMS")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--modes", nargs="+", default=["nms", "fuse"],
                    choices=["nms", "fuse"],
                    help="decode modes to sweep; both by default, since the "
                         "comparison is the point")
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes for the rasterization pass")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--rgb-masks", action="store_true",
                    help="rasterize onto the official 3-channel canvas instead "
                         "of 1 channel; 2.3x slower, provably the same IoU")
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

    nms_thres = float(head.test_cfg.get("nms_thres", 50.0))
    topk = int(head.test_cfg.get("nms_topk", 4))
    print(f"clustering at {nms_thres} px, keeping {topk} per image, "
          f"fusion weight = score ** {args.power}")

    loader_cfg = copy.deepcopy(
        cfg.test_dataloader if args.split == "test" else cfg.val_dataloader)
    loader = Runner.build_dataloader(loader_cfg)
    ev = cfg.test_evaluator if args.split == "test" else cfg.val_evaluator
    data_root, data_list = ev["data_root"], ev["data_list"]
    categories_dir = os.path.join(data_root, "list/test_split/")

    modes = tuple(args.modes)
    dumps = {mode: [] for mode in modes}
    seen = 0
    with torch.no_grad():
        for data in loader:
            data = model.data_preprocessor(data, False)
            outs = head(model.extract_feat(data["inputs"]))
            pred = outs[-1] if isinstance(outs, list) else outs
            prob = torch.softmax(pred["cls_logits"].float(), -1)[..., 1]

            for b, sample in enumerate(data["data_samples"]):
                keep = prob[b] >= args.dump_threshold
                name = sample.metainfo["sub_img_name"]
                if not bool(keep.any()):
                    for m in dumps:
                        dumps[m].append((name, [], np.zeros(0)))
                    continue
                xs_b = pred["xs"][b][keep].float()
                sc_b = prob[b][keep]
                ap_b = pred["anchor_params"][b][keep].float()
                ln_b = pred["lengths"][b][keep].float()
                for mode in modes:
                    fx, fs, fp, fl = cluster_and_fuse(
                        xs_b, sc_b, ap_b, ln_b, head.img_w, nms_thres, topk,
                        args.power, mode)
                    lanes = head.predictions_to_lanes(
                        fx, fp, torch.round(fl * head.n_strips), fs,
                        head.test_cfg.as_lanes, head.test_cfg.extend_bottom)
                    dumps[mode].append(
                        (name, lanes, fs.detach().cpu().numpy().astype(np.float64)))
            seen += len(data["data_samples"])
            if seen % 2000 < len(data["data_samples"]):
                print(f"  {seen} images", flush=True)
            if args.max_images is not None and seen >= args.max_images:
                break

    metric = CULaneMetric(data_root=data_root, data_list=data_list)
    start, stop, step = args.grid
    thresholds = [t for t in np.round(np.arange(start, stop + 1e-9, step), 4)
                  if t >= args.dump_threshold]

    rows, best = run_sweep(dumps, thresholds, data_root, data_list,
                           categories_dir, metric, jobs=args.jobs,
                           verify=not args.no_verify,
                           logger=MMLogger.get_current_instance(),
                           rgb_masks=args.rgb_masks)

    header = "".join(f"{m + ' F1':>14}" for m in modes)
    print(f"\n{'conf':>6}{header}")
    print("-" * (6 + 14 * len(modes)))
    for i, t in enumerate(thresholds):
        cells = "".join(f"{rows[m][i]['F1'] * 100:14.2f}" for m in modes)
        print(f"{t:6.2f}{cells}")

    for mode in modes:
        print(f"\n{mode:<5} best F1 {best[mode][1] * 100:.2f} "
              f"at conf {best[mode][0]:.2f}")
    if "nms" in best and "fuse" in best:
        gain = (best["fuse"][1] - best["nms"][1]) * 100
        print(f"\nfusion - selection: {gain:+.2f} F1")
        print("Compare this against rescore_eval.py's number: that one bounds any")
        print("ranking or cluster-collapse scheme, and this one is not bound by it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
