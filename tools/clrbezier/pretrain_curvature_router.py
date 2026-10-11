"""Pretrain the curvature router of ``bbox_head.curv_moe_cfg`` on a trained checkpoint.

The router reads the head's own RoI features, so it is trained on top of a
detector that is already trained, with everything else frozen and in eval mode
(BatchNorm statistics and dropout untouched). The output checkpoint holds the
base weights, the experts (copies of the base classifier), the trained router
and the curvature bin edges; the MoE fine-tune loads it with ``load_from``.

Steps:
  1. Bin edges on |curvature| (the measure set in curv_moe_cfg):
     [straight, s1*sigma, s2*sigma, ...], sigma = std of the signed curvature of
     the GT lanes in --sigma-batches training batches (after the training
     augmentation, as the router sees them). --edges or curv_moe_cfg.bin_edges
     skip this. num_experts must be len(edges) + 1.
  2. Router training: each query of each refinement stage is labelled with the
     bin of the GT lane its stage output overlaps most (LaneIoU >= label_iou_thr);
     other queries are ignored. Cross-entropy on the router only.
  3. Optional router accuracy on a held-out list (--val-list; list/val_gt.txt
     for CULane, same format as train_gt.txt), per bin and per stage, with the
     majority-class baseline.

    python tools/clrbezier/pretrain_curvature_router.py \\
        MOE_CONFIG.py BASE_CHECKPOINT.pth --out work_dirs/router/router.pth \\
        --val-list list/val_gt.txt

MOE_CONFIG must build the same network as BASE_CHECKPOINT plus curv_moe_cfg
(e.g. a config that _base_s the base config and only adds curv_moe_cfg).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config", help="MoE config (bbox_head.curv_moe_cfg set)")
    p.add_argument("checkpoint", help="trained base checkpoint of the same network")
    p.add_argument("--out", required=True, help="output checkpoint (.pth)")
    p.add_argument("--epochs", type=float, default=1.0, help="passes over the training set")
    p.add_argument("--max-iters", type=int, default=None, help="cap on router updates")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=None, help="default: the config's")
    p.add_argument("--workers", type=int, default=None, help="default: the config's")
    p.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 2.0],
                   help="bin edges after 'straight', in multiples of sigma")
    p.add_argument("--straight", type=float, default=None,
                   help="|curvature| at or below this is the straight bin "
                        "(default 2.0 px for measure sagitta_px; required for 'curvature')")
    p.add_argument("--edges", type=float, nargs="+", default=None,
                   help="explicit bin edges (skips the sigma pass)")
    p.add_argument("--sigma-batches", type=int, default=300)
    p.add_argument("--val-list", default=None,
                   help="annotated list for router accuracy, relative to data_root")
    p.add_argument("--val-batches", type=int, default=None, help="default: the whole list")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", help="fp16 autocast for the frozen network")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=50)
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# building (mmengine / mmdet); replaced by a fake in the tests


def _val_dataset_cfg(cfg, val_list):
    ds = copy.deepcopy(cfg.train_dataloader["dataset"])
    root = ds.get("data_root", "")
    ds["data_list"] = val_list if os.path.isabs(val_list) else os.path.join(root, val_list)
    ds["diff_file"] = None
    ds["test_mode"] = False
    val_al = cfg.get("val_al_pipeline", None)
    pipe = []
    for step in ds["pipeline"]:
        step = dict(step)
        if "pipelines" in step:
            if val_al is None:
                print("WARNING: no val_al_pipeline in the config; the held-out list "
                      "is read with the training augmentation")
            else:
                step["pipelines"] = copy.deepcopy(val_al)
        pipe.append(step)
    ds["pipeline"] = pipe
    return ds


def build_all(args):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    if not cfg.model["bbox_head"].get("curv_moe_cfg"):
        raise SystemExit("the config has no bbox_head.curv_moe_cfg")
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=False)

    loader_cfg = copy.deepcopy(cfg.train_dataloader)
    if args.batch_size:
        loader_cfg["batch_size"] = args.batch_size
    if args.workers is not None:
        loader_cfg["num_workers"] = args.workers
    loader_cfg["sampler"] = dict(type="DefaultSampler", shuffle=True, seed=args.seed)
    train_loader = Runner.build_dataloader(loader_cfg, seed=args.seed)
    val_loader = None
    if args.val_list:
        vcfg = copy.deepcopy(loader_cfg)
        vcfg["dataset"] = _val_dataset_cfg(cfg, args.val_list)
        vcfg["sampler"] = dict(type="DefaultSampler", shuffle=False)
        val_loader = Runner.build_dataloader(vcfg, seed=args.seed)
    return model, train_loader, val_loader


# --------------------------------------------------------------------------
# helpers


def gt_padded(head, data_samples, device):
    """GT lanes [B, M, 6+R] (px; non-lane slots invalid) and their mask [B, M]."""
    lanes = head.target_adapter.extract_lanes(data_samples, device)
    mask = lanes[..., 1] == 1
    gt = lanes.clone()
    gt[..., 6:] = torch.where(mask[..., None], gt[..., 6:], torch.full_like(gt[..., 6:], -1e5))
    return gt, mask


def frozen_forward(model, head, batch, amp):
    """Stage outputs and router inputs of the main branch, no gradient."""
    data = model.data_preprocessor(batch, False)
    device_type = next(model.parameters()).device.type
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=bool(amp)):
        feats = head._select_features(model.extract_feat(data["inputs"]))
        preds, _, extra = head._refine(feats, head.prior_bank(feats[-1].shape[0]),
                                       head.gsrc_tokens(feats))
    return data, [p.float() for p in preds], [x.float() for x in extra["curv_in"]]


def stage_labels(head, preds, gt, mask):
    from libs.clrbezier.curv_moe import router_labels
    moe = head.curv_moe
    bins, _ = moe.lane_bins(gt[..., 6:], head.prior_ys, head.img_w, head.img_h)
    bins = torch.where(mask, bins, torch.zeros_like(bins))
    return [router_labels(p, gt, bins, mask, head.lane_width, head.img_w, head.img_h,
                          (head.iou_w, head.iou_h), moe.label_iou_thr) for p in preds]


def estimate_edges(head, loader, device, args):
    from libs.clrbezier.curv_moe import curvature_bins, edges_from_sigma, lane_curvature
    moe = head.curv_moe
    straight = args.straight
    if straight is None:
        if moe.measure != "sagitta_px":
            raise SystemExit("--straight is required for measure 'curvature' (in 1/px)")
        straight = 2.0
    values = []
    for i, batch in enumerate(loader):
        if i >= args.sigma_batches:
            break
        gt, mask = gt_padded(head, batch["data_samples"], device)
        v = lane_curvature(gt[..., 6:], head.prior_ys, head.img_w, head.img_h, moe.measure)
        values.append(v[mask].cpu())
    values = torch.cat(values) if values else torch.zeros(0)
    if values.numel() < 10:
        raise SystemExit(f"only {values.numel()} GT lanes found in {args.sigma_batches} batches")
    edges, sigma = edges_from_sigma(values, straight, args.sigmas)
    a = values.abs()
    q = torch.quantile(a.double(), torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    unit = "px" if moe.measure == "sagitta_px" else "1/px"
    print(f"curvature ({moe.measure}) over {values.numel()} GT lanes: sigma {sigma:.4g} {unit}; "
          f"|c| median {q[0]:.4g}, p90 {q[1]:.4g}, p99 {q[2]:.4g}, max {a.max():.4g}")
    counts = torch.bincount(curvature_bins(values, torch.tensor(edges)), minlength=len(edges) + 1)
    _print_bins(edges, counts, unit)
    return edges, dict(sigma=sigma, num_lanes=int(values.numel()), straight=straight,
                       sigmas=list(args.sigmas), counts=counts.tolist())


def _bin_names(edges):
    names = [f"|c| <= {edges[0]:.4g}"]
    names += [f"{a:.4g} < |c| <= {b:.4g}" for a, b in zip(edges[:-1], edges[1:])]
    names.append(f"|c| > {edges[-1]:.4g}")
    return names


def _print_bins(edges, counts, unit):
    total = max(int(counts.sum()), 1)
    print(f"bins ({unit}):")
    for k, (name, c) in enumerate(zip(_bin_names(edges), counts.tolist())):
        print(f"  expert {k}: {name:<28} {c:8d} lanes ({100.0 * c / total:5.1f}%)")


@torch.no_grad()
def evaluate(model, head, loader, device, args, max_batches=None):
    moe = head.curv_moe
    K = moe.num_experts
    S = head.refine_layers
    conf = torch.zeros(S, K, K, dtype=torch.long)          # [stage, true, pred]
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        data, preds, inputs = frozen_forward(model, head, batch, args.amp)
        gt, mask = gt_padded(head, data["data_samples"], device)
        labels = stage_labels(head, preds, gt, mask)
        for s, (x, lab) in enumerate(zip(inputs, labels)):
            pred = moe.router(x).argmax(-1)
            keep = lab >= 0
            idx = (lab[keep] * K + pred[keep]).cpu()
            conf[s] += torch.bincount(idx, minlength=K * K).view(K, K)
    out = {}
    for s in range(S):
        c = conf[s].double()
        n = c.sum().clamp_min(1)
        recall = (c.diag() / c.sum(1).clamp_min(1)).tolist()
        out[f"stage{s}"] = dict(
            acc=float(c.diag().sum() / n), majority_acc=float(c.sum(1).max() / n),
            balanced_acc=float(sum(recall) / K), recall=recall,
            labelled=int(c.sum()), confusion=conf[s].tolist())
    return out


def _print_eval(res, K):
    for s, r in res.items():
        print(f"  {s}: acc {r['acc']:.3f} (majority class {r['majority_acc']:.3f}), "
              f"balanced {r['balanced_acc']:.3f}, {r['labelled']} labelled queries; "
              "recall per bin " + ", ".join(f"{v:.2f}" for v in r["recall"]))
    last = list(res.values())[-1]
    print("  final-stage confusion (rows: true bin, columns: routed expert):")
    for k, row in enumerate(last["confusion"]):
        print(f"    {k}: " + " ".join(f"{v:8d}" for v in row))


# --------------------------------------------------------------------------


def main(argv=None, build=build_all):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    model, train_loader, val_loader = build(args)
    head = model.bbox_head
    moe = getattr(head, "curv_moe", None)
    if moe is None:
        raise SystemExit("the head has no curv_moe (set bbox_head.curv_moe_cfg)")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    moe.router.requires_grad_(True)
    moe.capture = True

    # 1. bin edges
    if args.edges:
        edges, edge_info = [float(e) for e in args.edges], dict(source="--edges")
    elif moe.edges_from_cfg:
        edges, edge_info = moe.bin_edges.tolist(), dict(source="curv_moe_cfg.bin_edges")
    elif bool(torch.isfinite(moe.bin_edges).all()):
        edges, edge_info = moe.bin_edges.tolist(), dict(source="checkpoint")
    else:
        print(f"estimating the curvature spread from {args.sigma_batches} training batches")
        edges, edge_info = estimate_edges(head, train_loader, device, args)
        edge_info["source"] = "sigma"
    if len(edges) + 1 != moe.num_experts:
        raise SystemExit(f"{len(edges)} edges give {len(edges) + 1} bins but curv_moe_cfg "
                         f"has num_experts={moe.num_experts}; set num_experts="
                         f"{len(edges) + 1} (or change --sigmas)")
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])) or edges[0] < 0:
        raise SystemExit(f"bin edges must be >= 0 and increasing: {edges}")
    moe.bin_edges.copy_(torch.tensor(edges, dtype=moe.bin_edges.dtype))
    print("bin edges:", [round(e, 6) for e in edges], f"({edge_info['source']})")

    # 2. router training
    per_epoch = len(train_loader)
    total = int(math.ceil(args.epochs * per_epoch))
    if args.max_iters is not None:
        total = min(total, int(args.max_iters))
    opt = torch.optim.AdamW(moe.router.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: 0.5 * (1.0 + math.cos(math.pi * min(it, total) / max(total, 1))))
    K = moe.num_experts
    it, t0 = 0, time.time()
    run_loss, run_n, run_hit = 0.0, 0, 0
    run_counts = torch.zeros(K, dtype=torch.long)
    print(f"training the router for {total} iterations ({per_epoch} per epoch)")
    while it < total:
        for batch in train_loader:
            if it >= total:
                break
            data, preds, inputs = frozen_forward(model, head, batch, args.amp)
            gt, mask = gt_padded(head, data["data_samples"], device)
            labels = stage_labels(head, preds, gt, mask)
            logits = torch.cat([moe.router(x).reshape(-1, K) for x in inputs])
            lab = torch.cat([l.reshape(-1) for l in labels])
            keep = lab >= 0
            n = int(keep.sum())
            if n:
                loss = F.cross_entropy(logits, lab, ignore_index=-1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run_loss += float(loss.detach()) * n
                run_n += n
                run_hit += int((logits.argmax(-1)[keep] == lab[keep]).sum())
                run_counts += torch.bincount(lab[keep].cpu(), minlength=K)
            sched.step()
            it += 1
            if it % args.log_interval == 0 or it == total:
                share = (run_counts.double() / run_counts.sum().clamp_min(1)).tolist()
                print(f"it {it}/{total}  ce {run_loss / max(run_n, 1):.4f}  "
                      f"acc {run_hit / max(run_n, 1):.3f}  labelled {run_n}  "
                      f"label share " + "/".join(f"{v:.2f}" for v in share) +
                      f"  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
                run_loss, run_n, run_hit = 0.0, 0, 0
                run_counts.zero_()
    moe.router_trained.fill_(1)

    # 3. held-out accuracy
    val = None
    if val_loader is not None:
        print("router accuracy on", args.val_list)
        val = evaluate(model, head, val_loader, device, args, args.val_batches)
        _print_eval(val, K)

    moe.capture = False
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    meta = dict(curv_router=dict(edges=edges, edge_info=edge_info, measure=moe.measure,
                                 num_experts=K, base_checkpoint=args.checkpoint,
                                 iterations=total, val=val))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(dict(state_dict=state, meta=meta), args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {args.out} (load it with load_from in the MoE config)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
