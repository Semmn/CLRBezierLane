"""Recompute the head's BatchNorm running statistics from the main branch only.

Why: during training the collaborative auxiliary branch runs the same RoIGather
(and lateral-evidence) BatchNorm layers in a second forward call. Each call
normalizes with its own batch statistics, but both update the running mean/var,
so with the default ``aux_cfg bn_stats="shared"`` the statistics used at test
time are a ~47/53 blend of main and aux (perturbed-anchor) features, which the
main branch never saw during training.

This tool keeps every weight, resets only the head's BN running statistics,
and re-estimates them (cumulative average, no momentum) with main-branch-only
forward passes, exactly what ``aux_cfg bn_stats="main"`` would have tracked.
The backbone and neck are untouched (they run once per image already).
Evaluate the written checkpoint with the usual sweep to see whether the blended
statistics cost F1:

    python tools/clrbezier/recalibrate_head_bn.py CONFIG CHECKPOINT \
        [--split train|val] [--num-batches 200] [--batch-size N] [--out PATH]
    python tools/clrbezier/sweep_conf_eval.py CONFIG PATH --range 0.70 0.90 --step 0.02

``--split train`` (default) uses the training pipeline, i.e. the augmented
images the statistics were tracked on during training.
"""
from __future__ import annotations

import argparse
import os.path as osp

import torch
import torch.nn as nn


def head_bn_layers(head):
    return [(name, m) for name, m in head.named_modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]


@torch.no_grad()
def recalibrate(model, batches, num_batches, log_every=50):
    """Re-estimate ``model.bbox_head`` BN statistics on main-branch passes.

    Everything else stays in eval mode (frozen backbone/neck statistics, no
    dropout, test-mode head), so only the head's BN layers see batch statistics.
    Returns {layer name: (old mean, old var, new mean, new var)}.
    """
    head = model.bbox_head
    layers = head_bn_layers(head)
    if not layers:
        raise RuntimeError("the head has no BatchNorm layers with running statistics")
    old = {n: (m.running_mean.clone(), m.running_var.clone()) for n, m in layers}
    momenta = {n: m.momentum for n, m in layers}
    model.eval()
    for _, m in layers:
        m.reset_running_stats()
        m.momentum = None            # cumulative average over all batches
        m.train()
    seen = 0
    try:
        for data in batches:
            data = model.data_preprocessor(data, False)
            head(model.extract_feat(data["inputs"]))   # head.training is False: main branch only
            seen += 1
            if log_every and seen % log_every == 0:
                print(f"  {seen} batches", flush=True)
            if seen >= num_batches:
                break
    finally:
        for n, m in layers:
            m.momentum = momenta[n]
            m.eval()
    if seen == 0:
        raise RuntimeError("the dataloader yielded no batches")
    return {n: (*old[n], m.running_mean.clone(), m.running_var.clone()) for n, m in layers}, seen


def report(stats):
    """Per layer: mean |Δmean| in units of the old std, and the new/old var ratio."""
    print(f"{'layer':52s} {'|dmean|/std':>11s} {'var new/old':>16s}")
    for name, (m0, v0, m1, v1) in stats.items():
        shift = ((m1 - m0).abs() / (v0 + 1e-5).sqrt()).mean().item()
        ratio = (v1 + 1e-5) / (v0 + 1e-5)
        print(f"{name:52s} {shift:11.3f} {ratio.median().item():7.3f} "
              f"[{ratio.min().item():.2f}, {ratio.max().item():.2f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--split", choices=("train", "val"), default="train")
    ap.add_argument("--num-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="output checkpoint (default: CHECKPOINT with _mainbn suffix)")
    args = ap.parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)
    cfg.work_dir = osp.join(osp.dirname(osp.abspath(args.checkpoint)), "recalibrate_head_bn")
    cfg.load_from = args.checkpoint
    loader_cfg = cfg.train_dataloader if args.split == "train" else cfg.val_dataloader
    if args.batch_size:
        loader_cfg.batch_size = args.batch_size
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    loader = Runner.build_dataloader(loader_cfg, seed=0)

    stats, seen = recalibrate(model, loader, args.num_batches)
    print(f"re-estimated {len(stats)} head BN layers from {seen} batches ({args.split})")
    report(stats)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["state_dict"]
    for name, (_, _, mean, var) in stats.items():
        key = f"bbox_head.{name}."
        if key + "running_mean" not in state:
            raise KeyError(f"{key}running_mean not in the checkpoint state_dict")
        state[key + "running_mean"] = mean.cpu()
        state[key + "running_var"] = var.cpu()
    if "ema_state_dict" in ckpt:
        print("note: the checkpoint also has EMA weights; they were not recalibrated")
    out = args.out or osp.splitext(args.checkpoint)[0] + "_mainbn.pth"
    torch.save(ckpt, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
