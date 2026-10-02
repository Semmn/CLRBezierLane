"""Is the MoE actually there, and is it actually routing?

Memory is the wrong instrument for this. A soft mixture on the deformable offset
head adds about 4 MB over three stages -- the head emits 3 numbers, so E=4 costs
[B,K,S,4,3] of activation against [B,K,S,4,64] for the sampled features it sits
next to. PyTorch's caching allocator rounds that away, so identical ``nvidia-smi``
readings are the expected result and say nothing about whether the module exists.

This answers it directly: build the model from the config, find every MoE layer,
report what it added, and run one forward to confirm the router produces stats and
receives gradient.

    python tools/clrbezier/check_moe_wired.py CONFIG [--checkpoint CKPT]
"""
from __future__ import annotations

import argparse
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine and avoids competing with training")
    args = ap.parse_args()

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS

    from libs.clrbezier.moe import MoEFFN, MoEGate, MoELinear

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    model = MODELS.build(cfg.model)
    if args.checkpoint:
        from mmengine.runner.checkpoint import load_checkpoint
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(torch.device(args.device)).eval()

    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (MoELinear, MoEFFN))]
    gates = [(n, m) for n, m in model.named_modules() if isinstance(m, MoEGate)]

    print(f"\nconfig: {args.config}")
    if not layers:
        print("\nNO MoE LAYERS FOUND.\n")
        print("The config is not reaching the module. Check, in order:")
        print("  1. the key is spelled deform_moe_cfg (gather) or")
        print("     gsrc_cfg.context_cfg.moe_cfg (GSRC), not moe_cfg at the top level")
        print("  2. roi_gather_cfg.type is 'CurveAlignedDeformableROIGather'")
        print("  3. for GSRC, context_branch is 'transformer' -- moe_cfg has no")
        print("     effect on the 'segman' branch and now raises there")
        print("  4. the base config you inherit from actually builds that module")
        return 1

    print(f"\n{len(layers)} MoE layer(s), {len(gates)} gate(s):\n")
    total_expert, total_gate = 0, 0
    for name, mod in layers:
        experts = mod.experts if isinstance(mod, MoELinear) else mod.fc1
        e, out_dim, in_dim = experts.weight.shape
        n_expert = sum(p.numel() for n, p in mod.named_parameters() if "gate" not in n)
        n_gate = sum(p.numel() for n, p in mod.named_parameters() if "gate" in n)
        total_expert += n_expert
        total_gate += n_gate
        print(f"  {name}")
        print(f"    {type(mod).__name__}: E={e}, {in_dim} -> {out_dim}, "
              f"mode={mod.gate.mode}, top_k={mod.gate.top_k}")
        print(f"    expert params {n_expert}, gate params {n_gate}")
        print(f"    expert divergence {float(experts.divergence()):.4f} "
              f"(must be > 0, or the router's gradient is exactly zero)")
        # A single plain Linear doing the same job, for the comparison that
        # explains why memory did not move.
        print(f"    a plain Linear here would be {in_dim * out_dim + out_dim} params")

    print(f"\n  totals: experts {total_expert}, gates {total_gate}, "
          f"model {sum(p.numel() for p in model.parameters())}")

    print("\nrouter state at init:")
    for name, g in gates:
        w = g.proj.weight
        uniform = float(w.abs().max()) == 0.0
        print(f"  {name}: E={g.num_experts}, mode={g.mode}, "
              f"|W|max={float(w.abs().max()):.2e} "
              f"{'(uniform gate, as intended)' if uniform else '(NON-uniform at init)'}")
        print(f"    entropy should start at ln({g.num_experts}) = "
              f"{math.log(g.num_experts):.4f}")

    # One forward with gradient, to prove the router is live rather than present.
    print("\nlive check: one forward + backward on random input")
    dim = layers[0][1].gate.proj.in_features
    mod = layers[0][1]
    x = torch.randn(4, 7, dim, requires_grad=True)
    out, stats = mod(x)
    out.square().sum().backward()
    grad = mod.gate.proj.weight.grad
    gmax = 0.0 if grad is None else float(grad.abs().max())
    print(f"  output shape {tuple(out.shape)}")
    print(f"  stats reported: {sorted(stats)}")
    print(f"  entropy {float(stats['moe_entropy']):.6f} "
          f"(ln E = {math.log(mod.gate.num_experts):.6f})")
    print(f"  router gradient max |g| = {gmax:.3e}")
    ok = gmax > 1e-12 and "moe_entropy" in stats
    print(f"\n{'OK — the router is live and receiving gradient.' if ok else 'BROKEN — the router receives no gradient; the experts are probably tied or zeroed.'}")
    print("\nNow confirm it in training: moe_entropy must appear in scalars.json.")
    print("That, not GPU memory, is the signal. It starts at ln(E) and falling")
    print("below it is the entire experiment; flat means nothing specialized.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
