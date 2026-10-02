"""Self-test for the MoE module and the rewritten deformable gather.

This file exists because the old deformable branch was inert for a reason no
shape test would ever catch: a symmetric initialization made one gradient
*exactly zero*. ``tools/clrbezier/check_deform_init.py`` asserted that very
condition, so the tooling was certifying the bug. These tests assert the
opposite, and they demonstrate the old behaviour alongside the new one so the
difference is visible rather than asserted.

The same trap applies to the MoE: a router's gradient is
``J_softmax @ (expert_outputs @ dL/dy)``, which is exactly zero when the expert
outputs are identical. So "never zero-initialize an expert" gets a test too.

    python tools/clrbezier/test_deform_moe.py
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libs.clrbezier.curve_deformable_roi_gather import (  # noqa: E402
    CurveAlignedDeformableROIGather,
    CurveDeformableSampler,
    build_clr_curve_reference_points,
    masked_softmax,
    offset_fan,
)
from libs.clrbezier.moe import (  # noqa: E402
    MoEFFN, MoEGate, MoELinear, cv_squared, finalize_moe_stats, merge_moe_stats,
)

OK = True


def chk(name, cond, extra=""):
    global OK
    print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""))
    OK &= bool(cond)


def section(title):
    print(f"\n{title}")


# ======================================================================= MoE


def test_gate():
    section("MoEGate")
    torch.manual_seed(0)
    dim, E, N = 16, 4, 32
    x = torch.randn(N, dim)

    soft = MoEGate(dim, E, mode="soft")
    w, st = soft(x)
    chk("soft gate shape", tuple(w.shape) == (N, E), str(tuple(w.shape)))
    chk("soft gate rows sum to 1", torch.allclose(w.sum(-1), torch.ones(N), atol=1e-6))
    # The router is zero-initialized, so the gate starts uniform and the entropy
    # starts at ln(E). That is the baseline the sharpening diagnostic moves from.
    chk(f"zero-init router -> uniform gate (entropy = ln {E} = {math.log(E):.4f})",
        abs(float(st["moe_entropy"]) - math.log(E)) < 1e-5, f"{float(st['moe_entropy']):.6f}")
    chk("entropy fraction is 1.0 at init", abs(float(st["moe_entropy_frac"]) - 1.0) < 1e-5)
    chk("no balance loss in soft mode", "moe_balance_loss" not in st)
    chk("load is perfectly even at init", float(st["moe_load_cv2"]) < 1e-9)

    for k in (1, 2):
        hard = MoEGate(dim, E, mode="hard", top_k=k)
        w, st = hard(x)
        nz = (w > 0).sum(-1)
        chk(f"hard top-{k}: exactly {k} experts active per token",
            bool((nz == k).all()), f"counts {sorted(set(nz.tolist()))}")
        chk(f"hard top-{k}: rows still sum to 1",
            torch.allclose(w.sum(-1), torch.ones(N), atol=1e-6))
        chk(f"hard top-{k}: balance loss reported", "moe_balance_loss" in st)

    chk("cv_squared is 0 when uniform", float(cv_squared(torch.full((4,), 0.25))) < 1e-9)
    chk(f"cv_squared is E-1 = 3 when fully collapsed",
        abs(float(cv_squared(torch.tensor([1.0, 0, 0, 0]))) - 3.0) < 1e-5,
        f"{float(cv_squared(torch.tensor([1.0, 0, 0, 0]))):.4f}")

    for bad in (dict(mode="nope"), dict(top_k=0), dict(top_k=99), dict(temperature=0.0)):
        try:
            MoEGate(dim, E, **bad)
            chk(f"rejects {bad}", False)
        except ValueError:
            chk(f"rejects {bad}", True)


def test_router_symmetry():
    section("the router symmetry trap (the same shape as the deformable bug)")
    torch.manual_seed(0)
    din, dout, E, N = 8, 6, 4, 16
    x = torch.randn(N, din)

    layer = MoELinear(din, dout, num_experts=E)
    out, st = layer(x)
    chk("MoELinear shape", tuple(out.shape) == (N, dout), str(tuple(out.shape)))
    # At init the gate is uniform, so the mixture equals the averaged expert --
    # algebraically one linear layer. The capacity only appears as the gate
    # starts depending on x, which is what the entropy diagnostic tracks.
    mean_w = layer.experts.weight.mean(0)
    mean_b = layer.experts.bias.mean(0)
    chk("uniform gate == averaged expert (so capacity needs the gate to vary)",
        torch.allclose(out, x @ mean_w.T + mean_b, atol=1e-5))
    chk("experts are diverse at init", float(st["moe_expert_divergence"]) > 0.1,
        f"{float(st['moe_expert_divergence']):.4f}")

    def router_grad(mod):
        mod.zero_grad(set_to_none=True)
        y, _ = mod(x)
        y.square().sum().backward()
        g = mod.gate.proj.weight.grad
        return 0.0 if g is None else float(g.abs().max())

    diverse = router_grad(MoELinear(din, dout, num_experts=E))
    chk("diverse experts -> the router receives gradient", diverse > 1e-8,
        f"max |g| = {diverse:.3e}")

    tied = MoELinear(din, dout, num_experts=E)
    with torch.no_grad():                        # make every expert identical
        tied.experts.weight.copy_(tied.experts.weight[0:1].expand_as(tied.experts.weight))
        tied.experts.bias.copy_(tied.experts.bias[0:1].expand_as(tied.experts.bias))
    g_tied = router_grad(tied)
    chk("identical experts -> the router's gradient is EXACTLY zero, forever",
        g_tied < 1e-12, f"max |g| = {g_tied:.3e}")

    zeroed = MoELinear(din, dout, num_experts=E)
    with torch.no_grad():
        zeroed.experts.weight.zero_()
        zeroed.experts.bias.zero_()
    g_zero = router_grad(zeroed)
    chk("zero-initialized experts -> same trap (so never zero-init an expert)",
        g_zero < 1e-12, f"max |g| = {g_zero:.3e}")


def test_expert_gradients():
    section("where gradient reaches")
    torch.manual_seed(0)
    din, dout, E, N = 8, 6, 4, 32
    x = torch.randn(N, din)

    soft = MoELinear(din, dout, num_experts=E, mode="soft")
    soft(x)[0].square().sum().backward()
    per_expert = soft.experts.weight.grad.flatten(1).abs().sum(1)
    chk("soft mode: every expert receives gradient", bool((per_expert > 1e-9).all()),
        f"min {float(per_expert.min()):.3e}")

    hard = MoELinear(din, dout, num_experts=E, mode="hard", top_k=1)
    with torch.no_grad():
        hard.gate.proj.weight.normal_(0, 2.0)      # break the uniform tie
        gate_w, _ = hard.gate(x)
    selected = (gate_w > 0).any(0)                 # which experts any token routed to
    hard(x)[0].square().sum().backward()
    per_expert = hard.experts.weight.grad.flatten(1).abs().sum(1)
    got = per_expert > 1e-12
    chk(f"hard mode: exactly the routed experts receive gradient "
        f"({int(selected.sum())} of {E} routed)", bool((got == selected).all()),
        f"routed {selected.tolist()}, gradient {got.tolist()}")

    ffn = MoEFFN(din, hidden_dim=12, num_experts=E)
    out, st = ffn(torch.randn(3, 5, din))
    chk("MoEFFN shape", tuple(out.shape) == (3, 5, din), str(tuple(out.shape)))
    out.square().sum().backward()
    chk("MoEFFN: both expert stacks receive gradient",
        ffn.fc1.weight.grad.abs().max() > 1e-9 and ffn.fc2.weight.grad.abs().max() > 1e-9)

    # route_on lets the gate read something more informative than the layer input.
    lay = MoELinear(din, dout, num_experts=E, gate_dim=3)
    o2, _ = lay(x, route_on=torch.randn(N, 3))
    chk("route_on uses a separate gate input", tuple(o2.shape) == (N, dout))


def test_stat_merging():
    section("stat merging (version-tolerant, like the rank-loss fix)")
    acc = {}
    merge_moe_stats(acc, {"moe_entropy": torch.tensor(1.0)})
    merge_moe_stats(acc, {"moe_entropy": torch.tensor(3.0)})
    out = finalize_moe_stats(acc)
    chk("two layers average rather than sum", abs(float(out["moe_entropy"]) - 2.0) < 1e-6,
        f"{float(out['moe_entropy']):.4f}")
    chk("bookkeeping keys are dropped", not any(k.startswith("__n_") for k in out))
    acc2 = {}
    merge_moe_stats(acc2, {"brand_new_key": torch.tensor(1.0)}, prefix="s0_")
    chk("an unknown key passes through with its prefix", "s0_brand_new_key" in acc2)


# ============================================================ deformable gather


def test_offset_fan():
    section("the offset fan (the symmetry fix)")
    expected = {2: [0.90], 3: [0.90, -0.90], 4: [0.45, -0.45, 0.90],
                5: [0.45, -0.45, 0.90, -0.90]}
    for n, want in expected.items():
        got = torch.tanh(offset_fan(n))
        chk(f"num_offsets={n} -> {want}",
            got.shape[0] == len(want) and torch.allclose(got, torch.tensor(want), atol=1e-5),
            f"{[round(v, 4) for v in got.tolist()]}")
    for n in (2, 3, 4, 5, 8):
        fan = torch.tanh(offset_fan(n))
        chk(f"num_offsets={n}: every displacement is non-zero and distinct",
            bool((fan.abs() > 1e-3).all()) and len(set(round(v, 6) for v in fan.tolist())) == len(fan))


def test_masked_softmax():
    section("masked_softmax")
    logits = torch.randn(2, 5)
    mask = torch.tensor([[True, True, False, True, False],
                         [False, False, False, False, False]])
    w = masked_softmax(logits, mask, dim=-1)
    chk("masked entries are exactly zero", float(w[0, 2].abs() + w[0, 4].abs()) == 0.0)
    chk("valid entries sum to 1", abs(float(w[0].sum()) - 1.0) < 1e-6)
    chk("an all-masked row returns zeros, not a renormalized denominator",
        float(w[1].abs().sum()) == 0.0)


def test_reference_points():
    section("build_clr_curve_reference_points")
    B, K, P, S = 2, 6, 36, 18
    xs = torch.rand(B, K, P)
    ys = torch.linspace(1, 0, P)
    pts, mask = build_clr_curve_reference_points(xs, ys, S)
    chk("points shape", tuple(pts.shape) == (B, K, S, 2), str(tuple(pts.shape)))
    chk("mask shape", tuple(mask.shape) == (B, K, S), str(tuple(mask.shape)))
    chk("all in-range points are valid", bool(mask.all()))
    chk("points stay in [0, 1]", bool((pts >= 0).all() and (pts <= 1).all()))

    xs_bad = xs.clone()
    xs_bad[0, 0, :] = -5.0
    _, mask_bad = build_clr_curve_reference_points(xs_bad, ys, S)
    chk("out-of-range x is marked invalid", not bool(mask_bad[0, 0].any()))
    xs_nan = xs.clone()
    xs_nan[0, 1, 0] = float("nan")     # row 0 is always among the selected rows
    pts_nan, mask_nan = build_clr_curve_reference_points(xs_nan, ys, S)
    chk("NaN is masked and sanitized", bool(torch.isfinite(pts_nan).all())
        and not bool(mask_nan[0, 1].all()))
    for ydim in (ys, ys.expand(B, P), ys.view(1, 1, P).expand(B, K, P)):
        pts2, _ = build_clr_curve_reference_points(xs, ydim, S)
        chk(f"accepts prior_ys with {ydim.ndim} dims", tuple(pts2.shape) == (B, K, S, 2))
    for bad in (1, P + 1):
        try:
            build_clr_curve_reference_points(xs, ys, bad)
            chk(f"rejects num_curve_samples={bad}", False)
        except ValueError:
            chk(f"rejects num_curve_samples={bad}", True)


def test_curve_frame():
    section("curve_frame")
    pts = torch.stack([torch.linspace(0.2, 0.8, 12),
                       torch.linspace(0.9, 0.1, 12)], dim=-1).view(1, 1, 12, 2)
    t, n = CurveDeformableSampler.curve_frame(pts, 320, 800)
    chk("tangent is unit length", torch.allclose(t.norm(dim=-1), torch.ones(1, 1, 12), atol=1e-4))
    chk("normal is unit length", torch.allclose(n.norm(dim=-1), torch.ones(1, 1, 12), atol=1e-4))
    chk("normal is perpendicular to tangent", float((t * n).sum(-1).abs().max()) < 1e-5)
    # Coincident points would blow up a tight-epsilon normalize; the loose eps caps it.
    same = torch.full((1, 1, 5, 2), 0.5)
    t2, _ = CurveDeformableSampler.curve_frame(same, 320, 800)
    chk("coincident points do not produce NaN", bool(torch.isfinite(t2).all()))


def _sampler_case(init_mode, num_offsets=4, moe_cfg=None, seed=0):
    torch.manual_seed(seed)
    B, K, S, C, D = 2, 5, 9, 12, 16
    s = CurveDeformableSampler(C, D, num_curve_samples=S, num_offsets=num_offsets,
                               init_mode=init_mode, moe_cfg=moe_cfg)
    feature = torch.randn(B, C, 20, 50)
    query = torch.randn(B, K, D)
    pts = torch.rand(B, K, S, 2)
    mask = torch.ones(B, K, S, dtype=torch.bool)
    out, stats = s(feature, query, pts, mask)
    out.square().sum().backward()
    return s, out, stats, (B, K, D)


def test_sampler():
    section("CurveDeformableSampler — the regression the old version failed")
    s, out, stats, (B, K, D) = _sampler_case("symmetry_broken")
    chk("output shape", tuple(out.shape) == (B, K, D), str(tuple(out.shape)))
    chk("output is finite", bool(torch.isfinite(out).all()))
    chk("the offsets start spread apart, not collapsed",
        float(stats["deform_offset_spread"]) > 1e-3,
        f"spread = {float(stats['deform_offset_spread']):.4f} px")

    # THE regression test. With the old symmetric init every sample coincides, so
    # the attention over samples has exactly zero gradient and weight_head is dead.
    g_new = float(s.weight_head.weight.grad.abs().max())
    s_old, _, stats_old, _ = _sampler_case("zero")
    g_old = float(s_old.weight_head.weight.grad.abs().max())
    print(f"\n    weight_head gradient: symmetry_broken {g_new:.3e}  vs  zero-init {g_old:.3e}")
    print(f"    offset spread:        symmetry_broken "
          f"{float(stats['deform_offset_spread']):.4f}  vs  zero-init "
          f"{float(stats_old['deform_offset_spread']):.4f}\n")
    chk("zero-init reproduces the old bug: weight_head gets EXACTLY no gradient",
        g_old < 1e-12, f"{g_old:.3e}")
    chk("symmetry_broken fixes it: weight_head gets real gradient",
        g_new > 1e-9, f"{g_new:.3e}")
    chk("and zero-init collapses the offsets to a single point",
        float(stats_old["deform_offset_spread"]) < 1e-9)

    # Every parameter must receive gradient on the first backward. The old version
    # failed this too, because a zeroed output projection starves the whole branch.
    dead = [n for n, p in s.named_parameters()
            if p.grad is None or float(p.grad.abs().max()) == 0.0]
    chk("every sampler parameter receives gradient on step 1",
        not dead, f"dead: {dead}" if dead else "")

    for n_off in (2, 3, 5):
        s2, out2, st2, _ = _sampler_case("symmetry_broken", num_offsets=n_off)
        chk(f"num_offsets={n_off} runs and stays finite", bool(torch.isfinite(out2).all()))
        if n_off > 2:
            chk(f"num_offsets={n_off}: offsets spread",
                float(st2["deform_offset_spread"]) > 1e-3)

    try:
        CurveDeformableSampler(8, 8, num_offsets=1)
        chk("rejects num_offsets=1", False)
    except ValueError:
        chk("rejects num_offsets=1", True)


def test_sampler_moe():
    section("CurveDeformableSampler with MoE on the offsets")
    cfg = dict(num_experts=4, mode="soft")
    s, out, stats, (B, K, D) = _sampler_case("symmetry_broken", moe_cfg=cfg)
    chk("MoE sampler output shape", tuple(out.shape) == (B, K, D))
    chk("router stats surface through the sampler", "moe_entropy" in stats)
    chk("entropy starts at ln(4)", abs(float(stats["moe_entropy"]) - math.log(4)) < 1e-4,
        f"{float(stats['moe_entropy']):.5f}")
    chk("the fan is applied to every expert: offsets still spread",
        float(stats["deform_offset_spread"]) > 1e-3,
        f"{float(stats['deform_offset_spread']):.4f}")
    chk("the router receives gradient", s.offset_head.gate.proj.weight.grad is not None
        and float(s.offset_head.gate.proj.weight.grad.abs().max()) > 1e-12)
    dead = [n for n, p in s.named_parameters()
            if p.grad is None or float(p.grad.abs().max()) == 0.0]
    chk("every parameter still receives gradient with MoE on", not dead,
        f"dead: {dead}" if dead else "")
    try:
        CurveDeformableSampler(8, 8, init_mode="zero", moe_cfg=cfg)
        chk("rejects init_mode='zero' together with moe_cfg", False)
    except ValueError:
        chk("rejects init_mode='zero' together with moe_cfg", True)


def test_gather_dropin():
    section("CurveAlignedDeformableROIGather — drop-in compatibility")
    torch.manual_seed(0)
    B, K, P, C, D, R, S = 2, 6, 36, 10, 16, 3, 18
    g = CurveAlignedDeformableROIGather(C, K, P, D, R, deform_num_curve_samples=S)
    roi_features = [torch.randn(B * K, C, P, 1) for _ in range(2)]
    x = torch.randn(B, C, 20, 50)
    pts, mask = build_clr_curve_reference_points(torch.rand(B, K, P),
                                                torch.linspace(1, 0, P), S)
    out = g(roi_features, x, 1, reference_points=pts, reference_valid_mask=mask)
    chk("the head's exact call signature returns a bare [B, K, D] tensor",
        torch.is_tensor(out) and tuple(out.shape) == (B, K, D), str(tuple(out.shape)))
    chk("stats are exposed for logging", "deform_offset_px" in g.last_stats)

    out.square().sum().backward()
    dead = [n for n, p in g.named_parameters()
            if p.grad is None or float(p.grad.abs().max()) == 0.0]
    chk("every gather parameter receives gradient", not dead, f"dead: {dead}" if dead else "")

    # zero_init runs after the head's blanket re-init and must restore the fan
    # rather than flatten it -- the failure mode that produced 64.20.
    with torch.no_grad():
        for p in g.sampler.parameters():
            nn.init.trunc_normal_(p, std=0.02) if p.ndim > 1 else nn.init.constant_(p, 0.02)
    g.zero_init()
    fan_after = torch.tanh(g.sampler.offset_head.bias.detach())
    chk("zero_init restores the fan after a blanket re-init",
        torch.allclose(fan_after, torch.tanh(offset_fan(4)), atol=1e-5),
        f"{[round(v, 3) for v in fan_after.tolist()]}")
    chk("zero_init still zeroes the composed attention's output gate",
        float(g.attention.W.weight.abs().max()) == 0.0)

    # A non-deformable stage must not need reference points.
    g2 = CurveAlignedDeformableROIGather(C, K, P, D, R, deformable_stages=[2],
                                         deform_num_curve_samples=S)
    chk("a non-deformable stage runs without reference points",
        tuple(g2(roi_features, x, 1).shape) == (B, K, D))
    try:
        g(roi_features, x, 1)
        chk("a deformable stage demands reference points", False)
    except ValueError:
        chk("a deformable stage demands reference points", True)

    # Retired config keys must not crash an old config.
    g3 = CurveAlignedDeformableROIGather(C, K, P, D, R, norm_type="BN",
                                         global_dropout=0.1, use_conv_activation=True,
                                         use_curve_point_query_interaction=False)
    chk("retired config keys are accepted and reported", g3 is not None)
    try:
        CurveAlignedDeformableROIGather(C, K, P, D, R, nonsense_key=1)
        chk("a genuinely unknown key is rejected", False)
    except TypeError:
        chk("a genuinely unknown key is rejected", True)

    g4 = CurveAlignedDeformableROIGather(C, K, P, D, R, deform_num_curve_samples=S,
                                         deform_moe_cfg=dict(num_experts=2, mode="hard",
                                                             top_k=1))
    out4 = g4(roi_features, x, 1, reference_points=pts, reference_valid_mask=mask)
    chk("hard-routed MoE gather runs", tuple(out4.shape) == (B, K, D))
    chk("hard mode surfaces the balance loss", "moe_balance_loss" in g4.last_stats)


def main():
    test_gate()
    test_router_symmetry()
    test_expert_gradients()
    test_stat_merging()
    test_offset_fan()
    test_masked_softmax()
    test_reference_points()
    test_curve_frame()
    test_sampler()
    test_sampler_moe()
    test_gather_dropin()
    print("\nPASS" if OK else "\nFAILURES ABOVE")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
