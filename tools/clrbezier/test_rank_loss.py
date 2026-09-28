"""Self-test for the cluster ranking loss.

A ranking loss is exactly the kind of thing that silently trains the wrong
direction: flip one sign and it still decreases, still looks healthy, and
quietly degrades the model. These are the checks that catch that, on synthetic
data with a known answer.

Every expectation here is *derived* from the LaneIoU geometry rather than
hardcoded. The first version of this file asserted a pair count of 3 from
eyeballing the lane offsets, and the real answer was 2 — the test failed
against correct code. Deriving it keeps the test honest when the width or the
threshold changes.

    python tools/clrbezier/test_rank_loss.py
"""
import itertools
import math

import torch

from libs.clrbezier.lane_iou import pairwise_lane_iou
from libs.clrbezier.ranking import batch_cluster_rank_loss, cluster_rank_loss

LANE_WIDTH = 7.5 / 800
IMG_W, IMG_H = 800, 320


def straight_lane(x0, slope=0.0, rows=72):
    """x per row, bottom -> top, normalized by img_w - 1."""
    t = torch.linspace(0.0, 1.0, rows)
    return (x0 + slope * t).clamp(0.0, 0.999)


def mutual_iou(xs):
    with torch.no_grad():
        m = pairwise_lane_iou(xs, xs, LANE_WIDTH, IMG_W, IMG_H)
    return torch.nan_to_num(m, nan=0.0)


def expected_pairs(xs, quality, cluster_iou_thr, margin):
    """The pairs the rule should select, computed independently of the loss."""
    m = mutual_iou(xs)
    out = []
    for i, j in itertools.permutations(range(xs.shape[0]), 2):
        if i < j or True:  # oriented: i beats j
            if m[i, j] > cluster_iou_thr and (quality[i] - quality[j]) > margin:
                out.append((i, j))
    return out


def report(name, ok):
    print(("  ok   " if ok else "  FAIL ") + name)
    return ok


def main():
    torch.manual_seed(0)
    ok = True

    # Two vertical lanes separated by d, each of half-width w, have
    # IoU = (2w - d) / (2w + d). At w = 7.5/800 that crosses 0.35 at d = 7.8 px,
    # so lanes 8 px apart do NOT cluster and lanes 4 px apart do.
    w = LANE_WIDTH
    print(f"half-width {w * 800:.1f} px; IoU 0.35 at separation "
          f"{(1.35 / 1.3) * w * 800:.1f} px\n")

    # Three near-duplicates of one lane plus one far away.
    xs = torch.stack([straight_lane(0.50), straight_lane(0.505),
                      straight_lane(0.495), straight_lane(0.90)])
    quality = torch.tensor([0.90, 0.60, 0.30, 0.80])
    m = mutual_iou(xs)
    print("mutual LaneIoU:")
    for row in m.tolist():
        print("   " + "  ".join(f"{v:5.3f}" for v in row))
    print()

    # 1. Correct ordering must cost less than the reversed one.
    good = torch.tensor([3.0, 2.0, 1.0, 2.5])
    bad = torch.tensor([1.0, 2.0, 3.0, 2.5])
    loss_good, stats_good = cluster_rank_loss(good, xs, quality)
    loss_bad, stats_bad = cluster_rank_loss(bad, xs, quality)
    print(f"correct order  loss={float(loss_good):.4f} "
          f"pair_acc={float(stats_good['rank_pair_acc']):.2f}")
    print(f"reversed order loss={float(loss_bad):.4f} "
          f"pair_acc={float(stats_bad['rank_pair_acc']):.2f}")
    ok &= report("reversed ordering is penalized", float(loss_good) < float(loss_bad))
    ok &= report("pair accuracy tracks the ordering",
                 float(stats_good["rank_pair_acc"]) > 0.99
                 and float(stats_bad["rank_pair_acc"]) < 0.01)

    # Closed form, so a silent change to the weighting or temperature shows up.
    softplus = lambda z: math.log1p(math.exp(z))
    gaps = [0.90 - 0.60, 0.90 - 0.30]
    total = sum(gaps)
    want = sum((g / total) * softplus(-d / 0.5) for g, d in zip(gaps, [1.0, 2.0]))
    ok &= report(f"loss matches closed form ({want:.4f})",
                 abs(float(loss_good) - want) < 1e-4)

    # 2. Cluster membership, derived rather than assumed.
    want_pairs = expected_pairs(xs, quality, 0.35, 0.05)
    _, stats = cluster_rank_loss(good, xs, quality, cluster_iou_thr=0.35)
    print(f"\npairs formed {int(stats['rank_pairs'])}, derived {len(want_pairs)}: "
          f"{want_pairs}")
    ok &= report("pair set matches the geometry",
                 int(stats["rank_pairs"]) == len(want_pairs))
    ok &= report("the distant lane never joins the cluster",
                 all(3 not in pair for pair in want_pairs))

    # 3. The threshold is what gates membership: lanes 8 px apart join at 0.25.
    _, loose = cluster_rank_loss(good, xs, quality, cluster_iou_thr=0.25)
    ok &= report("lowering cluster_iou_thr admits the 8 px pair",
                 int(loose["rank_pairs"]) > int(stats["rank_pairs"]))

    # 4. A genuine three-member cluster gives three pairs.
    tight = torch.stack([straight_lane(0.500), straight_lane(0.503),
                         straight_lane(0.506)])
    tight_q = torch.tensor([0.90, 0.60, 0.30])
    _, tstats = cluster_rank_loss(torch.tensor([3.0, 2.0, 1.0]), tight, tight_q)
    print(f"\nthree mutually-clustered lanes -> {int(tstats['rank_pairs'])} pairs")
    ok &= report("a 3-member cluster yields 3 pairs", int(tstats["rank_pairs"]) == 3)

    # 5. Invariance. The loss reads score *differences*, so an additive shift
    #    leaves it exactly unchanged — that is what keeps the confidence
    #    threshold from moving. A multiplicative rescale is a sharpness change,
    #    not a level change, so it does move the loss; only the ordering (and
    #    hence pair accuracy) is invariant under both.
    shifted, _ = cluster_rank_loss(good + 5.0, xs, quality)
    ok &= report("loss is invariant to an additive shift",
                 abs(float(shifted) - float(loss_good)) < 1e-6)
    _, scaled_stats = cluster_rank_loss(good * 7.0, xs, quality)
    ok &= report("pair accuracy is invariant to a positive rescale",
                 abs(float(scaled_stats["rank_pair_acc"])
                     - float(stats_good["rank_pair_acc"])) < 1e-6)

    # 6. Near-equal qualities are noise and must be skipped.
    flat = torch.tensor([0.90, 0.89, 0.91, 0.80])
    _, fstats = cluster_rank_loss(good, xs, flat, margin=0.05)
    ok &= report("the margin filters uninformative pairs",
                 int(fstats["rank_pairs"]) == 0)

    # 7. Gradient direction: up for the better candidate, down for the worse.
    scores = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss, _ = cluster_rank_loss(scores, xs[:3], quality[:3])
    loss.backward()
    grad = scores.grad
    print(f"\ngrad {[round(g, 4) for g in grad.tolist()]}")
    ok &= report("gradient raises the best candidate and lowers the worst",
                 bool(grad[0] < 0 and grad[2] > 0))

    # 8. Batch wrapper and the degenerate cases real batches contain.
    batch_xs = xs.unsqueeze(0).expand(2, -1, -1).contiguous()
    loss, agg = batch_cluster_rank_loss(
        good.unsqueeze(0).expand(2, -1).contiguous(), batch_xs,
        quality.unsqueeze(0).expand(2, -1).contiguous())
    ok &= report("batch loss matches the per-image loss",
                 abs(float(loss) - float(loss_good)) < 1e-6)
    empty, _ = batch_cluster_rank_loss(torch.zeros(2, 4), batch_xs, torch.zeros(2, 4))
    ok &= report("an image with no usable pairs contributes nothing",
                 float(empty) == 0.0)

    print("\nPASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
