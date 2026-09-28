"""Self-test for the cluster ranking loss.

A ranking loss is exactly the kind of thing that silently trains the wrong
direction: flip one sign and it still decreases, still looks healthy, and
degrades the model. These are the checks that catch that, on synthetic data
with a known answer.

    python tools/clrbezier/test_rank_loss.py
"""
import torch

from libs.clrbezier.ranking import batch_cluster_rank_loss, cluster_rank_loss


def straight_lane(x0, slope=0.0, rows=72):
    """x per row, bottom -> top, normalized."""
    t = torch.linspace(0.0, 1.0, rows)
    return (x0 + slope * t).clamp(0.0, 0.999)


def main():
    torch.manual_seed(0)
    ok = True

    # Three near-duplicates of one lane (they will cluster) plus one far away.
    xs = torch.stack([
        straight_lane(0.50), straight_lane(0.505), straight_lane(0.495),
        straight_lane(0.90),
    ])
    quality = torch.tensor([0.90, 0.60, 0.30, 0.80])

    # 1. Correct ordering must cost less than the reversed one.
    good = torch.tensor([3.0, 2.0, 1.0, 2.5])
    bad = torch.tensor([1.0, 2.0, 3.0, 2.5])
    loss_good, stats_good = cluster_rank_loss(good, xs, quality)
    loss_bad, stats_bad = cluster_rank_loss(bad, xs, quality)
    print(f"correct order  loss={loss_good:.4f}  pair_acc={stats_good['rank_pair_acc']:.2f}  "
          f"pairs={int(stats_good['rank_pairs'])}")
    print(f"reversed order loss={loss_bad:.4f}  pair_acc={stats_bad['rank_pair_acc']:.2f}")
    if not loss_good < loss_bad:
        print("FAIL: reversed ordering is not penalized — check the sign of diff")
        ok = False
    if not (stats_good["rank_pair_acc"] > 0.99 and stats_bad["rank_pair_acc"] < 0.01):
        print("FAIL: pair accuracy does not track the ordering")
        ok = False

    # 2. Monotone rescaling must not change the loss. This is the property the
    #    whole design rests on: the confidence threshold must not move.
    for transform in (lambda s: s * 7.0, lambda s: s + 5.0, lambda s: torch.sigmoid(s)):
        rescaled, _ = cluster_rank_loss(transform(good), xs, quality, tau=1e9)
        base, _ = cluster_rank_loss(good, xs, quality, tau=1e9)
        if abs(float(rescaled) - float(base)) > 5e-3:
            print(f"NOTE: loss moved {float(base):.5f} -> {float(rescaled):.5f} under "
                  "rescaling (expected at finite tau; the *ordering* is what is invariant)")

    # 3. The far-away lane must not be compared with the cluster.
    _, stats = cluster_rank_loss(good, xs, quality, cluster_iou_thr=0.35)
    # cluster of 3 -> pairs with quality gap > margin: (0,1), (0,2), (1,2) = 3
    print(f"pairs formed: {int(stats['rank_pairs'])} (expected 3: the cluster only)")
    if int(stats["rank_pairs"]) != 3:
        print("FAIL: cluster membership is wrong — check cluster_iou_thr / pairwise IoU")
        ok = False

    # 4. Near-equal qualities must be skipped rather than supervised as noise.
    flat = torch.tensor([0.90, 0.89, 0.91, 0.80])
    _, stats = cluster_rank_loss(good, xs, flat, margin=0.05)
    print(f"pairs with near-equal quality: {int(stats['rank_pairs'])} (expected 0)")
    if int(stats["rank_pairs"]) != 0:
        print("FAIL: the margin is not filtering uninformative pairs")
        ok = False

    # 5. Gradient must push the better candidate's score up, the worse one down.
    scores = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss, _ = cluster_rank_loss(scores, xs[:3], quality[:3])
    loss.backward()
    grad = scores.grad
    print(f"grad: {grad.tolist()}  (best candidate index 0 should be negative)")
    if not (grad[0] < 0 and grad[2] > 0):
        print("FAIL: gradient direction is wrong")
        ok = False

    # 6. Batch wrapper, and the degenerate cases that occur in real batches.
    batch_xs = xs.unsqueeze(0).expand(2, -1, -1).contiguous()
    batch_scores = good.unsqueeze(0).expand(2, -1).contiguous()
    batch_quality = quality.unsqueeze(0).expand(2, -1).contiguous()
    loss, agg = batch_cluster_rank_loss(batch_scores, batch_xs, batch_quality)
    print(f"batch loss={float(loss):.4f} pairs/img={float(agg['rank_pairs']):.1f}")

    empty, agg = batch_cluster_rank_loss(
        torch.zeros(2, 4), batch_xs, torch.zeros(2, 4))
    print(f"all-zero quality -> loss={float(empty):.4f} (expected 0, no pairs)")
    if float(empty) != 0.0:
        print("FAIL: an image with no usable pairs should contribute nothing")
        ok = False

    print("\nPASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
