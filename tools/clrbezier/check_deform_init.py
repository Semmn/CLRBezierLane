"""Verify that the deformable RoI Gather starts on the lane, not beside it.

The head applies the official CLRerNet initialization policy over every module
it owns (kaiming for nn.Conv2d, trunc_normal std 0.02 for nn.Linear). The
deformable module relies on several of its own nn.Linear layers being exactly
zero so that, at iteration 0, it samples on the current curve with uniform
weights. A parent-level re-init silently replaces those zeros unless the module
is re-zeroed afterwards.

V11 never hit this because it ran the official init immediately after
CLRHead.__init__ and constructed the deformable module afterwards.

Usage:
    python tools/clrbezier/check_deform_init.py \
        configs/clrbezier/culane/clrbezier_deform_parity_r34.py

Exit status is non-zero if any layer that must start at zero does not, so this
can go in front of a training run.
"""
import argparse
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS

# Every tensor the deformable branch needs at exactly zero, as a path relative
# to the ROIGather module.
REQUIRED_ZEROS = (
    "global_output_projection.weight",
    "global_output_projection.bias",
    "curve_deformable_sampler.offset_head.weight",
    "curve_deformable_sampler.offset_head.bias",
    "curve_deformable_sampler.weight_head.weight",
    "curve_deformable_sampler.weight_head.bias",
    "curve_deformable_sampler.point_pooling_score.1.weight",
    "curve_deformable_sampler.point_pooling_score.1.bias",
    "curve_deformable_sampler.curve_pointwise_conv.weight",
    "curve_deformable_sampler.curve_pointwise_conv.bias",
    "curve_deformable_sampler.output_projection.weight",
    "curve_deformable_sampler.output_projection.bias",
)


def resolve(module, path):
    obj = module
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    model = MODELS.build(cfg.model)

    head = model.bbox_head
    gather = head.roi_gather
    if type(gather).__name__ != "CurveAlignedDeformableROIGather":
        print(f"roi_gather is {type(gather).__name__}; nothing to check.")
        return 0

    print(f"mid_channels          : {getattr(gather, 'mid_channels', '?')}")
    print(f"convs[0] out_channels : {gather.convs[0].out_channels}")
    print(f"catconv[0] in_channels: {gather.catconv[0].in_channels}")
    print(f"conv activation       : {gather.convs[0].activation is not None}")
    print(f"look_forward_twice    : {head.look_forward_twice}")
    print(f"detach_reference      : {getattr(head, 'deform_detach_reference', '?')}")
    print()

    failures = []
    for path in REQUIRED_ZEROS:
        tensor = resolve(gather, path)
        if tensor is None:
            print(f"  skip  {path} (not present in this configuration)")
            continue
        max_abs = tensor.detach().abs().max().item()
        status = "ok  " if max_abs == 0.0 else "FAIL"
        print(f"  {status}  {path:58s} max|w| = {max_abs:.3e}")
        if max_abs != 0.0:
            failures.append(path)

    # The catconv / convs channel contract: convs must reduce to mid_channels,
    # because catconv consumes mid_channels * (stage + 1).
    for stage in range(head.refine_layers):
        expected = gather.mid_channels * (stage + 1)
        actual = gather.catconv[stage].in_channels
        if expected != actual:
            failures.append(f"catconv[{stage}] expects {expected}, got {actual}")

    print()
    if failures:
        print("FAILED:")
        for item in failures:
            print(f"  {item}")
        print("\nThe deformable branch does not start on the current curve. Check that")
        print("zero_init_outputs is True and that head init calls roi_gather.zero_init().")
        return 1

    print("All zero-initialized layers are zero; the branch starts on the curve.")

    # Second check: with look_forward_twice on, does the sampling grid carry a
    # gradient back into the previous stage's reference?
    if head.look_forward_twice and not getattr(head, "deform_detach_reference", True):
        print("\nNOTE: look_forward_twice=True and detach_reference=False, so the")
        print("18 x num_offsets deformable sampling points send a feature-space")
        print("gradient into the previous stage's control points. V11 never had")
        print("that path. Set detach_reference=True to remove it.")
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
