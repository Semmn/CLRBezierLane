"""Probe the official CLRerNet (mmdet 3.x) interfaces and validate the CLRBezierHead port.

Run inside the official Docker container from the repo root:

    python tools/clrbezier/probe_official.py \
        configs/clrbezier/culane/clrbezier_culane_r34.py \
        --baseline-config configs/clrernet/culane/clrernet_culane_r34.py

It prints where the pipeline stores lane targets and masks, the backbone/neck
output containers, the GT start_y convention, BRR target fit quality, one
training step (losses, gradient sanity), and whether the official predict path
accepts CLRBezierHead outputs. Every check prints PASS/FAIL.
"""
import argparse
import math
import traceback

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet.registry import MODELS


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def status(ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
    return ok


def describe(obj, depth=0, max_depth=2):
    pad = "  " * depth
    if torch.is_tensor(obj):
        return f"{pad}Tensor{tuple(obj.shape)} {obj.dtype}"
    if isinstance(obj, (list, tuple)):
        lines = [f"{pad}{type(obj).__name__}(len={len(obj)})"]
        if depth < max_depth:
            lines += [describe(o, depth + 1, max_depth) for o in list(obj)[:4]]
        return "\n".join(lines)
    if isinstance(obj, dict):
        lines = [f"{pad}dict(keys={list(obj.keys())})"]
        if depth < max_depth:
            lines += [f"{pad}  {k}: " + describe(v, 0, 0) for k, v in list(obj.items())[:12]]
        return "\n".join(lines)
    return f"{pad}{type(obj).__name__}"


def unwrap_single_image(x):
    """mmdet3 get_lanes may return [[lane, ...]] (one list per image)."""
    while isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x)


def to_points(lane):
    if hasattr(lane, "points"):
        lane = lane.points
    if torch.is_tensor(lane):
        lane = lane.detach().cpu()
    return torch.as_tensor(lane, dtype=torch.float64).reshape(-1, 2)


def compare_decoded_with_gt(head, decoded, gt):
    """Match each decoded lane to a GT lane on identical rows; expect ~0 px error."""
    cfg = head.test_cfg
    ori_h, cut = float(cfg.ori_img_h), float(cfg.cut_height)
    ys = head.prior_ys.detach().cpu().double()
    ys_full = (ys * (ori_h - cut) + cut) / ori_h
    gt_xs = gt[:, 6:].detach().cpu().double()
    gt_valid = (gt_xs >= 0) & (gt_xs < head.img_w)
    matched = set()
    all_ok = True
    for i, lane in enumerate(decoded):
        pts = to_points(lane)
        best = (float("inf"), -1, 0)
        for g in range(gt_xs.shape[0]):
            rows = torch.nonzero(gt_valid[g]).squeeze(1)
            gt_pts_y = ys_full[rows]
            gt_pts_x = gt_xs[g, rows] / (head.img_w - 1)
            dy = (pts[:, 1:2] - gt_pts_y.view(1, -1)).abs()
            hit = dy.min(dim=1)
            common = hit.values < 1e-6
            if int(common.sum()) == 0:
                continue
            err_px = ((pts[common, 0] - gt_pts_x[hit.indices[common]]).abs().mean()
                      * (head.img_w - 1)).item()
            coverage = int(common.sum()) / max(1, rows.numel())
            if err_px < best[0]:
                best = (err_px, g, coverage)
        err_px, g, coverage = best
        ok = g >= 0 and err_px < 0.5 and coverage > 0.95 and g not in matched
        matched.add(g)
        all_ok &= ok
        print(f"  decoded lane {i}: matched GT {g}, mean |dx| = {err_px:.4f} px, "
              f"row coverage = {coverage:.2f}, first points = {pts[:2].tolist()}")
    status(all_ok and len(matched) == gt_xs.shape[0],
           "decoded points equal GT on every visible row (start_y convention end-to-end)")


def build(cfg_path, batch_size):
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    model = MODELS.build(cfg.model).cuda()
    model.init_weights()
    loader_cfg = cfg.train_dataloader.copy()
    loader_cfg["batch_size"] = batch_size
    loader_cfg["num_workers"] = 0
    loader_cfg["persistent_workers"] = False
    loader = Runner.build_dataloader(loader_cfg)
    return cfg, model, loader


def _flatten_decoded(obj):
    """Flatten official get_lanes output into a list of [N, 2] numpy point arrays.

    Handles a per-image nested list, (N, 1, 2) tensors, numpy arrays, and Lane
    objects (``.points``).
    """
    import numpy as np

    if obj is None:
        return []
    if hasattr(obj, "points"):
        return [np.asarray(obj.points, dtype=np.float64).reshape(-1, 2)]
    if torch.is_tensor(obj):
        return [obj.detach().cpu().double().numpy().reshape(-1, 2)]
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        return [obj.astype(np.float64).reshape(-1, 2)]
    out = []
    for item in obj:
        out.extend(_flatten_decoded(item))
    return out


@torch.no_grad()
def check_official_decoder_roundtrip(head, image_lanes, device, tol_px=1.0):
    """Feed GT lanes through the official get_lanes as perfect predictions.

    NMS is disabled for this check (neighboring GT lanes can legitimately
    suppress each other). Each decoded point is mapped back to its CLR row and
    compared with the GT x at that row. A correct start_y convention gives
    sub-pixel error for every lane.
    """
    import numpy as np

    t = image_lanes[image_lanes[:, 1] == 1]
    n = int(t.shape[0])
    if n == 0:
        print("first image has no GT lanes; skipped")
        return
    logits = torch.zeros((1, n, 2), device=device)
    logits[..., 1] = 10.0
    pred_dict = {
        "cls_logits": logits,
        "anchor_params": t[:, 2:5].unsqueeze(0),
        "lengths": (t[:, 5:6] / head.n_strips).unsqueeze(0),
        "xs": (t[:, 6:] / (head.img_w - 1)).unsqueeze(0),
    }

    cfg = head.test_cfg
    old_use_nms = cfg.get("use_nms", True)
    cfg["use_nms"] = False
    try:
        raw = head.get_lanes(pred_dict, as_lanes=False)
    finally:
        cfg["use_nms"] = old_use_nms

    print("raw get_lanes return:\n" + describe(raw, max_depth=3))
    # One image in, so a 2-element outer sequence is (lanes, scores).
    lanes_out = raw[0] if isinstance(raw, (tuple, list)) and len(raw) == 2 else raw
    decoded = _flatten_decoded(lanes_out)
    status(len(decoded) == n, f"official get_lanes decoded {len(decoded)} of {n} GT lanes (NMS off)")

    ori_h = float(cfg.get("ori_img_h", 590))
    cut = float(cfg.get("cut_height", 270))
    gt_xs = (t[:, 6:] / (head.img_w - 1)).cpu().double().numpy()
    gt_valid = ((t[:, 6:] >= 0) & (t[:, 6:] < head.img_w)).cpu().numpy()
    all_ok = True
    for i, pts in enumerate(decoded):
        y_crop = (pts[:, 1] * ori_h - cut) / (ori_h - cut)
        rows = np.rint((1.0 - y_crop) * head.n_strips).astype(int)
        inside = (rows >= 0) & (rows <= head.n_strips)
        rows, xs = rows[inside], pts[inside, 0]
        best_err, best_gt = float("inf"), -1
        for g in range(n):
            ok = gt_valid[g, rows]
            if ok.sum() < 2:
                continue
            err = np.abs(xs[ok] - gt_xs[g, rows[ok]]).mean() * (head.img_w - 1)
            if err < best_err:
                best_err, best_gt = err, g
        lane_ok = best_err <= tol_px
        all_ok &= lane_ok
        span = f"rows {rows.min()}-{rows.max()}" if rows.size else "no rows"
        print(f"  decoded lane {i}: {pts.shape[0]} points, {span}, "
              f"best GT {best_gt}, mean |dx| = {best_err:.3f} px")
    status(all_ok and len(decoded) == n,
           f"decoded x matches GT x at the decoded rows (tol {tol_px} px)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    from libs.clrbezier.data_adapters import describe_sample
    from libs.clrbezier.geometry import eval_global_cubic, fit_global_cubic_to_clr_rows

    banner("1. Build model and one training batch")
    cfg, model, loader = build(args.config, args.batch_size)
    head = model.bbox_head
    status(type(head).__name__ == "CLRBezierHead", f"bbox_head type = {type(head).__name__}")
    print("CLRBezierHead MRO:", [c.__name__ for c in type(head).__mro__[:4]])
    batch = next(iter(loader))
    data = model.data_preprocessor(batch, True)
    inputs, samples = data["inputs"], data["data_samples"]
    print("inputs:", describe(inputs))
    print("sample[0]:", describe_sample(samples[0]))

    banner("2. Backbone / neck containers")
    with torch.no_grad():
        bb = model.backbone(inputs)
        print("backbone output:\n" + describe(bb))
        feats = model.extract_feat(inputs)
        print("extract_feat output:\n" + describe(feats))
    try:
        selected = head._select_features(feats)
        status(True, "head feature order " + str([tuple(f.shape[-2:]) for f in selected]))
    except Exception as err:  # noqa: BLE001
        status(False, f"head feature selection: {err}")

    banner("3. Ground truth location and conventions")
    device = inputs.device
    lanes = head.target_adapter.extract_lanes(samples, device)
    print("lane source:", head.target_adapter._lane_source,
          "| convention:", head.target_adapter.start_y_convention,
          "| length unit:", head.target_adapter.length_unit)
    fg = lanes[..., 1] == 1
    sel = lanes[fg]
    valid = (sel[:, 6:] >= 0) & (sel[:, 6:] < head.img_w)
    first = torch.argmax(valid.int(), dim=-1).float()
    start_rows = torch.round((1.0 - sel[:, 2]) * head.n_strips)
    status(bool(((start_rows - first).abs() <= 1).float().mean() > 0.9),
           "canonical start_y (image y) matches first valid row")
    status(bool((valid.sum(-1).float() - sel[:, 5]).abs().max() <= 1.0),
           "length (row count) matches number of valid rows")
    contiguous = all(
        bool((torch.diff(torch.nonzero(v).squeeze(1)) == 1).all()) for v in valid if v.sum() > 1)
    print("valid rows contiguous for every lane:", contiguous)
    seg = head.target_adapter.extract_seg(samples, device, (head.img_h, head.img_w))
    print("seg source:", head.target_adapter._seg_source, "| classes present:",
          torch.unique(seg).tolist())

    banner("4. BRR control-point target fit quality")
    cp, ok = fit_global_cubic_to_clr_rows(sel[:, 6:], head.prior_ys, head.img_w,
                                          head.brr_cp_fit_ridge, head.cp_x_margin)
    fitted = eval_global_cubic(cp, head.prior_ys.to(cp)) * (head.img_w - 1)
    err = ((fitted - sel[:, 6:]).abs() * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    at_margin = ((cp <= -head.cp_x_margin + 1e-6) | (cp >= 1 + head.cp_x_margin - 1e-6)).any(-1)
    print(f"lanes={sel.shape[0]} fit_ok={int(ok.sum())} mean_px_err={float(err.mean()):.2f} "
          f"p95_px_err={float(err.quantile(0.95)) if err.numel() > 1 else float(err.mean()):.2f} "
          f"cp_at_margin={float(at_margin.float().mean()):.3f}")

    banner("5. One training step")
    model.train()
    losses = model.loss(inputs, samples)
    for k, v in losses.items():
        print(f"  {k}: {float(v.mean()):.5f}")
    total = sum(v for k, v in losses.items() if "loss" in k)
    status(bool(torch.isfinite(total)), f"total loss finite ({float(total):.4f})")
    total.backward()
    for name in ["reg_layers.weight", "cls_layers.weight", "prior_bank.prior_delta",
                 "roi_gather.fc.weight"]:
        p = dict(head.named_parameters())[name]
        g = None if p.grad is None else float(p.grad.abs().sum())
        status(g is not None and math.isfinite(g) and g > 0, f"grad {name}: {g}")
    model.zero_grad(set_to_none=True)

    banner("6. Official predict path with CLRBezierHead")
    model.eval()
    ours = None
    try:
        with torch.no_grad():
            ours = model.predict(inputs[:1], samples[:1])
        status(True, "model.predict ran")
        print(describe(ours))
        if isinstance(ours, (list, tuple)) and ours and hasattr(ours[0], "keys"):
            print("pred sample fields:", describe_sample(ours[0]))
    except Exception:  # noqa: BLE001
        status(False, "model.predict raised:")
        traceback.print_exc()

    banner("7. Official decoder round-trip on GT (convention end-to-end)")
    try:
        t = lanes[0][lanes[0][:, 1] == 1]
        n = int(t.shape[0])
        logits = torch.zeros((1, n, 2), device=device)
        logits[..., 1] = 10.0
        pred_dict = {
            "cls_logits": logits,
            "anchor_params": t[:, 2:5].unsqueeze(0),
            "lengths": (t[:, 5:6] / head.n_strips).unsqueeze(0),
            "xs": (t[:, 6:] / (head.img_w - 1)).unsqueeze(0),
        }
        out = head.get_lanes(pred_dict, as_lanes=False)
        print("get_lanes returned:\n" + describe(out, max_depth=3))
        decoded = out[0] if isinstance(out, tuple) and len(out) == 2 else out
        decoded = unwrap_single_image(decoded)
        status(len(decoded) == n,
               f"official get_lanes decoded {len(decoded)} of {n} GT lanes (per-image nesting removed)")
        compare_decoded_with_gt(head, decoded, t)
    except Exception:  # noqa: BLE001
        status(False, "official get_lanes round-trip raised:")
        traceback.print_exc()

    if args.baseline_config:
        banner("8. Baseline CLRerNet predict output structure")
        _, base_model, _ = build(args.baseline_config, 1)
        base_model.eval()
        with torch.no_grad():
            base_data = base_model.data_preprocessor(batch, False)
            ref = base_model.predict(base_data["inputs"][:1], base_data["data_samples"][:1])
        print(describe(ref))
        if ours is not None:
            status(type(ref) is type(ours), "same predict return type as baseline")
            if isinstance(ref, (list, tuple)) and ref and hasattr(ref[0], "keys"):
                status(sorted(ref[0].keys()) == sorted(ours[0].keys()),
                       "same prediction fields as baseline")


if __name__ == "__main__":
    main()
