"""Is the under-bending a read-out failure or a representation failure?

The question
------------
The model produces about 37% of the bow its curved GT lanes need. Two very
different causes, with opposite implications for where a mixture of experts
belongs:

  read-out failure       The pooled features DO encode that this lane is sharply
                         curved, and one shared linear map regresses to the mean
                         of a 98.8%-straight distribution. A conditionally
                         parameterized read-out (MoELinear) is then exactly the
                         fix, and the head is the right place.

  representation failure The features do not encode it, because ROI sampling
                         follows a near-straight reference and never looks where
                         the true lane goes. No read-out can recover it; the
                         mixture has to sit upstream, on the sampling offsets or
                         in GSRC.

This probes the frozen features that feed ``reg_layers`` for the residual bow --
the part of the GT's curvature the model failed to produce. High predictability
means the information is present and unused.

The controls are the point
--------------------------
A probe alone proves nothing: a feature that merely encodes "this lane is long
and starts here" would predict curvature on CULane through correlation. So the
features are scored against two nulls:

  geometry-only  ridge on [y_start, length, predicted bow] alone -- the things
                 the head already has explicitly. The features have to BEAT this
                 to be carrying anything new.
  shuffled       targets permuted across samples. Fixes the zero line, and
                 catches a leak if it ever reads above 0.

Split is by IMAGE, never by sample: several positives match the same GT lane, so
a sample-level split would put copies of one target on both sides.

    python tools/clrbezier/probe_bow.py CONFIG CKPT --max-images 3000
    python tools/clrbezier/probe_bow.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diagnose_assignment import straight_fit_bow  # noqa: E402  single source of truth


def ridge_r2(x_tr, y_tr, x_te, y_te, alpha=1.0):
    """Closed-form ridge, held-out R^2. Deterministic -- no optimizer, no seed.

    The bias column is appended and left unpenalized, otherwise the intercept
    shrinks toward zero and R^2 reads low for a reason that has nothing to do
    with the features.
    """
    def design(x):
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    a_tr, a_te = design(x_tr), design(x_te)
    d = a_tr.shape[1]
    penalty = np.eye(d) * alpha
    penalty[-1, -1] = 0.0
    w = np.linalg.solve(a_tr.T @ a_tr + penalty, a_tr.T @ y_tr)
    pred = a_te @ w
    sse = float(((y_te - pred) ** 2).sum())
    sst = float(((y_te - y_te.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def standardize(x_tr, x_te):
    mu, sd = x_tr.mean(0), x_tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x_tr - mu) / sd, (x_te - mu) / sd


def evaluate(features, geometry, target, image_id, alpha=1.0, holdout=0.3, seed=0):
    """Fit on one set of images, score on another. Returns a dict of R^2."""
    rng = np.random.default_rng(seed)
    images = np.unique(image_id)
    rng.shuffle(images)
    n_te = max(1, int(round(len(images) * holdout)))
    te_imgs = set(images[:n_te].tolist())
    te = np.array([i in te_imgs for i in image_id])
    tr = ~te
    if tr.sum() < 20 or te.sum() < 10:
        raise ValueError(f"not enough samples to split: {tr.sum()} train, {te.sum()} test")

    f_tr, f_te = standardize(features[tr], features[te])
    g_tr, g_te = standardize(geometry[tr], geometry[te])
    y_tr, y_te = target[tr], target[te]
    out = {
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_images": len(images), "target_std": float(y_te.std()),
        "features": ridge_r2(f_tr, y_tr, f_te, y_te, alpha),
        "geometry": ridge_r2(g_tr, y_tr, g_te, y_te, alpha),
        "features_plus_geometry": ridge_r2(
            np.concatenate([f_tr, g_tr], 1), y_tr,
            np.concatenate([f_te, g_te], 1), y_te, alpha),
        "shuffled": ridge_r2(f_tr, rng.permutation(y_tr), f_te, y_te, alpha),
    }
    out["features_over_geometry"] = out["features_plus_geometry"] - out["geometry"]
    return out


def report(res, unit="px"):
    print(f"\n{res['n_train']} train / {res['n_test']} held-out positives "
          f"from {res['n_images']} images. Residual bow std {res['target_std']:.2f} {unit}.\n")
    print(f"{'probe':<26} {'held-out R^2':>13}")
    print("-" * 41)
    for key, label in (("shuffled", "shuffled targets (null)"),
                       ("geometry", "geometry only"),
                       ("features", "pooled features"),
                       ("features_plus_geometry", "features + geometry")):
        print(f"{label:<26} {res[key]:13.4f}")
    print("-" * 41)
    print(f"{'features beyond geometry':<26} {res['features_over_geometry']:+13.4f}")

    gain = res["features_over_geometry"]
    print()
    if res["shuffled"] > 0.05:
        print("WARNING: the shuffled null reads above zero. The split is leaking -- "
              "check that image_id really\nseparates train from test before "
              "believing anything above.")
        return
    if res["features"] < 0.05 and res["geometry"] < 0.05:
        print("VERDICT: the residual bow is not predictable from the features OR from the\n"
              "geometry the head already has. Neither MoE placement addresses it, and the\n"
              "cause is upstream of both -- run diagnose_assignment.py if you have not.")
    elif gain > 0.05:
        print("VERDICT: the features carry curvature information the read-out is not using.\n"
              "This is a READ-OUT failure, so a conditionally parameterized read-out\n"
              "(MoELinear on reg_layers) is the targeted fix and the head is the right place.")
    else:
        print("VERDICT: the features add nothing beyond y_start, length and the predicted bow,\n"
              "which the head already has explicitly. This is a REPRESENTATION failure: the\n"
              "information never survived ROI sampling. Put the mixture upstream -- on the\n"
              "sampling offsets (MoE over where to look) or in GSRC. A head-side mixture\n"
              "would be conditioning on information that is not there.")


# --------------------------------------------------------------------------


def run(args):
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint
    from mmdet.registry import MODELS

    from libs.clrbezier.assigners import build_cost_cache

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    head = model.bbox_head

    # The features that actually feed the regression read-out, after reg_modules.
    captured = {}

    def grab(_module, inputs):
        captured["reg_f"] = inputs[0].detach()

    handle = head.reg_layers.register_forward_pre_hook(grab)

    loader_cfg = copy.deepcopy(cfg.train_dataloader)
    loader_cfg["batch_size"] = args.batch_size
    loader = Runner.build_dataloader(loader_cfg)

    prior_ys_px = head.prior_ys.detach().cpu().numpy() * float(head.img_h)
    stage = head.refine_layers - 1
    feats_list, geom_list, targ_list, img_list = [], [], [], []
    seen, image_counter = 0, 0

    with torch.no_grad():
        for data in loader:
            data = model.data_preprocessor(data, True)
            backbone = head._select_features(model.extract_feat(data["inputs"]))
            outs = head.forward_train(backbone)
            if "reg_f" not in captured:
                raise RuntimeError("the reg_layers hook never fired; the head's forward "
                                   "path changed and this probe needs updating")
            reg_f = captured["reg_f"]
            num_q = outs["main_preds"][stage].shape[1]
            # reg_f is [B*K, D] for the main branch, flattened in the same order.
            reg_f = reg_f[: data["inputs"].shape[0] * num_q].reshape(
                data["inputs"].shape[0], num_q, -1).cpu().numpy()

            lanes = head.target_adapter.extract_lanes(data["data_samples"], device)
            valid_targets = [t[t[:, 1] == 1] for t in lanes]
            pred = outs["main_preds"][stage].detach()

            for b, target in enumerate(valid_targets):
                image_counter += 1
                if target.shape[0] == 0:
                    continue
                assigner = head.main_stage_assigners.get(stage, head.main_assigner)
                cache = build_cost_cache(pred[b], target, head.img_w, head.img_h,
                                         head.lane_width, head.lane_width_cost,
                                         set(assigner.required_cache_keys) | {"lane_iou_dynamic"},
                                         iou_fns=head.iou_fns)
                rows, cols = assigner.assign(cache)
                if rows.numel() == 0:
                    continue
                pred_xs = pred[b, :, 6:].cpu().numpy() * float(head.img_w - 1)
                t_xs = target[:, 6:].cpu().numpy()
                p_state = outs["main_states"][stage][b].detach().cpu().numpy()

                for row, col in zip(rows.cpu().numpy(), cols.cpu().numpy()):
                    gt_bow = straight_fit_bow(t_xs[col], prior_ys_px, head.img_w)
                    pr_bow = straight_fit_bow(pred_xs[row], prior_ys_px, head.img_w)
                    if gt_bow is None or pr_bow is None:
                        continue
                    feats_list.append(reg_f[b, row])
                    # What the head already has explicitly: start, extent, and the
                    # bow it did produce. The features must beat this to count.
                    geom_list.append([p_state[row, 0], p_state[row, 1], pr_bow])
                    targ_list.append(gt_bow - pr_bow)
                    img_list.append(image_counter)

            seen += len(valid_targets)
            if seen % 500 < len(valid_targets):
                print(f"  {seen} images, {len(targ_list)} positives", flush=True)
            if args.max_images is not None and seen >= args.max_images:
                break
    handle.remove()

    if len(targ_list) < 50:
        print(f"only {len(targ_list)} positives collected; raise --max-images")
        return 1
    res = evaluate(np.asarray(feats_list, dtype=np.float64),
                   np.asarray(geom_list, dtype=np.float64),
                   np.asarray(targ_list, dtype=np.float64),
                   np.asarray(img_list), alpha=args.alpha, holdout=args.holdout)
    report(res)
    return 0


# --------------------------------------------------------------------------


def self_test():
    """Validates the ridge, the R^2 and the by-image split on constructed signal."""
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""))
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    n_img, per_img, d = 240, 4, 24
    image_id = np.repeat(np.arange(n_img), per_img)
    n = len(image_id)

    # Case A: the target is a linear function of the features and nothing else.
    w = rng.normal(size=d)
    feats = rng.normal(size=(n, d))
    geom = rng.normal(size=(n, 3))
    y = feats @ w + rng.normal(scale=0.1, size=n)
    res = evaluate(feats, geom, y, image_id)
    chk("recoverable signal -> features R^2 near 1", res["features"] > 0.95,
        f"{res['features']:.4f}")
    chk("geometry alone explains nothing", abs(res["geometry"]) < 0.1,
        f"{res['geometry']:.4f}")
    chk("shuffled null sits at or below zero", res["shuffled"] < 0.05,
        f"{res['shuffled']:.4f}")
    chk("features beyond geometry is large", res["features_over_geometry"] > 0.8,
        f"{res['features_over_geometry']:+.4f}")

    # Case B: the target depends ONLY on geometry, and the features are a noisy
    # copy of it. The verdict must be "representation failure", i.e. no gain.
    geom2 = rng.normal(size=(n, 3))
    y2 = geom2 @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=n)
    feats2 = np.concatenate([geom2 + rng.normal(scale=0.5, size=(n, 3)),
                             rng.normal(size=(n, d - 3))], axis=1)
    res2 = evaluate(feats2, geom2, y2, image_id)
    chk("geometry-driven target -> geometry R^2 high", res2["geometry"] > 0.9,
        f"{res2['geometry']:.4f}")
    chk("and the features add little beyond it", res2["features_over_geometry"] < 0.05,
        f"{res2['features_over_geometry']:+.4f}")

    # Case C: pure noise target -> everything at zero.
    res3 = evaluate(feats, geom, rng.normal(size=n), image_id)
    chk("unpredictable target -> every probe near zero",
        max(res3["features"], res3["geometry"], res3["shuffled"]) < 0.1,
        f"max {max(res3['features'], res3['geometry'], res3['shuffled']):.4f}")

    # The split must be by image: no image may appear on both sides.
    rng2 = np.random.default_rng(0)
    imgs = np.unique(image_id); rng2.shuffle(imgs)
    te = set(imgs[:max(1, int(round(len(imgs) * 0.3)))].tolist())
    tr_imgs = {i for i in image_id if i not in te}
    chk("train and test image sets are disjoint", not (tr_imgs & te))

    # A sample-level split would leak; show that it inflates R^2 on a target
    # that is constant within an image (which the real residual bow nearly is).
    y_img = np.repeat(rng.normal(size=n_img), per_img)
    feats_img = np.repeat(rng.normal(size=(n_img, d)), per_img, axis=0)
    by_image = evaluate(feats_img, geom, y_img, image_id)["features"]
    sample_split = evaluate(feats_img, geom, y_img, np.arange(n))["features"]
    print(f"\n  by-image split R^2 {by_image:.4f} vs sample split {sample_split:.4f} "
          f"on an image-constant target")
    chk("the by-image split is the conservative one", by_image <= sample_split + 1e-9)

    print("\nPASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--max-images", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.config and args.checkpoint):
        ap.error("CONFIG and CKPT are required unless --self-test")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
