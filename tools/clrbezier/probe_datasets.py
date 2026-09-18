"""Probe TuSimple / LLAMAS / CurveLanes support before training.

    python tools/clrbezier/probe_datasets.py configs/clrernet/tusimple/clrernet_tusimple_r34.py
    python tools/clrbezier/probe_datasets.py configs/clrbezier/curvelanes/clrbezier_curvelanes_r34.py

1. Training pipeline: a few augmented samples are packed; checks the image
   tensor, CLR lane targets, and segmentation classes, and saves overlays.
2. GT round trip: validation GT -> official val augmentation ->
   PackCLRNetInputs targets -> the model head's official ``get_lanes`` ->
   the configured metric (``partial_eval``). A correct crop / test_cfg / metric
   chain scores close to 100%. Any convention error shows up here as a
   low score.
"""
import argparse
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS, METRICS, MODELS

from libs.datasets.pipelines import Compose


def status(ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
    return ok


def pack_kwargs(pipeline_cfg):
    pack = [t for t in pipeline_cfg if t["type"] == "PackCLRNetInputs"][0]
    return dict(pack)


def draw_training_sample(data, out_path, n_strips=71):
    img = data["inputs"].permute(1, 2, 0).cpu().numpy().copy().astype(np.uint8)
    h, w = img.shape[:2]
    meta = data["data_samples"].metainfo
    lanes = meta["lanes"].cpu().numpy()
    masks = meta.get("gt_masks")
    if masks is not None:
        mask = np.asarray(masks[0])
        if mask.shape[:2] == (h, w):
            color = np.zeros_like(img)
            color[mask > 0] = (0, 0, 255)
            img = cv2.addWeighted(img, 1.0, color, 0.4, 0)
    ys = h - np.arange(n_strips + 1) * (h / n_strips)  # row 0 = bottom
    for lane in lanes[lanes[:, 1] == 1]:
        xs = lane[6:]
        for x, y in zip(xs, ys):
            if 0 <= x < w:
                cv2.circle(img, (int(x), int(min(y, h - 1))), 2, (0, 255, 0), -1)
    cv2.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--num-train", type=int, default=6)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--out-dir", default="work_dirs/probe_datasets")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(0)

    # ------------------------------------------------------------------ 1
    if not args.skip_train:
        print("\n=== 1. Training pipeline ===")
        train_ds = DATASETS.build(cfg.train_dataloader.dataset)
        print(f"train samples: {len(train_ds)}")
        max_lanes = pack_kwargs(cfg.train_pipeline)["max_lanes"]
        for k, idx in enumerate(random.sample(range(len(train_ds)), args.num_train)):
            data = train_ds[idx]
            meta = data["data_samples"].metainfo
            lanes = meta["lanes"]
            n_valid = int((lanes[:, 1] == 1).sum())
            status(tuple(data["inputs"].shape) == (3, 320, 800), f"[{idx}] image tensor {tuple(data['inputs'].shape)}")
            status(lanes.shape == (max_lanes, 78), f"[{idx}] lanes {tuple(lanes.shape)}, valid {n_valid}, gt_points {len(meta['gt_points'])}")
            masks = meta.get("gt_masks")
            if masks is not None:
                print(f"      seg classes: {np.unique(np.asarray(masks[0])).tolist()}")
            out = osp.join(args.out_dir, f"train_{k}.png")
            draw_training_sample(data, out)
        print(f"overlays written to {args.out_dir}")

    # ------------------------------------------------------------------ 2
    print("\n=== 2. GT round trip (val GT -> targets -> official get_lanes -> metric) ===")
    val_ds = DATASETS.build(cfg.val_dataloader.dataset)
    model = MODELS.build(cfg.model)
    head = model.bbox_head
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head.to(device).eval()
    print(f"head: {type(head).__name__}, test_cfg: {dict(head.test_cfg)}")

    val_pack = pack_kwargs(cfg.val_pipeline)
    val_pack["meta_keys"] = list(val_pack["meta_keys"]) + ["lanes"]
    val_pack["max_lanes"] = pack_kwargs(cfg.train_pipeline)["max_lanes"]
    val_pack.pop("type")
    rt_pipeline = Compose([
        dict(type="albumentation", pipelines=cfg.val_al_pipeline),
        dict(type="PackCLRNetInputs", **val_pack),
    ])

    metric_cfg = dict(cfg.val_evaluator)
    if metric_cfg["type"] == "CULaneMetric":
        print("CULaneMetric has no partial evaluation; round trip skipped.")
        return
    if metric_cfg["type"] in ("TuSimpleMetric", "LLAMASMetric"):
        metric_cfg["partial_eval"] = True  # CurvelanesMetric already scores processed images only
    metric = METRICS.build(metric_cfg)

    indices = sorted(random.sample(range(len(val_ds)), min(args.num_val, len(val_ds))))
    n_gt, n_pred = 0, 0
    for idx in indices:
        packed = rt_pipeline(val_ds.results_with_labels(idx))
        meta = packed["data_samples"].metainfo
        t = meta["lanes"].to(device)
        t = t[t[:, 1] == 1]
        n_gt += int(t.shape[0])
        logits = torch.zeros((1, t.shape[0], 2), device=device)
        logits[..., 1] = 10.0
        pred_dict = dict(
            cls_logits=logits,
            anchor_params=t[:, 2:5].unsqueeze(0),
            lengths=(t[:, 5:6] / head.n_strips).unsqueeze(0),
            xs=(t[:, 6:] / (head.img_w - 1)).unsqueeze(0),
        )
        with torch.no_grad():
            lanes, scores = head.get_lanes(pred_dict, as_lanes=True,
                                           extend_bottom=head.test_cfg.get("extend_bottom", True))
        lanes, scores = lanes[0], scores[0]
        n_pred += len(lanes)
        metric.process({}, [dict(lanes=lanes, scores=scores, metainfo=meta)])

    print(f"GT lanes {n_gt}, decoded lanes {n_pred} over {len(indices)} images")
    result = metric.compute_metrics(metric.results)
    print("round-trip metric:", result)
    key = next((k for k in ("Accuracy", "F1_0.50", "F1") if k in result), None)
    if key is not None:
        status(result[key] > 0.95, f"round-trip {key} = {result[key]:.4f} (expect > 0.95)")


if __name__ == "__main__":
    main()
