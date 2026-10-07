"""mmengine metrics wrapping the reference evaluation code.

Input: the official ``CLRerHead.predict`` output, i.e. one dict per image with
``lanes`` (``Lane`` objects, x normalized by the original width, y normalized
by the original height) and ``metainfo``.

    TuSimpleMetric    CLRNet TuSimple.pred2lanes + LaneEval.bench_one_submit
    LLAMASMetric      CLRNet LLAMAS.get_prediction_string + llamas_metric.eval_predictions
    CurvelanesMetric  curvelane-branch CurvelanesDataset.evaluate (vega LaneMetricCore)

``partial_eval=True`` evaluates only the processed images (GT round-trip probe).
Never use it for reported numbers.
"""
import json
import os
import os.path as osp
import shutil
import sys
import tempfile
from typing import Sequence

import numpy as np
from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from libs.utils.lane_utils import Lane



# =============================================================================
# TuSimple
# =============================================================================
@METRICS.register_module()
class TuSimpleMetric(BaseMetric):
    def __init__(self, gt_json, ori_img_w=1280, ori_img_h=720,
                 h_samples=tuple(range(160, 720, 10)), output_dir=None,
                 partial_eval=False, collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.gt_json = gt_json
        self.ori_img_w = float(ori_img_w)
        self.ori_img_h = float(ori_img_h)
        self.h_samples = list(h_samples)
        self.output_dir = output_dir
        self.partial_eval = bool(partial_eval)

    def pred2lanes(self, pred):
        # CLRNet TuSimple.pred2lanes
        ys = np.array(self.h_samples) / self.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())
        return lanes

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for result in data_samples:
            self.results.append(dict(
                raw_file=result["metainfo"]["sub_img_name"],
                lanes=self.pred2lanes(result["lanes"]),
                run_time=1.0,  # CLRNet default: 1e-3 s -> 1 ms
            ))

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        temp_dir = None if self.output_dir else tempfile.mkdtemp(prefix="tusimple_eval_")
        out_dir = self.output_dir or temp_dir
        try:
            os.makedirs(out_dir, exist_ok=True)
            pred_file = osp.join(out_dir, "tusimple_predictions.json")
            with open(pred_file, "w") as f:
                f.write("\n".join(json.dumps(r) for r in results))

            gt_file = self.gt_json
            if self.partial_eval:
                keep = {r["raw_file"] for r in results}
                gt_file = osp.join(out_dir, "partial_gt.json")
                with open(self.gt_json) as src, open(gt_file, "w") as dst:
                    dst.writelines(line for line in src
                                   if line.strip() and json.loads(line)["raw_file"] in keep)

            from .tusimple_eval import LaneEval  # lazy: needs scikit-learn
            result_json, accuracy = LaneEval.bench_one_submit(pred_file, gt_file)
            print_log(result_json, logger=logger)
            out = {item["name"]: float(item["value"]) for item in json.loads(result_json)}
            out["Accuracy"] = float(accuracy)
            return out
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# LLAMAS
# =============================================================================
@METRICS.register_module()
class LLAMASMetric(BaseMetric):
    def __init__(self, data_root, split="val", ori_img_w=1276, ori_img_h=717,
                 iou_thresholds=tuple(np.linspace(0.5, 0.95, 10)), width=30,
                 unofficial=False, output_dir=None, partial_eval=False,
                 collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.data_root = data_root
        self.split = split
        self.ori_img_w = float(ori_img_w)
        self.ori_img_h = float(ori_img_h)
        self.iou_thresholds = [float(t) for t in iou_thresholds]
        self.width = int(width)
        self.unofficial = bool(unofficial)
        self.output_dir = output_dir
        self.partial_eval = bool(partial_eval)

    def get_prediction_string(self, pred):
        # CLRNet LLAMAS.get_prediction_string
        ys = np.arange(300, 717, 1) / (self.ori_img_h - 1)
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * (self.ori_img_w - 1)
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * (self.ori_img_h - 1)
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = " ".join(["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != "":
                out.append(lane_str)
        return "\n".join(out)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        split_dir = "color_images/{}/".format("valid" if self.split == "val" else self.split)
        for result in data_samples:
            relative_path = result["metainfo"]["sub_img_name"]
            if split_dir not in relative_path:
                # test_dataloader/test_evaluator both default to val: overriding
                # only one of them would score (or submit) the wrong split.
                raise ValueError(f"LLAMASMetric(split={self.split!r}) got image "
                                 f"{relative_path!r}; set the dataset and evaluator "
                                 "split together")
            output_filename = "/".join(relative_path.split("/")[-2:]).replace(
                "_color_rect.png", ".lines.txt")
            self.results.append((output_filename, self.get_prediction_string(result["lanes"])))

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        out_dir = self.output_dir or tempfile.mkdtemp(prefix="llamas_eval_")
        # A temporary dir is deleted after scoring (each validation writes one
        # file per image, ~0.5 GB on LLAMAS valid). Test predictions are kept:
        # they are the submission. Set output_dir to keep val predictions.
        cleanup = self.output_dir is None and self.split != "test"
        for output_filename, output in results:
            path = osp.join(out_dir, output_filename)
            os.makedirs(osp.dirname(path), exist_ok=True)
            with open(path, "w") as out_file:
                out_file.write(output)
        if self.split == "test":
            print_log(f"LLAMAS test predictions written to {out_dir} (no public labels).", logger=logger)
            return {}

        anno_dir = osp.join(self.data_root, "labels/valid" if self.split == "val" else f"labels/{self.split}")
        if self.partial_eval:
            subset = tempfile.mkdtemp(prefix="llamas_partial_labels_")
            for output_filename, _ in results:
                src = osp.join(anno_dir, output_filename.replace(".lines.txt", ".json"))
                dst = osp.join(subset, output_filename.replace(".lines.txt", ".json"))
                os.makedirs(osp.dirname(dst), exist_ok=True)
                os.symlink(osp.abspath(src), dst)
            anno_dir = subset

        from ...datasets.metrics import llamas_eval  # lazy: multiprocessing + p_tqdm
        try:
            result = llamas_eval.eval_predictions(out_dir, anno_dir, width=self.width,
                                                  iou_thresholds=self.iou_thresholds,
                                                  unofficial=self.unofficial)
        finally:
            if cleanup:
                shutil.rmtree(out_dir, ignore_errors=True)
            if self.partial_eval:
                shutil.rmtree(anno_dir, ignore_errors=True)  # the symlink subset
        out = {}
        for thr, vals in result.items():
            key = "mean" if thr == "mean" else f"{float(thr):.2f}"
            for name in ("F1", "Precision", "Recall"):
                out[f"{name}_{key}"] = float(vals[name])
        print_log(f"LLAMAS F1@0.5 = {out.get('F1_0.50', float('nan')):.4f}", logger=logger)
        return out


# =============================================================================
# CurveLanes
# =============================================================================
@METRICS.register_module()
class CurvelanesMetric(BaseMetric):
    """Curvelane-branch evaluation.

    Expects the head decoded with ``test_cfg.ori_img_h=1, cut_height=0`` (lane y
    normalized within the crop). Each lane is mapped back to original-image
    normalized coordinates with the per-image ``crop_offset``, then evaluated
    exactly as the curvelane-branch ``CurvelanesDataset.evaluate``.
    """

    def __init__(self, eval_width=224, eval_height=224, iou_thresh=0.5, lane_width=5,
                 y_step=8, iou_thresholds=(0.5, 0.75), collect_device="cpu", prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.eval_width = eval_width
        self.eval_height = eval_height
        self.iou_thresh = iou_thresh
        self.lane_width = lane_width
        self.y_step = int(y_step)
        # Extra IoU thresholds reported as F1@50, F1@75, ... next to the
        # official f1_measure / precision / recall (which stay at iou_thresh).
        self.iou_thresholds = tuple(float(t) for t in (iou_thresholds or ()))

    @staticmethod
    def crop_to_original(lanes, ori_shape, crop_offset):
        ori_h = float(ori_shape[0])
        offset = float(crop_offset[1])
        out = []
        for lane in lanes:
            points = np.array(lane.points, dtype=np.float64, copy=True)
            points[:, 1] = (points[:, 1] * (ori_h - offset) + offset) / ori_h
            out.append(Lane(points=points, metadata=getattr(lane, "metadata", None)))
        return out

    def convert_coords_laneatt(self, lanes, ori_shape):
        # curvelane-branch CurvelanesDataset.convert_coords_laneatt
        ys = np.arange(0, ori_shape[0], self.y_step) / ori_shape[0]
        out = []
        for lane in lanes:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * ori_shape[1]
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * ori_shape[0]
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            out.append([{"x": x, "y": y} for x, y in zip(lane_xs, lane_ys)])
        return out

    @staticmethod
    def parse_anno(filename, formal=True):
        # curvelane-branch CurvelanesDataset.parse_anno
        anno_dir = filename.replace(".jpg", ".lines.txt")
        annos = []
        with open(anno_dir, "r") as anno_f:
            lines = anno_f.readlines()
        for line in lines:
            numbers = line.strip().split(" ")
            coords_tmp = [float(n) for n in numbers]
            coords = [(coords_tmp[2 * i], coords_tmp[2 * i + 1]) for i in range(len(coords_tmp) // 2)]
            annos.append(coords)
        if formal:
            annos = [[{"x": c[0], "y": c[1]} for c in lane] for lane in annos]
        return annos

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for result in data_samples:
            meta = result["metainfo"]
            ori_shape = tuple(meta["ori_shape"])
            lanes = self.crop_to_original(result["lanes"], ori_shape, meta["crop_offset"])
            self.results.append(dict(filename=meta["filename"], ori_shape=ori_shape,
                                     pred=self.convert_coords_laneatt(lanes, ori_shape)))

    def compute_metrics(self, results):
        """vega LaneMetricCore's F1 / precision / recall at ``iou_thresh`` (the
        official CurveLanes numbers), plus F1@50, F1@75, ... at
        ``iou_thresholds``. vega matches GT and predictions per image by the
        Hungarian algorithm on 1 - IoU, which does not depend on the threshold,
        so one IoU matrix per image gives every threshold (same cost as before).
        """
        try:
            from vega.metrics.pytorch.lane_metric import calc_iou, resize_lane  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "CurvelanesMetric needs vega's LaneMetricCore, as in the CLRerNet "
                "features/curvelane branch (pip install noah-vega).") from err
        thresholds = sorted(set(self.iou_thresholds) | {float(self.iou_thresh)})
        hits = {t: 0 for t in thresholds}
        gt_num = pr_num = 0
        for r in results:
            gt_wh = dict(height=r["ori_shape"][0], width=r["ori_shape"][1])
            gt_lines = [lane for lane in self.parse_anno(r["filename"]) if len(lane) > 0]
            pr_lines = [lane for lane in r["pred"] if len(lane) > 0]
            gt_num += len(gt_lines)
            pr_num += len(pr_lines)
            matched = self._matched_ious(gt_lines, pr_lines, gt_wh)
            for t in thresholds:
                hits[t] += int((matched > t).sum())   # vega: iou_val > iou_thresh
        precision, recall, f1 = self._prf(hits[float(self.iou_thresh)], pr_num, gt_num)
        out = dict(f1_measure=f1, precision=precision, recall=recall)
        for t in sorted(set(self.iou_thresholds)):
            out[f"F1@{int(round(t * 100))}"] = self._prf(hits[t], pr_num, gt_num)[2]
        print_log(str(out), logger=MMLogger.get_current_instance())
        return out

    @staticmethod
    def _prf(hits, pr_num, gt_num):
        """Same formula and epsilons as vega LaneMetricCore.summary."""
        eps = sys.float_info.epsilon
        precision = hits / (pr_num + eps)
        recall = hits / (gt_num + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return float(precision), float(recall), float(f1)

    def _matched_ious(self, gt_lanes, pr_lanes, gt_wh):
        """IoU of each Hungarian-matched (GT, prediction) pair of one image,
        computed exactly as vega's ``evaluate_core`` (GT and prediction share
        the image's shape here)."""
        from scipy.optimize import linear_sum_assignment
        from vega.metrics.pytorch.lane_metric import calc_iou, resize_lane
        if not gt_lanes or not pr_lanes:
            return np.zeros((0,))
        hyperp = dict(eval_width=self.eval_width, eval_height=self.eval_height,
                      iou_thresh=self.iou_thresh, lane_width=self.lane_width)
        y_ratio = np.true_divide(gt_wh["height"], self.eval_height)
        x_ratio = np.true_divide(gt_wh["width"], self.eval_width)
        gt_lanes = [resize_lane(lane, x_ratio, y_ratio) for lane in gt_lanes]
        pr_lanes = [resize_lane(lane, x_ratio, y_ratio) for lane in pr_lanes]
        iou_mat = np.zeros((len(gt_lanes), len(pr_lanes)))
        for i, gt_lane in enumerate(gt_lanes):
            for j, pr_lane in enumerate(pr_lanes):
                iou_mat[i, j] = calc_iou(gt_lane, pr_lane, hyperp)
        rows, cols = linear_sum_assignment(1 - iou_mat)
        return iou_mat[rows, cols]
