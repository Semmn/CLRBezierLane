import math

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from libs.utils.lane_utils import sample_lane_rows


@TRANSFORMS.register_module()
class PackCLRNetInputs(BaseTransform):
    def __init__(
        self,
        #keys=None,
        meta_keys=None,
        max_lanes=4,
        num_points=72,
        img_w=800,
        img_h=320,
        lane_interp="spline",
    ):
        #self.keys = keys
        self.meta_keys = meta_keys
        self.max_lanes = max_lanes
        self.n_offsets = num_points
        self.n_strips = num_points - 1
        self.strip_size = img_h / self.n_strips
        self.offsets_ys = np.arange(img_h, -1, -self.strip_size)
        self.img_w = img_w
        # "spline" (official) | "linear" | "pchip"; see sample_lane_rows
        if lane_interp not in ("spline", "linear", "pchip"):
            raise ValueError(f"Unknown lane_interp {lane_interp!r}")
        self.lane_interp = lane_interp

    def convert_targets(self, results):
        old_lanes = results["gt_points"]
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 2, old_lanes)

        lanes = (
            np.ones((self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32)
            * -1e5
        )
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            try:
                # x at every row from the bottom up to the lane top, in row order.
                # The official hstack((xs_outside_image, xs_inside_image)) moved every
                # outside row to the front, which shifts a lane that leaves the image
                # sideways (above its first visible row) up by that many rows.
                all_xs = sample_lane_rows(lane, self.offsets_ys, self.img_w,
                                          interp=self.lane_interp)
            except AssertionError:
                continue
            inside_rows = np.nonzero((all_xs >= 0) & (all_xs < self.img_w))[0]
            if len(inside_rows) <= 1:  # to calculate theta
                continue
            first_row, last_row = int(inside_rows[0]), int(inside_rows[-1])
            xs_inside_image = all_xs[inside_rows]
            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = (
                    math.atan(
                        (inside_rows[i] - first_row)
                        * self.strip_size
                        / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)
                    )
                    / math.pi
                )
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            theta_far = sum(thetas) / len(thetas)

            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = 1 - first_row / self.n_strips  # y0, relative
            lanes[lane_idx, 3] = xs_inside_image[0]  # x0, absolute
            lanes[lane_idx, 4] = theta_far  # theta
            lanes[lane_idx, 5] = last_row - first_row + 1  # length
            lanes[lane_idx, 6 : 6 + len(all_xs)] = all_xs  # xs, absolute

        results["lanes"] = to_tensor(lanes)
        return results

    def transform(self, results):
        data = {}
        img_meta = {}
        data_sample = DetDataSample()
        instance_data = InstanceData()
        if "img" in results:
            img = results["img"]
            img = to_tensor(img).permute(2, 0, 1).contiguous()
        if "lanes" in self.meta_keys:  # training
            results = self.convert_targets(results)
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.gt_instances = instance_data
        data_sample.set_metainfo(img_meta)
        data["data_samples"] = data_sample
        data["inputs"] = img
        return data
