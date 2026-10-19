"""Lane-aware foveation (clrbezierlane/lane_foveation_design.md), v1: one fixed far-field warp.

Every CurveLanes detector resizes the cropped image (up to 2560 x 800) uniformly to 800 x 320, so the far lanes
near the horizon keep about 1 px of width. Here the augmented image is kept at a larger intermediate size and
resampled into the same 800 x 320 input with a smooth separable warp that magnifies the far field (rows near the
top of the crop, columns near the centre) and compresses the near field. The detector is unchanged; lane points
map exactly through the warp, GT forward for training and predictions back for the metric.

Per axis, a normalized coordinate s in [0, 1] (0 = left / top edge of the crop, 1 = right / bottom) maps to
    F(s) = int_0^s d / int_0^1 d,   d(s) = 1 + amp * exp(-(s - centre)^2 / (2 sigma^2)),
monotone with F(0) = 0 and F(1) = 1; the magnification F'(s) = d(s) / Z is largest at the centre. Outside [0, 1]
F continues linearly with its end slopes. amp = 0 is the identity. Being separable and monotone, the warp keeps a
lane sorted bottom -> top sorted (the CLR row targets and cut_unsorted still apply).

    LaneFoveate               transform: after LaneAlbumentation / albumentation, before PackCLRNetInputs
    FoveatedCurvelanesMetric  CurvelanesMetric after mapping each predicted lane back through the inverse warp

Import with custom_imports "libs.datasets.foveation". Config: configs/*/curvelanes/dataset_curvelanes_*_fovea.py."""
import cv2
import numpy as np
from mmdet.registry import METRICS, TRANSFORMS

from libs.datasets.metrics import CurvelanesMetric
from libs.utils.lane_utils import Lane


class AxisWarp:
    """One axis of the warp: forward / inverse on normalized coordinates (arrays), and the magnification."""

    def __init__(self, centre=0.5, sigma=0.2, amp=0.0, grid=8193):
        if sigma <= 0 or amp < 0:
            raise ValueError(f"AxisWarp needs sigma > 0 and amp >= 0, got sigma={sigma}, amp={amp}")
        self.centre, self.sigma, self.amp = float(centre), float(sigma), float(amp)
        s = np.linspace(0.0, 1.0, int(grid))
        d = self.density(s)
        f = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(s))])
        self.z = float(f[-1])
        self.s, self.f = s, f / self.z
        self.f[-1] = 1.0
        self.slope0, self.slope1 = float(d[0] / self.z), float(d[-1] / self.z)

    def density(self, s):
        return 1.0 + self.amp * np.exp(-0.5 * ((np.asarray(s, dtype=np.float64) - self.centre) / self.sigma) ** 2)

    def magnification(self, s):
        """dF/ds at s: output length per input length (1 = as the uniform resize)."""
        return self.density(s) / self.z

    def forward(self, s):
        s = np.asarray(s, dtype=np.float64)
        u = np.interp(s, self.s, self.f)
        u = np.where(s < 0.0, s * self.slope0, u)
        return np.where(s > 1.0, 1.0 + (s - 1.0) * self.slope1, u)

    def inverse(self, u):
        u = np.asarray(u, dtype=np.float64)
        s = np.interp(u, self.f, self.s)
        s = np.where(u < 0.0, u / self.slope0, s)
        return np.where(u > 1.0, 1.0 + (u - 1.0) / self.slope1, s)


class LaneWarp:
    """The separable warp: x and y axis warps (dicts of AxisWarp arguments)."""

    def __init__(self, x=None, y=None):
        self.wx = AxisWarp(**(x or {}))
        self.wy = AxisWarp(**(y or {}))
        self._maps = {}

    def maps(self, src_w, src_h, out_w, out_h):
        """cv2.remap maps (float32 [out_h, out_w]): output pixel centre -> source pixel coordinates."""
        key = (src_w, src_h, out_w, out_h)
        if key not in self._maps:
            u = (np.arange(out_w) + 0.5) / out_w
            v = (np.arange(out_h) + 0.5) / out_h
            sx = (self.wx.inverse(u) * src_w - 0.5).astype(np.float32)
            sy = (self.wy.inverse(v) * src_h - 0.5).astype(np.float32)
            self._maps[key] = (np.ascontiguousarray(np.broadcast_to(sx[None, :], (out_h, out_w))),
                               np.ascontiguousarray(np.broadcast_to(sy[:, None], (out_h, out_w))))
        return self._maps[key]

    def points_forward(self, xs, ys, src_w, src_h, out_w, out_h):
        """Source px -> output px (same continuous convention as the image remap)."""
        return (self.wx.forward(np.asarray(xs, dtype=np.float64) / src_w) * out_w,
                self.wy.forward(np.asarray(ys, dtype=np.float64) / src_h) * out_h)

    def normalized_inverse(self, u, v):
        """Output-normalized -> source-normalized (the metric's lanes are normalized)."""
        return self.wx.inverse(u), self.wy.inverse(v)


@TRANSFORMS.register_module()
class LaneFoveate:
    """Resample results["img"] (any size) into out_w x out_h through the warp; gt_masks (nearest) and gt_points
    (flat [x0, y0, x1, y1, ...] lists, px) follow. x, y: AxisWarp arguments."""

    def __init__(self, out_w=800, out_h=320, x=None, y=None):
        self.out_w, self.out_h = int(out_w), int(out_h)
        self.warp = LaneWarp(x, y)

    def _remap(self, a, interp):
        h, w = a.shape[:2]
        mx, my = self.warp.maps(w, h, self.out_w, self.out_h)
        return cv2.remap(a, mx, my, interpolation=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def __call__(self, results):
        img = results["img"]
        src_h, src_w = img.shape[:2]
        results["img"] = self._remap(img, cv2.INTER_LINEAR)
        results["img_shape"] = results["img"].shape
        masks = results.get("gt_masks")
        if isinstance(masks, (list, tuple)):
            results["gt_masks"] = [self._remap(np.ascontiguousarray(m), cv2.INTER_NEAREST)
                                   if isinstance(m, np.ndarray) and m.ndim >= 2 and m.shape[:2] == (src_h, src_w)
                                   else m for m in masks]
        elif isinstance(masks, np.ndarray) and masks.ndim >= 2 and masks.shape[:2] == (src_h, src_w):
            results["gt_masks"] = self._remap(np.ascontiguousarray(masks), cv2.INTER_NEAREST)
        if results.get("gt_points"):
            out = []
            for lane in results["gt_points"]:
                u, v = self.warp.points_forward(lane[0::2], lane[1::2], src_w, src_h, self.out_w, self.out_h)
                out.append([float(c) for p in zip(u, v) for c in p])
            results["gt_points"] = out
        return results

    def __repr__(self):
        wx, wy = self.warp.wx, self.warp.wy
        return (f"LaneFoveate(out={self.out_w}x{self.out_h}, x=(c {wx.centre}, s {wx.sigma}, a {wx.amp}), "
                f"y=(c {wy.centre}, s {wy.sigma}, a {wy.amp}))")


@METRICS.register_module()
class FoveatedCurvelanesMetric(CurvelanesMetric):
    """CurvelanesMetric on lanes predicted in the warped input: each lane's points (normalized: x by the input
    width, y within the crop, as the heads decode with ori_img_w = ori_img_h = 1) go back through the inverse warp
    first. fovea: dict(x=..., y=...), the same as the pipeline's LaneFoveate."""

    def __init__(self, fovea=None, **kwargs):
        super().__init__(**kwargs)
        fovea = dict(fovea or {})
        self.warp = LaneWarp(fovea.get("x"), fovea.get("y"))

    def unwarp(self, lanes):
        out = []
        for lane in lanes:
            p = np.array(lane.points, dtype=np.float64, copy=True)
            p[:, 0], p[:, 1] = self.warp.normalized_inverse(p[:, 0], p[:, 1])
            out.append(Lane(points=p, metadata=getattr(lane, "metadata", None)))
        return out

    def process(self, data_batch, data_samples):
        super().process(data_batch, [dict(r, lanes=self.unwarp(r["lanes"])) for r in data_samples])
