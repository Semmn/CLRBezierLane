"""Read CLR lane targets and segmentation masks from the official pipeline.

The official CLRerNet mmdet3 pipeline packs lane targets and masks into each
data sample. This module does not assume a single storage location: it
searches the usual places, verifies the tensor layout, and verifies the
start_y convention against the x-validity pattern of every GT lane.

Run ``tools/clrbezier/probe_official.py`` once to print where the official
pipeline actually stores these fields, then pin ``lane_keys`` / ``seg_keys``
in the config.
"""
import warnings

import torch
import torch.nn.functional as F

_LANE_KEY_CANDIDATES = ("lanes", "gt_lanes", "lane_line", "lane_targets")
_SEG_KEY_CANDIDATES = ("gt_masks", "seg", "gt_seg", "gt_sem_seg", "lane_mask")


def _to_tensor(value, device):
    if value is None:
        return None
    if hasattr(value, "sem_seg"):  # PixelData
        value = value.sem_seg
    if hasattr(value, "masks"):  # BitmapMasks-like
        value = value.masks
    if hasattr(value, "to_tensor"):
        try:
            value = value.to_tensor(dtype=torch.float32, device=device)
        except TypeError:
            value = value.to_tensor()
    return torch.as_tensor(value, device=device)


def _lookup(sample, keys):
    """Search metainfo, gt_instances, and attributes of one sample."""
    if isinstance(sample, dict):
        for key in keys:
            if key in sample:
                return sample[key], f"dict['{key}']"
        return None, None
    metainfo = getattr(sample, "metainfo", {}) or {}
    for key in keys:
        if key in metainfo:
            return metainfo[key], f"metainfo['{key}']"
    for key in keys:
        if key in ("gt_sem_seg",) and hasattr(sample, key):
            return getattr(sample, key), key
        inst = getattr(sample, "gt_instances", None)
        if inst is not None and key in inst:
            return inst.get(key), f"gt_instances.{key}"
        if hasattr(sample, key):
            return getattr(sample, key), key
    return None, None


def describe_sample(sample):
    """Human-readable field listing (used in error messages and the probe)."""
    if isinstance(sample, dict):
        return f"dict keys={sorted(sample.keys())}"
    parts = [f"type={type(sample).__name__}"]
    metainfo = getattr(sample, "metainfo", None)
    if metainfo is not None:
        parts.append(f"metainfo keys={sorted(metainfo.keys())}")
    for name in ("gt_instances", "gt_sem_seg", "gt_panoptic_seg"):
        if hasattr(sample, name):
            parts.append(f"{name}={getattr(sample, name)!r:.200}")
    if hasattr(sample, "keys"):
        try:
            parts.append(f"data keys={sorted(sample.keys())}")
        except Exception:  # noqa: BLE001
            pass
    return "; ".join(parts)


class CLRTargetAdapter:
    """Extract [B, L, 6+R] lane targets (official convention) and [B, H, W] masks."""

    def __init__(self, n_offsets=72, img_w=800, img_h=320, lane_keys=None, seg_keys=None,
                 start_y_convention="auto", length_unit="auto", check_interval=500,
                 min_agreement=0.9):
        self.n_offsets = int(n_offsets)
        self.n_strips = self.n_offsets - 1
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.lane_keys = tuple(lane_keys) if lane_keys else _LANE_KEY_CANDIDATES
        self.seg_keys = tuple(seg_keys) if seg_keys else _SEG_KEY_CANDIDATES
        if start_y_convention not in ("auto", "image_y", "bottom_offset"):
            raise ValueError("start_y_convention must be 'auto', 'image_y', or 'bottom_offset'.")
        if length_unit not in ("auto", "count", "normalized"):
            raise ValueError("length_unit must be 'auto', 'count', or 'normalized'.")
        self.start_y_convention = start_y_convention
        self.length_unit = length_unit
        self.check_interval = int(check_interval)
        self.min_agreement = float(min_agreement)
        self._calls = 0
        self._lane_source = None
        self._seg_source = None

    # ------------------------------------------------------------------ lanes
    def extract_lanes(self, batch_data_samples, device):
        lanes = []
        for sample in batch_data_samples:
            value, source = _lookup(sample, self.lane_keys)
            if value is None:
                raise KeyError(
                    "CLRBezierHead could not find lane targets in the data sample. "
                    f"Tried {self.lane_keys}. Sample fields: {describe_sample(sample)}. "
                    "Set bbox_head.target_adapter.lane_keys in the config."
                )
            self._lane_source = source
            t = _to_tensor(value, device).float()
            if t.ndim == 1:
                t = t.view(1, -1)
            if t.shape[-1] != 6 + self.n_offsets:
                raise ValueError(
                    f"Lane target width {t.shape[-1]} != 6 + n_offsets ({6 + self.n_offsets}) "
                    f"(source: {source})."
                )
            lanes.append(t)
        max_lanes = max(1, max(int(t.shape[0]) for t in lanes))
        out = torch.full((len(lanes), max_lanes, 6 + self.n_offsets), -1e5, device=device)
        out[..., 0] = 1.0
        out[..., 1] = 0.0
        for i, t in enumerate(lanes):
            out[i, : t.shape[0]] = t
        return self._canonicalize(out)

    def _first_valid_row(self, xs_px):
        valid = (xs_px >= 0.0) & (xs_px < float(self.img_w))
        has = valid.any(-1)
        first = torch.argmax(valid.int(), dim=-1)
        return first, has, valid.sum(-1)

    @torch.no_grad()
    def _canonicalize(self, lanes):
        """Return targets with start_y = image y and length = row count."""
        self._calls += 1
        fg = lanes[..., 1] == 1
        if not bool(fg.any()):
            return lanes
        sel = lanes[fg]
        first, has, count = self._first_valid_row(sel[:, 6:])
        s = sel[:, 2]
        n = float(self.n_strips)
        agree_image = ((torch.round((1.0 - s) * n) - first.float()).abs() <= 1.0) & has
        agree_bottom = ((torch.round(s * n) - first.float()).abs() <= 1.0) & has
        frac_image = float(agree_image.float().mean())
        frac_bottom = float(agree_bottom.float().mean())

        convention = self.start_y_convention
        if convention == "auto":
            if max(frac_image, frac_bottom) < self.min_agreement:
                raise RuntimeError(
                    "Could not infer the GT start_y convention: agreement image_y="
                    f"{frac_image:.3f}, bottom_offset={frac_bottom:.3f}. Check the row "
                    "order of the target xs (expected bottom -> top) and set "
                    "target_adapter.start_y_convention explicitly."
                )
            convention = "image_y" if frac_image >= frac_bottom else "bottom_offset"
            self.start_y_convention = convention  # lock after the first batch
            print(f"[CLRBezierHead] GT start_y convention: {convention} "
                  f"(agreement image_y={frac_image:.3f}, bottom_offset={frac_bottom:.3f}); "
                  f"lane source: {self._lane_source}")
        elif self.check_interval > 0 and self._calls % self.check_interval == 1:
            frac = frac_image if convention == "image_y" else frac_bottom
            if frac < self.min_agreement:
                warnings.warn(
                    f"GT start_y agreement with '{convention}' dropped to {frac:.3f}.")

        length = sel[:, 5]
        unit = self.length_unit
        if unit == "auto":
            unit = "count" if float(length.max()) > 1.5 else "normalized"
            self.length_unit = unit

        out = lanes.clone()
        if convention == "bottom_offset":
            out[..., 2] = torch.where(fg, 1.0 - lanes[..., 2], lanes[..., 2])
        if unit == "normalized":
            out[..., 5] = torch.where(fg, lanes[..., 5] * n, lanes[..., 5])
        return out

    # -------------------------------------------------------------------- seg
    def extract_seg(self, batch_data_samples, device, size_hw):
        masks = []
        for sample in batch_data_samples:
            value, source = _lookup(sample, self.seg_keys)
            if value is None:
                raise KeyError(
                    "CLRBezierHead could not find the segmentation target. "
                    f"Tried {self.seg_keys}. Sample fields: {describe_sample(sample)}. "
                    "Set bbox_head.target_adapter.seg_keys, or set seg_loss_weight=0."
                )
            self._seg_source = source
            m = _to_tensor(value, device)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            elif m.ndim == 3 and m.shape[-1] == 1:
                m = m[..., 0]
            if m.ndim != 2:
                raise ValueError(
                    f"Expected a [H, W] lane-id mask from {source}, got shape {tuple(m.shape)}.")
            masks.append(m.long())
        masks = torch.stack(masks, dim=0)
        if tuple(masks.shape[-2:]) != tuple(size_hw):
            masks = F.interpolate(masks[:, None].float(), size=size_hw, mode="nearest")[:, 0].long()
        return masks
