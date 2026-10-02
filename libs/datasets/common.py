"""Shared helpers for TuSimple / LLAMAS / CurveLanes on the official CLRerNet (mmdet 3.x) pipeline."""
import copy

from mmdet.registry import TRANSFORMS

from libs.datasets.pipelines.alaug import Alaug


def lanes_to_clrernet_points(lanes):
    """Convert [(x, y), ...] lanes to the official CLRerNet ``gt_points`` format.

    The official pipeline (``Alaug.is_sorted`` and ``sample_lane``) requires every
    lane as a flat list [x0, y0, x1, y1, ...] with strictly decreasing y
    (bottom -> top). The ordering and duplicate-y filtering follow CLRNet
    ``GenerateLaneLine.transform_annotation``: sort by -y, keep the first point for
    each y, and drop lanes with fewer than two points.

    Returns:
        (gt_points, id_classes, id_instances) as in ``CulaneDataset.load_labels``.
    """
    shapes = []
    for lane in lanes:
        if len(lane) <= 1:
            continue
        lane = sorted(lane, key=lambda p: -p[1])
        used, kept = set(), []
        for x, y in lane:
            if y in used:
                continue
            used.add(y)
            kept.append((float(x), float(y)))
        if len(kept) <= 1:
            continue
        shapes.append([c for p in kept for c in p])
    return shapes, [1] * len(shapes), [i + 1 for i in range(len(shapes))]


@TRANSFORMS.register_module()
class LaneAlbumentation(Alaug):
    """Official ``Alaug`` plus the curvelane-branch ``cut_unsorted`` option.

    The main-branch ``Compose`` builds ``type="albumentation"`` without extra
    arguments, so CurveLanes training uses this registered transform instead:
    ``dict(type="LaneAlbumentation", pipelines=[...], cut_unsorted=True)``.
    """

    def __init__(self, pipelines, cut_unsorted=False):
        super().__init__(pipelines)
        self.cut_unsorted = bool(cut_unsorted)

    @staticmethod
    def cut_unsorted_points(lanes):
        # Verbatim from the curvelane branch Alaug.
        out_points = []
        for lane in lanes:
            out_points.append([])
            prev_y = 1e8
            for x, y in zip(lane[0::2], lane[1::2]):
                if y < prev_y:
                    out_points[-1].extend([x, y])
                    prev_y = y
                else:
                    continue
        return out_points

    def __call__(self, data):
        # Verbatim control flow from the curvelane branch Alaug.__call__.
        data_org = copy.deepcopy(data)
        for _ in range(30):
            data_aug = self.aug(data)
            if self.is_sorted(data_aug["gt_points"]):
                return data_aug
            data = copy.deepcopy(data_org)
        if self.cut_unsorted:
            # avoid lane sampling errors for sharp curve lanes
            data_aug["gt_points"] = self.cut_unsorted_points(data_aug["gt_points"])
            return data_aug
        raise ValueError("lane augmentation failed 30 times. modifying GT..")
