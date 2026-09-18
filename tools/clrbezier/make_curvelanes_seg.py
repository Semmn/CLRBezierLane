"""mmdet3 port of the curvelane-branch tools/make_seg.py.

Draws single-class lane masks (width 30, value 1) on the original CurveLanes
images and writes <data_root>/train/train_seg.txt ("<image> <mask>") used by
dataset_curvelanes_clrernet.py.

    python tools/clrbezier/make_curvelanes_seg.py configs/clrernet/curvelanes/clrernet_curvelanes_r34.py

Like the official tool, masks keep the image file name (.jpg). Pass --png to
store lossless masks instead (not the official behaviour).
"""
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS
from tqdm import tqdm

from libs.utils.lane_utils import interp
from libs.utils.visualizer import draw_lane


def make_seg_img(idx, dataset, rel_save_dir, width=30, png=False):
    save_dir = Path(dataset.img_prefix) / rel_save_dir
    img_info = dataset.img_infos[idx]
    imgname = Path(dataset.img_prefix) / img_info
    ori_shape = cv2.imread(str(imgname)).shape
    kps, _, _ = dataset.load_labels(idx)
    kps = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in kps]
    lanes = np.array([interp(kp, n=5) for kp in kps], dtype=object)
    img = np.zeros(ori_shape, dtype=np.uint8)
    for lane in lanes:
        img = draw_lane(lane, img=img, img_shape=ori_shape, width=width, color=(1, 1, 1))
    name = Path(imgname).with_suffix(".png").name if png else Path(imgname).name
    cv2.imwrite(str(save_dir / name), img)
    return img_info + " " + str(Path(rel_save_dir) / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--savedir", default="seg_mask")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))
    ds_cfg = cfg.train_dataloader.dataset.copy()
    ds_cfg["data_list"] = str(Path(ds_cfg["data_root"]) / "train.txt")
    ds_cfg["pipeline"] = []
    dataset = DATASETS.build(ds_cfg)
    (Path(dataset.img_prefix) / args.savedir).mkdir(parents=True, exist_ok=True)

    worker = partial(make_seg_img, dataset=dataset, rel_save_dir=args.savedir, png=args.png)
    with Pool(args.workers) as pool:
        lines = list(tqdm(pool.imap(worker, range(len(dataset)), chunksize=32), total=len(dataset)))
    with open(Path(dataset.img_prefix) / "train_seg.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
