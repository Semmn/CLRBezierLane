"""Record the images of every training batch whose loss or gradient spikes.

Add to a config:

    custom_hooks = [dict(type="LossSpikeHook", factor=2.0, window=200)]

After each iteration the hook compares the total loss and the pre-clip
gradient norm (logged by ``OptimWrapper`` when ``clip_grad`` is set) with the
median of the last ``window`` iterations. When either exceeds ``factor`` times
its median, the iteration's losses and the batch's file names go to
``<work_dir>/loss_spikes_rank<r>.jsonl`` (one file per GPU, since each GPU sees
its own batch). Feed those files to tools/clrbezier/scan_curvelanes_lanes.py.
"""
import json
import os.path as osp
from collections import deque

import numpy as np
from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LossSpikeHook(Hook):
    priority = "LOW"

    def __init__(self, factor=2.0, window=200, warmup=50, grad_factor=None,
                 meta_keys=("sub_img_name", "filename")):
        self.factor = float(factor)
        self.grad_factor = float(grad_factor if grad_factor is not None else factor)
        self.window = int(window)
        self.warmup = int(warmup)
        self.meta_keys = tuple(meta_keys)
        self.loss_hist = deque(maxlen=self.window)
        self.grad_hist = deque(maxlen=self.window)
        self.path = None

    def before_train(self, runner):
        self.path = osp.join(runner.work_dir, f"loss_spikes_rank{get_rank()}.jsonl")

    @staticmethod
    def _value(v):
        try:
            return float(v.detach().float().mean()) if hasattr(v, "detach") else float(v)
        except (TypeError, ValueError):
            return None

    def _names(self, data_batch):
        names = []
        for sample in (data_batch or {}).get("data_samples", []):
            meta = getattr(sample, "metainfo", {}) or {}
            name = next((meta[k] for k in self.meta_keys if k in meta), None)
            n_lanes = None
            lanes = meta.get("lanes")
            if lanes is not None and hasattr(lanes, "shape") and lanes.ndim == 2:
                n_lanes = int((lanes[:, 1] == 1).sum())
            names.append(dict(name=str(name), lanes=n_lanes))
        return names

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        losses = {k: self._value(v) for k, v in (outputs or {}).items()}
        losses = {k: v for k, v in losses.items() if v is not None}
        loss = losses.get("loss")
        grad = None
        try:
            grad = float(runner.message_hub.get_scalar("train/grad_norm").current())
        except Exception:  # no clip_grad: grad_norm is not logged
            pass
        hit = []
        if loss is not None and len(self.loss_hist) >= self.warmup:
            med = float(np.median(self.loss_hist))
            if loss > self.factor * med:
                hit.append(f"loss {loss:.3f} > {self.factor} x median {med:.3f}")
        if grad is not None and len(self.grad_hist) >= self.warmup:
            med = float(np.median(self.grad_hist))
            if grad > self.grad_factor * med:
                hit.append(f"grad_norm {grad:.2f} > {self.grad_factor} x median {med:.2f}")
        if hit:
            rec = dict(epoch=runner.epoch + 1, iter_in_epoch=batch_idx + 1, iter=runner.iter + 1,
                       reason=hit, grad_norm=grad, losses=losses, images=self._names(data_batch))
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if loss is not None:
            self.loss_hist.append(loss)
        if grad is not None:
            self.grad_hist.append(grad)
