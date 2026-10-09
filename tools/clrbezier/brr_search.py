"""Search training-time hyper-parameters (brr loss weights etc.) on a validation split.

A loss weight such as ``brr_cp_loss_weight`` changes what the network learns, so it
cannot be tuned by re-scoring one checkpoint: every grid point has to be trained.
What the validation split decides is which of the trained models is best. This
tool makes that affordable and keeps it honest:

* every grid point trains on the same short **proxy schedule** (fewer epochs, the
  cosine compressed to end at the proxy horizon, optionally a fixed random part of
  the train list), with the same seed, so the variants differ only by the grid
  values;
* each proxy checkpoint is scored by ``sweep_conf_eval.py`` on the **val split**
  at its own best confidence threshold (focal-trained variants move the score
  scale, so a fixed threshold would rank them unfairly);
* the base config is always run as the reference, and ``--noise-probe`` adds the
  base config again with another seed, which tells you how large a difference
  the proxy can resolve;
* optional successive halving (``--rungs``): all variants train to the first rung,
  the best ``--keep`` fraction continues (the base and the noise probe always do),
  training resumes from the rung checkpoint, so nothing is trained twice;
* ``--holdout N`` scores on N images held out from the train list instead of the
  config's val split. Use it for CurveLanes, whose valid split is also the split
  you report on; choosing settings on it would be tuning on the test set.

Usage (from the repository root, where tools/clrbezier/sweep_conf_eval.py lives):

    # 1. write the proxy configs (nothing is trained)
    python tools/clrbezier/brr_search.py plan \\
        configs/clrbezier/culane/clrbezier_global_rows_o2o_r34_batch.py \\
        --grid loss_cfg.brr_cp_loss_weight=0,0.05,0.1,0.2,0.4 \\
        --epochs 12 --train-fraction 0.5 --noise-probe \\
        --out work_dirs/brr_search/culane_cp_weight

    # 2. train + score every variant (resumable: rerun the same command after a crash)
    python tools/clrbezier/brr_search.py run work_dirs/brr_search/culane_cp_weight --gpus 4

    # 3. ranked table (also written at the end of every run)
    python tools/clrbezier/brr_search.py report work_dirs/brr_search/culane_cp_weight

Keys: a short key such as ``loss_cfg.brr_cp_loss_weight`` or ``brr_cp_loss_weight``
is resolved against the base config (here to ``model.bbox_head.loss_cfg...``); it
must name exactly one place. Several ``--grid`` options form their full product.
Values are Python literals (``0.1``, ``True``, ``None``, ``'rows'`` or just ``rows``);
lists need quotes in the shell: ``--grid "brr_loss_stages=[2],[0,1,2]"``.
``--set KEY=VALUE`` changes every variant, the reference included.

Each variant gets ``<out>/configs/<name>.py`` (proxy) and
``<out>/configs/full/<name>.py`` (the same setting on the base schedule, for the
final run of the winner). Both only ``_base_`` the base config and list the
changed keys, so you can read exactly what differs.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import os.path as osp
import random
import re
import shutil
import subprocess
import sys
import time

PLAN_FILE = "search.json"
_SCHED_EPOCH_KEYS = ("begin", "end", "T_max")
_SHORT = {
    "brr_cp_loss_weight": "cpw",
    "brr_support_loss_weight": "supw",
    "brr_cp_smooth_l1_beta": "cpbeta",
    "brr_support_smooth_l1_beta": "supbeta",
    "brr_cp_loss_space": "cpspace",
    "brr_length_target_min": "lenmin",
    "brr_loss_stages": "stages",
    "start_y_component_weight": "startw",
    "length_component_weight": "lenw",
    "iou_loss_weight": "iouw",
    "cls_loss_weight": "clsw",
    "seg_loss_weight": "segw",
}


# ----------------------------------------------------------------------------
# small helpers (no mmengine needed: unit-testable)
# ----------------------------------------------------------------------------
def split_top_level(text, sep=","):
    """Split on ``sep`` outside brackets and quotes: "[0,1],2" -> ["[0,1]", "2"]."""
    parts, depth, quote, cur = [], 0, None, []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur).strip())
    if any(p == "" for p in parts):
        raise ValueError(f"empty value in {text!r}")
    return parts


def parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text  # bare word, e.g. rows


def parse_assignment(text, multi):
    if "=" not in text:
        raise ValueError(f"expected KEY=VALUE, got {text!r}")
    key, val = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"empty key in {text!r}")
    if multi:
        return key, [parse_value(v) for v in split_top_level(val)]
    return key, parse_value(val.strip())


def _walk(tree, prefix=()):
    """Yield (path, value) for every dict entry, descending into dicts only."""
    for k, v in tree.items():
        path = prefix + (str(k),)
        yield path, v
        if isinstance(v, dict):
            yield from _walk(v, path)


def get_path(tree, path, default=KeyError):
    node = tree
    for p in path:
        if not isinstance(node, dict) or p not in node:
            if default is KeyError:
                raise KeyError(".".join(path))
            return default
        node = node[p]
    return node


def resolve_key(tree, key):
    """Short key -> full dotted path into the config, or ValueError if ambiguous/absent.

    The key's parent must be a dict in the config; the leaf itself may be missing
    (a head default that the config does not write out).
    """
    parts = tuple(key.split("."))
    parent = parts[:-1]
    # exact path from the root
    if isinstance(get_path(tree, parent, None), dict) and (
            len(parent) > 0 or parts[-1] in tree):
        return ".".join(parts)
    if not parent:
        hits = [p for p, _ in _walk(tree) if p[-1] == parts[-1]]
    else:
        parents = [p for p, v in _walk(tree)
                   if isinstance(v, dict) and p[-len(parent):] == parent]
        with_leaf = [p for p in parents if parts[-1] in get_path(tree, p)]
        parents = with_leaf or parents
        hits = [p + (parts[-1],) for p in parents]
    if len(hits) == 1:
        return ".".join(hits[0])
    if not hits:
        raise ValueError(f"{key!r} does not match any dict in the base config; "
                         "give the full path, e.g. model.bbox_head.loss_cfg.<name>")
    raise ValueError(f"{key!r} is ambiguous: " + ", ".join(".".join(h) for h in hits)
                     + "; give the full path")


def nest(flat):
    """{"a.b.c": 1, "a.d": 2} -> {"a": {"b": {"c": 1}, "d": 2}}."""
    out = {}
    for key, val in flat.items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = node.setdefault(p, {})
            if not isinstance(nxt, dict):
                raise ValueError(f"{key!r} conflicts with another override")
            node = nxt
        if isinstance(node.get(parts[-1]), dict) and not isinstance(val, dict):
            raise ValueError(f"{key!r} conflicts with another override")
        node[parts[-1]] = val
    return out


def py_literal(value, indent=0):
    """Config-file text for a value: dict(...) for dicts with identifier keys."""
    pad = " " * (indent + 4)
    if isinstance(value, dict):
        if not value:
            return "dict()"
        if all(isinstance(k, str) and k.isidentifier() for k in value):
            items = [f"{pad}{k}={py_literal(v, indent + 4)}," for k, v in value.items()]
            return "dict(\n" + "\n".join(items) + "\n" + " " * indent + ")"
        items = [f"{pad}{k!r}: {py_literal(v, indent + 4)}," for k, v in value.items()]
        return "{\n" + "\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[\n" + "\n".join(f"{pad}{py_literal(v, indent + 4)}," for v in value) \
            + "\n" + " " * indent + "]"
    if isinstance(value, tuple):
        inner = ", ".join(py_literal(v, indent) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return f"float({str(value)!r})"
    return repr(value)


def config_text(base_path, overrides, header_lines):
    nested = nest(overrides)
    lines = [f"# {h}" if h else "#" for h in header_lines]
    lines += [f"_base_ = [{base_path!r}]", ""]
    for key, val in nested.items():
        lines.append(f"{key} = {py_literal(val)}")
    return "\n".join(lines) + "\n"


def value_tag(v):
    if isinstance(v, bool) or v is None:
        s = str(v)
    elif isinstance(v, float):
        s = f"{v:g}"
    else:
        s = str(v)
    s = re.sub(r"[^A-Za-z0-9.+-]+", "-", s).strip("-")
    return s or "x"


def variant_name(point, used):
    parts = []
    for key, val in point.items():
        leaf = key.split(".")[-1]
        parts.append(_SHORT.get(leaf, leaf) + value_tag(val))
    name = "_".join(parts) or "point"
    base, i = name, 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def expand_grid(grid_items, points):
    """Product of --grid axes, then explicit points (dicts)."""
    out = []
    if grid_items:
        keys = [k for k, _ in grid_items]
        for combo in itertools.product(*[vals for _, vals in grid_items]):
            out.append(dict(zip(keys, combo)))
    out.extend(points or [])
    return out


def scale_epoch_value(v, e0, e):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v > e0:
        return v  # mmengine's INF default and anything past the schedule stay
    if v == e0:
        return e
    scaled = v * e / e0
    return int(round(scaled)) if isinstance(v, int) else scaled


def proxy_schedulers(scheds, e0, e, warmup_iters=None):
    """Compress epoch-based schedulers from e0 to e epochs; iteration-based ones
    (warm-up, in absolute steps) are kept unless warmup_iters is given."""
    out, notes, decays = [], [], False
    for s in scheds:
        s = copy.deepcopy(dict(s))
        if s.get("by_epoch", True):
            for k in _SCHED_EPOCH_KEYS:
                if k in s:
                    s[k] = scale_epoch_value(s[k], e0, e)
            if "milestones" in s:
                s["milestones"] = [scale_epoch_value(m, e0, e) for m in s["milestones"]]
            decays = True
        else:
            if warmup_iters is not None and s.get("begin", 0) == 0 and "end" in s:
                notes.append(f"{s.get('type')}: end {s['end']} -> {warmup_iters} iterations")
                s["end"] = int(warmup_iters)
        out.append(s)
    if not decays:
        notes.append("no epoch-based scheduler found: the LR will not decay within the "
                     "proxy; express the decay with by_epoch=True, convert_to_iter_based=True")
    return out, notes


def setting_text(v):
    return ", ".join("%s=%r" % (k.split(".")[-1], x) for k, x in v["point"].items()) \
        or v.get("note", "")


def tree_digest(tree):
    return hashlib.sha1(json.dumps(tree, sort_keys=True, default=str).encode()).hexdigest()


def files_digest(paths):
    h = hashlib.sha1()
    for p in paths:
        if p:
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def fingerprint(overrides, base_digest, data_digest):
    """What a variant's training depends on: its overrides, the merged base config
    and the proxy data lists. Results and checkpoints are only reused under the
    same fingerprint."""
    blob = json.dumps(dict(o=overrides, b=base_digest, d=data_digest), sort_keys=True,
                      default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def read_list(path):
    with open(path) as f:
        return f.read().splitlines()


def make_data_subsets(train_list, diffs, diff_thr, holdout, fraction, seed, out_dir):
    """Write the proxy train list (and aligned diffs) and the holdout list.

    The dataset filters line i by diffs[i], so the subset keeps each line's own diff.
    Holdout lines keep only the image path (CULaneMetric reads the whole line as a
    path). Returns a summary dict.
    """
    lines = read_list(train_list)
    idx = [i for i, l in enumerate(lines) if l.strip()]
    if diffs is not None and len(diffs) < len(lines):
        raise ValueError(f"diff file has {len(diffs)} entries for {len(lines)} list lines")

    def passes(i):
        return diffs is None or diffs[i] >= diff_thr

    rng = random.Random(seed)
    held = []
    if holdout:
        eligible = [i for i in idx if passes(i)]
        if holdout >= len(eligible):
            raise ValueError(f"--holdout {holdout} >= {len(eligible)} usable train images")
        held = sorted(rng.sample(eligible, holdout))
    held_set = set(held)
    rest = [i for i in idx if i not in held_set]
    if fraction < 1.0:
        k = max(1, int(round(fraction * len(rest))))
        rest = sorted(rng.sample(rest, k))
    os.makedirs(out_dir, exist_ok=True)
    info = dict(train_lines=len(rest), train_used=sum(1 for i in rest if passes(i)),
                holdout=len(held), source_lines=len(idx))
    train_out = osp.join(out_dir, "train.txt")
    with open(train_out, "w") as f:
        f.write("\n".join(lines[i] for i in rest) + "\n")
    info["train_list"] = osp.abspath(train_out)
    if diffs is not None:
        import numpy as np
        diff_out = osp.join(out_dir, "train_diffs.npz")
        np.savez(diff_out, data=np.asarray(diffs)[rest])
        info["diff_file"] = osp.abspath(diff_out)
    if held:
        held_out = osp.join(out_dir, "holdout.txt")
        with open(held_out, "w") as f:
            f.write("\n".join(lines[i].split()[0] for i in held) + "\n")
        info["holdout_list"] = osp.abspath(held_out)
    return info


def survivors(variants, results_prev, keep):
    """Names that continue to the next rung.

    variants: list of dicts with name / pinned; results_prev: name -> result dict for
    every variant that ran the previous rung. Pinned variants continue if they ran
    successfully; the rest keep the best ceil(keep * n) by score.
    """
    ran = [v for v in variants if v["name"] in results_prev]
    pinned = [v["name"] for v in ran if v.get("pinned")
              and results_prev[v["name"]].get("status") == "ok"]
    others = [v["name"] for v in ran if not v.get("pinned")]
    ok = [n for n in others if results_prev[n].get("status") == "ok"]
    n_keep = max(1, int(math.ceil(keep * len(others)))) if others else 0
    ok.sort(key=lambda n: (-results_prev[n]["score"], n))
    keep_set = set(pinned) | set(ok[:n_keep])
    return [v["name"] for v in variants if v["name"] in keep_set]


# ----------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------
def _to_dict(cfg):
    return cfg.to_dict() if hasattr(cfg, "to_dict") else cfg._cfg_dict.to_dict()


def load_config_tree(path):
    from mmengine.config import Config
    return _to_dict(Config.fromfile(path, import_custom_modules=False))


def estimate_iters(n_images, batch_per_gpu, gpus):
    per_rank = math.ceil(n_images / max(1, gpus))
    return math.ceil(per_rank / max(1, batch_per_gpu))


def cmd_plan(args):
    base = osp.abspath(args.config)
    out = osp.abspath(args.out)
    if osp.exists(osp.join(out, PLAN_FILE)) and not args.force:
        sys.exit(f"{out} already has a plan; pass --force to replace it (a variant "
                 "whose training inputs are unchanged keeps its runs; `run` refuses "
                 "to reuse one whose inputs changed)")
    tree = load_config_tree(base)
    base_digest = tree_digest(tree)

    # search points
    grid_items = []
    for g in args.grid or []:
        key, vals = parse_assignment(g, multi=True)
        grid_items.append((resolve_key(tree, key), vals))
    points = []
    if args.grid_file:
        with open(args.grid_file) as f:
            spec = json.load(f)
        for key, vals in (spec.get("grid") or {}).items():
            grid_items.append((resolve_key(tree, key), list(vals)))
        for p in spec.get("points") or []:
            points.append({resolve_key(tree, k): v for k, v in p.items()})
    keys = [k for k, _ in grid_items]
    if len(set(keys)) != len(keys):
        sys.exit("the same key appears in two --grid options")
    search = expand_grid(grid_items, points)
    if not search:
        sys.exit("nothing to search: give --grid KEY=V1,V2,... or --grid-file")

    common = {}
    for s in args.set or []:
        key, val = parse_assignment(s, multi=False)
        common[resolve_key(tree, key)] = val
    clash = set(common) & {k for p in search for k in p}
    if clash:
        sys.exit(f"{sorted(clash)} are both searched and --set")

    # reference value of every searched key in the base config
    base_vals = {}
    for p in search:
        for k in p:
            base_vals[k] = get_path(tree, tuple(k.split(".")), "<unset>")
    unset = sorted(k for k, v in base_vals.items() if v == "<unset>")

    variants, used = [], set()
    have_base = False
    for p in search:
        is_base = all(base_vals[k] != "<unset>" and base_vals[k] == v for k, v in p.items())
        if is_base:
            if have_base:
                continue
            have_base = True
            used.add("base")
            variants.append(dict(name="base", point={}, pinned=True, note="= base config"))
        else:
            variants.append(dict(name=variant_name(p, used), point=p, pinned=False))
    if not have_base and not args.no_baseline:
        used.add("base")
        variants.append(dict(name="base", point={}, pinned=True, note="base config"))
    seed = (tree.get("randomness") or {}).get("seed")
    seed_note = None
    if seed is None:
        seed = 0
        common["randomness.seed"] = 0
        seed_note = "base config has no seed: every variant now uses seed 0"
    if args.noise_probe:
        variants.append(dict(name=f"base_seed{seed + 1}", point={}, pinned=True,
                             seed=seed + 1, note="base config, another seed (noise probe)"))
    # references first, then the grid in its own order
    variants.sort(key=lambda v: (v["name"] != "base", not v["pinned"]))

    # proxy schedule
    e0 = int(get_path(tree, ("train_cfg", "max_epochs")))
    epochs = args.epochs or max(1, e0 // 3)
    rungs = sorted(set(args.rungs or [])) or [epochs]
    if rungs[-1] != epochs:
        if args.rungs and args.epochs is None:
            epochs = rungs[-1]
        else:
            sys.exit(f"the last rung ({rungs[-1]}) must equal --epochs ({epochs})")
    if rungs[0] < 1:
        sys.exit("rungs must be >= 1 epoch")
    scheds, sched_notes = proxy_schedulers(tree.get("param_scheduler") or [], e0, epochs,
                                           args.warmup_iters)
    proxy = {
        "train_cfg.max_epochs": epochs,
        "train_cfg.val_begin": 10 ** 9,  # no validation inside training; scored afterwards
        "param_scheduler": scheds,
        "default_hooks.checkpoint.interval": 1,
        "default_hooks.checkpoint.save_begin": 0,
        "default_hooks.checkpoint.max_keep_ckpts": 2,
    }

    # data
    data_info = None
    tds = get_path(tree, ("train_dataloader", "dataset"))
    if args.train_fraction < 1.0 or args.holdout:
        if not 0.0 < args.train_fraction <= 1.0:
            sys.exit("--train-fraction must be in (0, 1]")
        if "data_list" not in tds:
            sys.exit("train_dataloader.dataset has no data_list; subsets are not supported "
                     "for this dataset type")
        diffs = None
        if tds.get("diff_file"):
            import numpy as np
            diffs = np.load(tds["diff_file"])["data"]
        data_info = make_data_subsets(tds["data_list"], diffs, tds.get("diff_thr", 0),
                                      args.holdout, args.train_fraction, args.data_seed,
                                      osp.join(out, "data"))
        proxy["train_dataloader.dataset.data_list"] = data_info["train_list"]
        if "diff_file" in data_info:
            proxy["train_dataloader.dataset.diff_file"] = data_info["diff_file"]
        if args.holdout:
            proxy["val_dataloader.dataset.data_root"] = tds["data_root"]
            proxy["val_dataloader.dataset.data_list"] = data_info["holdout_list"]
            ev = tree.get("val_evaluator")
            if isinstance(ev, (list, tuple)):
                sys.exit("--holdout supports a single val_evaluator")
            for k, v in (("data_root", tds["data_root"]), ("data_list", data_info["holdout_list"])):
                if ev and k in ev:
                    proxy[f"val_evaluator.{k}"] = v
            if "culane" in str((ev or {}).get("type", "")).lower():
                print("[warn] CULane holdout images have neighbouring video frames in the "
                      "train list; the official list/val.txt is the cleaner choice")
    data_digest = files_digest([data_info.get("train_list"), data_info.get("diff_file"),
                                data_info.get("holdout_list")]) if data_info else None
    n_train = data_info["train_used"] if data_info else None
    if n_train is None and "data_list" in tds and osp.exists(tds["data_list"]):
        lines = [l for l in read_list(tds["data_list"]) if l.strip()]
        n_train = len(lines)
        if tds.get("diff_file") and osp.exists(tds["diff_file"]):
            import numpy as np
            d = np.load(tds["diff_file"])["data"]
            n_train = int(sum(1 for i in range(len(lines)) if d[i] >= tds.get("diff_thr", 0)))
    bs = int(get_path(tree, ("train_dataloader", "batch_size"), 1))
    iters = estimate_iters(n_train, bs, args.gpus) if n_train else None

    # write configs
    cfg_dir = osp.join(out, "configs")
    os.makedirs(osp.join(cfg_dir, "full"), exist_ok=True)
    for v in variants:
        ov = dict(v["point"])
        ov.update(common)
        if v.get("seed") is not None:
            ov["randomness.seed"] = v["seed"]
        shown = ", ".join(f"{k} = {val!r}" for k, val in v["point"].items()) or v.get("note", "")
        proxy_ov = dict(ov)
        proxy_ov.update(proxy)
        hdr = ["Generated by tools/clrbezier/brr_search.py plan, " + time.strftime("%Y-%m-%d %H:%M"),
               f"variant {v['name']}: {shown}",
               f"proxy run: {epochs} of {e0} epochs"
               + (f", {data_info['train_lines']} train lines" if data_info else "")
               + (f", holdout {data_info['holdout']} images as val" if data_info and data_info["holdout"] else "")]
        v["config"] = osp.join(cfg_dir, v["name"] + ".py")
        with open(v["config"], "w") as f:
            f.write(config_text(base, proxy_ov, hdr))
        if v.get("seed") is None:
            full_ov = dict(v["point"])
            full_ov.update(common)
            v["full_config"] = osp.join(cfg_dir, "full", v["name"] + ".py")
            with open(v["full_config"], "w") as f:
                f.write(config_text(base, full_ov, [
                    "Generated by tools/clrbezier/brr_search.py plan",
                    f"variant {v['name']} on the base schedule: {shown}"]))
        v["overrides"] = proxy_ov
        v["fingerprint"] = fingerprint(proxy_ov, base_digest, data_digest)
    verify_configs(variants)

    plan = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), base_config=base, out=out,
        base_epochs=e0, epochs=epochs, rungs=rungs, keep=args.keep, gpus=args.gpus,
        searched_keys=keys or sorted({k for p in search for k in p}),
        base_values={k: (None if v == "<unset>" else v) for k, v in base_vals.items()},
        unset_keys=unset, common=common, data=data_info, seed=seed,
        iters_per_epoch=iters, scheduler_notes=sched_notes,
        eval=dict(range=list(args.conf_range), step=args.conf_step, metric_key=args.metric_key),
        base_digest=base_digest, data_digest=data_digest,
        variants=[{k: v[k] for k in ("name", "point", "pinned", "config", "seed", "note",
                                     "full_config", "fingerprint") if k in v}
                  for v in variants],
    )
    with open(osp.join(out, PLAN_FILE), "w") as f:
        json.dump(plan, f, indent=1, default=str)

    print(f"plan: {len(variants)} variants -> {out}")
    for v in variants:
        flag = " (always continues)" if v["pinned"] and len(rungs) > 1 else ""
        print(f"  {v['name']:<28} {setting_text(v)}{flag}")
    print(f"proxy: {epochs} of {e0} epochs, rungs {rungs}, keep {args.keep}")
    if data_info:
        print(f"data: {data_info['train_lines']} of {data_info['source_lines']} train lines "
              f"({data_info['train_used']} after the diff filter)"
              + (f", {data_info['holdout']} held out for scoring" if data_info['holdout'] else ""))
    if iters:
        total = iters * epochs
        print(f"~{iters} iterations/epoch at {bs} x {args.gpus} GPUs, {total} per proxy run")
        for s in scheds:
            if not s.get("by_epoch", True) and s.get("end", 0) > 0.25 * total:
                print(f"[warn] {s.get('type')} spans {s['end']} iterations = "
                      f"{100 * s['end'] / total:.0f}% of the proxy run; consider --warmup-iters")
    for n in sched_notes:
        print(f"[note] scheduler: {n}")
    if unset:
        print(f"[note] {unset} are not written in the base config (the head's default "
              "applies); the base run is that default")
    if seed_note:
        print(f"[note] {seed_note}")
    n_runs = len(variants)
    print(f"budget without halving: {n_runs} x {epochs} epochs"
          + (f" x {args.train_fraction:g} of the data" if args.train_fraction < 1 else "")
          + f" = {n_runs * epochs * args.train_fraction / e0:.2f} base-length runs")
    if len(rungs) > 1:
        n_pin = sum(1 for v in variants if v["pinned"])
        n_free, used_ep, prev = n_runs - n_pin, 0, 0
        for r in rungs:
            used_ep += (n_pin + n_free) * (r - prev)
            prev, n_free = r, max(1, int(math.ceil(args.keep * n_free))) if n_free else 0
        print(f"budget with halving (no failures): {used_ep * args.train_fraction / e0:.2f} "
              "base-length runs")
    print(f"next: python {osp.relpath(__file__)} run {osp.relpath(out)} --gpus {args.gpus}")


def verify_configs(variants):
    """Load every written config and check that each override took effect."""
    try:
        from mmengine.config import Config
    except ImportError:
        print("[warn] mmengine not importable; generated configs were not verified")
        return
    for v in variants:
        tree = _to_dict(Config.fromfile(v["config"], import_custom_modules=False))
        for key, val in v["overrides"].items():
            got = get_path(tree, tuple(key.split(".")), "<missing>")
            if _norm(got) != _norm(val):
                raise RuntimeError(f"{v['config']}: {key} = {got!r}, expected {val!r}")


def _norm(x):
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
def load_plan(out):
    path = osp.join(out, PLAN_FILE)
    if not osp.exists(path):
        sys.exit(f"no {PLAN_FILE} in {out}; run `plan` first")
    with open(path) as f:
        return json.load(f)


def result_path(out, name, epoch):
    # keyed by epoch, not rung index, so a re-plan with other rungs cannot pick up
    # a score from a different epoch
    return osp.join(out, name, f"result_e{epoch}.json")


def load_results(plan, rung):
    res = {}
    for v in plan["variants"]:
        p = result_path(plan["out"], v["name"], plan["rungs"][rung])
        if osp.exists(p):
            with open(p) as f:
                res[v["name"]] = json.load(f)
    return res


def latest_checkpoint(work_dir):
    """(path, epoch) of the newest epoch checkpoint, or None."""
    cands = []
    marker = osp.join(work_dir, "last_checkpoint")
    if osp.exists(marker):
        with open(marker) as f:
            p = f.read().strip()
        if p and not osp.isabs(p):
            p = osp.join(work_dir, osp.basename(p))
        if p and osp.exists(p):
            cands.append(p)
    if osp.isdir(work_dir):
        cands += [osp.join(work_dir, f) for f in os.listdir(work_dir)
                  if re.fullmatch(r"epoch_\d+\.pth", f)]
    best = None
    for p in cands:
        m = re.search(r"epoch_(\d+)\.pth$", p)
        if m and (best is None or int(m.group(1)) > best[1]):
            best = (p, int(m.group(1)))
    return best


def train_command(repo, cfg, work_dir, epoch, resume, gpus, python):
    tail = ["--work-dir", work_dir]
    if resume:
        tail += ["--resume", resume]
    tail += ["--cfg-options", f"train_cfg.max_epochs={epoch}"]
    if gpus > 1:
        return ["bash", osp.join(repo, "tools", "dist_train.sh"), cfg, str(gpus)] + tail
    return [python, osp.join(repo, "tools", "train.py"), cfg] + tail


def eval_command(plan, v, ckpt, stem, sweep_script, python, keep_dump):
    ev = plan["eval"]
    cmd = [python, sweep_script, v["config"], ckpt, "--split", "val",
           "--range", str(ev["range"][0]), str(ev["range"][1]), "--step", str(ev["step"]),
           "--work-dir", osp.join(plan["out"], v["name"], "eval"),
           "--out", stem + "_sweep.txt", "--overwrite"]
    if ev.get("metric_key"):
        cmd += ["--metric-key", ev["metric_key"]]
    if keep_dump:
        cmd += ["--save-dump", stem + "_dump.pkl"]
    return cmd


def _run_logged(cmd, log_path, env, cwd, dry_run):
    shown = [osp.relpath(c, cwd) if osp.isabs(c) and c.startswith(cwd + os.sep) else c
             for c in cmd]
    print("  $ " + " ".join(shown), flush=True)
    if dry_run:
        return 0
    with open(log_path, "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    return proc.returncode


def cmd_run(args):
    out = osp.abspath(args.dir)
    plan = load_plan(out)
    repo = osp.abspath(args.repo or osp.join(osp.dirname(osp.abspath(__file__)), "..", ".."))
    sweep_script = osp.abspath(args.sweep_script or osp.join(osp.dirname(osp.abspath(__file__)),
                                                             "sweep_conf_eval.py"))
    if not osp.exists(sweep_script):
        sys.exit(f"{sweep_script} not found (pass --sweep-script)")
    gpus = args.gpus or plan["gpus"]
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PORT", str(args.port))
    eval_env = dict(env)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible:
        eval_env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0]
    variants = {v["name"]: v for v in plan["variants"]}
    order = [v["name"] for v in plan["variants"]]
    now = tree_digest(load_config_tree(plan["base_config"]))
    if now != plan["base_digest"]:
        sys.exit(f"{plan['base_config']} (or a file it inherits) changed since the plan was "
                 "written; the variants would no longer be comparable. Plan again into a "
                 "new --out.")
    for name in order:
        fp_file = osp.join(out, name, "fingerprint")
        if osp.exists(fp_file):
            with open(fp_file) as f:
                old_fp = f.read().strip()
            if old_fp != variants[name]["fingerprint"]:
                sys.exit(f"{osp.join(out, name)} was trained under a different plan "
                         "(settings, schedule, base config or data lists changed). Delete "
                         "that directory or plan into a new --out.")
        elif not args.dry_run:
            os.makedirs(osp.join(out, name), exist_ok=True)
            with open(fp_file, "w") as f:
                f.write(variants[name]["fingerprint"] + "\n")

    alive = list(order)
    for k, epoch in enumerate(plan["rungs"]):
        if k > 0:
            alive = survivors([variants[n] for n in alive], load_results(plan, k - 1), plan["keep"])
        print(f"\n=== rung {k}: {epoch} epochs, {len(alive)} variants: {', '.join(alive)}")
        for name in alive:
            v = variants[name]
            wd = osp.join(out, name)
            os.makedirs(wd, exist_ok=True)
            rp = result_path(out, name, epoch)
            if osp.exists(rp):
                with open(rp) as f:
                    old = json.load(f)
                if old.get("status") == "ok" or not args.retry_failed:
                    print(f"[{name}] rung {k} done before ({old.get('status')})")
                    continue
            res = dict(variant=name, rung=k, epoch=epoch, status="ok")
            last = latest_checkpoint(wd)
            ckpt = osp.join(wd, f"epoch_{epoch}.pth")
            tic = time.time()
            if last and last[1] > epoch and not osp.exists(ckpt):
                res.update(status="failed", reason=f"work dir is past epoch {epoch} "
                           f"({last[0]}) but epoch_{epoch}.pth is gone")
            elif not (last and last[1] >= epoch):
                resume = last[0] if last else None
                print(f"[{name}] train to epoch {epoch}" + (f" (resume {osp.basename(resume)})" if resume else "")
                      + f"; log: {osp.relpath(osp.join(wd, 'train.log'))}", flush=True)
                rc = _run_logged(train_command(repo, v["config"], wd, epoch, resume, gpus, args.python),
                                 osp.join(wd, "train.log"), env, repo, args.dry_run)
                if rc != 0 or (not args.dry_run and not osp.exists(ckpt)):
                    res.update(status="failed", reason=f"training exit code {rc}",
                               log=osp.join(wd, "train.log"))
            res["train_seconds"] = round(time.time() - tic, 1)
            if res["status"] == "ok":
                stem = osp.join(wd, f"e{epoch}")
                print(f"[{name}] score epoch_{epoch}.pth on the val split")
                tic = time.time()
                rc = _run_logged(eval_command(plan, v, ckpt, stem, sweep_script, args.python,
                                              args.keep_dumps),
                                 osp.join(wd, "eval.log"), eval_env, repo, args.dry_run)
                res["eval_seconds"] = round(time.time() - tic, 1)
                if args.dry_run:
                    continue
                try:
                    with open(stem + "_sweep.json") as f:
                        sw = json.load(f)
                    key = sw["metric_key"]
                    res.update(metric_key=key, score=float(sw["best"][key]),
                               conf=float(sw["best"]["threshold"]), best=sw["best"],
                               checkpoint=ckpt)
                    print(f"[{name}] {key} {res['score']:.4f} at conf {res['conf']:.2f}")
                except (OSError, KeyError, ValueError) as err:
                    res.update(status="failed", reason=f"evaluation failed ({rc}): {err}",
                               log=osp.join(wd, "eval.log"))
            if args.dry_run:
                continue
            if res["status"] != "ok":
                print(f"[{name}] FAILED: {res.get('reason')} (see {res.get('log', wd)})")
            with open(rp, "w") as f:
                json.dump(res, f, indent=1)
        if args.dry_run:
            print("(dry run: later rungs depend on scores, stopping here)")
            return 0
        done = load_results(plan, k)
        if not any(done.get(n, {}).get("status") == "ok" for n in alive):
            sys.exit(f"every variant failed at rung {k}; see the logs under {out}")
    print()
    print(make_report(plan))
    return 0


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def make_report(plan, write=True):
    out = plan["out"]
    rungs = plan["rungs"]
    results = [load_results(plan, k) for k in range(len(rungs))]
    names = [v["name"] for v in plan["variants"]]
    by_name = {v["name"]: v for v in plan["variants"]}

    def last_ok(n):
        for k in range(len(rungs) - 1, -1, -1):
            r = results[k].get(n)
            if r and r.get("status") == "ok":
                return k, r
        return -1, None

    key = next((r["metric_key"] for res in results for r in res.values() if r.get("metric_key")), "F1")
    ranked = sorted(names, key=lambda n: (-last_ok(n)[0], -(last_ok(n)[1] or {}).get("score", -1e9), n))
    head = f"{'variant':<28}" + "".join(f" {'e' + str(e):>16}" for e in rungs) + "  setting"
    lines = ["=" * len(head),
             f"search {out}",
             f"base {plan['base_config']}",
             f"proxy {plan['epochs']} of {plan['base_epochs']} epochs; scored by {key} on "
             + ("holdout of train" if (plan.get("data") or {}).get("holdout") else "the val split")
             + " at each variant's best confidence threshold (value @ conf)",
             "-" * len(head), head]
    rows = []
    for n in ranked:
        cells = []
        for k in range(len(rungs)):
            r = results[k].get(n)
            if r is None:
                cells.append("-")
            elif r.get("status") != "ok":
                cells.append("failed")
            else:
                cells.append(f"{r['score']:.4f} @ {r['conf']:.2f}")
        v = by_name[n]
        setting = setting_text(v)
        lines.append(f"{n:<28}" + "".join(f" {c:>16}" for c in cells) + f"  {setting}")
        k_last, r_last = last_ok(n)
        rows.append(dict(variant=n, setting=setting, pinned=v.get("pinned", False),
                         last_rung_epoch=rungs[k_last] if k_last >= 0 else None,
                         score=(r_last or {}).get("score"), conf=(r_last or {}).get("conf"),
                         **{f"{key}_e{rungs[k]}": (results[k].get(n) or {}).get("score")
                            for k in range(len(rungs))},
                         full_config=v.get("full_config", "")))
    lines.append("-" * len(head))
    final = results[-1]
    probe = next((n for n in names if n.startswith("base_seed")), None)
    if "base" in final and final["base"].get("status") == "ok":
        b = final["base"]["score"]
        if probe and final.get(probe, {}).get("status") == "ok":
            noise = abs(final[probe]["score"] - b)
            lines.append(f"noise: base vs {probe} differ by {noise:.4f} (one pair, a rough guide; "
                         "treat gaps below ~2x this as ties)")
        else:
            noise = None
        contenders = [n for n in names if not by_name[n].get("pinned")
                      and final.get(n, {}).get("status") == "ok"]
        if contenders:
            best = max(contenders, key=lambda n: final[n]["score"])
            gap = final[best]["score"] - b
            verdict = ""
            if noise is not None:
                verdict = ("  -> larger than 2x the noise" if gap > 2 * noise
                           else "  -> within 2x the noise: not resolved by this proxy")
            lines.append(f"best searched: {best} ({gap:+.4f} vs base){verdict}")
            if by_name[best].get("full_config"):
                lines.append(f"full-schedule config: {by_name[best]['full_config']}")
    lines.append("=" * len(head))
    text = "\n".join(lines)
    if write:
        with open(osp.join(out, "report.txt"), "w") as f:
            f.write(text + "\n")
        with open(osp.join(out, "report.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["variant"])
            w.writeheader()
            w.writerows(rows)
    return text


def cmd_report(args):
    print(make_report(load_plan(osp.abspath(args.dir))))
    return 0


# ----------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write proxy configs and the search plan")
    p.add_argument("config", help="base config (the reference setting)")
    p.add_argument("--out", required=True, help="search directory")
    p.add_argument("--grid", action="append", metavar="KEY=V1,V2,...",
                   help="values to try for one key; several --grid form their product")
    p.add_argument("--grid-file", help='JSON: {"grid": {key: [values]}, "points": [{key: value}]}')
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override applied to every variant, the base included")
    p.add_argument("--epochs", type=int, default=None,
                   help="proxy schedule length (default: a third of the base schedule)")
    p.add_argument("--rungs", type=int, nargs="+", default=None,
                   help="successive halving: score at these epochs, keep the best --keep "
                        "fraction each time (last rung = --epochs)")
    p.add_argument("--keep", type=float, default=0.5)
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="train on this random fraction of the train list (same for all)")
    p.add_argument("--holdout", type=int, default=0,
                   help="hold N train images out and score on them instead of the val split")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--noise-probe", action="store_true",
                   help="also run the base config with seed + 1")
    p.add_argument("--no-baseline", action="store_true",
                   help="do not add the base config when no grid point equals it")
    p.add_argument("--warmup-iters", type=int, default=None,
                   help="replace the end of iteration-based warm-up schedulers")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--conf-range", type=float, nargs=2, default=(0.05, 0.95),
                   metavar=("LO", "HI"),
                   help="threshold sweep per checkpoint (wide by default: focal and "
                        "softmax heads peak in different places)")
    p.add_argument("--conf-step", type=float, default=0.05)
    p.add_argument("--metric-key", default=None,
                   help="rank by this key of the metric (default: its F1)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", help="train and score (resumable)")
    r.add_argument("dir")
    r.add_argument("--gpus", type=int, default=None, help="default: the plan's --gpus")
    r.add_argument("--port", type=int, default=29511, help="PORT for dist_train.sh")
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--repo", default=None, help="repository root (default: from this file)")
    r.add_argument("--sweep-script", default=None)
    r.add_argument("--keep-dumps", action="store_true",
                   help="save each inference pass (sweep_conf_eval --save-dump)")
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="print the first rung's commands")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("report", help="print and write the ranked table")
    s.add_argument("dir")
    s.set_defaults(func=cmd_report)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
