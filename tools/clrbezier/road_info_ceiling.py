"""Do the other lanes of an image say where a lane's near end is? GT only, CPU, no checkpoint.

The near misses of score_shift_check.py (checks 6-9) are whole lanes turned about their far end: the far end is right
and the near field is about one stride-8 cell off. A road-level fix (scene / road / lane tokens, lane or road width,
attention or an SSM across the lanes) can only place them if the rest of the road determines a lane's near end to
within the metric's tolerance. This check measures that on the GT, as a ceiling: the other lanes are the true ones and
the lane's own far part is exact.

Per GT lane L, in the metric's eval space (224 x 224): F = its far end (the upper end; for a lane within 6 deg of
level, the end nearer the centre column), its far line = the principal direction of its far third (by arc length), N =
its near end, at distance a from F along the far line and t across it. A predictor guesses t; the placed lane is L
turned about F so that its near end lies at the guess, and it is a hit if its metric IoU with L is > 0.5. L's own shape
and extent are kept, so only the angle can be wrong, as in the near misses.

    far line            t = 0: the lane's own far third, extended
    equal spacing       check 9's rule with GT neighbours: the far line's x at the image's bottom row snapped to the
                        lattice of the other lanes' near-field lines there (near fields within 75 deg of vertical and
                        >= 2 neighbours; the rest is "n/a"); the lane is sheared about its far end as in check 9
    learned, own        an MLP fit on the train split on the lane's own far part: F, the far line's direction, its bend
                        and the image's aspect. Not its extent: a lane that runs off the image ends on the border, so
                        F, the direction and a would give t away
    learned, + others   the same plus every other GT lane: 6 points along each in L's far-line frame, its lateral offset
                        at a, its length, for the 3 nearest on each side, and the lane count on each side

Rows: hit rate (IoU > 0.5), the median |error| at the near end (eval px) and the share within 1.7 eval px (about 6 px
of the 800-px input: real lanes lose IoU 0.5 near there, check 1). Columns: all lanes, interior and outermost lanes
(check 8's rule), and with --lanes-csv (score_shift_check's lanes.csv of a run) by that run's outcome: hit, near only
(the lanes check 6 found placed one cell off, no copy above the threshold), low score, missed.

Read: "learned, + others" well above "learned, own" on the near-only lanes, and placing most of them, means the rest of
the road carries their near-field position, so a road-level module has something to learn. Close to "learned, own":
the other lanes do not locate them, whatever architecture reads them. Both MLPs see exact GT, so the numbers are upper
bounds for a network that must first find the other lanes; a small MLP on ~20k images is not the best possible
learner, so a gap of a few points between the two is not proof either way.

On synthetic roads (tests/test_road_info_ceiling.py) with one near-field bend shared by the lanes of an image,
"learned, + others" places 99.7% against 54% for the far line and "learned, own"; with an independent bend per lane
it places 57% like the other two.

    python tools/clrbezier/road_info_ceiling.py CONFIG [--lanes-csv work_dirs/score_shift_check/<config>/lanes.csv]
        [--train-images 20000] [--val-images 4000] [--jobs 8] [--seed 0] [--out-dir DIR]

Train images: a seeded sample of the config's train list. Val images: those of --lanes-csv (so the outcomes line up),
else the first --val-images of the test list.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
import os.path as osp
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

_HERE = osp.dirname(osp.abspath(__file__))
_spec = importlib.util.spec_from_file_location("score_shift_check", osp.join(_HERE, "score_shift_check.py"))
ssc = importlib.util.module_from_spec(_spec)
sys.modules["score_shift_check"] = ssc
_spec.loader.exec_module(ssc)
eb, rc = ssc.eb, ssc.rc

HIT = 0.5
TOL = 1.7                                        # eval px at the near end, ~6 px of the 800-px input
EVAL = dict(eval_width=224, eval_height=224, lane_width=5)
MIN_LEN = 10.0                                   # eval px of arc: shorter lanes are skipped
LEVEL = np.sin(np.radians(6.0))                  # ends within this rise per unit length: pick the far end by column
N_SIDE, N_PTS = 3, 6
PREDICTORS = ("far line", "equal spacing", "learned, own", "learned, + others")
STATUS = ("hit", "near only", "low score", "missed")


# ----------------------------------------------------------------------------
# geometry of one image
# ----------------------------------------------------------------------------
def _direction(pts):
    """Unit principal direction of [n, 2] points, oriented from the first point to the last."""
    c = pts - pts.mean(0)
    d = np.linalg.svd(c, full_matrices=False)[2][0] if len(pts) >= 3 else pts[-1] - pts[0]
    d = d / (np.linalg.norm(d) + 1e-12)
    return d if np.dot(d, pts[-1] - pts[0]) >= 0 else -d


def _oriented(poly, cum, width):
    """The polyline from its far end: the upper end, or for a lane within 6 deg of level the end nearer the centre."""
    p0, p1 = poly[0], poly[-1]
    if abs(p0[1] - p1[1]) >= LEVEL * cum[-1]:
        flip = p1[1] < p0[1]
    else:
        flip = abs(p1[0] - width / 2) < abs(p0[0] - width / 2)
    if flip:
        poly = poly[::-1].copy()
        cum = cum[-1] - cum[::-1]
    return poly, cum


def _bend(far):
    """Signed angle (rad) from the first half of the far part to the second."""
    k = len(far) // 2
    if k < 2 or len(far) - k < 2:
        return 0.0
    d1, d2 = _direction(far[:k + 1]), _direction(far[k:])
    return float(np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], np.dot(d1, d2)))


def image_lanes(gt, ori_hw, ev):
    """Per GT lane (original px [(x, y), ...]) its geometry in eval space, or None if shorter than MIN_LEN:
    dicts (original px), poly (oriented from F), F, d (far-line direction), v (across, d turned +90 deg), a, t."""
    out = []
    for ln in gt:
        dicts = [{"x": float(x), "y": float(y)} for x, y in ln]
        gl = eb.gt_polyline(dicts, ori_hw, ev) if len(dicts) >= 2 else None
        if gl is None or gl[1][-1] < MIN_LEN:
            out.append(None)
            continue
        poly, cum = _oriented(*gl, ev["eval_width"])
        far = poly[cum <= cum[-1] / 3.0]
        if len(far) < 2:
            far = poly[:2]
        F, d = poly[0], _direction(far)
        v = np.array([-d[1], d[0]])
        a, t = float(np.dot(poly[-1] - F, d)), float(np.dot(poly[-1] - F, v))
        out.append(dict(dicts=dicts, poly=poly, cum=cum, F=F, d=d, v=v, a=a, t=t, bend=_bend(far)))
    return out


def own_features(g, ori_hw, ev):
    w, h = ev["eval_width"], ev["eval_height"]
    return np.array([g["F"][0] / w, g["F"][1] / h, g["d"][0], g["d"][1], g["bend"], ori_hw[1] / ori_hw[0]],
                    np.float32)


def other_features(g, others, ev):
    """The other lanes in g's far-line frame: N_SIDE slots on each side (nearest first by the lateral offset at g's
    near-end distance a), each = present, offset at a, length, N_PTS points (along, across); then the side counts."""
    w = float(ev["eval_width"])
    left, right = [], []
    for o in others:
        idx = np.linspace(0, len(o["poly"]) - 1, N_PTS).round().astype(int)
        rel = o["poly"] - g["F"]
        al, ac = rel @ g["d"], rel @ g["v"]
        order = np.argsort(al)
        at_a = float(np.interp(g["a"], al[order], ac[order]))
        feat = np.concatenate([[1.0, at_a / w, o["cum"][-1] / w], np.stack([al[idx], ac[idx]], 1).ravel() / w])
        (left if at_a < 0 else right).append((abs(at_a), feat))
    n_slot = 3 + 2 * N_PTS
    slots = []
    for side in (left, right):
        side.sort(key=lambda p: p[0])
        for k in range(N_SIDE):
            slots.append(side[k][1] if k < len(side) else np.zeros(n_slot))
    return np.concatenate(slots + [np.array([len(left), len(right)], np.float64) / 4.0]).astype(np.float32)


def spacing_guess(g, others, ori_hw, ev):
    """Check 9's rule from GT neighbours: (x_true, x_new, y_ref) at the image's bottom row in original px, or None."""
    h, w = ori_hw
    y_ref = h - 1.0
    own = ssc.near_line(g["dicts"])
    sx, sy = w / ev["eval_width"], h / ev["eval_height"]
    dx, dy = g["d"][0] * sx, g["d"][1] * sy
    if own is None or abs(dy) < 1e-6 or abs(dx / dy) > ssc.MAX_STEEP:
        return None
    x_far = g["F"][0] * sx + dx * (y_ref - g["F"][1] * sy) / dy
    xs = [ln[0] * y_ref + ln[1] for ln in (ssc.near_line(o["dicts"]) for o in others) if ln is not None]
    x_new = ssc.lattice_x(x_far, xs, w)
    if x_new is None:
        return None
    return own[0] * y_ref + own[1], float(x_new), y_ref


def turned(g, t_hat, ori_hw, ev):
    """L (original px dicts) turned about F so that its near end lies t_hat across the far line."""
    alpha = np.arctan2(t_hat, g["a"]) - np.arctan2(g["t"], g["a"])
    c, s = np.cos(alpha), np.sin(alpha)
    sx, sy = ori_hw[1] / ev["eval_width"], ori_hw[0] / ev["eval_height"]
    p = np.array([[q["x"] / sx, q["y"] / sy] for q in g["dicts"]]) - g["F"]
    p = p @ np.array([[c, s], [-s, c]]) + g["F"]
    return [{"x": float(x * sx), "y": float(y * sy)} for x, y in p]


def lane_iou(a, b, ori_hw, ev):
    return float(rc.iou_matrix([rc.lane_mask(a, ori_hw, ev)], [rc.lane_mask(b, ori_hw, ev)])[0, 0])


# ----------------------------------------------------------------------------
# workers
# ----------------------------------------------------------------------------
_W = {}


def _init(state):
    torch.set_num_threads(1)
    _W.update(state)


def _read(path):
    return [ln for ln in rc.read_lines_txt(eb.gt_file(path)) if len(ln) > 0]


def extract(item):
    """Per usable lane of one image: (gt index, own features, other features, t, a, outermost, spacing guess)."""
    path, ori_hw = item
    ev = _W["ev"]
    gt = _read(path)
    geo = image_lanes(gt, ori_hw, ev)
    outer = ssc.outermost_lanes(gt) if gt else []
    rows = []
    for k, g in enumerate(geo):
        if g is None or g["a"] < 0.3 * np.hypot(g["a"], g["t"]):
            continue
        others = [o for j, o in enumerate(geo) if j != k and o is not None]
        rows.append(dict(gt=k, own=own_features(g, ori_hw, ev), oth=other_features(g, others, ev), t=g["t"],
                         a=g["a"], outermost=bool(outer[k]), spacing=spacing_guess(g, others, ori_hw, ev)))
    return path, ori_hw, rows


def place(task):
    """IoU of each predictor's placement with its lane, for the lanes of one image."""
    path, ori_hw, preds = task
    ev = _W["ev"]
    geo = image_lanes(_read(path), ori_hw, ev)
    out = {}
    for k, p in preds.items():
        g = geo[k]
        res = {}
        for name, t_hat in p.items():
            if name == "equal spacing":
                if t_hat is None:
                    res[name] = None
                    continue
                x_true, x_new, y_ref = t_hat
                lane = ssc.shear_lane(g["dicts"], x_true, x_new, y_ref)
            else:
                lane = turned(g, t_hat, ori_hw, ev)
            res[name] = lane_iou(g["dicts"], lane, ori_hw, ev)
        out[k] = res
    return path, out


def _pool(fn, items, jobs, state):
    if jobs <= 1:
        _init(state)
        return [fn(x) for x in items]
    with mp.get_context("fork").Pool(jobs, initializer=_init, initargs=(state,)) as pool:
        return pool.map(fn, items, chunksize=16)


# ----------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------
def fit_mlp(x, y, groups, seed=0, epochs=80, patience=8, hidden=256, say=print):
    """MLP regressing t (eval px), Huber 1 px; 10% of the train images held out for early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    hold = set(rng.choice(ug, max(1, len(ug) // 10), replace=False).tolist())
    va = np.array([gi in hold for gi in groups])
    mu, sd = x[~va].mean(0), x[~va].std(0) + 1e-6
    xt = torch.tensor((x - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.SiLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    tr_idx, va_idx = np.nonzero(~va)[0], np.nonzero(va)[0]
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.tensor(rng.permutation(tr_idx))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(net(xt[b])[:, 0], yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(xt[va_idx])[:, 0], yt[va_idx])) if len(va_idx) else 0.0
        if vl < best - 1e-4:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    say(f"    {x.shape[1]} inputs, {len(tr_idx)} train / {len(va_idx)} held-out lanes, stopped at epoch {ep + 1}, "
        f"held-out Huber {best:.3f}")

    def predict(xn):
        with torch.no_grad():
            return net(torch.tensor((xn - mu) / sd, dtype=torch.float32))[:, 0].numpy().astype(np.float64)
    return predict


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def group_masks(rows):
    m = {"all": [True] * len(rows), "interior": [not r["outermost"] for r in rows],
         "outermost": [r["outermost"] for r in rows]}
    if any(r.get("status") for r in rows):
        for s in STATUS:
            m[s] = [r.get("status") == s for r in rows]
        m["near only, outermost"] = [r.get("status") == "near only" and r["outermost"] for r in rows]
        m["near only, interior"] = [r.get("status") == "near only" and not r["outermost"] for r in rows]
    return {k: np.array(v, bool) for k, v in m.items()}


def summarize(rows, say):
    groups = group_masks(rows)
    out = {name: {} for name in PREDICTORS}
    say(f"{'lanes':>28}" + "".join(f"{name:>27}" for name in PREDICTORS))
    for gname, m in groups.items():
        sub = [r for r, k in zip(rows, m) if k]
        cells = []
        for name in PREDICTORS:
            iou = [r["iou"][name] for r in sub if r["iou"].get(name) is not None]
            err = [r["err"][name] for r in sub if r["err"].get(name) is not None]
            r = dict(n=len(sub), applicable=len(iou), hit=float(np.mean(np.array(iou) > HIT)) if iou else float("nan"),
                     err_med=float(np.median(err)) if err else float("nan"),
                     within_tol=float(np.mean(np.array(err) <= TOL)) if err else float("nan"))
            out[name][gname] = r
            cells.append(_cell(r, name))
        say(f"{gname:>20} {len(sub):>7}" + "".join(f"{c:>27}" for c in cells))
    say("cell: hit % (IoU > 0.5) / median |error| eval px / % within 1.7 eval px. Equal spacing: its error is at the "
        "image's bottom row, [share of the lanes it applies to].")
    return out


def _cell(r, name):
    if not r["applicable"]:
        return "-"
    s = f"{100 * r['hit']:.1f} / {r['err_med']:.2f} / {100 * r['within_tol']:.0f}"
    return s + (f" [{100 * r['applicable'] / max(1, r['n']):.0f}%]" if name == "equal spacing" else "")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def list_images(root, data_list):
    with open(data_list) as f:
        rels = [ln.strip().split(" ")[0].lstrip("/") for ln in f if ln.strip()]
    return [osp.join(root, r) for r in rels]


def with_sizes(paths):
    from PIL import Image
    items, skipped = [], 0
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        if (h, w) not in rc.CROP_OFFSETS:              # CurvelanesDataset drops these
            skipped += 1
            continue
        items.append((p, (h, w)))
    return items, skipped


def read_status(path):
    """(image basename, gt index) -> outcome, and the images in order, from score_shift_check's lanes.csv."""
    status, images = {}, []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("status"):
                status[(osp.basename(r["image"]), int(r["gt"]))] = r["status"]
            if not images or images[-1] != r["image"]:
                images.append(r["image"])
    return status, list(dict.fromkeys(images))


def collect(items, jobs, ev):
    res = _pool(extract, items, jobs, dict(ev=ev))
    rows = []
    for gi, (path, ori_hw, lanes) in enumerate(res):
        for r in lanes:
            rows.append(dict(r, image=path, ori_hw=ori_hw, group=gi))
    return rows


def run(train_items, val_items, args, ev, say, status=None):
    tic = time.time()
    tr = collect(train_items, args.jobs, ev)
    va = collect(val_items, args.jobs, ev)
    say(f"train: {len(train_items)} images, {len(tr)} lanes; val: {len(val_items)} images, {len(va)} lanes "
        f"({time.time() - tic:.0f} s)")
    if status is not None:
        for r in va:
            r["status"] = status.get((osp.basename(r["image"]), r["gt"]))
    y = np.array([r["t"] for r in tr])
    groups = np.array([r["group"] for r in tr])
    x_own = np.stack([r["own"] for r in tr])
    x_all = np.concatenate([x_own, np.stack([r["oth"] for r in tr])], 1)
    say("  learned, own:")
    f_own = fit_mlp(x_own, y, groups, seed=args.seed, say=say)
    say("  learned, + others:")
    f_all = fit_mlp(x_all, y, groups, seed=args.seed, say=say)
    v_own = np.stack([r["own"] for r in va])
    p_own = f_own(v_own)
    p_all = f_all(np.concatenate([v_own, np.stack([r["oth"] for r in va])], 1))
    tasks = {}
    for i, r in enumerate(va):
        r["pred"] = {"far line": 0.0, "learned, own": float(p_own[i]), "learned, + others": float(p_all[i]),
                     "equal spacing": r["spacing"]}
        tasks.setdefault(r["image"], (r["ori_hw"], {}))[1][r["gt"]] = r["pred"]
    placed = dict(_pool(place, [(p, hw, pr) for p, (hw, pr) in tasks.items()], args.jobs, dict(ev=ev)))
    for r in va:
        r["iou"] = placed[r["image"]][r["gt"]]
        r["err"] = {k: abs(r["pred"][k] - r["t"]) for k in ("far line", "learned, own", "learned, + others")}
        sp = r["spacing"]
        r["err"]["equal spacing"] = (abs(sp[1] - sp[0]) * ev["eval_width"] / r["ori_hw"][1]) if sp else None
    say(f"placed ({time.time() - tic:.0f} s)\n")
    return va, summarize(va, say)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--lanes-csv", default=None)
    ap.add_argument("--train-images", type=int, default=20000)
    ap.add_argument("--val-images", type=int, default=4000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    ns = dict(data_root=None, data_list=None, lane_interp=None, curve_points=100)
    st_tr = rc.settings_from_config(cfg, SimpleNamespace(split="train", **ns))
    st_va = rc.settings_from_config(cfg, SimpleNamespace(split="valid", **ns))
    ev = st_va["eval"]
    out_dir = args.out_dir or osp.join("work_dirs", "road_info_ceiling", osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    train = list_images(st_tr["data_root"], st_tr["data_list"])
    train = random.Random(args.seed).sample(train, min(args.train_images, len(train)))
    status = None
    if args.lanes_csv:
        status, val = read_status(args.lanes_csv)
    else:
        val = list_images(st_va["data_root"], st_va["data_list"])[:args.val_images]
    train_items, sk_tr = with_sizes(train)
    val_items, sk_va = with_sizes(val)
    say(f"road_info_ceiling: {args.config}; eval {ev}; skipped sizes: train {sk_tr}, val {sk_va}")
    va, summary = run(train_items, val_items, args, ev, say, status)
    with open(osp.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    keys = ["image", "gt", "status", "outermost", "a", "t"]
    with open(osp.join(out_dir, "lanes.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys + [f"pred {k}" for k in PREDICTORS[2:]] + [f"iou {k}" for k in PREDICTORS])
        for r in va:
            wr.writerow([r.get(k) for k in keys] + [r["pred"][k] for k in PREDICTORS[2:]]
                        + [r["iou"].get(k) for k in PREDICTORS])
    print(f"wrote {out_dir}/summary.txt, summary.json, lanes.csv")
    return summary


if __name__ == "__main__":
    main()
