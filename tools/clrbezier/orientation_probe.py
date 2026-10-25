"""Do the encoder features already carry a lane's direction, and how precisely? Eval-only, CurveLanes valid.

road_info_ceiling.py found that a near-only lane's own far part, extended, would place 70% of those lanes, yet the
detector turns the whole lane about its far end (score_shift_check check 7: offsets 0.94 / 2.61 / 4.53 eval px from
the top third to the bottom, a rotation). The head infers a lane's direction from features sampled at positions
along its curve. An orientation field would give it the direction as a quantity of its own. Before building one:
is the direction already in the features the decoder reads?

Points: along every GT lane of the first --max-images test images, --points points in its far third and as many in
its near third (by arc length in the network input, 800 x 320; the far end is the upper end, or for a lane within
6 deg of level the end nearer the centre column), each with the GT direction there (undirected, from the lane 8 px of
arc before and after it).
Features: the encoder's tokens at each point, bilinear on every context map the decoder reads, concatenated.
Predictors of the direction, fit on the even images and scored on the odd ones, each predicting (cos 2a, sin 2a):
    position only   a 2-layer MLP on the point's position and the image's aspect: the perspective prior (lanes on the
                    left lean one way), the floor any feature probe has to beat
    ridge           a linear readout of the features (its weight picked on a held-out fifth of the even images)
    MLP             a 2-layer MLP on the features and the position
Reference: the head's own direction at the same points, read off its output lane nearest the GT lane (score above
--thr, median distance to the lane's points under 24 px; points with no such lane have no head error).
With --lanes-csv (score_shift_check's lanes.csv of the same checkpoint and threshold) the rows also split by that
run's outcome per lane: hit, near only, low score, missed.

Rows: median and 90th-percentile angle error (deg), far third and near third. Read "near only, far third": features
well under the head and under position only means the direction is in the features and the head does not read it
precisely, so a fix is in how queries read it. Features near position only means they do not carry it at that
precision, and an orientation field needs its own supervision.

--probe offset: can the features on the head's own curve see where the lane is? The direction probe reads features
on the GT lane; a near miss's curve sits about 2 stride-8 cells to the side of it at the bottom, and curve attention
samples close to its own curve. Near-third points only:
    displaced GT    each GT point moved across its lane (along the normal) by U(-40, 40) px, --synthetic per point;
                    the probes are fit on these (even images): features at a point -> the signed offset to its lane
    head's curve    the foot of each GT point on the head's matched lane, with the head lane's normal there: the
                    offset is the head's own error at that point, measured along its normal to the GT lane
(points nearer another GT lane than their own, or whose normal line misses their own lane, are dropped.)
Features: point = the tokens at the point on every context map (what the decoder's samples see now); strip = point
plus the finest context map at 4 points each side across the curve, 8 px apart (what sampling across the curve would
see, +-32 px). Ridge and MLP for each, both with the point's position, aspect and normal.
Read "head's curve off >= 6 px, near only": % brought within 6 px (the lateral shift that keeps IoU 0.5 at lane
width 5 in 224 x 224). Point probes bring most of them in: the decoder's own samples already carry the correction
and it is not acted on (a training fix). Only the strip does: the lane is out of the queries' reach (sampling across
the curve). Neither: features 2 cells off the lane do not locate it at this precision.
--features backbone (offset probe only): the same probe on the backbone's maps before the encoder: point = C2
(stride 4, hooked off the first stage), C3, C4, C5 at the point; strip = + C2 across the curve. Locates the lane
where the encoder's tokens do not: the encoder loses the evidence. Neither does: the image at this resolution does not
hold it. A second table gives, on the head's curve off >= 6 px, how far each probe would move it and how often toward
the lane (50% = no information).

    python tools/clrbezier/orientation_probe.py CONFIG CHECKPOINT [--lanes-csv .../lanes.csv] [--thr THR]
        [--max-images 4000] [--points 4] [--out-dir DIR] [--device cuda:0] [--probe direction|offset]
        [--synthetic 2] [--features encoder|backbone]

Heads: FovealLaneHead (without interval_cfg / hier_cfg) and BezierFormerHead, as score_shift_check.py (next to it).
The head's own output is re-run as in score_shift_check (self-check: max difference from its forward, must be 0).
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = osp.dirname(osp.abspath(__file__))
_spec = importlib.util.spec_from_file_location("score_shift_check", osp.join(_HERE, "score_shift_check.py"))
ssc = importlib.util.module_from_spec(_spec)
sys.modules["score_shift_check"] = ssc
_spec.loader.exec_module(ssc)
eb, rc = ssc.eb, ssc.rc

ARC_DELTA = 8.0                                   # px of arc each side of a point for its GT direction
MIN_LEN = 24.0                                    # px of arc: shorter lanes give no points
MATCH_PX = 24.0                                   # the head's lane must lie within this median distance (3 cells)
LEVEL = np.sin(np.radians(6.0))
PARTS = ("far third", "near third")
PREDICTORS = ("head", "position only", "ridge", "MLP")
STATUS = ("hit", "near only", "low score", "missed")
OFFSET_RANGE = 40.0                               # px: displaced GT points move up to this far across their lane
STRIP, STRIP_STEP = 4, 8.0                        # strip features: 4 points each side across the curve, 8 px apart
HIT_PX = 6.0                                      # px across: about the shift that keeps IoU 0.5 (width 5 in 224)
OFF_PREDICTORS = ("as is", "point, ridge", "point, MLP", "strip, ridge", "strip, MLP")


# ----------------------------------------------------------------------------
# geometry (network input px)
# ----------------------------------------------------------------------------
def undirected(d):
    """Angle in [0, pi) of direction vectors d [..., 2]."""
    return np.mod(np.arctan2(d[..., 1], d[..., 0]), np.pi)


def angle_err(a, b):
    """Degrees between undirected angles a, b (radians), in [0, 90]."""
    d = np.mod(np.abs(np.asarray(a) - np.asarray(b)), np.pi)
    return np.degrees(np.minimum(d, np.pi - d))


def _arc(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def _at(p, cum, s):
    return np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], -1)


def to_input(lane, ori_hw, offset, img_wh):
    """Original px [(x, y), ...] -> the vega spline (step 1) in network input px, cut to the crop, [n, 2] or None."""
    h, w = ori_hw
    dicts = [{"x": float(x), "y": float(y)} for x, y in lane]
    if len(dicts) < 2:
        return None
    sp = rc.spline_interp(lane=dicts, step_t=1)
    p = np.array([[q["x"], q["y"]] for q in sp], np.float64).reshape(-1, 2)
    p = np.stack([p[:, 0] * img_wh[0] / w, (p[:, 1] - offset) * img_wh[1] / (h - offset)], 1)
    p = p[(p[:, 1] >= 0) & (p[:, 1] <= img_wh[1]) & (p[:, 0] >= 0) & (p[:, 0] <= img_wh[0])]
    p = eb._distinct(p, 1e-3) if len(p) else p
    return p if len(p) >= 2 else None


def lane_points(gt, ori_hw, offset, img_wh, k):
    """Per GT lane: k points in its far third and k in its near third with the GT direction there.
    -> gt index [n], part [n] (0 far, 1 near), points [n, 2] (input px), angle [n] (radians, undirected)."""
    gi, part, pts, ang = [], [], [], []
    for g, ln in enumerate(gt):
        p = to_input(ln, ori_hw, offset, img_wh)
        if p is None:
            continue
        cum = _arc(p)
        if cum[-1] < MIN_LEN:
            continue
        if abs(p[0, 1] - p[-1, 1]) >= LEVEL * cum[-1]:
            flip = p[-1, 1] < p[0, 1]
        else:
            flip = abs(p[-1, 0] - img_wh[0] / 2) < abs(p[0, 0] - img_wh[0] / 2)
        if flip:
            p, cum = p[::-1].copy(), cum[-1] - cum[::-1]
        third = cum[-1] / 3.0
        for prt, s in ((0, (np.arange(k) + 0.5) / k * third), (1, cum[-1] - (np.arange(k) + 0.5) / k * third)):
            d = _at(p, cum, np.clip(s + ARC_DELTA, 0, cum[-1])) - _at(p, cum, np.clip(s - ARC_DELTA, 0, cum[-1]))
            gi += [g] * k
            part += [prt] * k
            pts.append(_at(p, cum, s))
            ang.append(undirected(d))
    if not pts:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2)), np.zeros(0)
    return np.array(gi), np.array(part), np.concatenate(pts), np.concatenate(ang)


def point_to_polyline(pts, poly, with_foot=False):
    """Distance [n] of points [n, 2] to a polyline [m, 2] and the undirected angle [n] of its nearest segment
    (with_foot: also the nearest point on it [n, 2])."""
    a, ab = poly[:-1], np.diff(poly, axis=0)
    ln2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(np.einsum("nmc,mc->nm", pts[:, None] - a[None], ab) / ln2[None], 0.0, 1.0)
    foot = a[None] + t[..., None] * ab[None]
    dist = np.linalg.norm(pts[:, None] - foot, axis=2)
    k = dist.argmin(1)
    i = np.arange(len(pts))
    if with_foot:
        return dist[i, k], undirected(ab[k]), foot[i, k]
    return dist[i, k], undirected(ab[k])


def head_match(gi, pts, lines):
    """Per point, the index of the head's output lane nearest the point's GT lane (median distance over that lane's
    points < MATCH_PX), -1 where there is none."""
    out = np.full(len(pts), -1)
    for g in np.unique(gi):
        m = gi == g
        best = None
        for li, poly in enumerate(lines):
            if len(poly) < 2:
                continue
            med = float(np.median(point_to_polyline(pts[m], poly)[0]))
            if med < MATCH_PX and (best is None or med < best[0]):
                best = (med, li)
        if best is not None:
            out[m] = best[1]
    return out


def head_angles(gi, pts, lines):
    """Per point, the head's direction from its matched output lane (head_match), NaN where there is none."""
    li = head_match(gi, pts, lines)
    out = np.full(len(pts), np.nan)
    for k in np.unique(li[li >= 0]):
        m = li == k
        out[m] = point_to_polyline(pts[m], lines[k])[1]
    return out


def tangent_normal(a):
    """Unit tangent and normal of undirected angles a: normal (sin a, -cos a), pointing right (x >= 0)."""
    t = np.stack([np.cos(a), np.sin(a)], -1)
    return t, np.stack([t[..., 1], -t[..., 0]], -1)


def normal_offset(q, a, poly):
    """Signed offset [n] (px) along the normal of a [n] from points q [n, 2] to where that normal line meets the
    polyline [m, 2] (the nearest crossing), NaN where it does not meet it."""
    t, nrm = tangent_normal(np.asarray(a))
    d = poly[None] - q[:, None]
    u, s = np.einsum("nmc,nc->nm", d, t), np.einsum("nmc,nc->nm", d, nrm)
    u0, u1 = u[:, :-1], u[:, 1:]
    cross = ((u0 <= 0) & (u1 >= 0)) | ((u0 >= 0) & (u1 <= 0))
    du = u0 - u1
    w = np.divide(u0, du, out=np.full_like(u0, 0.5), where=np.abs(du) > 1e-9)
    sc = np.where(cross, s[:, :-1] + w * (s[:, 1:] - s[:, :-1]), np.inf)
    k = np.abs(sc).argmin(1)
    out = sc[np.arange(len(q)), k]
    return np.where(np.isfinite(out), out, np.nan)


def offset_samples(polys, gi, part, pts, ang, lines, img_wh, rng, n_synth):
    """Near-third points with the signed offset to their GT lane along their normal: kind 0, the GT point moved along
    its normal by U(-OFFSET_RANGE, OFFSET_RANGE), n_synth per point; kind 1, its foot on the head's matched lane
    (head_match) with the head lane's direction there. Points outside the input, nearer another GT lane than their
    own, or whose normal line misses their own lane are dropped. -> gi, kind, q [n, 2], a [n], off [n]"""
    near = np.nonzero(part == 1)[0]
    li = head_match(gi, pts, lines)
    rows = []
    for i in near:
        _, nrm = tangent_normal(ang[i])
        for _ in range(n_synth):
            rows.append((gi[i], 0, pts[i] + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE) * nrm, ang[i]))
        if li[i] >= 0:
            _, a_h, foot = point_to_polyline(pts[i:i + 1], lines[li[i]], with_foot=True)
            rows.append((gi[i], 1, foot[0], a_h[0]))
    if not rows:
        return dict(gi=np.zeros(0, int), kind=np.zeros(0, int), q=np.zeros((0, 2)), a=np.zeros(0), off=np.zeros(0))
    G, kind = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    Q, A = np.array([r[2] for r in rows], np.float64), np.array([r[3] for r in rows])
    off = np.full(len(G), np.nan)
    for g in np.unique(G):
        m = G == g
        off[m] = normal_offset(Q[m], A[m], polys[g])
        for o, poly in enumerate(polys):
            if o != g and poly is not None:
                near_other = point_to_polyline(Q[m], poly)[0] < np.abs(off[m])
                off[np.nonzero(m)[0][near_other]] = np.nan
    keep = (~np.isnan(off) & (Q[:, 0] >= 0) & (Q[:, 0] <= img_wh[0]) & (Q[:, 1] >= 0) & (Q[:, 1] <= img_wh[1]))
    return dict(gi=G[keep], kind=kind[keep], q=Q[keep], a=A[keep], off=off[keep])


# ----------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------
def sample_tokens(tokens, maps, frame, pts, b, ctx):
    """[n, D x maps]: the encoded tokens of image b at input px pts [n, 2], bilinear on every context map."""
    if not len(pts):
        return torch.zeros(0, 0)
    loc = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=tokens.device)[None]
    grid = frame.grid(loc, ctx)[:, :, None, :]                                   # [1, n, 1, 2]
    dim = tokens.shape[-1]
    out = []
    for mp in maps:
        if mp.view != ctx:
            continue
        v = tokens[b:b + 1, mp.start:mp.start + mp.n].float().transpose(1, 2).reshape(1, dim, mp.h, mp.w)
        s = F.grid_sample(v, grid, mode="bilinear", padding_mode="border", align_corners=False)
        out.append(s[0, :, :, 0].t())
    return torch.cat(out, 1)


def sample_dense(fmaps, frame, pts, b, ctx):
    """[n, sum C]: dense maps [B, C, h, w] spanning the context frame (the backbone's), bilinear at input px pts [n, 2]
    of image b."""
    if not len(pts):
        return torch.zeros(0, 0)
    loc = torch.as_tensor(np.asarray(pts), dtype=torch.float32, device=fmaps[0].device)[None]
    grid = frame.grid(loc, ctx)[:, :, None, :]
    return torch.cat([F.grid_sample(f[b:b + 1].float(), grid, mode="bilinear", padding_mode="border",
                                    align_corners=False)[0, :, :, 0].t() for f in fmaps], 1)


def _strip(point_fn, fine_fn, q, a):
    """[n, point dims + 2 STRIP x fine dims]: point_fn at q, then fine_fn at STRIP points each side of q across the
    curve (along the normal of a), STRIP_STEP px apart."""
    if not len(q):
        return torch.zeros(0, 0)
    _, nrm = tangent_normal(np.asarray(a))
    ks = [k for k in range(-STRIP, STRIP + 1) if k]
    side = fine_fn(np.concatenate([q + k * STRIP_STEP * nrm for k in ks]))
    side = side.reshape(len(ks), len(q), -1).transpose(0, 1).reshape(len(q), -1)
    return torch.cat([point_fn(q), side], 1)


def sample_strip(tokens, maps, frame, q, a, b, ctx):
    """The encoder's strip: the tokens at q on every context map (sample_tokens), then the finest context map across
    the curve (_strip)."""
    fine = max((mp for mp in maps if mp.view == ctx), key=lambda mp: mp.h * mp.w)
    return _strip(lambda p: sample_tokens(tokens, maps, frame, p, b, ctx),
                  lambda p: sample_tokens(tokens, [fine], frame, p, b, ctx), q, a)


def backbone_strip(c2, levels, frame, q, a, b, ctx):
    """The backbone's strip: C2 (stride 4) and the levels C3-C5 at q, then C2 across the curve (_strip)."""
    return _strip(lambda p: sample_dense([c2] + list(levels), frame, p, b, ctx),
                  lambda p: sample_dense([c2], frame, p, b, ctx), q, a)


def first_stage(backbone):
    """The backbone's stride-4 stage: mmdet ResNet's layer1 (res_layers[0]); the CPU tests' stub ResNet's layers[0]."""
    if hasattr(backbone, "res_layers"):
        return getattr(backbone, backbone.res_layers[0])
    return backbone.layers[0]


def head_lines(rows, scores, thr, decoder, img_wh):
    """The head's output lanes scored >= thr as polylines in input px."""
    out = []
    for q in np.nonzero(scores >= thr)[0]:
        p = decoder.points(rows[q])
        if p is not None and len(p) >= 2:
            out.append(np.asarray(p, np.float64)[:, :2] * np.array(img_wh, np.float64))
    return out


def collect(cfg, args, device):
    from mmengine.runner import Runner
    from libs.clrbezier.foveal_lane import CTX
    model = eb.build_model(cfg, args.checkpoint, device)
    head = model.bbox_head
    rescorer = ssc.rescorer_for(head)
    meta = eb.head_meta(head)
    decoder = eb.make_decoder(meta)
    thr = float(args.thr if args.thr is not None else meta["test_cfg"]["conf_threshold"])
    img_wh = (meta["img_w"], meta["img_h"])
    loader = Runner.build_dataloader(copy.deepcopy(cfg.test_dataloader))
    rng = np.random.default_rng(args.seed)
    stage = {}
    if args.features == "backbone":
        first_stage(model.backbone).register_forward_hook(lambda mod, inp, out: stage.update(c2=out))
    recs, diff, n_img, tic = [], 0.0, 0, time.time()
    with torch.no_grad():
        for data in loader:
            data = model.data_preprocessor(data, False)
            feats = model.extract_feat(data["inputs"])
            st = rescorer.prefix(feats)
            pred, _ = rescorer.last(st, torch.zeros_like(st["cps"][:, :, 0]), "out")
            rows, scores = eb.pred_rows(pred)
            if n_img < 32:                               # self-check on the first batches
                own, _ = eb.pred_rows(head(feats)[-1])
                diff = max(diff, float((own.float().cpu() - rows.float().cpu()).abs().max()) if own.numel() else 0.0)
            rows, scores = rows.float().cpu().numpy(), scores.float().cpu().numpy()
            for b, ds in enumerate(data["data_samples"]):
                m = ds.metainfo
                ori_hw, offset = (int(m["ori_shape"][0]), int(m["ori_shape"][1])), int(m["crop_offset"][1])
                gt = [ln for ln in rc.read_lines_txt(eb.gt_file(str(m["filename"]))) if len(ln) > 0]
                gi, part, pts, ang = lane_points(gt, ori_hw, offset, img_wh, args.points)
                lines = head_lines(rows[b], scores[b], thr, decoder, img_wh)
                rec = dict(image=str(m["filename"]), img=n_img, aspect=ori_hw[1] / ori_hw[0])
                if args.probe == "offset":
                    polys = [to_input(ln, ori_hw, offset, img_wh) for ln in gt]
                    o = offset_samples(polys, gi, part, pts, ang, lines, img_wh, rng, args.synthetic)
                    if args.features == "backbone":
                        levels = feats["ctx"] if isinstance(feats, dict) else feats
                        feat = backbone_strip(stage["c2"], levels, st["frame"], o["q"], o["a"], b, CTX)
                        d_point = stage["c2"].shape[1] + sum(f.shape[1] for f in levels)
                    else:
                        feat = sample_strip(st["tokens"], st["maps"], st["frame"], o["q"], o["a"], b, CTX)
                        d_point = sum(mp.view == CTX for mp in st["maps"]) * st["tokens"].shape[-1]
                    rec.update(o, d_point=d_point)
                else:
                    feat = sample_tokens(st["tokens"], st["maps"], st["frame"], pts, b, CTX)
                    rec.update(gi=gi, part=part, pts=pts, ang=ang, head=head_angles(gi, pts, lines))
                recs.append(dict(rec, feat=feat.cpu().numpy().astype(np.float16)))
                n_img += 1
            if n_img % 500 < len(data["data_samples"]):
                print(f"  {n_img} images ({time.time() - tic:.0f} s)", flush=True)
            if args.max_images is not None and n_img >= args.max_images:
                break
    return recs[:args.max_images] if args.max_images else recs, meta, thr, diff


# ----------------------------------------------------------------------------
# probes
# ----------------------------------------------------------------------------
def doubled(a):
    return np.stack([np.cos(2 * a), np.sin(2 * a)], 1)


def from_doubled(y):
    return np.mod(0.5 * np.arctan2(y[:, 1], y[:, 0]), np.pi)


def _holdout(groups, seed):
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    hold = set(rng.choice(ug, max(1, len(ug) // 5), replace=False).tolist())
    return np.array([g in hold for g in groups])


def offset_code(s):
    return (np.asarray(s, np.float64) / OFFSET_RANGE)[:, None]


def from_offset_code(y):
    return y[:, 0] * OFFSET_RANGE


def offset_err(p, s):
    return np.abs(np.asarray(p) - np.asarray(s))


def fit_ridge(x, a, groups, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), seed=0, enc=doubled, dec=from_doubled,
              err=angle_err):
    """Linear readout of enc(a) ((cos 2a, sin 2a) by default); its weight picked on a held-out fifth of the images.
    -> predict(x) -> a, weight."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = ((x - mu) / sd).astype(np.float32)
    y = enc(a)
    va = _holdout(groups, seed)

    def stats(zz, yy):
        ym = yy.mean(0)
        return (zz.T @ zz).astype(np.float64), (zz.T @ (yy - ym)).astype(np.float64), ym, len(zz)

    def solve(st, lam):
        g, r, ym, n = st
        return np.linalg.solve(g + lam * n * np.eye(len(g)), r), ym
    st = stats(z[~va], y[~va])
    best = None
    for lam in lams:
        w, ym = solve(st, lam)
        e = float(np.median(err(dec(z[va] @ w + ym), a[va])))
        if best is None or e < best[0]:
            best = (e, lam)
    w, ym = solve(stats(z, y), best[1])
    return (lambda xn: dec(((xn - mu) / sd).astype(np.float32) @ w + ym)), best[1]


def fit_mlp(x, a, groups, seed=0, hidden=512, epochs=60, patience=6, device="cpu", enc=doubled, dec=from_doubled):
    """2-layer MLP on enc(a) ((cos 2a, sin 2a) by default), MSE; a held-out fifth of the images for early stopping."""
    torch.manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor(enc(a), dtype=torch.float32, device=device)
    va = _holdout(groups, seed)
    tr_i, va_i = np.nonzero(~va)[0], np.nonzero(va)[0]
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.SiLU(), torch.nn.Linear(hidden, yt.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(tr_i)
        for i in range(0, len(perm), 1024):
            b = torch.as_tensor(perm[i:i + 1024], device=device)
            opt.zero_grad()
            F.mse_loss(net(xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(F.mse_loss(net(xt[va_i]), yt[va_i]))
        if vl < best - 1e-5:
            best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()

    def predict(xn):
        with torch.no_grad():
            out = [net(torch.tensor((xn[i:i + 65536] - mu) / sd, dtype=torch.float32, device=device)).cpu().numpy()
                   for i in range(0, len(xn), 65536)]
            return dec(np.concatenate(out) if out else np.zeros((0, yt.shape[1]), np.float32))
    return predict


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def flatten(recs, status):
    cols = {k: [] for k in ("img", "gi", "part", "pts", "ang", "aspect", "head", "feat", "status", "image")}
    for r in recs:
        n = len(r["gi"])
        if not n:
            continue
        base = osp.basename(r["image"])
        cols["img"].append(np.full(n, r["img"]))
        cols["gi"].append(r["gi"])
        cols["part"].append(r["part"])
        cols["pts"].append(r["pts"])
        cols["ang"].append(r["ang"])
        cols["aspect"].append(np.full(n, r["aspect"]))
        cols["head"].append(r["head"])
        cols["feat"].append(r["feat"])
        cols["status"] += [status.get((base, int(g)), "") if status else "" for g in r["gi"]]
        cols["image"] += [r["image"]] * n
    out = {k: (np.concatenate(v) if k not in ("status", "image") else np.array(v, dtype=object))
           for k, v in cols.items() if len(v)}
    return out


def run_probes(P, img_wh, say, device="cpu", seed=0):
    """Fit on even images, predict odd ones -> {predictor: error deg [n_odd]}, odd mask."""
    odd = P["img"] % 2 == 1
    pos = np.stack([P["pts"][:, 0] / img_wh[0], P["pts"][:, 1] / img_wh[1], P["aspect"]], 1).astype(np.float32)
    feat = P["feat"].astype(np.float32)
    tr, te = ~odd, odd
    err = {"head": angle_err(P["head"][te], P["ang"][te])}
    f_pos = fit_mlp(pos[tr], P["ang"][tr], P["img"][tr], seed=seed, hidden=128, device=device)
    err["position only"] = angle_err(f_pos(pos[te]), P["ang"][te])
    f_ridge, lam = fit_ridge(feat[tr], P["ang"][tr], P["img"][tr], seed=seed)
    err["ridge"] = angle_err(f_ridge(feat[te]), P["ang"][te])
    both = np.concatenate([feat, pos], 1)
    f_mlp = fit_mlp(both[tr], P["ang"][tr], P["img"][tr], seed=seed, device=device)
    err["MLP"] = angle_err(f_mlp(both[te]), P["ang"][te])
    say(f"probes: {int(tr.sum())} train points ({len(np.unique(P['img'][tr]))} images), {int(te.sum())} test points; "
        f"{feat.shape[1]} feature dims; ridge weight {lam:g}")
    return err, te


def _q(v, q):
    v = v[~np.isnan(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def summarize(P, err, te, say):
    status = P["status"][te]
    part = P["part"][te]
    groups = {"all": np.ones(len(part), bool)}
    if any(s for s in status):
        groups.update({s: status == s for s in STATUS})
    out = {}
    say(f"{'':24}{'points':>8}" + "".join(f"{n:>18}" for n in PREDICTORS))
    for pi, pname in enumerate(PARTS):
        for gname, gm in groups.items():
            m = gm & (part == pi)
            row = {"n": int(m.sum())}
            cells = []
            for name in PREDICTORS:
                e = err[name][m]
                row[name] = dict(median=_q(e, 50), p90=_q(e, 90), n=int((~np.isnan(e)).sum()))
                cells.append("-" if not row[name]["n"] else f"{row[name]['median']:.2f} / {row[name]['p90']:.1f}")
            out[f"{pname}, {gname}"] = row
            say(f"{pname + ', ' + gname:>24}{row['n']:>8}" + "".join(f"{c:>18}" for c in cells))
    say("cell: median / 90th percentile angle error (deg) on the odd images; head: points whose GT lane has an output "
        "lane above the threshold within 24 px")
    return out


def flatten_offsets(recs, status):
    cols = {k: [] for k in ("img", "gi", "kind", "q", "a", "off", "aspect", "feat", "status", "image")}
    for r in recs:
        n = len(r["gi"])
        if not n:
            continue
        base = osp.basename(r["image"])
        for k in ("gi", "kind", "q", "a", "off", "feat"):
            cols[k].append(r[k])
        cols["img"].append(np.full(n, r["img"]))
        cols["aspect"].append(np.full(n, r["aspect"]))
        cols["status"] += [status.get((base, int(g)), "") if status else "" for g in r["gi"]]
        cols["image"] += [r["image"]] * n
    out = {k: (np.concatenate(v) if k not in ("status", "image") else np.array(v, dtype=object))
           for k, v in cols.items() if len(v)}
    out["d_point"] = next(r["d_point"] for r in recs if len(r["gi"]))
    return out


def run_offset_probes(P, img_wh, say, device="cpu", seed=0):
    """Fit on the displaced GT points of the even images, predict every odd point -> {predictor: signed offset}, odd
    mask."""
    odd = P["img"] % 2 == 1
    tr = (P["kind"] == 0) & ~odd
    _, nrm = tangent_normal(P["a"])
    side = np.stack([P["q"][:, 0] / img_wh[0], P["q"][:, 1] / img_wh[1], P["aspect"], nrm[:, 0], nrm[:, 1]],
                    1).astype(np.float32)
    feats = {"point": P["feat"][:, :P["d_point"]], "strip": P["feat"]}
    off, te = P["off"], odd
    pred = {"as is": np.zeros(int(te.sum()))}
    kw = dict(enc=offset_code, dec=from_offset_code)
    for name, f in feats.items():
        x = np.concatenate([f.astype(np.float32), side], 1)
        f_r, lam = fit_ridge(x[tr], off[tr], P["img"][tr], seed=seed, err=offset_err, **kw)
        pred[f"{name}, ridge"] = f_r(x[te])
        pred[f"{name}, MLP"] = fit_mlp(x[tr], off[tr], P["img"][tr], seed=seed, device=device, **kw)(x[te])
        say(f"{name}: {x.shape[1] - side.shape[1]} feature dims; ridge weight {lam:g}")
    say(f"probes: {int(tr.sum())} displaced GT points ({len(np.unique(P['img'][tr]))} images) to fit; "
        f"{int((te & (P['kind'] == 0)).sum())} displaced and {int((te & (P['kind'] == 1)).sum())} head points to test")
    return pred, te


def summarize_offset(P, pred, te, say):
    status, kind, off = P["status"][te], P["kind"][te], P["off"][te]
    statuses = STATUS if any(s for s in status) else ()
    rows = []
    for kname, k in (("displaced GT", 0), ("head's curve", 1)):
        rows.append((f"{kname}, all", kind == k))
        rows += [(f"{kname}, {s}", (kind == k) & (status == s)) for s in statuses]
    missed_by = (kind == 1) & (np.abs(off) >= HIT_PX)
    rows.append((f"head's curve off >= {HIT_PX:g} px, all", missed_by))
    rows += [(f"head's curve off >= {HIT_PX:g} px, {s}", missed_by & (status == s)) for s in statuses]
    out = {}
    say(f"{'':38}{'points':>8}" + "".join(f"{n:>15}" for n in OFF_PREDICTORS))
    for rname, m in rows:
        row = {"n": int(m.sum())}
        cells = []
        for name in OFF_PREDICTORS:
            e = offset_err(pred[name][m], off[m])
            row[name] = dict(median=_q(e, 50), within=float((e < HIT_PX).mean()) if len(e) else float("nan"))
            cells.append("-" if not len(e) else f"{row[name]['median']:.1f} / {100 * row[name]['within']:.0f}%")
        out[rname] = row
        say(f"{rname:>38}{row['n']:>8}" + "".join(f"{c:>15}" for c in cells))
    say(f"near third only; cell: median |offset error| (input px) / % within {HIT_PX:g} px, on the odd images. as is: "
        "the point's own offset from its GT lane (for the head's curve, the head's error there). Probes fit on displaced "
        "GT points of the even images; point = the features at the point, strip = + the finest map across the curve")
    say("")
    say(f"{'':38}{'points':>8}{'offset':>15}" + "".join(f"{n:>15}" for n in OFF_PREDICTORS[1:]))
    for rname, m in rows:
        if not rname.startswith("head's curve off"):
            continue
        row = {"n": int(m.sum()), "offset": _q(np.abs(off[m]), 50)}
        cells = []
        for name in OFF_PREDICTORS[1:]:
            p = pred[name][m]
            row[name] = dict(moved=_q(np.abs(p), 50),
                             toward=float((np.sign(p) == np.sign(off[m])).mean()) if len(p) else float("nan"))
            cells.append("-" if not len(p) else f"{row[name]['moved']:.1f} / {100 * row[name]['toward']:.0f}%")
        out[rname].update(move=row)
        say(f"{rname:>38}{row['n']:>8}" + (f"{row['offset']:>15.1f}" if row["n"] else f"{'-':>15}")
            + "".join(f"{c:>15}" for c in cells))
    say("cell: median |predicted offset| (px), how far the probe would move the head's curve / % moved toward the lane "
        "(50% = no information); offset: the head's own median |offset| there")
    return out


def read_status(path):
    status = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("status"):
                status[(osp.basename(r["image"]), int(r["gt"]))] = r["status"]
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--lanes-csv", default=None)
    ap.add_argument("--thr", type=float, default=None, help="default: the config's test_cfg.conf_threshold")
    ap.add_argument("--max-images", type=int, default=4000)
    ap.add_argument("--points", type=int, default=4, help="points per third of each lane")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", choices=("direction", "offset"), default="direction")
    ap.add_argument("--synthetic", type=int, default=2, help="offset probe: displaced copies of each GT point")
    ap.add_argument("--features", choices=("encoder", "backbone"), default="encoder",
                    help="offset probe: the encoder's tokens, or the backbone's maps before the encoder")
    args = ap.parse_args(argv)
    if args.features == "backbone" and args.probe != "offset":
        ap.error("--features backbone needs --probe offset")
    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    recs, meta, thr, diff = collect(cfg, args, device)
    status = read_status(args.lanes_csv) if args.lanes_csv else None
    P = flatten_offsets(recs, status) if args.probe == "offset" else flatten(recs, status)
    out_dir = args.out_dir or osp.join("work_dirs", "orientation_probe", osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say(f"config {args.config}")
    say(f"checkpoint {args.checkpoint}")
    say(f"head {meta['head_cls']}; images {len(recs)}; threshold {thr:.2f}; {args.points} points per third"
        + (f"; {args.features} features" if args.probe == "offset" else ""))
    say(f"check: the re-run output vs the head's own forward, max difference {diff:.3g} (must be 0)")
    if status is not None:
        matched = int(sum(s != "" for s in P["status"]))
        say(f"lanes.csv: outcome found for {matched} of {len(P['status'])} points")
    if args.probe == "offset":
        return offset_report(P, meta, thr, diff, recs, args, out_dir, say, lines, str(device))
    err, te = run_probes(P, (meta["img_w"], meta["img_h"]), say, device=str(device), seed=args.seed)
    say("")
    summary = summarize(P, err, te, say)
    summary.update(config=args.config, checkpoint=args.checkpoint, images=len(recs), threshold=thr,
                   self_check_max_diff=diff)
    with open(osp.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    idx = np.nonzero(te)[0]
    with open(osp.join(out_dir, "points.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "gt", "part", "x", "y", "angle_deg", "status"] + [f"err {n}" for n in PREDICTORS])
        for j, i in enumerate(idx):
            wr.writerow([P["image"][i], int(P["gi"][i]), PARTS[P["part"][i]], f"{P['pts'][i, 0]:.1f}",
                         f"{P['pts'][i, 1]:.1f}", f"{np.degrees(P['ang'][i]):.2f}", P["status"][i]]
                        + [f"{err[n][j]:.3f}" for n in PREDICTORS])
    print(f"wrote {out_dir}/summary.txt, summary.json, points.csv")
    return dict(summary=summary, P=P, err=err, te=te, diff=diff)


def offset_report(P, meta, thr, diff, recs, args, out_dir, say, lines, device):
    pred, te = run_offset_probes(P, (meta["img_w"], meta["img_h"]), say, device=device, seed=args.seed)
    say("")
    summary = summarize_offset(P, pred, te, say)
    summary.update(config=args.config, checkpoint=args.checkpoint, images=len(recs), threshold=thr,
                   self_check_max_diff=diff, features=args.features)
    tag = "offset" if args.features == "encoder" else "offset_backbone"
    with open(osp.join(out_dir, f"summary_{tag}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(osp.join(out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    idx = np.nonzero(te)[0]
    with open(osp.join(out_dir, f"{tag}s.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "gt", "kind", "x", "y", "offset", "status"] + [f"pred {n}" for n in OFF_PREDICTORS[1:]])
        for j, i in enumerate(idx):
            wr.writerow([P["image"][i], int(P["gi"][i]), ("displaced GT", "head's curve")[P["kind"][i]],
                         f"{P['q'][i, 0]:.1f}", f"{P['q'][i, 1]:.1f}", f"{P['off'][i]:.2f}", P["status"][i]]
                        + [f"{pred[n][j]:.2f}" for n in OFF_PREDICTORS[1:]])
    print(f"wrote {out_dir}/summary_{tag}.txt, summary_{tag}.json, {tag}s.csv")
    return dict(summary=summary, P=P, pred=pred, te=te, diff=diff)


if __name__ == "__main__":
    main()
