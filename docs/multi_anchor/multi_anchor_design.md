# Multi-anchor lane head: design

2026-11-26. Thread "Explore a new lane detection design". YSY's instruction: think deeply about the core components,
keep everything else on standard choices, then train and evaluate. The base is the pointcond run
(foveal_lane_off_bfstyle_o2o_pointcond_curvelanes_r34: 86.58 at ep24).

## 1. The problem it has to fix

The measured failure is the near miss. The lane has the right far end and a wrong near-field angle: a rigid
rotation about the far end, |d| = 4.5 / 2.6 / 0.9 eval px at the bottom, middle and top. These lanes are mostly the
outermost lane, an unpainted edge, or a lane under a vehicle. The far part is visible, but the near part has no
local evidence. The position oracle is worth +5.35 F1.

Three Bezier models share this error (near-third r 0.80, same side 88%). Swapping models inside the Bezier family
does not help. No other anchor family has been checked.

Two earlier results limit what can work:

- A local near-end readout from the current features places only about 14% of the near-only lanes. The evidence
  is not there.
- A neighbour-spacing prior broke 72% of the correct lanes. Lanes are not evenly spaced at the bottom row.

So the near field of these lanes has to come from somewhere other than the local image or the neighbours.

## 2. Core idea: one query, three anchored forms of the same lane

Each query carries three parametric forms of its lane, one per anchor family. Each form is pinned at a different
place and reads its anchor parameters at a different place.

| Form | Family | Pinned at | Parameters (px, rad) | Where the near field comes from |
|---|---|---|---|---|
| S | line anchor (CLRNet-style start + angle) | near end | a (near point), phi (absolute heading at a), L, c2, c3 | the near evidence, extrapolated forwards |
| B | endpoint Bezier (the current head) | both ends | P0..P3 (monotone step, unchanged) | one pooled query vector moves all four points |
| V | polar anchor (a ray from the vanishing point) | far end | f (far point), delta (heading at f relative to the VP ray through f), L, c2, c3 | the VP ray through the visible far part |

S and V are cubic graphs in their anchor's tangent frame. With the anchor at A, the unit heading d and its normal n,
and s from 0 to 1 away from the anchor:

`p(s) = A + L s d + L (c2 s^2 + c3 s^3) n`

The tangent at s = 0 is exactly d, so the heading is the anchor's own angle. S uses d = (cos phi, sin phi). V uses
d = rotate(unit(f - v), delta), where v is the predicted vanishing point.

### Why V is the one with a mechanism for the near miss

For V, delta = 0 and c = 0 is a straight line through the VP. On flat roads that is almost always right: flat
lanes lie 0.9 deg from the VP ray (polar-view gate), against 79.4 deg from vertical. The near-miss error is about
2 deg.

So V's regression target for a straight lane is about zero, whatever the lane's slant in the image. When the near
part has no evidence, the target that is easiest to learn is "continue along the ray through the far part". The far
part is right on these lanes, and the VP is a scene-level estimate made from all lanes.

S and B regress an absolute angle. Without evidence, they fall back to a typical angle, which is the error we
measure.

This is different from the earlier VP work (polar view, polar features, ray aggregation: no gain). Those changed
the features. Here the VP changes the parameter space of the lane. The output is still an ordinary image-space
curve.

### Why keep S and B

- B is the current model. Keeping it as a member means the fused model can never lose B's own strengths, and it
  is the control inside the model.
- S is the classic line anchor. Its errors grow from the bottom up, the opposite of V. It helps where the near
  evidence is strong and the far end is unclear (curves, crests). That matches YSY's line / Bezier / polar framing.

## 3. Components

### 3.1 Initialization from one proposal (no new proposal machinery)

The proposal cubic (unchanged) starts all three forms, each as its family's conventional anchor:

- B = the proposal cubic, as now.
- S = the line from the proposal's near end along its near tangent: phi = angle(P1 - P0), L = |P3 - P0|, c = 0.
  This is a CLRNet line anchor.
- V = the ray through the VP and the proposal's far end: f = P3, delta = 0, c = 0, L = |P3 - P0|. This is a
  polar anchor.

All are detached from the proposal, as cps0 is now.

### 3.2 Readout: where each form gets its update

The decoder steps are unchanged: self-attention, point-conditioned curve attention, FFN on the shared query h. The
reference points come from the fused curve.

- B: `update(h)`, the existing linear layer and monotone step. Bit for bit the current head.
- S and V: an MLP on [h, local], where local is the stride-8 encoder map read at three points of the form's
  own curve next to its anchor (s = 0, 0.1, 0.2), then averaged. S reads its near end and V reads its far end. The
  MLP has 2 layers and outputs (dA_x, dA_y, d angle, d log L, d c2, d c3). The last layer starts at std 1e-2, as
  B's update does.

The update scales are fixed, not tuned: px for the anchor point, 0.05 rad per unit for the angle, 0.1 per unit for
log L and the curvatures.

The three forms keep their own states through all steps. They are three tracks, not one re-projected track,
because re-fitting every form to a consensus each step would erase the decorrelation the design is for.

### 3.3 Fusion: a learned profile along the lane, not a score

Each form is resampled at the same arc-length fractions u (0 = near, 1 = far; 73 points, as the output now).

`p(u) = sum_k w_k(u) p_k(u)`, with `w(u) = softmax_k(a_k(u))`

a_k(u) is a learned profile: 8 knots per form, linear in between, shared by all lanes and steps. That is 24
parameters, starting equal (1/3 each).

- It is not a per-lane gate and not a score. Per-lane learned gates collapsed before (curvature MoE, slot
  experts). A global profile cannot collapse lane by lane, and it is directly readable: it says where along a lane
  each anchor family is trusted.
- The prediction to check after training is near: V high, far: B/S. If the profile stays flat, the forms agree and
  the idea has no effect.

The fused curve is the lane. Matching, the standard losses, the score heads, the self-attention positions and the
output all use it, exactly as they use B's curve now. For the output tangents beyond the ends, the fused curve's own
end segments stand in for P1 - P0 and P3 - P2.

### 3.4 Losses: every form must be a full lane

- The standard loss (LaneIoU + endpoint, on o2o matches of the fused curve) is unchanged.
- Each form gets the same LaneIoU + endpoint loss on the same matched pairs, at every step, with weight 0.5
  (form_loss_weight). This is deep supervision of each branch, as in Co-DETR's auxiliary heads.
- This loss is what stops the fusion from collapsing onto one form. Each form stays a complete estimate of the whole
  lane, so the profile chooses between forms that are trained, not between a trained form and an untrained one.
- The VP point head: in off mode the column head becomes the point head of the polar mode (column + row), with its
  existing smooth-L1 loss on the GT VP (vanishing_point()). V uses it detached, so the lane losses do not pull the
  VP.

## 4. What stays fixed (the pair)

The new config is the pointcond config with one block added: `anchor_cfg` (+ the VP head becomes the 2-D point head,
which V needs).

Everything else is the same as the pointcond config:

- data, augmentation, schedule, optimizer;
- backbone, encoder, proposals;
- the decoder (6 steps, point-conditioned curve attention);
- o2o matching and cost, the loss weights, threshold sweep, eval.

anchor_cfg None = bit-identical to the current head, and a test checks it.

Pair: pointcond 86.58 (ep24). It is one seed each. The run variance is below +-1 F1, so a gain under ~1 is not
evidence.

## 5. Cost

Per query per step:

- B: unchanged.
- S and V: 3 bilinear reads + a 512 -> 256 -> 6 MLP each.
- 3 x 73-point arc-length resampling and fusion.

The total is well under 0.2 GMACs/img on top of 34.3, plus about 0.3 M parameters. It is real-time neutral. The VP
point head adds a 1-layer MLP.

## 6. What to read after the run

1. F1 (threshold swept on val) against 86.58.
2. The learned profile w_k(u).
3. Each form's own F1 (the tool can output any single form) and the fused F1. Fused > best single form means the
   fusion itself works.
4. Near-miss count (curvelanes_error_breakdown) against pointcond's. The claim is specifically fewer near-only
   misses.
5. If fused = B and the profile sits on B: V does not carry the near field. Then the anchor family is not what
   limits these lanes, and the evidence comes back to the image.

## 7. Risks, stated in advance

- V needs a VP. On curves, crests or with < 2 lanes the VP is poor (no GT VP in some images). V must then fit with
  delta and the curvatures, which is what per-form supervision trains. The global profile cannot switch V off for
  those images only; a per-image profile is the natural next step if the profile is the limit.
- V's ray is only as good as the VP relative to how far the far end is from it. A VP error e at distance r turns
  the ray by about e / r, so with delta = 0 the near end moves by about e x L / r (e.g. 5 px x 300 / 50 = 30 px). V is
  then a far-anchored line whose heading is regressed relative to a good, slant-normalized reference frame: an
  anchor that does not have to be exact. delta must carry the rest. Logged as vp_err_px and form_v_delta_deg. In the
  CPU overfit, V lagged S and B while the VP error was 5-12 px, and the profile moved V's near weight from 0.33 to
  0.25.
- If the near-miss error comes from the far part's angle and not from extrapolation, V inherits it. The
  measurements (far |d| 0.9 px) say the far part is right.
- One run of about 1.5-2 days on one GPU (batch 16 on GPU 3, as YSY runs).
