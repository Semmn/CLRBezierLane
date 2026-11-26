# Multi-anchored lanes: what actually differs between anchorings, and a premise test

2026-11-26. Thread "Explore a new lane detection design". Status: analysis + premise test proposal. No code yet.

## 1. The idea as it stands

YSY's starting point: line, Bezier and polar anchors each have strengths, so let the detector use several (pick one,
or the top 2-3 per lane). Prior art (clrbezierlane/prior_art/new_design_leads_20261126.md, Idea 5): per-lane
selection between representations exists (UFLDv2 row/column hybrid, CANet row/column per instance). Not found:
several anchorings of the SAME lane fused by image region. That is the version kept here.

The measured failure it has to fix (CurveLanes, o2o ckpt, [near-miss-diagnosis]): near misses are full-length lanes
with a correct far end and a wrong near-field angle, i.e. a rigid rotation about the far end (|d| bottom/mid/top
4.5 / 2.6 / 0.9 eval px). Place-oracle +5.35 F1 (91.56), drop-oracle 88.94.

## 2. What differs between anchorings (the core argument)

Three things could differ when the same lane is written in another form. Only some of them are new.

| Property | Line (CLRNet start + angle) | Bezier (ours) | Polar / VP ray | Consequence |
|---|---|---|---|---|
| Shapes it can draw | lines (+ row offsets) | cubics, includes lines | lines through a centre | Nested. Cubic GT fit = 99.43 F1, so expressiveness is not the gap. |
| Where it is pinned | bottom start point | both endpoints P0, P3 | VP / centre | Bezier already has a near-field anchor (P0). A start-anchored form adds no new degree of freedom. |
| Where each parameter's evidence is read | CLRNet: pooled RoI over the whole line | `update = Linear(h)`: all 4 CP deltas from ONE pooled query vector (foveal_lane.py `_forward`, hblrnet.py:729) | — | This is the real difference. The near endpoint is corrected from the same vector as the far end. |
| How errors propagate | angle error grows with distance from the start | an error in P1/P2 rotates the near field about P3 | angle error grows away from the centre | Matches our failure: near field = extrapolation of far evidence. |

So "start-anchored vs VP-anchored" reduces to two concrete questions:

- **(M1) Decorrelated errors.** Do differently anchored models fail on different lanes? If yes, region fusion helps
  like a structured ensemble (near field from the start-anchored estimate, far field from the far/VP-anchored one).
- **(M2) Local read-out per anchor.** If each anchor's parameters were updated from features sampled AT that anchor
  instead of from one pooled vector, could the near end move without the far end? This is a property of the
  read-out, not of the representation. Bezier with per-endpoint read-out would do it.

Known evidence on each:

- M2 overlaps ideas already ruled out as read-outs from current features: near-field correction, strip refiner
  (85.19 vs 85.27), point-conditioned curve attention (85.70). The frozen-encoder offset probe placed near-only lanes
  within 6 px in 55% of cases when started near the GT, but only 14% from the head's own curve. So a local near-end
  read-out has a ceiling of about 14-55% of the near-only lanes, and the realistic end is near 14%.
- M1 has never been measured. We have models with different anchorings: o2o (Bezier endpoints), the old row cascade
  (CLR-style start-anchored rows), and the polar-view run (VP-centred view, Bezier output). We have never compared
  their near-only lanes lane by lane.

## 3. Premise test (eval-only, no training)

Goal: does M1 hold, i.e. does a start-anchored model place the near field of the lanes our Bezier model misses?

Inputs: per GT lane on the same 4000 CurveLanes val images, the best-matching prediction (curve points + score)
from two checkpoints: A = o2o (FovealLaneHead, lanes.csv from score_shift_check already has curve points), and
B = the old start-anchored cascade (needs a small dump script for its head) and/or the polar-view run.

Measures (metric = CurvelanesMetric 224x224 w5, IoU 0.5):

1. **Cross-hit rate.** Of A's near-only lanes (A hits far part, misses lane), the share that B hits outright, and the
   share where B's near third is within the metric tolerance.
2. **Error correlation.** Signed lateral error at the bottom third for A vs B on lanes both detect: Pearson r. r near
   1 = same failure (M1 dead); low r = complementary.
3. **Region-fusion oracle F1.** Output A's curve with its near third replaced by B's (smooth blend over the middle
   third), at A's score and threshold. This is the most region fusion could give with these two models, no scoring.

Read:

- Fusion oracle >= +1.0 F1 over A and cross-hit rate >= 30% -> M1 holds. Design the joint model (one query, two
  anchored forms, fixed region blend; section 4).
- Fusion oracle < +0.5 or r > 0.8 -> anchorings fail on the same lanes. Drop multi-anchoring; the failure is in the
  evidence, not in the form.

Caveat: the cascade is older and weaker (84.59 vs 86.06), so a null partly reflects model quality. A positive result
is still informative.

## 4. If M1 holds: the joint design (sketch, to be developed)

- One query per lane, two anchored forms decoded from it: S (near endpoint + near tangent + curvature) and F (far
  endpoint + far tangent + curvature). Each form's anchor parameters come from features sampled in a window at its
  own anchor (local read-out, M2); shape terms come from the pooled query.
- Output p(t) = w(t) p_S(t) + (1 - w(t)) p_F(t) on a shared lane parameter t (0 = near, 1 = far). w is a FIXED
  profile: 1 on the near third, 0 on the far third, linear between. No per-lane gate and no score, because learned
  gates collapsed before (curvature MoE, slot experts) and the role of each region is known in advance.
- Matching and LaneIoU loss on the fused curve; each form also gets a LaneIoU loss restricted to its own region, so
  each is trained to be right where it is pinned.
- Cost: two small heads and a second sampling window per query, about +1-2 GMACs on top of 34 GMACs.

## 5. Open question for YSY

Which of these checkpoints still exist on the server? o2o ep24 (86.06), the old start-anchored cascade on the
regenerated GT (84.59 or 85.27 base), the polar-view run. The test needs A plus at least one of the others.
