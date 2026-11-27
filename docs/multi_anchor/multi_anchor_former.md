# Multi-anchor BezierFormer: design

2026-11-27. Thread "Explore a new lane detection design". YSY's revisions of 05:00Z / 05:08Z:

- all four Bezier control points move freely;
- the polar anchor follows Polar R-CNN;
- no foveal modules;
- the base is the BezierFormer replica, the transformer network that does well on CurveLanes.

The design should show what multiple anchors are worth, with everything else held at the replica's settings.

Code: `libs/clrbezier/multi_anchor.py` (MultiAnchorFormerHead, a subclass of BezierFormerHead). Config:
`configs/clrbezier/curvelanes/multi_anchor_former_curvelanes_r18.py`. Pair:
`bezierformer_replica_simota_focalcost_flip_curvelanes_r18.py`.

## 1. What stays the replica's

The config differs from the pair in four places only. A test checks this.

1. The head type.
2. `anchor_cfg`.
3. `cfg_name`.
4. Batch 16 on one GPU.

Everything else is the replica's:

- R18 backbone and the 4-scale deformable encoder;
- E_0 and C_0 = MLP(X);
- the 6 decoder layers: self-attention, then Bezier curve attention at 5 reference points, then the FFN;
- per-layer focal classification;
- the loss on all 7 layers: Chamfer IoU, length and endpoint;
- SimOTA top-1 matching on the paper's cost;
- the 72-row test path.

With `forms=("B",)` the head computes the replica's lane through the fusion path. The overfit passes 5/5, as the
replica does. Its parameters are identical to the replica's at the same seed (tested).

## 2. The three anchored forms of one query

Each query carries its lane in three parametric forms. Each form has its own C_0 MLP and its own update FFN per
layer. The FFNs read the same query embedding E_i and are refined the way the replica refines C:

`state_i = state_{i-1}.detach() + FFN_i(E_i)`

| Form | Anchor family | Parameters | Lane points |
|---|---|---|---|
| B | BezierFormer | 4 free 2D control points, with no y order | the cubic |
| S | CLRNet line anchor | start (x_s, y_s), angle theta_s to the x axis (0..pi), far row y_e, x offsets dx_j | x_j = x_s + (y_s - y_j) cot theta_s + dx_j |
| V | Polar R-CNN polar anchor | theta_v, r about a FIXED global pole c, rows y_s, y_e, x offsets dx_j | cos theta_v (x - c_x) - sin theta_v (y - c_y) = r, so x_j = c_x + (r + sin theta_v (y_j - c_y)) / cos theta_v + dx_j |

All three forms start as the replica's fan of 16 straight lines, written in each form's own parameters. They are the
same lines at step 0 (tested). The fan meets at (399.5, 22.8) px, which is the default pole.

### Alignment with Polar R-CNN (arXiv 2411.01499)

**The same as Polar R-CNN:**

- One global pole, fixed, near the dataset's static vanishing point (Sec. III.A).
  `tools/clrbezier/static_pole.py` measures the pole as the median of the per-image GT vanishing points.
- The straight anchor in the normal form (theta, r), with theta in (-pi/2, pi/2) and r signed (Eq. A2 / 7). Their
  frame is y-up; here the image frame is y-down, so the sign of the sin term flips.
- The lane as that line plus per-row x offsets.

**Different on purpose:**

- *The offsets.* Polar R-CNN has 72 offsets at fixed image rows, each supervised by its own smooth-L1 x target. The
  replica's losses have no row correspondence, and YSY's rule is no point-by-point supervision. Under those losses,
  36 free offsets go unconstrained and jagged, and they slow every form down. In the CPU overfit:
  - S+B with 36 rows: 3/5 passed, S's own Chamfer IoU 0.16;
  - S+B with 36 offsets on fixed image rows: 3/5;
  - S+B with no offsets: 5/5;
  - S+B with 8 rows: 5/5, S's own Chamfer IoU 0.74.

  So S and V use 8 rows spaced over the lane's own extent. Its ends are differentiable, which the length and
  endpoint losses need. That is 12 numbers per lane, against B's 8.
- *The proposals.* Polar R-CNN's proposals come from the LPM: a 6 x 13 local-pole grid on P3 for CurveLanes, top-K
  50, followed by one refinement step. Here the proposals and refinement are the replica's (C_0 = MLP(X), then 6
  decoder layers), so that only the anchor family changes against the pair.

S follows CLRNet's anchor: the start point, the angle to the x axis, and per-row offsets.

## 3. Fusion: a learned profile along the lane

Every form's polyline is resampled at 73 arc-length fractions u, from 0 at the near end to 1 at the far end. The
fused lane is

`p(u) = sum_k w_k(u) p_k(u)`, with `w = softmax_k(profile_k(u))`

- The profile has 8 knots per form, linear in between, and is shared by all lanes and layers.
- It starts at zero, so all forms have equal weight at first.
- It is not a per-lane gate and not a score.

The fused lane is what the rest of the network uses:

- the next layer's reference points and position encoding;
- matching;
- the replica's loss;
- the output.

The output's end tangents come from the fused polyline's own end segments.

## 4. Loss

The replica's loss is applied to the fused lane at every layer.

On the same pairs, each form adds `0.5 x` (Chamfer IoU + length + endpoint). These terms are logged as
`loss_form_loc / _len / _end`. They keep each form a complete lane estimate, so the profile chooses between trained
forms.

## 5. Cost

- R18 at 800 x 320: 21.106 to 21.114 GMACs/img (+0.04%). CPU time is the same within noise (360 vs 342 ms, one
  measurement each).
- +1.7 M parameters. Most of them are the S and V C_0 MLPs, which output 16 x 12 numbers each from the pooled
  tokens. These are the same size as the replica's C_0 MLP for B.

## 6. What to read after the run

1. Fused F1 (threshold swept on val) against the pair's F1.
2. Each form alone: `--cfg-options model.bbox_head.anchor_cfg.output=S|B|V`. If fused beats the best single form,
   the fusion itself works. If B inside the multi-anchor model beats the pair, the other forms' supervision helps the
   shared queries.
3. The profile `form_w_<k>_near / _mid / _far`: where along a lane each anchor family is trusted. If it stays flat,
   the forms agree.
4. The near-miss count (curvelanes_error_breakdown) against the pair's.

## 7. CPU overfit (tests/multi_anchor_former/test_multi_anchor_former_overfit.py, 400 steps, seed 0)

| forms | rows | passed | fused Chamfer IoU | per-form Chamfer IoU |
|---|---|---|---|---|
| B | - | 5/5 | 0.765 | B 0.765 |
| S, B | 36 | 3/5 | 0.544 | S 0.16, B 0.62 |
| B, V | 36 | 3/5 | 0.453 | B 0.41, V 0.39 |
| S, B | 8 | 5/5 | 0.799 | S 0.74, B 0.73 |
| B, V | 8 | 5/5 | 0.790 | B 0.78, V 0.71 |
| **S, B, V (config)** | 8 | **5/5** | **0.866** | S 0.77, B 0.83, V 0.77 |

With all three forms, each form alone also finds every lane on the test path. The fused lane fits better than B
alone (0.866 vs 0.765), and B inside the model fits better than the replica's B. This is one seed of a 2-image
overfit, so it shows that the parts train together, not that they will gain on CurveLanes.
