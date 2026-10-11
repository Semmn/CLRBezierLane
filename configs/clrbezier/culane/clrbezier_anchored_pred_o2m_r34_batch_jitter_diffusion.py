# Truncated anchored diffusion (bbox_head.diffusion_cfg, libs/clrbezier/diffusion.py)
# on the O2M jitter config, trained from scratch with the same 20-epoch schedule,
# so it compares directly with clrbezier_anchored_pred_o2m_r34_batch_jitter.py.
#
# Training: the main branch starts from noisy anchors, anchor + sigma(t) * eps on
# the 4 control-point x's and the start y, t uniform in 0..trunc_t. The noise is
# centred on each anchor (no DDPM shrink), so a noisy anchor stays local; at
# t = trunc_t its std is sigma_x_px in x and sigma_y in start y. The cascade
# predicts the clean lane and its score; a zero-initialised timestep embedding
# scales and shifts every stage's RoI feature. Assignment is unchanged: SimOTA on
# each stage's predictions. The base's main-branch stage jitter (after stages 0
# and 1) stays on: the noise moves the stage-0 input, the jitter the later
# handoffs. Rollout passes run as at test time, without jitter or dropout.
# Inference: DDIM over the whole cascade at infer_steps timesteps from the clean
# anchors (t = 50 -> 25 with 2 steps). infer_steps=1 costs what the base costs.
_base_ = ["./clrbezier_anchored_pred_o2m_r34_batch_jitter.py"]
cfg_name = "clrbezier_anchored_pred_o2m_r34_batch_jitter_diffusion.py"

model = dict(
    bbox_head=dict(
        diffusion_cfg=dict(
            timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2,
            trunc_t=50,          # train t in 0..50 of 1000 (DDPM noise growth over that range)
            schedule="ddpm",     # "ddpm" | "linear" (sigma proportional to t)
            sigma_x_px=16.0,     # x noise std at t = trunc_t, px of the 800-wide input
            sigma_y=0.02,        # start-y noise std at t = trunc_t (~1.4 rows)
            infer_steps=2,       # DDIM steps at test (each = one full cascade pass)
            infer_noise=False,   # True: start from anchor + noise (seeded) instead of the anchor
            infer_seed=0,
            rollout_prob=0.5,    # share of iterations that train a later sampler step
            time_cond=True,      # timestep scale/shift on the RoI features
        ),
    ),
)
