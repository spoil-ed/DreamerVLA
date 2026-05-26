# DreamerVLA Eval Diagnostic Protocol

This is the fixed comparison we should run for every new DreamerVLA checkpoint
before trusting LIBERO eval videos or success rates.

## Goal

Each run must answer three questions:

1. Is the eval input identical to the training/preprocess input?
2. Is the action-hidden / WM latent distribution aligned between offline data
   and online rollout?
3. Are both the original VLA actor head and the trained DreamerVLA actor
   producing usable actions?

The current tool is:

```bash
python -u scripts/diagnose_dreamervla_latent_distribution.py \
  --ckpt /path/to/dreamervla.ckpt \
  --encoder-ckpt /path/to/pi0_query_vla.ckpt \
  --tasks 0,3,6,8 \
  --episodes-per-task 1 \
  --online-steps 20 \
  --offline-batches 4 \
  --offline-batch-size 4 \
  --out data/outputs/diagnostics/<run_name>.json
```

The script writes both:

- `<run_name>.json`: full machine-readable metrics
- `<run_name>.md`: short human-readable report

## 1. Input Alignment

Record the effective input contract:

- VLA/model path
- encoder checkpoint
- action head type, expected `pi0_query`
- `prompt_style`, expected `vla_policy`
- `history`, expected `2`
- `include_state`, expected `true`
- `rotate_images_180`, expected `true`
- `obs_hidden_source`, expected `action_query`
- `time_horizon`, expected `5`
- prompt text format:
  `Finish the task: {task_description}.<|state|><|image|>*4`

Then compare online vs offline `obs_embedding`:

- `obs_embedding.std_ratio_online_over_offline`
- `obs_embedding.row_norm_ratio_online_over_offline`
- `obs_embedding.centered_norm_ratio_online_over_offline`

Expected: std and norm ratios should be close to `1.0`. If these are far from
1.0, debug prompt/history/image rotation before looking at actor behavior.

## 2. Action Hidden / Latent Distribution

Record offline and online distributions for:

- `obs_embedding`
- `posterior/deter_h`
- `posterior/stoch_z_flat`
- `posterior/feature_hz`
- `posterior/actor_input`
- posterior entropy and max probability if available

Required pair metrics:

- `hidden_live_vs_recon/cos`
- `hidden_live_vs_recon/mse`
- `action_original_live_vs_original_recon_raw/cos`
- `action_original_live_vs_original_recon_raw/mse`

Interpretation:

- If offline hidden live-vs-recon cosine is high and MSE is low, WM
  reconstruction is not the primary failure.
- If online hidden live-vs-recon is much worse than offline, check online RSSM
  rollout drift, previous-action convention, and posterior correction.
- If original actor live-vs-recon action stays close, reconstructed action
  hidden is usable by the VLA head.

## 3. Original Actor Vs Trained Actor

Always compare four paths on the same samples:

- `original_live`: original VLA `output_projection` on live action hidden
- `original_recon`: original VLA `output_projection` on WM reconstructed action hidden
- `trained_live`: trained Dreamer actor on live action hidden
- `trained_recon`: trained Dreamer actor on WM reconstructed action hidden

Record for each:

- raw action mean/std/min/max
- env action mean/std/min/max
- env action absolute mean
- env action row norm
- env action saturation rate, `abs(action) > 0.95`
- per-dimension env action mean/abs/min/max for:
  `dx, dy, dz, droll, dpitch, dyaw, grip`

Required pair metrics:

- `action_original_live_vs_original_recon_raw/cos`
- `action_original_live_vs_original_recon_raw/mse`
- `action_original_live_vs_trained_live_raw/cos`
- `action_original_live_vs_trained_live_raw/mse`
- `action_original_live_vs_trained_recon_raw/cos`
- `action_original_live_vs_trained_recon_raw/mse`

Interpretation:

- `original_live` is the VLA reference.
- `original_recon` should remain close to `original_live`; otherwise WM hidden
  reconstruction or online latent drift is suspect.
- `trained_live` isolates whether the trained actor adapter itself is healthy.
- `trained_recon` is the actual DreamerVLA actor path used in rollout.
- If `trained_live` and `trained_recon` both diverge from `original_live`, the
  trained actor/adapter is likely corrupting otherwise usable action hidden.

## Current Baseline Finding

For checkpoint:

`data/outputs/dreamervla/pi0_action_hidden_vla_policy_h2_actor_20260514_gpu4567/checkpoints/epoch=008-epoch_returns_mean=0.6341.ckpt`

Diagnostic output:

`data/outputs/diagnostics/dreamervla_actor_compare_h2_e8_perdim_20260514.json`

Main findings:

- `obs_embedding` online/offline std ratio: `1.0004`
- `obs_embedding` online/offline norm ratio: `1.0004`
- offline hidden live-vs-recon cosine: `0.9991`
- offline original actor live-vs-recon raw action MSE: `0.0010`
- online hidden live-vs-recon cosine: `0.9458`
- online original actor live-vs-recon raw action MSE: `0.0547`
- online original actor live-vs-trained recon raw action MSE: `1.5828`

Per-dimension online env action means:

| path | dx | dy | dz | droll | dpitch | dyaw | grip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_live | 0.270 | 0.124 | -0.574 | -0.009 | -0.003 | -0.006 | -0.977 |
| original_recon | 0.373 | 0.287 | -0.716 | 0.039 | 0.065 | -0.096 | -0.979 |
| trained_live | 0.934 | 0.211 | -0.937 | 0.233 | 0.375 | -0.364 | -0.995 |
| trained_recon | 0.938 | 0.142 | -0.938 | 0.247 | 0.375 | -0.364 | -1.000 |

Conclusion for this baseline: input/action-hidden alignment is mostly OK, but
the trained actor adapter pushes actions toward saturated fixed values. Future
runs should be compared against this table.
