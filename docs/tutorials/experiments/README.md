# Experiment tutorials

Current recipes use Hydra configuration. See the
[complete loop](../../architecture/04_complete_loop.md) for orchestration and
[PARAMETERS.md](../../PARAMETERS.md) for launcher overrides. Training defaults
use eight GPUs; batch sizes, horizons, and checkpoint paths belong to the
selected recipe.

## Recipes

| Workflow | Guide | Configuration |
| --- | --- | --- |
| OpenVLA one-trajectory mainline | [Collection, warmup, cotrain, evaluation](OpenVLA_Onetraj_LIBERO.md) | `collect_rollouts`, `openvla_libero`, `eval_cotrain` |
| Aggressive Dreamer comparison | [Experiment guide](2026-07-20_Aggressive_Dreamer_LIBERO.md) | `openvla_libero_aggressive` |
| Imagined-success SFT signal probe | [Experiment guide](2026-07-20_Imagined_Success_SFT_Probe_LIBERO.md) | `openvla_libero_success_sft_probe` |
| π0.5 SFT with local LeRobot v3 | [Data layout](../../data_layout.md), [installation](../../install.md) | `pi05_libero_sft` |
| π0.5 prefix-input WM | [Representation and launch](../../pi05_prefix_input_latent.md) | `wm_pi05_prefix_input_train` |
| π0.5 V-JEPA2-AC WM | [Latent supervision and decoder evaluation](../../vjepa2_ac_initialization.md) | `wm_pi05_vjepa2_latent_train` |

The aggressive comparison and SFT probe are explicit opt-in experiments.
Their guides describe their image/checkpoint requirements; the default
`openvla_libero` route uses failure-conditioned imagined PPO with frozen
encoder, world model, and classifier.

## Diagnostics

The [single-trajectory overfit launcher](../../../scripts/experiments/single_trajectory_overfit/train.sh)
checks its inputs before training. Its
[Python entrypoint](../../../dreamervla/diagnostics/benchmarks/wm_single_trajectory_overfit.py)
records true, zero, and random action-chunk rollout comparisons in
`metrics.jsonl`, `summary.json`, and `summary.md`.
