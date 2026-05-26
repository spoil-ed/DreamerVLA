# Script Entry Points

Scripts are thin wrappers around `python -m src.cli.train` or small diagnostic
tools. Keep durable logic in `src/`; keep scripts as reproducible launch
recipes.

## Data Preparation

Historical data-preparation shell recipes are archived under
`scripts/archive/uncertain_shells/`. Current training launchers take task
metadata from `configs/task/*.yaml`.

## Training

| Script | Main Config | Public Workspace | Purpose |
| --- | --- | --- | --- |
| `train_vla.sh` | `vla_pi0_query` | `VLASFTWorkspace` | VLA SFT |
| `train_vla_nongoal_45.sh` | `vla_pi0_query` | `VLASFTWorkspace` | LIBERO non-goal VLA SFT on GPUs 4,5; switch task with `TAG=<tag>` |
| `train_wm.sh` | `world_model_rssm_step`, `world_model_dinowm_step`, `world_model_dinowm_chunk` | route-specific `src.workspace.*` target from config | WM training |
| `train_dreamervla.sh` | `dreamervla_pi0_action_hidden_head_actor`, `dreamervla_rynn_dino_wm_actor_critic`, `dreamervla_rynn_dino_wm_wmpo_outcome` | `JointDreamerVLAWorkspace` | DreamerVLA training |

Configs point directly at the route-specific workspace class.

Most wrappers accept standard environment variables:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
BATCH_SIZE=96
NUM_WORKERS=2
RUN_TAG=my_run
OUT_DIR_BASE=/path/to/output/root
```

`DETACH=1` backgrounds training and writes a `train.pid`; omit `DETACH` to keep
logs in the terminal.

Non-goal LIBERO VLA SFT uses suite-specific pretrained weights under
`data/ckpts/VLA_model_256/<suite>`:

```bash
bash scripts/train_vla_nongoal_45.sh libero_10
TAG=libero_object bash scripts/train_vla_nongoal_45.sh
TAG=libero_spatial bash scripts/train_vla_nongoal_45.sh
```

## Evaluation

| Script | Purpose |
| --- | --- |
| `eval_libero_vla.sh` | Single-process LIBERO rollout eval for VLA or Dreamer checkpoints |
| `eval_libero.sh` | Legacy LIBERO eval wrapper |
| `evals_libero/*.sh` | Task-suite specific eval wrappers |

For Dreamer checkpoints, `eval_libero_vla.sh` supports:

```text
eval.ckpt_kind=dreamer
eval.dreamer_policy_source=ckpt|init
eval.dreamer_actor_input_source=rssm|encoder|encoder_sequence
```

These switches are useful for actor and hidden ablations.

## Diagnostics

| Script | Purpose |
| --- | --- |
| `analyze_rynn_hidden_action_metrics.py` | Offline hidden/action mismatch metrics |
| `monitor_dreamer_vla_metrics.py` | Summarize training log trends |
| `visualize_dreamervla_reward.py` | Reward-model visualization helper |
| `smoke_libero_online_env.py` | Smoke test for online LIBERO env wiring |

Diagnostic outputs should go under:

```text
data/outputs/eval/
```

## Script Hygiene

- Do not hard-code a new experiment path if an env var or Hydra override is enough.
- Put long-lived launch defaults in the wrapper; put experiment identity in `RUN_TAG`.
- Keep logs under `data/outputs/...`; do not write logs into the repo root.
- If a script becomes a stable entry point, list it here.
