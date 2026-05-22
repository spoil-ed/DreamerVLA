# Script Entry Points

Scripts are thin wrappers around `python -m src.cli.train` or small diagnostic
tools. Keep durable logic in `src/`; keep scripts as reproducible launch
recipes.

## Data Preparation

| Script | Purpose |
| --- | --- |
| `env_libero_goal_pi0_query.sh` | Current pi0 action-hidden LIBERO-goal path / checkpoint / horizon registry |
| `env_libero_goal.sh` | Shared LIBERO-goal defaults for data prep and VLA SFT |
| `download_hf.sh` | Download model checkpoints from Hugging Face into `data/ckpts/` |
| `prepare_data.sh` | Standard LIBERO data preparation pipeline |
| `run_pi0_query_hidden_pipeline.sh` | Current action-hidden pipeline wrapper; preprocess, WM, and actor stages |
| `preprocess_rynn_pixel_hidden.sh` | Generate pi0 action-query hidden sidecar HDF5 files |
| `preprocess_rynn_pixel_hidden.py` | Python implementation for action-hidden sidecar generation |
| `prepare_dreamervla_data.sh` | DreamerVLA data preparation wrapper |
| `preprocess/*.sh` | Lower-level preprocessing steps retained for reproducibility |

Current canonical action-hidden sidecar for LIBERO-goal:

```text
data/processed_data/libero_goal_no_noops_t_256_pi0_action_hidden_vla_policy_h2
```

The current action-hidden wrappers source `env_libero_goal_pi0_query.sh`, which keeps these values
aligned:

```text
VLA_INIT_CKPT
VLA_STATE_CKPT / ENCODER_STATE_CKPT
ACTION_HORIZON / TIME_HORIZON
ACTION_HEAD_TYPE=pi0_query
PI0_ACTION_HIDDEN_DIR
PI0_QUERY_PROMPT_STYLE=vla_policy
PI0_QUERY_HISTORY=2
PI0_QUERY_INCLUDE_STATE=1
PI0_QUERY_ROTATE_IMAGES_180=1
```

## Training

| Script | Main Config | Public Workspace | Purpose |
| --- | --- | --- | --- |
| `pretokenize_train_vla.sh` | `pretokenize_vla_libero_goal_pi0_query` | `VLASFTWorkspace` | Current pi0 action-query VLA action head SFT when `ACTION_HEAD_TYPE=pi0_query` |
| `run_pi0_query_hidden_pipeline.sh` | `rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed` | `ActionHiddenWMWorkspace` | Current preprocess + action-hidden WM pipeline |
| `train_pi0_action_hidden_dreamerv3_wm.sh` | `rynn_backbone_dreamerv3_action_hidden_wm_libero_goal_precomputed` | `ActionHiddenWMWorkspace` | Current action-hidden DreamerV3 WM training |
| `run_pi0_action_hidden_reconstruct_actor.sh` | `dreamer_vla_libero_goal_pi0_action_hidden_head_actor` | `JointDreamerVLAWorkspace` | Current action-hidden actor training |
| `run_pi0_action_hidden_head_actor_variants.sh` | `dreamer_vla_libero_goal_pi0_action_hidden_head_actor` | `JointDreamerVLAWorkspace` | Current actor-head adapter sweep |
| `train_dreamerv3_pixel.sh` | `dreamerv3_pixel_libero_goal` | `PixelWMWorkspace` | Secondary pixel DreamerV3 baseline |
| `train_dreamerv3_token.sh` | `dreamerv3_token_libero_goal` | `TokenWMWorkspace` | Secondary token DreamerV3 baseline |
| `train_wm.sh` | env-selected | route-specific `src.workspace.*` target from config | Generic WM training wrapper |
| `train_chameleon_ladiwm_wm.sh` | `chameleon_latent_action_wm_libero_goal` | `ChameleonLatentWMWorkspace` | Chameleon / LaDiWM-style WM baseline |
| `train_dreamer_vla.sh` | `dreamer_vla_libero_goal_pi0_action_hidden_head_actor` | `JointDreamerVLAWorkspace` | Generic action-hidden DreamerVLA training wrapper |

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
