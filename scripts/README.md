# Script Entry Points

Scripts are thin wrappers around `python -m src.cli.train` or small diagnostic
tools. Keep durable logic in `src/`; keep scripts as reproducible launch
recipes.

## Data Preparation

| Script | Purpose |
| --- | --- |
| `env_libero_goal.sh` | Canonical LIBERO-goal path / checkpoint / horizon registry sourced by launch wrappers |
| `download_hf.sh` | Download model checkpoints from Hugging Face into `data/ckpts/` |
| `prepare_data.sh` | Standard LIBERO data preparation pipeline |
| `prepare_dreamervla_data.sh` | From-zero DreamerVLA data pipeline, including Rynn full token hidden sidecar |
| `prepare_latent_data.sh` | One-shot wrapper for hidden / latent sidecar preparation |
| `preprocess_rynn_pixel_hidden.sh` | Generate RynnVLA hidden sidecar HDF5 files |
| `preprocess_rynn_pixel_hidden.py` | Python implementation for the hidden sidecar generator |
| `preprocess/*.sh` | Lower-level preprocessing steps retained for reproducibility |

Current canonical hidden sidecars for LIBERO-goal:

```text
data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_h5_epoch000
data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_h5_epoch000_fullseq
```

The active wrappers source `env_libero_goal.sh`, which keeps these values
aligned:

```text
VLA_INIT_CKPT
VLA_STATE_CKPT / ENCODER_STATE_CKPT
ACTION_HORIZON / TIME_HORIZON
RYNN_HIDDEN_DIR
RYNN_HIDDEN_FULLSEQ_DIR
```

## Training

| Script | Main Config | Purpose |
| --- | --- | --- |
| `pretokenize_train_vla.sh` | `pretokenize_vla_libero_goal` | Train / finetune VLA action head |
| `train_rynn_backbone_dreamerv3_wm.sh` | `rynn_backbone_dreamerv3_pixel_wm_libero_goal_precomputed` | Train Rynn-hidden DreamerV3 world model |
| `train_dreamer_vla_rynn_pixel.sh` | `dreamer_vla_libero_goal_rynn_pixel_precomputed_vlaactor` | Train DreamerVLA with Rynn-pixel WM and VLA action-head actor |
| `train_dreamerv3_pixel.sh` | `dreamerv3_pixel_libero_goal` | Pixel DreamerV3 baseline |
| `train_dreamerv3_token.sh` | `dreamerv3_token_libero_goal` | Token DreamerV3 baseline |
| `train_wm.sh` | env-selected | Generic WM training wrapper |
| `train_dreamer_vla.sh` | env-selected | Generic DreamerVLA training wrapper |

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
| `eval_wm.sh` | World-model diagnostics / reconstruction eval |
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
| `diagnose_wm.sh` | WM diagnostic wrapper |

Diagnostic outputs should go under:

```text
data/outputs/eval/
```

## Script Hygiene

- Do not hard-code a new experiment path if an env var or Hydra override is enough.
- Put long-lived launch defaults in the wrapper; put experiment identity in `RUN_TAG`.
- Keep logs under `data/outputs/...`; do not write logs into the repo root.
- If a script becomes a stable entry point, list it here.
