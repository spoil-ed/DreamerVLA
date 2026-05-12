# Config Registry

This directory contains Hydra experiment configs. Treat configs as part of the
public API: scripts should select a config by name and pass only small,
explicit overrides for dataset paths, checkpoints, batch size, and GPUs.

## Current Mainline

Use these configs for the current LIBERO-goal / Rynn-pixel DreamerVLA path.

| Stage | Config | Entry Script | Status |
| --- | --- | --- | --- |
| VLA SFT | `pretokenize_vla_libero_goal.yaml` | `scripts/pretokenize_train_vla.sh` | active |
| Rynn-pixel WM | `rynn_backbone_dreamerv3_pixel_wm_libero_goal_precomputed.yaml` | `scripts/train_rynn_backbone_dreamerv3_wm.sh` | active |
| DreamerVLA cotrain | `dreamer_vla_libero_goal_rynn_pixel_precomputed_vlaactor.yaml` | `scripts/train_dreamer_vla_rynn_pixel.sh` | active |
| LIBERO eval | `eval_libero_vla.yaml` | `scripts/eval_libero_vla.sh` | active |

Canonical goal checkpoint / hidden pairing for the next clean run:

```text
VLA base:
  data/ckpts/VLA_model_256/libero_goal

VLA action head / encoder state:
  data/outputs/vla/pretokenize_vla/pretokenize_vla_libero_goal_libero_goal_h5_20260508_060320/checkpoints/goal_h5_epoch000_train_vla_loss_1p323.ckpt

Precomputed hidden sidecar:
  data/processed_data/libero_goal_no_noops_t_256_rynn_hidden_goal_h5_epoch000

Action horizon:
  5
```

Do not mix this with the older `libero_goal_no_noops_t_256_rynn_hidden`
sidecar. That sidecar was generated from a `libero_10`, horizon-10 VLA model
and is not compatible with the current goal action head.

The active Rynn-hidden dataset configs set:

```text
dataset.expected_model_path
dataset.expected_encoder_state_ckpt
dataset.expected_time_horizon
```

`LIBEROPixelRynnHiddenSequenceDataset` checks those against the sidecar
`preprocess_config.json` at startup, so stale hidden sidecars fail fast instead
of silently training.

## Active Baselines

These configs are still useful as maintained baselines.

| Config | Purpose |
| --- | --- |
| `dreamerv3_pixel_libero_goal.yaml` | DreamerV3 pixel world model baseline on LIBERO-goal |
| `dreamerv3_token_libero_goal.yaml` | DreamerV3 token world model baseline on LIBERO-goal |
| `dreamer_vla_libero_goal_dreamerv3_pixel_actor.yaml` | DreamerVLA with pixel WM and small actor |
| `dreamer_vla_libero_goal_dreamerv3_pixel_vlaactor.yaml` | DreamerVLA with pixel WM and VLA action-head actor |
| `dreamer_vla_libero_goal_dreamerv3_token_actor.yaml` | DreamerVLA token-WM actor baseline |

LIBERO-10 configs are archived under `configs/archive/libero10_legacy/` for
reproducibility only. The current mainline should use `libero_goal` naming and
data.

## Historical / Ablation Configs

These are kept to reproduce earlier experiments. They should not be used as
defaults for the current mainline without checking paths and horizons.

- `archive/libero10_legacy/dreamer_vla_libero_10*.yaml`
- `archive/libero10_legacy/pretokenize_wm_libero_10*.yaml`
- `archive/libero10_legacy/pretokenize_vla_libero_10.yaml`
- `archive/libero10_legacy/chameleon_latent_action_wm_libero_10.yaml`
- `pretokenize_wm_libero_goal_transdreamer.yaml`
- `chameleon_latent_action_wm_libero_goal.yaml`

## Output Layout

All configs should write under these roots:

```text
data/outputs/vla/
data/outputs/worldmodel/
data/outputs/dreamervla/
data/outputs/eval/
```

Do not create new top-level output categories unless the artifact type is
actually new. `data/` is intentionally ignored by git.

## Config Hygiene Rules

- Keep dataset names explicit: use `libero_goal`, `libero_10`,
  `libero_object`, or `libero_spatial`; avoid ambiguous tags.
- Keep VLA checkpoint, hidden sidecar, and `time_horizon` aligned.
- Put temporary experiments in `RUN_TAG`, not by copying another YAML.
- Prefer script environment variables for operational choices:
  `CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `BATCH_SIZE`, `NUM_WORKERS`,
  `RUN_TAG`, `OUT_DIR_BASE`.
- If a config is a new publishable baseline, add it to this file.
