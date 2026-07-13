# Config Registry

Hydra owns DreamerVLA experiment configuration. `configs/train.yaml` is the
stable training entrypoint; `experiment=<name>` selects one recipe under
`configs/experiment/`.

```text
configs/
├── train.yaml
├── experiment/
├── dreamervla/
├── classifier/
├── pre_mainline/
├── evaluation/
├── task/
├── logger/
├── precision/
├── parallelism/
└── scheduler/
```

Shell launchers should stay thin. Put experiment behavior in config or Python
runners, and keep shell overrides limited to GPUs, data roots, output roots,
and smoke-run limits.

## Common Commands

```bash
python -m dreamervla.train experiment=openvla_onetraj_libero_cotrain_noray task=openvla_onetraj_coldstart_libero
python -m dreamervla.train experiment=openvla_onetraj_libero_cotrain_ray task=openvla_onetraj_coldstart_libero
python -m dreamervla.train experiment=wm_full_dataset_train task=openvla_onetraj_coldstart_libero
python -m dreamervla.train experiment=dreamervla_frozen_models_rl task=openvla_onetraj_libero \
  init.world_model_state_ckpt=<wm.ckpt> init.classifier_state_ckpt=<classifier.ckpt>
python -m dreamervla.train experiment=dreamervla_frozen_models_rl_ray task=openvla_onetraj_libero \
  init.world_model_state_ckpt=<wm.ckpt> init.classifier_state_ckpt=<classifier.ckpt>
python -m dreamervla.train experiment=eval_libero_vla task=openvla_onetraj_libero
```

`logger=tensorboard_wandb` is the default grouped logger. Set
`runner.logger.wandb_mode=offline` for local-only W&B files, or select a single
backend with `logger=tensorboard` / `logger=wandb`.

## Entry Points

| Stage | Script | Default Config |
| --- | --- | --- |
| Trainable WM/CLS cotrain | `scripts/experiments/cotrain/train.sh` | `dreamervla_wmcls_cotrain_ray` |
| Cotrain policy eval | `scripts/experiments/cotrain/eval.sh` | `eval_cotrain` |
| Official-data WM upper bound | `scripts/experiments/world_model_training/train.sh` | `wm_official_upper_bound` |
| Bounded WM timing profile | `scripts/experiments/world_model_training/profile.sh` | `wm_official_upper_bound_profile` |
| Official-data classifier upper bound | `scripts/experiments/classifier_training/train.sh` | `classifier_official_upper_bound` |
| Cold-start collect/warmup pipeline | `python -m dreamervla.launchers.coldstart_warmup_cotrain` | `configs/scripts/coldstart_warmup_cotrain.yaml` |
| Pre-mainline frozen-model proof | `python -m dreamervla.launchers.frozen_model_pre_mainline` | `configs/scripts/frozen_model_pre_mainline.yaml` |

## Experiments

| Experiment | Module group |
| --- | --- |
| `collect_rollouts_onetraj` | rollout collection |
| `collect_rollouts_ray` | Ray rollout collection |
| `openvla_onetraj_libero_cotrain_noray` | sync warmup + cotrain |
| `openvla_onetraj_libero_cotrain_ray` | Ray manual cotrain |
| `wm_full_dataset_train` | full-replay WM warmup |
| `wm_official_upper_bound` | pre-mainline WM training from official data |
| `wm_official_upper_bound_profile` | bounded 8-GPU timing run of the same optimized WM route |
| `classifier_official_upper_bound` | pre-mainline classifier training from official data |
| `dreamervla_frozen_models_rl` | policy-only imagined RL with immutable WM/CLS |
| `dreamervla_frozen_models_rl_ray` | 8-GPU Ray/FSDP policy-only imagined RL with immutable WM/CLS |
| `latent_classifier_openvla_onetraj_libero_goal_h1` | classifier warmup |
| `wmpo_token_classifier_openvla_onetraj_libero_goal_h1` | token classifier recipe |
| `eval_libero_vla` | LIBERO rollout eval |

The release training path is OpenVLA-OFT one-trajectory cold-start cotrain.
The two `*_official_upper_bound` stages plus the single-process and Ray
`frozen_models_rl*` realizations form an isolated `libero_goal`-only
pre-mainline feasibility gate and are not release-mainline aliases.
The official classifier stage and frozen-RL stage both select
`classifier=openvla_oft_spatial`; construction is shared as one Hydra component
instead of being copied into experiment-specific Python classes.
All three stages select `pre_mainline=libero_goal_official` for the immutable
ten-task/ten-shard official-data manifest.

## Task Configs

Concrete LIBERO task configs live under `configs/task/`:

```text
task/libero_goal.yaml
task/libero_object.yaml
task/libero_spatial.yaml
task/libero_10.yaml
task/openvla_onetraj_libero.yaml
task/openvla_onetraj_libero_object.yaml
task/openvla_onetraj_libero_spatial.yaml
task/openvla_onetraj_libero_10.yaml
task/openvla_onetraj_coldstart_libero*.yaml
```

Cold-start configs reroot HDF5 paths under collected rollout directories. The
OpenVLA-OFT block carries checkpoint paths, sidecar expectations, token
dimensions, proprio dimensions, model dimensions, actor/classifier targets, and
world-model sequence length. The one-trajectory mainline contract is
`task.openvla_oft.hidden_token` with `hidden_token [256,4096]` and
`wm_obs_dim=1048576`.

## Runtime Artifacts

Runners write under one `${training.out_dir}`:

```text
${training.out_dir}/
├── resolved_config.yaml
├── run_manifest.json
├── checkpoints/
├── log/
│   ├── tensorboard/
│   └── wandb/
├── video/
└── diagnostics/
```

Warmup pipeline checkpoints are written under `${RUN_ROOT}/cotrain/ckpt/`.
