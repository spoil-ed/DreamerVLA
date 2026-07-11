# Config Registry

Hydra owns DreamerVLA experiment configuration. `configs/train.yaml` is the
stable training entrypoint; `experiment=<name>` selects one recipe under
`configs/experiment/`.

```text
configs/
├── train.yaml
├── experiment/
├── dreamervla/
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
python -m dreamervla.train experiment=eval_libero_vla task=openvla_onetraj_libero
```

`logger=tensorboard_wandb` is the default grouped logger. Set
`runner.logger.wandb_mode=offline` for local-only W&B files, or select a single
backend with `logger=tensorboard` / `logger=wandb`.

## Entry Points

| Stage | Script | Default Config |
| --- | --- | --- |
| Ray cold-start cotrain | `scripts/e2e_coldstart_warmup_cotrain_ray.sh` | `configs/scripts/coldstart_warmup_cotrain.yaml` |
| Sync cold-start cotrain | `scripts/e2e_coldstart_warmup_cotrain_noray.sh` | `configs/scripts/coldstart_warmup_cotrain.yaml` |
| Manual async cotrain | `scripts/e2e_manual_cotrain_async.sh` | `openvla_onetraj_libero_cotrain_ray` |
| Full-dataset WM warmup | `scripts/experiments/world_model_training/train.sh` | `wm_full_dataset_train` |
| LIBERO eval | `scripts/eval_libero_vla.sh` | `eval_libero_vla` |

## Experiments

| Experiment | Module group |
| --- | --- |
| `collect_rollouts_onetraj` | rollout collection |
| `collect_rollouts_ray` | Ray rollout collection |
| `openvla_onetraj_libero_cotrain_noray` | sync warmup + cotrain |
| `openvla_onetraj_libero_cotrain_ray` | Ray manual cotrain |
| `wm_full_dataset_train` | full-replay WM warmup |
| `latent_classifier_openvla_onetraj_libero_goal_h1` | classifier warmup |
| `wmpo_token_classifier_openvla_onetraj_libero_goal_h1` | token classifier recipe |
| `eval_libero_vla` | LIBERO rollout eval |

The release training path is OpenVLA-OFT one-trajectory cold-start cotrain.

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
`task.openvla_oft.input_tokens` with `input_token_embedding [256,4096]` and
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
