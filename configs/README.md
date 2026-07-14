# Config Registry

Hydra owns DreamerVLA experiment configuration. `configs/train.yaml` is the
stable training entrypoint; `experiment=<name>` selects one recipe under
`configs/experiment/`.

```text
configs/
├── train.yaml
├── experiment/
├── dreamervla/
├── worldmodel/
├── classifier/
├── pre_mainline/
├── evaluation/
├── task/
└── logger/
```

`configs/scripts/` is intentionally separate and contains only `install/`,
`download/`, and `preprocess/`. Training/evaluation shell entries select
`configs/experiment/` recipes directly; experiment `launch` blocks own local
torchrun/GPU metadata.

Shell launchers should stay thin. Put experiment behavior in config or Python
runners, and keep shell overrides limited to GPUs, data roots, output roots,
and smoke-run limits.

## Common Commands

```bash
python -m dreamervla.train experiment=openvla_onetraj_libero_cotrain task=openvla_onetraj_coldstart_libero
bash scripts/experiments/collect_rollouts/train.sh task=openvla_onetraj_coldstart_libero
python -m dreamervla.train experiment=wm_full_dataset_train task=openvla_onetraj_coldstart_libero
bash scripts/experiments/world_model_training/train.sh --config dino-wm
bash scripts/experiments/world_model_training/train.sh --config dreamer-wm
python -m dreamervla.train experiment=eval_libero_vla task=openvla_onetraj_libero
```

`logger=tensorboard_wandb` is the default grouped logger. Set
`runner.logger.wandb_mode=offline` for local-only W&B files, or select a single
backend with `logger=tensorboard` / `logger=wandb`.

## Entry Points

| Stage | Script | Default Config |
| --- | --- | --- |
| Failure-conditioned imagined RL | `scripts/experiments/cotrain/train.sh --config openvla_libero --wm_ckpt <wm> --cls_ckpt <cls>` | `openvla_libero` |
| Cotrain policy eval | `scripts/experiments/cotrain/eval.sh` | `eval_cotrain` |
| Official-data world model | `scripts/experiments/world_model_training/train.sh --config dino-wm\|dreamer-wm` | `dino-wm` / `dreamer-wm` |
| Bounded WM timing profile | `scripts/experiments/world_model_training/profile.sh` | `wm_official_upper_bound_profile` |
| Official-data classifier upper bound | `scripts/experiments/classifier_training/train.sh` | `classifier_official_upper_bound` |
| Rollout collection | `scripts/experiments/collect_rollouts/train.sh` | `collect_rollouts` |

## Experiments

| Experiment | Module group |
| --- | --- |
| `collect_rollouts` | Ray rollout collection |
| `openvla_onetraj_libero_cotrain` | canonical Ray cotrain base recipe |
| `wm_full_dataset_train` | full-replay WM warmup |
| `wm_official_upper_bound` | pre-mainline WM training from official data |
| `wm_dino_token_official` | DINO-WM architecture/data protocol over official OpenVLA-OFT tokens |
| `dino-wm` | user-facing DINO-WM recipe with Dreamer-WM-aligned batch size and learning rate |
| `dreamer-wm` | user-facing official-data Chunk-WM recipe |
| `wm_official_upper_bound_profile` | bounded 8-GPU timing run of the same optimized WM route |
| `classifier_official_upper_bound` | pre-mainline classifier training from official data |
| `wmpo_token_classifier_openvla_onetraj_libero_goal_h1` | token classifier recipe |
| `openvla_libero` | frozen-WM/CLS failure-conditioned OpenVLA imagined RL (Ray backend) |
| `eval_libero_vla` | LIBERO rollout eval |

The release training path is OpenVLA-OFT one-trajectory cold-start cotrain.
The two `*_official_upper_bound` stages form an isolated `libero_goal`-only
pre-mainline capacity check and are not release-mainline aliases. World-model
and classifier role construction live only in their component groups; experiments
select those groups and cotrain recipes do not duplicate their parameters. The
concrete classifier model, dataset targets, and input contract are owned together by
`task.classifier`; `classifier=dreamer-cls` consumes that task contract.
Both stages select `pre_mainline=libero_goal_official` for the immutable
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
├── logs/
├── tensorboard/
├── wandb/
├── video/{train,eval}/
├── diagnostics/
└── .hydra/
```

The default run root is
`${RUN_ROOT:-${DVLA_DATA_ROOT}/outputs}/${run.name}/${run.timestamp}`. Warmup and
periodic checkpoints are written under its `checkpoints/` directory. Supplying
`--resume <run-or-checkpoint>` to a train launcher restores the checkpoint and keeps
writing into that same run root.
