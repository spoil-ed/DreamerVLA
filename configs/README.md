# Config Registry

Hydra owns DreamerVLA experiment configuration. `configs/train.yaml` is the
stable training entrypoint; `experiment=<name>` selects one recipe under
`configs/experiment/`.

```text
configs/
├── train.yaml
├── experiment/
├── profile/
├── launch/
├── dreamervla/
├── worldmodel/
├── classifier/
├── pre_mainline/
├── evaluation/
├── task/
└── logger/
```

`configs/scripts/` is intentionally separate and contains `install/`, `download/`,
`preprocess/`, and the public `reproduce/` orchestration configs. Ordinary
training/evaluation shell entries select
`configs/experiment/` recipes directly; experiment `launch` blocks own local
torchrun/GPU metadata.

Shell launchers should stay thin. Put experiment behavior in config or Python
runners, and keep shell overrides limited to GPUs, data roots, output roots,
and smoke-run limits.

## Common Commands

```bash
python -m dreamervla.train experiment=openvla_onetraj_libero_cotrain profile=production
bash scripts/experiments/collect_rollouts/train.sh task=openvla_onetraj_coldstart_libero
python -m dreamervla.train experiment=wm_full_dataset_train task=openvla_onetraj_coldstart_libero
bash scripts/experiments/world_model_training/train.sh --config dino-wm
bash scripts/experiments/world_model_training/train.sh --config dreamer-wm
python -m dreamervla.train experiment=eval_libero_vla task=openvla_onetraj_libero
```

`logger=tensorboard_wandb` is the default grouped logger. Set
`runner.logger.wandb_mode=offline` for local-only W&B files, or select a single
backend with `logger=tensorboard` / `logger=wandb`.

Logger resume follows the checkpoint-owning run root. TensorBoard writes another
event file in the same `tensorboard/` directory and purges the abandoned tail at
the restored step. W&B persists its stable ID in `wandb/run_id.txt`; online mode
resumes that run directly and, on SDKs with `resume_from`, truncates its abandoned
tail at the restored step. Offline mode writes one local `offline-run-*` segment per
process with the same ID. On a networked CPU host that shares the run directory,
stream the active logical run with the official W&B CLI:

```bash
wandb beta sync --live /path/to/run_root/wandb
```

## Entry Points

| Stage | Script | Default Config |
| --- | --- | --- |
| Docker asset preparation | `scripts/reproduce/01_prepare_assets.sh` | `scripts/reproduce/prepare_assets` |
| Docker WM/CLS/Dreamer chain | `scripts/reproduce/02_train_dreamer.sh` | `scripts/reproduce/train_dreamer` |
| Aggressive Dreamer from WM/CLS checkpoints | `scripts/reproduce/02_train_dreamer.sh --config reproduce/train_dreamer_aggressive --wm_ckpt <wm> --cls_ckpt <cls>` | `scripts/reproduce/train_dreamer_aggressive` |
| Imagined-success SFT signal probe | `scripts/reproduce/02_train_dreamer.sh --config reproduce/train_dreamer_success_sft_probe --wm_ckpt <wm> --cls_ckpt <cls>` | `scripts/reproduce/train_dreamer_success_sft_probe` |
| Full online cotrain | `scripts/experiments/cotrain/train.sh --config openvla_onetraj_libero_cotrain --wm_ckpt <wm> --cls_ckpt <cls>` | `openvla_onetraj_libero_cotrain` |
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
| `openvla_libero_aggressive` | opt-in 20-step frozen-WM/CLS route that imagines from all replay trajectories and performs every-step resident WM/CLS evaluation |
| `openvla_libero_success_sft_probe` | one-step frozen-WM/CLS signal probe that SFTs only classifier-success imagined trajectories |
| `eval_libero_vla` | LIBERO rollout eval |

`profile=production` preserves the selected experiment. `profile=debug` declares
short budgets, while `profile=smoke` declares a complete two-GPU real+WM topology.
Profiles are composed after experiments, so every effective budget remains visible
in Hydra's `.hydra/config.yaml`.

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
├── run_manifest.json
├── checkpoints/
│   ├── latest.ckpt
│   └── epoch=<epoch>-<metric>=<value>.ckpt
├── checkpoint_hf/              # explicit HF export only
├── logs/
├── tensorboard/
├── wandb/
├── video/{train,eval}/
├── diagnostics/
└── .hydra/
```

The default run root is
`${RUN_ROOT:-${DVLA_DATA_ROOT}/outputs}/${run.name}/${run.timestamp}`. Warmup and
periodic checkpoints are flat files under its `checkpoints/` directory. Supplying
`--resume <run-or-checkpoint>` to a train launcher restores the checkpoint and keeps
writing into that same run root, including the TensorBoard timeline and W&B run
identity.

Evaluation instead owns `${run.output_root}/eval/${eval.task_suite_name}` directly;
the task directory is the run root and has no timestamp child. Evaluation accepts a
specific checkpoint, `checkpoints/`, or a training run root and reads `latest.ckpt`.
