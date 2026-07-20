# Script registry

Shell files in this directory are entrypoints, not configuration sources. Training and
evaluation defaults live in `configs/`; pass changes as Hydra `key=value` overrides.
Repository data paths are described in `docs/data_layout.md`.

On a networked CPU host that can read the GPU host's shared run directory, stream an
active offline W&B run after authenticating once:

```bash
wandb login
wandb beta sync --live /path/to/run_root/wandb
```

For noninteractive authentication, provide `WANDB_API_KEY` instead. Use W&B 0.24.1
or newer and start the command after `wandb/offline-run-*` exists. No repository
wrapper is provided; W&B reads entity, project, and run identity from the stream.

Training and evaluation entries under `scripts/experiments/` select recipes directly
from `configs/experiment/`. Configs under `configs/scripts/` are grouped as `install/`,
`download/`, `preprocess/`, and `reproduce/` workflows.

## Frozen bootstrap surface

The shell files in `download/`, `install/`, and `preprocess/` are the finalized
bootstrap surface for this repository. These files
must not be modified, renamed, reordered, or copied in part. Run them through the registered entrypoints and change supported
inputs only through Hydra overrides or the environment variables already declared by
the workflow configs.

This restriction applies to every `*.sh` file in those three directories, including
the unnumbered preprocessing wrappers. Keep the wrapper, Hydra config, numbered stage,
tests, and documentation as one indivisible contract. New training or evaluation
behavior belongs in `dreamervla/` and `configs/`, not in these frozen bootstrap files.

## Workflow entrypoints

- `install_env.sh`
- `download_assets.sh`
- `preprocess_libero.sh`

These run the Hydra install, download, and LIBERO preprocessing workflows,
respectively.

The workflow implementation steps are:

- `install/00_apt_tools.sh`
- `install/10_conda_env.sh`
- `install/20_torch.sh`
- `install/30_python_deps.sh`
- `install/40_third_party.sh`
- `install/50_special_packages.sh`
- `install/60_verify.sh`
- `download/00_openvla_oft.sh`
- `download/10_openvla_oft_one_trajectory.sh`
- `download/20_libero_dataset.sh`
- `download/30_calvin_dataset.sh`
- `preprocess/00_hdf5_reward.sh`
- `preprocess/10_oft_hidden_token.sh`
- `preprocess/20_validate.sh`
- `preprocess/prepare_libero_data.sh`
- `preprocess/process_all_libero_data.sh`
- `preprocess/validate_libero_data.sh`

## Experiment entrypoints

Docker reproduction:

- `reproduce/01_prepare_assets.sh`
- `reproduce/02_train_dreamer.sh`

Both are intended for `spoil/dreamervla:cu124-h100-v1`; see
`docs/docker_reproduction.md`. The first prepares and validates public assets. The
second runs WM 30 epochs, CLS 8 epochs, then frozen-WM/CLS Dreamer for 20,000 global
steps with automatic resume.

To reuse existing component checkpoints and run the isolated aggressive Dreamer
recipe, select its reproduce config explicitly:

```bash
bash scripts/reproduce/02_train_dreamer.sh \
  --config reproduce/train_dreamer_aggressive \
  --wm_ckpt /path/to/wm.ckpt \
  --cls_ckpt /path/to/classifier.ckpt
```

The checkpoint flags are atomic: both are required. This config skips WM/CLS
training, runs `openvla_libero_aggressive` for 20 global steps, imagines from all
replay episode starts, and evaluates every step. The resident eval logs
`eval/wm_trajectory_cosine`, `eval/cls_trajectory_f1`, and
`eval/cls_trajectory_accuracy` to the configured logger backends.

Mainline rollout collection:

- `experiments/collect_rollouts/train.sh`

Classifier training:

- `experiments/classifier_training/train.sh`

Official OpenVLA-OFT evaluation:

- `experiments/openvla_oft_official_eval/eval.sh`

Single-trajectory world-model overfit diagnostic:

- `experiments/single_trajectory_overfit/train.sh`
- `experiments/single_trajectory_overfit/eval.sh`

Full-dataset world-model training, profiling, and evaluation:

- `experiments/world_model_training/train.sh`
- `experiments/world_model_training/profile.sh`
- `experiments/world_model_training/eval.sh`

Select the world-model implementation through its Hydra experiment config:

```bash
bash scripts/experiments/world_model_training/train.sh --config dino-wm
bash scripts/experiments/world_model_training/train.sh --config dreamer-wm
```

The model configs live at `configs/worldmodel/dino-wm.yaml` and
`configs/worldmodel/dreamer-wm.yaml`; the same-named files under
`configs/experiment/` are thin experiment selectors. DINO-WM and Dreamer-WM both
use a per-rank batch size of 16 and learning rate `3e-5`; DINO-WM applies that
rate to both its predictor and conditioning optimizers. Training and optimizer
parameters remain under `configs/experiment/`; the shell launcher contains no
training defaults.

Full cotrain and frozen-WM/CLS imagined RL share one train/eval launcher pair:

- `experiments/cotrain/train.sh`
- `experiments/cotrain/eval.sh`

The release full-cotrain route updates WM/CLS and the policy:

```bash
bash scripts/experiments/cotrain/train.sh \
  --config openvla_onetraj_libero_cotrain \
  --wm_ckpt /path/to/wm.ckpt \
  --cls_ckpt /path/to/classifier.ckpt
```

The supporting `openvla_libero` route freezes WM/CLS, so both component
checkpoints are required unless the whole run is resumed:

```bash
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm.ckpt \
  --cls_ckpt /path/to/classifier.ckpt
```

`openvla_libero` selects the dedicated frozen imagined-RL `DreamerRunner`
internally. It deliberately uses this same cotrain launcher, so checkpoint and
resume parsing have only one shell entrypoint.

Checkpoints are flat files below the owning run root: `checkpoints/latest.ckpt` plus
configured `epoch=<epoch>-<metric>=<value>.ckpt` top-k files. Resume accepts a run
root, `checkpoints/`, or a checkpoint file and continues in the original run root:

```bash
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --resume /path/to/openvla_libero/<timestamp>
```

Evaluation requires a policy checkpoint as a Hydra override. Its fixed 100-episode,
25-environment protocol is defined by `configs/experiment/eval_cotrain.yaml`. The
launcher defaults to 8 GPUs, shards tasks across ranks, and rank 0 renders one global
episode progress line:

```bash
bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/openvla_libero/<timestamp>
```

Use `ngpu=<N> gpus=<comma-separated-ids>` to override the default GPU set.
Evaluation writes to `outputs/eval/<task-suite>/` with no timestamp child.
