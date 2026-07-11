# DreamerVLA

DreamerVLA is a single-machine multi-GPU research stack for LIBERO VLA
rollout collection, world-model warmup, success-classifier warmup, and online
cotrain.

```text
LIBERO rollouts
  -> reward + hidden HDF5 shards
  -> world model + success classifier warmup
  -> OpenVLA-OFT cotrain
  -> LIBERO rollout evaluation
```

## Quick Start

```bash
git clone <repo> && cd DreamerVLA
export DVLA_DATA_ROOT=data
bash scripts/install_env.sh
conda activate dreamervla
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]
bash scripts/download_assets.sh only=[40_libero_dataset] env.LIBERO_SUITES=libero_goal

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=8 profile=multi_gpu render_backend=osmesa
```

For the independent official-data upper-bound jobs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  GPU_COUNT=8 \
  DVLA_DATA_ROOT=/path/to/data \
  bash scripts/experiments/world_model_training/train.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  GPU_COUNT=8 \
  DVLA_DATA_ROOT=/path/to/data \
  bash scripts/experiments/classifier_training/train.sh
```

## Pre-Mainline Frozen-Model Feasibility Test

Before entering the formal cotrain mainline, the isolated causal test trains WM
and classifier upper bounds from official LIBERO data, freezes both modules, and
trains only the DreamerVLA policy through imagined LUMOS rollouts. This first
proof route is intentionally `libero_goal`-only:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_frozen_model_pre_mainline.sh task=goal ngpu=8
```

It then evaluates the unmodified one-trajectory OpenVLA-OFT checkpoint and the
learned policy with identical real-LIBERO metadata. The gate passes only for a
strict success-rate improvement, an updated policy hash, at least one applied
policy step, unchanged WM/CLS hashes, and exact evaluated-checkpoint hash
binding. Use `stage=wm|classifier|rl|eval` to resume by stage or `dry_run=true`
to inspect commands. This test does not replace the
`collect -> warmup -> online cotrain` mainline.

## Reproduction Route

1. Install the environment with `scripts/install_env.sh`.
2. Download OpenVLA-OFT one-trajectory checkpoints and LIBERO data with
   `scripts/download_assets.sh`.
3. Collect rollout shards with `scripts/e2e_coldstart_warmup_cotrain_ray.sh`
   or `scripts/e2e_coldstart_warmup_cotrain_noray.sh`.
4. Warm up the world model and classifier from collected replay.
5. Continue with online cotrain when `online_rollout.total_env_steps` is raised
   above zero.
6. Evaluate with `scripts/eval_libero_vla.sh`.

## Repository Layout

```text
dreamervla/        Python package: runners, models, datasets, algorithms, envs
configs/            Hydra recipes and LIBERO task configs
scripts/            shell launchers for install, download, preprocess, train, eval
tests/              unit and smoke tests
third_party/        ignored, read-only upstream runtime dependencies
data/               relative default runtime data root
docs/               documentation index, references, tutorials, reports, papers
```

## Entry Points

| Stage | Command |
| --- | --- |
| Install | `bash scripts/install_env.sh` |
| Download OpenVLA-OFT one-trajectory | `bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]` |
| Download LIBERO | `bash scripts/download_assets.sh only=[40_libero_dataset] env.LIBERO_SUITES=libero_goal` |
| Ray cold-start cotrain | `bash scripts/e2e_coldstart_warmup_cotrain_ray.sh task=goal ngpu=8 profile=multi_gpu render_backend=osmesa` |
| Sync cold-start cotrain | `bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=goal ngpu=8 profile=multi_gpu render_backend=osmesa` |
| Full-dataset WM warmup | `bash scripts/experiments/world_model_training/train.sh` |
| Pre-mainline frozen WM/CLS policy test | `bash scripts/e2e_frozen_model_pre_mainline.sh task=goal ngpu=8` |
| Eval | `bash scripts/eval_libero_vla.sh gpus=0 eval.ckpt_path=<ckpt> eval.ckpt_kind=auto` |

Common overrides:

```bash
DVLA_DATA_ROOT=data
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=8 profile=multi_gpu render_backend=osmesa
bash scripts/experiments/world_model_training/train.sh \
  training.out_dir="${DVLA_DATA_ROOT}/outputs/wm_full_dataset_train/run"
```

`DVLA_DATA_ROOT` is independent of `DVLA_ROOT`; use a separate disk or shared
storage path when that is more convenient.

## Config Fields

- `offline_warmup.data_dir`: collected reward-HDF5 replay directory.
- `offline_warmup.hidden_dir`: collected hidden-sidecar directory.
- `task.openvla_oft.input_tokens.*`: projected input-token dimensions and sidecar contract.
- `training.wm_warmup_steps`: world-model warmup update budget.
- `training.classifier_warmup_steps`: success-classifier warmup update budget.
- `dataloader.batch_size`: per-rank replay batch size.
- `online_rollout.sequence_length`: replay window length.
- `online_rollout.total_env_steps`: online cotrain budget.

## Verify

```bash
python -m pytest tests/unit_tests -q
```

See [SETUP.md](SETUP.md) for the full workflow and
[docs/data_layout.md](docs/data_layout.md) for path conventions.
