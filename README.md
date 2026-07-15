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
bash scripts/download_assets.sh download.openvla_one_traj=true only=[10_openvla_oft_one_trajectory]
bash scripts/download_assets.sh only=[20_libero_dataset] env.LIBERO_SUITES=libero_goal

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm_warmup.ckpt \
  --cls_ckpt /path/to/classifier_warmup.ckpt

bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/manual_cotrain.ckpt
```

For the independent official-data upper-bound jobs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DVLA_DATA_ROOT=/path/to/data \
  bash scripts/experiments/world_model_training/train.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DVLA_DATA_ROOT=/path/to/data \
  bash scripts/experiments/classifier_training/train.sh
```

## Pre-Mainline Frozen-Model Feasibility Test

Before entering the main cotrain workflow, the isolated causal test trains WM
and classifier upper bounds from official LIBERO data, freezes both modules, and
trains only the DreamerVLA policy through imagined LUMOS rollouts. This first
proof route is intentionally `libero_goal`-only:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m dreamervla.launchers.frozen_model_pre_mainline task=goal ngpu=8
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
3. Train with `scripts/experiments/cotrain/train.sh`, explicitly supplying the
   frozen WM and classifier checkpoints.
4. Evaluate an explicit policy checkpoint with
   `scripts/experiments/cotrain/eval.sh`.

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
| Download OpenVLA-OFT one-trajectory | `bash scripts/download_assets.sh download.openvla_one_traj=true only=[10_openvla_oft_one_trajectory]` |
| Download LIBERO | `bash scripts/download_assets.sh only=[20_libero_dataset] env.LIBERO_SUITES=libero_goal` |
| WM/CLS cotrain | `bash scripts/experiments/cotrain/train.sh --config openvla_libero --wm_ckpt <wm-ckpt> --cls_ckpt <cls-ckpt>` |
| Full-dataset WM warmup | `bash scripts/experiments/world_model_training/train.sh` |
| Pre-mainline frozen WM/CLS policy test | `python -m dreamervla.launchers.frozen_model_pre_mainline task=goal ngpu=8` |
| Cotrain eval | `bash scripts/experiments/cotrain/eval.sh eval.ckpt_path=<ckpt>` |

Common overrides:

```bash
DVLA_DATA_ROOT=data
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm_warmup.ckpt \
  --cls_ckpt /path/to/classifier_warmup.ckpt \
  manual_cotrain.global_steps=20000
bash scripts/experiments/world_model_training/train.sh \
  training.out_dir="${DVLA_DATA_ROOT}/outputs/wm_full_dataset_train/run"
```

`DVLA_DATA_ROOT` is independent of `DVLA_ROOT`; use a separate disk or shared
storage path when that is more convenient.

Shell entrypoints do not define training or evaluation defaults. They select complete
recipes directly under `configs/experiment/`; use Hydra `key=value` overrides for
changes. `configs/scripts/` is reserved for install, download, and preprocess.

## Config Fields

- `offline_warmup.data_dir`: collected reward-HDF5 replay directory.
- `offline_warmup.hidden_dir`: collected hidden-sidecar directory.
- `task.openvla_oft.hidden_token.*`: projected hidden-token dimensions and sidecar contract.
- `training.wm_warmup_steps`: world-model warmup update budget.
- `training.classifier_warmup_steps`: success-classifier warmup update budget.
- `dataloader.batch_size`: per-rank replay batch size.
- `online_rollout.sequence_length`: replay window length.
- `manual_cotrain.global_steps`: Ray online-cotrain update budget.

## Verify

```bash
python -m pytest tests/unit_tests -q
```

See [SETUP.md](SETUP.md) for the full workflow and
[docs/data_layout.md](docs/data_layout.md) for path conventions.
