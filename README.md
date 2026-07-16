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

For the pinned 8xH100 Docker reproduction, see
[Docker Reproduction](docs/docker_reproduction.md):

```bash
docker pull spoil/dreamervla:cu124-h100-v1
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  -v "$PWD/dreamervla-data:/data" spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/01_prepare_assets.sh
docker run --rm --gpus all --ipc=host --network=host --shm-size=100g \
  -v "$PWD/dreamervla-data:/data" spoil/dreamervla:cu124-h100-v1 \
  bash scripts/reproduce/02_train_dreamer.sh
```

The image contains the source and pinned third-party environment; weights, datasets,
and outputs stay in the mounted `/data` directory.

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
  --wm_ckpt /path/to/wm-run/checkpoints/latest.ckpt \
  --cls_ckpt /path/to/classifier-run/checkpoints/latest.ckpt

bash scripts/experiments/cotrain/eval.sh \
  eval.ckpt_path=/path/to/cotrain-run
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

## Reproduction Route

1. Install the environment with `scripts/install_env.sh`.
2. Download OpenVLA-OFT one-trajectory checkpoints and LIBERO data with
   `scripts/download_assets.sh`.
3. Train with `scripts/experiments/cotrain/train.sh --config
   openvla_libero`, explicitly supplying the warmup WM and
   classifier checkpoints.
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
| Docker asset preparation | `bash scripts/reproduce/01_prepare_assets.sh` |
| Docker WM/CLS/Dreamer reproduction | `bash scripts/reproduce/02_train_dreamer.sh` |
| Install | `bash scripts/install_env.sh` |
| Download OpenVLA-OFT one-trajectory | `bash scripts/download_assets.sh download.openvla_one_traj=true only=[10_openvla_oft_one_trajectory]` |
| Download LIBERO | `bash scripts/download_assets.sh only=[20_libero_dataset] env.LIBERO_SUITES=libero_goal` |
| Full WM/CLS cotrain | `bash scripts/experiments/cotrain/train.sh --config openvla_onetraj_libero_cotrain --wm_ckpt <wm-ckpt> --cls_ckpt <cls-ckpt>` |
| Frozen WM/CLS imagined RL | `bash scripts/experiments/cotrain/train.sh --config openvla_libero --wm_ckpt <wm-ckpt> --cls_ckpt <cls-ckpt>` |
| Full-dataset WM warmup | `bash scripts/experiments/world_model_training/train.sh` |
| Cotrain eval | `bash scripts/experiments/cotrain/eval.sh eval.ckpt_path=<ckpt>` |

Common overrides:

```bash
DVLA_DATA_ROOT=data
bash scripts/experiments/cotrain/train.sh \
  --config openvla_libero \
  --wm_ckpt /path/to/wm-run/checkpoints/latest.ckpt \
  --cls_ckpt /path/to/classifier-run/checkpoints/latest.ckpt \
  manual_cotrain.global_steps=20000
bash scripts/experiments/world_model_training/train.sh \
  training.out_dir="${DVLA_DATA_ROOT}/outputs/wm_full_dataset_train/run"
```

`DVLA_DATA_ROOT` is independent of `DVLA_ROOT`; use a separate disk or shared
storage path when that is more convenient.

Training artifacts live at `outputs/<experiment>/<timestamp>/`. Its flat
`checkpoints/` contains `latest.ckpt` and optional
`epoch=<epoch>-<metric>=<value>.ckpt` top-k files. Explicit HF export writes the
sibling `checkpoint_hf/`. Evaluation writes to `outputs/eval/<task-suite>/` and
accepts a concrete checkpoint, `checkpoints/`, or a training run root.

Shell entrypoints do not define training or evaluation defaults. They select complete
recipes directly under `configs/experiment/`; use Hydra `key=value` overrides for
changes. `configs/scripts/` is reserved for install, download, and preprocess.
Use `profile=debug` or `profile=smoke` for declared reduced budgets; Runner code
never rewrites production budgets at runtime.

On a networked CPU host that can read the GPU host's shared run directory, stream an
active offline W&B run with the official W&B CLI:

```bash
wandb login
wandb beta sync --live /path/to/run_root/wandb
```

Start the command after the GPU process has created `wandb/offline-run-*`. Use W&B
0.24.1 or newer; see the experiment tutorial for crash recovery and legacy layouts.

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
