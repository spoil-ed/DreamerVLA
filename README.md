# DreamerVLA

DreamerVLA is a single-machine multi-GPU research stack for LIBERO VLA training,
world-model learning, and Dreamer-style policy optimization.

```text
LIBERO HDF5
  -> no-op filtering and reward labels
  -> RynnVLA / OpenVLA-OFT action-hidden sidecar
  -> DINO-style chunk world model
  -> DreamerVLA actor-critic or WMPO outcome
  -> LIBERO rollout evaluation
```

## Quick Start

```bash
git clone <repo> && cd DreamerVLA
bash scripts/install_env.sh
conda activate dreamervla
export DVLA_DATA_ROOT=/path/to/dvla_data   # optional; defaults to ./data
bash scripts/download_assets.sh
TASK=libero_goal bash scripts/preprocess/prepare_libero_data.sh
CONFIG=vla_rynnvla_action_head bash scripts/train_vla.sh task=libero_goal
```

## Reproduction Route

1. Install the environment with `scripts/install_env.sh`.
2. Download weights and LIBERO assets with `scripts/download_assets.sh`.
3. Build filtered HDF5 files, reward labels, manifests, and sidecars with
   `scripts/preprocess/prepare_libero_data.sh`.
4. Train a VLA checkpoint with `scripts/train_vla.sh`.
5. Train a chunk world model with `scripts/train_wm.sh`.
6. Train DreamerVLA with `scripts/train_dreamervla.sh`.
7. Evaluate with `scripts/eval_libero_vla.sh`.

## Repository Layout

```text
dreamer_vla/        Python package: runners, models, datasets, algorithms, envs
configs/            Hydra routes and LIBERO task configs
scripts/            shell launchers for install, download, preprocess, train, eval
tests/              unit and smoke tests
third_party/        editable upstream dependencies
data/               default DVLA_DATA_ROOT
docs/               setup and data-layout reference
```

## Entry Points

| Stage | Command |
| --- | --- |
| Install | `bash scripts/install_env.sh` |
| Download | `bash scripts/download_assets.sh` |
| Preprocess | `TASK=libero_goal bash scripts/preprocess/prepare_libero_data.sh` |
| VLA SFT | `CONFIG=vla_rynnvla_action_head bash scripts/train_vla.sh task=libero_goal` |
| One-trajectory VLA | `CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_goal` |
| World model | `CONFIG=world_model_dinowm_chunk bash scripts/train_wm.sh task=libero_goal` |
| Classifier | `CONFIG=latent_classifier_libero_goal_chunk bash scripts/train_wm.sh` |
| DreamerVLA | `CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome bash scripts/train_dreamervla.sh` |
| Eval | `bash scripts/eval_libero_vla.sh eval.ckpt_path=<ckpt> eval.ckpt_kind=vla` |

Common overrides:

```bash
DVLA_DATA_ROOT=/path/to/dvla_data
CUDA_VISIBLE_DEVICES=0,1,2,3
NGPU=4
RUN_TAG=my_run
OUT_DIR="${DVLA_DATA_ROOT:-data}/outputs/<stage>/<run>"
```

## Config Fields

- `task.vla_ckpt_path`: RynnVLA init or SFT checkpoint directory.
- `task.pretokenize_config_path`: VLA SFT manifest.
- `task.hdf5_dir`: filtered LIBERO HDF5 directory.
- `task.hdf5_reward_dir`: reward-labeled HDF5 directory.
- `task.rynnvla_action_hidden_dir`: action-hidden sidecar.
- `init.world_model_state_ckpt`: DreamerVLA world-model checkpoint.
- `init.classifier_state_ckpt`: WMPO outcome classifier checkpoint.

## Verify

```bash
python -m pytest tests/unit_tests -q
```

See [SETUP.md](SETUP.md) for the full workflow and
[docs/data_layout.md](docs/data_layout.md) for path conventions.
