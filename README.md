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
export DVLA_DATA_ROOT=/path/to/dvla_data
bash scripts/install_env.sh
conda activate dreamervla
bash scripts/download_assets.sh
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal
bash scripts/train_vla.sh experiment=vla_rynnvla_action_head task=libero_goal
```

## Reproduction Route

1. Install the environment with `scripts/install_env.sh` or a single
   `scripts/install/*.sh` step.
2. Download weights and LIBERO assets with `scripts/download_assets.sh` or a
   single `scripts/download/*.sh` step.
3. Build filtered HDF5 files, reward labels, manifests, and sidecars with
   `scripts/preprocess/prepare_libero_data.sh`.
4. Train a VLA checkpoint with `scripts/train_vla.sh`.
5. Train a chunk world model with `scripts/train_wm.sh`.
6. Train DreamerVLA with `scripts/train_dreamervla.sh`.
7. Evaluate with `scripts/eval_libero_vla.sh`.

## Repository Layout

```text
dreamervla/        Python package: runners, models, datasets, algorithms, envs
configs/            Hydra routes and LIBERO task configs
scripts/            shell launchers for install, download, preprocess, train, eval
tests/              unit and smoke tests
third_party/        editable upstream dependencies
data/               relative default runtime data root
docs/               setup and data-layout reference
```

## Entry Points

| Stage | Command |
| --- | --- |
| Install | `bash scripts/install_env.sh` |
| Download all | `bash scripts/download_assets.sh` |
| Download RynnVLA weights | `bash scripts/download_assets.sh only=[10_rynnvla] env.LIBERO_SUITES=libero_goal` |
| Download OpenVLA-OFT | `bash scripts/download_assets.sh download.openvla_oft=true only=[20_openvla_oft] env.OPENVLA_OFT_REPOS=owner/repo:libero_goal_hdf5_latest_6650` |
| Download OpenVLA-OFT one-trajectory | `bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]` |
| Download LIBERO | `bash scripts/download_assets.sh download.rynnvla=false download.libero=true env.LIBERO_SUITES=libero_goal` |
| Download CALVIN | `bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true` |
| Preprocess | `bash scripts/preprocess/prepare_libero_data.sh task=libero_goal` |
| VLA SFT | `bash scripts/train_vla.sh experiment=vla_rynnvla_action_head task=libero_goal` |
| One-trajectory VLA | `bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=libero_goal` |
| World model | `bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal` |
| Classifier | `bash scripts/train_wm.sh experiment=latent_classifier_libero_goal_chunk` |
| DreamerVLA | `bash scripts/train_dreamervla.sh experiment=dreamervla_rynn_dino_wm_wmpo_outcome task=libero_goal` |
| Eval | `bash scripts/eval_libero_vla.sh gpus=0 eval.ckpt_path=<ckpt> eval.ckpt_kind=vla` |

Common overrides:

```bash
DVLA_DATA_ROOT=/path/to/dvla_data
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal \
  gpus=0,1,2,3 ngpu=4 batch_size=16 run_tag=my_run
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal gpus=0 ngpu=1 num_procs=8
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal \
  training.out_dir="${DVLA_DATA_ROOT:-data}/outputs/<stage>/<run>"
```

`DVLA_DATA_ROOT` is independent of `DVLA_ROOT`; use a separate disk or shared
storage path when that is more convenient.

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
