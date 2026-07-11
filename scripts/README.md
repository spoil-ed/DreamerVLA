# Script Registry

`scripts/` contains shell launchers only. Python implementation code lives under
the `dreamervla` package and is launched with `python -m`.
Runtime paths are documented in [docs/data_layout.md](../docs/data_layout.md).

## Main Path

| Script | Purpose |
| --- | --- |
| `install_env.sh` | Hydra wrapper for resumable install steps |
| `download_assets.sh` | Hydra wrapper for selected asset downloads |
| `preprocess_libero.sh` | Hydra wrapper for LIBERO preprocessing |
| `preprocess/prepare_libero_data.sh` | One-suite LIBERO preprocessing workflow |
| `collect_parallel.sh` | Data-parallel cold-start collection helper |
| `e2e_coldstart_warmup_cotrain_ray.sh` | Ray collection followed by WM/classifier warmup and optional cotrain |
| `e2e_coldstart_warmup_cotrain_noray.sh` | Sync collection followed by WM/classifier warmup and optional cotrain |
| `e2e_manual_cotrain_async.sh` | Manual async OpenVLA-OFT cotrain resume launcher |
| `e2e_frozen_model_pre_mainline.sh` | Pre-mainline official-data WM/CLS, frozen policy RL, and matched real-eval proof |
| `experiments/single_trajectory_overfit/train.sh` | Single-trajectory overfit training diagnostic |
| `experiments/single_trajectory_overfit/eval.sh` | Single-trajectory overfit eval summary |
| `experiments/classifier_training/train.sh` | Full classifier training recipe |
| `experiments/classifier_training/eval.sh` | Full classifier eval summary |
| `experiments/world_model_training/train.sh` | Full-replay world-model training recipe |
| `experiments/world_model_training/eval.sh` | Full-replay world-model eval diagnostic |
| `eval_libero_vla.sh` | LIBERO rollout eval |
| `eval/launch_openvla_oft_official_libero_eval.sh` | Official OpenVLA-OFT LIBERO eval wrapper |
| `train_dreamervla.sh` | Hydra training entrypoint |
| `run_wandb_relay_sync.sh` | CPU-side W&B offline relay helper |
| `start_ray.sh` | Start a local single-node Ray head |
| `check_ray.sh` | Inspect the active Ray cluster |

## Install Steps

| Script | Purpose |
| --- | --- |
| `install/00_apt_tools.sh` | System packages |
| `install/10_conda_env.sh` | Conda environment |
| `install/20_torch.sh` | PyTorch CUDA wheel set |
| `install/30_python_deps.sh` | Python runtime and dev dependencies |
| `install/40_third_party.sh` | LIBERO, robosuite stack, OpenSora, and OpenVLA-OFT packages |
| `install/50_special_packages.sh` | flash-attn, egl_probe, and optional GPU extensions |
| `install/60_verify.sh` | Import and CUDA visibility check |

## Download Steps

| Script | Purpose |
| --- | --- |
| `download/20_openvla_oft.sh` | Download user-provided OpenVLA-OFT checkpoints |
| `download/30_openvla_oft_one_trajectory.sh` | Download OpenVLA-OFT one-trajectory checkpoints |
| `download/40_libero_dataset.sh` | Download LIBERO suites |
| `download/50_calvin_dataset.sh` | Download optional CALVIN tasks |

Download steps are intentionally serial and numbered. To add a new asset family,
create `download/NN_name.sh`, write outputs only under `${DVLA_DATA_ROOT}`, then
append the script to `configs/scripts/download.yaml`.

## Preprocessing

| Script | Purpose |
| --- | --- |
| `preprocess_libero.sh` | Top-level wrapper around one-suite preprocessing |
| `preprocess/prepare_libero_data.sh` | One-suite preprocessing workflow |
| `preprocess/process_all_libero_data.sh` | Multi-suite OpenVLA input-token preprocessing wrapper |
| `preprocess/10_hdf5_reward.sh` | Write LIBERO config, mark/filter HDF5 files, and add reward labels |
| `preprocess/35_oft_input_tokens.sh` | Extract canonical OpenVLA-OFT input-token sidecars `[T,256,4096]` |
| `preprocess/40_validate.sh` | Validate the exact input-token metadata and every HDF5 demo |
| `preprocess/validate_libero_data.sh` | Validate canonical preprocessing outputs for selected suites |

Common launcher flags:

```bash
bash scripts/install_env.sh only=[20_torch] force=true
bash scripts/download_assets.sh only=[40_libero_dataset] env.LIBERO_SUITES=libero_goal
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal gpus=0 ngpu=1 num_procs=8
```

Single-suite preprocessing:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal \
  gpus=0 ngpu=1 num_procs=8
```

The workflow has exactly three stages: reward HDF5 preparation,
`input_token_embedding` extraction, and strict sidecar validation. The persisted
observation shape is always `[T,256,4096]`.

Cold-start warmup launchers run a two-stage flow: collect generated rollouts,
then point `offline_warmup.data_dir` and `offline_warmup.hidden_dir` at the
collected output for WM/classifier warmup. The Ray variant uses
`experiment=collect_rollouts_ray`; the sync variant uses
`experiment=collect_rollouts_onetraj`.

Dry-run the launcher command assembly with:

```bash
bash scripts/e2e_coldstart_warmup_cotrain_ray.sh dry_run=true
bash scripts/e2e_coldstart_warmup_cotrain_noray.sh task=spatial dry_run=true
bash scripts/e2e_frozen_model_pre_mainline.sh task=goal dry_run=true
```

Check collection completeness before a long run:

```bash
python -m dreamervla.diagnostics.check_collection_completeness \
  --reward-dir data/collected_rollouts/libero_goal/reward \
  --hidden-dir data/collected_rollouts/libero_goal/hidden \
  --target-episodes 500 --num-tasks 10 --json
```

Grouped training defaults to `logger=tensorboard_wandb`, writing TensorBoard
events under `${training.out_dir}/log/tensorboard` and W&B files under
`${training.out_dir}/log/wandb`.
