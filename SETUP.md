# Setup

`DVLA_ROOT` points at the source checkout. `DVLA_DATA_ROOT` points at runtime
assets and may live on any disk with enough space:

```bash
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
cd "${DVLA_ROOT}"
```

See [docs/data_layout.md](docs/data_layout.md) for the complete runtime tree.

## Install

Use the resumable one-command installer:

```bash
bash scripts/install_env.sh
conda activate dreamervla
```

The installer runs `scripts/install/*.sh` in order and records completed steps
under `${DVLA_DATA_ROOT}/install_state/`. Re-run the same command after a
failed step; completed steps are skipped. To force one step:

```bash
bash scripts/install_env.sh only=[20_torch] force=true
```

Useful single-step debugging commands:

```bash
bash scripts/install/00_apt_tools.sh
bash scripts/install/10_conda_env.sh
bash scripts/install/20_torch.sh
bash scripts/install/30_python_deps.sh
bash scripts/install/40_third_party.sh
bash scripts/install/50_special_packages.sh
bash scripts/install/60_verify.sh
```

## Download

Download the OpenVLA-OFT one-trajectory checkpoints and LIBERO data used by the
current cotrain path:

```bash
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]
bash scripts/download_assets.sh only=[40_libero_dataset] env.LIBERO_SUITES=libero_goal
```

For the full four-suite matrix:

```bash
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]
bash scripts/download_assets.sh only=[40_libero_dataset] \
  env.LIBERO_SUITES='"libero_goal libero_object libero_spatial libero_10"'
```

CALVIN is optional and is not needed for LIBERO cotrain:

```bash
bash scripts/download_assets.sh download.libero=false download.calvin=true
bash scripts/download_assets.sh download.libero=false download.calvin=true \
  env.HF_ENDPOINT=https://hf-mirror.com env.CALVIN_DOWNLOAD_METHOD=hf_shards
bash scripts/download_assets.sh download.libero=false download.calvin=true \
  env.HF_ENDPOINT=https://hf-mirror.com env.CALVIN_DOWNLOAD_METHOD=hf_subsets
bash scripts/download_assets.sh download.libero=false download.calvin=true \
  env.CALVIN_DOWNLOAD_METHOD=opendatalab
```

## Canonical LIBERO Preprocessing

The offline route has one observation contract:
`hidden_token [T,256,4096]`. For one suite, run:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal gpus=0 ngpu=1
```

The workflow executes these stages in order:

1. `10_hdf5_reward`: replay LIBERO data, mark no-ops with `keep_noops=true`,
   filter the marked files with `filter_noops=true` through
   `dreamervla.preprocess.filter_marked_libero_hdf5`, then add rewards. Its
   intermediate paths are
   `${DVLA_DATA_ROOT}/processed_data/${TASK}/marked_t_256` and
   `${DVLA_DATA_ROOT}/processed_data/${TASK}/no_noops_t_256`.
2. `35_oft_hidden_token`: write the exact projected OpenVLA hidden-token
   sidecar.
3. `40_validate`: verify metadata plus every `obs_embedding` dataset before the
   artifact can be used for training.

Run all four suites in one command:

```bash
bash scripts/preprocess_libero.sh
bash scripts/preprocess_libero.sh tasks='"libero_goal libero_object"'
```

The default suite set is
`libero_goal libero_object libero_spatial libero_10`.

## Cold-Start Cotrain

The main entrypoint is the launcher that collects rollouts, seeds replay,
warms up the world model and classifier, then optionally continues online:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_ray.sh \
  task=goal ngpu=8 profile=multi_gpu render_backend=osmesa
```

The sync launcher uses the same warmup code path without Ray collection:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/e2e_coldstart_warmup_cotrain_noray.sh \
  task=goal ngpu=8 profile=multi_gpu render_backend=osmesa
```

Common task shorthands are `goal`, `object`, `spatial`, and `10`. The launcher
config is `configs/scripts/coldstart_warmup_cotrain.yaml`; training recipes live
under `configs/experiment/`.

## World-Model Full-Replay Warmup

For offline WM analysis on the full collected replay:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  GPU_COUNT=8 \
  DVLA_DATA_ROOT=/path/to/data \
  bash scripts/experiments/world_model_training/train.sh
```

The shell file intentionally exposes one command. Batch size, LR, sequence
length, rollout depth, checkpoint cadence, and profiling knobs are owned by
Hydra config and the script-level defaults in
`scripts/experiments/world_model_training/train.sh`.

## Evaluation

Evaluate a VLA or Dreamer checkpoint with:

```bash
bash scripts/eval_libero_vla.sh \
  gpus=0 \
  eval.ckpt_path=/path/to/checkpoint \
  eval.ckpt_kind=auto \
  eval.task_suite_name=libero_goal
```

## Verification

Run the install check before long jobs:

```bash
bash scripts/install/60_verify.sh
```

Run focused unit checks:

```bash
python -m pytest tests/unit_tests -q
ruff check dreamervla tests
```

`scripts/install/30_python_deps.sh` installs the `pyproject.toml` `dev`
dependency group by default so `pytest`, `ruff`, and `pre-commit` are available
inside the conda environment. Use
`bash scripts/install_env.sh env.INSTALL_DEV_TOOLS=false` to skip developer
tools.
