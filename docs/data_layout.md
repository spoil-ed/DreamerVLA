# Data Layout

`DVLA_ROOT` is the source checkout. `DVLA_DATA_ROOT` is the runtime asset root
for checkpoints, datasets, collected rollouts, processed data, logs, and
outputs. If unset, release scripts use `data` under the repository root.
DVLA_DATA_ROOT does not need to live inside DVLA_ROOT.

The tracked repository does not include `data/` or `third_party/`; both are
local state.

Canonical roots are `${DVLA_DATA_ROOT}/checkpoints`,
`${DVLA_DATA_ROOT}/datasets`, and `${DVLA_DATA_ROOT}/processed_data`.

## Tree

```text
${DVLA_DATA_ROOT}/
|-- checkpoints/
|   |-- OpenVLA-OFT/<run>/
|   `-- Openvla-oft-SFT-traj1/<name>/
|-- datasets/
|   |-- libero/<suite>/
|   `-- calvin/
|-- collected_rollouts/
|   `-- <suite>/{reward,hidden}/
|-- processed_data/
|-- configs/
|-- outputs/
|-- logs/
|-- wheels/
|-- tmp_ckpts/
`-- .libero/config.yaml
```

## Checkpoints

| Path | Content |
| --- | --- |
| `checkpoints/OpenVLA-OFT/<run>/` | OpenVLA-OFT model and component checkpoints |
| `checkpoints/Openvla-oft-SFT-traj1/<name>/` | one-trajectory OpenVLA-OFT checkpoints |

Download current model assets with:

```bash
bash scripts/download_assets.sh download.openvla_one_traj=true only=[10_openvla_oft_one_trajectory]
```

The registered checkpoint steps are `scripts/download/00_openvla_oft.sh` and
`scripts/download/10_openvla_oft_one_trajectory.sh`.

## Datasets

Raw LIBERO suites resolve to `${DVLA_DATA_ROOT}/datasets/libero/<suite>`:

```text
libero_goal/
libero_object/
libero_spatial/
libero_10/
```

Download LIBERO data into the canonical tree with:

```bash
bash scripts/download_assets.sh only=[20_libero_dataset] env.LIBERO_SUITES=libero_goal
bash scripts/download_assets.sh only=[20_libero_dataset] \
  env.LIBERO_SUITES='"libero_goal libero_object libero_spatial libero_10"'
```

The LIBERO download implementation is `scripts/download/20_libero_dataset.sh`.

CALVIN data is optional and resolves under `${DVLA_DATA_ROOT}/datasets/calvin/`:

```bash
bash scripts/download_assets.sh download.libero=false download.calvin=true
```

## Collected Rollouts

Cold-start launchers write generated replay under:

```text
collected_rollouts/<suite>/reward/
collected_rollouts/<suite>/hidden/
```

These directories are passed to `offline_warmup.data_dir` and
`offline_warmup.hidden_dir` for WM/classifier warmup.

## Processed Data

Offline preprocessing writes under `${DVLA_DATA_ROOT}/processed_data/<artifact>`:

```text
processed_data/<artifact>/marked_t_256/
processed_data/<artifact>/no_noops_t_256/
processed_data/<artifact>/no_noops_t_256_remaining_reward/
processed_data/<artifact>/no_noops_t_256_oft_hidden_token_vla_policy_h1/
processed_data/<artifact>/metainfo.json
```

The hidden-token directory contains `preprocess_config.json` and HDF5
`obs_embedding` datasets with the fixed shape `[T,256,4096]`. No alternate
56-token observation sidecar is accepted.

Run preprocessing for one suite with:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal
```

Run the same canonical workflow for all four suites with:

```bash
bash scripts/preprocess_libero.sh
```

## Outputs

Training runs default to
`${RUN_ROOT:-${DVLA_DATA_ROOT}/outputs}/${run.name}/${run.timestamp}`. Every invocation
owns one run root with sibling `checkpoints/`, `logs/`, `tensorboard/`, `wandb/`,
`video/`, `diagnostics/`, and `.hydra/` directories. Collection, warmup, cotrain, and
other training routes use their experiment names as `run.name`; resuming keeps the
original root. `checkpoints/` is flat and contains `latest.ckpt` plus configured
`epoch=<epoch>-<metric>=<value>.ckpt` top-k files. Explicit HF export uses the sibling
`checkpoint_hf/`.

Evaluation uses `${RUN_ROOT:-${DVLA_DATA_ROOT}/outputs}/eval/${eval.task_suite_name}`.
That task directory is the eval run root; it has no timestamp layer and does not own
training checkpoints.

## Move Data

```bash
rsync -a old:dvla_data/ new:dvla_data/
export DVLA_DATA_ROOT=/new/path/dvla_data
```
