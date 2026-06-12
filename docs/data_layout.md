# Data Layout

`DVLA_ROOT` is the source checkout. `DVLA_DATA_ROOT` is the runtime asset root
for checkpoints, datasets, processed data, logs, and outputs.
`DVLA_DATA_ROOT does not need to live inside DVLA_ROOT`. If it is unset, release
scripts use the relative path `data` after entering the repository root.

## Tree

```text
${DVLA_DATA_ROOT}/
|-- checkpoints/
|   |-- VLA_model_256/<suite>/
|   |-- Action_World_model_512/<suite>/
|   |-- chameleon/tokenizer/
|   |-- models--Alpha-VLLM--Lumina-mGPT-7B-768/
|   |-- OpenVLA-OFT/<run>/
|   `-- Openvla-oft-SFT-traj1/<name>/
|-- datasets/
|   |-- libero/<suite>/
|   `-- calvin/
|-- processed_data/
|-- configs/<suite>/
|-- outputs/
|-- logs/
|-- wheels/
|-- tmp_ckpts/
`-- .libero/config.yaml
```

## Checkpoints

Checkpoint assets resolve under `${DVLA_DATA_ROOT}/checkpoints`.

| Path | Content |
| --- | --- |
| `checkpoints/VLA_model_256/<suite>/` | RynnVLA-002 init / SFT model files |
| `checkpoints/Action_World_model_512/<suite>/` | RynnVLA action world-model init |
| `checkpoints/chameleon/tokenizer/` | Chameleon tokenizer and VQGAN files |
| `checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768/` | Lumina tokenizer and backbone files |
| `checkpoints/OpenVLA-OFT/<run>/` | OpenVLA-OFT model and component checkpoints |
| `checkpoints/Openvla-oft-SFT-traj1/<name>/` | one-trajectory OpenVLA-OFT checkpoints |

Download all default Hugging Face weights with:

```bash
bash scripts/download_assets.sh
```

Download model families one at a time with:

```bash
bash scripts/download/10_worldvla.sh
bash scripts/download/20_lumina.sh
LIBERO_SUITES="libero_goal libero_object" bash scripts/download/30_rynnvla.sh
DOWNLOAD_ACTION_WM=0 LIBERO_SUITES=libero_goal bash scripts/download/30_rynnvla.sh
```

`scripts/download/10_worldvla.sh` creates `checkpoints/chameleon/` and related
WorldVLA files. `scripts/download/20_lumina.sh` creates
`checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768/`.
`scripts/download/30_rynnvla.sh` creates `checkpoints/VLA_model_256/<suite>/`
and, by default, `checkpoints/Action_World_model_512/<suite>/`.

## Datasets

| Path | Content |
| --- | --- |
| `datasets/libero/<suite>/*.hdf5` | raw LIBERO demonstrations |
| `datasets/calvin/` | CALVIN zip files and extracted data |

Raw LIBERO suites resolve to `${DVLA_DATA_ROOT}/datasets/libero/<suite>`.
The standard suite directories are:

```text
${DVLA_DATA_ROOT}/datasets/libero/libero_goal/
${DVLA_DATA_ROOT}/datasets/libero/libero_object/
${DVLA_DATA_ROOT}/datasets/libero/libero_spatial/
${DVLA_DATA_ROOT}/datasets/libero/libero_10/
```

Download LIBERO data into the canonical tree with:

```bash
bash scripts/download/40_libero_dataset.sh
LIBERO_SUITES="libero_goal libero_10" bash scripts/download/40_libero_dataset.sh
```

The LIBERO downloader writes into `${DVLA_DATA_ROOT}/datasets/libero` by default.
Do not put suites directly under `${DVLA_DATA_ROOT}/datasets`; task configs and
launch scripts expect the `datasets/libero/<suite>/` layer.

CALVIN data resolves under:

```text
${DVLA_DATA_ROOT}/datasets/calvin/
${DVLA_DATA_ROOT}/datasets/calvin/task_ABCD_D.zip
${DVLA_DATA_ROOT}/datasets/calvin/task_ABCD_D/
```

Download CALVIN data with:

```bash
bash scripts/download/50_calvin_dataset.sh
EXTRACT_CALVIN=1 bash scripts/download/50_calvin_dataset.sh
```

## Processed Data

Generated datasets resolve under `${DVLA_DATA_ROOT}/processed_data`.

`TASK=<suite> bash scripts/preprocess/prepare_libero_data.sh` writes:

```text
processed_data/<suite>_marked_t_256/
processed_data/<suite>_no_noops_t_256/
processed_data/<suite>_no_noops_t_256_pi06_remaining_reward/
processed_data/<suite>_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2/
processed_data/<suite>_image_state_action_t_256/
processed_data/convs/
processed_data/tokens/
processed_data/concate_tokens/
processed_data/<suite>_metainfo.json
configs/<suite>/his_1_third_view_wrist_w_state_1_256_pretokenize*.yaml
```

If `DVLA_DATA_ROOT` changes after preprocessing, regenerate stage 4-5 or update
absolute prefixes inside generated YAML / JSON manifests.

## Outputs

Training runs write under `outputs/<route>/<run>/checkpoints/`. Evaluation
runs write under `outputs/eval/`.

## LIBERO Config

Launch scripts write `${DVLA_DATA_ROOT}/.libero/config.yaml` with raw demos at
`${DVLA_DATA_ROOT}/datasets/libero`.

## Move Data

```bash
rsync -a old:/path/to/dvla_data/ new:/path/to/dvla_data/
export DVLA_DATA_ROOT=/path/to/dvla_data
```

Create assets with `bash scripts/download_assets.sh`.
Create processed data with `TASK=<suite> bash scripts/preprocess/prepare_libero_data.sh`.
