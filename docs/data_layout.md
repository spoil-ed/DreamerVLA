# Data Layout

`DVLA_ROOT` is the source checkout. `DVLA_DATA_ROOT` is the runtime asset root
for checkpoints, datasets, processed data, logs, and outputs.
`DVLA_DATA_ROOT does not need to live inside DVLA_ROOT`. If it is unset, release
scripts use the relative path `data` after entering the repository root.

The tracked repository does not include `data/` or `third_party/`; both are
local state. `scripts/install_env.sh` creates/populates `third_party/`,
`scripts/download_assets.sh` downloads raw datasets and checkpoints, and
`scripts/preprocess_libero.sh` generates `processed_data/*` from those raw HDF5
files. A directory name alone is not a completed stage; each stage is complete
only when the files consumed by the next stage exist.

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
bash scripts/download_assets.sh only=[10_rynnvla] env.LIBERO_SUITES='"libero_goal libero_object"'
bash scripts/download_assets.sh only=[10_rynnvla] env.LIBERO_SUITES=libero_goal \
  env.DOWNLOAD_RYNNVLA_LUMINA=false env.DOWNLOAD_ACTION_WM=false
bash scripts/download_assets.sh download.openvla_oft=true only=[20_openvla_oft] \
  env.OPENVLA_OFT_REPOS=owner/repo:libero_goal_hdf5_latest_6650
bash scripts/download_assets.sh download.openvla_one_traj=true only=[30_openvla_oft_one_trajectory]
```

`scripts/download/10_rynnvla.sh` creates RynnVLA Chameleon assets,
`checkpoints/models--Alpha-VLLM--Lumina-mGPT-7B-768/`,
`checkpoints/VLA_model_256/<suite>/`, and, by default,
`checkpoints/Action_World_model_512/<suite>/`. The script is split into
weight-level steps so each part can be disabled with environment flags.
`scripts/download/20_openvla_oft.sh` writes HDF5 SFT checkpoints under
`checkpoints/OpenVLA-OFT/<run>/`. `scripts/download/30_openvla_oft_one_trajectory.sh`
writes one-trajectory checkpoints under `checkpoints/Openvla-oft-SFT-traj1/<name>/`.

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
bash scripts/download_assets.sh download.rynnvla=false download.libero=true
bash scripts/download_assets.sh download.rynnvla=false download.libero=true \
  env.LIBERO_SUITES='"libero_goal libero_10"'
```

The LIBERO downloader writes into `${DVLA_DATA_ROOT}/datasets/libero` by default.
The workflow step is `scripts/download/40_libero_dataset.sh`.
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
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true \
  env.EXTRACT_CALVIN=true
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true \
  env.HF_ENDPOINT=https://hf-mirror.com env.CALVIN_DOWNLOAD_METHOD=hf_shards
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true \
  env.HF_ENDPOINT=https://hf-mirror.com env.CALVIN_DOWNLOAD_METHOD=hf_subsets
bash scripts/download_assets.sh download.rynnvla=false download.libero=false download.calvin=true \
  env.CALVIN_DOWNLOAD_METHOD=opendatalab
```

`CALVIN_DOWNLOAD_METHOD=official` downloads Freiburg zip files directly.
`hf_shards` uses the Hugging Face dataset `VyoJ/calvin-ABCD-D-shards`, which
stores `task_ABCD_D` as 30 GB multi-part zip shards under
`datasets/calvin/task_ABCD_D_shards/`. `hf_subsets` uses
`VyoJ/calvin-ABCD-D-subsets`, which stores complete structured subset zips under
`datasets/calvin/task_ABCD_D_subsets/`. `opendatalab` uses
`OpenDataLab/CALVIN` through the `openxlab` CLI and writes under
`datasets/calvin/opendatalab/`.

## Processed Data

Generated datasets resolve under `${DVLA_DATA_ROOT}/processed_data`.

`bash scripts/preprocess/prepare_libero_data.sh task=<suite>` writes:

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

Preprocessing follows the same orchestrator-and-step style as install and
download scripts. Run the full path with:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal
```

Run or reproduce one step by calling the numbered child script directly:

```bash
TASK=libero_goal GPUS=0 PRETOKENIZE_PROCS=8 bash scripts/preprocess/20_pretokenize_dataset.sh
```

The numbered steps are:

```text
10_hdf5_reward -> 20_pretokenize_dataset -> 30_action_hidden -> 40_validate
```

If `DVLA_DATA_ROOT` changes after preprocessing, regenerate stage 4-5 or update
absolute prefixes inside generated YAML / JSON manifests.

Validate the generated tree with:

```bash
bash scripts/preprocess/validate_libero_data.sh --suites libero_goal
bash scripts/preprocess/validate_libero_data.sh \
  --suites libero_goal libero_object libero_spatial libero_10
```

The validator checks that HDF5/reward files exist, conv JSONs are non-empty,
token pkl counts match conv counts, record files match token counts, the
concatenated manifest matches records, and generated configs point at existing
JSON files. Add `--check-action-hidden` when validating the legacy action-hidden
sidecar after stage 5.

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
Create processed data with `bash scripts/preprocess/prepare_libero_data.sh task=<suite>`.
