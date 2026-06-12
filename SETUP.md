# DreamerVLA Setup

All commands run from the repository root. `DVLA_ROOT` is the checkout.
`DVLA_DATA_ROOT` is the runtime asset root and can be any writable path. If
unset, release scripts use relative `data`. See `docs/data_layout.md`.

A fresh checkout contains source code, configs, scripts, tests, and docs only.
`third_party/` and everything under `data/` are local install/download/generated
state. Build the tree in order: install third-party code, download raw
datasets/checkpoints, then generate `processed_data/*`.

```bash
cd /path/to/DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT=/path/to/dvla_data
```

## 1. Install

One-command install:

```bash
bash scripts/install_env.sh
conda activate dreamervla
```

The installer runs these resumable steps and writes
`${DVLA_DATA_ROOT:-data}/install_state/*.done`:

```text
scripts/install/00_apt_tools.sh
scripts/install/10_conda_env.sh
scripts/install/20_torch.sh
scripts/install/30_python_deps.sh
scripts/install/40_third_party.sh
scripts/install/50_special_packages.sh
scripts/install/60_verify.sh
```

Run a single step when needed:

```bash
bash scripts/install/00_apt_tools.sh
bash scripts/install/10_conda_env.sh
bash scripts/install/20_torch.sh
bash scripts/install/30_python_deps.sh
bash scripts/install/40_third_party.sh
bash scripts/install/50_special_packages.sh
bash scripts/install/60_verify.sh
```

Install step map:

| Step | Scope |
| --- | --- |
| `00_apt_tools.sh` | apt packages such as git-lfs, ffmpeg, OpenGL / OSMesa, cmake, ninja |
| `10_conda_env.sh` | conda environment and Python version |
| `20_torch.sh` | PyTorch / torchvision / torchaudio CUDA wheels |
| `30_python_deps.sh` | DreamerVLA editable install and curated pip requirements |
| `40_third_party.sh` | LIBERO, robosuite-family packages, OpenSora, OpenVLA-OFT helpers |
| `50_special_packages.sh` | flash-attn, egl_probe, optional apex / TensorNVMe |
| `60_verify.sh` | import and CUDA visibility checks |

## 2. Download Assets

One-command download:

```bash
bash scripts/download_assets.sh
```

Single asset steps:

```bash
LIBERO_SUITES="libero_goal libero_object libero_spatial libero_10" bash scripts/download/10_rynnvla.sh
OPENVLA_OFT_REPOS="owner/repo:libero_goal_hdf5_latest_6650" bash scripts/download/20_openvla_oft.sh
bash scripts/download/30_openvla_oft_one_trajectory.sh
LIBERO_SUITES="libero_goal libero_object libero_spatial libero_10" bash scripts/download/40_libero_dataset.sh
bash scripts/download/50_calvin_dataset.sh
```

Useful variants:

```bash
LIBERO_SUITES="libero_goal libero_object" bash scripts/download_assets.sh
DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=1 LIBERO_SUITES=libero_spatial bash scripts/download_assets.sh
DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=0 DOWNLOAD_CALVIN=1 CALVIN_TASKS=task_ABCD_D bash scripts/download_assets.sh
DOWNLOAD_ONLY=10_rynnvla bash scripts/download_assets.sh
DOWNLOAD_OPENVLA_ONE_TRAJ=1 DOWNLOAD_ONLY=30_openvla_oft_one_trajectory bash scripts/download_assets.sh
```

CALVIN domestic / mirror-friendly options:

```bash
# Hugging Face mirror, sharded 30 GB multi-part zip files for task_ABCD_D
HF_ENDPOINT=https://hf-mirror.com CALVIN_DOWNLOAD_METHOD=hf_shards bash scripts/download/50_calvin_dataset.sh

# Hugging Face mirror, structured subset zips for task_ABCD_D
HF_ENDPOINT=https://hf-mirror.com CALVIN_DOWNLOAD_METHOD=hf_subsets bash scripts/download/50_calvin_dataset.sh

# OpenDataLab domestic platform; requires login through openxlab
pip install -U openxlab
openxlab login
CALVIN_DOWNLOAD_METHOD=opendatalab bash scripts/download/50_calvin_dataset.sh
```

OpenVLA-OFT one-trajectory checkpoints support both Hugging Face download
methods:

```bash
# Method 1: git clone with Git LFS
OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=git bash scripts/download/30_openvla_oft_one_trajectory.sh

# Method 2: huggingface-hub; set HF_ENDPOINT=https://hf-mirror.com if needed
OPENVLA_ONE_TRAJ_DOWNLOAD_METHOD=hf bash scripts/download/30_openvla_oft_one_trajectory.sh
```

Canonical dataset roots:

```text
${DVLA_DATA_ROOT}/datasets/libero/libero_goal/
${DVLA_DATA_ROOT}/datasets/libero/libero_object/
${DVLA_DATA_ROOT}/datasets/libero/libero_spatial/
${DVLA_DATA_ROOT}/datasets/libero/libero_10/
${DVLA_DATA_ROOT}/datasets/calvin/
```

## 3. Preprocess LIBERO

Preprocess all four standard LIBERO suites in one run:

```bash
bash scripts/preprocess_libero.sh
```

By default this runs `libero_goal libero_object libero_spatial libero_10` and
delegates each suite to `scripts/preprocess/prepare_libero_data.sh`. To process
a subset, set `LIBERO_SUITES`:

```bash
LIBERO_SUITES="libero_goal libero_object" bash scripts/preprocess_libero.sh
```

For a single suite or a resumable rerun of one suite:

```bash
TASK=libero_goal bash scripts/preprocess/prepare_libero_data.sh
```

The preprocessing path is split into numbered child scripts, matching the
download and install flow. To rerun only one part:

```bash
TASK=libero_goal PREPROCESS_ONLY=20_pretokenize_dataset bash scripts/preprocess/prepare_libero_data.sh
TASK=libero_goal bash scripts/preprocess/20_pretokenize_dataset.sh
```

The default `FILTER_NOOPS=1` path does no-op handling before reward and hidden
sidecar generation:

```text
input: downloaded raw LIBERO HDF5 files
  ${DVLA_DATA_ROOT}/datasets/libero/${TASK}/*.hdf5

stage 1: replay and mark no-ops
  python -m dreamer_vla.preprocess.libero_utils.regenerate_libero_dataset_filter_no_op --keep-noops
  writes ${TASK}_marked_t_256 with data/demo_*/noop_mask

stage 2: filter marked no-ops
  python -m dreamer_vla.preprocess.filter_marked_libero_hdf5 --filter-noops
  writes ${TASK}_no_noops_t_256
```

Keep `FILTER_NOOPS=1` for the standard configs. Setting `FILTER_NOOPS=0`
writes `${TASK}_with_noops_t_256`, but the pretokenize configs in this release
target `*_no_noops_t_*` paths, so also set `RUN_PRETOKENIZE=0` when using that
debug path.

Outputs:

```text
${DVLA_DATA_ROOT:-data}/processed_data/${TASK}_marked_t_256
${DVLA_DATA_ROOT:-data}/processed_data/${TASK}_no_noops_t_256
${DVLA_DATA_ROOT:-data}/processed_data/${TASK}_no_noops_t_256_pi06_remaining_reward
${DVLA_DATA_ROOT:-data}/processed_data/${TASK}_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2
${DVLA_DATA_ROOT:-data}/configs/${TASK}/his_1_third_view_wrist_w_state_1_256_pretokenize*.yaml
```

Useful variants:

```bash
TASK=libero_10 ACTION_HIDDEN_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/preprocess/prepare_libero_data.sh
TASK=libero_goal RUN_ACTION_HIDDEN=0 bash scripts/preprocess/prepare_libero_data.sh
```

## 4. Train

VLA SFT:

```bash
CONFIG=vla_rynnvla_action_head NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_vla.sh task=libero_goal
```

One-trajectory VLA:

```bash
CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_goal
CONFIG=vla_sft_one_trajectory bash scripts/train_vla.sh task=libero_object dataset.trajectory_offset=2
CONFIG=openvla_oft_hdf5_one_trajectory bash scripts/train_vla.sh task=libero_10
CONFIG=openvla_oft_hdf5_one_trajectory_l1 bash scripts/train_vla.sh task=libero_goal
```

`openvla_oft_hdf5_one_trajectory` trains the discrete LM-head action-token
variant; `openvla_oft_hdf5_one_trajectory_l1` keeps the standard OFT L1
regression head, whose component checkpoints feed the action-hidden sidecar
and world-model chain.

OpenVLA-OFT action-hidden sidecar (feeds the OFT WM/classifier/DreamerVLA
routes; both checkpoint formats extract the same backbone layer):

```bash
# Component-wise L1 checkpoint (auto-detected):
TASK=libero_goal OFT_CKPT=data/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650 \
bash scripts/preprocess/35_oft_action_hidden.sh

# Downloaded discrete one-trajectory weights (single view, no history/proprio):
TASK=libero_goal \
OFT_CKPT=data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1 \
OFT_POLICY_MODE=discrete OFT_HISTORY=1 OFT_IMAGE_KEYS=agentview_rgb \
bash scripts/preprocess/35_oft_action_hidden.sh
```

First run on a new checkpoint: smoke-test by invoking the module directly with
`--max-files 1 --max-demos-per-file 2` before committing GPU hours.
Per model-on-dataset details (checkpoint formats, hidden shapes, expected
attrs): [docs/model_datasets/openvla_oft_libero_goal.md](docs/model_datasets/openvla_oft_libero_goal.md)
and [docs/model_datasets/rynnvla_libero_goal.md](docs/model_datasets/rynnvla_libero_goal.md).

Discrete sidecars are written with `action_head_type=oft_discrete_token`; when
training the WM on them, point the route at the sidecar and align the expected
attrs, e.g.:

```bash
CONFIG=oft_world_model_dinowm_chunk bash scripts/train_wm.sh task=libero_goal \
  task.openvla_oft.ckpt_path=/abs/path/to/Openvla-oft-SFT-libero-goal-traj1 \
  task.openvla_oft.action_hidden_dir=/abs/path/to/<sidecar_dir> \
  task.openvla_oft.expected_action_head_type=oft_discrete_token \
  task.openvla_oft.expected_history=1 \
  task.openvla_oft.expected_include_state=false
```

World model:

```bash
CONFIG=world_model_dinowm_chunk NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_wm.sh task=libero_goal
```

Classifier:

```bash
CONFIG=latent_classifier_libero_goal_chunk bash scripts/train_wm.sh
```

DreamerVLA:

```bash
CONFIG=dreamervla_rynn_dino_wm_wmpo_outcome NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_dreamervla.sh \
  task=libero_goal \
  init.world_model_state_ckpt=/abs/path/to/wm.ckpt \
  init.classifier_state_ckpt=/abs/path/to/classifier.ckpt
```

Hydra overrides are passed after the script command, for example
`task=libero_object`, `training.max_steps=1`, or
`task.hdf5_dir=/abs/path`.

## 5. Evaluate

VLA checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_vla.sh \
  eval.ckpt_kind=vla \
  eval.ckpt_path=/abs/path/to/vla.ckpt \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

Dreamer checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_libero_vla.sh \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=/abs/path/to/dreamer.ckpt \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

OpenVLA-OFT one-trajectory checkpoint:

```bash
CKPT_ROOT="${DVLA_DATA_ROOT:-data}/checkpoints/Openvla-oft-SFT-traj1" \
SUITE=libero_goal bash scripts/eval/launch_openvla_oft_official_libero_eval.sh
```

## 6. Verify

```bash
python -m pytest tests/unit_tests -q
```

Path checks:

```bash
test -d "${DVLA_DATA_ROOT:-data}/checkpoints/VLA_model_256/libero_goal"
test -d "${DVLA_DATA_ROOT:-data}/datasets/libero/libero_goal"
test -d "${DVLA_DATA_ROOT:-data}/processed_data/libero_goal_no_noops_t_256"
```

Smoke train:

```bash
OUT_DIR=/tmp/dvla_wm_smoke CONFIG=world_model_dinowm_chunk \
bash scripts/train_wm.sh task=libero_goal training.max_steps=1 dataloader.num_workers=0
```
