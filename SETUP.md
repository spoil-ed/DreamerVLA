# DreamerVLA Setup

All commands run from the repository root. `DVLA_ROOT` is the checkout.
`DVLA_DATA_ROOT` is the runtime asset root and can be any writable path. If
unset, release scripts use relative `data`. See `docs/data_layout.md`.

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
scripts/install/20_python_deps.sh
scripts/install/30_third_party.sh
scripts/install/40_verify.sh
```

Run a single step when needed:

```bash
bash scripts/install/00_apt_tools.sh
bash scripts/install/10_conda_env.sh
bash scripts/install/20_python_deps.sh
bash scripts/install/30_third_party.sh
bash scripts/install/40_verify.sh
```

## 2. Download Assets

One-command download:

```bash
bash scripts/download_assets.sh
```

Single asset steps:

```bash
bash scripts/download/10_worldvla.sh
bash scripts/download/20_lumina.sh
LIBERO_SUITES="libero_goal libero_object libero_spatial libero_10" bash scripts/download/30_rynnvla.sh
LIBERO_SUITES="libero_goal libero_object libero_spatial libero_10" bash scripts/download/40_libero_dataset.sh
bash scripts/download/50_calvin_dataset.sh
```

Useful variants:

```bash
LIBERO_SUITES="libero_goal libero_object" bash scripts/download_assets.sh
DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=1 LIBERO_SUITES=libero_spatial bash scripts/download_assets.sh
DOWNLOAD_WEIGHTS=0 DOWNLOAD_LIBERO=0 DOWNLOAD_CALVIN=1 CALVIN_TASKS=task_ABCD_D bash scripts/download_assets.sh
DOWNLOAD_ONLY=10_worldvla bash scripts/download_assets.sh
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

```bash
TASK=libero_goal bash scripts/preprocess/prepare_libero_data.sh
```

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
