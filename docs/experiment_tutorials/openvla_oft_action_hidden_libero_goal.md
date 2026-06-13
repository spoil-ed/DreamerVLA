# OpenVLA-OFT Scheme A on LIBERO-Goal

Goal: train the OFT action-hidden route. The WM observes 56 OFT action-slot
hidden tokens per frame (`8 actions x 7 dims`, token dim 4096).

Use a component-wise L1 OFT checkpoint for the full WMPO chain. Discrete OFT
checkpoints can produce Scheme-A sidecars for WM experiments, but the current
DreamerVLA action actor route expects an L1 action head.

## 0. System

```bash
cd /path/to/DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_ROOT}/data"

bash scripts/install_env.sh
conda activate dreamervla
```

## 1. Download

Download LIBERO-Goal:

```bash
bash scripts/download_assets.sh download.rynnvla=false download.libero=true \
  only=[40_libero_dataset] env.LIBERO_SUITES=libero_goal
```

Use an existing L1 OFT checkpoint at:

```text
${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650
```

Or download one from a user-provided Hugging Face repo:

```bash
bash scripts/download_assets.sh download.openvla_oft=true only=[20_openvla_oft] \
  env.OPENVLA_OFT_REPOS=owner/repo:libero_goal_hdf5_latest_6650
```

To train a one-trajectory L1 OFT checkpoint locally:

```bash
bash scripts/train_vla.sh experiment=openvla_oft_hdf5_one_trajectory_l1 task=libero_goal \
  gpus=0 ngpu=1 batch_size=1 num_workers=4
```

## 2. Preprocess

Build the reward HDF5:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal only=[10_hdf5_reward]
```

Extract Scheme-A OFT sidecars:

```bash
TASK=libero_goal \
GPUS=0 \
OFT_ACTION_HIDDEN_GPUS=1 \
OFT_CKPT="${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650" \
OFT_POLICY_MODE=auto \
OFT_LATENT_SCHEME=action_hidden \
bash scripts/preprocess/35_oft_action_hidden.sh
```

The wrapper writes:

```text
${DVLA_DATA_ROOT}/processed_data/libero_goal/libero_goal_no_noops_t_256_oft_legacy_action_hidden_vla_policy_h2
```

## 3. World Model

Pass the sidecar path if you use the wrapper output above:

```bash
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk task=libero_goal \
  gpus=0 ngpu=1 batch_size=2 num_workers=4 \
  task.openvla_oft.action_hidden_dir="${DVLA_DATA_ROOT}/processed_data/libero_goal/libero_goal_no_noops_t_256_oft_legacy_action_hidden_vla_policy_h2"
```

For a smoke run:

```bash
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk task=libero_goal \
  gpus=0 ngpu=1 batch_size=1 num_workers=0 max_steps=1 out_dir=/tmp/dvla_oft_a_wm_smoke \
  task.openvla_oft.action_hidden_dir="${DVLA_DATA_ROOT}/processed_data/libero_goal/libero_goal_no_noops_t_256_oft_legacy_action_hidden_vla_policy_h2"
```

## 4. Classifier

Failure rollout HDF5 files and matching OFT action-hidden failure sidecars are
required.

```bash
bash scripts/train_wm.sh experiment=oft_latent_classifier_chunk task=libero_goal gpus=0 \
  task.openvla_oft.action_hidden_dir="${DVLA_DATA_ROOT}/processed_data/libero_goal/libero_goal_no_noops_t_256_oft_legacy_action_hidden_vla_policy_h2" \
  task.openvla_oft.failure_hdf5_dir=/abs/path/to/libero_goal_failures \
  task.openvla_oft.failure_action_hidden_dir=/abs/path/to/libero_goal_failures_oft_action_hidden \
  batch_size=8 num_workers=4
```

## 5. DreamerVLA

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_oft_dino_wm_wmpo_outcome \
  task=libero_goal \
  gpus=0 ngpu=1 batch_size=2 num_workers=2 \
  task.openvla_oft.action_hidden_dir="${DVLA_DATA_ROOT}/processed_data/libero_goal/libero_goal_no_noops_t_256_oft_legacy_action_hidden_vla_policy_h2" \
  init.world_model_state_ckpt=/abs/path/to/oft_world_model.ckpt \
  init.classifier_state_ckpt=/abs/path/to/oft_classifier.ckpt
```

## 6. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=/abs/path/to/oft_dreamervla.ckpt \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

To evaluate the raw OFT checkpoint before DreamerVLA:

```bash
CKPT="${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650" \
SUITE=libero_goal \
GPU_ID=0 \
POLICY_MODE=l1 \
CAMERA_INPUTS=primary,wrist \
NUM_IMAGES=2 \
USE_PROPRIO=1 \
bash scripts/eval/launch_openvla_oft_official_libero_eval.sh
```
