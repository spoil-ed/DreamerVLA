# OpenVLA_Onetraj_LIBERO Pipeline

Goal: run the OpenVLA-OFT one-trajectory + LIBERO-Goal pipeline with matching
Hydra task and processed-data artifact names.

Canonical task name:

```text
OpenVLA_Onetraj_LIBERO
```

This writes intermediate data under:

```text
${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO/
```

The raw benchmark suite is still `libero_goal`; `OpenVLA_Onetraj_LIBERO` is the
pipeline/artifact name.

Use an L1 OFT checkpoint for the full DreamerVLA WMPO chain. Discrete
OpenVLA-OFT checkpoints can be used for sidecar experiments, but the current
DreamerVLA actor route expects an L1 action head.

## 0. System

```bash
cd /path/to/DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_ROOT}/data"

bash scripts/install_env.sh
conda activate dreamervla
```

## 1. Download

Download LIBERO-Goal and one-trajectory OpenVLA-OFT assets:

```bash
bash scripts/download_assets.sh \
  download.rynnvla=false \
  download.libero=true \
  download.openvla_one_traj=true \
  env.LIBERO_SUITES=[libero_goal] \
  only=[30_openvla_oft_one_trajectory,40_libero_dataset]
```

The default one-trajectory checkpoint path is:

```text
${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1
```

To train a local L1 one-trajectory checkpoint instead:

```bash
bash scripts/train_vla.sh \
  experiment=openvla_oft_hdf5_one_trajectory_l1 \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=1 num_workers=4
```

Use the resulting checkpoint directory as `OFT_CKPT` during sidecar extraction.
For DreamerVLA actor initialization, set
`task.openvla_oft.action_head_ckpt` to that checkpoint directory's
`action_head--<step>_checkpoint.pt`.

## 2. Preprocess

Build the renamed reward HDF5:

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=OpenVLA_Onetraj_LIBERO \
  libero_suite=libero_goal \
  only=[10_hdf5_reward] \
  gpus=0 ngpu=1
```

Extract OpenVLA-OFT action-hidden Scheme A sidecars:

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=OpenVLA_Onetraj_LIBERO \
  libero_suite=libero_goal \
  only=[35_oft_action_hidden] \
  gpus=0 ngpu=1 \
  env.OFT_LATENT_SCHEME=action_hidden \
  env.OFT_POLICY_MODE=l1 \
  env.OFT_CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
```

Expected artifacts:

```text
${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO/OpenVLA_Onetraj_LIBERO_no_noops_t_256
${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO/OpenVLA_Onetraj_LIBERO_no_noops_t_256_pi06_remaining_reward
${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO/OpenVLA_Onetraj_LIBERO_no_noops_t_256_oft_legacy_action_hidden_vla_policy_h2
```

Optional input-token Scheme B:

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=OpenVLA_Onetraj_LIBERO \
  libero_suite=libero_goal \
  only=[35_oft_action_hidden] \
  gpus=0 ngpu=1 \
  env.OFT_LATENT_SCHEME=input_tokens \
  env.OFT_POLICY_MODE=l1 \
  env.OFT_CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
```

## 3. World Model

```bash
bash scripts/train_wm.sh \
  experiment=oft_world_model_dinowm_chunk \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=2 num_workers=4
```

Smoke run:

```bash
bash scripts/train_wm.sh \
  experiment=oft_world_model_dinowm_chunk \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=1 num_workers=0 max_steps=1 \
  out_dir=/tmp/openvla_onetraj_libero_wm_smoke
```

## 4. Classifier

WMPO needs failure rollout HDF5 files and matching OFT failure sidecars.

```bash
bash scripts/train_wm.sh \
  experiment=oft_latent_classifier_chunk \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 \
  task.openvla_oft.failure_hdf5_dir=/abs/path/to/OpenVLA_Onetraj_LIBERO_failures \
  task.openvla_oft.failure_action_hidden_dir=/abs/path/to/OpenVLA_Onetraj_LIBERO_failures_oft_action_hidden \
  batch_size=8 num_workers=4
```

## 5. DreamerVLA

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_oft_dino_wm_wmpo_outcome \
  task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=2 num_workers=2 \
  task.openvla_oft.action_head_ckpt=/abs/path/to/action_head--step_checkpoint.pt \
  init.world_model_state_ckpt=/abs/path/to/oft_world_model.ckpt \
  init.classifier_state_ckpt=/abs/path/to/oft_classifier.ckpt
```

## 6. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=/abs/path/to/openvla_onetraj_dreamervla.ckpt \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

Raw OpenVLA-OFT checkpoint eval:

```bash
CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1" \
SUITE=libero_goal \
GPU_ID=0 \
POLICY_MODE=l1 \
CAMERA_INPUTS=primary,wrist \
NUM_IMAGES=2 \
USE_PROPRIO=1 \
bash scripts/eval/launch_openvla_oft_official_libero_eval.sh
```
