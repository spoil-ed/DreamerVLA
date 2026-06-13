# OpenVLA-OFT Scheme B on LIBERO-Goal

Goal: train the OFT frame-token route. The WM observes current-frame projected
vision patch tokens, not OFT action-slot hidden tokens.

Use a component-wise L1 OFT checkpoint for DreamerVLA, because the bridge actor
still needs an L1 output action head.

## 0. System

```bash
cd /path/to/DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_ROOT}/data"

bash scripts/install_env.sh
conda activate dreamervla
```

## 1. Download

```bash
bash scripts/download_assets.sh download.rynnvla=false download.libero=true \
  only=[40_libero_dataset] env.LIBERO_SUITES=libero_goal
```

Use or download an L1 OFT checkpoint:

```bash
bash scripts/download_assets.sh download.openvla_oft=true only=[20_openvla_oft] \
  env.OPENVLA_OFT_REPOS=owner/repo:libero_goal_hdf5_latest_6650
```

## 2. Preprocess

Build the reward HDF5:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal only=[10_hdf5_reward]
```

Extract Scheme-B OFT input-token sidecars:

```bash
TASK=libero_goal \
GPUS=0 \
OFT_ACTION_HIDDEN_GPUS=1 \
OFT_CKPT="${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/libero_goal_hdf5_latest_6650" \
OFT_POLICY_MODE=auto \
OFT_LATENT_SCHEME=input_tokens \
bash scripts/preprocess/35_oft_action_hidden.sh
```

Expected sidecar:

```text
${DVLA_DATA_ROOT}/processed_data/libero_goal/libero_goal_no_noops_t_256_oft_input_token_embedding_vla_policy_h2
```

## 3. World Model

```bash
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk_input_tokens task=libero_goal \
  gpus=0 ngpu=1 batch_size=1 num_workers=4
```

For a smoke run:

```bash
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk_input_tokens task=libero_goal \
  gpus=0 ngpu=1 batch_size=1 num_workers=0 max_steps=1 out_dir=/tmp/dvla_oft_b_wm_smoke
```

## 4. Classifier

Failure rollout HDF5 files and matching OFT input-token failure sidecars are
required.

```bash
bash scripts/train_wm.sh experiment=oft_latent_classifier_chunk_input_tokens task=libero_goal gpus=0 \
  task.openvla_oft.failure_hdf5_dir=/abs/path/to/libero_goal_failures \
  task.openvla_oft.failure_input_token_hidden_dir=/abs/path/to/libero_goal_failures_oft_input_tokens \
  batch_size=8 num_workers=4
```

## 5. DreamerVLA

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_oft_dino_wm_wmpo_outcome_input_tokens \
  task=libero_goal \
  gpus=0 ngpu=1 batch_size=1 num_workers=2 \
  init.world_model_state_ckpt=/abs/path/to/oft_input_token_world_model.ckpt \
  init.classifier_state_ckpt=/abs/path/to/oft_input_token_classifier.ckpt
```

Scheme B cannot distinguish action slots in the latent tokens. The WM remains
action-conditioned, and DreamerVLA uses `LatentToActionHiddenActor` to bridge
frame tokens to an L1 action head.

## 6. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=/abs/path/to/oft_input_token_dreamervla.ckpt \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```
