# RynnVLA_LIBERO Pipeline

Goal: run the default RynnVLA + LIBERO-Goal pipeline with a matching Hydra task
name and processed-data artifact name.

Canonical task name:

```text
RynnVLA_LIBERO
```

This writes intermediate data under:

```text
${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/
```

The raw benchmark suite is still `libero_goal`; `RynnVLA_LIBERO` is the
pipeline task name and `RynnVLA_LIBERO_libero_goal` is the preprocessing
artifact name.

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
bash scripts/download_assets.sh env.LIBERO_SUITES=[libero_goal]
```

This downloads RynnVLA assets and the LIBERO-Goal raw dataset.

## 2. Preprocess

Action-hidden Scheme A:

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=RynnVLA_LIBERO \
  libero_suite=libero_goal \
  only=[10_hdf5_reward,30_action_hidden] \
  gpus=0 ngpu=1
```

Expected artifacts:

```text
${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256
${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256_remaining_reward
${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256_legacy_action_hidden_vla_policy_h2
```

## 3. Optional VLA SFT

Use the downloaded RynnVLA checkpoint by default. To train a one-trajectory VLA
checkpoint:

```bash
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=RynnVLA_LIBERO \
  gpus=0 ngpu=1 batch_size=4 num_workers=4
```

If you use a new VLA checkpoint, rerun `30_action_hidden` with `VLA_CKPT` or
`ENCODER_STATE_CKPT` pointing to that checkpoint.

## 4. World Model

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=RynnVLA_LIBERO \
  gpus=0 ngpu=1 batch_size=16 num_workers=4
```

Smoke run:

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=RynnVLA_LIBERO \
  gpus=0 ngpu=1 batch_size=2 num_workers=0 max_steps=1 out_dir=/tmp/rynnvla_libero_wm_smoke
```

## 5. Classifier

WMPO needs failure rollout HDF5 files and matching failure sidecars.

```bash
bash scripts/train_wm.sh experiment=latent_classifier_libero_goal_chunk task=RynnVLA_LIBERO gpus=0 \
  data.failure_dir_raw=/abs/path/to/RynnVLA_LIBERO_libero_goal_failures \
  data.failure_dir_hidden=/abs/path/to/RynnVLA_LIBERO_libero_goal_failures_action_hidden \
  batch_size=32 num_workers=4
```

## 6. DreamerVLA

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_rynn_dino_wm_wmpo_outcome \
  task=RynnVLA_LIBERO \
  gpus=0 ngpu=1 batch_size=4 num_workers=2 \
  init.world_model_state_ckpt=/abs/path/to/world_model.ckpt \
  init.classifier_state_ckpt=/abs/path/to/classifier.ckpt
```

Without a classifier, use the actor-critic fallback:

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_rynn_dino_wm_actor_critic \
  task=RynnVLA_LIBERO \
  gpus=0 ngpu=1 batch_size=4 num_workers=2 \
  init.world_model_state_ckpt=/abs/path/to/world_model.ckpt
```

## 7. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=/abs/path/to/dreamervla.ckpt \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```
