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

RynnVLA action-hidden WM/DreamerVLA routes use reward-labeled HDF5 from
`10_hdf5_reward` plus the legacy RynnVLA sidecar from `30_action_hidden`. They
do not need `20_pretokenize_dataset`; that step is for token-record SFT and
older pretokenized dataset routes. OpenVLA-OFT uses the same reward stage but a
different sidecar extractor, `35_oft_action_hidden`, because its action head and
hidden-state layout are different.

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

Manual integrity check:

```bash
python -m dreamervla.preprocess.check_artifacts hdf5-dir \
  --dir "${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256_remaining_reward" \
  --reference-dir "${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256" \
  --match-reference-demos \
  --match-reference-lengths

python -m dreamervla.preprocess.check_artifacts hdf5-dir \
  --dir "${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256_legacy_action_hidden_vla_policy_h2" \
  --reference-dir "${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256_remaining_reward" \
  --match-reference-demos \
  --match-reference-lengths \
  --require-complete-attr \
  --require-config
```

This checks that the reward HDF5 files match the no-noops source files and that
the action-hidden sidecar has the same file set, demo keys, per-demo lengths,
`complete=true` markers, and `preprocess_config.json` schema metadata.

If `.tmp` or `.rank*.tmp` files remain under the artifact directories, the usual
reason is that preprocessing was interrupted before the atomic rename to the
final `.hdf5` completed. Re-running the same preprocessing step removes the old
rank-local tmp for that output before writing it again. Only delete tmp files by
hand after confirming no preprocessing process is still running:

```bash
find "${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal" \
  -type f \( -name "*.tmp" -o -name "*.rank*.tmp" \) -print
```

## 3. Optional VLA SFT

Use the downloaded RynnVLA checkpoint by default. To train a one-trajectory VLA
checkpoint:

```bash
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=RynnVLA_LIBERO \
  gpus=0 ngpu=1 batch_size=4 num_workers=4
```

Add logging overrides to any training command. The project uses TensorBoard
event files for the local TensorFlow-compatible log viewer:

```bash
logger=tensorboard
logger=wandb
logger=tensorboard_wandb runner.logger.wandb_mode=online
logger=tensorboard_wandb runner.logger.wandb_mode=offline
```

TensorBoard writes `${training.out_dir}/log/tensorboard`; W&B writes
`${training.out_dir}/log/wandb`. `wandb_mode=offline` keeps the W&B run local for
later sync.

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
