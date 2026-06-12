# RynnVLA Scheme B on LIBERO-Goal

Goal: train the frame-level input-token variant. The WM observes current-frame
Chameleon image-token embeddings, not action-query tokens.

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
bash scripts/download_assets.sh env.LIBERO_SUITES=libero_goal
```

## 2. Preprocess

Run the standard LIBERO preprocessing first:

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal gpus=0 ngpu=1
```

Then extract Scheme-B input-token sidecars:

```bash
TASK=libero_goal GPUS=0 ACTION_HIDDEN_GPUS=1 bash scripts/preprocess/32_input_token_hidden.sh
```

Expected sidecar:

```text
${DVLA_DATA_ROOT}/processed_data/libero_goal_no_noops_t_256_pi0_input_token_embedding_vla_policy_h2
```

## 3. Optional VLA SFT

```bash
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=libero_goal \
  gpus=0 ngpu=1 batch_size=4 num_workers=4
```

If you train a new VLA checkpoint, re-extract the input-token sidecar before
training the WM.

## 4. World Model

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk_input_tokens task=libero_goal \
  gpus=0 ngpu=1 batch_size=4 num_workers=4
```

For a smoke run:

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk_input_tokens task=libero_goal \
  gpus=0 ngpu=1 batch_size=1 num_workers=0 max_steps=1 out_dir=/tmp/dvla_rynn_b_wm_smoke
```

## 5. Classifier

Failure rollout HDF5 files and matching input-token failure sidecars are
required.

```bash
bash scripts/train_wm.sh experiment=latent_classifier_libero_goal_chunk_input_tokens gpus=0 \
  data.failure_dir_raw=/abs/path/to/libero_goal_failures \
  data.failure_dir_hidden=/abs/path/to/libero_goal_failures_input_tokens \
  batch_size=32 num_workers=4
```

## 6. DreamerVLA

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_rynn_dino_wm_wmpo_outcome_input_tokens \
  task=libero_goal \
  gpus=0 ngpu=1 batch_size=2 num_workers=2 \
  init.world_model_state_ckpt=/abs/path/to/input_token_world_model.ckpt \
  init.classifier_state_ckpt=/abs/path/to/input_token_classifier.ckpt
```

Scheme B cannot decode actions directly from a latent action-token axis. This
route uses `LatentToActionHiddenActor` to bridge frame tokens to action slots.

## 7. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path=/abs/path/to/dreamervla_input_token.ckpt \
  eval.dreamer_policy_source=ckpt \
  eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal \
  eval.num_episodes_per_task=10 \
  training.device=cuda:0
```
