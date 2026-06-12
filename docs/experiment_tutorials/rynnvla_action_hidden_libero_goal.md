# RynnVLA Scheme A on LIBERO-Goal

Goal: reproduce the RynnVLA-002-style action-hidden workflow. The WM observes
35 action-query tokens per frame (`5 actions x 7 dims`, token dim 1024).

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

This downloads the RynnVLA assets and the LIBERO-Goal dataset under
`${DVLA_DATA_ROOT}`.

## 2. Preprocess

```bash
bash scripts/preprocess/prepare_libero_data.sh task=libero_goal gpus=0 ngpu=1
```

The standard preprocessing chain writes the reward HDF5 and the Scheme-A
RynnVLA sidecar:

```text
${DVLA_DATA_ROOT}/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward
${DVLA_DATA_ROOT}/processed_data/libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2
```

To rerun only the sidecar:

```bash
TASK=libero_goal GPUS=0 ACTION_HIDDEN_GPUS=1 bash scripts/preprocess/30_action_hidden.sh
```

## 3. Optional VLA SFT

Use the downloaded RynnVLA checkpoint by default. To train a one-trajectory VLA
checkpoint:

```bash
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=libero_goal \
  gpus=0 ngpu=1 batch_size=4 num_workers=4
```

If you use a new VLA checkpoint, re-extract the sidecar with that checkpoint
before WM training.

## 4. World Model

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal \
  gpus=0 ngpu=1 batch_size=16 num_workers=4
```

For a smoke run:

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=libero_goal \
  gpus=0 ngpu=1 batch_size=2 num_workers=0 max_steps=1 out_dir=/tmp/dvla_rynn_a_wm_smoke
```

## 5. Classifier

This step requires failure rollout HDF5 files and matching failure sidecars.

```bash
bash scripts/train_wm.sh experiment=latent_classifier_libero_goal_chunk gpus=0 \
  data.failure_dir_raw=/abs/path/to/libero_goal_failures \
  data.failure_dir_hidden=/abs/path/to/libero_goal_failures_action_hidden \
  batch_size=32 num_workers=4
```

## 6. DreamerVLA

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_rynn_dino_wm_wmpo_outcome \
  task=libero_goal \
  gpus=0 ngpu=1 batch_size=4 num_workers=2 \
  init.world_model_state_ckpt=/abs/path/to/world_model.ckpt \
  init.classifier_state_ckpt=/abs/path/to/classifier.ckpt
```

If you do not have a classifier yet, use
`experiment=dreamervla_rynn_dino_wm_actor_critic` as the non-WMPO fallback.

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
