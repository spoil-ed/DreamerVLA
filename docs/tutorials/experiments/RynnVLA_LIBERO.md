# RynnVLA + LIBERO-Goal

Background & parameters: [EXPLAINED.md](EXPLAINED.md) · [../../PARAMETERS.md](../../PARAMETERS.md).

## 1. Install

```bash
cd DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
bash scripts/install_env.sh
conda activate dreamervla
```

## 2. Download

```bash
bash scripts/download_assets.sh env.LIBERO_SUITES=[libero_goal]
```

## 3. Preprocess (reward HDF5 + action-hidden sidecar)

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=rynnvla_libero libero_suite=libero_goal \
  only=[10_hdf5_reward,30_action_hidden] gpus=0 ngpu=1
```

```bash
python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
  dir="${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256_remaining_reward" \
  reference_dir="${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256" \
  match_reference_demos=true match_reference_lengths=true

python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
  dir="${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256_legacy_action_hidden_vla_policy_h2" \
  reference_dir="${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal/no_noops_t_256_remaining_reward" \
  match_reference_demos=true match_reference_lengths=true require_complete_attr=true require_config=true

find "${DVLA_DATA_ROOT}/processed_data/RynnVLA_LIBERO_libero_goal" \
  -type f \( -name "*.tmp" -o -name "*.rank*.tmp" \) -print
```

## 4. (Optional) VLA SFT

```bash
bash scripts/train_vla.sh experiment=vla_sft_one_trajectory task=rynnvla_libero \
  gpus=0 ngpu=1 batch_size=4 num_workers=4
```

## 5. World model

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=rynnvla_libero \
  gpus=0 ngpu=1 batch_size=16 num_workers=4
```

Smoke:

```bash
bash scripts/train_wm.sh experiment=world_model_dinowm_chunk task=rynnvla_libero \
  gpus=0 ngpu=1 batch_size=2 num_workers=0 num_epochs=1 out_dir=/tmp/rynnvla_libero_wm_smoke
```

## 6. Classifier

```bash
bash scripts/train_wm.sh experiment=latent_classifier_libero_goal_chunk task=rynnvla_libero gpus=0 \
  data.failure_dir_raw="${DVLA_DATA_ROOT}/collected_rollouts/RynnVLA_LIBERO_libero_goal_failures/reward" \
  data.failure_dir_hidden="${DVLA_DATA_ROOT}/collected_rollouts/RynnVLA_LIBERO_libero_goal_failures/hidden" \
  batch_size=32 num_workers=4
```

## 7. DreamerVLA

```bash
bash scripts/train_dreamervla.sh experiment=dreamervla_rynn_dino_wm_lumos task=rynnvla_libero \
  gpus=0 ngpu=1 batch_size=4 num_workers=2 \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT}/outputs/classifier/<run>/checkpoints/latest.ckpt"
```

Actor-critic fallback (no classifier):

```bash
bash scripts/train_dreamervla.sh experiment=dreamervla_rynn_dino_wm_actor_critic task=rynnvla_libero \
  gpus=0 ngpu=1 batch_size=4 num_workers=2 \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT}/outputs/worldmodel/<run>/checkpoints/latest.ckpt"
```

## 8. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path="${DVLA_DATA_ROOT}/outputs/dreamervla/<run>/checkpoints/latest.ckpt" \
  eval.dreamer_policy_source=ckpt eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal eval.num_episodes_per_task=10 training.device=cuda:0
```
