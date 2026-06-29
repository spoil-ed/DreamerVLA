# OpenVLA-OFT one-trajectory + LIBERO-Goal (discrete-token route)

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
bash scripts/download_assets.sh \
  download.rynnvla=false download.libero=true download.openvla_one_traj=true \
  env.LIBERO_SUITES=[libero_goal] \
  only=[30_openvla_oft_one_trajectory,40_libero_dataset]
```

## 3. Preprocess

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=rynnvla_libero libero_suite=libero_goal \
  only=[20_pretokenize_dataset] gpus=0 ngpu=1
```

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=openvla_onetraj_libero libero_suite=libero_goal \
  only=[10_hdf5_reward] gpus=0 ngpu=1
```

```bash
OFT_DISCRETE_CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
bash scripts/preprocess/prepare_libero_data.sh \
  task=openvla_onetraj_libero libero_suite=libero_goal \
  only=[35_oft_action_hidden] gpus=0 ngpu=1 \
  env.OFT_LATENT_SCHEME=action_hidden env.OFT_POLICY_MODE=discrete \
  env.OFT_HISTORY=1 env.OFT_IMAGE_KEYS=agentview_rgb \
  env.OFT_CKPT="${OFT_DISCRETE_CKPT}"
```

```bash
python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
  dir="${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward" \
  reference_dir="${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256" \
  match_reference_demos=true match_reference_lengths=true

python -m dreamervla.preprocess.check_artifacts command=hdf5-dir \
  dir="${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1" \
  reference_dir="${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward" \
  match_reference_demos=true match_reference_lengths=true require_complete_attr=true require_config=true
```

## 4. World model (discrete-token)

```bash
bash scripts/train_wm.sh experiment=oft_discrete_token_world_model_chunk \
  task=openvla_onetraj_libero gpus=0 ngpu=1 batch_size=2 num_workers=4
```

Smoke:

```bash
bash scripts/train_wm.sh experiment=oft_discrete_token_world_model_chunk \
  task=openvla_onetraj_libero gpus=0 ngpu=1 batch_size=1 num_workers=0 num_epochs=1 \
  out_dir=/tmp/openvla_onetraj_libero_discrete_wm_smoke
```

## 5. Classifier

```bash
bash scripts/train_wm.sh experiment=oft_latent_classifier_chunk task=openvla_onetraj_libero \
  gpus=0 batch_size=8 num_workers=4 -- \
  task.openvla_oft.action_hidden_dir="${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h1" \
  task.openvla_oft.expected_action_head_type=oft_discrete_token \
  task.openvla_oft.expected_include_state=false task.openvla_oft.expected_history=1 \
  task.openvla_oft.failure_hdf5_dir="${DVLA_DATA_ROOT}/collected_rollouts/OpenVLA_Onetraj_LIBERO_libero_goal_failures/reward" \
  task.openvla_oft.failure_action_hidden_dir="${DVLA_DATA_ROOT}/collected_rollouts/OpenVLA_Onetraj_LIBERO_libero_goal_failures/hidden"
```

## 6. DreamerVLA

```bash
bash scripts/train_dreamervla.sh experiment=dreamervla_oft_discrete_token_wm_lumos \
  task=openvla_onetraj_libero gpus=0 ngpu=1 batch_size=2 num_workers=2 -- \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT}/outputs/classifier/<run>/checkpoints/latest.ckpt"
```

## 7. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path="${DVLA_DATA_ROOT}/outputs/dreamervla/<run>/checkpoints/latest.ckpt" \
  eval.dreamer_policy_source=ckpt eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal eval.num_episodes_per_task=10 training.device=cuda:0
```

Raw OpenVLA-OFT checkpoint eval:

```bash
bash scripts/eval/launch_openvla_oft_official_libero_eval.sh \
  ckpt="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1" \
  suite=libero_goal gpu_id=0 policy_mode=discrete camera_inputs=primary \
  num_images=1 use_proprio=false
```
