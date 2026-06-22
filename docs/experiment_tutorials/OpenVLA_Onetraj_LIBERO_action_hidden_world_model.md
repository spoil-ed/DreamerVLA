# OpenVLA-OFT one-trajectory — action-hidden WM (Scheme A)

Background, WM architecture, memory/OOM & known gaps: [EXPLAINED.md](EXPLAINED.md) ·
parameters: [../PARAMETERS.md](../PARAMETERS.md).

## 0. Env

```bash
cd DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
conda activate dreamervla
export MUJOCO_GL=osmesa
```

## 1. Preprocess (reward HDF5 + OFT action-hidden sidecar)

```bash
bash scripts/preprocess/prepare_libero_data.sh \
  task=openvla_onetraj_libero libero_suite=libero_goal \
  only=[10_hdf5_reward] gpus=0 ngpu=1

OFT_CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
bash scripts/preprocess/prepare_libero_data.sh \
  task=openvla_onetraj_libero libero_suite=libero_goal \
  only=[35_oft_action_hidden] gpus=0 ngpu=1 \
  env.OFT_LATENT_SCHEME=action_hidden env.OFT_POLICY_MODE=discrete \
  env.OFT_HISTORY=1 env.OFT_IMAGE_KEYS=agentview_rgb \
  env.OFT_CKPT="${OFT_CKPT}"
```

## 2. World model

```bash
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk task=openvla_onetraj_libero \
  gpus=0 ngpu=1 batch_size=2 num_workers=4
```

## 3. Classifier

```bash
bash scripts/train_wm.sh experiment=oft_latent_classifier_chunk task=openvla_onetraj_libero \
  gpus=0 batch_size=8 num_workers=4
```

## 4. DreamerVLA (wmpo_outcome)

```bash
bash scripts/train_dreamervla.sh experiment=dreamervla_oft_dino_wm_wmpo_outcome task=openvla_onetraj_libero \
  gpus=0 ngpu=1 batch_size=2 num_workers=2 -- \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT}/outputs/classifier/<run>/checkpoints/latest.ckpt"
```

## 5. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path="${DVLA_DATA_ROOT}/outputs/dreamervla/<run>/checkpoints/latest.ckpt" \
  eval.dreamer_policy_source=ckpt eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal eval.num_episodes_per_task=10 training.device=cuda:0
```

## 6. Unified online cotrain (one Hydra call)

```bash
# full (N GPUs = N parallel rollouts + DDP cotrain)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m dreamervla.train experiment=online_cotrain_oft_action_hidden
# one-command dry-run
WANDB_MODE=disabled python -m dreamervla.train \
  experiment=online_cotrain_oft_action_hidden training.debug=true
```

## 7. Tiny offline smoke (1 file, balanced set → ckpt)

`SC` (hidden sidecar) **must match the experiment's expected OFT metadata** —
`action_head_type`, `history`, `include_state`, `obs_hidden_source` are cross-checked
against `task.openvla_oft.expected_*` and the run aborts on any mismatch. For
`task=openvla_onetraj_libero` the expectation is `oft_discrete_token`, `history=1`,
`include_state=false`, `obs_hidden_source=action_query` (the `_oft_legacy_action_hidden_vla_policy_h1`
sidecar from the discrete recipe). Set `SC`/`RW` to a sidecar+reward pair you generated
with those settings; the on-disk `*_oft_*_legacy_action_hidden_vla_policy_h2` dumps are
the **L1-regression** route (`oft_l1_regression`, `history=2`) and will be rejected by the
discrete WM. The WM/classifier/DreamerVLA offline routes below were each run end-to-end to
a `latest.ckpt` against a metadata-matching discrete sidecar.

```bash
SC=<discrete action-query sidecar matching task.openvla_oft.expected_* (history=1)>
RW=<matching reward HDF5 dir>
MP="${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/libero_goal"
PY="PYTHONPATH=. python"

# WM smoke
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa WANDB_MODE=disabled $PY -m dreamervla.train \
  experiment=oft_world_model_dinowm_chunk task=openvla_onetraj_libero logger=tensorboard \
  training.out_dir=/tmp/oft_onetraj_wm_smoke training.num_epochs=1 \
  dataloader.batch_size=1 dataloader.num_workers=0 \
  task.hdf5_reward_dir=$RW task.openvla_oft.hdf5_reward_dir=$RW task.openvla_oft.action_hidden_dir=$SC \
  dataset.hdf5_dir=$RW dataset.hidden_dir=$SC dataset.expected_model_path=$MP \
  dataset.max_files=1 dataset.max_demos_per_file=3 dataset.balanced_length=8

# classifier smoke
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa WANDB_MODE=disabled $PY -m dreamervla.train \
  experiment=oft_latent_classifier_chunk task=openvla_onetraj_libero logger=tensorboard \
  training.out_dir=/tmp/oft_onetraj_cls_smoke training.num_epochs=1 \
  training.batch_size=2 training.num_workers=0 training.episode_eval_enabled=false \
  task.openvla_oft.hdf5_dir=$RW task.openvla_oft.action_hidden_dir=$SC \
  data.success_dir_raw=$RW data.success_dir_hidden=$SC

# DreamerVLA wmpo_outcome smoke (memory-bounded imagination)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa WANDB_MODE=disabled \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m dreamervla.train \
  experiment=dreamervla_oft_dino_wm_wmpo_outcome task=openvla_onetraj_libero logger=tensorboard \
  training.out_dir=/tmp/oft_onetraj_dvla_smoke training.num_epochs=1 \
  dataloader.batch_size=1 dataloader.num_workers=0 dataloader.multiprocessing_context=null dataloader.persistent_workers=false \
  task.hdf5_reward_dir=$RW task.openvla_oft.hdf5_reward_dir=$RW task.openvla_oft.action_hidden_dir=$SC \
  dataset.hdf5_dir=$RW dataset.hidden_dir=$SC dataset.expected_model_path=$MP \
  dataset.max_files=1 dataset.max_demos_per_file=3 dataset.balanced_length=4 \
  algorithm.wmpo.episode_max_steps=40 algorithm.ppo_rollouts_per_start=2 algorithm.imag_last=2 \
  init.world_model_state_ckpt=/tmp/oft_onetraj_wm_smoke/ckpt/latest.ckpt \
  init.classifier_state_ckpt=/tmp/oft_onetraj_cls_smoke/best_format.ckpt
```
