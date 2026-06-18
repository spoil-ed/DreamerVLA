# OpenVLA-OFT One-Trajectory — Action-Hidden World Model DreamerVLA

Complete, **verified-runnable** script for the OpenVLA-OFT one-trajectory
action-hidden DreamerVLA pipeline on LIBERO-Goal.

Latent route: **Scheme A — action hidden** (the action-slot hidden tokens the
OFT action head consumes, `[T, 56, 4096]` per frame; flat `obs_embedding` dim
`56*4096 = 229376`). The world model predicts future action-hidden; the
classifier scores imagined latent windows; the DreamerVLA actor is trained by
`wmpo_outcome` imagined rollout.

Pipeline (each stage is its own Hydra entry; chained by checkpoint path):

```
preprocess (reward HDF5 + OFT action-hidden sidecar)
   → world model         (experiment=oft_world_model_dinowm_chunk)
   → success classifier  (experiment=oft_latent_classifier_chunk)
   → DreamerVLA actor    (experiment=dreamervla_oft_dino_wm_wmpo_outcome,
                          init from WM + classifier ckpts; wmpo_outcome)
   → LIBERO eval
```

> The sub-training logic is **unchanged** — every stage runs the existing real
> runner (`OFTDinoWMRunner` / `LatentClassifierRunner` / `JointDreamerVLARunner`
> with the `wmpo_outcome` route). This recipe only wires data/config/ckpt paths.

## Unified online cotrain (single Hydra call)

`dreamervla.runners.OnlineCotrainRunner` implements the unified pipeline in **one
`train` call**: one-traj VLA → **parallel online rollout** (one
`DreamerVLAOnlineTrainEnv` per `torchrun` rank) → `OnlineReplay` →
**warmup (WM + classifier only, `training.warmup_steps`)** →
**cotrain (WM + classifier + slow-policy RL, `dino_wmpo_outcome_step`)**. It
reuses the existing step functions (`world_model_pretrain_step`,
`online_classifier_update_step`, `dino_wmpo_outcome_step`); WM/actor are frozen
during WM/classifier updates and the actor only updates in the RL phase
(asserted at runtime).

```bash
# full (N GPUs = N parallel rollouts + DDP cotrain)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m dreamervla.train experiment=online_cotrain_oft_action_hidden
# one-command dry-run (rollout -> replay -> WM/cls warmup -> RL cotrain)
WANDB_MODE=disabled python -m dreamervla.train \
  experiment=online_cotrain_oft_action_hidden training.debug=true
```

Config: `configs/experiment/online_cotrain_oft_action_hidden.yaml` (`latent_type=action_hidden`,
`warmup_steps`, `train_actor_after_warmup`, `train_classifier_inline`,
`online_rollout.*`, `optim.policy.lr=5e-7` slow policy). The **online env-rollout
policy is the RynnVLA one-trajectory VLA** (the ~40% seed) since the online env
emits the `action_query` latent via `RynnVLAEncoder`; the **OpenVLA-OFT**
one-trajectory checkpoint uses the OFFLINE staged path below. (Scheme 1 /
backbone-latent online rollout is a separate, not-yet-wired path — see the
backbone-latent tutorial.)

## 0. System

```bash
cd DreamerVLA
export DVLA_ROOT="$(pwd -P)"
export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
conda activate dreamervla
# LIBERO rendering on this host: EGL crashes in robosuite read_pixels; use osmesa.
export MUJOCO_GL=osmesa
# Grouped training defaults to tensorboard+wandb(online). For local runs use
# tensorboard only (no W&B API key needed):
#   logger=tensorboard
```

Required assets (already present on this host):

```text
data/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1   # OFT one-traj ckpt (discrete)
data/datasets/libero/libero_goal/*.hdf5                                    # raw LIBERO-Goal demos
```

## 1. Preprocess — reward HDF5 + OFT action-hidden sidecar

```bash
# 1a. no-op filter + remaining-steps reward HDF5
bash scripts/preprocess/prepare_libero_data.sh \
  task=OpenVLA_Onetraj_LIBERO libero_suite=libero_goal \
  only=[10_hdf5_reward] gpus=0 ngpu=1

# 1b. OFT Scheme-A action-hidden sidecar (discrete one-traj ckpt)
OFT_CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
bash scripts/preprocess/prepare_libero_data.sh \
  task=OpenVLA_Onetraj_LIBERO libero_suite=libero_goal \
  only=[35_oft_action_hidden] gpus=0 ngpu=1 \
  env.OFT_LATENT_SCHEME=action_hidden env.OFT_POLICY_MODE=discrete \
  env.OFT_HISTORY=2 env.OFT_IMAGE_KEYS="agentview_rgb eye_in_hand_rgb" \
  env.OFT_CKPT="${OFT_CKPT}"
```

Artifacts:

```text
data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_remaining_reward
data/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/no_noops_t_256_oft_legacy_action_hidden_vla_policy_h2
```

The OFT preprocess module accepts `--max-files` / `--max-demos-per-file` and
`--fake-oft-components` (structural-only, no 7B inference) for fast plumbing
tests.

## 2. World model

```bash
bash scripts/train_wm.sh \
  experiment=oft_world_model_dinowm_chunk task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=2 num_workers=4
```

Produces a `ChunkAwareDinoWMWorldModel` checkpoint under
`${training.out_dir}/ckpt/latest.ckpt` (obs latent dim `229376`).

## 3. Success / reward classifier

```bash
bash scripts/train_wm.sh \
  experiment=oft_latent_classifier_chunk task=OpenVLA_Onetraj_LIBERO \
  gpus=0 batch_size=8 num_workers=4
```

`LatentSuccessClassifier` over flat `obs_embedding` windows. Failure rollouts
are optional (`data.failure_dir_*` default to null → success-only). The
DreamerVLA stage loads the classifier **best-format** checkpoint
(`{model, config.classifier, threshold}`); the best ckpt is produced when
`training.episode_eval_enabled=true`.

## 4. DreamerVLA actor (wmpo_outcome)

```bash
bash scripts/train_dreamervla.sh \
  experiment=dreamervla_oft_dino_wm_wmpo_outcome task=OpenVLA_Onetraj_LIBERO \
  gpus=0 ngpu=1 batch_size=2 num_workers=2 \
  -- \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT}/outputs/classifier/<run>/checkpoints/latest.ckpt"
```

WM + classifier are frozen; the actor is updated by `dino_wmpo_outcome_step`
(imagined chunk rollout → classifier outcome reward → PPO), slow policy
`optim.policy.lr=5e-7`.

> OFT memory note: the OFT action-hidden latent (`229376`) is ~6.5× the RynnVLA
> latent, so the imagined video tensor is large. If you hit CUDA OOM, set
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and/or lower
> `algorithm.wmpo.episode_max_steps` and `algorithm.ppo_rollouts_per_start`.

## 5. Eval

```bash
bash scripts/eval_libero_vla.sh gpus=0 \
  eval.ckpt_kind=dreamer \
  eval.ckpt_path="${DVLA_DATA_ROOT}/outputs/dreamervla/<run>/checkpoints/latest.ckpt" \
  eval.dreamer_policy_source=ckpt eval.dreamer_actor_input_source=rssm \
  eval.task_suite_name=libero_goal eval.num_episodes_per_task=10 \
  training.device=cuda:0
```

---

## Verified smoke (CPU/GPU, reuses existing data, no full training)

The following exact commands were run on this host and **complete without
error**, exercising the real WM / classifier / DreamerVLA training logic on the
existing LIBERO-Goal OFT action-hidden sidecar. They point the
`OpenVLA_Onetraj_LIBERO` task at the already-present `libero_goal_*` artifacts
(only `expected_model_path` differs, so it is overridden to match the sidecar).

```bash
cd DreamerVLA
SC=data/processed_data/libero_goal_no_noops_t_256_oft_official_legacy_action_hidden_vla_policy_h2
RW=data/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward
MP="${DVLA_DATA_ROOT}/checkpoints/OpenVLA-OFT/libero_goal"   # sidecar's stored model_path
PY="PYTHONPATH=. python"

# --- WM smoke (8 steps, 1 file, tiny balanced set) → ckpt
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa WANDB_MODE=disabled $PY -m dreamervla.train \
  experiment=oft_world_model_dinowm_chunk task=OpenVLA_Onetraj_LIBERO logger=tensorboard \
  training.out_dir=/tmp/oft_onetraj_wm_smoke training.num_epochs=1 \
  dataloader.batch_size=1 dataloader.num_workers=0 \
  task.hdf5_reward_dir=$RW task.openvla_oft.hdf5_reward_dir=$RW task.openvla_oft.action_hidden_dir=$SC \
  dataset.hdf5_dir=$RW dataset.hidden_dir=$SC dataset.expected_model_path=$MP \
  dataset.max_files=1 dataset.max_demos_per_file=3 dataset.balanced_length=8

# --- classifier smoke (success-only) → ckpt
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa WANDB_MODE=disabled $PY -m dreamervla.train \
  experiment=oft_latent_classifier_chunk task=OpenVLA_Onetraj_LIBERO logger=tensorboard \
  training.out_dir=/tmp/oft_onetraj_cls_smoke training.num_epochs=1 \
  training.batch_size=2 training.num_workers=0 training.episode_eval_enabled=false \
  task.openvla_oft.hdf5_dir=$RW task.openvla_oft.action_hidden_dir=$SC \
  data.success_dir_raw=$RW data.success_dir_hidden=$SC

# --- DreamerVLA wmpo_outcome smoke → ckpt
#  (init from the WM + classifier smoke ckpts; OFT latent is large, so cap the
#   imagined horizon and enable expandable_segments)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa WANDB_MODE=disabled \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m dreamervla.train \
  experiment=dreamervla_oft_dino_wm_wmpo_outcome task=OpenVLA_Onetraj_LIBERO logger=tensorboard \
  training.out_dir=/tmp/oft_onetraj_dvla_smoke training.num_epochs=1 \
  dataloader.batch_size=1 dataloader.num_workers=0 dataloader.multiprocessing_context=null dataloader.persistent_workers=false \
  task.hdf5_reward_dir=$RW task.openvla_oft.hdf5_reward_dir=$RW task.openvla_oft.action_hidden_dir=$SC \
  dataset.hdf5_dir=$RW dataset.hidden_dir=$SC dataset.expected_model_path=$MP \
  dataset.max_files=1 dataset.max_demos_per_file=3 dataset.balanced_length=4 \
  algorithm.wmpo.episode_max_steps=40 algorithm.ppo_rollouts_per_start=2 \
  init.world_model_state_ckpt=/tmp/oft_onetraj_wm_smoke/ckpt/latest.ckpt \
  init.classifier_state_ckpt=/tmp/oft_onetraj_cls_smoke/best_format.ckpt
```

Notes that made the smoke run (all of these are general OFT-route facts, not
hacks):

1. **`dataloader.multiprocessing_context=null`** is required when
   `num_workers=0` (the joint runner does not sanitize the worker kwargs).
2. **Classifier ckpt format.** `LatentClassifierRunner` writes a snapshot
   (`{cfg, state_dicts, pickles}`); `JointDreamerVLARunner` expects the
   best-format `{model, config.classifier, threshold}`. Run the classifier with
   `episode_eval_enabled=true` to emit a best ckpt, or re-wrap the snapshot:
   ```python
   import torch
   from omegaconf import OmegaConf
   d = torch.load("…/checkpoints/latest.ckpt", map_location="cpu", weights_only=False)
   m = {k[7:] if k.startswith("module.") else k: v for k, v in d["state_dicts"]["model"].items()}
   blob = OmegaConf.to_container(d["cfg"].classifier, resolve=True); blob.pop("_target_", None)
   blob["latent_dim"] = int(next(v for k, v in m.items() if k.endswith("input_proj.weight")).shape[1])
   torch.save({"model": m, "config": {"classifier": blob}, "threshold": 0.5}, "…/best_format.ckpt")
   ```
3. **Tokenized-latent flatten (code fix applied).** The OFT world model emits a
   4-D imagined `hidden_seq` `[B, T, 56, 4096]`; `dino_wmpo_outcome_step` and
   `LatentSuccessClassifier.predict_success` assume the flat `[B, T, 229376]`
   contract (matching how the classifier is trained on flat `obs_embedding`).
   `dreamervla/algorithms/ppo/outcome.py` now flattens the trailing token axes
   for any `ndim > 3` latent before scoring (no-op for the already-flat RynnVLA
   latent).

## Known gaps / not covered

- **Online env rollout for OpenVLA-OFT is not wired.** The online trainer
  (`dreamervla/runners/online_dreamervla.py`) builds a `RynnVLAEncoder` only
  (`--action-head-type` is `["legacy"]`). This recipe is the **offline staged**
  pipeline; live OFT rollout-into-replay would need an OFT env-encoder.
- **Backbone-latent (Scheme B) variant.** The analogous input-token route
  (`experiment=oft_world_model_dinowm_chunk_input_tokens`,
  `oft_latent_classifier_chunk_input_tokens`,
  `dreamervla_oft_dino_wm_wmpo_outcome_input_tokens`, policy
  `LatentToActionHiddenActor`) composes but is not smoked here; the OFT
  one-traj checkpoint is discrete (`action_head_ckpt=null`), so its
  `oft_l1_regression` actor head needs an L1 action-head checkpoint.
- The smoke uses tiny budgets and a random-init WM/classifier, so RL advantages
  are zero-variance (groups filtered, `G=0`); it verifies the pipeline runs,
  not convergence.
