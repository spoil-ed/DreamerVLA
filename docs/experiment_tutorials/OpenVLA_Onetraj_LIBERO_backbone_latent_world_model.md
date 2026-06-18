# OpenVLA_Onetraj_LIBERO — Backbone-Latent World Model (Scheme 1)

WM placed **before the Action Query**. Prediction target = **future backbone /
DINO-style visual-language latent** (OpenVLA-OFT current-frame projected vision
patch tokens, `[512, 4096]`; flat `512*4096`). The actor maps a predicted future
backbone latent back to an action via learned **Action Queries + a
TransformerDecoder bridge + the frozen action head** — `LatentToActionHiddenActor`
(no hard-coding into the base policy; it is a clean adapter).

This is a **policy/representation latent world model**, not the original
pixel/observation DINO-WM.

## Modules (all reused, `dreamervla.*`)

| Capability | Where |
| --- | --- |
| backbone latent extractor | preprocess `scripts/preprocess/32_input_token_hidden.sh` / OFT `35_oft_action_hidden.sh OFT_LATENT_SCHEME=input_tokens` → `obs_embedding` sidecar |
| backbone latent → action adapter | `dreamervla.models.actor.LatentToActionHiddenActor` (`_source_tokens` validates shape; `source_proj` → `action_queries` + `TransformerDecoder` bridge → `action_hidden_proj` → frozen action head) |
| world model (horizon rollout) | `dreamervla.models.world_model.dino_wm_chunk.ChunkAwareDinoWMWorldModel` (latent-agnostic; `token_count=512, token_dim=4096`) |
| classifier | `dreamervla.models.reward.LatentSuccessClassifier` |
| replay buffer | `dreamervla.runners.online_replay.OnlineReplay` |
| online cotrain runner | `dreamervla.runners.OnlineCotrainRunner` |

## Core data flow

```
preprocess input-token sidecar  (obs_embedding = [T, 512*4096] current-frame vision patches)
  → ChunkAwareDinoWMWorldModel predicts FUTURE backbone latent (horizon rollout)
  → LatentSuccessClassifier scores backbone-latent windows
  → predicted future backbone latent
      → LatentToActionHiddenActor:  source_proj → Action Queries + TransformerDecoder bridge
                                    → action_hidden → frozen action head → action [T, 7]
```

## Runnable path today — OFFLINE OFT input-token (one Hydra entry per stage)

`OpenVLA_Onetraj_LIBERO` task; data under
`${DVLA_DATA_ROOT}/processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/`.

```bash
cd DreamerVLA
export DVLA_DATA_ROOT="$(pwd -P)/data"; export MUJOCO_GL=osmesa; conda activate dreamervla

# 1. reward HDF5 + OFT input-token (backbone) sidecar
bash scripts/preprocess/prepare_libero_data.sh task=OpenVLA_Onetraj_LIBERO \
  libero_suite=libero_goal only=[10_hdf5_reward] gpus=0 ngpu=1
OFT_CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
bash scripts/preprocess/prepare_libero_data.sh task=OpenVLA_Onetraj_LIBERO \
  libero_suite=libero_goal only=[35_oft_action_hidden] gpus=0 ngpu=1 \
  env.OFT_LATENT_SCHEME=input_tokens env.OFT_POLICY_MODE=discrete env.OFT_HISTORY=2 \
  env.OFT_CKPT="${OFT_CKPT}"

# 2. world model (backbone/input-token latent)
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk_input_tokens \
  task=OpenVLA_Onetraj_LIBERO gpus=0 ngpu=1 batch_size=2

# 3. classifier
bash scripts/train_wm.sh experiment=oft_latent_classifier_chunk_input_tokens \
  task=OpenVLA_Onetraj_LIBERO gpus=0 batch_size=8

# 4. DreamerVLA actor (LatentToActionHiddenActor over backbone latent)
bash scripts/train_dreamervla.sh experiment=dreamervla_oft_dino_wm_wmpo_outcome_input_tokens \
  task=OpenVLA_Onetraj_LIBERO gpus=0 ngpu=1 batch_size=2 -- \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT}/outputs/classifier/<run>/checkpoints/latest.ckpt"
```

## Unified online cotrain (warmup → cotrain) and the current gap

The unified online cotrain runner `OnlineCotrainRunner` implements:
one-traj VLA → parallel rollout (1 env/rank) → replay → **warmup (WM+classifier
only, `training.warmup_steps`)** → **cotrain (WM+classifier + slow-policy RL)**.

```bash
python -m dreamervla.train experiment=online_cotrain_oft_backbone_latent
WANDB_MODE=disabled python -m dreamervla.train \
  experiment=online_cotrain_oft_backbone_latent training.debug=true
```

**KNOWN GAP (clear error, no silent fail):** online env rollout for
`backbone_latent` is **not wired** — `DreamerVLAOnlineTrainEnv.obs_hidden_source`
is `Literal["action_query"]` (it only emits the action-hidden latent). So
`OnlineCotrainRunner` **raises a clear `NotImplementedError`** for
`latent_type=backbone_latent`, pointing to the offline path above.
**Next step to enable online backbone rollout:** add an input-token obs source to
the env/encoder (an `obs_to_input_token` analogous to
`dreamervla.runners.online_utils.obs_to_action_hidden`, plus an
`obs_hidden_source="input_token_embedding"` branch in `DreamerVLAOnlineTrainEnv`).

**Discrete OFT checkpoint fallback:** the downloaded OFT one-trajectory checkpoint
is discrete and has no L1 action head. `LatentToActionHiddenActor`
(`head_type=oft_l1_regression`, `init_action_head_ckpt=<oft ckpt>`) only loads the
output projection when present; with a discrete ckpt it logs how many tensors
matched (`missing/unexpected`) and leaves a randomly-initialised (frozen) head —
supply a real OFT L1 `action_head--*.pt` to make the actor head meaningful.

## Config knobs (`configs/experiment/online_cotrain_oft_backbone_latent.yaml`)

`latent_type` · `warmup_steps` · `train_actor_after_warmup` ·
`train_classifier_inline` · `online_rollout.{buffer_size, sequence_length,
min_replay, train_every, rollout_policy_source, debug_*}` ·
`world_model.{token_count=512, token_dim=4096, obs_dim, chunk_size}` ·
`optim.{world_model, policy(slow 5e-7), critic, classifier}` ·
`init.{vla_ckpt_path, world_model_state_ckpt, classifier_state_ckpt}`.

## Key shapes (printed/validated)

- backbone latent (`obs_embedding`): `[T, 512*4096]` (current-frame projected vision patches)
- predicted future latent: same per-frame dim, `[B, T, 512*4096]`
- action_hidden (bridge output): `[B, time_horizon, action_hidden_dim]`
- action: `[B, time_horizon, 7]` (`_source_tokens` raises on shape mismatch)
