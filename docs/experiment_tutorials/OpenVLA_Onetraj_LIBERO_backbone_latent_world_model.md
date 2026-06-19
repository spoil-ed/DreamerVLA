# OpenVLA_Onetraj_LIBERO — Backbone-Latent World Model (Scheme 1)

WM placed **before the Action Query**. Prediction target = **future backbone /
DINO-style visual-language latent** (OpenVLA-OFT current-frame projected vision
patch tokens, `[512, 4096]`; flat `512*4096`). The actor maps a predicted future
backbone latent back to an action via learned **Action Queries + a
TransformerDecoder bridge + the frozen OpenVLA LM head** —
`LatentToOpenVLADiscreteTokenActor`. This route stays discrete end-to-end; no L1
action head is constructed or loaded.

This is a **policy/representation latent world model**, not the original
pixel/observation DINO-WM.

## World Model architecture (latest, 2026-06-19)

Same DINO-WM concat conditioning and autoregressive recursion as the action-hidden
route (`num_hist=3` sliding window; predicted latents feed back, by step 4 all
three inputs are predictions), but on the **backbone / input-token** latent:
`token_count=512`, `token_dim=4096`, `model_dim = 4096 + 10 = 4106`.

- **Predictor sizing (lean-debottlenecked, ~313M):** `depth=6, heads=16,
  dim_head=128` (inner = 2048 = 0.5×`model_dim`, 8× the old compact attention
  inner=256), lean `mlp_dim=2048`. The 1536-token sequence
  (`num_hist*token_count`) is ~9× the action-hidden route, so capacity is kept
  lean here for efficiency. Under the 1B cap. Values resolve from
  `configs/worldmodel/openvla_oft_input_token_chunk.yaml` /
  `configs/dreamervla/openvla_oft_input_token_wmpo_outcome.yaml`.
- This is the most faithful migration of the DINO-WM paradigm (latent taken
  *before* action conditioning, like DINO patch tokens), and is the scheme used by
  the offline WM upper-bound probe.
- **Online** rollout for this scheme is **now wired**: `OnlineCotrainRunner`
  uses `OFTRolloutHiddenExtractor(obs_hidden_source="input_token_embedding")`
  to produce the online counterpart of the offline OFT input-token sidecar, and
  the actor is `LatentToOpenVLADiscreteTokenActor` (discrete LM-head bridge,
  **no L1 head**). The offline path below is still what the WM ceiling probe
  uses.

## Modules (all reused, `dreamervla.*`)

| Capability | Where |
| --- | --- |
| backbone latent extractor | preprocess `scripts/preprocess/32_input_token_hidden.sh` / OFT `35_oft_action_hidden.sh OFT_LATENT_SCHEME=input_tokens` → `obs_embedding` sidecar |
| backbone latent → action adapter | `dreamervla.models.actor.LatentToOpenVLADiscreteTokenActor` (`_source_tokens` validates shape; `source_proj` → `action_queries` + `TransformerDecoder` bridge → `action_hidden_proj` → frozen OpenVLA LM head) |
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
      → LatentToOpenVLADiscreteTokenActor:
          source_proj → Action Queries + TransformerDecoder bridge
          → action_hidden → OpenVLA LM-head action-token logits → action [T, 7]
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

# 4. DreamerVLA actor (discrete bridge over backbone latent)
bash scripts/train_dreamervla.sh experiment=dreamervla_oft_dino_wm_wmpo_outcome_input_tokens \
  task=OpenVLA_Onetraj_LIBERO gpus=0 ngpu=1 batch_size=2 -- \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT}/outputs/classifier/<run>/checkpoints/latest.ckpt"
```

## Unified online cotrain (warmup → cotrain)

The unified online cotrain runner `OnlineCotrainRunner` implements:
one-traj VLA → parallel rollout (1 env/rank) → replay → **warmup (WM+classifier
only, `training.warmup_steps`)** → **cotrain (WM+classifier + slow-policy RL)**.

```bash
python -m dreamervla.train experiment=online_cotrain_oft_backbone_latent
WANDB_MODE=disabled python -m dreamervla.train \
  experiment=online_cotrain_oft_backbone_latent training.debug=true
```

**Online backbone rollout is wired (no longer a gap):** `OnlineCotrainRunner`
handles `latent_type=backbone_latent` by extracting the pre-Action-Query
input-token latent via `OFTRolloutHiddenExtractor` with
`obs_hidden_source=input_token_embedding`, and by setting the rollout env to the
same input-token contract.

**Discrete actor (no L1 head):** the backbone-latent route uses
`LatentToOpenVLADiscreteTokenActor` — Action-Queries + a TransformerDecoder bridge
produce action-hidden slots, then the OpenVLA **LM-head categorical** action-token
decoder maps them to actions (`init_lm_head_ckpt=<discrete oft ckpt>`,
`head_type=oft_discrete_token`). This matches the discrete VLA; no `action_head--*.pt`
L1 component is needed. (The older L1 adapter `LatentToActionHiddenActor`
[`head_type=oft_l1_regression`] still exists for L1 checkpoints but is not used on
this discrete route.)

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
- action_hidden slots (bridge output): `[B, time_horizon*7, 4096]`
- action token ids: `[B, time_horizon, 7]`
- action: `[B, time_horizon, 7]` (`_source_tokens` raises on shape mismatch)
