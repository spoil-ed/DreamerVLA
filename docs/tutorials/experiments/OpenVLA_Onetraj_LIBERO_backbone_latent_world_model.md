# OpenVLA-OFT one-trajectory — backbone-latent WM (Scheme 1)

Background (Scheme 1 vs Scheme A, WM sizing, online wiring, memory): [EXPLAINED.md](EXPLAINED.md) ·
parameters: [../../PARAMETERS.md](../../PARAMETERS.md).

## Offline staged path

```bash
cd DreamerVLA
export DVLA_DATA_ROOT="$(pwd -P)/data"; export MUJOCO_GL=osmesa; conda activate dreamervla

# 1. reward HDF5 + OFT input-token (backbone) sidecar
bash scripts/preprocess/prepare_libero_data.sh task=openvla_onetraj_libero \
  libero_suite=libero_goal only=[10_hdf5_reward] gpus=0 ngpu=1
OFT_CKPT="${DVLA_DATA_ROOT}/checkpoints/Openvla-oft-SFT-traj1/Openvla-oft-SFT-libero-goal-traj1"
bash scripts/preprocess/prepare_libero_data.sh task=openvla_onetraj_libero \
  libero_suite=libero_goal only=[35_oft_action_hidden] gpus=0 ngpu=1 \
  env.OFT_LATENT_SCHEME=input_tokens env.OFT_POLICY_MODE=discrete env.OFT_HISTORY=2 \
  env.OFT_CKPT="${OFT_CKPT}"

# 2. world model (backbone/input-token latent)
bash scripts/train_wm.sh experiment=oft_world_model_dinowm_chunk_input_tokens \
  task=openvla_onetraj_libero gpus=0 ngpu=1 batch_size=2

# 3. classifier
bash scripts/train_wm.sh experiment=oft_latent_classifier_chunk_input_tokens \
  task=openvla_onetraj_libero gpus=0 batch_size=8

# 4. DreamerVLA actor (discrete bridge over backbone latent)
bash scripts/train_dreamervla.sh experiment=dreamervla_oft_dino_wm_lumos_input_tokens \
  task=openvla_onetraj_libero gpus=0 ngpu=1 batch_size=2 -- \
  init.world_model_state_ckpt="${DVLA_DATA_ROOT}/outputs/worldmodel/<run>/checkpoints/latest.ckpt" \
  init.classifier_state_ckpt="${DVLA_DATA_ROOT}/outputs/classifier/<run>/checkpoints/latest.ckpt"
```

## Unified online cotrain

```bash
python -m dreamervla.train experiment=online_cotrain_oft_backbone_latent
# dry-run
WANDB_MODE=disabled python -m dreamervla.train \
  experiment=online_cotrain_oft_backbone_latent training.debug=true
```
