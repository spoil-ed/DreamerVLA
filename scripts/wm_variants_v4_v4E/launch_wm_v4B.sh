#!/usr/bin/env bash
# v4-B: WM pretrain with bigger hidden_decoder (units 2048 → 8192, layers 1 → 2).
# Init from merged v3 ckpt. Same image_decoder/hidden_rec rescaling as v4-A.
# Goal: cosine_loss < 0.05.
set +e
cd /home/user01/liops/workspace/DreamerVLA
export PATH=/home/user01/miniconda3/envs/dreamervla/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6,7
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=29502

TS=$(date +%Y%m%d_%H%M%S)
RUN=wm_pretrain_legacy_v4B_u8192L2_gpu67_${TS}
OUT=/home/user01/liops/workspace/DreamerVLA/data/outputs/dreamervla_diag/${RUN}
WM_INIT=/home/user01/liops/workspace/DreamerVLA/data/outputs/dreamervla_diag/v3_merged_v2base_perwindow_ft/v3_e7_ft_perwindow.ckpt
export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

echo "===== ${RUN} start ====="; date
echo "init from: ${WM_INIT}"
echo "OUT:       ${OUT}"

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --master-port=29502 -m src.cli.train \
  --config-name dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor \
  training.out_dir="$OUT" \
  training.run_actor_critic_phase=false \
  training.run_wm_phase=true \
  training.num_epochs=10 \
  training.checkpoint_every=1 \
  init.world_model_state_ckpt="$WM_INIT" \
  init.require_encoder_state_ckpt=false \
  world_model.hidden_decoder_units=8192 \
  world_model.hidden_decoder_layers=2 \
  world_model.rec_scale=0.01 \
  world_model.hidden_rec_scale=1000.0 \
  dataloader.batch_size=12 \
  dataloader.num_workers=2 \
  dataset.sequence_length=8 \
  2>&1 | tee -a "$OUT/wm_pretrain.log"
echo "===== exit=${PIPESTATUS[0]} ====="; date
