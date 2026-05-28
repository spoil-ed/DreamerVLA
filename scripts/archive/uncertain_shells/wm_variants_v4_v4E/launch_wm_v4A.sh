#!/usr/bin/env bash
# v4-A: WM pretrain with larger hidden_decoder (units 2048 → 4096, layers=1).
# Init from merged v3 ckpt. Cut image_decoder, boost hidden_rec to dominate gradient.
# Goal: cosine_loss < 0.10 (vs v3's 0.32).
set +e
cd /mnt/data/spoil/workspace/DreamerVLA
export PATH=/home/user01/miniconda3/envs/dreamervla/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4,5
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHELASTIC_USE_AGENT_STORE=0  # avoid port collision with v4B
export MASTER_PORT=29501

TS=$(date +%Y%m%d_%H%M%S)
RUN=wm_pretrain_legacy_v4A_u4096_gpu45_${TS}
OUT="${OUT_DIR:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/wm_pretrain_legacy_v4/${RUN}}"
WM_INIT=/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/dreamervla_diag/v3_merged_v2base_perwindow_ft/v3_e7_ft_perwindow.ckpt
export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

echo "===== ${RUN} start ====="; date
echo "init from: ${WM_INIT}"
echo "OUT:       ${OUT}"

python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --master-port=29501 -m dreamer_vla.cli.train \
  --config-name dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor \
  training.out_dir="$OUT" \
  training.run_actor_critic_phase=false \
  training.run_wm_phase=true \
  training.num_epochs=10 \
  training.checkpoint_every=1 \
  init.world_model_state_ckpt="$WM_INIT" \
  init.require_encoder_state_ckpt=false \
  world_model.hidden_decoder_units=4096 \
  world_model.hidden_decoder_layers=1 \
  world_model.rec_scale=0.01 \
  world_model.hidden_rec_scale=1000.0 \
  dataloader.batch_size=16 \
  dataloader.num_workers=2 \
  dataset.sequence_length=8 \
  2>&1 | tee -a "$OUT/wm_pretrain.log"
echo "===== exit=${PIPESTATUS[0]} ====="; date
