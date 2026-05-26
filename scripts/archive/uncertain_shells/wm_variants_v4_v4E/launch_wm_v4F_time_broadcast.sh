#!/usr/bin/env bash
# V4-F: pi0_time_broadcast hidden_decoder.
# Predicts 5 time tokens (T_q=5) and broadcasts across 7 joints; matches the
# data manifold structure documented in docs/hidden_token_structure_report.md
# (same-t residual cosine = 0.996, joint axis is statistically degenerate).
# DDP on GPUs 6+7. Per-rank batch=24, effective batch=48 (matches v4-D baseline).
set +e
cd /mnt/data/spoil/workspace/DreamerVLA
export PATH=/home/user01/miniconda3/envs/dreamervla/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6,7
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
RUN=wm_pretrain_legacy_v4F_time_broadcast_L4d1024m8_Tq5J7_gpu67_${TS}
OUT="${OUT_DIR:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/wm_pretrain_legacy_v4/${RUN}}"
WM_INIT=/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/dreamervla_diag/v3_merged_v2base_perwindow_ft/v3_e7_ft_perwindow.ckpt
export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

echo "===== ${RUN} start ====="; date
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --master-port=29521 -m src.cli.train \
  --config-name dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor \
  training.out_dir="$OUT" \
  training.run_actor_critic_phase=false \
  training.run_wm_phase=true \
  training.num_epochs=8 \
  training.checkpoint_every=1 \
  init.world_model_state_ckpt="$WM_INIT" \
  init.require_encoder_state_ckpt=false \
  world_model.hidden_decoder_kind=pi0_time_broadcast \
  world_model.hidden_decoder_layers=4 \
  world_model.hidden_decoder_units=0 \
  world_model.hidden_decoder_d_model=1024 \
  world_model.hidden_decoder_nhead=8 \
  world_model.hidden_decoder_mem_tokens=8 \
  world_model.hidden_decoder_n_time_queries=5 \
  world_model.hidden_decoder_joint_broadcast=7 \
  world_model.hidden_decoder_dropout=0.0 \
  world_model.rec_scale=0.01 \
  world_model.hidden_rec_scale=1000.0 \
  dataloader.batch_size=24 \
  dataloader.num_workers=4 \
  dataset.sequence_length=8 \
  2>&1 | tee -a "$OUT/wm_pretrain.log"
echo "===== exit=${PIPESTATUS[0]} ====="; date
