#!/usr/bin/env bash
# V4-H: pi0_time_broadcast scaled ×7 in per-query capacity.
# Same architecture as v4-F (5 time queries + joint broadcast), but each query
# now has ~7× the transformer capacity. Tests whether the cos=0.92 plateau
# seen in v4-F is a property of the broadcast structure (residual joint info
# we threw away) or a property of decoder capacity within that structure.
#
#   kind        pi0_time_broadcast  (5 time queries × broadcast across 7 joints)
#   d_model     1024 -> 2688  (= floor(sqrt(7)*1024); 2688/nhead=8 -> head=336)
#                              transformer params ~ 12·d^2·L grows by 6.89×
#   token_dim   1024  (NEW: per-query output width, decoupled from d_model;
#                       preserves out_dim = 5*7*1024 = 35840)
#   mem_tokens  8     (unchanged)
#   layers      4     (unchanged)
#   n_time_q    5     (unchanged)
#   joint_bcast 7     (unchanged, hard tie)
#
# Comparison matrix at epoch 7 (hidden_rec_scale=1000):
#   v4-D pi0xform (35 queries, d=1024, no broadcast)   hidden_rec=0.1695
#   v4-F broadcast (5 queries, d=1024,    broadcast)   hidden_rec=0.1597
#   v4-G x7 (35 queries, d=2688, no broadcast)         hidden_rec=?  (running on 6+7)
#   v4-H x7 (5 queries,  d=2688,    broadcast)         hidden_rec=?  (this run)
#
# DDP on GPUs 4+5. Per-rank batch=24, effective batch=48.
set +e
cd /mnt/data/spoil/workspace/DreamerVLA
export PATH=/home/user01/miniconda3/envs/dreamervla/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4,5
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
RUN=wm_pretrain_legacy_v4H_time_broadcast_x7_L4d2688tok1024_gpu45_${TS}
OUT="${OUT_DIR:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/wm_pretrain_legacy_v4/${RUN}}"
WM_INIT=/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/dreamervla_diag/v3_merged_v2base_perwindow_ft/v3_e7_ft_perwindow.ckpt
export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

echo "===== ${RUN} start ====="; date
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --master-port=29541 -m src.cli.train \
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
  world_model.hidden_decoder_d_model=2688 \
  world_model.hidden_decoder_token_dim=1024 \
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
