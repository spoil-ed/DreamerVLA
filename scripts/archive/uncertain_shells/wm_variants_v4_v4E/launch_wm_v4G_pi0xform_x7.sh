#!/usr/bin/env bash
# V4-G: pi0_transformer with per-query capacity ×7. Tests whether the
# cos=0.92 plateau seen in v4-D / v4-F is decoder-capacity-bound or data-bound.
#   d_model     1024 -> 2688  (=floor(sqrt(7)*1024); 2688/nhead=8 -> head=336)
#                              transformer params ~ 12·d^2·L grows by 6.89×
#   token_dim   1024 -> 1024  (NEW arg; decouples query output width from
#                              d_model so num_queries=35 is preserved while
#                              d_model can be any value)
#   mem_tokens  8    -> 8     (unchanged; mem×d_model in feat_proj is NOT
#                              "per-query capacity", scaling it would just
#                              blow up the input projection)
#   layers      4    -> 4     (unchanged, avoid depth instability)
#   n_queries   35   -> 35    (no broadcast; one query per output token)
# DDP on GPUs 6+7. Per-rank batch=24, effective batch=48 (matches v4-F/v4-D).
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
RUN=wm_pretrain_legacy_v4G_pi0xform_x7_L4d2688tok1024_gpu67_${TS}
OUT="${OUT_DIR:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/wm_pretrain_legacy_v4/${RUN}}"
WM_INIT=/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/dreamervla_diag/v3_merged_v2base_perwindow_ft/v3_e7_ft_perwindow.ckpt
export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

echo "===== ${RUN} start ====="; date
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 --master-port=29531 -m dreamer_vla.cli.train \
  --config-name dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor \
  training.out_dir="$OUT" \
  training.run_actor_critic_phase=false \
  training.run_wm_phase=true \
  training.num_epochs=8 \
  training.checkpoint_every=1 \
  init.world_model_state_ckpt="$WM_INIT" \
  init.require_encoder_state_ckpt=false \
  world_model.hidden_decoder_kind=pi0_transformer \
  world_model.hidden_decoder_layers=4 \
  world_model.hidden_decoder_units=0 \
  world_model.hidden_decoder_d_model=2688 \
  world_model.hidden_decoder_token_dim=1024 \
  world_model.hidden_decoder_nhead=8 \
  world_model.hidden_decoder_mem_tokens=8 \
  world_model.hidden_decoder_dropout=0.0 \
  world_model.rec_scale=0.01 \
  world_model.hidden_rec_scale=1000.0 \
  dataloader.batch_size=24 \
  dataloader.num_workers=4 \
  dataset.sequence_length=8 \
  2>&1 | tee -a "$OUT/wm_pretrain.log"
echo "===== exit=${PIPESTATUS[0]} ====="; date
