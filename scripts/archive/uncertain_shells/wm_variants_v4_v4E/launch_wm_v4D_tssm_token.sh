#!/usr/bin/env bash
# V4: TSSM (Transformer State Space Model) replacing RSSM. Train from scratch (no compatible v3 ckpt).
# Uses dedicated cfg file directly.
set +e
cd /mnt/data/spoil/workspace/DreamerVLA
export PATH=/home/user01/miniconda3/envs/dreamervla/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${TSSM_GPUS:-${TSSM_GPU:-5}}
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
RUN_GPU_TAG=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '-')
RUN=wm_pretrain_legacy_v4D_tssm_token_concatO_L6d600w8_gpu${RUN_GPU_TAG}_${TS}
OUT="${OUT_DIR:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/wm_pretrain_legacy_v4/${RUN}}"
export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

echo "===== ${RUN} start ====="; date
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=${TSSM_NPROC:-1} --master-port=${TSSM_PORT:-29514} -m src.cli.train \
  --config-name dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor_v4d_tssm_token \
  training.out_dir="$OUT" \
  training.run_actor_critic_phase=false \
  training.run_wm_phase=true \
  training.num_epochs=8 \
  training.checkpoint_every=1 \
  init.world_model_state_ckpt=null \
  init.require_encoder_state_ckpt=false \
  world_model.rec_scale=0.01 \
  world_model.hidden_rec_scale=1000.0 \
  dataloader.batch_size=${TSSM_BATCH_SIZE:-8} \
  dataloader.num_workers=${TSSM_NUM_WORKERS:-4} \
  dataloader.prefetch_factor=${TSSM_PREFETCH_FACTOR:-2} \
  dataset.sequence_length=8 \
  2>&1 | tee -a "$OUT/wm_pretrain.log"
echo "===== exit=${PIPESTATUS[0]} ====="; date
