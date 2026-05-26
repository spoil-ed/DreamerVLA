#!/usr/bin/env bash
# V4-I: current legacy 35-token target with overcapacity RSSM and ResNet decoder.
#
# Keeps output fixed at 35 * 1024 = 35840, then increases:
#   RSSM feature: 8192 + 32*64 = 10240 -> 12288 + 48*64 = 15360
#                 h:z ratio stays 8192:2048 = 12288:3072 = 4:1
#   decoder: ResMLP L6×8192 -> L8×8192
#
# Defaults to the latest 35840 ResMLP L6 checkpoint; override WM_INIT to use
# another compatible checkpoint or set WM_INIT=null to train from random init.
set +e
cd /mnt/data/spoil/workspace/DreamerVLA
export PATH=/home/user01/miniconda3/envs/dreamervla/bin:$PATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MUJOCO_GL=osmesa
unset MUJOCO_EGL_DEVICE_ID
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
RUN=${RUN_TAG:-wm_pretrain_legacy_v4I_overdim_d12288s48_resnet_L8u8192_${TS}}
OUT="${OUT_DIR:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/wm_pretrain_legacy_v4/${RUN}}"
WM_INIT_DEFAULT=/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/dreamervla_diag/wm_pretrain_legacy_v4D_resnet_L6u8192_gpu7_20260521_154718/ckpt/latest.ckpt
WM_INIT=${WM_INIT:-$WM_INIT_DEFAULT}
NUM_GPUS=${NUM_GPUS:-2}
MASTER_PORT=${MASTER_PORT:-29551}
# Per-GPU batch size. With the default 2 GPUs this gives an effective batch 192.
BATCH_SIZE=${BATCH_SIZE:-96}
NUM_WORKERS=${NUM_WORKERS:-4}
SEQ_LEN=${SEQ_LEN:-8}

export MPLCONFIGDIR=/tmp/matplotlib-${RUN}
mkdir -p "$MPLCONFIGDIR" "$OUT"

INIT_OVERRIDES=()
if [[ -n "$WM_INIT" && "$WM_INIT" != "null" ]]; then
  INIT_OVERRIDES+=("init.world_model_state_ckpt=$WM_INIT")
fi

echo "===== ${RUN} start ====="; date
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NUM_GPUS" --master-port="$MASTER_PORT" -m src.cli.train \
  --config-name dreamer_vla_libero_goal_pi0_legacy_action_hidden_head_actor_v4i_overdim \
  training.out_dir="$OUT" \
  dataloader.batch_size="$BATCH_SIZE" \
  dataloader.num_workers="$NUM_WORKERS" \
  dataset.sequence_length="$SEQ_LEN" \
  "${INIT_OVERRIDES[@]}" \
  "$@" \
  2>&1 | tee -a "$OUT/wm_pretrain.log"
echo "===== exit=${PIPESTATUS[0]} ====="; date
