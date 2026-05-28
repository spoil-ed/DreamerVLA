#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
MASTER_PORT="${MASTER_PORT:-29531}"
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

WATCH_LOG="${WATCH_LOG:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/wait_g45_multienv_smoke_${TS}.log}"
SMOKE_LOG="${SMOKE_LOG:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs/smoke_multienv_g45_${TS}.log}"
SMOKE_OUT="${SMOKE_OUT:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/dreamervla/smoke_multienv_chunk5_g45/${TS}}"

CONFIG="${CONFIG:-/mnt/data/spoil/workspace/DreamerVLA/configs/online_wmpo_outcome_libero_goal.yaml}"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/step_00015000.ckpt}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/dreamervla/outcome_classifier/libero_goal/wmpo_aligned_small_tf_chunk_minsteps32/ckpt/best_episode_f10.9705_th0.67.ckpt}"
VLA_CKPT_PATH="${VLA_CKPT_PATH:-/mnt/data/spoil/workspace/DreamerVLA/data/ckpts/frozen_backbones/rynnvla_libero_goal_pi0_query/base_model}"

mkdir -p "$(dirname "$WATCH_LOG")" "$(dirname "$SMOKE_OUT")"

echo "[wait-g45] watch_log=$WATCH_LOG"
echo "[wait-g45] smoke_log=$SMOKE_LOG"
echo "[wait-g45] smoke_out=$SMOKE_OUT"

while true; do
  g4="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 | tr -d ' ')"
  g5="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 | tr -d ' ')"
  echo "$(date +%F_%T) gpu4_mem=${g4} gpu5_mem=${g5}" | tee -a "$WATCH_LOG"
  if [ "$g4" -lt "$MEM_THRESHOLD_MB" ] && [ "$g5" -lt "$MEM_THRESHOLD_MB" ]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "$(date +%F_%T) GPUs 4/5 free; launching multi-env smoke" | tee -a "$WATCH_LOG"
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=4,5 MUJOCO_GL=osmesa \
  "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 --master_port="$MASTER_PORT" \
  scripts/training/train_online_pi0_action_hidden_dreamervla_multienv.py \
  --config "$CONFIG" \
  --out-dir "$SMOKE_OUT" \
  --world-model-ckpt "$WORLD_MODEL_CKPT" \
  --classifier-ckpt "$CLASSIFIER_CKPT" \
  --vla-ckpt-path "$VLA_CKPT_PATH" \
  --encoder-state-ckpt "" \
  --action-head-type legacy \
  --task-suite libero_goal \
  --task-ids 0,1,2,3 \
  --episode-horizon 40 \
  --total-env-steps 90 \
  --train-ratio 0 \
  --batch-size 2 \
  --replay-size 200 \
  --min-replay 999999 \
  --min-episodes-per-task 0 \
  --global-coverage-train-start \
  --task-balanced-replay \
  --replay-capacity-mode per_task \
  --wm-refresh-updates-before-ppo 0 \
  --actor-update-kind outcome \
  --collect-chunk-steps 5 \
  --num-envs-per-rank 2 \
  --log-every 10 \
  --save-every 100 \
  --rssm-action-scale env \
  --bc-to-ref 0.1 \
  --freeze-log-std \
  2>&1 | tee "$SMOKE_LOG"
