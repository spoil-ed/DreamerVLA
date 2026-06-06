#!/usr/bin/env bash
# Launch online WMPO outcome PPO for all LIBERO-Goal tasks.
#
# Default run:
#   bash scripts/run_online_dreamervla_wmpo_alltasks_g67.sh
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=4,5 MASTER_PORT=29518 \
#   TOTAL_ENV_STEPS=10000 MAX_TRAIN_UPDATES=2200 \
#   bash scripts/run_online_dreamervla_wmpo_alltasks_g67.sh
#
# Attach:
#   tmux attach -t ppo_alltasks_trace_wmclf_g67
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${DVLA_ROOT}"

SESSION="${SESSION:-ppo_alltasks_trace_wmclf_g67}"
MASTER_PORT="${MASTER_PORT:-29519}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
NGPU="${NGPU:-2}"

CONFIG="${CONFIG:-${DVLA_ROOT}/configs/online_wmpo_outcome_libero_goal.yaml}"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-${DVLA_ROOT}/data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/step_00015000.ckpt}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${DVLA_ROOT}/data/outputs/dreamervla/outcome_classifier/libero_goal/wmpo_aligned_small_tf_chunk_minsteps32/ckpt/best_episode_f10.9705_th0.67.ckpt}"
VLA_CKPT_PATH="${VLA_CKPT_PATH:-${DVLA_ROOT}/data/ckpts/VLA_model_256/libero_goal}"

TASK_SUITE="${TASK_SUITE:-libero_goal}"
TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
EPISODE_HORIZON="${EPISODE_HORIZON:-300}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-10000}"
MAX_TRAIN_UPDATES="${MAX_TRAIN_UPDATES:-2200}"
TRAIN_RATIO="${TRAIN_RATIO:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
REPLAY_SIZE="${REPLAY_SIZE:-3000}"
MIN_REPLAY="${MIN_REPLAY:-64}"
MIN_EPISODES_PER_TASK="${MIN_EPISODES_PER_TASK:-1}"
WM_REFRESH_UPDATES_BEFORE_PPO="${WM_REFRESH_UPDATES_BEFORE_PPO:-75}"
SAVE_EVERY="${SAVE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-10}"
BC_TO_REF="${BC_TO_REF:-0.1}"
COLLECT_CHUNK_STEPS="${COLLECT_CHUNK_STEPS:-5}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${DVLA_ROOT}/data/outputs/dreamervla/online_wmpo_outcome/alltasks_frozenlegacy_ppo_g67_trace_wmclf_online_replay3k_freezeenc_globalcov_wmrefresh75_long300_chunk5/${TS}}"
LOG_FILE="${LOG_FILE:-${DVLA_ROOT}/data/outputs/logs/ppo_alltasks_trace_wmclf_replay3k_freezeenc_globalcov_chunk5_g67_${TS}.log}"

mkdir -p "$(dirname "$LOG_FILE")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[run_online_wmpo_alltasks] tmux session already exists: $SESSION"
  echo "  attach: tmux attach -t $SESSION"
  exit 1
fi

echo "[run_online_wmpo_alltasks] session=$SESSION"
echo "[run_online_wmpo_alltasks] gpus=$CUDA_VISIBLE_DEVICES ngpu=$NGPU master_port=$MASTER_PORT"
echo "[run_online_wmpo_alltasks] out_dir=$OUT_DIR"
echo "[run_online_wmpo_alltasks] log_file=$LOG_FILE"
echo "[run_online_wmpo_alltasks] task_ids=$TASK_IDS total_env_steps=$TOTAL_ENV_STEPS max_train_updates=$MAX_TRAIN_UPDATES collect_chunk_steps=$COLLECT_CHUNK_STEPS"

tmux new-session -d -s "$SESSION" -c "${DVLA_ROOT}" \
  "PYTHONPATH=$DVLA_ROOT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MUJOCO_GL=$MUJOCO_GL $PYTHON -m torch.distributed.run \
  --standalone --nproc_per_node=$NGPU --master_port=$MASTER_PORT \
  scripts/training/train_online_rynnvla_action_hidden_dreamervla.py \
  --config $CONFIG \
  --out-dir $OUT_DIR \
  --world-model-ckpt $WORLD_MODEL_CKPT \
  --classifier-ckpt $CLASSIFIER_CKPT \
  --vla-ckpt-path $VLA_CKPT_PATH \
  --encoder-state-ckpt '' \
  --action-head-type legacy \
  --task-suite $TASK_SUITE \
  --task-ids $TASK_IDS \
  --episode-horizon $EPISODE_HORIZON \
  --total-env-steps $TOTAL_ENV_STEPS \
  --max-train-updates $MAX_TRAIN_UPDATES \
  --train-ratio $TRAIN_RATIO \
  --batch-size $BATCH_SIZE \
  --replay-size $REPLAY_SIZE \
  --min-replay $MIN_REPLAY \
  --min-episodes-per-task $MIN_EPISODES_PER_TASK \
  --global-coverage-train-start \
  --task-balanced-replay \
  --replay-capacity-mode per_task \
  --failure-prefix-steps 40 \
  --failure-prefix-ratio 0.2 \
  --wm-refresh-updates-before-ppo $WM_REFRESH_UPDATES_BEFORE_PPO \
  --no-freeze-wm-after-refresh \
  --freeze-wm-encoder \
  --update-classifier-online \
  --actor-update-kind outcome \
  --collect-chunk-steps $COLLECT_CHUNK_STEPS \
  --log-every $LOG_EVERY \
  --save-every $SAVE_EVERY \
  --rssm-action-scale env \
  --bc-to-ref $BC_TO_REF \
  --freeze-log-std \
  2>&1 | tee $LOG_FILE; exec bash"

echo "[run_online_wmpo_alltasks] launched."
echo "  attach: tmux attach -t $SESSION"
echo "  tail:   tail -f $LOG_FILE"
