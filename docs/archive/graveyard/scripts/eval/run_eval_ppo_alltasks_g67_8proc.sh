#!/usr/bin/env bash
# Fast LIBERO-Goal eval for the PPO/DreamerVLA checkpoint on GPUs 6 and 7.
#
# It runs four single-process eval workers per GPU. Each worker loads one copy
# of the model, so this is intended for high-memory GPUs. The eval path must use
# action-query hidden states because this WM expects legacy 35*1024 action
# hidden inputs, not pooled 4096-dim VLA embeddings.
#
# Usage:
#   bash scripts/eval/run_eval_ppo_alltasks_g67_8proc.sh
#
# Useful overrides:
#   CKPT=/path/to/latest.ckpt EPISODES=5 bash scripts/eval/run_eval_ppo_alltasks_g67_8proc.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
CKPT="${CKPT:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/dreamervla/online_wmpo_outcome/alltasks_frozenlegacy_ppo_g67_trace_wmclf_online_replay3k_freezeenc_globalcov_wmrefresh75_long300_chunk5/20260526_194027/checkpoints/latest.ckpt}"
EPISODES="${EPISODES:-10}"
ACTION_STEPS="${ACTION_STEPS:-5}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
BASE="${BASE:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/eval/ppo_alltasks_g67_latest_actionquery_8proc_${TS}}"
LOGDIR="${LOGDIR:-/mnt/data/spoil/workspace/DreamerVLA/data/outputs/logs}"

mkdir -p "$BASE" "$LOGDIR"

launch_eval() {
  local session="$1"
  local gpu="$2"
  local port="$3"
  local tasks="$4"
  local outname="$5"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" -c "$PWD" \
    "PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=osmesa $PYTHON -u -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node=1 --master_port=$port --module src.cli.train \
      --config-name eval_libero_vla \
      training.out_dir=$BASE/$outname \
      eval.ckpt_path=$CKPT \
      eval.ckpt_kind=dreamer \
      eval.task_suite_name=libero_goal \
      'eval.task_ids=$tasks' \
      eval.num_episodes_per_task=$EPISODES \
      eval.action_steps=$ACTION_STEPS \
      eval.save_video=false \
      eval.obs_hidden_source=action_query \
      2>&1 | tee $LOGDIR/${session}_${TS}.log; exec bash"
}

launch_eval eval8_g6_t0  6 29700 '[0]'   task0
launch_eval eval8_g6_t1  6 29701 '[1]'   task1
launch_eval eval8_g6_t2  6 29702 '[2]'   task2
launch_eval eval8_g6_t34 6 29703 '[3,4]' task3_4
launch_eval eval8_g7_t5  7 29704 '[5]'   task5
launch_eval eval8_g7_t6  7 29705 '[6]'   task6
launch_eval eval8_g7_t7  7 29706 '[7]'   task7
launch_eval eval8_g7_t89 7 29707 '[8,9]' task8_9

echo "[eval-ppo-8proc] launched"
echo "  base: $BASE"
echo "  ckpt: $CKPT"
echo "  logs: $LOGDIR/eval8_g{6,7}_*_${TS}.log"
echo "  attach: tmux attach -t eval8_g6_t0"
