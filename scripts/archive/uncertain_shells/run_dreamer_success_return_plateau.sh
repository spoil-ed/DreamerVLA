#!/usr/bin/env bash
# Continue the DINO-WM Dreamer actor-critic success-to-go auxiliary run until
# the critic/return curves plateau.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
source "${SCRIPT_DIR}/lib/output_layout.sh"

OUT_DIR="${OUT_DIR:-$(output_layout_path "${PROJECT_ROOT}" dreamervla actor_critic_ppo success_return_s025_aux16_init0 gpu45_plateau)}"
CHUNK_STEPS="${CHUNK_STEPS:-5000}"
MAX_CHUNKS="${MAX_CHUNKS:-20}"
WINDOW="${WINDOW:-500}"
PATIENCE="${PATIENCE:-3}"
MIN_REL_IMPROVE="${MIN_REL_IMPROVE:-0.02}"
RETURN_REL_IMPROVE="${RETURN_REL_IMPROVE:-0.02}"

export PATH="/home/user01/miniconda3/envs/dreamervla/bin:${PATH}"
export PYTHON="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CONFIG_NAME="${CONFIG_NAME:-dreamer_vla_libero_goal_rynn_dino_wm_actor_critic}"
export RUN_TAG="${RUN_TAG:-success_return_s025_aux16_init0_plateau}"
export OUT_DIR
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

LOG_PATH="${OUT_DIR}/dreamer_vla_logs.json.txt"
STATE_PATH="${OUT_DIR}/plateau_state.json"
STATUS_PATH="${OUT_DIR}/plateau_status.jsonl"

mkdir -p "${OUT_DIR}"

restart_hidden_watchers() {
  if [[ "${RESTART_HIDDEN_WATCHERS:-1}" != "1" ]]; then
    return 0
  fi
  if command -v tmux >/dev/null 2>&1; then
    if ! tmux has-session -t hidden_spatial_when_ready_0523 2>/dev/null; then
      tmux new-session -d -s hidden_spatial_when_ready_0523 \
        "cd ${PROJECT_ROOT} && LIBERO_TASK_SUITE=libero_spatial GPU=4 POLL_SECONDS=120 bash scripts/wait_and_run_libero_hidden.sh" \
        || true
    fi
    if ! tmux has-session -t hidden_object_when_ready_0523 2>/dev/null; then
      tmux new-session -d -s hidden_object_when_ready_0523 \
        "cd ${PROJECT_ROOT} && LIBERO_TASK_SUITE=libero_object GPU=5 POLL_SECONDS=120 bash scripts/wait_and_run_libero_hidden.sh" \
        || true
    fi
  fi
}

trap restart_hidden_watchers EXIT

for chunk in $(seq 1 "${MAX_CHUNKS}"); do
  echo "[plateau] chunk=${chunk}/${MAX_CHUNKS} out_dir=${OUT_DIR} chunk_steps=${CHUNK_STEPS}"
  bash scripts/train_dreamer_vla.sh \
    training.resume=true \
    training.num_epochs=999 \
    training.max_train_steps="${CHUNK_STEPS}" \
    training.checkpoint_every=1 \
    +dataset.success_to_go_discount=0.97 \
    +world_model.success_return_head_type=binary \
    +world_model.success_return_loss_scale=16.0 \
    +world_model.success_return_hidden_dim=1024 \
    +world_model.success_return_init_logit=0.0 \
    +world_model.success_return_loss_type=bce \
    +algorithm.success_return_shaping_scale=0.25 \
    +algorithm.success_return_shaping_discount=0.99

  set +e
  "${PYTHON}" - "${LOG_PATH}" "${STATE_PATH}" "${STATUS_PATH}" \
    "${WINDOW}" "${PATIENCE}" "${MIN_REL_IMPROVE}" "${RETURN_REL_IMPROVE}" "${chunk}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
state_path = Path(sys.argv[2])
status_path = Path(sys.argv[3])
window = int(sys.argv[4])
patience = int(sys.argv[5])
min_rel = float(sys.argv[6])
return_rel = float(sys.argv[7])
chunk = int(sys.argv[8])

lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
tail = lines[-window:] if len(lines) > window else lines

def mean(key: str) -> float:
    vals = [float(item[key]) for item in tail if key in item and math.isfinite(float(item[key]))]
    return sum(vals) / max(len(vals), 1)

critic = mean("train_critic_loss")
raw_return = mean("train_raw_returns_mean")
success_loss = mean("train_wm_success_return_loss")
success_pred = mean("train_wm_success_return_pred_mean")
success_target = mean("train_wm_success_return_target_mean")
reward = mean("train_reward_mean")
step = int(lines[-1].get("global_step", -1)) if lines else -1

if state_path.exists():
    state = json.loads(state_path.read_text())
else:
    state = {
        "best_critic": float("inf"),
        "best_return": -float("inf"),
        "bad_chunks": 0,
    }

best_critic = float(state.get("best_critic", float("inf")))
best_return = float(state.get("best_return", -float("inf")))
critic_improved = critic < best_critic * (1.0 - min_rel)
return_improved = raw_return > best_return * (1.0 + return_rel)
if not math.isfinite(best_return):
    return_improved = True

if critic_improved or return_improved:
    state["best_critic"] = min(best_critic, critic)
    state["best_return"] = max(best_return, raw_return)
    state["bad_chunks"] = 0
else:
    state["bad_chunks"] = int(state.get("bad_chunks", 0)) + 1

record = {
    "chunk": chunk,
    "global_step": step,
    "window": len(tail),
    "critic_loss_mean": critic,
    "raw_return_mean": raw_return,
    "success_return_loss_mean": success_loss,
    "success_return_pred_mean": success_pred,
    "success_return_target_mean": success_target,
    "reward_mean": reward,
    "best_critic": state["best_critic"],
    "best_return": state["best_return"],
    "bad_chunks": state["bad_chunks"],
    "patience": patience,
    "stop": int(state["bad_chunks"] >= patience),
}
state_path.write_text(json.dumps(state, indent=2) + "\n")
with status_path.open("a", buffering=1) as handle:
    handle.write(json.dumps(record) + "\n")
print("[plateau-status] " + json.dumps(record, sort_keys=True))
sys.exit(10 if record["stop"] else 0)
PY
  status=$?
  set -e
  if [[ "${status}" -eq 10 ]]; then
    echo "[plateau] stop: patience reached"
    exit 0
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "[plateau] metric checker failed with status=${status}" >&2
    exit "${status}"
  fi
done

echo "[plateau] stop: reached MAX_CHUNKS=${MAX_CHUNKS}"
