#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../common_env.sh"
cd "${DVLA_ROOT}"

MASTER_PORT="${MASTER_PORT:-29541}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NGPU="${NGPU:-2}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

CONFIG="${CONFIG:-${DVLA_ROOT}/configs/online_wmpo_outcome_libero_goal.yaml}"
WORLD_MODEL_CKPT="${WORLD_MODEL_CKPT:-${DVLA_ROOT}/data/outputs/worldmodel/dinowm_chunk/20260525_221114/ckpt/step_00015000.ckpt}"
VLA_CKPT_PATH="${VLA_CKPT_PATH:-${DVLA_ROOT}/data/ckpts/VLA_model_256/libero_goal}"

OUT_DIR="${OUT_DIR:-${DVLA_ROOT}/data/outputs/dreamervla/smoke_multiproc_collector_g45/${TS}}"
LOG_FILE="${LOG_FILE:-${DVLA_ROOT}/data/outputs/logs/smoke_multiproc_collector_g45_${TS}.log}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "[multiproc-smoke] out_dir=$OUT_DIR"
echo "[multiproc-smoke] log_file=$LOG_FILE"

PYTHONPATH="${DVLA_ROOT}" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" MUJOCO_GL="${MUJOCO_GL}" \
  "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  scripts/training/train_online_rynnvla_action_hidden_dreamervla_multiproc.py \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --world-model-ckpt "$WORLD_MODEL_CKPT" \
  --vla-ckpt-path "$VLA_CKPT_PATH" \
  --encoder-state-ckpt "" \
  --action-head-type legacy \
  --task-suite libero_goal \
  --task-ids 0,1,2,3 \
  --episode-horizon 40 \
  --total-env-steps 120 \
  --num-collectors-per-rank 2 \
  --encoder-batch-size 8 \
  --encoder-batch-timeout-ms 30 \
  --collect-chunk-steps 5 \
  --sequence-length 32 \
  --replay-size 200 \
  --log-every 20 \
  --rssm-action-scale env \
  --bc-to-ref 0.1 \
  --freeze-log-std \
  2>&1 | tee "$LOG_FILE"
