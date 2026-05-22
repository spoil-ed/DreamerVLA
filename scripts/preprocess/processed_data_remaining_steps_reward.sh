#!/usr/bin/env bash
# Rewrite sparse 0/1 LIBERO rewards into pi0.6-style normalized
# remaining-steps-to-success targets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/processed_data/${LIBERO_TASK_SUITE}_no_noops_t_${IMAGE_RESOLUTION}_pi06_remaining_reward}"
PYTHON_BIN="${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}"
METAINFO_JSON_PATH="${METAINFO_JSON:-${METINFO_JSON:-}}"

ARGS=(
  "$ROOT_DIR/scripts/preprocess_remaining_steps_reward.py"
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --failure-value "${REMAINING_REWARD_FAILURE_VALUE:-0.0}"
  --min-value "${REMAINING_REWARD_MIN_VALUE:-0.0}"
  --max-value "${REMAINING_REWARD_MAX_VALUE:-1.0}"
)

if [[ -n "${METAINFO_JSON_PATH}" ]]; then
  ARGS+=(--metainfo-json "${METAINFO_JSON_PATH}")
elif [[ -f "$ROOT_DIR/${LIBERO_TASK_SUITE}_metainfo.json" ]]; then
  ARGS+=(--metainfo-json "$ROOT_DIR/${LIBERO_TASK_SUITE}_metainfo.json")
fi
if [[ -n "${MAX_FILES:-}" ]]; then
  ARGS+=(--max-files "$MAX_FILES")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

echo "[remaining-reward] input:  ${INPUT_DIR}"
echo "[remaining-reward] output: ${OUTPUT_DIR}"
"${PYTHON_BIN}" "${ARGS[@]}"
