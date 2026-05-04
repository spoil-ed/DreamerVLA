#!/usr/bin/env bash
# WM Checklist: prior/posterior alignment diagnostic.
#
# Runs `src/cli/diagnose_wm.py` against a saved TSSMWorldModelTransDreamer (or
# its discrete subclass) checkpoint and writes a JSON report to
# data/outputs/eval/eval_wm/.
#
# Usage:
#   conda activate dreamervla
#   bash scripts/diagnose_wm.sh                                  # run with defaults
#   CKPT=path/to/file.ckpt bash scripts/diagnose_wm.sh           # override ckpt
#   CONFIG_NAME=pretokenize_wm_libero_10_discrete \
#     CKPT=... bash scripts/diagnose_wm.sh                       # discrete WM
#
# Override via env vars:
#   CONFIG_NAME    Hydra config name (default: pretokenize_wm_libero_10)
#   CKPT           Path to .ckpt file (REQUIRED if no default ckpt set below)
#   NUM_SAMPLES    Number of samples to aggregate over (default: 128)
#   DEVICE         CUDA device (default: cuda:0)
#   DATASET_KEY    Which cfg dataset to use: dataset / dataset_val_ind / dataset_val_ood
#                  (default: dataset_val_ind)
#   OUT            Output JSON path (default: data/outputs/eval/eval_wm/wm_checklist_<tag>_s<N>.json)
#
# Trailing args are forwarded to the python script verbatim.
#
# ── What gets measured ───────────────────────────────────────────────────────
# Level 0  representation-space consistency
#   feature_post_norm ≈ feature_prior_norm
#   feature_diff_l2_mean << feature_post_norm
#   feature_cos_mean → 0.9+
#   ||z_post − z_prior|| small
#   ||h_post − h_prior|| small  (= 0 by construction in TransDreamer; same h)
#
# Level 1  KL / distribution health
#   kl_mean in 0.1–3
#   ||μ_post − μ_prior|| small
#   std_post ≈ std_prior
#   std_post not too large (~0.5–1)
#   std_prior not too small (>0.05)
#
# Level 2  transition collapse
#   prior_relative_centered_norm > 0.1
#   prior_pairwise_l2_mean nonzero
#   prior_adjacent_cos_mean < 0.999
#
# Level 3  action conditioning
#   real_to_target_l2 < zero_to_target_l2
#   real_to_target_l2 < shuffle_to_target_l2
#   margin_zero > 0
#   margin_shuffle > 0
#
# Level 4  multi-step rollout       (skipped for T=2 pretokenized batches)
#   KL@1 < KL@5 < KL@10 < KL@20  (slow growth)
#   feature_l2 grows smoothly with horizon
#   no flat / explosion
#
# Level 5  token-level
#   dynamic_acc clearly above random
#   dynamic_CE can decrease
#   dynamic-static gap is controllable
#
# Minimum-required (quick-look)
#   feature_post_norm / feature_prior_norm / feature_diff_l2 / feature_cos
#   std_post_mean / std_prior_mean
#   prior_relative_centered_norm
#   real_to_target_l2 / zero_to_target_l2 / shuffle_to_target_l2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_NAME="${CONFIG_NAME:-pretokenize_wm_libero_10}"
CKPT="${CKPT:-}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
DEVICE="${DEVICE:-cuda:0}"
DATASET_KEY="${DATASET_KEY:-dataset_val_ind}"

if [[ -z "${CKPT}" ]]; then
  echo "ERROR: CKPT not set. Pass via env var, e.g.:"
  echo "  CKPT=data/outputs/worldmodel/pretokenize_wm/<run>/checkpoints/<file>.ckpt \\"
  echo "    bash scripts/diagnose_wm.sh"
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: CKPT file not found: ${CKPT}"
  exit 1
fi

# Auto-derive OUT from ckpt path if not specified.
if [[ -z "${OUT:-}" ]]; then
  CKPT_STEM="$(basename "${CKPT}" .ckpt)"
  RUN_NAME="$(basename "$(dirname "$(dirname "${CKPT}")")")"
  OUT="${PROJECT_ROOT}/data/outputs/eval/eval_wm/wm_checklist_${RUN_NAME}_${CKPT_STEM}_s${NUM_SAMPLES}.json"
fi

mkdir -p "$(dirname "${OUT}")"

echo "[diagnose_wm.sh] CONFIG_NAME = ${CONFIG_NAME}"
echo "[diagnose_wm.sh] CKPT        = ${CKPT}"
echo "[diagnose_wm.sh] NUM_SAMPLES = ${NUM_SAMPLES}"
echo "[diagnose_wm.sh] DEVICE      = ${DEVICE}"
echo "[diagnose_wm.sh] DATASET_KEY = ${DATASET_KEY}"
echo "[diagnose_wm.sh] OUT         = ${OUT}"

python -m src.cli.diagnose_wm \
  --config-name "${CONFIG_NAME}" \
  --ckpt "${CKPT}" \
  --num-samples "${NUM_SAMPLES}" \
  --device "${DEVICE}" \
  --dataset-key "${DATASET_KEY}" \
  --out "${OUT}" \
  "$@"
