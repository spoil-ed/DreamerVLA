#!/usr/bin/env bash
# Phase 0 of docs/classifier_revision_plan.md — one-command end-to-end:
#   1. Fit sklearn LR ceiling on real hidden
#   2. Episode-eval every existing classifier ckpt + LR ceiling under WMPO protocol
#
# Output: data/outputs/dreamervla/outcome_classifier/_compare_v0/
#   ├── lr_ceiling/              ← LR ckpt + train_log.jsonl + config.yaml
#   └── phase0_eval.json         ← per-ckpt × per-threshold episode F1 table
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/classifier_phase0_baseline.sh
#
# Pre-conditions:
#   - data/processed_data/libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2 exists
#   - data/processed_data/libero_goal_no_noops_t_256_failures_pi0_legacy_action_hidden_vla_policy_h2 exists
#   - Existing v1/v2/v3b ckpts under data/outputs/dreamervla/outcome_classifier/libero_goal/*

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY=/home/user01/miniconda3/envs/dreamervla/bin/python
CFG=configs/wmpo_classifier_libero_goal_v4_real_hidden.yaml
OUTDIR=data/outputs/dreamervla/outcome_classifier/_compare_v0
mkdir -p "$OUTDIR"

LR_DIR="$OUTDIR/lr_ceiling"
LR_CKPT="$LR_DIR/best.ckpt"

if [[ -f "$LR_CKPT" ]] && [[ "${RETRAIN_LR:-0}" != "1" ]]; then
  echo "[phase0] LR ckpt already exists at $LR_CKPT (set RETRAIN_LR=1 to re-fit)"
else
  echo "[phase0] fitting sklearn LR ceiling → $LR_DIR"
  PYTHONUNBUFFERED=1 "$PY" -u scripts/train_logreg_classifier.py \
    --config "$CFG" \
    --out "$LR_DIR" \
    --C 0.01 \
    --stride-train 8 --stride-val 1 \
    --max-iter 200 \
    --thresh-steps 30 \
    2>&1 | tee "$LR_DIR.stdout.log"
fi

CKPT_BASE=data/outputs/dreamervla/outcome_classifier/libero_goal

# Collect all existing ckpts that should be benchmarked side-by-side.
declare -a CKPT_ARGS=()
add_if_exists() {
  local tag="$1" path="$2"
  if [[ -f "$path" ]]; then
    CKPT_ARGS+=( "--ckpt" "$tag=$path" )
    echo "  + $tag → $path"
  else
    echo "  - $tag (not found, skipping): $path"
  fi
}

echo "[phase0] collecting ckpts for episode-eval:"
add_if_exists v1_demo_only             "$CKPT_BASE/v1_demo_only/best.ckpt"
add_if_exists v2_wm_replay             "$CKPT_BASE/v2_wm_replay/best.ckpt"
add_if_exists v3b_finish_step          "$CKPT_BASE/v3b_finish_step/best.ckpt"
add_if_exists v3b_swap_plus_failures   "$CKPT_BASE/v3b_swap_plus_failures/best.ckpt"
add_if_exists v3_with_failures         "$CKPT_BASE/v3_with_failures/best.ckpt"
add_if_exists lr_ceiling               "$LR_CKPT"

if [[ ${#CKPT_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: no ckpts found to evaluate." >&2
  exit 1
fi

echo
echo "[phase0] running WMPO-protocol episode eval (real hidden, stride=1, min_steps=64)"
PYTHONUNBUFFERED=1 "$PY" -u scripts/eval_latent_classifier_episode.py \
  --config "$CFG" \
  --hidden-mode real \
  --threshold-sweep 0.5,0.7,0.85,0.93,0.97,0.99 \
  --stride 1 --min-steps 64 --batch-size 32 \
  --device cuda:0 \
  "${CKPT_ARGS[@]}" \
  --out "$OUTDIR/phase0_eval.json" \
  2>&1 | tee "$OUTDIR/phase0_eval.stdout.log"

echo
echo "[phase0] done. compare table at end of $OUTDIR/phase0_eval.stdout.log"
echo "         full json: $OUTDIR/phase0_eval.json"
