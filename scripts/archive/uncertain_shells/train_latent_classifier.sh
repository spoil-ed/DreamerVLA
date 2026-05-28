#!/usr/bin/env bash
# Thin wrapper around `python -m dreamer_vla.cli.train` for the LatentSuccessClassifier
# workspace. Picks env defaults; the heavy lifting lives in the Hydra config.
#
# Usage:
#   bash scripts/train_latent_classifier.sh                                  # default: linear head
#   bash scripts/train_latent_classifier.sh latent_classifier_libero_goal_small_tf  # small TF ablation
#   CUDA_VISIBLE_DEVICES=6 bash scripts/train_latent_classifier.sh
#   bash scripts/train_latent_classifier.sh latent_classifier_libero_goal training.max_steps=200  # smoke
#
# Outputs land under cfg.training.out_dir:
#   ├── ckpt/best_window_f1{F}_th{T}.ckpt       ← best window-level
#   ├── ckpt/best_episode_f1{F}_th{T}.ckpt      ← best episode-level (WMPO any-positive)
#   ├── ckpt/latest.ckpt + final.ckpt           ← BaseRunner standard tags
#   ├── log/train_log.jsonl                     ← per-step + per-eval events
#   └── summary.json                            ← terminal-state best F1s

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PY=${PY:-/home/user01/miniconda3/envs/dreamervla/bin/python}
CONFIG_NAME=${1:-latent_classifier_libero_goal}
shift || true        # remaining args become Hydra overrides
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

echo "[train_latent_classifier] config=${CONFIG_NAME}"
echo "[train_latent_classifier] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train_latent_classifier] overrides: $*"

exec "${PY}" -m dreamer_vla.cli.train --config-name "${CONFIG_NAME}" "$@"
