#!/usr/bin/env bash
# One-command LIBERO preprocessing for the standard RynnVLA-002 route.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-data}"
TASK="${TASK:-libero_goal}"
PYTHON="${PYTHON:-python}"
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-8}"
GPUS="${GPUS:-0}"
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-${NGPU:-}}"
OVERWRITE="${OVERWRITE:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --data-root) DVLA_DATA_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --gpus)
      GPUS="$2"
      export CUDA_VISIBLE_DEVICES="$2"
      if [[ -z "${ACTION_HIDDEN_GPUS}" ]]; then
        gpu_count=0
        for _gpu in ${2//,/ }; do gpu_count=$((gpu_count + 1)); done
        ACTION_HIDDEN_GPUS="${gpu_count}"
      fi
      shift 2
      ;;
    --ngpu) ACTION_HIDDEN_GPUS="$2"; shift 2 ;;
    --num-procs) PRETOKENIZE_PROCS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-1}"
export DVLA_DATA_ROOT TASK PYTHON PRETOKENIZE_PROCS GPUS ACTION_HIDDEN_GPUS OVERWRITE
cd "${DVLA_ROOT}"

LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/datasets/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF

echo "[prepare_libero_data] TASK=${TASK} DVLA_DATA_ROOT=${DVLA_DATA_ROOT}"
bash scripts/preprocess/10_hdf5_reward.sh
bash scripts/preprocess/20_pretokenize_dataset.sh
bash scripts/preprocess/30_action_hidden.sh
bash scripts/preprocess/40_validate.sh

# Optional Scheme-B / OpenVLA-OFT sidecars, kept as direct copyable commands:
# bash scripts/preprocess/32_input_token_hidden.sh --task "${TASK}"
# bash scripts/preprocess/35_oft_action_hidden.sh --task "${TASK}"
