#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/data/spoil/workspace/DreamerVLA}
PYTHON=${PYTHON:-/home/user01/miniconda3/envs/dreamervla/bin/python}
GPU=${GPU:-6}
OFT_GPU=${OFT_GPU:-7}
CHUNK_SIZE=${CHUNK_SIZE:-16}
SLEEP_SECONDS=${SLEEP_SECONDS:-300}
BACKBONES=${BACKBONES:-all}

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}
export TF_FORCE_GPU_ALLOW_GROWTH=${TF_FORCE_GPU_ALLOW_GROWTH:-true}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${ROOT}/data/outputs"
cd "${ROOT}"

count_hdf5_files() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name "*.hdf5" 2>/dev/null | wc -l
}

wait_for_source_files() {
  local dir="$1"
  local expected="${2:-10}"
  while true; do
    local count
    count="$(count_hdf5_files "$dir")"
    if [[ "$count" -ge "$expected" ]]; then
      echo "[wait] source ready: $dir ($count/$expected)"
      return 0
    fi
    echo "[wait] $dir files=$count/$expected; sleeping ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
  done
}

run_rynn_suite() {
  local suite="$1"
  local horizon="$2"
  local src="${ROOT}/data/processed_data/libero_${suite}_no_noops_t_256_pi06_remaining_reward"
  local out="${ROOT}/data/processed_data/libero_${suite}_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2"
  local ckpt="${ROOT}/data/ckpts/VLA_model_256/libero_${suite}"
  local log="${ROOT}/data/outputs/hidden_libero_${suite}_rynn_pi06_gpu${GPU}_0523.log"

  wait_for_source_files "$src" 10
  echo "[rynn] suite=${suite} src=${src} out=${out} ckpt=${ckpt}"
  RYNN_HIDDEN_RUN_ID="${suite}_rynn_pi06_gpu${GPU}_20260523" \
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/preprocess_rynn_pixel_hidden.py \
    --hdf5-dir "${src}" \
    --out-dir "${out}" \
    --chunk-size "${CHUNK_SIZE}" \
    --output-dtype float16 \
    --compression none \
    --obs-hidden-source action_query \
    --prompt-style vla_policy \
    --history 2 \
    --model-path "${ckpt}" \
    --time-horizon "${horizon}" \
    --action-head-type legacy \
    --save-action-hidden \
    --action-trigger-token-id 10004 \
    --include-state \
    --rotate-images-180 \
    2>&1 | tee "${log}"
}

run_oft_suite() {
  local suite="$1"
  local src="${ROOT}/data/processed_data/libero_${suite}_no_noops_t_256_pi06_remaining_reward"
  local out="${ROOT}/data/processed_data/libero_${suite}_no_noops_t_256_oft_official_legacy_action_hidden_vla_policy_h2"
  local ckpt="${ROOT}/data/ckpts/OpenVLA-OFT/libero_${suite}"
  local key="libero_${suite}_no_noops"
  local log="${ROOT}/data/outputs/hidden_libero_${suite}_oft_official_pi06_gpu${OFT_GPU}_0523.log"

  wait_for_source_files "$src" 10
  echo "[oft] suite=${suite} src=${src} out=${out} ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES="${OFT_GPU}" MANUAL_SHARD_RANK=0 MANUAL_SHARD_WORLD_SIZE=1 \
  "${PYTHON}" scripts/preprocess_oft_action_hidden.py \
    --openvla-oft-dir "${ROOT}/../openvla-oft" \
    --hdf5-dir "${src}" \
    --out-c-dir "${ROOT}/data/processed_data/libero_${suite}_no_noops_t_256_oft_official_action_hidden_c_h2_h8" \
    --out-d-dir "${ROOT}/data/processed_data/libero_${suite}_no_noops_t_256_oft_official_action_hidden_d_h2_h8" \
    --out-action-dir "${out}" \
    --oft-ckpt "${ckpt}" \
    --unnorm-key "${key}" \
    --image-keys agentview_rgb eye_in_hand_rgb \
    --num-images-in-input 4 \
    --include-state \
    --center-crop \
    --rotate-images-180 \
    --history 2 \
    --chunk-size "${CHUNK_SIZE}" \
    --output-dtype float16 \
    --skip-cd-sidecars \
    2>&1 | tee "${log}"
}

echo "[plan] non-90 LIBERO suites: goal object spatial 10"
echo "[plan] source: *_no_noops_t_256_pi06_remaining_reward"
echo "[plan] RynnVLA GPU=${GPU}; OpenVLA-OFT GPU=${OFT_GPU}; chunk=${CHUNK_SIZE}"
echo "[plan] backbones=${BACKBONES}"

if [[ "${BACKBONES}" == "all" || "${BACKBONES}" == "rynn" ]]; then
  run_rynn_suite goal 5
  run_rynn_suite object 5
  run_rynn_suite spatial 10
  run_rynn_suite 10 10
fi

if [[ "${BACKBONES}" == "all" || "${BACKBONES}" == "oft" ]]; then
  run_oft_suite goal
  run_oft_suite object
  run_oft_suite spatial
  run_oft_suite 10
fi

echo "[done] all non-90 LIBERO no-op/precompute jobs completed"
