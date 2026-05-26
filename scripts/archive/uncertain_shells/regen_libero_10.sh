#!/usr/bin/env bash
# All preprocessing for libero_10:
# NOTE: libero_10 has no existing pi0_legacy sidecar dir yet — Phase 1 creates from scratch.
# This is the largest job: ~386 success demos + 114 failure demos at horizon=10.
#         success sidecar + failure demo replay + failure sidecar
#
# Phases (sequential):
#   1. success sidecar  (GPU, fills missing files in *_pi0_legacy_action_hidden_vla_policy_h2/)
#   2. failure demo     (CPU sim, replays metainfo failures into *_failures/)
#   3. failure sidecar  (GPU, pi0 encoder on *_failures/ into *_failures_pi0_legacy_action_hidden_vla_policy_h2/)
#
# Resources defaults (override via env):
#   CUDA_VISIBLE_DEVICES=6,7  NUM_GPUS=2  MASTER_PORT=29581  MAX_RETRIES=3
#
# Each phase has 3-retry built in for GPU jobs; CPU phase is single-pass.
# Phases skip themselves cleanly if their outputs already exist.
#
# Usage:
#   bash scripts/regen_libero_10.sh
#   CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 bash scripts/regen_libero_10.sh
set -uo pipefail

SUITE=10
HORIZON=10         # libero_10 uses action horizon = 10
EXPECTED_SUCCESS=10
DEFAULT_PORT=29581

PROJECT_ROOT=/mnt/data/spoil/workspace/DreamerVLA
PYTHON=/home/user01/miniconda3/envs/dreamervla/bin/python
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=osmesa

GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"
N="${NUM_GPUS:-2}"
BASE_PORT="${MASTER_PORT:-${DEFAULT_PORT}}"
MAX_RETRIES="${MAX_RETRIES:-3}"

SUCCESS_SRC="${PROJECT_ROOT}/data/processed_data/libero_${SUITE}_no_noops_t_256_pi06_remaining_reward"
SUCCESS_SIDECAR_OUT="${PROJECT_ROOT}/data/processed_data/libero_${SUITE}_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2"
FAILURE_HDF5_OUT="${PROJECT_ROOT}/data/processed_data/libero_${SUITE}_no_noops_t_256_failures"
FAILURE_SIDECAR_OUT="${PROJECT_ROOT}/data/processed_data/libero_${SUITE}_no_noops_t_256_failures_pi0_legacy_action_hidden_vla_policy_h2"
RAW_DIR="${PROJECT_ROOT}/data/libero/datasets/libero_${SUITE}"
METAINFO="${PROJECT_ROOT}/libero_${SUITE}_metainfo.json"
CKPT="${PROJECT_ROOT}/data/ckpts/VLA_model_256/libero_${SUITE}"

LOG_DIR="${PROJECT_ROOT}/data/outputs"
mkdir -p "${LOG_DIR}"

[[ -d "${SUCCESS_SRC}" ]] || { echo "ERROR: missing ${SUCCESS_SRC}"; exit 2; }
[[ -d "${RAW_DIR}" ]]     || { echo "ERROR: missing ${RAW_DIR}"; exit 2; }
[[ -f "${METAINFO}" ]]    || { echo "ERROR: missing ${METAINFO}"; exit 2; }
[[ -d "${CKPT}" ]]        || { echo "ERROR: missing VLA ckpt ${CKPT}"; exit 2; }

# Count metainfo failures so failure sidecar knows when it's done
EXPECTED_FAILURE_TASKS=$(python -c "
import json
m = json.load(open('${METAINFO}'))
n = 0
for t, demos in m.items():
    if any(not i.get('success') for i in demos.values()):
        n += 1
print(n)
")

echo "════════════════════════════════════════════════════════════════"
echo " regen_libero_${SUITE}.sh   horizon=${HORIZON}  GPUs=${GPUS} (nproc=${N})"
echo "  expected success sidecars: ${EXPECTED_SUCCESS}"
echo "  expected failure task hdf5s: ${EXPECTED_FAILURE_TASKS}"
echo "════════════════════════════════════════════════════════════════"

# ───────────────────────── helpers ──────────────────────────
_run_sidecar() {
  # $1=src $2=out $3=run_id_prefix $4=expected $5=base_port_offset
  local src="$1" out="$2" prefix="$3" expected="$4" port_off="$5"
  mkdir -p "${out}"
  for attempt in $(seq 1 "${MAX_RETRIES}"); do
    local pre=$(ls "${out}"/*.hdf5 2>/dev/null | wc -l)
    if [[ "${pre}" -ge "${expected}" ]]; then
      echo "  [${prefix}] already complete (${pre}/${expected})"; return 0
    fi
    local port=$((BASE_PORT + port_off + attempt - 1))
    local log="${LOG_DIR}/${prefix}_attempt${attempt}_$(date +%Y%m%d_%H%M%S).log"
    local run_id="${prefix}_a${attempt}_$(date +%Y%m%d_%H%M%S)"

    echo ""
    echo "  [${prefix}] ATTEMPT ${attempt}/${MAX_RETRIES}: pre=${pre}/${expected} port=${port}"
    echo "  [${prefix}] log: ${log}"
    rm -rf "${out}/.progress" 2>/dev/null || true

    RYNN_HIDDEN_RUN_ID="${run_id}" \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
      "${PYTHON}" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc-per-node="${N}" --master_port="${port}" \
        "${PROJECT_ROOT}/scripts/preprocess_rynn_pixel_hidden.py" \
        --hdf5-dir "${src}" --out-dir "${out}" \
        --chunk-size 16 --output-dtype float16 --compression none \
        --obs-hidden-source action_query --prompt-style vla_policy --history 2 \
        --model-path "${CKPT}" --time-horizon "${HORIZON}" \
        --action-head-type legacy --save-action-hidden --action-trigger-token-id 10004 \
        --include-state --rotate-images-180 \
        > "${log}" 2>&1
    local rc=$?
    local post=$(ls "${out}"/*.hdf5 2>/dev/null | wc -l)
    echo "  [${prefix}] attempt ${attempt}: exit=${rc}  sidecars ${pre} -> ${post}/${expected}"

    if [[ "${post}" -ge "${expected}" ]]; then
      echo "  [${prefix}] COMPLETE"; return 0
    fi
    if [[ "${attempt}" -lt "${MAX_RETRIES}" ]]; then
      local sleep_for=15; [[ "${post}" -eq "${pre}" ]] && sleep_for=60
      sleep "${sleep_for}"
    fi
  done
  echo "  [${prefix}] GAVE UP after ${MAX_RETRIES} attempts"
  return 1
}

# ───────────────────────── Phase 1: success sidecar ──────────────────────────
echo ""
echo "── Phase 1/3: success sidecar ──"
_run_sidecar "${SUCCESS_SRC}" "${SUCCESS_SIDECAR_OUT}" \
  "regen_success_sidecar_libero_${SUITE}" "${EXPECTED_SUCCESS}" 0

# ───────────────────────── Phase 2: failure demo replay (CPU) ──────────────────────────
echo ""
echo "── Phase 2/3: failure demo replay (CPU sim) ──"
pre=$(ls "${FAILURE_HDF5_OUT}"/*.hdf5 2>/dev/null | wc -l)
if [[ "${pre}" -ge "${EXPECTED_FAILURE_TASKS}" ]]; then
  echo "  [failure_demo] already complete (${pre}/${EXPECTED_FAILURE_TASKS})"
else
  mkdir -p "${FAILURE_HDF5_OUT}"
  LOG="${LOG_DIR}/regen_failure_demo_libero_${SUITE}_$(date +%Y%m%d_%H%M%S).log"
  echo "  [failure_demo] log: ${LOG}"
  pushd "${PROJECT_ROOT}/src/utils/libero_utils" > /dev/null
  "${PYTHON}" regenerate_libero_failure_demos.py \
    --libero_task_suite libero_${SUITE} \
    --libero_raw_data_dir "${RAW_DIR}" \
    --libero_target_dir "${FAILURE_HDF5_OUT}" \
    --libero_metainfo_json "${METAINFO}" \
    --image_resolution 256 \
    > "${LOG}" 2>&1
  rc=$?
  popd > /dev/null
  post=$(ls "${FAILURE_HDF5_OUT}"/*.hdf5 2>/dev/null | wc -l)
  echo "  [failure_demo] exit=${rc}  hdf5 ${pre} -> ${post}/${EXPECTED_FAILURE_TASKS}"
  [[ "${rc}" -ne 0 ]] && echo "  [failure_demo] WARNING non-zero exit; continuing to phase 3 with what we have"
fi

# ───────────────────────── Phase 3: failure sidecar ──────────────────────────
echo ""
echo "── Phase 3/3: failure sidecar ──"
failure_hdf5_count=$(ls "${FAILURE_HDF5_OUT}"/*.hdf5 2>/dev/null | wc -l)
if [[ "${failure_hdf5_count}" -eq 0 ]]; then
  echo "  [failure_sidecar] no failure HDF5s found, skipping"
else
  _run_sidecar "${FAILURE_HDF5_OUT}" "${FAILURE_SIDECAR_OUT}" \
    "regen_failure_sidecar_libero_${SUITE}" "${failure_hdf5_count}" 100
fi

# ───────────────────────── final summary ──────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " FINAL: libero_${SUITE}"
echo "  success sidecar: $(ls ${SUCCESS_SIDECAR_OUT}/*.hdf5 2>/dev/null | wc -l)/${EXPECTED_SUCCESS}"
echo "  failure HDF5:    $(ls ${FAILURE_HDF5_OUT}/*.hdf5 2>/dev/null | wc -l)/${EXPECTED_FAILURE_TASKS}"
echo "  failure sidecar: $(ls ${FAILURE_SIDECAR_OUT}/*.hdf5 2>/dev/null | wc -l)/${failure_hdf5_count:-0}"
echo "════════════════════════════════════════════════════════════════"
