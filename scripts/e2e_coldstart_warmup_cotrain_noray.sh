#!/usr/bin/env bash
# E2E launcher: pure-Hydra cold-start collection -> offline-warmup online cotrain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
# DVLA_ROOT (code) and DVLA_DATA_ROOT (data) are INDEPENDENT. Data resolves under
# DVLA_DATA_ROOT; only when it is unset do we fall back to <DVLA_ROOT>/data. Echo both
# so a wrong/defaulted data root is visible up front (instead of a late asset-check fail).
if [[ -n "${DVLA_DATA_ROOT:-}" ]]; then
  _DVLA_DATA_ROOT_SRC="from environment"
else
  export DVLA_DATA_ROOT="${DVLA_ROOT}/data"
  _DVLA_DATA_ROOT_SRC="DEFAULTED from DVLA_ROOT — export DVLA_DATA_ROOT=<dir with checkpoints/ and datasets/> if data lives elsewhere"
fi
echo "[e2e] DVLA_ROOT      = ${DVLA_ROOT}  (code root)" >&2
echo "[e2e] DVLA_DATA_ROOT = ${DVLA_DATA_ROOT}  (${_DVLA_DATA_ROOT_SRC})" >&2
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${DVLA_ROOT}"

CONDA_ENV_NAME="${DVLA_CONDA_ENV:-dreamervla}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

# Multi-GPU DDP cotrain stability (harmless for single-GPU); override by exporting it.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export ABSL_MIN_LOG_LEVEL="${ABSL_MIN_LOG_LEVEL:-3}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export GYM_DISABLE_WARNINGS="${GYM_DISABLE_WARNINGS:-1}"
export USE_TF="${USE_TF:-0}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
_DVLA_PYTHONWARNINGS="ignore::FutureWarning:libero.libero.benchmark,ignore:enable_nested_tensor is True.*:UserWarning:torch.nn.modules.transformer"
export PYTHONWARNINGS="${_DVLA_PYTHONWARNINGS}${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"

python -m dreamervla.launchers.coldstart_warmup_cotrain mode=noray "$@"
