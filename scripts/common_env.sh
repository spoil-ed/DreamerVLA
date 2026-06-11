#!/usr/bin/env bash
# DEPRECATED for formal entrypoints -- they are now self-contained and read
# DVLA_DATA_ROOT (see docs/data_layout.md). Kept only for the legacy
# machine-specific scripts (*_45.sh, *_g67.sh, smoke/, archive/) that still
# source it.

_DREAMERVLA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${_DREAMERVLA_SCRIPT_DIR}/.." && pwd -P)}"
export PROJECT_ROOT="${PROJECT_ROOT:-${DVLA_ROOT}}"

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-dreamervla}"
_DREAMERVLA_CONDA_BIN="${CONDA_ENV_BIN:-${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin}"
if [[ -z "${PYTHON:-}" && -x "${_DREAMERVLA_CONDA_BIN}/python" ]]; then
  export PATH="${_DREAMERVLA_CONDA_BIN}:${PATH}"
  export PYTHON="${_DREAMERVLA_CONDA_BIN}/python"
else
  export PYTHON="${PYTHON:-python}"
fi

case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "${MUJOCO_GL}" == "egl" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
elif [[ "${MUJOCO_GL}" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_ROOT}/third_party/LIBERO/libero/datasets
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
fi

unset _DREAMERVLA_SCRIPT_DIR
unset _DREAMERVLA_CONDA_BIN
