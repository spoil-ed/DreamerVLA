 #!/usr/bin/env bash
# Compatibility wrapper for preprocessing all standard LIBERO suites.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/datasets/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
fi

cd "${DVLA_ROOT}"

DEFAULT_SUITES=(libero_goal libero_object libero_spatial libero_10)
suite_list=()
if [[ -n "${TASK:-}" ]]; then
  suite_list=("${TASK}")
elif [[ -n "${LIBERO_SUITES:-}" ]]; then
  read -r -a suite_list <<< "${LIBERO_SUITES}"
elif [[ -n "${SUITES:-}" ]]; then
  read -r -a suite_list <<< "${SUITES}"
else
  suite_list=("${DEFAULT_SUITES[@]}")
fi

echo "[preprocess_libero] root=${DVLA_ROOT} data_root=${DVLA_DATA_ROOT}"
echo "[preprocess_libero] suites=${suite_list[*]}"

for suite in "${suite_list[@]}"; do
  echo "[preprocess_libero] running TASK=${suite}"
  TASK="${suite}" bash "${DVLA_ROOT}/scripts/preprocess/prepare_libero_data.sh" "$@"
done

echo "[preprocess_libero] complete"
