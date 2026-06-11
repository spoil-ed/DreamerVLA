# Portable Paths + Self-Contained Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple project root (`DVLA_ROOT`) from data root (`DVLA_DATA_ROOT`), rewrite the 9 formal entry scripts as self-contained (RLinf style, no `common_env.sh`), and add a portable dataset manifest (`docs/data_layout.md`).

**Architecture:** Configs read `${oc.env:DVLA_DATA_ROOT,data}/...` (pure text replacement of `${oc.env:DVLA_ROOT,.}/data/...`). Each formal script carries its own env block: derive `DVLA_ROOT` from script location, default `DVLA_DATA_ROOT=${DVLA_ROOT}/data`, write the LIBERO path config pointing `datasets:` at the data root. No conda autodetection — user activates the env; scripts honor `$PYTHON` (default `python` from PATH).

**Tech Stack:** bash, Hydra/OmegaConf `oc.env` resolver, LIBERO config.yaml mechanism.

**Spec:** `docs/superpowers/specs/2026-06-11-portable-paths-self-contained-scripts-design.md`

---

## Reference: the standard env block

Every rewritten script starts with this block (shown here once for the reader's
orientation; each task below repeats the exact content for its script, with
`DVLA_ROOT` derivation adjusted for scripts living in subdirectories).
The LIBERO sub-block is included in every runtime script (anything that may
import `libero` or touch LIBERO data); `install_env.sh` omits it.

```bash
# ---- environment (self-contained; no common_env.sh) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"          # scripts/ -> repo root
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

# ---- LIBERO paths (datasets live under the data root) -----------------------
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/dataset/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
fi
```

Notes:
- `PYOPENGL_PLATFORM` defaults to `${MUJOCO_GL}` — same effective mapping as
  the old egl/osmesa case statement.
- `DREAMERVLA_WRITE_LIBERO_CONFIG=0` skip-guard is preserved from the old
  `common_env.sh`.
- `.libero/` moves from `${DVLA_ROOT}/.libero` to `${DVLA_DATA_ROOT}/.libero`.

---

### Task 1: Config sweep — `DVLA_ROOT` → `DVLA_DATA_ROOT`

**Files:**
- Modify: `configs/*.yaml`, `configs/task/*.yaml` (18 files, ~70 references)

- [ ] **Step 1: Confirm every config reference matches the replace pattern**

Run:
```bash
cd /mnt/data/spoil/workspace/DreamerVLA
grep -rn 'oc.env:DVLA_ROOT' configs/ | grep -v 'oc.env:DVLA_ROOT,.}/data/' || echo "ALL-MATCH"
grep -rn 'oc.env:PROJECT_ROOT' configs/ || echo "NO-PROJECT-ROOT"
```
Expected: `ALL-MATCH` and `NO-PROJECT-ROOT` (no references outside the
pattern). If any line appears, handle it case-by-case with the same semantics
(data paths → `DVLA_DATA_ROOT`, code paths stay `DVLA_ROOT`) before the sed.

- [ ] **Step 2: Apply the text replacement**

```bash
sed -i 's|${oc.env:DVLA_ROOT,.}/data/|${oc.env:DVLA_DATA_ROOT,data}/|g' \
  configs/*.yaml configs/task/*.yaml
```

- [ ] **Step 3: Verify zero residue and resolution in both modes**

```bash
grep -rn 'oc.env:DVLA_ROOT' configs/ && echo "FAIL: residue" || echo "OK: no residue"
# Unset mode: paths resolve relative to CWD, same as old `.` default
python -m dreamer_vla.cli.train --config-name vla_rynnvla_action_head --cfg job 2>/dev/null \
  | grep -m3 'data/ckpts'
# Set mode: paths resolve under the external root
DVLA_DATA_ROOT=/tmp/dvla_data_test \
  python -m dreamer_vla.cli.train --config-name vla_rynnvla_action_head --cfg job 2>/dev/null \
  | grep -m3 '/tmp/dvla_data_test/ckpts'
```
Expected: "OK: no residue"; first grep shows `data/ckpts/...` paths; second
shows `/tmp/dvla_data_test/ckpts/...` paths. Repeat the two `--cfg job` checks
for `world_model_dinowm_chunk`, `dreamervla_rynn_dino_wm_wmpo_outcome`,
`eval_libero_vla` (grep `ckpts` or `processed_data`).

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -s -m "refactor: point config data paths at DVLA_DATA_ROOT"
```

---

### Task 2: Rewrite the four core Hydra launchers

**Files:**
- Modify: `scripts/train_vla.sh`, `scripts/train_wm.sh`,
  `scripts/train_dreamervla.sh`, `scripts/eval_libero_vla.sh`

Each keeps its header comment (CONFIG registry + examples) and launch logic
verbatim; only the `source common_env.sh` line is replaced by the standard env
block. The four scripts differ only in header comment, default `CONFIG`,
default `MASTER_PORT`, and echo tag.

- [ ] **Step 1: Rewrite `scripts/train_vla.sh`**

Keep lines 1–25 (shebang + header comment block) unchanged. Replace everything
from `set -euo pipefail` to the end with:

```bash
set -euo pipefail

# ---- environment (self-contained; no common_env.sh) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

# ---- LIBERO paths (datasets live under the data root) -----------------------
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/dataset/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
fi

cd "${DVLA_ROOT}"

# ---- knobs -------------------------------------------------------------------
CONFIG="${CONFIG:-vla_rynnvla_action_head}"
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

# ---- launch ------------------------------------------------------------------
echo "[train_vla] python=$(command -v "${PYTHON}")"
echo "[train_vla] root=${DVLA_ROOT}  data_root=${DVLA_DATA_ROOT}"
echo "[train_vla] config=${CONFIG}  ngpu=${NGPU}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[train_vla] out_dir=${OUT_DIR:-<config default: \${DVLA_DATA_ROOT}/outputs/vla/.../<timestamp>>}"
echo "[train_vla] extra hydra args: $*"

if [ "${NGPU}" -gt 1 ]; then
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${NGPU}" --master_port="${MASTER_PORT}" \
    -m dreamer_vla.cli.train --config-name "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m dreamer_vla.cli.train --config-name "${CONFIG}" "$@"
fi
```

- [ ] **Step 2: Rewrite `scripts/train_wm.sh`**

Same as Step 1 with these substitutions: keep its own header comment
(lines 1–24); `CONFIG="${CONFIG:-world_model_dinowm_chunk}"`;
`MASTER_PORT="${MASTER_PORT:-29500}"`; echo tag `[train_wm]`; out_dir echo
says `outputs/worldmodel/...`.

- [ ] **Step 3: Rewrite `scripts/train_dreamervla.sh`**

Same with: keep its header comment (lines 1–25);
`CONFIG="${CONFIG:-dreamervla_rynn_dino_wm_wmpo_outcome}"`;
`MASTER_PORT="${MASTER_PORT:-29502}"`; echo tag `[train_dreamervla]`; out_dir
echo says `outputs/dreamervla/...`.

- [ ] **Step 4: Rewrite `scripts/eval_libero_vla.sh`**

Same env block, then (no NGPU/torchrun in this script):

```bash
cd "${DVLA_ROOT}"

CONFIG="${CONFIG:-eval_libero_vla}"

echo "[eval_libero_vla] python=$(command -v "${PYTHON}")"
echo "[eval_libero_vla] root=${DVLA_ROOT}  data_root=${DVLA_DATA_ROOT}"
echo "[eval_libero_vla] config=${CONFIG}  gpus=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "[eval_libero_vla] extra hydra args: $*"

exec "${PYTHON}" -m dreamer_vla.cli.train --config-name "${CONFIG}" "$@"
```

- [ ] **Step 5: Syntax-check and verify no common_env reference**

```bash
bash -n scripts/train_vla.sh scripts/train_wm.sh scripts/train_dreamervla.sh scripts/eval_libero_vla.sh
grep -l common_env scripts/train_vla.sh scripts/train_wm.sh scripts/train_dreamervla.sh scripts/eval_libero_vla.sh || echo OK
```
Expected: no output from `bash -n`; `OK`.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_vla.sh scripts/train_wm.sh scripts/train_dreamervla.sh scripts/eval_libero_vla.sh
git commit -s -m "refactor: make core launchers self-contained with DVLA_DATA_ROOT"
```

---

### Task 3: Rewrite `scripts/download_assets.sh`

**Files:**
- Modify: `scripts/download_assets.sh` (full rewrite)

- [ ] **Step 1: Rewrite the script**

```bash
#!/usr/bin/env bash
# Download model weights and benchmark datasets used by formal DreamerVLA flows.
# All assets land under ${DVLA_DATA_ROOT} (default: <repo>/data).
set -euo pipefail

# ---- environment (self-contained; no common_env.sh) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"
cd "${DVLA_ROOT}"

RYNNVLA_REPO="${RYNNVLA_REPO:-Alibaba-DAMO-Academy/RynnVLA-002}"
WORLDVLA_REPO="${WORLDVLA_REPO:-Alibaba-DAMO-Academy/WorldVLA}"
LUMINA_REPO="${LUMINA_REPO:-Alpha-VLLM/Lumina-mGPT-7B-768}"
LIBERO_SUITES="${LIBERO_SUITES:-libero_goal libero_object libero_spatial libero_10}"
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DOWNLOAD_LIBERO="${DOWNLOAD_LIBERO:-1}"
DOWNLOAD_CALVIN="${DOWNLOAD_CALVIN:-0}"
DOWNLOAD_ACTION_WM="${DOWNLOAD_ACTION_WM:-1}"

CKPT_DIR="${DVLA_DATA_ROOT}/ckpts"
LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${DVLA_DATA_ROOT}/dataset/libero}"
mkdir -p "${CKPT_DIR}" "${LIBERO_DATASET_DIR}"

echo "[download_assets] data_root=${DVLA_DATA_ROOT}"

normalize_list() {
  printf '%s\n' "$1" | tr ',' ' '
}

if [[ "${DOWNLOAD_WEIGHTS}" == "1" ]]; then
  echo "[download_assets] Hugging Face weights -> ${CKPT_DIR}"
  hf download "${WORLDVLA_REPO}" --repo-type model \
    --local-dir "${CKPT_DIR}" \
    --include "chameleon/tokenizer/*" "chameleon/base_model/*" "base_model/*" "chameleon/starting_point/*"

  hf download "${LUMINA_REPO}" --repo-type model \
    --local-dir "${CKPT_DIR}/models--Alpha-VLLM--Lumina-mGPT-7B-768"

  for suite in $(normalize_list "${LIBERO_SUITES}"); do
    [[ -n "${suite}" ]] || continue
    hf download "${RYNNVLA_REPO}" --repo-type model \
      --local-dir "${CKPT_DIR}" \
      --include "VLA_model_256/${suite}/*"
    if [[ "${DOWNLOAD_ACTION_WM}" == "1" ]]; then
      hf download "${RYNNVLA_REPO}" --repo-type model \
        --local-dir "${CKPT_DIR}" \
        --include "Action_World_model_512/${suite}/*"
    fi
  done
fi

if [[ "${DOWNLOAD_LIBERO}" == "1" ]]; then
  echo "[download_assets] LIBERO datasets -> ${LIBERO_DATASET_DIR}"
  if [[ ! -f "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" ]]; then
    echo "Missing third_party/LIBERO. Run scripts/install_env.sh first." >&2
    exit 2
  fi
  for suite in $(normalize_list "${LIBERO_SUITES}"); do
    [[ -n "${suite}" ]] || continue
    "${PYTHON}" "${DVLA_ROOT}/third_party/LIBERO/benchmark_scripts/download_libero_datasets.py" \
      --download-dir "${LIBERO_DATASET_DIR}" \
      --datasets "${suite}" --use-huggingface
  done
fi

if [[ "${DOWNLOAD_CALVIN}" == "1" ]]; then
  echo "[download_assets] CALVIN datasets"
  CALVIN_BASE_URL="${CALVIN_BASE_URL:-http://calvin.cs.uni-freiburg.de/dataset}"
  CALVIN_TASKS="${CALVIN_TASKS:-task_ABCD_D}"
  CALVIN_DIR="${CALVIN_DIR:-${DVLA_DATA_ROOT}/dataset/calvin}"
  mkdir -p "${CALVIN_DIR}"
  for task in $(normalize_list "${CALVIN_TASKS}"); do
    [[ -n "${task}" ]] || continue
    archive="${CALVIN_DIR}/${task}.zip"
    if [[ ! -f "${archive}" ]]; then
      curl -L -C - "${CALVIN_BASE_URL}/${task}.zip" -o "${archive}"
    fi
    if [[ "${EXTRACT_CALVIN:-0}" == "1" ]]; then
      "${PYTHON}" -m zipfile -e "${archive}" "${CALVIN_DIR}/${task}"
    fi
  done
fi

echo "[download_assets] complete"
```

Notes vs. old version: weights/datasets go to `${DVLA_DATA_ROOT}`; LIBERO gets
explicit `--download-dir` (the LIBERO downloader places suites in
`<download-dir>/<suite>/`); the CALVIN extract uses `${PYTHON}` (old script
had a bare `python`). No LIBERO config block needed — the downloader takes the
explicit dir.

- [ ] **Step 2: Syntax-check**

```bash
bash -n scripts/download_assets.sh
```
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/download_assets.sh
git commit -s -m "refactor: download assets into DVLA_DATA_ROOT"
```

---

### Task 4: Rewrite `scripts/install_env.sh` header

**Files:**
- Modify: `scripts/install_env.sh:1-7,45-46`

- [ ] **Step 1: Replace the sourcing header**

Replace lines 1–7:

```bash
#!/usr/bin/env bash
# Formal single-machine DreamerVLA environment installer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${DVLA_ROOT}"
```

with:

```bash
#!/usr/bin/env bash
# Formal single-machine DreamerVLA environment installer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
cd "${DVLA_ROOT}"
```

(The installer creates and activates its own conda env below, so no `PYTHON`
default or LIBERO block is needed here.)

- [ ] **Step 2: Point the wheel cache at the data root**

Replace:
```bash
  mkdir -p "${DVLA_ROOT}/data/wheels"
  FLASH_ATTN_WHEEL="${DVLA_ROOT}/data/wheels/$(basename "${FLASH_ATTN_WHEEL_URL}")"
```
with:
```bash
  mkdir -p "${DVLA_DATA_ROOT}/wheels"
  FLASH_ATTN_WHEEL="${DVLA_DATA_ROOT}/wheels/$(basename "${FLASH_ATTN_WHEEL_URL}")"
```

- [ ] **Step 3: Syntax-check and commit**

```bash
bash -n scripts/install_env.sh
git add scripts/install_env.sh
git commit -s -m "refactor: self-contained install_env without common_env.sh"
```

---

### Task 5: Rewrite the preprocess launchers

**Files:**
- Modify: `scripts/preprocess/prepare_libero_data.sh:1-44`
- Modify: `scripts/preprocess/process_all_libero_data.sh:19-53,73-75`

- [ ] **Step 1: Replace `prepare_libero_data.sh` header and path defaults**

Replace lines 1–44 (through the `ACTION_HIDDEN_GPUS` line) with:

```bash
#!/usr/bin/env bash
# One-command LIBERO preprocessing for the formal DreamerVLA data layout.
set -euo pipefail

# ---- environment (self-contained; no common_env.sh) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

# ---- LIBERO paths (datasets live under the data root) -----------------------
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/dataset/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
fi

cd "${DVLA_ROOT}"

TASK="${TASK:-libero_goal}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
HIS="${HIS:-1}"
ACTION_HORIZON="${ACTION_HORIZON:-1}"
FILTER_NOOPS="${FILTER_NOOPS:-1}"
RUN_MARKED="${RUN_MARKED:-1}"
RUN_REWARD="${RUN_REWARD:-1}"
RUN_PRETOKENIZE="${RUN_PRETOKENIZE:-1}"
RUN_ACTION_HIDDEN="${RUN_ACTION_HIDDEN:-1}"
OVERWRITE="${OVERWRITE:-0}"

case "${TASK}" in
  libero_goal|libero_object) DEFAULT_TIME_HORIZON=5 ;;
  libero_spatial|libero_10) DEFAULT_TIME_HORIZON=10 ;;
  *) DEFAULT_TIME_HORIZON="${ACTION_HORIZON}" ;;
esac
TIME_HORIZON="${TIME_HORIZON:-${DEFAULT_TIME_HORIZON}}"

PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${DVLA_DATA_ROOT}/processed_data}"
RAW_LIBERO_DIR="${RAW_LIBERO_DIR:-${DVLA_DATA_ROOT}/dataset/libero/${TASK}}"
MARKED_DIR="${MARKED_DIR:-${PROCESSED_DATA_ROOT}/${TASK}_marked_t_${IMAGE_RESOLUTION}}"
if [[ "${FILTER_NOOPS}" == "1" ]]; then
  HDF5_DIR="${HDF5_DIR:-${PROCESSED_DATA_ROOT}/${TASK}_no_noops_t_${IMAGE_RESOLUTION}}"
else
  HDF5_DIR="${HDF5_DIR:-${PROCESSED_DATA_ROOT}/${TASK}_with_noops_t_${IMAGE_RESOLUTION}}"
fi
REWARD_DIR="${REWARD_DIR:-${HDF5_DIR}_pi06_remaining_reward}"
HIDDEN_DIR="${HIDDEN_DIR:-${HDF5_DIR}_pi0_legacy_action_hidden_vla_policy_h2}"
META_JSON="${META_JSON:-${PROCESSED_DATA_ROOT}/${TASK}_metainfo.json}"
VLA_CKPT="${VLA_CKPT:-${DVLA_DATA_ROOT}/ckpts/VLA_model_256/${TASK}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${DVLA_DATA_ROOT}/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768}"
TEXT_TOKENIZER_PATH="${TEXT_TOKENIZER_PATH:-${DVLA_DATA_ROOT}/ckpts/chameleon/tokenizer/text_tokenizer.json}"
CHAMELEON_VQGAN_CONFIG="${CHAMELEON_VQGAN_CONFIG:-${DVLA_DATA_ROOT}/ckpts/chameleon/tokenizer/vqgan.yaml}"
CHAMELEON_VQGAN_CKPT="${CHAMELEON_VQGAN_CKPT:-${DVLA_DATA_ROOT}/ckpts/chameleon/tokenizer/vqgan.ckpt}"
ACTION_HIDDEN_GPUS="${ACTION_HIDDEN_GPUS:-${NGPU:-1}}"
```

Then replace the next line
`mkdir -p "${PROCESSED_DATA_ROOT}" "${DVLA_ROOT}/data/logs/libero_data_prep"`
with
`mkdir -p "${PROCESSED_DATA_ROOT}" "${DVLA_DATA_ROOT}/logs/libero_data_prep"`.
Everything from the first `echo "[prepare_libero_data] ..."` to the end of the
file stays unchanged (the raw-dir change is in defaults above:
`RAW_LIBERO_DIR` now points at `${DVLA_DATA_ROOT}/dataset/libero/${TASK}`
instead of `third_party/LIBERO/libero/datasets/${TASK}`).

- [ ] **Step 2: Replace `process_all_libero_data.sh` env header**

Replace lines 19–53 (from `set -uo pipefail` through the `mkdir -p
"${CONVS_DIR}" ...` line) with:

```bash
set -uo pipefail

# ---- environment (self-contained; no common_env.sh) -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
PROJECT_ROOT="${DVLA_ROOT}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
PYTHON="${PYTHON:-python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
cd "${PROJECT_ROOT}"

LOG_DIR="${DVLA_DATA_ROOT}/logs/libero_data_prep"
mkdir -p "${LOG_DIR}"

SUITES="${SUITES:-libero_10 libero_object libero_spatial}"
GPUS="${GPUS:-4,5}"
PRETOKENIZE_PROCS="${PRETOKENIZE_PROCS:-16}"
IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-256}"
ACTION_HORIZON="${ACTION_HORIZON:-1}"           # atomic pretokenize
HIS="${HIS:-1}"
FORCE="${FORCE:-0}"

TOKENIZER_PATH="${DVLA_DATA_ROOT}/ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768"
PROCESSED_DATA_ROOT="${DVLA_DATA_ROOT}/processed_data"

CONVS_DIR="${PROCESSED_DATA_ROOT}/convs"
TOKENS_DIR="${PROCESSED_DATA_ROOT}/tokens"
CONCATE_DIR="${PROCESSED_DATA_ROOT}/concate_tokens"
mkdir -p "${CONVS_DIR}" "${TOKENS_DIR}" "${CONCATE_DIR}"
```

(This drops the old `CONDA_SH` activation block — env activation is the
user's job now. `PROJECT_ROOT` is kept as an alias so the rest of the script
is untouched.)

- [ ] **Step 3: Point the generated yaml-config dir at the data root**

In `process_one_suite()`, replace
`local CONFIG_DIR="${PROJECT_ROOT}/data/configs/${SUITE}"` with
`local CONFIG_DIR="${DVLA_DATA_ROOT}/configs/${SUITE}"`.

- [ ] **Step 4: Verify no other `data/` literals remain in either script**

```bash
bash -n scripts/preprocess/prepare_libero_data.sh scripts/preprocess/process_all_libero_data.sh
grep -n 'DVLA_ROOT}/data\|PROJECT_ROOT}/data\|common_env' \
  scripts/preprocess/prepare_libero_data.sh scripts/preprocess/process_all_libero_data.sh || echo OK
```
Expected: `bash -n` silent; `OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/preprocess/prepare_libero_data.sh scripts/preprocess/process_all_libero_data.sh
git commit -s -m "refactor: preprocess launchers read DVLA_DATA_ROOT"
```

---

### Task 6: Rewrite `scripts/eval/launch_openvla_oft_official_libero_eval.sh` header

**Files:**
- Modify: `scripts/eval/launch_openvla_oft_official_libero_eval.sh:12-26,73-77`

This 200-line script only needs its env plumbing swapped; worker/tmux logic is
untouched.

- [ ] **Step 1: Replace the sourcing header (lines 12–26)**

Replace:
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/../common_env.sh"

ROOT="${ROOT:-${DVLA_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON}}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-${ROOT}/third_party/openvla-oft}"

SUITE="${SUITE:-libero_goal}"
case "${SUITE}" in
  libero10|libero_long) SUITE="libero_10" ;;
esac
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/data/ckpts/Openvla-oft-SFT-traj1}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/outputs/eval/openvla_oft_official_libero}"
STAGED_CKPT_ROOT="${STAGED_CKPT_ROOT:-${ROOT}/data/tmp_ckpts/openvla_oft_official_eval}"
USE_STAGED_CKPT="${USE_STAGED_CKPT:-1}"
```
with:
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
case ":${PYTHONPATH:-}:" in
  *":${DVLA_ROOT}:"*) ;;
  *) export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

# ---- LIBERO paths (datasets live under the data root) -----------------------
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ "${DREAMERVLA_WRITE_LIBERO_CONFIG:-1}" == "1" ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/dataset/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF
fi

ROOT="${ROOT:-${DVLA_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
OPENVLA_OFT_ROOT="${OPENVLA_OFT_ROOT:-${ROOT}/third_party/openvla-oft}"

SUITE="${SUITE:-libero_goal}"
case "${SUITE}" in
  libero10|libero_long) SUITE="libero_10" ;;
esac
CKPT_ROOT="${CKPT_ROOT:-${DVLA_DATA_ROOT}/ckpts/Openvla-oft-SFT-traj1}"
OUT_ROOT="${OUT_ROOT:-${DVLA_DATA_ROOT}/outputs/eval/openvla_oft_official_libero}"
STAGED_CKPT_ROOT="${STAGED_CKPT_ROOT:-${DVLA_DATA_ROOT}/tmp_ckpts/openvla_oft_official_eval}"
USE_STAGED_CKPT="${USE_STAGED_CKPT:-1}"
```

- [ ] **Step 2: Pin the renderer defaults where the old common_env defaults applied**

Old behavior: `common_env.sh` set `MUJOCO_GL=egl` first, so the script's later
`MUJOCO_GL="${MUJOCO_GL:-osmesa}"` effectively yielded **egl**. Preserve that:
replace (around old lines 73–76):
```bash
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
```
with:
```bash
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
```

- [ ] **Step 3: Pass `DVLA_DATA_ROOT` through to tmux workers**

In `launch_worker_session()`, the long `env ...` string already forwards
`ROOT='${ROOT}'`; add `DVLA_ROOT='${DVLA_ROOT}' DVLA_DATA_ROOT='${DVLA_DATA_ROOT}'`
right after `env DREAMERVLA_OFFICIAL_LIBERO_WORKER=1 `.

- [ ] **Step 4: Syntax-check and commit**

```bash
bash -n scripts/eval/launch_openvla_oft_official_libero_eval.sh
grep -n common_env scripts/eval/launch_openvla_oft_official_libero_eval.sh || echo OK
git add scripts/eval/launch_openvla_oft_official_libero_eval.sh
git commit -s -m "refactor: self-contained OFT official eval launcher"
```

---

### Task 7: Deprecation note on `common_env.sh`

**Files:**
- Modify: `scripts/common_env.sh:1-2`

- [ ] **Step 1: Update the header comment**

Replace line 2
(`# Shared environment defaults for formal DreamerVLA shell entrypoints.`)
with:

```bash
# DEPRECATED for formal entrypoints — they are now self-contained and read
# DVLA_DATA_ROOT (see docs/data_layout.md). Kept only for the legacy
# machine-specific scripts (*_45.sh, *_g67.sh, smoke/, archive/) that still
# source it.
```

- [ ] **Step 2: Commit**

```bash
git add scripts/common_env.sh
git commit -s -m "docs: mark common_env.sh deprecated for formal entrypoints"
```

---

### Task 8: Write `docs/data_layout.md` (portable dataset manifest)

**Files:**
- Create: `docs/data_layout.md`

- [ ] **Step 1: Write the manifest**

```markdown
# Data layout (`DVLA_DATA_ROOT`)

Single source of truth for everything DreamerVLA reads or writes outside the
repo. Code and data are connected by exactly one environment variable:

| Variable | Meaning | Default |
|---|---|---|
| `DVLA_ROOT` | Project root (code). Auto-derived by every formal script. | repo checkout dir |
| `DVLA_DATA_ROOT` | Data root — every path below lives under it. | `${DVLA_ROOT}/data` |

Migrating to a new machine = rsync `${DVLA_DATA_ROOT}` + clone the repo + set
the variable. Provisioning from scratch = run the Provenance commands below.

## Directory tree

​```
${DVLA_DATA_ROOT}/
├── ckpts/                          # pretrained weights (downloaded)
│   ├── VLA_model_256/<suite>/
│   ├── Action_World_model_512/<suite>/
│   ├── chameleon/tokenizer/
│   ├── models--Alpha-VLLM--Lumina-mGPT-7B-768/
│   ├── OpenVLA-OFT/<run>/
│   └── Openvla-oft-SFT-traj1/<name>/
├── dataset/
│   ├── libero/<suite>/             # raw LIBERO demos (downloaded)
│   └── calvin/                     # optional
├── processed_data/                 # generated by preprocessing
│   ├── <suite>_marked_t_256/
│   ├── <suite>_no_noops_t_256/
│   ├── <suite>_no_noops_t_256_pi06_remaining_reward/
│   ├── <suite>_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2/
│   ├── <suite>_image_state_action_t_256/
│   ├── convs/  tokens/  concate_tokens/
│   └── <suite>_metainfo.json
├── configs/<suite>/                # generated pretokenize dataset configs
├── outputs/                        # training/eval runs (checkpoints, logs)
├── logs/                           # preprocessing logs
├── wheels/                         # cached flash-attn wheel
├── tmp_ckpts/                      # staged eval checkpoints (scratch)
└── .libero/config.yaml             # generated LIBERO path config
​```

## Asset classes

### Pretrained checkpoints — `ckpts/`

| Path | Content / format | Needed by | Provenance |
|---|---|---|---|
| `ckpts/VLA_model_256/<suite>/` | RynnVLA-002 VLA init (HF safetensors + config) | VLA SFT, action-hidden sidecar, online RL | `bash scripts/download_assets.sh` (HF `Alibaba-DAMO-Academy/RynnVLA-002`) |
| `ckpts/Action_World_model_512/<suite>/` | RynnVLA action-WM init | optional WM experiments | same, `DOWNLOAD_ACTION_WM=1` (default) |
| `ckpts/chameleon/tokenizer/` | `text_tokenizer.json`, `vqgan.yaml`, `vqgan.ckpt` | pretokenize, VLA SFT, sidecar | same (HF `Alibaba-DAMO-Academy/WorldVLA`) |
| `ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/` | Lumina-mGPT tokenizer/backbone files | pretokenize, VLA SFT | same (HF `Alpha-VLLM/Lumina-mGPT-7B-768`) |
| `ckpts/OpenVLA-OFT/<run>/` | OFT SFT checkpoint: HF model dir + `action_head--*.pt`, `proprio_projector--*.pt`, `dataset_statistics.json` | OFT routes (`configs/task/*.yaml: openvla_oft.*`) | produced by `CONFIG=openvla_oft_hdf5 bash scripts/train_vla.sh` |
| `ckpts/Openvla-oft-SFT-traj1/<name>/` | pretrained one-trajectory OFT eval ckpts (merged discrete head) | OFT official eval only — NOT usable for the WM chain | `hf download Haozhan72/...` (see SETUP.md) |

### Raw datasets — `dataset/`

| Path | Content / format | Needed by | Provenance |
|---|---|---|---|
| `dataset/libero/<suite>/*.hdf5` | raw LIBERO demos; HDF5, one file per task, demos under `data/demo_*` | all preprocessing | `bash scripts/download_assets.sh` (HF `yifengzhu-hf/LIBERO-datasets`, Box fallback); suites: `libero_goal`, `libero_object`, `libero_spatial`, `libero_10` |
| `dataset/calvin/` | CALVIN zips | optional CALVIN experiments | `DOWNLOAD_CALVIN=1 bash scripts/download_assets.sh` |

### Preprocessed products — `processed_data/` (generated, per suite)

All produced by `TASK=<suite> bash scripts/preprocess/prepare_libero_data.sh`
(stages 1–5). Stage 4 fans out to `process_all_libero_data.sh`.

| Path | Content / format | Stage |
|---|---|---|
| `<suite>_marked_t_256/` | replayed demos with no-op marks; HDF5 | 1 |
| `<suite>_no_noops_t_256/` | filtered training view; HDF5 | 2 |
| `<suite>_no_noops_t_256_pi06_remaining_reward/` | + per-step remaining-steps reward; HDF5 | 3 |
| `<suite>_image_state_action_t_256/` | extracted images/state/action trees (png + npy per task) | 4 |
| `convs/*.json` | conversation JSONs (train / val_ind / val_ood) | 4 |
| `tokens/<run>/files/*.pkl`, `tokens/<run>/record.json` | pretokenized samples | 4 |
| `concate_tokens/<suite>_his_1_*.json` | concat manifest | 4 |
| `<suite>_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2/` | action-hidden sidecar (token_dim 1024, history 2, action_query source — must match `task.legacy_action_hidden.*` expectations) | 5 |
| `<suite>_metainfo.json` | task metainfo (init states / language) | 1 |
| `<suite>_no_noops_t_256_failures*` | optional failure rollouts + sidecar for the latent classifier | see `docs/classifier_revision_plan.md` |

### Generated configs — `configs/<suite>/`

`his_1_third_view_wrist_w_state_1_256_pretokenize{,_val_ind,_val_ood}.yaml`,
written by preprocessing stage 5; referenced by
`configs/task/<suite>.yaml: pretokenize_*_config_path`. YAML with absolute
manifest paths — regenerate (rerun stage 4–5) after moving the data root,
or sed the old root inside them.

### Outputs — `outputs/`

Training runs: `outputs/<route>/<run>/checkpoints/` (+ `ema/` when
`training.use_ema=true`), JSON logs. Eval runs under `outputs/eval/`.
Safe to prune old runs; nothing else references them except `init.*_ckpt`
fields you set explicitly.

### LIBERO path config — `.libero/config.yaml`

Regenerated by every formal script launch (`DREAMERVLA_WRITE_LIBERO_CONFIG=0`
to keep manual edits). Points LIBERO at `${DVLA_ROOT}/third_party/LIBERO`
internals and `datasets: ${DVLA_DATA_ROOT}/dataset/libero`.

## Migration checklist

1. `rsync -a old:${DVLA_DATA_ROOT}/ new:/path/to/data_root/`
2. `git clone` the repo; `bash scripts/install_env.sh`
3. `export DVLA_DATA_ROOT=/path/to/data_root` (add to your shell rc)
4. Pretokenize manifests embed absolute paths — if the data-root path changed,
   rerun preprocessing stage 4–5 or sed the old prefix in
   `processed_data/{tokens,concate_tokens}/**/*.json` and `configs/<suite>/*.yaml`.
5. Verify: `python -m dreamer_vla.cli.train --config-name vla_rynnvla_action_head --cfg job | grep ckpts`
```

(Replace the `​```` fence escapes with normal fences when writing the file.)

- [ ] **Step 2: Commit**

```bash
git add docs/data_layout.md
git commit -s -m "docs: add portable data-layout manifest"
```

---

### Task 9: Update SETUP.md, scripts/README.md, README.md

**Files:**
- Modify: `SETUP.md` (env section ~line 33; verification block ~line 235)
- Modify: `scripts/README.md` (common_env mention)
- Modify: `README.md` (common_env mention)

- [ ] **Step 1: SETUP.md — add quickstart, replace common_env section**

At the top of SETUP.md (after the title/intro), insert:

```markdown
## 新机器 5 步快速开始

​```bash
git clone <repo> && cd DreamerVLA
bash scripts/install_env.sh                       # conda env + 依赖 + third_party
conda activate dreamervla
export DVLA_DATA_ROOT=/mnt/bigdisk/dvla_data      # 可选；不设则用 repo/data
bash scripts/download_assets.sh                   # 权重 + 数据集 → 数据根
bash scripts/preprocess/prepare_libero_data.sh    # 预处理产物 → 数据根
bash scripts/train_vla.sh                         # 开始训练
​```

代码与数据由唯一变量 `DVLA_DATA_ROOT` 连接（默认 `${DVLA_ROOT}/data`）。
完整数据布局与迁移清单见 [docs/data_layout.md](docs/data_layout.md)。
```

Then replace the section at line ~33 describing `scripts/common_env.sh`
(through its variable list) with:

```markdown
正式 shell 入口均为自包含脚本（不再 source `scripts/common_env.sh`）：脚本顶部
自行推导 `DVLA_ROOT`、默认 `DVLA_DATA_ROOT=${DVLA_ROOT}/data`、设置
`PYTHONPATH` / `MUJOCO_GL`，并生成 LIBERO 路径配置
（`${DVLA_DATA_ROOT}/.libero/config.yaml`，`datasets:` 指向
`${DVLA_DATA_ROOT}/dataset/libero`）。运行前先 `conda activate dreamervla`。
```

Read the surrounding section first and adapt the splice points to the actual
prose; preserve everything else. Update the verification block at ~line 235 to
use the data root:

```bash
test -d "${DVLA_DATA_ROOT:-data}/ckpts/VLA_model_256/libero_goal"
test -d "${DVLA_DATA_ROOT:-data}/processed_data/libero_goal_no_noops_t_256"
test -d "${DVLA_DATA_ROOT:-data}/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward"
test -d "${DVLA_DATA_ROOT:-data}/processed_data/libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2"
```
and the literal `data/processed_data/...` path mentions at ~lines 81–85 and
the `--local-dir data/ckpts/...` example at ~line 150 to
`${DVLA_DATA_ROOT:-data}/...` equivalents.

- [ ] **Step 2: scripts/README.md and README.md**

In both files, find the `common_env.sh` mentions (`grep -n common_env
scripts/README.md README.md`) and rewrite those sentences to state: formal
entrypoints are self-contained; paths are controlled by `DVLA_DATA_ROOT`
(default `repo/data`); see `docs/data_layout.md`. Keep surrounding content
unchanged.

- [ ] **Step 3: Cross-check docs against code**

```bash
grep -rn 'common_env' SETUP.md README.md scripts/README.md docs/data_layout.md
```
Expected: only mentions that describe it as deprecated/legacy.

- [ ] **Step 4: Commit**

```bash
git add SETUP.md README.md scripts/README.md
git commit -s -m "docs: document DVLA_DATA_ROOT and self-contained launchers"
```

---

### Task 10: Reconcile this machine's live layout + end-to-end verification

The live `data/` dir predates the canonical layout: `processed_data/` and
`configs/` sit under `data/dataset/`, and raw LIBERO data is under
`data/dataset/libero/datasets/` and/or `third_party/LIBERO/libero/datasets/`.
Reconcile with symlinks (never move/delete data).

- [ ] **Step 1: Inspect what actually exists**

```bash
ls -la data/ data/dataset/ data/dataset/libero/ 2>/dev/null
ls third_party/LIBERO/libero/datasets/ 2>/dev/null
ls data/dataset/processed_data/ 2>/dev/null | head
```

- [ ] **Step 2: Create reconciling symlinks (adapt to Step 1 findings)**

```bash
# canonical processed_data -> live location (only if target exists and canonical missing)
[[ -d data/processed_data || ! -d data/dataset/processed_data ]] \
  || ln -s dataset/processed_data data/processed_data
# canonical generated-configs dir
[[ -d data/configs || ! -d data/dataset/configs ]] \
  || ln -s dataset/configs data/configs
# canonical raw LIBERO suites: data/dataset/libero/<suite>
for suite in libero_goal libero_object libero_spatial libero_10; do
  if [[ ! -e "data/dataset/libero/${suite}" ]]; then
    if [[ -d "data/dataset/libero/datasets/${suite}" ]]; then
      ln -s "datasets/${suite}" "data/dataset/libero/${suite}"
    elif [[ -d "third_party/LIBERO/libero/datasets/${suite}" ]]; then
      ln -s "../../../third_party/LIBERO/libero/datasets/${suite}" "data/dataset/libero/${suite}"
    fi
  fi
done
ls -la data/ data/dataset/libero/
```
Expected: `data/processed_data`, `data/configs`, and per-suite
`data/dataset/libero/<suite>` all resolve (`ls -L` works on each).

- [ ] **Step 3: Full verification sweep**

```bash
# 1. syntax
bash -n scripts/train_vla.sh scripts/train_wm.sh scripts/train_dreamervla.sh \
  scripts/eval_libero_vla.sh scripts/download_assets.sh scripts/install_env.sh \
  scripts/preprocess/prepare_libero_data.sh scripts/preprocess/process_all_libero_data.sh \
  scripts/eval/launch_openvla_oft_official_libero_eval.sh
# 2. no formal script references common_env
grep -l common_env scripts/*.sh scripts/preprocess/*.sh scripts/eval/launch_openvla_oft_official_libero_eval.sh \
  | grep -v -E '_(45|g67)|common_env.sh' || echo OK
# 3. config residue
grep -rn 'oc.env:DVLA_ROOT' configs/ || echo OK
# 4. resolution, both modes (paths must exist locally for the unset case)
python -m dreamer_vla.cli.train --config-name vla_rynnvla_action_head --cfg job | grep -m3 ckpts
DVLA_DATA_ROOT=/tmp/dvla_data_test python -m dreamer_vla.cli.train \
  --config-name vla_rynnvla_action_head --cfg job | grep -m3 /tmp/dvla_data_test
# 5. data presence with reconciled symlinks
test -d data/ckpts/VLA_model_256/libero_goal && echo ckpts-ok
test -d data/processed_data/libero_goal_no_noops_t_256 && echo processed-ok
# 6. LIBERO env smoke (writes .libero config via a formal script first)
DREAMERVLA_WRITE_LIBERO_CONFIG=1 CONFIG=eval_libero_vla bash -c \
  'source /dev/stdin <<< "$(sed -n "/---- environment/,/^fi$/p" scripts/eval_libero_vla.sh)"; \
   cat "${LIBERO_CONFIG_PATH}/config.yaml"'
python scripts/smoke/smoke_libero_online_env.py
```
Expected: all OK / ok markers; smoke script completes without path errors.
If step 5/6 fail because this machine's live data genuinely lacks those dirs,
record what's missing in the final report rather than fabricating links.

- [ ] **Step 4: Commit (symlinks are not tracked — commit only if any tracked file changed)**

```bash
git status --short
# expect empty (data/ is gitignored); nothing to commit for symlinks
```

---

### Task 11: First-person new-machine migration walkthrough + machine-specificity audit

Goal: simulate "I just cloned this repo on a brand-new machine with an empty
data disk" and surface every point of friction; verify no mainline code is
specific to this machine.

- [ ] **Step 1: Fresh-clone dry-run (no data)**

```bash
TMP=$(mktemp -d)
git clone --no-hardlinks . "${TMP}/DreamerVLA" && cd "${TMP}/DreamerVLA"
# Walk the SETUP.md quickstart on paper: which commands would fail, in what
# order, with what message? Run the cheap ones for real:
bash -n scripts/*.sh scripts/preprocess/*.sh
DVLA_DATA_ROOT="${TMP}/data" python -m dreamer_vla.cli.train \
  --config-name vla_rynnvla_action_head --cfg job | grep -m3 "${TMP}/data" || echo "RESOLUTION-FAIL"
cd - && rm -rf "${TMP}"
```

- [ ] **Step 2: Machine-specificity sweep over mainline code**

```bash
# absolute paths bound to this machine
grep -rn '/mnt/data\|/home/user01\|/data/spoil' --include='*.sh' --include='*.yaml' --include='*.py' \
  scripts/ configs/ dreamer_vla/ | grep -v -E '_(45|g67)|archive/|smoke/|wm_variants'
# hardcoded GPU ids / device lists in formal scripts
grep -n 'GPUS=\|GPU_A=\|GPU_B=\|CUDA_VISIBLE_DEVICES=' scripts/*.sh scripts/preprocess/*.sh \
  scripts/eval/launch_openvla_oft_official_libero_eval.sh
# configs defaulting to this machine's past run artifacts (timestamped out_dirs)
grep -rn 'data/outputs/.*20[0-9]\{6\}' configs/
```

- [ ] **Step 3: Triage findings**

In scope (fix now): machine-bound defaults inside the 9 formal scripts.
Out of scope (report only): legacy `_45/_g67` scripts, `init.*_ckpt` config
defaults pointing at past local runs (these are experiment pins, but on a new
machine they 404 — recommend follow-up), anything in `third_party/`.

- [ ] **Step 4: Write the first-person walkthrough**

Deliver in the final report, structured as: step-by-step "I do X → I hit Y"
for the 5-step quickstart, then the machine-specificity findings table
(file:line, why it breaks on a new machine, fixed / reported).

---

## Self-review notes

- Spec coverage: env contract (T1–T6), self-contained scripts (T2–T6),
  common_env deprecation (T7), data manifest (T8), docs + quickstart (T9),
  migration + verification (T10). LIBERO raw relocation: T3 (download dir),
  T5 (RAW_LIBERO_DIR), env blocks (`datasets:` field), T10 (symlinks).
- The pretokenize-manifest absolute-path caveat surfaced while drafting the
  manifest (T8): generated record/manifest JSONs embed absolute paths, so
  data-root moves need a regenerate-or-sed step. Documented in
  `docs/data_layout.md` Migration checklist; no code change required.
```
