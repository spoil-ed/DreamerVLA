# Portable paths + self-contained launch scripts

Date: 2026-06-11
Status: approved

## Goal

Make DreamerVLA easy to migrate between machines and easy to set up on a new
machine. Two changes accomplish this:

1. **Decouple project root from data root.** Code (`DVLA_ROOT`) and data
   (`DVLA_DATA_ROOT`) connect through a single environment variable. Migrating
   = rsync one directory + set one variable.
2. **Self-contained launch scripts (RLinf style).** Each formal entry script
   carries its own environment block at the top — visible, editable, no hidden
   `source common_env.sh` indirection.

## Environment variable contract

| Variable | Meaning | Default |
|---|---|---|
| `DVLA_ROOT` | Project root (code). | Auto-derived from script location: `$(dirname "${SCRIPT_DIR}")` |
| `DVLA_DATA_ROOT` | Data root: `ckpts/`, `dataset/`, `processed_data/`, `outputs/`, `logs/`, `.libero/` | `${DVLA_ROOT}/data` |

- Zero-config default: with no variables set, everything lands in `repo/data`
  — identical to today's behavior.
- LIBERO raw HDF5 canonical location moves from
  `third_party/LIBERO/libero/datasets` to
  `${DVLA_DATA_ROOT}/dataset/libero/<suite>`. The generated LIBERO
  `config.yaml` (`datasets:` field) and the download script's
  `--download-dir` both point there.
- No conda autodetection. The user activates the env first
  (`conda activate dreamervla`); each script prints `which python` at startup
  for diagnosis. Rationale: the old probe hardcoded
  `~/miniconda3/envs/dreamervla` and silently failed on machines where conda
  lives elsewhere.

## Script template

Every formal entry script is self-contained:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ---- paths -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DVLA_ROOT="${DVLA_ROOT:-$(dirname "${SCRIPT_DIR}")}"
export DVLA_DATA_ROOT="${DVLA_DATA_ROOT:-${DVLA_ROOT}/data}"
export PYTHONPATH="${DVLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- runtime -----------------------------------------------
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

# ---- LIBERO (only in scripts that touch LIBERO sim/data) ----
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${DVLA_DATA_ROOT}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${DVLA_ROOT}/third_party/LIBERO/libero/libero
bddl_files: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/bddl_files
init_states: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/init_files
datasets: ${DVLA_DATA_ROOT}/dataset/libero
assets: ${DVLA_ROOT}/third_party/LIBERO/libero/libero/assets
EOF

# ---- knobs (per-script tunables, RLinf style) ----------------
CONFIG="${CONFIG:-vla_rynnvla_action_head}"
NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "Using Python at $(which python)"
# ... launch ...
```

Scripts that never touch LIBERO sim/data (e.g. pure WM training on
preprocessed shards) omit the LIBERO block. Each script keeps only the knobs
it actually uses.

## Config plumbing

Pure text replacement across `configs/*.yaml` and `configs/task/*.yaml`
(~70 references, 18 files):

```
${oc.env:DVLA_ROOT,.}/data/   →   ${oc.env:DVLA_DATA_ROOT,data}/
```

When running `python -m dreamer_vla.cli.train` directly without a wrapper
script and without the variable set, paths fall back to relative `data/` from
the CWD — same as today's `./data` behavior. The existing nested-resolver
pattern `${oc.env:OUT_DIR,${oc.env:DVLA_ROOT,.}/data/outputs/...}` becomes
`${oc.env:OUT_DIR,${oc.env:DVLA_DATA_ROOT,data}/outputs/...}`.

No Python code reads `DVLA_ROOT` directly; configs and shell are the only
consumers.

## File-by-file changes

**Rewritten as self-contained (9 formal entries):**

- `scripts/train_vla.sh`
- `scripts/train_wm.sh`
- `scripts/train_dreamervla.sh`
- `scripts/eval_libero_vla.sh`
- `scripts/download_assets.sh`
- `scripts/install_env.sh`
- `scripts/preprocess/prepare_libero_data.sh`
- `scripts/preprocess/process_all_libero_data.sh`
- `scripts/eval/launch_openvla_oft_official_libero_eval.sh`

**`download_assets.sh` targets:**

- HF weights → `${DVLA_DATA_ROOT}/ckpts`
- LIBERO datasets → `--download-dir ${DVLA_DATA_ROOT}/dataset/libero`
- CALVIN → `${DVLA_DATA_ROOT}/dataset/calvin`

**`prepare_libero_data.sh`:** `RAW_LIBERO_DIR` default becomes
`${DVLA_DATA_ROOT}/dataset/libero/${TASK}`; `PROCESSED_DATA_ROOT` default
becomes `${DVLA_DATA_ROOT}/processed_data`; ckpt path defaults switch to
`${DVLA_DATA_ROOT}/ckpts/...`. All remain overridable.

**`common_env.sh`:** kept in place untouched except a one-line header comment
noting formal entries no longer use it. The machine-suffixed one-off scripts
(`*_45.sh`, `*_g67.sh`, smoke, archive) still source it and are out of scope.

**Configs:** the text replacement above.

**Docs:** update `SETUP.md` (add "new machine in 5 steps" quickstart at top,
document `DVLA_DATA_ROOT`), `scripts/README.md`, and `README.md` references
to `common_env.sh` / data layout.

**New: `docs/data_layout.md` — portable dataset manifest.** A single
self-contained document that fully specifies the data root, so a new machine
can be provisioned (or an old one audited) from this file alone. For every
asset class it records:

- **Content**: what it is and which training route needs it
  (required / optional per route).
- **Canonical path** under `${DVLA_DATA_ROOT}` (the single source of truth;
  configs and scripts must agree with this file).
- **Format**: HDF5 / safetensors / pt / json / yaml, plus key internal
  structure where it matters (e.g. HDF5 demo keys, sidecar tensor shapes).
- **Provenance**: the exact command that downloads it
  (`download_assets.sh` flags) or the script that generates it
  (`prepare_libero_data.sh` stage), with upstream repo names.

Covered asset classes: pretrained ckpts (RynnVLA `VLA_model_256`,
`Action_World_model_512`, WorldVLA chameleon tokenizer, Lumina-mGPT,
OpenVLA-OFT), raw LIBERO HDF5 per suite, optional CALVIN, preprocessed
products (marked / no_noops / reward / action-hidden sidecars / pretokenize
configs), task metainfo JSONs, and training outputs. Linked from `SETUP.md`.

Note: the current machine's live layout deviates from the canonical one
(e.g. `processed_data` sits under `data/dataset/` while configs expect
`data/processed_data`). Implementation reconciles via symlinks and the
manifest documents only the canonical layout.

## Migration note (existing machines)

On machines with LIBERO raw data already under
`third_party/LIBERO/libero/datasets`, create a one-time symlink (do not move
data):

```bash
mkdir -p "${DVLA_DATA_ROOT:-data}/dataset"
ln -s "$(pwd)/third_party/LIBERO/libero/datasets" "${DVLA_DATA_ROOT:-data}/dataset/libero"
```

Documented in SETUP.md.

## New-machine story (acceptance narrative)

```bash
git clone <repo> && cd DreamerVLA
bash scripts/install_env.sh
export DVLA_DATA_ROOT=/mnt/bigdisk/dvla_data   # optional
bash scripts/download_assets.sh
bash scripts/preprocess/prepare_libero_data.sh
bash scripts/train_vla.sh
```

## Verification

1. `bash -n` every rewritten script → syntax clean.
2. `python -m dreamer_vla.cli.train --config-name <X> --cfg job` with
   `DVLA_DATA_ROOT` unset and set to an external dir → all resolved paths
   switch correctly in both cases.
3. `grep -r "oc.env:DVLA_ROOT" configs/` → zero remaining references.
4. After the symlink migration on this machine, run
   `scripts/smoke/smoke_libero_online_env.py` → LIBERO path resolution works.

## Out of scope

- Machine-suffixed one-off scripts (`*_45.sh`, `*_g67.sh`), `scripts/smoke/`,
  `scripts/archive/`, `scripts/wm_variants_v4_v4E/`.
- Splitting outputs into a separate `DVLA_OUTPUT_ROOT` (decided against:
  one variable, one directory).
- Hydra `paths:` config group (decided against: larger sweep, higher
  regression risk than env-var text replacement).
