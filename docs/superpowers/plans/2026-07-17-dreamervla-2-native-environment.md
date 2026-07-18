# DreamerVLA-2 Native Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a fresh `dreamervla-2` Conda environment through every repository setup stage, run bounded real asset/training flow, repair proven dependency gaps, and publish a detailed native reproduction record.

**Architecture:** The existing Hydra workflow remains the setup and reproduction source of truth. Host-specific logs and snapshots are written under ignored `data/reproduction/environment/dreamervla-2/`; durable conclusions and commands are written under `docs/`. Setup changes are test-first and limited to failures reproduced in the new environment.

**Tech Stack:** Bash, Conda, uv/pip, Hydra/OmegaConf, PyTorch 2.5.1 cu124, Ray 2.55.1, OpenVLA-OFT, LIBERO, pytest, Ruff.

## Global Constraints

- The environment name is exactly `dreamervla-2`.
- Execute every installer stage `00_apt_tools` through `60_verify` with fresh evidence.
- Temporary clean clones verify source acquisition but must not become editable runtime paths.
- Real bounded training replaces full 30/8/20,000 budgets; dry-run and fixtures do not count.
- Never edit `third_party/` or overwrite existing data/checkpoints.
- Add dependencies only after reproducing and root-causing an actual missing/incompatible package.
- Preserve unrelated workspace changes and do not commit without explicit repository-owner direction.

---

### Task 1: Baseline and Evidence Directory

**Files:**
- Create at runtime: `data/reproduction/environment/dreamervla-2/logs/`
- Create at runtime: `data/reproduction/environment/dreamervla-2/snapshots/`
- Update: `.planning/2026-07-17-dreamervla-2-environment/{task_plan,findings,progress}.md`

**Interfaces:**
- Consumes: current checkout and host state.
- Produces: immutable pre-run facts and an isolated evidence root used by all later tasks.

- [ ] **Step 1: Capture the pre-run baseline**

Run read-only commands for `git status`, `git rev-parse HEAD`, `/etc/os-release`,
`uname -a`, `conda --version`, `conda env list`, `nvidia-smi`, `df -h`, and relevant
environment variables. Save the combined transcript in
`data/reproduction/environment/dreamervla-2/snapshots/host-before.txt`.

- [ ] **Step 2: Confirm target isolation**

Run `conda env list` and verify that `dreamervla-2` is absent before creation. If it
already exists, inspect it rather than deleting it; use the install stages to repair
it in place and record that the precondition differed.

- [ ] **Step 3: Record repository-owned pins**

Save the resolved install Hydra config and the configured third-party revisions in
the evidence directory. Expected: Python 3.11, Torch 2.5.1/cu124, flash-attn
2.7.1.post1, Ray 2.55.1, and the commits in `configs/scripts/install/config.yaml`.

### Task 2: Clean Third-Party Acquisition Audit

**Files:**
- Create at runtime: `data/reproduction/environment/dreamervla-2/snapshots/third-party-clean-clones.txt`

**Interfaces:**
- Consumes: URLs and revisions from `scripts/install/40_third_party.sh` and `configs/scripts/install/config.yaml`.
- Produces: source-clone proof independent of existing `third_party/` directories.

- [ ] **Step 1: Create a safe temporary root**

Use `mktemp -d /tmp/dreamervla-2-third-party.XXXXXX` and validate that the returned
path starts with `/tmp/dreamervla-2-third-party.` before placing clones below it.

- [ ] **Step 2: Clone and check out mandatory repositories**

Clone LIBERO, robosuite, robosuite-task-zoo, robomimic, mimicgen, OpenVLA-OFT,
egl_probe, dlimp_openvla, and transformers-openvla-oft from the installer-declared
URLs. Check out the exact configured revision for each and record `remote.origin.url`
plus `rev-parse HEAD`.

- [ ] **Step 3: Compare clean and configured revisions**

Verify every clean-clone SHA equals its configured pin. A mismatch is a setup
source failure; do not continue with an unpinned branch head.

- [ ] **Step 4: Remove only the validated temporary root**

After the snapshot is safely stored, remove the explicit validated `mktemp` path.
Do not remove or modify repository `third_party/` directories.

### Task 3: Execute Every Installer Stage

**Files:**
- Create at runtime: `data/reproduction/environment/dreamervla-2/logs/install-<stage>.log`
- Potentially modify after a proven failure: `requirements.txt`, `pyproject.toml`, or one file under `scripts/install/`
- Test before any setup fix: `tests/unit_tests/test_setup_scripts.py` or `tests/unit_tests/test_verify_install.py`

**Interfaces:**
- Consumes: `env.CONDA_ENV_NAME=dreamervla-2` and repository install config.
- Produces: a real Python 3.11 environment with persistent editable paths under this checkout.

- [ ] **Step 1: Run apt stage**

Run `bash scripts/install_env.sh only=[00_apt_tools] force=true env.CONDA_ENV_NAME=dreamervla-2`
with pipefail and a transcript. Expected: apt update/install exits zero, including
OpenGL/OSMesa, ffmpeg, build tools, Git LFS, Ninja, CMake, curl, and wget.

- [ ] **Step 2: Create Conda environment**

Run the same command with `only=[10_conda_env]`. Expected: `dreamervla-2` exists and
`python --version` inside it reports Python 3.11.x.

- [ ] **Step 3: Install Torch**

Run with `only=[20_torch]`. Expected: Torch 2.5.1, torchvision 0.20.1, and
torchaudio 2.5.1 install from the cu124 index.

- [ ] **Step 4: Install curated Python dependencies**

Run with `only=[30_python_deps]`. Expected: pinned requirements, editable
DreamerVLA, Ray extra, and dev group install successfully.

- [ ] **Step 5: Install mandatory third parties**

Run with `only=[40_third_party]`. Expected: pinned persistent checkouts, editable
embodiment dependencies, OpenVLA-OFT runtime additions, and the custom Transformers
fork install successfully.

- [ ] **Step 6: Install special packages**

Run with `only=[50_special_packages]`. Expected: pinned flash-attn wheel and
egl_probe build/install complete.

- [ ] **Step 7: Run installer verification**

Run with `only=[60_verify]`. Expected: critical versions, import locations, CUDA,
custom bidirectional Transformers fork, and PEFT compatibility pass.

- [ ] **Step 8: Diagnose and repair only actual setup failures**

For each failure, preserve the full traceback, reproduce the failing import/command
directly in `dreamervla-2`, compare it with the owning setup manifest, and form one
root-cause hypothesis. Before modifying setup behavior, add a focused test that
fails for the missing pin/import contract, run it to see the expected failure,
apply the smallest setup change, rerun the test, rerun the failed install stage,
then rerun `60_verify`.

### Task 4: Environment Integrity and Runtime Probes

**Files:**
- Create at runtime: `data/reproduction/environment/dreamervla-2/snapshots/conda-explicit.txt`
- Create at runtime: `data/reproduction/environment/dreamervla-2/snapshots/pip-freeze.txt`
- Create at runtime: `data/reproduction/environment/dreamervla-2/snapshots/runtime-probes.txt`

**Interfaces:**
- Consumes: completed environment from Task 3.
- Produces: exact package/runtime evidence before expensive assets or training.

- [ ] **Step 1: Capture package snapshots**

Activate `dreamervla-2`, run `conda list --explicit`, `python -m pip freeze --all`,
`python -m pip check`, and `uv pip check`; save complete output. Both dependency
checks must exit zero or enter Task 3 Step 8.

- [ ] **Step 2: Probe CUDA and compiled extensions**

Run Python probes that allocate tensors on all visible GPUs, report Torch CUDA and
cuDNN versions, import flash-attn, and execute a small FlashAttention-compatible
CUDA path. Record results without starting distributed training.

- [ ] **Step 3: Probe LIBERO rendering and Ray startup**

Run the repository's gated LIBERO/Ray environment smokes with OSMesa first and EGL
where configured. Record skips separately from passes; a skipped test is not a pass.

### Task 5: Public Asset Preparation and Validation

**Files:**
- Create at runtime: `data/reproduction/manifests/assets.json`
- Create at runtime: `data/reproduction/environment/dreamervla-2/logs/prepare-assets.log`

**Interfaces:**
- Consumes: pinned OpenVLA-OFT asset, LIBERO goal data, all eight GPUs, completed environment.
- Produces: validated reward/hidden preprocessing directories and complete asset manifest.

- [ ] **Step 1: Inspect exact public targets without mutation**

Resolve `configs/scripts/reproduce/prepare_assets.yaml`, inspect whether each target
is absent, complete, or incomplete, and record sizes/revisions. Never delete an
incomplete existing target.

- [ ] **Step 2: Run the public preparation command**

In `dreamervla-2`, with the current `DVLA_ROOT` and repository `data/` root, run
`bash scripts/reproduce/01_prepare_assets.sh`. This must execute real hardware,
third-party, asset, and preprocessing validation; do not pass `dry_run=true`.

- [ ] **Step 3: Verify the manifest and HDF5 contracts**

Require `status=complete`, the configured profile, exact asset and third-party
revisions, ten LIBERO files, matched reward/hidden demos and lengths, and hidden
shape metadata `[256,4096]` with history/chunk size 1.

### Task 6: Bounded Real WM, Classifier, and Dreamer Training

**Files:**
- Create at runtime: isolated run roots below `data/outputs/reproduction-validation/dreamervla-2/`
- Create at runtime: `data/reproduction/environment/dreamervla-2/logs/train-bounded.log`
- Create at runtime: isolated training state JSON under the validation evidence tree.

**Interfaces:**
- Consumes: complete asset manifest and prepared reward/hidden data.
- Produces: selected WM checkpoint, selected classifier checkpoint, and Dreamer `latest.ckpt`.

- [ ] **Step 1: Resolve and inspect the bounded commands**

Run the reproduction launcher once with `dry_run=true`, an isolated output/state
path, WM budget key `training.warmup_replay_max_steps`, classifier budget key
`training.max_train_steps`, and all budgets set to 1. Verify the printed commands
still select `dreamer-wm`, `classifier_official_upper_bound`, and `openvla_libero`.

- [ ] **Step 2: Execute bounded public reproduction**

Run the same `scripts/reproduce/02_train_dreamer.sh` overrides without dry-run.
Use all eight configured H100s and an isolated state/output root. Do not substitute
test fixtures or mock workers.

- [ ] **Step 3: Validate WM evidence**

Require at least one non-empty train loss metric, optimizer step evidence, a
resumable `latest.ckpt`, and a loss-named selected checkpoint loadable by the
classifier/Dreamer workflow.

- [ ] **Step 4: Validate classifier evidence**

Require at least one classifier train/validation metric, optimizer step evidence,
a resumable `latest.ckpt`, and an F1-named selected checkpoint loadable by Dreamer.

- [ ] **Step 5: Validate Dreamer evidence**

Require successful Ray group startup, WM/CLS checkpoint loads, real/imagined
rollout evidence, at least one actor update metric, and a non-empty
`checkpoints/latest.ckpt` that contains completed-step state.

### Task 7: Durable Documentation and Regression Coverage

**Files:**
- Modify: `docs/install.md`
- Modify: `docs/README.md`
- Create: `docs/native_environment_reproduction.md`
- Modify before doc behavior change: `tests/unit_tests/test_reproduction_workflow.py`

**Interfaces:**
- Consumes: exact logs, snapshots, manifests, checkpoints, and any setup fixes from Tasks 1–6.
- Produces: reproducible custom-environment instructions and an evidence-backed host record.

- [ ] **Step 1: Add a failing documentation contract test**

Extend the existing public-doc test to require
`env.CONDA_ENV_NAME=dreamervla-2`,
`CONDA_ENV_NAME=dreamervla-2 bash scripts/install/60_verify.sh`, and a docs-index
link to `native_environment_reproduction.md`. Run the focused test and confirm it
fails because the documentation is absent.

- [ ] **Step 2: Update native install instructions**

Document default and custom environment naming, activation, forced single-stage
debugging, and the distinction between Hydra config override and the direct shell
environment variable used by `60_verify.sh`.

- [ ] **Step 3: Write the evidence report**

Record date, commit, OS/kernel/driver/GPU/disk, install config, every command and
exit result, critical and transitive package snapshots, third-party pins, asset
manifest, bounded training artifacts/metrics, dependency gaps and fixes, full
quality-gate results, formal-budget commands, and limitations. Do not paste secrets,
tokens, W&B credentials, or enormous raw logs.

- [ ] **Step 4: Link the report from the docs index**

Add one concise entry to `docs/README.md` and rerun the focused documentation test.

### Task 8: Fresh Final Verification

**Files:**
- Update: `.planning/2026-07-17-dreamervla-2-environment/{task_plan,findings,progress}.md`

**Interfaces:**
- Consumes: final setup/docs changes and completed environment.
- Produces: fresh completion evidence and a clean handoff.

- [ ] **Step 1: Rerun install verification and dependency checks**

Run `CONDA_ENV_NAME=dreamervla-2 bash scripts/install/60_verify.sh`,
`python -m pip check`, and `uv pip check` from the final state.

- [ ] **Step 2: Run focused and full repository gates**

Run focused setup/reproduction/verify tests, `python -m pytest tests/unit_tests -q`,
and `ruff check dreamervla tests`. Record exact pass/fail/skip counts and exit codes.

- [ ] **Step 3: Re-check artifacts and git diff**

Verify the asset/training manifests and checkpoint hashes still resolve, inspect
`git status --short` and `git diff --check`, and ensure no `third_party/` or unrelated
files are staged/modified.

- [ ] **Step 4: Complete the task record**

Mark only evidence-backed phases complete, list any remaining external limitation,
and provide the user with the environment activation command, durable report link,
modified files, and fresh verification summary.
