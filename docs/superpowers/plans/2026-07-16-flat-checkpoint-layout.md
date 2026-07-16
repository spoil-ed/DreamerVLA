# Flat Checkpoint Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every training route write flat Torch checkpoints as `checkpoints/latest.ckpt` plus metric-named top-k files, make HF export an explicit `checkpoint_hf/` sibling, and give evaluation the dedicated `outputs/eval/<task-name>/` run root.

**Architecture:** Centralize filename construction, top-k bookkeeping, atomic materialization, and directory resolution in checkpoint utilities. `BaseRunner` owns common Torch/HF paths and resume behavior; specialized runners only build payloads and provide the Hydra-selected epoch and metric. Historical layouts remain read-only fallbacks, while every new write uses the flat contract.

**Tech Stack:** Python 3.11, PyTorch checkpoints, Hydra/OmegaConf, pytest, Ruff, shell syntax checks.

---

### Task 1: Shared checkpoint naming and discovery

**Files:**
- Modify: `dreamervla/utils/checkpoint_util.py`
- Modify: `dreamervla/utils/run_paths.py`
- Create: `tests/unit_tests/test_checkpoint_util.py`
- Modify: `tests/unit_tests/test_run_paths.py`

- [ ] **Step 1: Write failing naming tests**

Add parameterized tests proving `format_metric_checkpoint_name(epoch=3, metric_name=name, metric_value=0.25)` always returns one basename beginning with `epoch=0003-` and ending with `=0.250000.ckpt` for `loss`, `eval/accuracy`, `../loss`, and `a\\b`. Add rejection tests for negative epochs and `nan`/`inf` metric values.

- [ ] **Step 2: Run RED naming tests**

Run: `pytest -q tests/unit_tests/test_checkpoint_util.py`

Expected: import or assertion failure because the formatter does not exist.

- [ ] **Step 3: Implement naming and top-k invariants**

Implement `format_metric_checkpoint_name(*, epoch: int, metric_name: str, metric_value: float) -> str`. Require a non-negative completed epoch and finite metric, sanitize every non `[A-Za-z0-9_.-]` run to `_`, reject an empty sanitized name, pad epochs to at least four digits, and format values with six decimals. Refactor `TopKCheckpointManager` to accept `monitor_key`, optional filesystem `metric_name`, `mode`, and `k`; require `data["epoch"]`; construct names only through the formatter; and delete only registered metric files, never `latest.ckpt`.

- [ ] **Step 4: Write failing discovery tests**

Cover a concrete checkpoint file, a `checkpoints/` directory, a run root, a valid `checkpoint_hf/`, a canonical directory missing latest, and historical nested fallback. The run-root and `checkpoints/` inputs must both resolve exactly to `checkpoints/latest.ckpt`.

- [ ] **Step 5: Implement canonical-first discovery**

Teach `infer_run_root` that `checkpoints`, legacy `ckpt`, and `checkpoint_hf` are owned by their parent. Make `resolve_resume_checkpoint` prefer canonical latest, accept a valid HF directory, and only then scan existing legacy manual/warmup/progress patterns. Failure must name both the requested directory and expected `latest.ckpt`.

- [ ] **Step 6: Verify and commit**

Run: `pytest -q tests/unit_tests/test_checkpoint_util.py tests/unit_tests/test_run_paths.py`

Commit only these four files with `git commit -s -m "refactor: centralize flat checkpoint paths"`.

### Task 2: BaseRunner Torch/HF and resume contract

**Files:**
- Modify: `dreamervla/runners/base_runner.py`
- Modify: `dreamervla/runtime/libero_vla_evaluation_base.py`
- Modify: `tests/unit_tests/test_runner_artifacts.py`
- Modify: `tests/unit_tests/test_base_runner_shared_helpers.py`
- Modify: `tests/unit_tests/test_hf_module.py`
- Modify: `tests/unit_tests/test_checkpoint_format_version.py`

- [ ] **Step 1: Write failing common-runner tests**

Assert the default checkpoint format is Torch-only, `get_hf_checkpoint_path()` is `<run-root>/checkpoint_hf`, run-root resume loads canonical latest, and a runner declaring `checkpoint_output_enabled = False` does not create `checkpoints/` during setup.

- [ ] **Step 2: Run RED common-runner tests**

Run: `pytest -q tests/unit_tests/test_runner_artifacts.py tests/unit_tests/test_base_runner_shared_helpers.py tests/unit_tests/test_hf_module.py tests/unit_tests/test_checkpoint_format_version.py`

Expected: failures expose the current `both` default, `latest_hf` path, and unconditional checkpoint-directory creation.

- [ ] **Step 3: Implement BaseRunner behavior**

Default `_checkpoint_format()` to `torch`; add `checkpoint_output_enabled = True`; conditionally create `checkpoints/`; return `<run-root>/checkpoint_hf` for new HF writes while keeping old HF paths under `prefer_existing=True`; and route every explicit resume file/directory through the shared resolver before choosing Torch or HF loading.

- [ ] **Step 4: Make VLA HF export explicit and singular**

`LIBEROVLAEvaluationBase._save_checkpoint_sidecars` must require both `checkpoint_save_hf()` and `checkpoint.save_hf_encoder`. It must export only to `self.get_hf_checkpoint_path()` and skip metric-copy callbacks, so no `<stem>_hf`, `latest_hf`, or top-k HF duplicates are produced.

- [ ] **Step 5: Verify and commit**

Rerun the Step 2 command and commit the six files with `git commit -s -m "refactor: unify runner checkpoint contract"`.

### Task 3: Offline WM, DINO-WM, classifier, and VLA writers

**Files:**
- Modify: `dreamervla/runners/world_model_training_runner.py`
- Modify: `dreamervla/runners/dino_token_world_model_training_runner.py`
- Modify: `dreamervla/runners/success_classifier_training_runner.py`
- Modify: `dreamervla/runtime/world_model_training_base.py`
- Modify: `dreamervla/runtime/libero_vla_evaluation_base.py`
- Modify: `configs/experiment/wm_full_dataset_train.yaml`
- Modify: `configs/experiment/wm_dino_token_official.yaml`
- Modify: `configs/experiment/wmpo_token_classifier_openvla_onetraj_libero_goal_h1.yaml`
- Modify: `configs/evaluation/libero_vla.yaml`
- Modify: `tests/unit_tests/test_world_model_training_runner.py`
- Modify: `tests/unit_tests/test_dino_token_training_runner.py`
- Modify: `tests/unit_tests/test_success_classifier_training_runner.py`

- [ ] **Step 1: Write failing flat-writer tests**

Replace expectations for `wm_warmup.ckpt`, `classifier_warmup.ckpt`, `model.ckpt`, `global_step_*`, and `warmup_topk/` with canonical latest plus `epoch=0003-loss=0.250000.ckpt` or the configured classifier metric. Assert every child of `checkpoints/` is a file and that a selected metric file is byte-identical to the latest payload from that save event.

- [ ] **Step 2: Run RED route tests**

Run: `pytest -q tests/unit_tests/test_world_model_training_runner.py tests/unit_tests/test_dino_token_training_runner.py tests/unit_tests/test_success_classifier_training_runner.py`

Expected: failures reference retired aliases and directories.

- [ ] **Step 3: Flatten offline WM and classifier warmup**

Atomically write every progress/final payload to `checkpoints/latest.ckpt`; materialize top-k from the completed warmup epoch and Hydra monitor metric. Stop writing `warmup_progress/`, `warmup_topk/`, WM/classifier aliases, and route-specific HF directories. Keep old lookups only for legacy reads.

- [ ] **Step 4: Flatten DINO and generic WM/VLA saves**

Use completed epoch plus `eval/loss` (or configured monitor), serialize latest once, and pass the top-k destination through `extra_paths`. Remove DINO's nested `global_step_N/model.ckpt`. Apply the same one-serialization behavior to generic WM/VLA periodic saves.

- [ ] **Step 5: Flatten classifier selected checkpoints**

Replace `_save_named` partial payloads with full `BaseRunner.save_checkpoint()` payloads. Use completed `self.epoch` and the Hydra-selected F1/accuracy metric for the flat filename; final save writes only latest.

- [ ] **Step 6: Migrate Hydra selection keys**

Declare cadence, `checkpoint.topk.monitor_key`, filesystem `metric_name`, `mode`, and `k` in active experiment configs. Remove `format_str` and route-specific directory settings. No experiment produces HF unless it explicitly selects `hf` or `both`.

- [ ] **Step 7: Verify and commit**

Run the Step 2 command plus `pytest -q tests/unit_tests/test_runner_public_api.py`. Commit with `git commit -s -m "refactor: flatten offline training checkpoints"`.

### Task 4: Cotrain and Dreamer writer

**Files:**
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `dreamervla/config.py`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain_base.yaml`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain.yaml`
- Modify: `configs/dreamervla/wmcls_cotrain.yaml`
- Modify: `tests/unit_tests/test_cotrain_resume.py`
- Modify: `tests/unit_tests/test_cotrain_launcher.py`
- Modify: `tests/unit_tests/test_dreamer_runner.py`
- Modify: `tests/unit_tests/test_cotrain_config_validation.py`

- [ ] **Step 1: Write failing cotrain tests**

Assert cadence writes only `checkpoints/latest.ckpt`, Dreamer evaluation success selects `epoch=0010-accuracy=<value>.ckpt`, no checkpoint manifest or subdirectory exists, and run-root resume resolves latest. Retain a fixture proving a concrete historical `manual_cotrain.ckpt` still loads.

- [ ] **Step 2: Run RED cotrain tests**

Run: `pytest -q tests/unit_tests/test_cotrain_resume.py tests/unit_tests/test_cotrain_launcher.py tests/unit_tests/test_dreamer_runner.py tests/unit_tests/test_cotrain_config_validation.py`

Expected: failures expose `global_step_N/manual_cotrain.ckpt`, manifests, and last-directory retention.

- [ ] **Step 3: Consolidate the payload**

Keep the complete policy/optimizer/WM/classifier/RNG payload, add completed cotrain `epoch`, serialize atomically to latest once, and materialize a top-k file only when the configured monitor metric exists. Remove manifests, nested mkdir, directory pruning, and `keep_last_checkpoints`.

- [ ] **Step 4: Preserve legacy reads**

Make directory inputs use the shared resolver while retaining explicit historical manifest and `manual_cotrain.ckpt` interpretation. Init and resume must accept a concrete file, `checkpoints/`, or run root.

- [ ] **Step 5: Migrate cotrain config and verify**

Replace obsolete filename/latest/last-retention keys with top-k config. Dreamer monitors `eval/success_rate` and names it `accuracy`; missing evaluation metric means latest-only. Rerun Step 2 and commit with `git commit -s -m "refactor: flatten cotrain checkpoints"`.

### Task 5: Evaluation input and output roots

**Files:**
- Modify: `dreamervla/runners/libero_vla_evaluation_runner.py`
- Modify: `configs/evaluation/libero_vla.yaml`
- Modify: `configs/experiment/eval_libero_vla.yaml`
- Modify: `configs/experiment/eval_cotrain.yaml`
- Modify: `tests/unit_tests/test_libero_eval_protocol_compat.py`
- Modify: `tests/unit_tests/test_run_config.py`
- Modify: `tests/unit_tests/test_cotrain_launcher.py`
- Modify: `tests/unit_tests/test_runner_artifacts.py`

- [ ] **Step 1: Write failing eval tests**

Compose both eval experiments and assert `training.out_dir == run.output_root/eval/eval.task_suite_name` with no timestamp. Exercise `eval.ckpt_path` as a concrete file, `checkpoints/`, and run root; all must reach the same payload before checkpoint-kind detection.

- [ ] **Step 2: Run RED eval tests**

Run: `pytest -q tests/unit_tests/test_libero_eval_protocol_compat.py tests/unit_tests/test_run_config.py tests/unit_tests/test_cotrain_launcher.py tests/unit_tests/test_runner_artifacts.py`

Expected: directory input is not resolved and eval still inherits a timestamped experiment root.

- [ ] **Step 3: Implement eval layout and resolver**

Set `LIBEROVLAEvaluationRunner.checkpoint_output_enabled = False`; resolve non-null `eval.ckpt_path` through `resolve_resume_checkpoint()` before HF/type inspection; and set output to `${run.output_root}/eval/${eval.task_suite_name}`. The task directory itself is the run root and checkpoints remain inputs only.

- [ ] **Step 4: Verify and commit**

Rerun Step 2 and commit with `git commit -s -m "refactor: separate evaluation run roots"`.

### Task 6: Active docs and retired-name audit

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `configs/README.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `docs/data_layout.md`
- Modify: `scripts/README.md`
- Modify: `spec/04_complete_loop.md`
- Modify: `spec/99_manual_notes.md`

- [ ] **Step 1: Update active documentation**

Document training as `outputs/<experiment>/<timestamp>/`, eval as `outputs/eval/<task>/`, Torch as latest plus `epoch=...-metric=value.ckpt`, and optional HF as sibling `checkpoint_hf/`. Use run roots, latest, or explicit metric files in examples.

- [ ] **Step 2: Audit retired names**

Run `rg -n 'manual_cotrain\.ckpt|wm_warmup\.ckpt|classifier_warmup\.ckpt|model\.ckpt|latest_hf|warmup_topk|global_step_.*/' dreamervla configs scripts README.md README.zh-CN.md AGENTS.md docs spec`.

Expected: matches remain only in legacy read compatibility or historical `docs/superpowers` records, never active writers/current examples.

- [ ] **Step 3: Verify and commit**

Run `pytest -q tests/unit_tests/test_run_config.py tests/unit_tests/test_world_model_training_config.py tests/unit_tests/test_cotrain_config_validation.py` and commit with `git commit -s -m "docs: document flat checkpoint layout"`.

### Task 7: Full verification, separate Ruff commit, and push

**Files:**
- Verify: all Task 1-6 changes
- Preserve and separately commit: the pre-existing safe Ruff expansion already in the worktree

- [ ] **Step 1: Run targeted checkpoint regression**

Run the union of all test commands above. Expected: all checkpoint, resume, eval, and config tests pass.

- [ ] **Step 2: Run repository gates**

Run `ruff check .`, `ruff format --check .`, `bash -n scripts/experiments/cotrain/train.sh scripts/experiments/cotrain/eval.sh`, and `pytest -q tests/unit_tests`. Every command must exit 0.

- [ ] **Step 3: Review final invariants**

Run `git diff --check`, `git status --short`, and the final retired-name audit. Confirm no active writer creates a directory below `checkpoints/`, default saving creates no `checkpoint_hf/`, and unrelated changes are not staged.

- [ ] **Step 4: Commit the Ruff expansion separately**

After full verification, stage only the pre-existing Ruff rule expansion and commit it with `git commit -s -m "style: enable additional safe ruff rules"`.

- [ ] **Step 5: Push**

Run `git push origin main`. Expected: origin advances through the design, implementation, docs, and separate Ruff commit.
