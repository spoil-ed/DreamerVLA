# Hydra-Centered Experiment and Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Hydra the complete experiment source of truth, separate full cotrain from frozen Dreamer, and provide bounded resume outputs without replay state.

**Architecture:** Keep `dreamervla.train` as the native Runner entrypoint and `DreamerRunner` as a thin `CotrainRunner` specialization. Move budgets into Hydra profiles, validate Ray placement before setup, and restore checkpoint state before any logger is initialized.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, PyTorch, Ray, pytest, Ruff, Bash.

---

### Task 1: Explicit Experiment and Full Cotrain Topology

**Files:**
- Modify: `configs/train.yaml`
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain.yaml`
- Modify: `dreamervla/config.py`
- Test: `tests/unit_tests/test_runner_public_api.py`
- Test: `tests/unit_tests/test_openvla_traj1_libero_matrix.py`
- Test: `tests/unit_tests/test_cotrain_config_validation.py`

- [ ] **Step 1: Write failing tests**

```python
def test_train_requires_explicit_experiment():
    cfg = compose(config_name="train")
    assert OmegaConf.select(cfg, "_target_", default=None) is None

def test_full_cotrain_has_real_and_wm_workers():
    cfg = _compose(["experiment=openvla_onetraj_libero_cotrain"])
    assert cfg.manual_cotrain.training_mode == "staged_full_cotrain"
    assert cfg.manual_cotrain.learner_updates_enabled is True
    assert cfg.manual_cotrain.staged_policy_update is True
    plan = CotrainRunner(cfg)._placement_plan()
    assert plan.real_env_ranks and plan.wm_env_ranks

def test_validate_rejects_missing_wm_worker():
    cfg = _compose(["experiment=openvla_onetraj_libero_cotrain"])
    cfg.manual_cotrain.ngpu = 1
    cfg.manual_cotrain.real_env_workers = 1
    cfg.cluster.component_placement = {"env": 0, "rollout": 0, "actor": 0}
    with pytest.raises(ValueError, match="WMEnvWorker"):
        validate_cfg(cfg)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/unit_tests/test_runner_public_api.py tests/unit_tests/test_openvla_traj1_libero_matrix.py tests/unit_tests/test_cotrain_config_validation.py`

Expected: implicit selection, frozen flags, and missing placement validation fail.

- [ ] **Step 3: Implement minimum behavior**

Set `experiment: null`. Configure `openvla_onetraj_libero_cotrain` for eight GPUs with `component_placement: null`, `training_mode: staged_full_cotrain`, `learner_updates_enabled: true`, and `staged_policy_update: true`. Add a validator using `build_manual_cotrain_placement` and reject empty real/WM rank lists.

- [ ] **Step 4: Verify GREEN and commit**

Run the Step 2 command. Then stage the six listed files and commit with `git commit -s -m "fix: validate Hydra cotrain topology"`.

### Task 2: Hydra Profiles Replace Runtime Budget Mutation

**Files:**
- Create: `configs/profile/production.yaml`
- Create: `configs/profile/debug.yaml`
- Create: `configs/profile/smoke.yaml`
- Modify: `configs/train.yaml`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `dreamervla/runners/dreamer_runner.py`
- Modify: `dreamervla/runners/world_model_training_runner.py`
- Modify: `dreamervla/runtime/world_model_training_base.py`
- Modify: `dreamervla/runtime/libero_vla_evaluation_base.py`
- Test: `tests/unit_tests/test_cotrain_debug_config.py`
- Test: `tests/unit_tests/test_world_model_training_runner.py`

- [ ] **Step 1: Write failing profile tests**

```python
def test_debug_profile_owns_budget():
    cfg = _compose_experiment("openvla_libero", ["profile=debug"])
    assert cfg.manual_cotrain.global_steps == 10
    assert cfg.manual_cotrain.checkpoint_every == 1
    assert cfg.manual_cotrain.real_rollout_target_trajectories == 8
    assert cfg.manual_cotrain.wm_rollout_target_trajectories == 256

def test_dreamer_constructor_does_not_mutate_config():
    cfg = _compose_experiment("openvla_libero", ["profile=debug"])
    before = OmegaConf.to_container(cfg, resolve=True)
    DreamerRunner(cfg)
    assert OmegaConf.to_container(cfg, resolve=True) == before
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/unit_tests/test_cotrain_debug_config.py tests/unit_tests/test_world_model_training_runner.py`

Expected: the profile group is absent and constructors/runtime code still change budgets.

- [ ] **Step 3: Implement profiles**

Compose `profile: production` before `experiment`. Put the current reduced values in `profile/debug.yaml` and smallest valid geometry in `profile/smoke.yaml`. Delete `training.debug` assignments that choose steps, epochs, batches, cadence, or workers. Keep per-rank batch values local.

- [ ] **Step 4: Verify GREEN and commit**

Run the Step 2 command. Stage the eleven listed files and commit with `git commit -s -m "refactor: move run budgets into Hydra profiles"`.

### Task 3: Remove Parallel Launcher Semantics

**Files:**
- Modify: `dreamervla/launchers/train.py`
- Modify: `dreamervla/launchers/contracts.py`
- Test: `tests/unit_tests/test_unified_train_launcher.py`
- Test: `tests/unit_tests/test_experiment_launchers.py`

- [ ] **Step 1: Write failing launcher tests**

```python
def test_launcher_does_not_translate_batch_alias():
    with pytest.raises(SystemExit, match="Hydra key=value"):
        build_launch(["--config", "dreamer-wm", "batch_size=8"])

def test_contract_ignores_step_environment(monkeypatch):
    monkeypatch.setenv("WMCLS_COTRAIN_GLOBAL_STEPS", "3")
    values = CotrainLaunchContract().normalize_argv(["manual_cotrain.global_steps=11"])
    assert "manual_cotrain.global_steps=3" not in values
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/unit_tests/test_unified_train_launcher.py tests/unit_tests/test_experiment_launchers.py`

Expected: aliases and semantic environment variables remain active.

- [ ] **Step 3: Implement minimum launcher surface**

Remove aliases for batch, workers, steps, epochs, and output. Remove `WORLD_MODEL_CKPT`, `CLASSIFIER_CKPT`, `WMCLS_COTRAIN_GLOBAL_STEPS`, and `COTRAIN_DRY_RUN`. Retain documented `--wm_ckpt`, `--cls_ckpt`, and `--resume` normalization.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2 and `bash -n scripts/experiments/cotrain/train.sh`. Stage the four files and commit with `git commit -s -m "refactor: keep experiment semantics in Hydra"`.

### Task 4: Classifier Restore-First Logging and Canonical Checkpoint

**Files:**
- Modify: `dreamervla/runners/success_classifier_training_runner.py`
- Test: `tests/unit_tests/test_success_classifier_training_runner.py`
- Test: `tests/unit_tests/test_metric_logger.py`
- Test: `tests/unit_tests/test_run_paths.py`

- [ ] **Step 1: Write failing resume tests**

```python
def test_classifier_resume_precedes_first_log(runner, monkeypatch):
    events = []
    monkeypatch.setattr(runner, "resume", lambda cfg: (setattr(runner, "global_step", 17), events.append("resume")))
    monkeypatch.setattr(runner, "_log", lambda payload: events.append(f"log:{runner.global_step}"))
    runner._finish_setup_after_optimizer()
    assert events[:2] == ["resume", "log:17"]

def test_classifier_resume_keeps_jsonl(tmp_path, runner):
    path = tmp_path / "logs" / "train_log.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"event":"before"}\n', encoding="utf-8")
    runner._prepare_train_log(resume=True)
    assert path.read_text(encoding="utf-8") == '{"event":"before"}\n'
```

- [ ] **Step 2: Write failing canonical checkpoint test**

```python
def test_classifier_final_save_writes_warmup_and_latest(tmp_path, runner):
    runner._save_final_checkpoint()
    warmup = tmp_path / "checkpoints" / "classifier_warmup.ckpt"
    latest = tmp_path / "checkpoints" / "latest.ckpt"
    assert warmup.is_file() and latest.samefile(warmup)
```

- [ ] **Step 3: Verify RED**

Run: `python -m pytest -q tests/unit_tests/test_success_classifier_training_runner.py tests/unit_tests/test_metric_logger.py tests/unit_tests/test_run_paths.py`

Expected: logging precedes restore, JSONL is truncated, and the canonical artifact is absent.

- [ ] **Step 4: Implement and verify GREEN**

Introduce `_finish_setup_after_optimizer`, `_prepare_train_log`, and `_save_final_checkpoint` with the signatures exercised above. Restore immediately after optimizer construction and before `_log`. Preserve JSONL on resume. Save one BaseRunner-format `classifier_warmup.ckpt`, link/copy it to `latest.ckpt`, and gate named snapshots behind `training.topk_k > 0`. Run Step 3 and commit the four files with `git commit -s -m "fix: resume classifier logs from checkpoint step"`.

### Task 5: Replay-Free Cotrain Checkpoints and Retention

**Files:**
- Modify: `configs/dreamervla/openvla_onetraj_libero_cotrain.yaml`
- Modify: `configs/dreamervla/wmcls_cotrain.yaml`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Test: `tests/unit_tests/test_cotrain_resume.py`
- Test: `tests/unit_tests/test_cotrain_phase_progress.py`

- [ ] **Step 1: Write failing output tests**

```python
def _save_steps(tmp_path, steps):
    runner = _make_manual_checkpoint_runner(tmp_path, keep_last=2)
    groups = {"ActorGroup": _ActorGroup(), "LearnerGroup": _LearnerGroup()}
    return [runner._maybe_save_manual_checkpoint(groups, step, {}) for step in steps]

def test_checkpoint_has_no_replay(tmp_path):
    path = _save_steps(tmp_path, [1])[0]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "replay" not in payload
    assert "replay_sampling_state" not in payload

def test_retention_keeps_two(tmp_path):
    _save_steps(tmp_path, [1, 2, 3])
    checkpoint_dir = tmp_path / "checkpoints"
    assert sorted(p.name for p in checkpoint_dir.glob("global_step_*")) == ["global_step_2", "global_step_3"]

def test_progress_reuses_current_dir(tmp_path):
    runner = _make_manual_checkpoint_runner(tmp_path, keep_last=2)
    assert runner._prepare_manual_cotrain_progress_dir(1) == runner._prepare_manual_cotrain_progress_dir(2)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/unit_tests/test_cotrain_resume.py tests/unit_tests/test_cotrain_phase_progress.py`

Expected: replay fields remain, old checkpoints accumulate, and progress paths differ.

- [ ] **Step 3: Implement bounded replay-free output**

Add `_make_manual_checkpoint_runner` beside the existing `_ActorGroup` and `_LearnerGroup` fixtures; it constructs an uninitialized `CotrainRunner`, assigns `cfg`, `config`, `_output_dir`, policy hashes, applied steps, and `manual_cotrain.keep_last_checkpoints`. Remove `save_replay_state` and all replay/sampling serialization/restoration; ignore legacy fields. Add `manual_cotrain.keep_last_checkpoints: 2`, prune older numeric step directories after refreshing `latest.ckpt`, and reuse `diagnostics/manual_cotrain_progress/current/`.

- [ ] **Step 4: Verify GREEN and commit**

Run Step 2. Stage the five files and commit with `git commit -s -m "fix: bound replay-free cotrain checkpoints"`.

### Task 6: Remove Stale Summarizer and Separate Probes

**Files:**
- Delete: `dreamervla/diagnostics/experiment_stage_checks.py`
- Delete: `scripts/experiments/classifier_training/eval.sh`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`
- Modify: `tests/unit_tests/test_repository_hygiene.py`
- Modify: `scripts/README.md`

- [ ] **Step 1: Write failing surface test**

```python
def test_stale_classifier_summarizer_is_removed():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "dreamervla/diagnostics/experiment_stage_checks.py").exists()
    assert not (root / "scripts/experiments/classifier_training/eval.sh").exists()
```

Require each retained `scripts/experiments/**/*.sh` to contain `dreamervla.launchers.train`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/unit_tests/test_experiment_stage_scripts.py tests/unit_tests/test_repository_hygiene.py`

Expected: stale files exist and probes remain under experiments.

- [ ] **Step 3: Implement, verify, and commit**

Delete the stale files. Move standalone probe shells to `scripts/diagnostics/` without changing Python targets. Run Step 2 and `find scripts/experiments scripts/diagnostics -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n`. Stage affected files and commit with `git commit -s -m "refactor: keep probes outside experiment scripts"`.

### Task 7: Documentation and Full Verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `spec/04_complete_loop.md`
- Modify: `docs/data_layout.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `configs/README.md`
- Modify: `scripts/README.md`

- [ ] **Step 1: Update contracts**

Document explicit experiment selection, full Cotrain versus frozen Dreamer, profiles, replay-free checkpoints, canonical classifier warmup, keep-last retention, and `bash scripts/utils/wandb_sync.sh /path/to/run_root/wandb`.

- [ ] **Step 2: Run focused verification**

Run: `python -m pytest -q tests/unit_tests/test_runner_public_api.py tests/unit_tests/test_openvla_traj1_libero_matrix.py tests/unit_tests/test_cotrain_config_validation.py tests/unit_tests/test_cotrain_resume.py tests/unit_tests/test_cotrain_phase_progress.py tests/unit_tests/test_success_classifier_training_runner.py tests/unit_tests/test_metric_logger.py tests/unit_tests/test_wandb_sync_launcher.py tests/unit_tests/test_experiment_stage_scripts.py tests/unit_tests/test_repository_hygiene.py`

Expected: zero failures.

- [ ] **Step 3: Run complete verification**

```bash
python -m pytest -q tests/unit_tests
python -m ruff check dreamervla tests
python -m ruff format --check dreamervla tests
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check
```

Expected: all commands exit zero; only existing intentional skips/warnings remain.

- [ ] **Step 4: Inspect and commit documentation**

Run `git status --short` and `git diff --stat`, then stage the six documentation files and commit with `git commit -s -m "docs: document Hydra experiment contracts"`.
