# Training Output, Resume, and W&B Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give WM warmup, success-classifier training, and cotrain one shallow artifact layout, strict epoch/global-unit resume semantics, resumable TensorBoard/W&B history, and a one-argument offline W&B sync command.

**Architecture:** Hydra creates and owns the run root plus `.hydra/`; `BaseRunner` owns non-Hydra runtime metadata and canonical checkpoint paths. Each route writes one authoritative resumable checkpoint family and restores all model, optimizer, progress, threshold/best-metric, and per-process RNG state before creating metric loggers. TensorBoard and W&B share a route-correct `metric_resume_step`; W&B stores a stable logical run ID under the flat `wandb/` directory. Legacy paths are read-only compatibility inputs.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, PyTorch/DDP/FSDP, Ray worker groups, TensorBoard, W&B SDK/CLI, Bash, pytest, Ruff.

---

## Scope guardrails

- Do not add, remove, migrate, validate, or test replay-buffer payloads, replay sampling state, replay schema, or replay restore behavior.
- Save WM and standalone-classifier checkpoints only after a completed epoch. Cotrain's existing global unit is its checkpoint boundary.
- Do not save a dataloader cursor or prefetched batches.
- Keep top-k checkpoints and Hugging Face exports opt-in.
- Read legacy `ckpt/`, `warmup_progress/`, nested `wandb/wandb/`, and root `resolved_config.yaml` where compatibility is required; never produce those forms in a new run.
- Preserve unrelated working-tree changes, including the existing untracked planning files.

## Task 1: Restore native Hydra runtime and the shallow run root

**Files:**

- Modify: `dreamervla/train.py`
- Modify: `dreamervla/runners/base_runner.py`
- Modify: `tests/unit_tests/test_runner_public_api.py`
- Modify: `tests/unit_tests/test_runner_artifacts.py`
- Create: `tests/unit_tests/test_native_hydra_artifacts.py`

- [ ] **Step 1: Replace parser expectations with a failing native-Hydra test**

  Delete tests for `_parse_hydra_like_args`. Add assertions that the public entry is Hydra-decorated and still delegates the composed config to `run`:

  ```python
  def test_main_is_native_hydra_entrypoint() -> None:
      assert hasattr(train.main, "__wrapped__")
      assert train.main.__wrapped__.__name__ == "main"
  ```

- [ ] **Step 2: Add a failing subprocess test for Hydra-owned metadata**

  In `test_native_hydra_artifacts.py`, run a tiny helper in a subprocess which imports `dreamervla.train`, replaces `run` with a no-op, sets `sys.argv` overrides for a temporary output directory, and invokes `main()`. Assert:

  ```python
  assert (out_dir / ".hydra" / "config.yaml").is_file()
  assert (out_dir / ".hydra" / "overrides.yaml").is_file()
  assert (out_dir / ".hydra" / "hydra.yaml").is_file()
  assert not (out_dir / "resolved_config.yaml").exists()
  ```

  Run:

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_runner_public_api.py \
    tests/unit_tests/test_native_hydra_artifacts.py -q
  ```

  Expected: FAIL because `train.py` manually composes Hydra config and does not enter Hydra's runtime.

- [ ] **Step 3: Restore the native Hydra entrypoint**

  Replace `_parse_hydra_like_args` and manual `initialize_config_dir`/`compose` use with:

  ```python
  @hydra.main(
      version_base=None,
      config_path="../configs",
      config_name="train",
  )
  def main(cfg: DictConfig) -> None:
      run(cfg)
  ```

  Keep `register_dreamervla_resolvers()` at import time before the decorator can compose the config, and retain `run(cfg)` as the independently testable lifecycle function.

- [ ] **Step 4: Make `BaseRunner` create only route-independent canonical artifacts**

  Update `test_runner_artifacts.py` first so it expects only:

  ```python
  assert (out_dir / "checkpoints").is_dir()
  assert (out_dir / "run_manifest.json").is_file()
  assert not (out_dir / "resolved_config.yaml").exists()
  assert not (out_dir / "logs").exists()
  assert not (out_dir / "video").exists()
  assert not (out_dir / "diagnostics").exists()
  ```

  `MetricLogger` will create `tensorboard/` and `wandb/` lazily only when enabled. Route code creates video/diagnostic/log directories only when it writes those artifacts.

- [ ] **Step 5: Trim `run_manifest.json` to runtime-only facts**

  Remove `run_dir`, `artifact_dirs`, `state`, `logging.log_path`, and `config.resolved_config_path`. Retain the schema version, creation time, runner identity, distributed topology, logger backends, Git metadata, and later `model` summary:

  ```python
  return {
      "schema_version": 2,
      "created_at_utc": datetime.now(UTC).isoformat(),
      "runner": {
          "class": type(self).__name__,
          "name": str(self.runner_name),
          "family": str(self.runner_family),
          "status": str(self.runner_status),
      },
      "distributed": {
          "strategy": str(
              OmegaConf.select(
                  self.cfg,
                  "training.distributed_strategy",
                  default="ddp",
              )
          ),
          "rank": int(getattr(distributed, "rank", 0) or 0),
          "local_rank": int(getattr(distributed, "local_rank", 0) or 0),
          "world_size": int(getattr(distributed, "world_size", 1) or 1),
      },
      "logging": {"backends": backends},
      "git": self._git_metadata(),
  }
  ```

  Remove `get_resolved_config_path()` and update `print_config()`'s comment to name `.hydra/config.yaml`.

- [ ] **Step 6: Run focused tests**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_runner_public_api.py \
    tests/unit_tests/test_native_hydra_artifacts.py \
    tests/unit_tests/test_runner_artifacts.py \
    tests/unit_tests/test_base_runner_config_gate.py -q
  ```

  Expected: PASS.

- [ ] **Step 7: Commit the native-Hydra boundary**

  ```bash
  git add dreamervla/train.py dreamervla/runners/base_runner.py \
    tests/unit_tests/test_runner_public_api.py \
    tests/unit_tests/test_native_hydra_artifacts.py \
    tests/unit_tests/test_runner_artifacts.py
  git commit -s -m "refactor: make hydra own run metadata"
  ```

## Task 2: Centralize canonical run-config discovery

**Files:**

- Create: `dreamervla/utils/run_config.py`
- Modify: `dreamervla/diagnostics/eval_chunkwm_closeloop.py`
- Modify: `dreamervla/diagnostics/eval_dino_token_wm.py`
- Modify: `dreamervla/diagnostics/wm_single_episode_overfit.py`
- Modify: `dreamervla/diagnostics/experiment_stage_checks.py`
- Modify: `tests/unit_tests/test_eval_chunkwm_closeloop.py`
- Modify: `tests/unit_tests/test_wm_single_episode_overfit_diagnostic.py`
- Create: `tests/unit_tests/test_run_config.py`
- Modify: `tests/e2e_tests/test_world_model_env_ray_smoke.py`
- Modify: `tests/e2e_tests/test_cotrain_smoke.py`

- [ ] **Step 1: Write failing discovery and precedence tests**

  Cover a checkpoint path nested below `checkpoints/global_step_7/`, canonical `.hydra/config.yaml`, legacy root `resolved_config.yaml`, and precedence when both exist:

  ```python
  def test_find_run_config_prefers_hydra_snapshot(tmp_path: Path) -> None:
      canonical = tmp_path / ".hydra" / "config.yaml"
      legacy = tmp_path / "resolved_config.yaml"
      canonical.parent.mkdir()
      canonical.write_text("value: canonical\n")
      legacy.write_text("value: legacy\n")
      checkpoint = tmp_path / "checkpoints/global_step_7/manual_cotrain.ckpt"
      assert find_run_config(checkpoint) == canonical
  ```

  Add a resolver test with a DreamerVLA interpolation so loader behavior, not just path lookup, is covered.

- [ ] **Step 2: Run the new test and confirm failure**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_run_config.py -q
  ```

  Expected: FAIL because the shared loader does not exist.

- [ ] **Step 3: Implement the shared loader**

  Add these public functions:

  ```python
  def find_run_config(path: str | Path) -> Path:
      start = Path(path).expanduser().resolve()
      parents = (start, *start.parents) if start.is_dir() else start.parents
      for parent in parents:
          canonical = parent / ".hydra" / "config.yaml"
          if canonical.is_file():
              return canonical
      for parent in parents:
          legacy = parent / "resolved_config.yaml"
          if legacy.is_file():
              return legacy
      raise FileNotFoundError(f"no run config found above {start}")


  def load_run_config(path: str | Path) -> DictConfig:
      setup_globals()
      register_dreamervla_resolvers()
      cfg = OmegaConf.load(find_run_config(path))
      OmegaConf.resolve(cfg)
      return cfg
  ```

  Import `setup_globals` from `hydra.core.utils`; this restores Hydra's standard `now`, `hydra`, and `python_version` resolvers before resolving the native application snapshot. Do not copy or rewrite Hydra's file.

- [ ] **Step 4: Migrate every run-config consumer**

  Replace local parent walks and hard-coded `resolved_config.yaml` assumptions with `find_run_config`/`load_run_config`. CLI help should say “run config” rather than “resolved_config.yaml”. Keep explicit `--config-path` accepting either file form for backward compatibility.

  Update e2e assertions to read `.hydra/config.yaml` and assert the root copy is absent.

- [ ] **Step 5: Run diagnostics-focused tests**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_run_config.py \
    tests/unit_tests/test_eval_chunkwm_closeloop.py \
    tests/unit_tests/test_wm_single_episode_overfit_diagnostic.py -q
  ```

  Expected: PASS for canonical and legacy inputs.

- [ ] **Step 6: Commit config discovery**

  ```bash
  git add dreamervla/utils/run_config.py \
    dreamervla/diagnostics/eval_chunkwm_closeloop.py \
    dreamervla/diagnostics/eval_dino_token_wm.py \
    dreamervla/diagnostics/wm_single_episode_overfit.py \
    dreamervla/diagnostics/experiment_stage_checks.py \
    tests/unit_tests/test_run_config.py \
    tests/unit_tests/test_eval_chunkwm_closeloop.py \
    tests/unit_tests/test_wm_single_episode_overfit_diagnostic.py \
    tests/e2e_tests/test_world_model_env_ray_smoke.py \
    tests/e2e_tests/test_cotrain_smoke.py
  git commit -s -m "refactor: load run config from hydra metadata"
  ```

## Task 3: Make RNG a strict new-checkpoint contract

**Files:**

- Modify: `dreamervla/constants.py`
- Modify: `dreamervla/utils/seed.py`
- Modify: `dreamervla/runtime/distributed.py`
- Modify: `dreamervla/runners/base_runner.py`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py`
- Modify: `dreamervla/workers/actor/learner_worker.py`
- Modify: `tests/unit_tests/test_rng_checkpoint.py`
- Modify: `tests/unit_tests/test_cotrain_resume.py`
- Create: `tests/unit_tests/test_cotrain_worker_rng.py`
- Modify: `tests/unit_tests/test_checkpoint_format_version.py`
- Modify: `tests/unit_tests/test_checkpoint_version_guard.py`

- [ ] **Step 1: Add failing NumPy and rank-selection RNG tests**

  Extend the CPU round-trip test to draw from Python, NumPy, and Torch before and after restore:

  ```python
  set_seed(17)
  state = capture_rng_state()
  expected = (random.random(), np.random.random(), torch.rand(3))
  restore_rng_state(state, strict=True)
  actual = (random.random(), np.random.random(), torch.rand(3))
  assert actual[0] == expected[0]
  assert actual[1] == expected[1]
  torch.testing.assert_close(actual[2], expected[2])
  ```

  Keep the CUDA round trip gated by `torch.cuda.is_available()`.

- [ ] **Step 2: Run RNG tests and confirm failure**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_rng_checkpoint.py -q
  ```

  Expected: FAIL because NumPy is outside the current contract.

- [ ] **Step 3: Add NumPy and rank-aware helpers**

  Seed NumPy in `set_seed`, add `"numpy": np.random.get_state()` to snapshots, restore it, and require it in strict mode. Add:

  ```python
  def select_rank_rng_state(states: object, rank: int) -> Mapping[str, Any] | None:
      if isinstance(states, list) and 0 <= rank < len(states):
          item = states[rank]
          return item if isinstance(item, Mapping) else None
      return states if isinstance(states, Mapping) else None
  ```

  Legacy missing RNG must emit one `RuntimeWarning`; malformed RNG in a new-format checkpoint is an error.

- [ ] **Step 4: Version the stricter checkpoint schema**

  Increment `CHECKPOINT_FORMAT_VERSION` from `1` to `2`. Treat version `2` as requiring `rng_by_rank`; version `1` or a missing version is a legacy checkpoint that warns once when RNG is unavailable. Keep the future-version rejection test intact and add a version-1 compatibility test.

- [ ] **Step 5: Add distributed object gathering**

  Add to `NopretokenizeSFTDistributedHelper`:

  ```python
  def all_gather_objects(self, value: Any) -> list[Any]:
      if not self.is_distributed:
          return [value]
      gathered: list[Any] = [None] * self.world_size
      dist.all_gather_object(gathered, value)
      return gathered
  ```

  All ranks must enter this collective before the existing main-rank-only filesystem write.

- [ ] **Step 6: Store and restore per-rank RNG in `BaseRunner` checkpoints**

  Before the non-main early return in `save_checkpoint`, capture local RNG and gather it. Write `payload["rng_by_rank"]`. In `load_payload`, restore the current rank after model/optimizer/pickled progress is restored and before returning. New payloads require RNG; legacy payloads warn once.

  Add a checkpoint round-trip test that perturbs RNG after saving and proves the next draw is restored.

- [ ] **Step 7: Expose worker RNG RPCs for cotrain**

  Add the same method to both trainable worker classes:

  ```python
  def rng_state_dict(self) -> dict[str, Any]:
      return capture_rng_state()

  def load_rng_state_dict(self, states: object) -> None:
      state = select_rank_rng_state(states, self.rank)
      if state is None:
          raise RuntimeError(f"checkpoint has no RNG state for worker rank {self.rank}")
      restore_rng_state(state, strict=True)
  ```

  At cotrain save time, add `format_version=CHECKPOINT_FORMAT_VERSION`, collect `actor_rng_by_rank` and `learner_rng_by_rank` from worker groups, and capture the controller's `rng`. Do not query or change `ReplayGroup` for this task.

- [ ] **Step 8: Restore cotrain RNG after worker initialization**

  After actor and learner models/optimizers are loaded, dispatch each saved state to the same worker rank. Restore controller RNG before the loop and before metric logger creation. Legacy manual checkpoints without these keys warn once and continue from the configured seed.

  Add fake-group tests proving each rank receives its own state, and that replay fake call counts are unchanged.

- [ ] **Step 9: Run RNG/resume tests**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_rng_checkpoint.py \
    tests/unit_tests/test_checkpoint_format_version.py \
    tests/unit_tests/test_checkpoint_version_guard.py \
    tests/unit_tests/test_cotrain_resume.py \
    tests/unit_tests/test_cotrain_worker_rng.py -q
  ```

  Expected: PASS.

- [ ] **Step 10: Commit RNG resume**

  ```bash
  git add dreamervla/constants.py dreamervla/utils/seed.py \
    dreamervla/runtime/distributed.py \
    dreamervla/runners/base_runner.py dreamervla/runners/cotrain_runner.py \
    dreamervla/workers/actor/embodied_fsdp_actor.py \
    dreamervla/workers/actor/learner_worker.py \
    tests/unit_tests/test_rng_checkpoint.py \
    tests/unit_tests/test_checkpoint_format_version.py \
    tests/unit_tests/test_checkpoint_version_guard.py \
    tests/unit_tests/test_cotrain_resume.py \
    tests/unit_tests/test_cotrain_worker_rng.py
  git commit -s -m "feat: resume per-process training rng"
  ```

## Task 4: Replace warmup progress files with epoch-boundary route checkpoints

**Files:**

- Modify: `dreamervla/runners/world_model_training_runner.py`
- Modify: `dreamervla/runners/success_classifier_training_runner.py`
- Modify: `dreamervla/utils/run_paths.py`
- Modify: `dreamervla/config.py`
- Modify: `dreamervla/diagnostics/experiment_stage_checks.py`
- Modify: `configs/experiment/wm_full_dataset_train.yaml`
- Modify: `configs/experiment/wm_official_upper_bound_profile.yaml`
- Modify: `configs/experiment/wmpo_token_classifier_openvla_onetraj_libero_goal_h1.yaml`
- Modify: `tests/unit_tests/test_world_model_training_runner.py`
- Modify: `tests/unit_tests/test_success_classifier_training_runner.py`
- Modify: `tests/unit_tests/test_world_model_training_config.py`
- Modify: `tests/unit_tests/test_run_paths.py`

- [ ] **Step 1: Write failing WM canonical-overwrite tests**

  Replace the current `warmup_progress` accumulation assertions with:

  ```python
  first = runner._save_wm_warmup_checkpoint(step=7, epoch=1, complete=False, metrics={})
  second = runner._save_wm_warmup_checkpoint(step=14, epoch=2, complete=False, metrics={})
  assert first == second == tmp_path / "checkpoints/wm_warmup.ckpt"
  assert not (tmp_path / "checkpoints/warmup_progress").exists()
  payload = torch.load(second, map_location="cpu", weights_only=False)
  assert payload["warmup_epoch"] == 2
  assert payload["warmup_step"] == 14
  assert payload["complete"] is False
  ```

  Add an analogous classifier test for `classifier_warmup.ckpt`. Assert each payload includes model, optimizer, progress, best/threshold fields where applicable, and RNG.

- [ ] **Step 2: Add failing epoch-resume behavior tests**

  Prove that:

  - an incomplete canonical checkpoint resumes at the next epoch;
  - a complete canonical checkpoint skips that component;
  - legacy `warmup_progress/*_step_*.ckpt` remains readable;
  - no save occurs inside an epoch callback;
  - a missing optimizer in strict resume raises rather than becoming a warm start.

  Run:

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_success_classifier_training_runner.py \
    tests/unit_tests/test_world_model_training_config.py \
    tests/unit_tests/test_run_paths.py -q
  ```

  Expected: FAIL against the current step-based progress writer.

- [ ] **Step 3: Rename checkpoint cadence to epochs**

  Replace `training.warmup_checkpoint_every` with `training.warmup_checkpoint_every_epochs`, defaulting active recipes to `1`. Replace standalone classifier `training.ckpt_every` with `training.checkpoint_every_epochs`, also defaulting to `1`. Update `validate_cfg` to reject negative values and reject `0` when resume-capable training requests more than one epoch.

  Do not reinterpret the old field silently; config validation should identify the removed field and tell users the replacement name. Update `experiment_stage_checks.py` to emit the new override names.

- [ ] **Step 4: Write canonical WM/classifier checkpoint helpers**

  Both helpers atomically overwrite their fixed path and carry an explicit component/version marker:

  ```python
  payload = {
      "format_version": CHECKPOINT_FORMAT_VERSION,
      "component": component,
      "complete": bool(complete),
      "warmup_epoch": int(epoch),
      "warmup_step": int(step),
      "state_dicts": {
          component: _cpu_state_dict(model),
          f"{component}_optimizer": _cpu_tree(optimizer.state_dict()),
      },
      "rng_by_rank": self._gather_checkpoint_rng(),
      "metrics": dict(metrics),
  }
  _atomic_torch_save(payload, path)
  ```

  For classifier payloads, include calibrated threshold and best validation metric/path when they exist. Save `complete=False` after each configured completed epoch and overwrite the same file with `complete=True` at route completion.

- [ ] **Step 5: Resume at the next complete epoch**

  Convert `warmup_epoch` to the starting epoch and derive the corresponding component step from the route's deterministic `steps_per_epoch`. Restore model, optimizer, scalar state, and rank RNG before constructing/logging the next epoch. Do not add a batch cursor.

  Keep `_latest_warmup_progress_path` only as a legacy-read fallback; delete all calls that write new files there.

  Refactor each warmup call into deterministic epoch chunks: calculate `steps_per_epoch`, call the existing trainer for exactly the remaining steps in one epoch, return to the route, then checkpoint. Early stopping must mark the component complete and must not fabricate an unfinished epoch.

- [ ] **Step 6: Move standalone classifier saves out of the batch loop**

  Delete the `global_step % ckpt_every` write inside the training loop. After validation and `self.finish_epoch()`, save `latest.ckpt` only when the completed epoch matches `checkpoint_every_epochs`; preserve explicit top-k copies as opt-in destinations to the same atomic serialization.

- [ ] **Step 7: Expand resume-path compatibility**

  `resolve_resume_checkpoint` should search in this order:

  1. `checkpoints/latest.ckpt`
  2. newest `checkpoints/global_step_*/manual_cotrain.ckpt`
  3. `checkpoints/wm_warmup.ckpt`
  4. `checkpoints/classifier_warmup.ckpt`
  5. the same canonical filenames under legacy `ckpt/`
  6. historical `warmup_progress/*.ckpt`

  New writes stay under `checkpoints/`.

- [ ] **Step 8: Run focused checkpoint tests**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_success_classifier_training_runner.py \
    tests/unit_tests/test_world_model_training_config.py \
    tests/unit_tests/test_run_paths.py \
    tests/unit_tests/test_cotrain_resume.py -q
  ```

  Expected: PASS, with no replay-buffer assertions added or changed.

- [ ] **Step 9: Commit epoch checkpoints**

  ```bash
  git add dreamervla/runners/world_model_training_runner.py \
    dreamervla/runners/success_classifier_training_runner.py \
    dreamervla/utils/run_paths.py dreamervla/config.py \
    dreamervla/diagnostics/experiment_stage_checks.py \
    configs/experiment/wm_full_dataset_train.yaml \
    configs/experiment/wm_official_upper_bound_profile.yaml \
    configs/experiment/wmpo_token_classifier_openvla_onetraj_libero_goal_h1.yaml \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_success_classifier_training_runner.py \
    tests/unit_tests/test_world_model_training_config.py \
    tests/unit_tests/test_run_paths.py
  git commit -s -m "refactor: checkpoint warmup at epoch boundaries"
  ```

## Task 5: Correct metric resume steps for all three routes

**Files:**

- Modify: `dreamervla/runners/world_model_training_runner.py`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `tests/unit_tests/test_world_model_training_runner.py`
- Modify: `tests/unit_tests/test_cotrain_resume.py`
- Modify: `tests/unit_tests/test_metric_logger.py`

- [ ] **Step 1: Add failing route-axis tests**

  Add explicit assertions:

  ```python
  # WM-only resume
  assert runner._metric_resume_step == restored_wm_step

  # classifier phase on a shared WM+classifier axis
  assert runner._metric_resume_step == wm_total_steps + restored_classifier_step

  # cotrain
  assert runner._metric_resume_step == restored_global_step
  ```

  Also assert `_metric_logger is None` at the moment each resume setter is called.

- [ ] **Step 2: Confirm current failures**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_cotrain_resume.py -q
  ```

  Expected: FAIL for combined classifier warmup and cotrain.

- [ ] **Step 3: Set route-correct resume steps before logging**

  In combined warmup, make the loader return component progress without directly applying it to the metric axis. The route applies:

  ```python
  metric_resume_step = (
      restored_wm_step
      if active_component == "world_model"
      else wm_total_steps + restored_classifier_step
  )
  self.set_metric_resume_step(metric_resume_step)
  ```

  In cotrain's manual-resume path:

  ```python
  self.global_step = resume_step
  self.set_metric_resume_step(resume_step)
  ```

  Keep standalone classifier using restored `self.global_step` from `BaseRunner`.

- [ ] **Step 4: Verify TensorBoard purge semantics**

  Use the fake `SummaryWriter` in `test_metric_logger.py` to assert exactly the same value becomes `purge_step`; a fresh run must pass `None`.

- [ ] **Step 5: Run focused tests and commit**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_cotrain_resume.py \
    tests/unit_tests/test_metric_logger.py -q
  git add dreamervla/runners/world_model_training_runner.py \
    dreamervla/runners/cotrain_runner.py \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_cotrain_resume.py \
    tests/unit_tests/test_metric_logger.py
  git commit -s -m "fix: align metric history on resume"
  ```

## Task 6: Flatten TensorBoard and W&B output

**Files:**

- Modify: `dreamervla/utils/metric_logger.py`
- Modify: `tests/unit_tests/test_metric_logger.py`
- Modify: `tests/unit_tests/test_runner_artifacts.py`

- [ ] **Step 1: Add failing filesystem-layout tests**

  Make the fake W&B SDK emulate the real SDK by creating its binary stream under `Path(init_kwargs["dir"]) / "wandb" / "offline-run-20260715_120000-stableid"`. Assert:

  ```python
  assert init_kwargs["dir"] == str(run_root)
  assert (run_root / "wandb" / "run_id.txt").is_file()
  assert list((run_root / "wandb").glob("offline-run-*"))
  assert not (run_root / "wandb" / "wandb").exists()
  assert not (run_root / "tensorboard" / "config.yaml").exists()
  ```

  Keep legacy nested run-ID discovery tests.

- [ ] **Step 2: Confirm current failure**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_metric_logger.py \
    tests/unit_tests/test_runner_artifacts.py -q
  ```

  Expected: FAIL because W&B receives `run_root / "wandb"` as `dir` and creates another `wandb/`, while TensorBoard writes a config copy.

- [ ] **Step 3: Separate W&B identity location from SDK base directory**

  Use the W&B child of the SDK base directory only for `run_id.txt` lookup and validation, but pass the SDK base directory itself with `wandb.init(dir=str(sdk_base))`:

  ```python
  sdk_base = Path(log_path) / log_path_suffix
  identity_dir = sdk_base / "wandb"
  identity_dir.mkdir(parents=True, exist_ok=True)
  run_id, existing = _resolve_wandb_run_id(wandb, identity_dir, resume=self.resume)
  init_kwargs["dir"] = str(sdk_base)
  ```

  Preserve stable offline IDs, online `resume_from=f"{run_id}?_step={resume_step}"` when supported, and `resume="allow"` fallback.

- [ ] **Step 4: Remove TensorBoard's config copy**

  Delete the complete `OmegaConf.save` call that writes `tensorboard_log_path / "config.yaml"`. The sole configuration source is `.hydra/config.yaml`.

- [ ] **Step 5: Run tests and commit**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_metric_logger.py \
    tests/unit_tests/test_runner_artifacts.py -q
  git add dreamervla/utils/metric_logger.py \
    tests/unit_tests/test_metric_logger.py \
    tests/unit_tests/test_runner_artifacts.py
  git commit -s -m "fix: flatten metric logger output"
  ```

## Task 7: Add one-command offline W&B upload

**Files:**

- Create: `scripts/utils/wandb_sync.sh`
- Create: `tests/unit_tests/test_wandb_sync_script.py`
- Modify: `scripts/README.md`

- [ ] **Step 1: Write a fake-CLI test harness**

  The test creates canonical and legacy offline segments and puts an executable fake `wandb` first on `PATH`. The fake records every argument and optionally returns a requested nonzero status. Cover:

  - one canonical segment;
  - multiple chronological segments (`--append` only after the first logical upload);
  - legacy `wandb/wandb/offline-run-*`;
  - `.synced` marker skipping and idempotent rerun;
  - missing CLI, invalid directory, no segment, invalid/conflicting IDs;
  - CLI failure propagation;
  - local files remain present and are never renamed/deleted by the script.

- [ ] **Step 2: Run the new test and confirm failure**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_wandb_sync_script.py -q
  ```

  Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement the one-argument shell interface**

  The script must use `set -euo pipefail`, require exactly one argument, verify `command -v wandb`, and validate IDs against `^[A-Za-z0-9_-]+$`. Discovery must include only:

  ```text
  "$wandb_dir"/{offline-run,run}-*/run-*.wandb
  "$wandb_dir"/wandb/{offline-run,run}-*/run-*.wandb
  ```

  Include `.wandb.synced` files only as markers, not upload inputs. Sort by run-directory timestamp/name, use `run_id.txt` when present, otherwise derive the ID from the earliest stream filename, and reject any other segment ID that disagrees.

  Invoke the CLI as argument arrays, never `eval`:

  ```bash
  if [[ "$logical_run_already_synced" == true ]]; then
      wandb sync --append --id "$run_id" "$stream"
  else
      wandb sync --id "$run_id" "$stream"
      logical_run_already_synced=true
  fi
  ```

  Do not pass entity/project flags; W&B reads offline metadata. Do not perform login, delete files, or rename files.

- [ ] **Step 4: Verify shell behavior**

  ```bash
  bash -n scripts/utils/wandb_sync.sh
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_wandb_sync_script.py -q
  ```

  Expected: PASS.

- [ ] **Step 5: Document and commit**

  Add exactly this primary usage to `scripts/README.md`:

  ```bash
  wandb login
  bash scripts/utils/wandb_sync.sh /path/to/run_root/wandb
  ```

  Mention `WANDB_API_KEY` as the noninteractive login alternative and nothing else as required configuration.

  ```bash
  git add scripts/utils/wandb_sync.sh \
    tests/unit_tests/test_wandb_sync_script.py scripts/README.md
  git commit -s -m "feat: sync offline wandb runs"
  ```

## Task 8: Update repository documentation and remove stale layout claims

**Files:**

- Modify: `AGENTS.md`
- Modify: `configs/README.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `docs/data_layout.md`
- Modify: `docs/repository_structure.md`
- Modify: `spec/04_complete_loop.md`
- Modify: `spec/99_manual_notes.md` only if it currently states a conflicting artifact path

- [ ] **Step 1: Search for stale contracts**

  ```bash
  rg -n "resolved_config\.yaml|warmup_progress|wandb/wandb|warmup_checkpoint_every|ckpt_every" \
    AGENTS.md configs docs spec dreamervla tests scripts
  ```

  Classify each hit as current contract, deliberate legacy compatibility, or stale documentation. Do not remove compatibility code/tests merely to make the search empty.

- [ ] **Step 2: Update the canonical documentation**

  Document the five-entry shallow tree, `.hydra/config.yaml` as canonical config, runtime-only manifest, epoch checkpoint cadence names, stable W&B ID, and the sync command. Explicitly state replay buffer is outside this resume contract.

- [ ] **Step 3: Run documentation/config tests**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_world_model_training_config.py \
    tests/unit_tests/test_runner_artifacts.py \
    tests/unit_tests/test_run_config.py -q
  ```

- [ ] **Step 4: Commit documentation**

  ```bash
  git add AGENTS.md configs/README.md docs/PARAMETERS.md docs/data_layout.md \
    docs/repository_structure.md spec/04_complete_loop.md spec/99_manual_notes.md
  git commit -s -m "docs: define canonical training artifacts"
  ```

## Task 9: Full verification and clean-tree audit

**Files:**

- Verify only; modify the smallest responsible source/test if a check exposes a defect.

- [ ] **Step 1: Run formatting and static checks**

  ```bash
  ruff format --check dreamervla tests
  ruff check dreamervla tests
  bash -n scripts/utils/wandb_sync.sh
  git diff --check
  ```

- [ ] **Step 2: Run the complete focused suite**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/unit_tests/test_runner_public_api.py \
    tests/unit_tests/test_native_hydra_artifacts.py \
    tests/unit_tests/test_runner_artifacts.py \
    tests/unit_tests/test_run_config.py \
    tests/unit_tests/test_metric_logger.py \
    tests/unit_tests/test_rng_checkpoint.py \
    tests/unit_tests/test_run_paths.py \
    tests/unit_tests/test_world_model_training_config.py \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_success_classifier_training_runner.py \
    tests/unit_tests/test_cotrain_resume.py \
    tests/unit_tests/test_cotrain_worker_rng.py \
    tests/unit_tests/test_wandb_sync_script.py -q
  ```

- [ ] **Step 3: Run broader unit coverage**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests -q
  ```

  Do not treat replay-specific pre-existing failures as authorization to modify replay behavior. Record any unrelated failure with its exact command and traceback.

- [ ] **Step 4: Run gated smoke tests when their prerequisites are available**

  ```bash
  /home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
    tests/e2e_tests/test_world_model_env_ray_smoke.py \
    tests/e2e_tests/test_cotrain_smoke.py -q
  ```

  If the tests skip because Ray/GPU/LIBERO prerequisites are absent, report the skips rather than claiming execution coverage.

- [ ] **Step 5: Audit a real temporary run tree**

  Run the smallest existing offline smoke recipe with a temporary output root, then assert:

  ```text
  /tmp/dvla_artifact_smoke/checkpoints/
  /tmp/dvla_artifact_smoke/tensorboard/                 # only if enabled
  /tmp/dvla_artifact_smoke/wandb/                       # only if enabled
  /tmp/dvla_artifact_smoke/.hydra/{config,overrides,hydra}.yaml
  /tmp/dvla_artifact_smoke/run_manifest.json
  ```

  Confirm there is no root `resolved_config.yaml`, no `tensorboard/config.yaml`, no new `warmup_progress/`, and no `wandb/wandb/`. Resume the same run once and confirm it reuses the original run root and stable W&B ID.

- [ ] **Step 6: Inspect final changes**

  ```bash
  git status --short
  git diff --stat HEAD~8..HEAD
  git log --oneline --decorate -12
  ```

  Verify that no replay-buffer implementation/test files were changed for this work and that unrelated user files remain untouched.

- [ ] **Step 7: Commit any verification-only correction**

  Only if Step 1–6 required a fix, stage each corrected path explicitly after reviewing `git diff`, then commit with `git commit -s -m "fix: complete resume verification"`. Do not use `git add -A`.

## Acceptance checklist

- [ ] New invocations use Hydra's native runtime and native `.hydra/` snapshots.
- [ ] Root `resolved_config.yaml`, TensorBoard config copies, and nested W&B directories are not produced.
- [ ] Manifest contains runtime facts without duplicated configuration or derivable paths.
- [ ] WM and classifier write one canonical epoch-boundary route checkpoint; cotrain keeps point-in-time checkpoints plus `latest.ckpt`.
- [ ] Strict resume restores every enabled model and optimizer, progress, applicable threshold/best metric, and per-process Python/NumPy/Torch CPU/CUDA RNG.
- [ ] TensorBoard purge and W&B rewind/append use the correct route-global step.
- [ ] Offline W&B upload requires only `bash scripts/utils/wandb_sync.sh /path/to/run_root/wandb` after login/API-key setup.
- [ ] Legacy artifacts remain readable, while all new writes use canonical paths.
- [ ] Replay-buffer behavior was neither modified nor added to the acceptance criteria.
