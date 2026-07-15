# Diagnostics and Runtime Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the retired non-Ray online-cotrain implementation and unreachable diagnostics/runtime code while preserving Ray cotrain, standalone warmup, evaluation, checkpoint/logger resume, and all replay behavior.

**Architecture:** `WorldModelTrainingRunner` becomes an offline warmup-only runner and rejects nonzero online environment budgets during config validation. Ray `CotrainRunner` and its Actor/Rollout/Env/Learner groups remain the sole online-cotrain implementation. Production evaluation support moves from `diagnostics/` into `runtime/`; low-risk single-consumer helpers are colocated without flattening the existing runtime boundaries.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, PyTorch, Ray, pytest, Ruff, Bash.

---

## Scope guardrails

- Do not change `online_replay.py`, `offline_seed.py`, Ray replay workers, replay checkpoint payloads, replay schemas, sampling, or restore behavior.
- Preserve `wm_warmup.ckpt`, `classifier_warmup.ckpt`, optimizer/RNG resume, TensorBoard purge-step behavior, W&B run identity, and `scripts/utils/wandb_sync.sh`.
- Preserve current Ray collection/cotrain and LIBERO evaluation behavior.
- Do not add compatibility shims for private orphan modules. Update repository consumers atomically.
- Preserve unrelated untracked files `findings.md`, `progress.md`, and `task_plan.md`.

## Target file structure

After the cleanup:

- `dreamervla/diagnostics/` contains only executable install/eval/smoke/measurement tools with a current launcher, documentation reference, or explicit troubleshooting role.
- `dreamervla/runtime/cotrain_eval.py` contains the cotrain evaluation observer used by `LIBEROVLAEvaluationRunner`.
- `dreamervla/runtime/metrics.py` contains `SuccessTracker`; callers no longer import `online_utils.py`.
- `dreamervla/runtime/rollout_collection_ray.py` owns its one-consumer collection-config builder.
- `dreamervla/runtime/world_model_training_common.py` contains only construction and validation helpers required by offline WM/classifier warmup.

## Task 1: Delete definite orphan files and symbols

**Files:**

- Delete: `dreamervla/diagnostics/_common.py`
- Delete: `dreamervla/diagnostics/wandb_relay_sync.py`
- Delete: `dreamervla/diagnostics/wm_single_trajectory_raw_overfit.py`
- Delete: `dreamervla/diagnostics/wm_single_trajectory_vla_overfit.py`
- Delete: `tests/unit_tests/test_wm_single_trajectory_raw_overfit.py`
- Delete: `tests/unit_tests/test_wm_single_trajectory_vla_overfit.py`
- Modify: `dreamervla/diagnostics/eval_dino_token_wm.py`
- Modify: `dreamervla/runtime/oft_collect.py`
- Modify: `dreamervla/runtime/libero_rollout.py`
- Modify: `dreamervla/runtime/rollout_collection_ray.py`
- Modify: `dreamervla/runtime/libero_vla_evaluation_base.py`
- Modify: `dreamervla/runtime/libero_vla_eval_latent.py`
- Modify: `dreamervla/runtime/world_model_training_base.py`
- Modify: `tests/unit_tests/test_ray_coldstart_real_config.py`
- Modify: `tests/unit_tests/test_libero_rollout_runner.py`
- Modify: `tests/unit_tests/test_eval_dino_token_wm.py`
- Modify: `tests/unit_tests/test_repository_hygiene.py`

- [ ] **Step 1: Add a repository-hygiene test describing the deleted surface**

  In `tests/unit_tests/test_repository_hygiene.py`, add a test with the exact retired paths and symbols:

  ```python
  def test_retired_diagnostics_and_runtime_helpers_are_absent() -> None:
      root = Path(__file__).resolve().parents[2]
      for relative in (
          "dreamervla/diagnostics/_common.py",
          "dreamervla/diagnostics/wandb_relay_sync.py",
          "dreamervla/diagnostics/wm_single_trajectory_raw_overfit.py",
          "dreamervla/diagnostics/wm_single_trajectory_vla_overfit.py",
      ):
          assert not (root / relative).exists()

      forbidden = {
          "dreamervla/runtime/oft_collect.py": ("OFTOpenLoopStep", "oft_open_loop_action"),
          "dreamervla/runtime/libero_rollout.py": ("build_grid_work_list", "SuccessTally"),
          "dreamervla/runtime/rollout_collection_ray.py": ("_next_ray_task_id", "_ray_start_episode_id"),
          "dreamervla/runtime/libero_vla_evaluation_base.py": ("_EvalInferResult",),
          "dreamervla/runtime/libero_vla_eval_latent.py": ("_dreamer_action_from_latent",),
          "dreamervla/runtime/world_model_training_base.py": ("_make_obs_dict",),
          "dreamervla/diagnostics/eval_dino_token_wm.py": ("load_dino_token_world_model",),
      }
      for relative, names in forbidden.items():
          tree = ast.parse((root / relative).read_text())
          definitions = {
              node.name
              for node in ast.walk(tree)
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
          }
          assert definitions.isdisjoint(names)
  ```

- [ ] **Step 2: Run the hygiene test and verify it fails**

  ```bash
  python -m pytest \
    tests/unit_tests/test_repository_hygiene.py::test_retired_diagnostics_and_runtime_helpers_are_absent -q
  ```

  Expected: FAIL because the retired files and definitions still exist.

- [ ] **Step 3: Delete the unreachable files, definitions, and definition-only tests**

  Remove the exact files and definitions listed in Step 1. In
  `test_ray_coldstart_real_config.py`, delete only
  `test_ray_task_scheduler_expands_all_and_reserves_round_robin` and
  `test_ray_start_episode_id_adds_resume_offset_to_scheduled_index`. In
  `test_libero_rollout_runner.py`, delete the imports and tests for
  `build_grid_work_list`/`SuccessTally`, retaining `run_vectorized_rollout` coverage.

- [ ] **Step 4: Run focused tests**

  ```bash
  python -m pytest \
    tests/unit_tests/test_repository_hygiene.py \
    tests/unit_tests/test_ray_coldstart_real_config.py \
    tests/unit_tests/test_libero_rollout_runner.py \
    tests/unit_tests/test_eval_dino_token_wm.py -q
  ```

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add -A dreamervla/diagnostics dreamervla/runtime \
    tests/unit_tests/test_repository_hygiene.py \
    tests/unit_tests/test_ray_coldstart_real_config.py \
    tests/unit_tests/test_libero_rollout_runner.py \
    tests/unit_tests/test_eval_dino_token_wm.py \
    tests/unit_tests/test_wm_single_trajectory_raw_overfit.py \
    tests/unit_tests/test_wm_single_trajectory_vla_overfit.py
  git commit -s -m "refactor: remove orphan diagnostics helpers"
  ```

## Task 2: Retire the non-Ray online-cotrain route

**Files:**

- Modify: `dreamervla/config.py`
- Modify: `dreamervla/runners/world_model_training_runner.py`
- Modify: `dreamervla/runtime/world_model_training_common.py`
- Delete: `tests/unit_tests/test_cotrain_render_backend.py`
- Modify: `tests/unit_tests/test_config_validation.py`
- Modify: `tests/unit_tests/test_world_model_training_runner.py`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`

- [ ] **Step 1: Write the failing config-boundary test**

  Add to `tests/unit_tests/test_config_validation.py` using the existing minimal valid
  `WorldModelTrainingRunner` fixture/config helper:

  ```python
  def test_world_model_training_runner_rejects_non_ray_online_cotrain(tmp_path: Path) -> None:
      cfg = _valid_world_model_training_cfg(tmp_path)
      OmegaConf.update(cfg, "online_rollout.total_env_steps", 1, force_add=True)
      with pytest.raises(
          ValueError,
          match="WorldModelTrainingRunner only supports offline warmup.*CotrainRunner",
      ):
          validate_cfg(cfg)
  ```

  The helper must create the required offline data/hidden directories so the new assertion
  is the first failing invariant.

- [ ] **Step 2: Run the test and verify it fails**

  ```bash
  python -m pytest \
    tests/unit_tests/test_config_validation.py::test_world_model_training_runner_rejects_non_ray_online_cotrain -q
  ```

  Expected: FAIL because nonzero `total_env_steps` is still accepted.

- [ ] **Step 3: Add the explicit validation boundary**

  At the start of `_validate_online_cotrain_pipeline` after checking the target, add:

  ```python
  total_env_steps = int(
      OmegaConf.select(cfg, "online_rollout.total_env_steps", default=0) or 0
  )
  if total_env_steps != 0:
      raise ValueError(
          "WorldModelTrainingRunner only supports offline warmup; "
          "use CotrainRunner and the Ray cotrain experiment for online training"
      )
  ```

- [ ] **Step 4: Make `WorldModelTrainingRunner.run()` end after warmup**

  Delete `_prepare_online_resume`, online-only encoder restoration, the `[3/3] ONLINE COTRAIN`
  banner, updates to `training.warmup_steps`/`online_rollout.debug_warmup_steps`, and the call
  to `_online_cotrain_loop`. The final branch becomes:

  ```python
  if self.distributed.is_main_process:
      self.console_banner("WARMUP COMPLETE", done=True)
  return []
  ```

  Keep all model/optimizer construction, offline replay seeding, WM/classifier epoch loops,
  canonical checkpoint writes, top-k behavior, and strict resume logic.

- [ ] **Step 5: Remove the retired common implementation and its tests**

  From `world_model_training_common.py`, delete online-only imports, free functions
  `build_cotrain_replay_transition`, `validate_rollout_cfg`,
  `build_rollout_progress_metrics`, `_cfg_select`, `_cotrain_render_gpu_pool`, and
  `build_rollout_vec_env`. Delete methods whose dependency closure is rooted at
  `_online_cotrain_loop`, including environment construction, hidden extraction, actor rollout,
  vectorized rollout, training bursts, online sidecars, and online checkpoint saving.

  Retain these shared definitions and every helper transitively used by offline warmup:

  ```python
  _component_hydra_cfg
  _world_model_ddp_wrap_kwargs
  validate_task_conditioning_cfg
  _WorldModelTrainingCommon._build_trainable_classifier
  _WorldModelTrainingCommon._load_world_model_init_ckpt
  _WorldModelTrainingCommon._assert_optimizers_disjoint
  _WorldModelTrainingCommon._build_components
  ```

  Make `_build_components` always skip rollout-only encoder/processor/extractor construction:

  ```python
  self.encoder = None
  self.processor = None
  self._oft_hidden_token_extractor = None
  ```

  Delete `test_cotrain_render_backend.py` and online-only tests in
  `test_world_model_training_runner.py`: registry-driven actor update, training bursts,
  online/vectorized rollout, online environment contract, online checkpoint sidecars, and
  online-resume ownership. Rewrite orchestration assertions to expect warmup completion without
  an online-loop call. Retain all offline warmup/checkpoint/RNG/metric-axis/calibration tests.

- [ ] **Step 6: Prove the old route is absent and the warmup route still works**

  ```bash
  ! rg -n "_online_cotrain_loop|_prepare_online_resume|build_cotrain_replay_transition|build_rollout_vec_env" \
    dreamervla tests/unit_tests
  python -m pytest \
    tests/unit_tests/test_config_validation.py \
    tests/unit_tests/test_world_model_training_config.py \
    tests/unit_tests/test_world_model_training_device.py \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_task_conditioning.py -q
  ```

  Expected: no `rg` matches and all tests PASS.

- [ ] **Step 7: Commit**

  ```bash
  git add -A dreamervla/config.py dreamervla/runners/world_model_training_runner.py \
    dreamervla/runtime/world_model_training_common.py \
    tests/unit_tests/test_cotrain_render_backend.py \
    tests/unit_tests/test_config_validation.py \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_experiment_stage_scripts.py
  git commit -s -m "refactor: retire non-ray online cotrain"
  ```

## Task 3: Trim `experiment_stage_checks` to its live classifier-eval command

**Files:**

- Modify: `dreamervla/diagnostics/experiment_stage_checks.py`
- Modify: `tests/unit_tests/test_experiment_stage_scripts.py`
- Modify: `tests/unit_tests/test_repository_hygiene.py`

- [ ] **Step 1: Change the command-surface test to the live contract**

  Replace the obsolete all-command assertion with:

  ```python
  def test_experiment_stage_checks_only_exposes_classifier_eval() -> None:
      parser = experiment_stage_checks.build_parser()
      subparsers = next(
          action
          for action in parser._actions
          if isinstance(action, argparse._SubParsersAction)
      )
      assert set(subparsers.choices) == {"cls-eval"}
  ```

- [ ] **Step 2: Run the test and verify it fails**

  ```bash
  python -m pytest \
    tests/unit_tests/test_experiment_stage_scripts.py::test_experiment_stage_checks_only_exposes_classifier_eval -q
  ```

  Expected: FAIL because the parser still exposes the retired stage commands.

- [ ] **Step 3: Reduce the module to the live command dependency closure**

  Keep `_print_json`, `_require_path`, `_latest_child`, `_read_jsonl`, `cls_eval`,
  `build_parser`, and `main`, plus standard-library imports used by them. Remove collection,
  original-data, warmup, non-Ray RL, pack-init, and cotrain-check functions and parser branches.
  Keep the CLI entrypoint:

  ```python
  if __name__ == "__main__":
      raise SystemExit(main())
  ```

  Delete tests for removed commands. Preserve the `cls-eval` subprocess/behavior tests and the
  shell launcher tests proving `scripts/experiments/classifier_training/eval.sh` calls it.

- [ ] **Step 4: Run focused tests and shell syntax**

  ```bash
  python -m pytest \
    tests/unit_tests/test_experiment_stage_scripts.py \
    tests/unit_tests/test_repository_hygiene.py -q
  bash -n scripts/experiments/classifier_training/eval.sh
  ```

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add dreamervla/diagnostics/experiment_stage_checks.py \
    tests/unit_tests/test_experiment_stage_scripts.py \
    tests/unit_tests/test_repository_hygiene.py
  git commit -s -m "refactor: trim retired experiment stage commands"
  ```

## Task 4: Move production cotrain evaluation out of diagnostics

**Files:**

- Create: `dreamervla/runtime/cotrain_eval.py`
- Delete: `dreamervla/diagnostics/eval_cotrain_transaction.py`
- Modify: `dreamervla/runners/libero_vla_evaluation_runner.py`
- Modify: `tests/unit_tests/test_cotrain_transaction_eval.py`
- Modify: `tests/unit_tests/test_repository_hygiene.py`

- [ ] **Step 1: Write the architecture-boundary assertion**

  Add to `test_repository_hygiene.py`:

  ```python
  def test_cotrain_eval_observer_lives_in_runtime() -> None:
      root = Path(__file__).resolve().parents[2]
      runner = (root / "dreamervla/runners/libero_vla_evaluation_runner.py").read_text()
      assert "from dreamervla.runtime.cotrain_eval import CotrainEvalObserver" in runner
      assert not (root / "dreamervla/diagnostics/eval_cotrain_transaction.py").exists()
  ```

- [ ] **Step 2: Run the test and verify it fails**

  ```bash
  python -m pytest \
    tests/unit_tests/test_repository_hygiene.py::test_cotrain_eval_observer_lives_in_runtime -q
  ```

  Expected: FAIL because production still imports from `diagnostics`.

- [ ] **Step 3: Move the module and update imports without behavior changes**

  Move the full implementation to `dreamervla/runtime/cotrain_eval.py`. Update runner and test
  imports to:

  ```python
  from dreamervla.runtime.cotrain_eval import CotrainEvalObserver
  ```

  Do not rename public classes/functions or change metric keys, accumulation, trajectory encoding,
  classifier scoring, or closed-loop rollout behavior.

- [ ] **Step 4: Run evaluation tests**

  ```bash
  python -m pytest \
    tests/unit_tests/test_cotrain_transaction_eval.py \
    tests/unit_tests/test_libero_eval_protocol_compat.py \
    tests/unit_tests/test_repository_hygiene.py -q
  ```

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add -A dreamervla/runtime/cotrain_eval.py \
    dreamervla/diagnostics/eval_cotrain_transaction.py \
    dreamervla/runners/libero_vla_evaluation_runner.py \
    tests/unit_tests/test_cotrain_transaction_eval.py \
    tests/unit_tests/test_repository_hygiene.py
  git commit -s -m "refactor: move cotrain evaluation into runtime"
  ```

## Task 5: Consolidate low-risk runtime fragments

**Files:**

- Create: `dreamervla/runtime/metrics.py`
- Delete: `dreamervla/runtime/online_utils.py`
- Delete: `dreamervla/runtime/rollout_collection_config.py`
- Delete: `dreamervla/runtime/world_model_training_utils.py`
- Modify: `dreamervla/runners/base_runner.py`
- Modify: `dreamervla/runtime/rollout_collection_ray.py`
- Modify: `dreamervla/runtime/world_model_training_base.py`
- Modify: `tests/unit_tests/test_console.py`
- Delete: `tests/unit_tests/test_wm_state_loader.py`
- Modify: `tests/unit_tests/test_rng_checkpoint.py`
- Modify: `tests/unit_tests/test_ray_coldstart_real_config.py`
- Modify: `tests/unit_tests/test_repository_hygiene.py`

- [ ] **Step 1: Add module-boundary tests**

  Add assertions that `SuccessTracker` imports from `runtime.metrics`, the three fragment files are
  absent, and `build_oft_collect_config` remains importable from `rollout_collection_ray`:

  ```python
  def test_runtime_fragments_are_consolidated() -> None:
      from dreamervla.runtime.metrics import SuccessTracker
      from dreamervla.runtime.rollout_collection_ray import build_oft_collect_config

      assert SuccessTracker(window=2).rate() == 0.0
      assert callable(build_oft_collect_config)
      root = Path(__file__).resolve().parents[2]
      for relative in (
          "dreamervla/runtime/online_utils.py",
          "dreamervla/runtime/rollout_collection_config.py",
          "dreamervla/runtime/world_model_training_utils.py",
      ):
          assert not (root / relative).exists()
  ```

- [ ] **Step 2: Run the test and verify it fails**

  ```bash
  python -m pytest \
    tests/unit_tests/test_repository_hygiene.py::test_runtime_fragments_are_consolidated -q
  ```

  Expected: FAIL because the fragment modules still exist.

- [ ] **Step 3: Move the two live helpers to their consumers/domains**

  Move `SuccessTracker` unchanged into `runtime/metrics.py`, including its bounded deque behavior:

  ```python
  class SuccessTracker:
      def __init__(self, window: int) -> None:
          self._buf: deque[float] = deque(maxlen=max(1, int(window)))
          self._best: float = 0.0
          self._last_printed: float | None = None

      def update(self, success: bool) -> None:
          self._buf.append(1.0 if success else 0.0)
          if len(self._buf) == self._buf.maxlen:
              self._best = max(self._best, self.rate())

      def rate(self) -> float:
          return (sum(self._buf) / len(self._buf)) if self._buf else 0.0

      @property
      def best(self) -> float:
          return self._best

      def delta(self) -> float:
          return 0.0 if self._last_printed is None else self.rate() - self._last_printed

      def mark_printed(self) -> None:
          self._last_printed = self.rate()

      def __len__(self) -> int:
          return len(self._buf)
  ```

  Migrate `BaseRunner` and `test_console.py` imports together without changing call sites.

  Move `build_oft_collect_config` and its imports into `rollout_collection_ray.py`; update its tests
  to import it from that module. Delete `rollout_collection_config.py`.

- [ ] **Step 4: Delete obsolete checkpoint helpers without weakening shared RNG tests**

  Move `save_viz_strip` into `world_model_training_base.py` immediately above its only consumer,
  preserving exact rendering behavior. Delete `DreamerCkptResumeMixin` and `to_device` with
  `world_model_training_utils.py`. Delete only the two DreamerV3 mixin tests at the end of
  `test_rng_checkpoint.py`; retain all shared BaseRunner Python/NumPy/Torch CPU/CUDA RNG tests.
  Delete `test_wm_state_loader.py` together with the uncalled loaders from `online_utils.py`.

- [ ] **Step 5: Run focused tests**

  ```bash
  python -m pytest \
    tests/unit_tests/test_console.py \
    tests/unit_tests/test_rng_checkpoint.py \
    tests/unit_tests/test_ray_coldstart_real_config.py \
    tests/unit_tests/test_world_model_training_device.py \
    tests/unit_tests/test_repository_hygiene.py -q
  ```

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add -A dreamervla/runtime dreamervla/runners/base_runner.py \
    tests/unit_tests/test_console.py tests/unit_tests/test_wm_state_loader.py \
    tests/unit_tests/test_rng_checkpoint.py \
    tests/unit_tests/test_ray_coldstart_real_config.py \
    tests/unit_tests/test_world_model_training_device.py \
    tests/unit_tests/test_repository_hygiene.py
  git commit -s -m "refactor: consolidate runtime helpers"
  ```

## Task 6: Update documentation and run release verification

**Files:**

- Modify: `AGENTS.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `docs/repository_structure.md`
- Modify: `docs/README.md`
- Modify: any current docs found by the exact retired-name scan

- [ ] **Step 1: Find and remove active documentation for retired surfaces**

  ```bash
  rg -n "non-Ray|_online_cotrain_loop|wandb_relay_sync|eval_cotrain_transaction|wm_single_trajectory_(raw|vla)_overfit|online_rollout.total_env_steps" \
    AGENTS.md README.md README.zh-CN.md docs configs scripts dreamervla tests \
    --glob '!docs/superpowers/**'
  ```

  Update active docs so they say:

  ```text
  WorldModelTrainingRunner performs offline world-model warmup only.
  SuccessClassifierTrainingRunner performs classifier warmup.
  CotrainRunner is the Ray online-cotrain route.
  ```

  Keep `online_rollout.total_env_steps: 0` only where an active static config still needs it for
  compatibility; remove its parameter-table claim as an online budget for
  `WorldModelTrainingRunner`.

- [ ] **Step 2: Run static validation**

  ```bash
  python -m ruff check dreamervla tests
  python -m ruff format --check dreamervla tests
  find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
  git diff --check
  ```

  Expected: all commands exit 0.

- [ ] **Step 3: Run route-focused regression tests**

  ```bash
  python -m pytest \
    tests/unit_tests/test_world_model_training_runner.py \
    tests/unit_tests/test_success_classifier_training_runner.py \
    tests/unit_tests/test_cotrain_resume.py \
    tests/unit_tests/test_rng_checkpoint.py \
    tests/unit_tests/test_metric_logger.py \
    tests/unit_tests/test_wandb_sync_launcher.py \
    tests/unit_tests/test_cotrain_transaction_eval.py \
    tests/unit_tests/test_repository_hygiene.py -q
  ```

- [ ] **Step 4: Run the complete unit test suite**

  ```bash
  python -m pytest tests/unit_tests -q
  ```

  Expected: PASS. GPU/Ray/real-environment e2e tests may remain gated, but collection/cotrain module
  import tests must pass.

- [ ] **Step 5: Prove no retired production references remain**

  ```bash
  ! rg -n "dreamervla\.diagnostics\.eval_cotrain_transaction|dreamervla\.runtime\.(online_utils|rollout_collection_config|world_model_training_utils)|_online_cotrain_loop|_prepare_online_resume" \
    dreamervla tests configs scripts AGENTS.md README.md README.zh-CN.md docs \
    --glob '!docs/superpowers/**'
  git status --short
  ```

  Expected: the reference scan is empty; status contains only intended task changes and the user's
  pre-existing untracked planning files.

- [ ] **Step 6: Commit documentation and final hygiene changes**

  ```bash
  git add AGENTS.md docs/PARAMETERS.md docs/repository_structure.md docs/README.md \
    README.md README.zh-CN.md dreamervla tests configs scripts
  git commit -s -m "docs: align runtime ownership with ray cotrain"
  ```
