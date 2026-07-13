# Mainline Runner Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace experimental runner branches with five canonical production runners, preserve independent World Model and Success Classifier training, run staged cotrain with resident Ray models and resident evaluation, expose phase-specific progress, and prove debug limits in a real eight-GPU run.

**Architecture:** A Python launcher routes checkpoint-presence cases through collection and independent component training before entering one `CotrainRunner`. During cotrain, Actor, Rollout, Learner, WorldModelEnv, real environment, evaluation environment, and replay groups are launched once; barriers serialize real rollout, VLA SFT, WM/CLS training, imagined rollout, VLA PPO, and read-only evaluation. The public runner package exports only complete role names and retains no aliases for removed experiments.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, PyTorch FSDP/DDP, Ray actors and channels, OpenVLA-OFT/Hugging Face checkpoints, LIBERO/robosuite OSMesa, pytest.

---

### Task 1: Lock the canonical public runner API

**Files:**
- Modify: `tests/unit_tests/test_runner_public_api.py`
- Modify: `dreamervla/runners/__init__.py`
- Create: `dreamervla/runners/rollout_collection_runner.py`
- Create: `dreamervla/runners/world_model_training_runner.py`
- Create: `dreamervla/runners/success_classifier_training_runner.py`
- Create: `dreamervla/runners/cotrain_runner.py`
- Create: `dreamervla/runners/libero_vla_evaluation_runner.py`

- [ ] **Step 1: Write the failing public-surface test**

```python
def test_public_runner_surface_contains_only_mainline_roles() -> None:
    import dreamervla.runners as runners

    assert runners.PUBLIC_RUNNERS == [
        "RolloutCollectionRunner",
        "WorldModelTrainingRunner",
        "SuccessClassifierTrainingRunner",
        "CotrainRunner",
        "LIBEROVLAEvaluationRunner",
    ]
    for removed in (
        "JointDreamerVLARunner",
        "ManualCotrainRayRunner",
        "OnlineCotrainRunner",
        "OnlineCotrainPipelineRunner",
        "OnlineCotrainRayRunner",
        "FrozenModelPolicyRunner",
    ):
        assert not hasattr(runners, removed)
```

- [ ] **Step 2: Run the test and verify that old names fail the contract**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_runner_public_api.py`

Expected: FAIL because `_RUNNER_SPECS` still exports legacy runners.

- [ ] **Step 3: Replace `_RUNNER_SPECS` with the five canonical names**

Use direct implementation classes in the new modules and keep lazy loading only for
the five public targets. Each class declares a complete `runner_name`,
`runner_status = "current"`, and role-appropriate `runner_family`.

- [ ] **Step 4: Run the public API test**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_runner_public_api.py`

Expected: PASS.

- [ ] **Step 5: Commit only canonical API files when the worktree permits isolated staging**

```bash
git add dreamervla/runners/__init__.py dreamervla/runners/*_runner.py \
  tests/unit_tests/test_runner_public_api.py
git commit -s -m "refactor: define canonical mainline runners"
```

### Task 2: Consolidate rollout collection behind `RolloutCollectionRunner`

**Files:**
- Modify: `tests/unit_tests/test_cold_start_ray_collect_runner.py`
- Modify: `tests/unit_tests/test_collect_rollouts_runner.py`
- Modify: `dreamervla/runners/rollout_collection_runner.py`
- Modify: `configs/experiment/collect_rollouts_ray.yaml`
- Modify: `configs/experiment/collect_rollouts_onetraj.yaml`

- [ ] **Step 1: Add backend-selection tests**

```python
@pytest.mark.parametrize("backend", ["ray", "vectorized"])
def test_rollout_collection_runner_selects_configured_backend(backend: str) -> None:
    cfg = _collection_cfg(backend=backend)
    runner = RolloutCollectionRunner(cfg)
    assert runner.collection_backend == backend


def test_rollout_collection_runner_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="collect.backend"):
        RolloutCollectionRunner(_collection_cfg(backend="legacy"))
```

- [ ] **Step 2: Run both collection test modules**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cold_start_ray_collect_runner.py tests/unit_tests/test_collect_rollouts_runner.py`

Expected: FAIL because no unified runner/backend contract exists.

- [ ] **Step 3: Move the two existing collection execution paths into private backend methods**

`RolloutCollectionRunner.execute()` dispatches only on `collect.backend`:

```python
def execute(self) -> dict[str, float | int]:
    if self.collection_backend == "ray":
        return self._run_ray_collection()
    if self.collection_backend == "vectorized":
        return self._run_vectorized_collection()
    raise AssertionError(self.collection_backend)
```

Both paths write the same reward/hidden shards and `collection_manifest.json`.

- [ ] **Step 4: Point both collection experiments at the canonical target**

Set `_target_: dreamervla.runners.RolloutCollectionRunner` and select
`collect.backend=ray|vectorized` in the two experiment configs.

- [ ] **Step 5: Run collection and config-composition tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cold_start_ray_collect_runner.py tests/unit_tests/test_collect_rollouts_runner.py tests/unit_tests/test_config_composition.py`

Expected: PASS.

### Task 3: Extract independent World Model training

**Files:**
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py`
- Create: `tests/unit_tests/test_world_model_training_runner.py`
- Modify: `dreamervla/runners/world_model_training_runner.py`
- Modify: `configs/experiment/wm_full_dataset_train.yaml`
- Modify: `configs/experiment/wm_official_upper_bound.yaml`

- [ ] **Step 1: Add a World-Model-only state-change test**

```python
def test_world_model_training_updates_only_world_model(tmp_path: Path) -> None:
    runner = WorldModelTrainingRunner(_world_model_cfg(tmp_path, steps=2))
    before = runner.component_hashes()
    metrics = runner.run()
    after = runner.component_hashes()
    assert before["world_model"] != after["world_model"]
    assert set(after) == {"world_model"}
    assert metrics["train/world_model_optimizer_steps"] == 2
```

- [ ] **Step 2: Run the new test**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_world_model_training_runner.py`

Expected: FAIL because the canonical runner is not implemented.

- [ ] **Step 3: Extract offline replay seeding, WM update, top-k checkpoint, and resume logic**

Move only the World Model warmup responsibilities from the old pipeline runner.
`WorldModelTrainingRunner` must construct no VLA actor, classifier optimizer, real
environment, or online rollout loop. Its checkpoint payload contains `world_model`,
`world_model_optimizer`, `global_step`, and resolved component metadata.

- [ ] **Step 4: Update WM experiments to use the canonical target**

Set `_target_: dreamervla.runners.WorldModelTrainingRunner`; retain the existing
official/collected dataset overrides and checkpoint cadence.

- [ ] **Step 5: Run World Model and resume tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_world_model_training_runner.py tests/unit_tests/test_cotrain_resume.py -k world_model`

Expected: PASS.

### Task 4: Rename and isolate Success Classifier training

**Files:**
- Modify: `tests/unit_tests/test_latent_classifier_runner.py`
- Modify: `dreamervla/runners/success_classifier_training_runner.py`
- Modify: `configs/experiment/latent_classifier_openvla_onetraj_libero_goal_h1.yaml`
- Modify: `configs/experiment/wmpo_token_classifier_openvla_onetraj_libero_goal_h1.yaml`
- Modify: `configs/experiment/classifier_official_upper_bound.yaml`

- [ ] **Step 1: Change classifier tests to the complete public name**

```python
from dreamervla.runners import SuccessClassifierTrainingRunner


def test_success_classifier_training_checkpoint_contains_calibration(tmp_path: Path) -> None:
    runner = SuccessClassifierTrainingRunner(_classifier_cfg(tmp_path))
    runner.run()
    payload = torch.load(runner.latest_checkpoint, map_location="cpu")
    assert "classifier" in payload
    assert "classifier_optimizer" in payload
    assert 0.0 <= float(payload["classifier_threshold"]) <= 1.0
```

- [ ] **Step 2: Run classifier tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_latent_classifier_runner.py`

Expected: FAIL on the removed `LatentClassifierRunner` name.

- [ ] **Step 3: Move the existing classifier implementation under the complete name**

Retain dataset splitting, balancing, F1 selection, threshold calibration, checkpoint,
and resume behavior. Remove latent-classifier compatibility exports.

- [ ] **Step 4: Update all classifier experiment targets**

Set `_target_: dreamervla.runners.SuccessClassifierTrainingRunner` in the three
active classifier experiments.

- [ ] **Step 5: Run classifier tests and composition checks**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_latent_classifier_runner.py tests/unit_tests/test_config_composition.py -k classifier`

Expected: PASS.

### Task 5: Establish the canonical `CotrainRunner` and delete frozen behavior

**Files:**
- Modify: `tests/unit_tests/test_manual_cotrain_ray_runner.py`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `configs/dreamervla/wmcls_cotrain_ray.yaml`
- Modify: `configs/experiment/dreamervla_wmcls_cotrain_ray.yaml`

- [ ] **Step 1: Rename staged causal tests and assert trainable groups**

```python
def test_cotrain_runner_builds_trainable_mainline_groups() -> None:
    runner = CotrainRunner(_cfg(learner_updates_enabled=True, real_env_enabled=True))
    assert runner._target_group_names() == [
        "LearnerGroup",
        "ActorGroup",
        "RolloutGroup",
        "EnvironmentGroup",
    ]
    assert runner._staged_policy_update_enabled() is True
```

- [ ] **Step 2: Run staged cotrain tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_manual_cotrain_ray_runner.py`

Expected: FAIL until imports and canonical group naming are updated.

- [ ] **Step 3: Preserve only the current staged trainable transaction**

Move `ManualCotrainRayRunner` to `CotrainRunner`, retain the real-SFT -> re-encode ->
WM/CLS -> imagine -> PPO order, and delete branches guarded by
`not learner_updates_enabled`, frozen hash auditing, frozen summaries, frozen source
checkpoint loading, and policy-only RL finalization.

- [ ] **Step 4: Require both trainable components at cotrain entry**

`CotrainRunner` accepts independent WM and classifier component checkpoints or one
consolidated cotrain checkpoint. Missing components are a launcher routing error,
not random initialization inside cotrain.

- [ ] **Step 5: Run cotrain unit tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_manual_cotrain_ray_runner.py tests/unit_tests/test_manual_resource_config_groups.py`

Expected: PASS.

### Task 6: Make Ray models resident and execute phases serially

**Files:**
- Modify: `dreamervla/workers/cotrain/placement.py`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Create: `tests/unit_tests/test_cotrain_resident_placement.py`

- [ ] **Step 1: Add the fixed eight-GPU placement test**

```python
def test_eight_gpu_cotrain_placement_is_resident_and_explicit() -> None:
    plan = build_cotrain_placement(8, real_env_workers=1)
    assert [x.gpu_ids for x in plan.actor_specs] == [[i] for i in range(8)]
    assert [x.gpu_ids for x in plan.rollout_specs] == [[i] for i in range(8)]
    assert plan.learner_spec.gpu_ids == [0]
    assert [x.gpu_ids for x in plan.wm_env_specs] == [[i] for i in range(1, 8)]
    assert plan.real_env_spec.gpu_ids == []
    assert plan.evaluation_env_spec.gpu_ids == []
```

- [ ] **Step 2: Run the placement test**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_resident_placement.py`

Expected: FAIL because real/eval CPU roles are absent and naming is manual-cotrain.

- [ ] **Step 3: Define `CotrainPlacementPlan` with all resident roles**

Actor and Rollout occupy GPUs 0-7; Learner occupies GPU 0; WorldModelEnv occupies
GPUs 1-7; RealEnvironment, EvaluationEnvironment, Replay, and driver are CPU roles.
Remove alternate component-placement/frozen topologies from the mainline builder.

- [ ] **Step 4: Add explicit phase barriers without worker destruction**

The global-step method calls and waits in this order:

```python
real = self._collect_real_trajectories(groups, global_step)
sft = self._train_vla_on_real(groups, real, global_step)
learner = self._train_world_model_classifier(groups, real, global_step)
imagined = self._collect_imagined_trajectories(groups, global_step)
ppo = self._train_vla_ppo(groups, imagined, global_step)
evaluation = self._maybe_evaluate_resident_policy(groups, global_step)
```

Each method returns only after its active group completes. Group creation occurs once
before the global-step loop; teardown occurs once after it.

- [ ] **Step 5: Run placement and causal-order tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_resident_placement.py tests/unit_tests/test_manual_cotrain_ray_runner.py -k 'placement or staged or order'`

Expected: PASS.

### Task 7: Add resident read-only evaluation

**Files:**
- Create: `dreamervla/workers/env/evaluation_env_worker.py`
- Modify: `dreamervla/workers/env/__init__.py`
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `dreamervla/runners/libero_chunk_eval.py`
- Create: `tests/unit_tests/test_cotrain_resident_eval.py`

- [ ] **Step 1: Write a no-reload/no-write evaluation test**

```python
def test_resident_eval_reuses_rollout_group_and_is_read_only() -> None:
    groups = _fake_resident_groups()
    runner = CotrainRunner(_cfg(eval_interval_global_steps=1))
    metrics = runner._evaluate_resident_policy(groups, global_step=1)
    assert groups.rollout.load_calls == 0
    assert groups.rollout.sync_calls == 1
    assert groups.replay.write_calls == 0
    assert groups.actor.optimizer_steps == 0
    assert metrics["eval/episodes"] == 100
```

- [ ] **Step 2: Run the resident-eval test**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_resident_eval.py`

Expected: FAIL because evaluation currently runs in a checkpoint-loading subprocess.

- [ ] **Step 3: Implement CPU `EvaluationEnvironmentWorker`**

It owns the fixed LIBERO vector environments and a dedicated evaluation channel. It
sends observations to the existing `RolloutGroup`, records first-result-per-reset-id
success, and returns task records without replay or training side effects.

- [ ] **Step 4: Invoke resident evaluation at the cotrain barrier**

Evaluate step 0 when configured, then accepted steps divisible by the interval.
Synchronize Actor -> Rollout immediately before evaluation. Do not stop Ray, save/load
a VLA checkpoint, or create another policy object.

- [ ] **Step 5: Run resident and standalone evaluation tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_resident_eval.py tests/unit_tests/test_libero_chunk_eval.py tests/unit_tests/test_libero_eval_protocol_compat.py`

Expected: PASS.

### Task 8: Split every cotrain phase into an independent progress stream

**Files:**
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `dreamervla/workers/env/trajectory_env_worker.py`
- Modify: `dreamervla/workers/env/evaluation_env_worker.py`
- Modify: `dreamervla/workers/learner/learner_worker.py`
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py`
- Create: `tests/unit_tests/test_cotrain_phase_progress.py`

- [ ] **Step 1: Add monotonic independent-progress tests**

```python
def test_cotrain_progress_streams_are_independent() -> None:
    progress = _capture_progress()
    _run_fake_global_step(progress)
    assert list(dict.fromkeys(progress.names)) == [
        "cotrain-real-rollout/00000001",
        "cotrain-vla-real-sft/00000001",
        "cotrain-wmcls-training/00000001",
        "cotrain-imagined-rollout/00000001",
        "cotrain-vla-ppo/00000001",
        "eval/00000001",
        "cotrain",
    ]
    for records in progress.by_name.values():
        assert [x.done for x in records] == sorted(x.done for x in records)
        assert records[-1].done == records[-1].total
```

- [ ] **Step 2: Run the progress test**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_phase_progress.py`

Expected: FAIL because real/imagined share one env total and learner/eval lack progress.

- [ ] **Step 3: Filter rollout progress by phase and role**

Real progress reports real trajectory target, chunks, and successes. Imagined progress
reports imagined trajectory target, chunks, and classifier-positive rate. Do not sum
their totals.

- [ ] **Step 4: Instrument VLA SFT, WM/CLS, PPO, and eval**

Use `BaseRunner.console_progress` on rank 0 at the configured interval. The WM/CLS
status contains learner update/target, WM loss, classifier loss/F1/threshold, and early
stop. Actor SFT and PPO use their own operation totals and KL/loss fields. Eval uses
episode target, successes, rate, and chunk throughput.

- [ ] **Step 5: Run progress and worker tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_phase_progress.py tests/unit_tests/test_manual_cotrain_ray_runner.py tests/unit_tests/test_online_cotrain_ray_runner.py -k progress`

Expected: PASS for retained tests; tests tied only to the removed runner are deleted in Task 11.

### Task 9: Enforce debug limits in the actual driver loop

**Files:**
- Modify: `dreamervla/runners/cotrain_runner.py`
- Modify: `dreamervla/launchers/cotrain.py`
- Modify: `configs/dreamervla/wmcls_cotrain_ray.yaml`
- Modify: `tests/unit_tests/test_cotrain_debug_limits.py`

- [ ] **Step 1: Add actual-loop budget tests**

```python
def test_debug_driver_accepts_exactly_ten_steps() -> None:
    runner, groups = _runner_with_fake_groups(debug=True, configured_steps=20_000)
    history = runner.run()
    assert groups.accepted_global_steps == list(range(1, 11))
    assert groups.real_targets == [8] * 10
    assert groups.imagined_targets == [256] * 10
    assert groups.saved_steps == list(range(1, 11))
    assert groups.evaluated_steps == list(range(1, 11))
    assert history["global_step"] == 10
```

- [ ] **Step 2: Run the debug budget test**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_debug_limits.py`

Expected: FAIL if launcher values override the runner cap or evaluation segmentation
changes the absolute target.

- [ ] **Step 3: Centralize debug resolution before group launch**

When `training.debug=true`, set exactly: global steps 10, checkpoint interval 1,
evaluation interval 1, real trajectories 8, imagined trajectories 256. Do not change
rollout horizon, learner updates, batch sizes, environment counts, or other production
values. The driver loop reads only the resolved values.

- [ ] **Step 4: Remove launcher-side debug target duplication**

The launcher passes `training.debug`; it does not create segmented absolute targets or
append a later production `manual_cotrain.global_steps` override.

- [ ] **Step 5: Run debug and launcher tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_debug_limits.py tests/unit_tests/test_frozen_model_cotrain_ray_launcher.py -k 'debug or wmcls'`

Expected: PASS.

### Task 10: Route missing component checkpoints through independent training

**Files:**
- Create: `dreamervla/launchers/cotrain.py`
- Modify: `configs/scripts/coldstart_warmup_cotrain.yaml`
- Modify: `scripts/experiments/cotrain/train.sh`
- Create: `tests/unit_tests/test_cotrain_launcher.py`

- [ ] **Step 1: Add all four checkpoint-routing cases**

```python
@pytest.mark.parametrize(
    ("has_wm", "has_classifier", "expected"),
    [
        (False, False, ["collect", "world_model", "classifier", "cotrain"]),
        (True, False, ["collect", "classifier", "cotrain"]),
        (False, True, ["collect", "world_model", "cotrain"]),
        (True, True, ["cotrain"]),
    ],
)
def test_cotrain_launcher_routes_missing_components(has_wm, has_classifier, expected):
    plan = build_cotrain_plan(_launcher_cfg(has_wm, has_classifier, valid_data=False))
    assert [stage.name for stage in plan.stages] == expected
```

- [ ] **Step 2: Run launcher tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_launcher.py`

Expected: FAIL because the current launcher requires a pair or random initialization.

- [ ] **Step 3: Implement manifest-aware pipeline routing**

Resolve optional `WORLD_MODEL_CKPT` and `CLASSIFIER_CKPT` independently. Add collection
only when a required component is missing and no valid collected dataset exists. Pass
the resulting two component checkpoint paths to `CotrainRunner`; never freeze them.

- [ ] **Step 4: Make the shell entrypoint one command**

`scripts/experiments/cotrain/train.sh` calls `python -m dreamervla.launchers.cotrain`
with Hydra overrides. It contains no pinned checkpoint path, loops, or custom parser.

- [ ] **Step 5: Run routing and script tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_launcher.py tests/unit_tests/test_experiment_stage_scripts.py tests/unit_tests/test_setup_scripts.py`

Expected: PASS.

### Task 11: Consolidate standalone evaluation and remove legacy runners

**Files:**
- Modify: `dreamervla/runners/libero_vla_evaluation_runner.py`
- Modify: `configs/evaluation/libero_vla.yaml`
- Modify: `configs/experiment/eval_libero_vla.yaml`
- Modify: `scripts/experiments/cotrain/eval.sh`
- Delete: `dreamervla/runners/cold_start_ray_collect_runner.py`
- Delete: `dreamervla/runners/collect_rollouts_runner.py`
- Delete: `dreamervla/runners/dreamervla_runner.py`
- Delete: `dreamervla/runners/embodied_eval_runner.py`
- Delete: `dreamervla/runners/frozen_model_policy_runner.py`
- Delete: `dreamervla/runners/latent_classifier_runner.py`
- Delete: `dreamervla/runners/manual_cotrain_ray_runner.py`
- Delete: `dreamervla/runners/online_cotrain_pipeline_runner.py`
- Delete: `dreamervla/runners/online_cotrain_ray_runner.py`
- Delete: `dreamervla/runners/online_cotrain_runner.py`
- Delete: `dreamervla/runners/pretokenize_vla_runner.py`

- [ ] **Step 1: Move retained standalone evaluation behavior under the canonical runner**

Preserve strict full-policy restore, native cotrain checkpoint loading, Hugging Face
base checkpoint loading/export, OpenVLA-OFT action postprocessing, and `rlinf_chunk`
evaluation. Update tests to import `LIBEROVLAEvaluationRunner` only.

- [ ] **Step 2: Delete legacy runner implementations and their experiment-only tests**

Remove tests that prove frozen or alternate cotrain behavior. Move still-valid helper
tests to the canonical runner test modules before deleting their source imports.

- [ ] **Step 3: Verify there are no legacy references**

Run:

```bash
rg -n "ManualCotrain|FrozenModelPolicy|OnlineCotrain|DreamerVLARunner|LatentClassifierRunner|EmbodiedEvalRunner|PretokenizeVLARunner" \
  dreamervla configs scripts tests docs spec README.md README.zh-CN.md
```

Expected: no active code/config/script/test references; historical design documents
may be removed or explicitly marked superseded rather than used as commands.

- [ ] **Step 4: Run public, evaluation, and repository-hygiene tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_runner_public_api.py tests/unit_tests/test_libero_eval_protocol_compat.py tests/unit_tests/test_repository_hygiene.py`

Expected: PASS.

### Task 12: Update documentation and run CPU verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `spec/00_overview.md`
- Modify: `spec/04_complete_loop.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `docs/repository_structure.md`
- Modify: `configs/README.md`
- Modify: `scripts/README.md`

- [ ] **Step 1: Replace command and runner inventories with the five canonical names**

Document the optional-component routing matrix, the resident eight-GPU map, serial
cotrain transaction, resident periodic evaluation, progress stream names, and the
single `scripts/experiments/cotrain/train.sh` entrypoint.

- [ ] **Step 2: Run formatting and static checks**

Run:

```bash
ruff check dreamervla tests
ruff format --check dreamervla tests
python -m compileall -q dreamervla
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Run the complete unit suite**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests`

Expected: PASS with no collection errors or references to deleted runner modules.

### Task 13: Run the real local eight-GPU debug acceptance

**Files:**
- Runtime output only: `${DVLA_DATA_ROOT}/outputs/cotrain/<timestamp>/`

- [ ] **Step 1: Confirm eight GPUs are visible and idle enough for the test**

Run: `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader`

Expected: GPU indices 0 through 7 are present; no unrelated process consumes enough
memory to prevent resident Actor/Rollout/Learner/WorldModelEnv placement.

- [ ] **Step 2: Launch the actual debug route**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/experiments/cotrain/train.sh \
  ++training.debug=true
```

Expected: all resident Ray groups launch once; phase-specific progress appears; the
process exits successfully after accepted global step 10.

- [ ] **Step 3: Verify runtime artifacts and actual budgets**

Check the resolved config, metrics, checkpoint directories, and evaluation summary.
There must be exactly accepted steps 1-10, real target 8 and imagined target 256 on
each step, checkpoints on steps 1-10, and resident eval records on steps 1-10. The log
must contain all seven progress names and no second VLA checkpoint load during periodic
evaluation.

- [ ] **Step 4: Re-run focused regression tests after the GPU job**

Run: `PYTHONPATH="$PWD" pytest -q tests/unit_tests/test_cotrain_debug_limits.py tests/unit_tests/test_cotrain_resident_eval.py tests/unit_tests/test_cotrain_phase_progress.py`

Expected: PASS.

- [ ] **Step 5: Commit the verified implementation when all overlapping user changes are explicitly in scope**

```bash
git add AGENTS.md README.md README.zh-CN.md configs dreamervla scripts spec docs tests
git commit -s -m "refactor: converge on the mainline cotrain pipeline"
```
