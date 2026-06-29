# Manual Cotrain Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the manual-notes cotrain route start successfully for 0-5 GPUs and complete one short global-step run.

**Architecture:** Keep the existing four-group target route: `LearnerGroup`, `ActorGroup`, `RolloutGroup`, and `EnvGroup`. Preserve compatibility with the current runner, worker, and Hydra patterns while aligning names and startup behavior with `spec/99_manual_notes.md`.

**Tech Stack:** Python 3.11, Hydra/OmegaConf, Ray, DreamerVLA `WorkerGroup`/`Channel`, PyTorch, existing tiny test models/envs, pytest.

---

## Source Of Truth

- Primary architecture source: `spec/99_manual_notes.md`.
- Supporting architecture docs: `spec/00_overview.md`, `spec/04_complete_loop.md`, and `spec/02_naming.md`.
- Do not use migrated `docs/` architecture copies as source-of-truth.
- `99_manual_notes.md` may only be edited for repository naming alignment. Do not change its design logic or flow.

## Current Difference Review

1. The target route exists as untracked/manual-route files, but its naming still contains a legacy term that the user rejected. That term appears in runner class names, helper names, config keys, experiment names, tests, launcher helpers, archived docs, and a stale plan.
2. The route can start a tiny CPU/0-GPU run, but the user requirement is explicit 0-5 GPU startup support with the final neutral names.
3. `BaseTrajectoryEnvWorker` currently builds one env object per slot. `99_manual_notes.md` describes one worker-rank env object that batch-manages multiple parallel slot states.
4. `LatentWorldModelEnv` currently holds one latent state and runs model inference with batch size 1. For `WMEnvWorker`, target behavior is one world model plus one classifier/reward model managing a batch of slot states.
5. The tiny startup config currently defaults into online logging unless overridden. Short-flow verification should be deterministic and local/offline.
6. `configs/scripts/coldstart_warmup_cotrain.yaml` and `dreamervla/launchers/coldstart_warmup_cotrain.py` still emit legacy experiment names and override keys for async cotrain.

## File Map

- Modify: `dreamervla/workers/cotrain/placement.py`
  - Rename placement dataclass/function to neutral manual-cotrain names.
- Modify: `dreamervla/workers/cotrain/__init__.py`
  - Export neutral placement names only.
- Modify: `dreamervla/runners/manual_cotrain_ray_runner.py`
  - Rename class, `runner_name`, config reads, channel prefixes, Ray actor names, and status banner.
- Modify: `dreamervla/runners/__init__.py`
  - Export `ManualCotrainRayRunner` and remove the legacy runner export.
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`
  - Emit `manual_cotrain.*` overrides for the manual route.
- Modify: `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`
  - Rename config group from legacy key to `manual_cotrain`.
- Modify: `configs/experiment/manual_cotrain_ray_oft_backbone_latent.yaml`
  - Compose the neutral dreamervla config.
- Modify: `configs/experiment/manual_cotrain_ray_tiny.yaml`
  - Tiny 0-5 startup config using neutral names and local/offline logging.
- Modify: `configs/scripts/coldstart_warmup_cotrain.yaml`
  - Set `cotrain_async_experiment: manual_cotrain_ray_oft_backbone_latent`.
- Modify: `dreamervla/envs/world_model/latent_world_model_env.py`
  - Add batched slot state support while keeping single-slot API compatible.
- Modify: `dreamervla/workers/env/trajectory_env_worker.py`
  - Prefer one batched env object when supported; keep per-slot fallback for existing envs.
- Modify: `dreamervla/workers/rollout/multistep_rollout_worker.py`
  - Remove rejected route naming from comments/docstrings.
- Modify: `dreamervla/workers/actor/embodied_fsdp_actor.py`
  - Remove rejected route naming from comments/docstrings.
- Modify: `tests/unit_tests/test_manual_cotrain_placement.py`
- Modify: `tests/unit_tests/test_manual_cotrain_ray_runner.py`
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
- Modify: `tests/unit_tests/test_trajectory_env_worker.py`
- Modify: `tests/unit_tests/test_latent_world_model_env.py`
- Modify: `spec/99_manual_notes.md`
  - Naming-only alignment if any stale repository name appears.
- Modify/delete stale plan/archive files only to remove rejected naming; do not use them as architecture sources.

---

### Task 1: Neutral Naming Purge

**Files:**
- Modify: `dreamervla/workers/cotrain/placement.py`
- Modify: `dreamervla/workers/cotrain/__init__.py`
- Modify: `dreamervla/runners/manual_cotrain_ray_runner.py`
- Modify: `dreamervla/runners/__init__.py`
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`
- Modify: `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`
- Modify: `configs/experiment/manual_cotrain_ray_oft_backbone_latent.yaml`
- Modify: `configs/experiment/manual_cotrain_ray_tiny.yaml`
- Modify: `configs/scripts/coldstart_warmup_cotrain.yaml`
- Modify: `tests/unit_tests/test_manual_cotrain_placement.py`
- Modify: `tests/unit_tests/test_manual_cotrain_ray_runner.py`
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`
- Modify/delete stale plan/archive files containing rejected naming.

- [ ] **Step 1: Replace public route names**

Rename symbols and strings:

```text
ManualCotrainPlacementPlan
build_manual_cotrain_placement
ManualCotrainRayRunner
manual_cotrain_ray
manual_cotrain
manual-cotrain
ManualReplay
ManualRealEnvWorker
ManualWMEnvWorker
ManualRolloutWorker
ManualActor
ManualLearner
```

Implementation requirements:

- `dreamervla.runners.ManualCotrainRayRunner` must resolve through the existing lazy runner registry.
- The Hydra experiment must be `experiment=manual_cotrain_ray_oft_backbone_latent`.
- The tiny startup experiment must be `experiment=manual_cotrain_ray_tiny`.
- Runtime override keys must use `manual_cotrain.ngpu`, `manual_cotrain.envs_per_worker`, `manual_cotrain.global_steps`, `manual_cotrain.rollout_epoch`, `manual_cotrain.max_steps_per_rollout_epoch`, `manual_cotrain.num_action_chunks`, `manual_cotrain.learner_update_step`, `manual_cotrain.sync_every`, and `manual_cotrain.task_id`.
- Do not keep compatibility aliases with the rejected name.

- [ ] **Step 2: Update launcher tests**

Update assertions in `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`:

```python
assert "experiment=manual_cotrain_ray_oft_backbone_latent" in plan.cotrain_online_cmd
assert "manual_cotrain.ngpu=2" in plan.cotrain_online_cmd
assert "cluster.component_placement=null" in plan.cotrain_online_cmd
```

Rename test functions so they use `manual_cotrain` in their names.

- [ ] **Step 3: Run naming-focused tests**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
  tests/unit_tests/test_manual_cotrain_placement.py \
  tests/unit_tests/test_manual_cotrain_ray_runner.py \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_async_cotrain_engine_splits_warmup_and_ray_online \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_async_cotrain_online_command_targets_manual_cotrain_runner \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Verify rejected token is gone**

Run a case-insensitive repository grep for the rejected legacy token supplied by the user.

Expected: no matches in `dreamervla/`, `configs/`, `tests/`, `spec/`, or active `docs/` files. Archived files may be rewritten or removed if they still match.

---

### Task 2: Batched Latent World-Model Env

**Files:**
- Modify: `dreamervla/envs/world_model/latent_world_model_env.py`
- Modify: `tests/unit_tests/test_latent_world_model_env.py`

- [ ] **Step 1: Add failing batched-state test**

Add a test that constructs one `LatentWorldModelEnv(num_envs=3)` and verifies one model/classifier pair manages three independent slots:

```python
def test_latent_world_model_env_batches_independent_slots() -> None:
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=10.0,
        num_envs=3,
    )

    obs0, _ = env.reset_slot(0, task_id=0, episode_id=0)
    obs1, _ = env.reset_slot(1, task_id=1, episode_id=11)
    obs2, _ = env.reset_slot(2, task_id=2, episode_id=22)

    assert obs0["latent"].shape == (2,)
    assert obs1["episode_id"] == 11
    assert obs2["task_id"] == 2

    next_obs0, reward0, done0, truncated0, info0 = env.step_slot(
        0, np.array([1.0, 0.0], dtype=np.float32)
    )
    next_obs1, reward1, done1, truncated1, info1 = env.step_slot(
        1, np.array([0.0, 2.0], dtype=np.float32)
    )

    assert next_obs0["latent"].tolist() == [1.0, 0.0]
    assert next_obs1["latent"].tolist() == [0.0, 2.0]
    assert reward0 == 1.0
    assert reward1 == 2.0
    assert done0 is False
    assert done1 is False
    assert truncated0 is False
    assert truncated1 is False
    assert info0["slot_id"] == 0
    assert info1["slot_id"] == 1
```

- [ ] **Step 2: Implement batched slot API**

Add constructor argument:

```python
num_envs: int = 1
```

Maintain:

```python
reset(task_id=..., episode_id=...)
step(action)
chunk_step(action_chunk)
make_transition(...)
```

Add:

```python
reset_slot(slot_id: int, *, task_id: int = 0, episode_id: int = 0)
step_slot(slot_id: int, action: Any)
reset_batch(task_ids: Sequence[int], episode_ids: Sequence[int])
step_batch(actions: Any, env_ids: Sequence[int] | None = None)
```

Implementation details:

- Store latent as `[num_envs, latent_dim]`.
- Store elapsed steps, task ids, and episode ids as per-slot arrays.
- `reset()` and `step()` delegate to slot 0 for backward compatibility.
- `step_batch()` runs one world-model forward over the addressed slots.
- `step_slot()` may delegate to `step_batch()` with one slot.
- `load_world_model_state()` and `load_classifier_state()` still load one model/classifier pair.

- [ ] **Step 3: Run tests**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_latent_world_model_env.py -q
```

Expected: all tests pass.

---

### Task 3: One Env Object Per Worker Rank When Supported

**Files:**
- Modify: `dreamervla/workers/env/trajectory_env_worker.py`
- Modify: `dreamervla/workers/env/_test_envs.py`
- Modify: `tests/unit_tests/test_trajectory_env_worker.py`

- [ ] **Step 1: Add batched-env worker test**

Add a test env class or extend `_test_envs.py` with a `BatchedCounterEnv` exposing:

```python
reset_slot(slot_id: int, *, task_id: int = 0, episode_id: int = 0)
step_slot(slot_id: int, action: Any)
make_transition(...)
close()
```

Add test:

```python
def test_trajectory_env_worker_uses_single_batched_env_when_available() -> None:
    worker = WMEnvWorker(
        env_cfg={
            "target": "dreamervla.workers.env._test_envs:BatchedCounterEnv",
            "kwargs": {"num_envs": 3, "horizon": 2, "embedding_dim": 4},
        },
        num_slots=3,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=2,
        task_id=0,
    )
    try:
        worker.init()
        assert len(worker.envs) == 1
        messages = worker.bootstrap_obs()
        assert [msg.slot_id for msg in messages] == [0, 1, 2]
    finally:
        worker.close()
```

- [ ] **Step 2: Implement worker batched mode**

In `BaseTrajectoryEnvWorker.init()`:

- Build one env first.
- If it supports `reset_slot` and `step_slot`, store `self.envs = [env]` and mark batched mode.
- Otherwise preserve existing per-slot fallback: build one env per slot.

In `_reset_slot()`:

- Use `env.reset_slot(slot_id, task_id=..., episode_id=...)` in batched mode.
- Use existing per-slot env reset in fallback mode.

In `_step_slot()`:

- Use `env.step_slot(slot_id, action)` in batched mode.
- Use existing per-slot env step in fallback mode.

In `_ensure_initialized()`:

- Accept either one batched env or `num_slots` fallback envs.

In `load_world_model_state()` and `load_classifier_state()`:

- Load once per env object. In batched mode this means one WM/classifier pair per worker rank.

- [ ] **Step 3: Run tests**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_trajectory_env_worker.py -q
```

Expected: all tests pass.

---

### Task 4: Manual Cotrain Runner Startup Compatibility

**Files:**
- Modify: `dreamervla/runners/manual_cotrain_ray_runner.py`
- Modify: `configs/experiment/manual_cotrain_ray_tiny.yaml`
- Modify: `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`
- Modify: `tests/unit_tests/test_manual_cotrain_ray_runner.py`

- [ ] **Step 1: Add config/read tests for neutral key**

Update helper config in `tests/unit_tests/test_manual_cotrain_ray_runner.py` to use:

```python
"manual_cotrain": {
    "ngpu": ngpu,
    "global_steps": 1,
    "learner_update_step": 1,
    "rollout_epoch": 1,
    "max_steps_per_rollout_epoch": 2,
    "num_action_chunks": 2,
    "envs_per_worker": 1,
    "sync_every": 1,
}
```

Assert `_target_ == "dreamervla.runners.ManualCotrainRayRunner"` for the composed OFT experiment.

- [ ] **Step 2: Update runner config selectors**

All private accessors must read from `manual_cotrain.*`.

Required accessors:

```python
_ngpu()
_global_steps()
_sync_every()
_learner_update_step()
_rollout_epoch()
_max_steps_per_rollout_epoch()
_num_action_chunks()
_envs_per_worker()
_task_id()
```

- [ ] **Step 3: Make tiny route deterministic and local**

In `configs/experiment/manual_cotrain_ray_tiny.yaml`:

- Set `_target_: dreamervla.runners.ManualCotrainRayRunner`.
- Set `runner.logger.logger_backends: []`.
- Set `manual_cotrain.ngpu: 0`.
- Use `manual_cotrain.envs_per_worker: 2`.
- Use `manual_cotrain.learner_update_step: 999999` so the startup smoke does not depend on replay contents for WM/classifier update.
- Configure WM env with `num_envs: ${manual_cotrain.envs_per_worker}`.

In `configs/dreamervla/manual_cotrain_ray_oft_backbone_latent.yaml`:

- Set `_target_: dreamervla.runners.ManualCotrainRayRunner`.
- Use `manual_cotrain` config group.
- Configure WM env with `num_envs: ${manual_cotrain.envs_per_worker}`.
- Keep `device: cpu` unless an explicit override sets otherwise; the startup goal is placement/topology, not GPU-heavy WM training.

- [ ] **Step 4: Run runner tests**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest tests/unit_tests/test_manual_cotrain_ray_runner.py -q
```

Expected: all tests pass.

---

### Task 5: Launcher Async Route Alignment

**Files:**
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py`
- Modify: `configs/scripts/coldstart_warmup_cotrain.yaml`
- Modify: `tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py`

- [ ] **Step 1: Rename helper**

Rename `_manual_cotrain_online_overrides` to:

```python
def _manual_cotrain_online_overrides(...)
```

It must emit:

```text
manual_cotrain.ngpu=<requested count>
manual_cotrain.envs_per_worker=<profile online_rollout_envs_per_gpu when configured>
```

For `ngpu=0`, keep CPU-safe overrides:

```text
actor.train_cfg.fsdp.strategy=none
actor.train_cfg.device=cpu
learner.train_cfg.device=cpu
rollout.train_cfg.device=cpu
```

- [ ] **Step 2: Detect manual route by neutral experiment prefix**

In async plan construction, apply manual-cotrain overrides when:

```python
async_exp.startswith("manual_cotrain_")
```

- [ ] **Step 3: Update default async experiment**

In `configs/scripts/coldstart_warmup_cotrain.yaml`:

```yaml
cotrain_async_experiment: manual_cotrain_ray_oft_backbone_latent
```

- [ ] **Step 4: Run launcher tests**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_async_cotrain_engine_splits_warmup_and_ray_online \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_async_cotrain_online_command_targets_manual_cotrain_runner \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py::test_ngpu_zero_does_not_emit_torchrun_or_gpu_ray_placement \
  -q
```

Expected: all selected tests pass.

---

### Task 6: Naming-Only Spec Alignment

**Files:**
- Modify: `spec/99_manual_notes.md` only if repository names need alignment.
- Modify: stale plan/archive files under `spec/` or `docs/` only to remove rejected naming.

- [ ] **Step 1: Check `99_manual_notes.md`**

Run:

```bash
rg -n -i "manual_cotrain|manual-cotrain|ManualCotrain|legacy" spec/99_manual_notes.md
```

Expected: no design changes needed. If a stale repository name appears, replace it with the neutral repository name only.

- [ ] **Step 2: Remove rejected naming from active docs**

Run the case-insensitive repository grep for the rejected legacy token supplied by the user.

Expected: no matches in `spec/`, `dreamervla/`, `configs/`, `tests/`, or active `docs/`.

- [ ] **Step 3: Preserve architecture logic**

After editing, verify `spec/99_manual_notes.md` still contains the same target flow:

```text
LearnerGroup -> world model/classifier
ActorGroup -> VLA PPO
RolloutGroup -> no-grad policy inference
EnvGroup -> RealEnvWorker / WMEnvWorker
GPU0 -> real env rank
GPU1.. -> WM env ranks
```

---

### Task 7: Unit Verification Set

**Files:**
- No implementation files unless failures reveal a defect.

- [ ] **Step 1: Run focused manual-cotrain tests**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
  tests/unit_tests/test_cotrain_messages.py \
  tests/unit_tests/test_manual_cotrain_placement.py \
  tests/unit_tests/test_multistep_rollout_worker.py \
  tests/unit_tests/test_trajectory_env_worker.py \
  tests/unit_tests/test_embodied_fsdp_actor.py \
  tests/unit_tests/test_manual_cotrain_ray_runner.py \
  tests/unit_tests/test_latent_world_model_env.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run launcher regression subset**

Run:

```bash
/home/user01/miniconda3/envs/dreamervla/bin/python -m pytest \
  tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py \
  -q
```

Expected: all tests in the launcher file pass.

---

### Task 8: 0-5 GPU Startup Verification

**Files:**
- No implementation files unless startup reveals a defect.

- [ ] **Step 1: Verify 0-GPU CPU startup**

Run:

```bash
WANDB_MODE=offline \
/home/user01/miniconda3/envs/dreamervla/bin/python -m dreamervla.train \
  experiment=manual_cotrain_ray_tiny \
  training.out_dir=/tmp/dvla_manual_cotrain_startup_gpu0 \
  hydra.run.dir=/tmp/dvla_manual_cotrain_startup_gpu0 \
  runner.logger.logger_backends=[] \
  manual_cotrain.ngpu=0 \
  cluster.num_gpus=0 \
  manual_cotrain.global_steps=1 \
  manual_cotrain.rollout_epoch=1 \
  manual_cotrain.max_steps_per_rollout_epoch=2 \
  manual_cotrain.envs_per_worker=2
```

Expected:

- Process exits 0.
- Logs include `groups=LearnerGroup,ActorGroup,RolloutGroup,EnvGroup`.
- Metrics include `env/steps`.

- [ ] **Step 2: Verify 1-5 GPU startup**

Run the same tiny experiment for `N=1,2,3,4,5` with matching visible devices.

Example for 5 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
WANDB_MODE=offline \
/home/user01/miniconda3/envs/dreamervla/bin/python -m dreamervla.train \
  experiment=manual_cotrain_ray_tiny \
  training.out_dir=/tmp/dvla_manual_cotrain_startup_gpu5 \
  hydra.run.dir=/tmp/dvla_manual_cotrain_startup_gpu5 \
  runner.logger.logger_backends=[] \
  manual_cotrain.ngpu=5 \
  cluster.num_gpus=5 \
  manual_cotrain.global_steps=1 \
  manual_cotrain.rollout_epoch=1 \
  manual_cotrain.max_steps_per_rollout_epoch=2 \
  manual_cotrain.envs_per_worker=2
```

Expected for each `N`:

- Process exits 0.
- Placement has one real env rank and `max(0, N - 1)` WM env ranks.
- The run completes one short global step.

- [ ] **Step 3: Final grep**

Run the case-insensitive repository grep for the rejected legacy token supplied by the user.

Expected: no matches.

---

## Completion Criteria

The task is complete only when:

- The manual cotrain route starts and completes one short global step.
- The startup route uses the four groups described in `spec/99_manual_notes.md`.
- Startup verification passes for 0, 1, 2, 3, 4, and 5 GPUs.
- The rejected legacy naming is removed from active code, configs, tests, and `spec/`.
- `99_manual_notes.md` has no design logic changes.
- Focused unit tests and launcher regression tests pass.
