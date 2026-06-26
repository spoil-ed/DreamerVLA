# Cotrain P1 Replay Ready and Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make full-run replay readiness depend on complete episodes, sampleable windows, and classifier positive/negative evidence while preserving three-pool sampling with latest-online priority.

**Architecture:** Extend `OnlineReplay.ready_for_training()` with explicit full-run gates and thread the same contract through DDP packing and Ray `ReplayWorker.ready()`. Keep imagined rollouts out of persistent replay; `OnlineReplay` remains real replay only, with `source` distinguishing `coldstart` and `online`.

**Tech Stack:** Python 3.11, PyTorch distributed tensor packing, pytest, Hydra/OmegaConf.

---

## File Structure

- Modify: `dreamervla/runners/online_replay.py:242`
  - Add `min_sampleable_windows` and `require_classifier_evidence` to ready checks and DDP helpers.
- Modify: `dreamervla/runners/online_cotrain_runner.py:735`
  - Read full-run ready knobs from `online_rollout` and pass them to global readiness.
- Modify: `dreamervla/workers/replay/replay_worker.py:79`
  - Route Ray readiness through `OnlineReplay.ready_for_training()`.
- Modify: `dreamervla/runners/online_cotrain_ray_runner.py:388`
  - Pass Ray readiness knobs into `ReplayWorker.ready()`.
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml:87`
  - Declare default full-run gates without code defaults deciding training behavior.
- Modify: `configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml:136`
  - Mirror Ray replay gates.
- Modify: `tests/unit_tests/test_online_replay_task_balanced.py`
  - Unit-test sampleable-window and classifier evidence readiness.
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py`
  - Unit-test no-Ray runner passes ready knobs.
- Modify: `tests/unit_tests/test_online_cotrain_ray_runner.py`
  - Unit-test Ray runner passes ready knobs.

## Task 1: Add Full-Run Gates to `OnlineReplay`

**Files:**
- Modify: `dreamervla/runners/online_replay.py:242`
- Modify: `tests/unit_tests/test_online_replay_task_balanced.py:311`

- [ ] **Step 1: Write the failing ready-gate tests**

Append to `tests/unit_tests/test_online_replay_task_balanced.py`:

```python
def test_online_replay_readiness_requires_sampleable_window_budget() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3)
    replay.add_episode(_episode(task_id=0, length=6, success=True))

    assert replay.sampleable_window_count() == 4
    assert replay.ready_for_training(
        min_transitions=3,
        task_ids=(0,),
        min_episodes_per_task=1,
        min_sampleable_windows=5,
    ) is False
    assert replay.ready_for_training(
        min_transitions=3,
        task_ids=(0,),
        min_episodes_per_task=1,
        min_sampleable_windows=4,
    ) is True


def test_ddp_replay_readiness_packs_sampleable_window_budget() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3)
    replay.add_episode(_episode(task_id=0, length=6, success=True))

    packed = pack_replay_task_stats_for_ddp(
        replay,
        task_ids=(0,),
        min_transitions=3,
        min_episodes_per_task=1,
        min_sampleable_windows=5,
    )
    _stats, coverage_ready, all_ranks_ready = unpack_replay_task_stats_from_ddp(
        packed,
        task_ids=(0,),
        world_size=1,
        min_transitions=3,
        min_episodes_per_task=1,
        min_sampleable_windows=5,
    )

    assert coverage_ready is False
    assert all_ranks_ready is False
```

- [ ] **Step 2: Run the focused replay tests**

Run:

```bash
pytest tests/unit_tests/test_online_replay_task_balanced.py::test_online_replay_readiness_requires_sampleable_window_budget tests/unit_tests/test_online_replay_task_balanced.py::test_ddp_replay_readiness_packs_sampleable_window_budget -q
```

Expected before implementation: FAIL with an unexpected keyword argument for `min_sampleable_windows`.

- [ ] **Step 3: Extend `ready_for_training()`**

Update `dreamervla/runners/online_replay.py`:

```python
    def ready_for_training(
        self,
        *,
        min_transitions: int,
        task_ids: tuple[int, ...],
        min_episodes_per_task: int,
        min_sampleable_windows: int = 0,
        require_classifier_evidence: bool = False,
    ) -> bool:
        if self.num_transitions < int(min_transitions):
            return False
        if int(min_sampleable_windows) > 0:
            if self.sampleable_window_count() < int(min_sampleable_windows):
                return False
        min_eps = int(min_episodes_per_task)
        if min_eps <= 0:
            ready = bool(self._valid_records())
        else:
            counts = self.task_episode_counts()
            ready = all(counts[int(task_id)] >= min_eps for task_id in task_ids)
        if not ready:
            return False
        if require_classifier_evidence and not self.classifier_ready(task_ids=task_ids):
            return False
        return True
```

- [ ] **Step 4: Thread the gates through DDP helpers**

Update signatures in `dreamervla/runners/online_replay.py`:

```python
def pack_replay_task_stats_for_ddp(
    replay: OnlineReplay,
    *,
    task_ids: tuple[int, ...],
    min_transitions: int,
    min_episodes_per_task: int,
    min_sampleable_windows: int = 0,
    require_classifier_evidence: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
```

Inside `pack_replay_task_stats_for_ddp()`, call:

```python
        replay.ready_for_training(
            min_transitions=int(min_transitions),
            task_ids=task_ids,
            min_episodes_per_task=int(min_episodes_per_task),
            min_sampleable_windows=int(min_sampleable_windows),
            require_classifier_evidence=bool(require_classifier_evidence),
        )
```

Update `unpack_replay_task_stats_from_ddp()`:

```python
def unpack_replay_task_stats_from_ddp(
    packed: torch.Tensor,
    *,
    task_ids: tuple[int, ...],
    world_size: int,
    min_transitions: int = 0,
    min_episodes_per_task: int = 1,
    min_sampleable_windows: int = 0,
) -> tuple[dict[str, dict[str, int]], bool, bool]:
```

Compute window coverage from packed stats:

```python
    total_sampleable_windows = sum(
        stats[str(int(task_id))]["sampleable_windows"] for task_id in task_ids
    )
    global_coverage_ready = (
        total_transitions >= int(min_transitions)
        and total_sampleable_windows >= int(min_sampleable_windows)
        and global_task_ready
    )
```

Update `get_replay_task_stats_global()` to accept and pass:

```python
    min_sampleable_windows: int = 0,
    require_classifier_evidence: bool = False,
```

- [ ] **Step 5: Verify replay readiness**

Run:

```bash
pytest tests/unit_tests/test_online_replay_task_balanced.py -q
```

Expected: all replay tests pass.

## Task 2: Wire No-Ray Cotrain Readiness from Hydra

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py:735`
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml:87`
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py`

- [ ] **Step 1: Add a runner-level wiring test**

Add this test to `tests/unit_tests/test_online_cotrain_pipeline.py`:

```python
def test_online_cotrain_loop_passes_full_ready_gates(monkeypatch):
    from types import SimpleNamespace

    import torch
    from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner

    captured = {}
    runner = OnlineCotrainRunner.__new__(OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner.global_step = 0
    runner.distributed = SimpleNamespace(world_size=1, rank=0, is_main_process=True)

    class Replay:
        num_transitions = 16

    def fake_global_ready(replay, **kwargs):
        del replay
        captured.update(kwargs)
        return {}, False, False

    monkeypatch.setattr(
        "dreamervla.runners.online_cotrain_runner.get_replay_task_stats_global",
        fake_global_ready,
    )

    stop = runner._run_training_bursts(
        env_step=1,
        total_env_steps=1,
        replay=Replay(),
        env_task_ids=(0,),
        knobs={
            "train_trigger": "episode_end",
            "updates_per_episode": 1,
            "updates_per_train": 1,
            "train_every": 1,
            "min_replay": 12,
            "min_eps": 1,
            "min_sampleable_windows": 9,
            "require_classifier_evidence": True,
            "is_dist": False,
            "batch_size": 1,
            "max_train_updates": 0,
            "warmup_steps": 0,
        },
        counters={"n_episodes": 1, "n_success": 0},
        history=[],
        episode_added=True,
    )

    assert stop is False
    assert captured["min_sampleable_windows"] == 9
    assert captured["require_classifier_evidence"] is True
```

- [ ] **Step 2: Run the wiring test**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_online_cotrain_loop_passes_full_ready_gates -q
```

Expected before implementation: FAIL because `_run_training_bursts()` does not pass the new gate keys.

- [ ] **Step 3: Read Hydra keys into `knobs`**

In `dreamervla/runners/online_cotrain_runner.py`, after `min_replay` and `min_eps` are read, add:

```python
        min_sampleable_windows = int(
            OmegaConf.select(oc, "min_sampleable_windows", default=0)
        )
        require_classifier_evidence = bool(
            OmegaConf.select(oc, "require_classifier_evidence", default=False)
        )
```

When constructing `knobs`, include:

```python
            "min_sampleable_windows": min_sampleable_windows,
            "require_classifier_evidence": require_classifier_evidence,
```

In `_run_training_bursts()`, update the global readiness call:

```python
        _stats, _cov_ready, all_ready = get_replay_task_stats_global(
            replay,
            task_ids=env_task_ids,
            min_transitions=knobs["min_replay"],
            min_episodes_per_task=knobs["min_eps"],
            min_sampleable_windows=knobs.get("min_sampleable_windows", 0),
            require_classifier_evidence=knobs.get("require_classifier_evidence", False),
            device=self.device,
            is_dist=knobs["is_dist"],
            world_size=self._world_size,
        )
```

- [ ] **Step 4: Declare defaults in pipeline config**

In `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml` under `online_rollout`, add:

```yaml
  min_sampleable_windows: 0
  require_classifier_evidence: true
```

Keep `min_sampleable_windows: 0` as the default because small smokes may have few windows; full runs override it from observed replay size.

- [ ] **Step 5: Verify no-Ray readiness wiring**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_online_cotrain_loop_passes_full_ready_gates tests/unit_tests/test_online_replay_task_balanced.py -q
```

Expected: selected tests pass.

## Task 3: Wire Ray Replay Readiness to the Same Contract

**Files:**
- Modify: `dreamervla/workers/replay/replay_worker.py:79`
- Modify: `dreamervla/runners/online_cotrain_ray_runner.py:388`
- Modify: `configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml:136`
- Modify: `tests/unit_tests/test_online_cotrain_ray_runner.py`

- [ ] **Step 1: Add a Ray runner readiness-call test**

Add to `tests/unit_tests/test_online_cotrain_ray_runner.py`:

```python
def test_ray_runner_passes_replay_ready_gates(monkeypatch) -> None:
    from omegaconf import OmegaConf
    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    captured = {}

    class ReadyResult:
        refs = ["ready"]
        def wait(self):
            return [False]

    class Replay:
        def ready(self, **kwargs):
            captured.update(kwargs)
            return ReadyResult()

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create(
        {
            "rollout": {
                "steps": 0,
                "min_replay_episodes": 2,
                "min_replay_transitions": 24,
                "min_sampleable_windows": 12,
                "require_classifier_evidence": True,
            },
            "sync": {"weight_sync_every": 1},
            "learner": {"train_cfg": {"mode": "dreamervla_cotrain"}},
        }
    )

    runner._run_loop_overlap(
        {
            "envs": type("EnvGroup", (), {"current_obs": lambda self: ReadyResult()})(),
            "infer": object(),
            "replay": Replay(),
            "learner": object(),
            "store_name": "test_store",
            "num_envs": 0,
        }
    )

    assert captured["min_episodes_per_task"] == 2
    assert captured["min_transitions"] == 24
    assert captured["min_sampleable_windows"] == 12
    assert captured["require_classifier_evidence"] is True
```

- [ ] **Step 2: Run the Ray wiring test**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_passes_replay_ready_gates -q
```

Expected before implementation: FAIL because Ray calls `replay.ready(min_episodes)` positionally.

- [ ] **Step 3: Update `ReplayWorker.ready()`**

In `dreamervla/workers/replay/replay_worker.py`, replace `ready()` with:

```python
    def ready(
        self,
        min_episodes_per_task: int,
        *,
        min_transitions: int = 0,
        task_ids: tuple[int, ...] | None = None,
        min_sampleable_windows: int = 0,
        require_classifier_evidence: bool = False,
    ) -> bool:
        replay = self._replay()
        selected_task_ids = (
            tuple(int(task_id) for task_id in task_ids)
            if task_ids is not None
            else tuple(int(task_id) for task_id in (replay.task_ids or (0,)))
        )
        return replay.ready_for_training(
            min_transitions=int(min_transitions),
            task_ids=selected_task_ids,
            min_episodes_per_task=int(min_episodes_per_task),
            min_sampleable_windows=int(min_sampleable_windows),
            require_classifier_evidence=bool(require_classifier_evidence),
        )
```

- [ ] **Step 4: Update Ray runner call site**

In `dreamervla/runners/online_cotrain_ray_runner.py`, read:

```python
        min_transitions = self._int_from(
            ("rollout.min_replay_transitions", "min_replay_transitions"), 0
        )
        min_sampleable_windows = self._int_from(
            ("rollout.min_sampleable_windows", "min_sampleable_windows"), 0
        )
        require_classifier_evidence = bool(
            OmegaConf.select(self.cfg, "rollout.require_classifier_evidence", default=False)
        )
        replay_task_ids = tuple(self._rollout_task_ids() or [0])
```

Replace `replay.ready(min_episodes).wait()[0]` with:

```python
            if not bool(
                replay.ready(
                    min_episodes,
                    min_transitions=min_transitions,
                    task_ids=replay_task_ids,
                    min_sampleable_windows=min_sampleable_windows,
                    require_classifier_evidence=require_classifier_evidence,
                ).wait()[0]
            ):
                return
```

- [ ] **Step 5: Declare Ray defaults**

In `configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml`, under `rollout`, add:

```yaml
  min_replay_transitions: ${ray_data.sequence_length}
  min_sampleable_windows: 0
  require_classifier_evidence: true
```

- [ ] **Step 6: Verify Ray replay wiring**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_ray_runner.py tests/unit_tests/test_replay_client_sample_forwarding.py -q
```

Expected: selected Ray unit tests pass.

## Task 4: Keep Three-Pool Sampling and Latest Online Priority Stable

**Files:**
- Modify: `tests/unit_tests/test_online_replay_task_balanced.py:240`
- Verify: `dreamervla/runners/online_replay.py:310`

- [ ] **Step 1: Add empty-pool fallback coverage**

Append to `tests/unit_tests/test_online_replay_task_balanced.py`:

```python
def test_online_replay_three_pool_sampling_falls_back_to_available_pool() -> None:
    random.seed(7)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_balanced=False,
        replay_sampling={
            "enabled": True,
            "recent_episode_count": 8,
            "mix": {
                "online_recent": 0.0,
                "online_replay": 0.0,
                "coldstart_anchor": 1.0,
            },
        },
    )
    online = replay.add_episode(_episode(task_id=0, length=6, success=True), source="online")

    batch = replay.sample(3)

    assert online is not None
    assert set(batch["episode_ids"].tolist()) == {int(online["episode_id"])}
    assert set(batch["replay_source_ids"].tolist()) == {1}
```

- [ ] **Step 2: Run sampling tests**

Run:

```bash
pytest tests/unit_tests/test_online_replay_task_balanced.py::test_online_replay_three_pool_sampling_falls_back_to_available_pool tests/unit_tests/test_online_replay_task_balanced.py::test_online_replay_latest_online_required_samples_new_episode_first -q
```

Expected: both tests pass. If fallback fails, change `_choose_pool_name()` so it renormalizes over non-empty pools only, using the existing pool weights.

- [ ] **Step 3: Run the full replay suite**

Run:

```bash
pytest tests/unit_tests/test_online_replay_task_balanced.py tests/unit_tests/test_online_cotrain_pipeline.py tests/unit_tests/test_online_cotrain_ray_runner.py -q
```

Expected: selected suites pass.

- [ ] **Step 4: Commit**

```bash
git add dreamervla/runners/online_replay.py dreamervla/runners/online_cotrain_runner.py dreamervla/workers/replay/replay_worker.py dreamervla/runners/online_cotrain_ray_runner.py configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml tests/unit_tests/test_online_replay_task_balanced.py tests/unit_tests/test_online_cotrain_pipeline.py tests/unit_tests/test_online_cotrain_ray_runner.py
git commit -s -m "feat(cotrain): gate replay readiness on sampleable windows"
```
