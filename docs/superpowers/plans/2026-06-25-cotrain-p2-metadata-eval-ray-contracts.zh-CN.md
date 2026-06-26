# Cotrain P2 Metadata Eval and Ray Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make full cotrain runs auditable by writing episode-level metadata, scheduling real eval windows, and separating Ray OFT fixed-base rollout from Ray learned-actor rollout contracts.

**Architecture:** Store scalar/string episode metadata as HDF5 demo attrs and collection-level metadata in `collection_manifest.json`. Add a CPU-testable real-eval scheduler that triggers only after completed real episodes or learner updates. Keep Ray rollout modes explicit in config so OFT fixed-base open-loop chunk execution and learned-actor rollout are not mixed.

**Tech Stack:** Python 3.11, h5py, JSON, pytest, Hydra/OmegaConf, Ray worker configs.

---

## File Structure

- Modify: `dreamervla/dataset/rollout_dump_writer.py:73`
  - Add `episode_metadata` attr writer with scalar/string filtering.
- Modify: `dreamervla/workers/rollout/dump_worker.py:63`
  - Pass collection metadata through Ray dump worker.
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py:600`
  - Enrich `collection_manifest.json`.
- Modify: `tests/unit_tests/test_rollout_dump_writer.py`
  - Test demo attrs for chunk/action/hidden schema.
- Modify: `tests/unit_tests/test_collection_manifest.py`
  - Test manifest schema fields.
- Create: `dreamervla/runners/real_eval_schedule.py`
  - Pure helper for periodic real eval decisions.
- Create: `tests/unit_tests/test_real_eval_schedule.py`
  - CPU tests for episode/update triggers.
- Modify: `dreamervla/runners/online_cotrain_runner.py`
  - Call eval scheduler from the training burst after real episode counters update.
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml`
  - Declare `online_eval` default-off schedule.
- Modify: `dreamervla/runners/online_cotrain_ray_runner.py`
  - Validate Ray rollout mode and keep OFT fixed-base action chunk execution explicit.
- Modify: `configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml`
  - Declare `ray_rollout.mode: oft_fixed_base`.
- Modify: `tests/unit_tests/test_online_cotrain_ray_runner.py`
  - Test Ray mode validation.

## Task 1: Write Episode-Level Metadata Attrs

**Files:**
- Modify: `dreamervla/dataset/rollout_dump_writer.py:73`
- Modify: `dreamervla/workers/rollout/dump_worker.py:63`
- Modify: `tests/unit_tests/test_rollout_dump_writer.py`

- [ ] **Step 1: Add writer metadata test**

Add to `tests/unit_tests/test_rollout_dump_writer.py`:

```python
def test_writer_episode_metadata_attrs(tmp_path: Path) -> None:
    import h5py
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter

    reward_dir = tmp_path / "reward"
    hidden_dir = tmp_path / "hidden"
    writer = RolloutDumpWriter(
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        shard_name="shard_000.hdf5",
    )
    writer.write_demo(
        index=0,
        steps=_make_episode(success=True),
        preprocess_config=PREPROCESS_CONFIG,
        task_id=2,
        episode_id=7,
        episode_success=True,
        episode_horizon=300,
        episode_metadata={
            "suite": "libero_goal",
            "task_name": "open drawer",
            "global_episode_index": 123,
            "policy_name": "openvla_oft_default",
            "policy_ckpt": "/ckpts/policy",
            "policy_version": 5,
            "success_step": 9,
            "timeout": False,
            "chunk_size": 8,
            "action_scale": "raw",
            "seed": 17,
            "render_backend": "egl",
            "hidden_key": "obs_embedding",
            "hidden_dim": HIDDEN_DIM,
            "token_count": 56,
            "token_dim": 4096,
        },
    )
    writer.close()

    with h5py.File(reward_dir / "shard_000.hdf5", "r") as f:
        attrs = f["data"]["demo_0"].attrs
        assert attrs["task_id"] == 2
        assert attrs["episode_id"] == 7
        assert attrs["suite"] == "libero_goal"
        assert attrs["chunk_size"] == 8
        assert attrs["action_scale"] == "raw"
        assert attrs["hidden_dim"] == HIDDEN_DIM
        assert attrs["token_count"] == 56
        assert attrs["token_dim"] == 4096
```

- [ ] **Step 2: Run the metadata writer test**

Run:

```bash
pytest tests/unit_tests/test_rollout_dump_writer.py::test_writer_episode_metadata_attrs -q
```

Expected before implementation: FAIL because `write_demo()` does not accept `episode_metadata`.

- [ ] **Step 3: Extend `RolloutDumpWriter.write_demo()`**

In `dreamervla/dataset/rollout_dump_writer.py`, import:

```python
from collections.abc import Mapping
```

Add parameter:

```python
        episode_metadata: Mapping[str, Any] | None = None,
```

After existing per-demo attrs, add:

```python
        if episode_metadata is not None:
            for key, value in episode_metadata.items():
                if value is None:
                    continue
                if isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating)):
                    demo_grp.attrs[str(key)] = value
```

- [ ] **Step 4: Pass metadata through Ray dump worker**

In `dreamervla/workers/rollout/dump_worker.py`, when calling `write_demo()`, pass:

```python
            episode_metadata=episode[-1].get("episode_metadata", None),
```

When no step contains metadata, this keeps current behavior unchanged.

- [ ] **Step 5: Verify writer suite**

Run:

```bash
pytest tests/unit_tests/test_rollout_dump_writer.py tests/unit_tests/test_rollout_dump_writer_identity.py -q
```

Expected: selected writer tests pass.

## Task 2: Enrich Collection Manifest

**Files:**
- Modify: `dreamervla/launchers/coldstart_warmup_cotrain.py:600`
- Modify: `tests/unit_tests/test_collection_manifest.py`

- [ ] **Step 1: Add manifest schema test**

Add to `tests/unit_tests/test_collection_manifest.py`:

```python
def test_write_collection_manifest_records_hidden_schema(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import dreamervla.launchers.coldstart_warmup_cotrain as launcher

    collected_root = tmp_path / "collected"
    reward_dir = collected_root / "reward"
    hidden_dir = collected_root / "hidden"
    reward_dir.mkdir(parents=True)
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "preprocess_config.json").write_text(
        json.dumps(
            {
                "hidden_key": "obs_embedding",
                "chunk_size": 8,
                "token_count": 56,
                "token_dim": 4096,
                "output_dtype": "float16",
            }
        ),
        encoding="utf-8",
    )
    _write_shard_with_task_ids(reward_dir / "shard_000.hdf5", [0, 1])
    plan = SimpleNamespace(
        task="openvla_onetraj_coldstart_libero",
        mode="full",
        profile="release",
        reward_dir=reward_dir,
        hidden_dir=hidden_dir,
        collected_root=collected_root,
        run_root=tmp_path / "run",
        collect_cmd=["python", "-m", "dreamervla.train"],
    )

    launcher._write_collection_manifest(plan, target_episodes=10, num_tasks=2)

    manifest = json.loads((collected_root / "collection_manifest.json").read_text())
    assert manifest["hidden_schema"]["hidden_key"] == "obs_embedding"
    assert manifest["hidden_schema"]["chunk_size"] == 8
    assert manifest["hidden_schema"]["token_count"] == 56
    assert manifest["hidden_schema"]["token_dim"] == 4096
    assert manifest["backend"] in {"unknown", "egl", "osmesa"}
```

- [ ] **Step 2: Run manifest schema test**

Run:

```bash
pytest tests/unit_tests/test_collection_manifest.py::test_write_collection_manifest_records_hidden_schema -q
```

Expected before implementation: FAIL because manifest lacks `hidden_schema`.

- [ ] **Step 3: Read preprocess schema in launcher**

In `_write_collection_manifest()`, before `write_manifest()`, add:

```python
    hidden_schema: dict[str, object] = {}
    preprocess_path = plan.hidden_dir / "preprocess_config.json"
    if preprocess_path.is_file():
        try:
            preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            preprocess = {}
        for key in ("hidden_key", "chunk_size", "token_count", "token_dim", "output_dtype"):
            if key in preprocess:
                hidden_schema[key] = preprocess[key]
```

Add to the manifest dict:

```python
            "backend": os.environ.get("MUJOCO_GL", "unknown"),
            "hidden_schema": hidden_schema,
```

- [ ] **Step 4: Verify manifest tests**

Run:

```bash
pytest tests/unit_tests/test_collection_manifest.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -q
```

Expected: selected tests pass.

## Task 3: Add Real Eval Scheduling Helper

**Files:**
- Create: `dreamervla/runners/real_eval_schedule.py`
- Create: `tests/unit_tests/test_real_eval_schedule.py`
- Modify: `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml`

- [ ] **Step 1: Write scheduler tests**

Create `tests/unit_tests/test_real_eval_schedule.py`:

```python
from __future__ import annotations

from dreamervla.runners.real_eval_schedule import RealEvalState, should_run_real_eval


def test_real_eval_schedule_triggers_on_completed_episode_window() -> None:
    state = RealEvalState(last_eval_episode=0, last_eval_update=0)

    assert should_run_real_eval(
        enabled=True,
        completed_episodes=5,
        learner_updates=0,
        every_episodes=5,
        every_learner_updates=0,
        state=state,
    ) is True


def test_real_eval_schedule_does_not_trigger_without_new_budget() -> None:
    state = RealEvalState(last_eval_episode=5, last_eval_update=10)

    assert should_run_real_eval(
        enabled=True,
        completed_episodes=7,
        learner_updates=12,
        every_episodes=5,
        every_learner_updates=5,
        state=state,
    ) is False


def test_real_eval_schedule_triggers_on_update_window() -> None:
    state = RealEvalState(last_eval_episode=0, last_eval_update=10)

    assert should_run_real_eval(
        enabled=True,
        completed_episodes=0,
        learner_updates=15,
        every_episodes=0,
        every_learner_updates=5,
        state=state,
    ) is True
```

- [ ] **Step 2: Run scheduler tests**

Run:

```bash
pytest tests/unit_tests/test_real_eval_schedule.py -q
```

Expected before implementation: FAIL because `dreamervla.runners.real_eval_schedule` does not exist.

- [ ] **Step 3: Implement scheduler helper**

Create `dreamervla/runners/real_eval_schedule.py`:

```python
"""Pure scheduling helper for periodic real-eval windows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RealEvalState:
    last_eval_episode: int = 0
    last_eval_update: int = 0


def should_run_real_eval(
    *,
    enabled: bool,
    completed_episodes: int,
    learner_updates: int,
    every_episodes: int,
    every_learner_updates: int,
    state: RealEvalState,
) -> bool:
    if not bool(enabled):
        return False
    if int(every_episodes) > 0:
        if int(completed_episodes) - int(state.last_eval_episode) >= int(every_episodes):
            return True
    if int(every_learner_updates) > 0:
        if int(learner_updates) - int(state.last_eval_update) >= int(every_learner_updates):
            return True
    return False


__all__ = ["RealEvalState", "should_run_real_eval"]
```

- [ ] **Step 4: Declare default-off schedule**

In `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml`, add:

```yaml
online_eval:
  enabled: false
  every_episodes: 25
  every_learner_updates: 0
  num_episodes: 10
```

- [ ] **Step 5: Verify helper**

Run:

```bash
pytest tests/unit_tests/test_real_eval_schedule.py -q
```

Expected: `3 passed`.

## Task 4: Hook Real Eval Schedule into Online Cotrain

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py`
- Modify: `tests/unit_tests/test_online_cotrain_pipeline.py`

- [ ] **Step 1: Add runner hook test**

Add to `tests/unit_tests/test_online_cotrain_pipeline.py`:

```python
def test_online_cotrain_real_eval_schedule_invokes_hook(monkeypatch):
    from types import SimpleNamespace
    import torch
    from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner

    runner = OnlineCotrainRunner.__new__(OnlineCotrainRunner)
    runner.device = torch.device("cpu")
    runner.global_step = 5
    runner.distributed = SimpleNamespace(world_size=1, rank=0, is_main_process=True)
    runner._real_eval_calls = 0
    runner._real_eval_state = SimpleNamespace(last_eval_episode=0, last_eval_update=0)
    runner._maybe_run_periodic_real_eval = lambda counters, learner_updates: setattr(
        runner, "_real_eval_calls", runner._real_eval_calls + 1
    )

    class Replay:
        num_transitions = 0

    monkeypatch.setattr(
        "dreamervla.runners.online_cotrain_runner.get_replay_task_stats_global",
        lambda *args, **kwargs: ({}, False, False),
    )

    runner._run_training_bursts(
        env_step=1,
        total_env_steps=1,
        replay=Replay(),
        env_task_ids=(0,),
        knobs={
            "train_trigger": "episode_end",
            "updates_per_episode": 1,
            "updates_per_train": 1,
            "train_every": 1,
            "min_replay": 1,
            "min_eps": 1,
            "is_dist": False,
            "batch_size": 1,
            "max_train_updates": 0,
            "warmup_steps": 0,
        },
        counters={"n_episodes": 25, "n_success": 0},
        history=[],
        episode_added=True,
    )

    assert runner._real_eval_calls == 1
```

- [ ] **Step 2: Run hook test**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_pipeline.py::test_online_cotrain_real_eval_schedule_invokes_hook -q
```

Expected before implementation: FAIL because `_run_training_bursts()` does not invoke the eval hook.

- [ ] **Step 3: Add hook call after rollout metrics are built**

In `_run_training_bursts()`, after `metrics.update(build_rollout_progress_metrics(...))`, call:

```python
            self._maybe_run_periodic_real_eval(
                counters=counters,
                learner_updates=int(self.global_step),
            )
```

Add method to `OnlineCotrainRunner`:

```python
    def _maybe_run_periodic_real_eval(
        self,
        *,
        counters: dict[str, int],
        learner_updates: int,
    ) -> None:
        from dreamervla.runners.real_eval_schedule import RealEvalState, should_run_real_eval

        eval_cfg = OmegaConf.select(self.cfg, "online_eval", default={}) or {}
        if OmegaConf.is_config(eval_cfg):
            eval_cfg = OmegaConf.to_container(eval_cfg, resolve=True)
        eval_cfg = dict(eval_cfg)
        state = getattr(self, "_real_eval_state", None)
        if state is None:
            state = RealEvalState()
            self._real_eval_state = state
        completed = int(counters.get("n_episodes", 0))
        if not should_run_real_eval(
            enabled=bool(eval_cfg.get("enabled", False)),
            completed_episodes=completed,
            learner_updates=int(learner_updates),
            every_episodes=int(eval_cfg.get("every_episodes", 0) or 0),
            every_learner_updates=int(eval_cfg.get("every_learner_updates", 0) or 0),
            state=state,
        ):
            return
        state.last_eval_episode = completed
        state.last_eval_update = int(learner_updates)
        self.console_banner(
            "REAL EVAL",
            subtitle=f"episodes={completed} learner_updates={int(learner_updates)}",
        )
```

This hook schedules and records the eval boundary without changing default runs. Full real-eval execution remains controlled by the GPU-gated command in Task 6, so CPU tests verify cadence while LIBERO runs verify real completed-episode metrics.

- [ ] **Step 4: Verify eval schedule hook**

Run:

```bash
pytest tests/unit_tests/test_real_eval_schedule.py tests/unit_tests/test_online_cotrain_pipeline.py::test_online_cotrain_real_eval_schedule_invokes_hook -q
```

Expected: selected tests pass.

## Task 5: Make Ray Rollout Mode Explicit

**Files:**
- Modify: `dreamervla/runners/online_cotrain_ray_runner.py`
- Modify: `configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml`
- Modify: `tests/unit_tests/test_online_cotrain_ray_runner.py`

- [ ] **Step 1: Add Ray mode validation tests**

Add to `tests/unit_tests/test_online_cotrain_ray_runner.py`:

```python
def test_ray_runner_accepts_declared_oft_fixed_base_mode():
    from omegaconf import OmegaConf
    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({"ray_rollout": {"mode": "oft_fixed_base"}})

    assert runner._ray_rollout_mode() == "oft_fixed_base"


def test_ray_runner_rejects_unknown_rollout_mode():
    import pytest
    from omegaconf import OmegaConf
    from dreamervla.runners.online_cotrain_ray_runner import OnlineCotrainRayRunner

    runner = OnlineCotrainRayRunner.__new__(OnlineCotrainRayRunner)
    runner.cfg = OmegaConf.create({"ray_rollout": {"mode": "mixed"}})

    with pytest.raises(ValueError, match="ray_rollout.mode"):
        runner._ray_rollout_mode()
```

- [ ] **Step 2: Run Ray mode tests**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_accepts_declared_oft_fixed_base_mode tests/unit_tests/test_online_cotrain_ray_runner.py::test_ray_runner_rejects_unknown_rollout_mode -q
```

Expected before implementation: FAIL because `_ray_rollout_mode()` does not exist.

- [ ] **Step 3: Implement mode helper**

In `dreamervla/runners/online_cotrain_ray_runner.py`, add:

```python
    def _ray_rollout_mode(self) -> str:
        mode = str(OmegaConf.select(self.cfg, "ray_rollout.mode", default="oft_fixed_base"))
        allowed = {"oft_fixed_base", "learned_actor"}
        if mode not in allowed:
            raise ValueError(
                f"ray_rollout.mode must be one of {sorted(allowed)}, got {mode!r}"
            )
        return mode
```

At the start of `_build_components()` or `_run_loop_overlap()`, call:

```python
        rollout_mode = self._ray_rollout_mode()
```

For `oft_fixed_base`, keep using `RolloutInferenceWorker` and its `action_steps` queue. For `learned_actor`, raise a clear error until the learned-actor Ray inference worker is wired:

```python
        if rollout_mode == "learned_actor":
            raise ValueError(
                "ray_rollout.mode=learned_actor requires a learned-actor inference worker; "
                "use no-Ray OnlineCotrainRunner or select ray_rollout.mode=oft_fixed_base."
            )
```

- [ ] **Step 4: Declare Ray config mode**

In `configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml`, add:

```yaml
ray_rollout:
  mode: oft_fixed_base
```

- [ ] **Step 5: Verify Ray mode suite**

Run:

```bash
pytest tests/unit_tests/test_online_cotrain_ray_runner.py tests/unit_tests/test_rollout_inference_worker.py -q
```

Expected: selected tests pass.

## Task 6: Run P2 Metadata/Eval/Ray Regression Suite

**Files:**
- Verify: all files modified in this plan.

- [ ] **Step 1: Run CPU tests**

Run:

```bash
pytest tests/unit_tests/test_rollout_dump_writer.py tests/unit_tests/test_collection_manifest.py tests/unit_tests/test_real_eval_schedule.py tests/unit_tests/test_online_cotrain_ray_runner.py tests/unit_tests/test_rollout_inference_worker.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run GPU-gated full-run verification on a GPU/LIBERO host**

Run only when assets and a free GPU are available:

```bash
python -m dreamervla.train experiment=online_cotrain_pipeline_oft_action_hidden_smoke task=openvla_onetraj_coldstart_libero
```

Expected:
- Cold-start data is present or collection has already populated `collection_manifest.json`.
- Warmup checkpoints are written under `${training.out_dir}/ckpt`.
- At least one completed real episode logs `rollout/success_rate`.
- Any `LUMOS/success_rate` appears only as imagined diagnostic, not as rollout success.

- [ ] **Step 3: Commit**

```bash
git add dreamervla/dataset/rollout_dump_writer.py dreamervla/workers/rollout/dump_worker.py dreamervla/launchers/coldstart_warmup_cotrain.py dreamervla/runners/real_eval_schedule.py dreamervla/runners/online_cotrain_runner.py dreamervla/runners/online_cotrain_ray_runner.py configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml tests/unit_tests/test_rollout_dump_writer.py tests/unit_tests/test_collection_manifest.py tests/unit_tests/test_real_eval_schedule.py tests/unit_tests/test_online_cotrain_pipeline.py tests/unit_tests/test_online_cotrain_ray_runner.py tests/unit_tests/test_rollout_inference_worker.py
git commit -s -m "feat(cotrain): add metadata eval and ray mode contracts"
```
