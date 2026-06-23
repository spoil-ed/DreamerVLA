# Cotrain vectorized egl rollout — RLinf parallel-env-worker alignment (Option 1)

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this task-by-task. Steps use
> checkbox (`- [ ]`) syntax. **All Python runs in the `dreamervla` conda env**
> (`conda run -n dreamervla …`); base env yields ~13 spurious failures, clean baseline = 582 passed.

**Goal:** Switch the online cotrain *rollout* from a single env + `MUJOCO_GL=osmesa` (CPU, ~130 s/demo)
to N parallel `VecRolloutEnv` child-process envs rendering with `egl` (GPU, ~30 s/demo, ~4×), reusing
the data-collection vectorized facilities — without changing the RL/WM/classifier training math.

**Architecture:** Mirror the proven `vectorized_collect.collect_vectorized` continuous K-slot loop
inside the cotrain runner: env instances live in spawned children (egl, isolated GL context → no
robosuite `read_pixels` SIGABRT); the parent runs per-slot OFT extractors on GPU, drives all envs via
`VecRolloutEnv.step`, rebuilds replay transitions from each child's `full_record`, and interleaves the
existing training burst every `train_every` env-steps.

**RLinf alignment:** This is the **parallel-env-worker + egl** layer of RLinf (each env an isolated
worker process). The fuller **async pipeline** (GPU-infer ∥ CPU-step ∥ learner overlap, Ray backend)
is Option 2, a separate plan. This plan keeps the default single-machine Runner path and adds early
config validation + RLinf metric namespaces (`rollout/`, `time/`).

---

## Problem (grounded in current code)

`OnlineCotrainRunner._online_cotrain_loop` (`dreamervla/runners/online_cotrain_runner.py:483–694`)
is single-env and synchronous:

- Module top forces CPU rendering: `os.environ.setdefault("MUJOCO_GL", "osmesa")` +
  `PYOPENGL_PLATFORM=osmesa` (`online_cotrain_runner.py:38–39`). This is the 2026-06-23 SIGABRT
  workaround (robosuite egl `read_pixels` crashes in the single training process). osmesa is correct
  but ~4× slower than egl — the throughput bottleneck.
- One env built once (`_build_env` → `online_cotrain_runner.py:171–206`), one global action queue
  `self._rollout_action_queue` (`:228–229`), one shared extractor `self._oft_action_hidden_extractor`.
- Loop `for env_step in range(1, total_env_steps + 1)` (`:555`): one step/iter →
  `_rollout_action` (`:559`) → `env.step` (`:562`) → `env.make_transition` (`:565`) → append; on `done`
  `replay.add_episode` + `env.reset`; train burst every `train_every` env-steps (`:595`).

The data-collection path already solved parallel egl rollout: `VecRolloutEnv`
(`dreamervla/runners/vec_rollout_env.py`) spawns K child envs and passes `env_vars` (incl.
`MUJOCO_GL`) into each child; `collect_vectorized` (`vectorized_collect.py:75–219`) drives them with a
per-slot action queue + per-slot extractor. Cotrain just never adopted it.

## Goal & success criteria

1. `num_envs>1` cotrain rolls out across K egl child envs, **no SIGABRT**, reaches an RL/actor update.
2. The replay transitions built in multi-env mode are **numerically identical** to what single-env
   `make_transition` produces (same `image`, `state`, `wm_action`, flags) — proven by unit test.
3. `num_envs==1` keeps the **legacy single-env osmesa path unchanged** (known-good A/B baseline).
4. Full unit suite green in `dreamervla` env; `ruff check` clean.
5. GPU smoke: `debug=true`, `num_envs=4`, `render_backend=egl` → no crash, reaches a training burst,
   measurably faster wall-clock per env-step than osmesa.

## Key facts that make it correct (verified, with file:line)

- **`state` is already reproducible from `full_record`.**
  `vectorized_collect.proprio_from_record(rec)` (`vectorized_collect.py:39–41`,
  `_PROPRIO_KEYS=("ee_pos","ee_ori","gripper_states")`) concatenates to 8-dim float32 and its docstring
  states it *"matches env._format_obs 'state'"*. `_format_obs` builds `state` as
  `eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2)` (`train_env.py:566–575`) → identical.
- **`image` is reproducible from `full_record`.** `_format_obs` builds the Dreamer image as
  `concat([resize(agentview, S).CHW, resize(wrist, S).CHW])` uint8 (`train_env.py:540–548`), where
  `agentview`/`wrist` are `_camera_image(raw, …, rotate_180=cfg.pixel_rotate_180)` — the SAME tensors
  `full_record()` returns as `agentview_rgb`/`eye_in_hand_rgb` (`train_env.py:280–317`). Only resize +
  CHW-concat remains; `_resize_hwc_uint8` is a pure staticmethod (`train_env.py:514–524`),
  `cfg.image_size` default 64 (`train_env.py:67`).
- **`wm_action` build is a one-liner reusable as-is.** Single-env stores
  `wm_action = info.get("wm_action", policy_action)[:7]` (`online_cotrain_runner.py:564`).
  `VecRolloutEnv.step` returns the child's `info`, so multi-env uses the identical expression with
  `action_sent` in place of `policy_action`.
- **Replay only consumes a small key set.** `OnlineReplay.sample` reads per step:
  `image, obs_embedding, reward, done, is_terminal, is_last, wm_action`
  (`online_replay.py:263–298`); `add_episode` reads `task_id` (+ optional `success`)
  (`online_replay.py:68–101`). `state`, `task_description`, `discount`, `is_first`, `step` are stored
  for parity/audit but only the listed keys gate training.
- **Per-slot isolation is already the collection idiom.** `collect_vectorized` keeps
  `action_queues[k]` and one `extractors[k]` per slot, slot-0 reusing the main extractor
  (`collect_parallel_rollouts.py:284–294`). Each `OFTRolloutHiddenExtractor` carries per-view history
  buffers, so sharing one across slots would corrupt history — must be one-per-slot + `.reset()` on
  episode start.
- **egl in a child is safe and fast.** SIGABRT only happens with egl in the single training process;
  spawned children get isolated GL contexts (collection runs egl multi-env without crashing).
  `env_vars` is the lever: `collect_parallel_rollouts.py:271–275` builds it as
  `{k: os.environ[k] for k in ("MUJOCO_GL","PYOPENGL_PLATFORM","DVLA_DATA_ROOT","LIBERO_CONFIG_PATH") if k in os.environ}`.

## Design

### Decisions (review these)

- **D1 — `num_envs` knob + legacy fallback.** New `online_rollout.num_envs` (runner default `1`).
  `==1` → existing single-env osmesa path, untouched. `>1` → new vectorized egl path. Cotrain config
  ships `num_envs: 4` (tunable; main throughput/GPU-memory tradeoff — parent also holds WM+policy+
  classifier for the burst, and egl contexts consume GPU per child).
- **D2 — scope: OFT `action_hidden` path only.** The vectorized path supports the default OFT
  `action_hidden` rollout (the one online cotrain uses). The `backbone_latent` (WM-latent) path stays
  single-env; assert `num_envs==1` when that path is selected. (Per-slot WM latent = Option-2 scope.)
- **D3 — render backend.** Parent keeps `osmesa` setdefault (it never renders — harmless). Children
  get `render_backend` (new `online_rollout.render_backend`, default `egl`) injected via `env_vars`.
  Early validation: `num_envs>1` requires `render_backend in {"egl","osmesa"}`; log the resolved value.
- **D4 — transitions rebuilt in the parent**, not via `env.make_transition` (env is in children).
  A pure helper `build_cotrain_replay_transition(...)` replicates `make_transition` from `full_record`
  + per-slot state. Proven equivalent to single-env by unit test.
- **D5 — infinite refill.** Unlike `collect_vectorized` (drains a finite work-list), cotrain refills a
  finished slot with the next episode of the task set (cycling `env.task_ids`) until `total_env_steps`.
- **D6 — env-step accounting.** Each loop iteration advances `len(active_slots)` env-steps. Train burst
  fires when the env-step counter crosses a `train_every` boundary. Replay fills ~K× faster (note for
  tuning `min_replay`/`train_every`; defaults unchanged in this plan).

### File structure

| File | Change |
|------|--------|
| `dreamervla/runners/vectorized_collect.py` | **Add** `dreamer_image_from_record(rec, image_size)` next to `proprio_from_record`/`extractor_obs_from_record`. |
| `dreamervla/runners/online_cotrain_runner.py` | **Add** `build_cotrain_replay_transition(...)` helper; **add** `_vectorized_cotrain_rollout(...)` loop + per-slot setup (extractors, queues, `env_vars`, `VecRolloutEnv`); **branch** `num_envs>1` in `_online_cotrain_loop`; **add** config validation. Legacy path untouched. |
| `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml` | **Add** `online_rollout.num_envs: 4`, `online_rollout.render_backend: egl`. |
| `configs/dreamervla/_base_wmpo_outcome.yaml` | **Add** `online_rollout.num_envs: 1`, `online_rollout.render_backend: egl` defaults (so validation has keys). |
| `tests/unit_tests/test_cotrain_vec_rollout.py` | **Create** — pure-CPU tests (helpers + loop with fakes). |

### Continuous loop (mirror of `vectorized_collect.py:163–216`, cotrain deltas)

```
build env_vars (D3); extractors[0..K-1] (slot0 = main, rest fresh); VecRolloutEnv(num_envs=K, cfg_kwargs, env_vars)
action_queues=[[]]*K ; episode=[[] for _]; slot_task/slot_ep/slot_step per slot
recs = vec.reset(task_ids=initial, episode_ids=initial)   # per-slot full_record
env_step = 0
while env_step < total_env_steps:
    ids = list(range(K))                                   # all slots always active (infinite refill)
    preps  = [extractors[k].step(extractor_obs_from_record(recs[k]), slot_desc[k]) for k in ids]   # (chunk, hidden) on GPU
    acts   = [pop_open_loop_action(preps[k][0], action_queues[k], action_steps) for k in ids]
    results = vec.step(acts, env_ids=ids)                  # [(reward, term, trunc, info, rec_after)]
    for k in ids:
        reward, term, trunc, info, rec_after = results[k]
        wm_action = np.asarray(info.get("wm_action", acts[k]), np.float32).reshape(-1)[:7]
        obs_emb   = preps[k][1].reshape(-1).cpu().float().numpy()
        tr = build_cotrain_replay_transition(recs[k], obs_emb, wm_action, reward, term, trunc,
                                             task_id=slot_task[k], task_description=slot_desc[k],
                                             step=slot_step[k], is_first=(slot_step[k]==0),
                                             image_size=cfg.image_size)
        episode[k].append(tr); slot_step[k]+=1; env_step+=1
        recs[k] = rec_after
        if term or trunc or slot_step[k] >= episode_horizon:
            replay.add_episode(episode[k])
            episode[k]=[]; action_queues[k].clear(); extractors[k].reset()
            slot_ep[k]+=1; slot_task[k]=next_task(); slot_step[k]=0
            recs[k] = vec.reset([slot_task[k]],[slot_ep[k]], env_ids=[k])[0]
    if replay_ready and crossed_train_every(env_step): run_training_burst()   # existing burst, unchanged
```

`next_task()` cycles `env.task_ids` (`online_cotrain_runner.py:527`). `run_training_burst()` is the
existing block `online_cotrain_runner.py:584–686` extracted/called as-is.

---

## Tasks

### Task 1: `dreamer_image_from_record` shared helper

**Files:**
- Modify: `dreamervla/runners/vectorized_collect.py` (add after `proprio_from_record`, ~line 41)
- Test: `tests/unit_tests/test_cotrain_vec_rollout.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/test_cotrain_vec_rollout.py
import numpy as np
from dreamervla.runners.vectorized_collect import dreamer_image_from_record
from dreamervla.envs.train_env import DreamerVLAOnlineTrainEnv as Env


def _rec(h=120, w=160):
    rng = np.random.default_rng(0)
    return {
        "agentview_rgb": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
        "eye_in_hand_rgb": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
    }


def test_dreamer_image_from_record_matches_format_obs_formula():
    rec, size = _rec(), 64
    out = dreamer_image_from_record(rec, size)
    assert out.shape == (6, size, size) and out.dtype == np.uint8
    third = Env._resize_hwc_uint8(rec["agentview_rgb"], size).transpose(2, 0, 1)
    wrist = Env._resize_hwc_uint8(rec["eye_in_hand_rgb"], size).transpose(2, 0, 1)
    expected = np.concatenate([third, wrist], axis=0).astype(np.uint8)
    np.testing.assert_array_equal(out, expected)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_cotrain_vec_rollout.py::test_dreamer_image_from_record_matches_format_obs_formula -q`
Expected: FAIL — `ImportError: cannot import name 'dreamer_image_from_record'`.

- [ ] **Step 3: Implement (single source of truth via the env staticmethod)**

```python
# dreamervla/runners/vectorized_collect.py  (after proprio_from_record)
def dreamer_image_from_record(rec: dict[str, Any], image_size: int) -> np.ndarray:
    """Dreamer (6,S,S) uint8 image from a full_record (matches env._format_obs 'image')."""
    from dreamervla.envs.train_env import DreamerVLAOnlineTrainEnv

    third = DreamerVLAOnlineTrainEnv._resize_hwc_uint8(rec["agentview_rgb"], image_size)
    wrist = DreamerVLAOnlineTrainEnv._resize_hwc_uint8(rec["eye_in_hand_rgb"], image_size)
    return np.concatenate(
        [third.transpose(2, 0, 1), wrist.transpose(2, 0, 1)], axis=0
    ).astype(np.uint8, copy=False)
```

(Local import avoids a module-load import cycle; `_resize_hwc_uint8` is the single resize primitive →
no drift from `_format_obs`.)

- [ ] **Step 4: Run it, verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/vectorized_collect.py tests/unit_tests/test_cotrain_vec_rollout.py
git commit -s -m "feat(cotrain): dreamer_image_from_record helper for record-sourced transitions"
```

### Task 2: `build_cotrain_replay_transition` (parent-side transition, equivalent to make_transition)

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py` (module-level helper near top, after imports)
- Test: `tests/unit_tests/test_cotrain_vec_rollout.py`

- [ ] **Step 1: Write the failing test (parity with `make_transition`)**

```python
from dreamervla.runners.online_cotrain_runner import build_cotrain_replay_transition
from dreamervla.runners.vectorized_collect import dreamer_image_from_record, proprio_from_record


def _full_record():
    rng = np.random.default_rng(1)
    return {
        "agentview_rgb": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
        "eye_in_hand_rgb": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
        "ee_pos": rng.standard_normal(3).astype(np.float64),
        "ee_ori": rng.standard_normal(3).astype(np.float64),
        "gripper_states": rng.standard_normal(2).astype(np.float64),
    }


def test_build_transition_has_replay_keys_and_dtypes():
    rec = _full_record()
    emb = np.arange(229376, dtype=np.float32)
    wm = np.arange(7, dtype=np.float32)
    tr = build_cotrain_replay_transition(
        rec, emb, wm, reward=1.5, terminated=True, truncated=False,
        task_id=3, task_description="pick up the bowl", step=4, is_first=False, image_size=64,
    )
    # keys OnlineReplay.sample / add_episode require
    for k in ("image", "obs_embedding", "reward", "done", "is_terminal", "is_last", "wm_action", "task_id"):
        assert k in tr
    np.testing.assert_array_equal(tr["image"], dreamer_image_from_record(rec, 64))
    np.testing.assert_array_equal(tr["state"], proprio_from_record(rec))
    np.testing.assert_array_equal(tr["wm_action"], wm)
    assert tr["reward"].dtype == np.float32 and float(tr["reward"]) == 1.5
    assert float(tr["done"]) == 1.0 and float(tr["is_terminal"]) == 1.0 and float(tr["discount"]) == 0.0
    assert tr["task_id"] == 3 and tr["task_description"] == "pick up the bowl"


def test_build_transition_truncated_keeps_discount_one():
    tr = build_cotrain_replay_transition(
        _full_record(), np.zeros(8, np.float32), np.zeros(7, np.float32),
        reward=0.0, terminated=False, truncated=True,
        task_id=0, task_description="t", step=10, is_first=False, image_size=64,
    )
    assert float(tr["done"]) == 1.0 and float(tr["is_terminal"]) == 0.0 and float(tr["discount"]) == 1.0
```

- [ ] **Step 2: Run, verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_cotrain_vec_rollout.py -q -k build_transition`
Expected: FAIL — import error.

- [ ] **Step 3: Implement (mirror `train_env.make_transition:336–367`)**

```python
# dreamervla/runners/online_cotrain_runner.py  (module level, after imports)
from dreamervla.runners.vectorized_collect import (
    dreamer_image_from_record,
    proprio_from_record,
)


def build_cotrain_replay_transition(
    rec: dict[str, Any],
    obs_embedding: np.ndarray,
    wm_action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    *,
    task_id: int,
    task_description: str,
    step: int,
    is_first: bool,
    image_size: int,
) -> dict[str, Any]:
    """Replay transition rebuilt in the parent from a child full_record.

    Numerically equivalent to DreamerVLAOnlineTrainEnv.make_transition for the OFT
    action_hidden rollout path (env action scale == wm_action scale).
    """
    done = bool(terminated or truncated)
    wm = np.asarray(wm_action, dtype=np.float32).reshape(-1)[:7]
    return {
        "image": dreamer_image_from_record(rec, image_size),
        "state": proprio_from_record(rec),
        "action": wm,
        "wm_action": wm,
        "obs_embedding": np.asarray(obs_embedding, dtype=np.float32),
        "reward": np.float32(reward),
        "done": np.float32(done),
        "discount": np.float32(0.0 if terminated else 1.0),
        "is_first": bool(is_first),
        "is_terminal": bool(terminated),
        "is_last": bool(done),
        "task_id": int(task_id),
        "step": int(step),
        "task_description": str(task_description),
    }
```

- [ ] **Step 4: Run, verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py tests/unit_tests/test_cotrain_vec_rollout.py
git commit -s -m "feat(cotrain): parent-side replay transition builder for vectorized rollout"
```

### Task 3: config knobs + early validation

**Files:**
- Modify: `configs/dreamervla/_base_wmpo_outcome.yaml`, `configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml`
- Modify: `dreamervla/runners/online_cotrain_runner.py` (validation in `_online_cotrain_loop`, near the
  `total_env_steps` read at `:511`)
- Test: `tests/unit_tests/test_cotrain_vec_rollout.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from dreamervla.runners.online_cotrain_runner import validate_rollout_cfg


def test_validate_rollout_cfg_rejects_bad_backend():
    with pytest.raises(ValueError, match="render_backend"):
        validate_rollout_cfg(num_envs=4, render_backend="vulkan", rollout_path="action_hidden")


def test_validate_rollout_cfg_rejects_multienv_backbone_latent():
    with pytest.raises(ValueError, match="backbone_latent"):
        validate_rollout_cfg(num_envs=4, render_backend="egl", rollout_path="backbone_latent")


def test_validate_rollout_cfg_accepts_singleenv_anything():
    validate_rollout_cfg(num_envs=1, render_backend="osmesa", rollout_path="backbone_latent")
```

- [ ] **Step 2: Run, verify it fails** — `… -k validate_rollout_cfg`; Expected FAIL (no `validate_rollout_cfg`).

- [ ] **Step 3: Implement**

```python
# dreamervla/runners/online_cotrain_runner.py (module level)
def validate_rollout_cfg(num_envs: int, render_backend: str, rollout_path: str) -> None:
    if num_envs < 1:
        raise ValueError(f"online_rollout.num_envs must be >=1, got {num_envs}")
    if num_envs > 1:
        if render_backend not in ("egl", "osmesa"):
            raise ValueError(
                f"online_rollout.render_backend must be 'egl' or 'osmesa' for num_envs>1, "
                f"got {render_backend!r}"
            )
        if rollout_path == "backbone_latent":
            raise ValueError(
                "vectorized rollout (num_envs>1) supports the OFT action_hidden path only; "
                "backbone_latent requires num_envs=1"
            )
```

Call it once in `_online_cotrain_loop` after reading the knobs:

```python
num_envs = int(OmegaConf.select(oc, "num_envs", default=1))
render_backend = str(OmegaConf.select(oc, "render_backend", default="egl"))
validate_rollout_cfg(num_envs, render_backend, self._rollout_path())   # _rollout_path() returns the selected path
```

- [ ] **Step 4: Run, verify it passes.** Then add config keys:

```yaml
# configs/dreamervla/_base_wmpo_outcome.yaml  (under online_rollout:)
  num_envs: 1
  render_backend: egl
```
```yaml
# configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml  (under online_rollout:)
  num_envs: 4        # parallel egl child envs; main throughput/GPU-mem knob — tune up if headroom
  render_backend: egl
```

- [ ] **Step 5: Commit**

```bash
git add configs/dreamervla/_base_wmpo_outcome.yaml configs/dreamervla/online_cotrain_pipeline_libero_goal.yaml dreamervla/runners/online_cotrain_runner.py tests/unit_tests/test_cotrain_vec_rollout.py
git commit -s -m "feat(cotrain): num_envs/render_backend knobs + early rollout validation"
```

### Task 4: vectorized continuous rollout loop (behavioral test with fakes)

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py` — add `_vectorized_cotrain_rollout(...)` (the loop
  in Design above) and the per-slot setup. Factor the existing burst block (`:584–686`) into a
  `self._maybe_train_burst(env_step, replay, …)` so both paths call it.
- Test: `tests/unit_tests/test_cotrain_vec_rollout.py`

- [ ] **Step 1: Write the failing behavioral test (CPU-only, fakes — no LIBERO/GPU)**

```python
class _FakeVec:
    """Stand-in for VecRolloutEnv: deterministic full_records, done after `horizon` steps."""
    def __init__(self, num_envs, horizon, full_record_fn):
        self.n, self.h, self._fr = num_envs, horizon, full_record_fn
        self._step = [0] * num_envs
    def reset(self, task_ids, episode_ids, env_ids=None):
        ids = range(self.n) if env_ids is None else list(env_ids)
        for k in ids: self._step[k] = 0
        return [self._fr() for _ in ids]
    def step(self, actions, env_ids=None):
        ids = list(env_ids) if env_ids is not None else range(self.n)
        out = []
        for k in ids:
            self._step[k] += 1
            term = self._step[k] >= self.h
            out.append((1.0, term, False, {}, self._fr()))
        return out

class _FakeExtractor:
    def __init__(self): self.resets = 0
    def reset(self): self.resets += 1
    def step(self, obs, desc):
        import torch
        return ([np.zeros(7, np.float32)], torch.zeros(8))   # (action_chunk, flat_hidden)
```

```python
def test_vectorized_rollout_isolated_queues_and_episode_grouping(monkeypatch):
    captured = []
    class _Replay:
        sequence_length = 2
        def add_episode(self, ep): captured.append(list(ep)); return None
    runner = _make_min_runner(monkeypatch)                 # helper: runner w/ device='cpu', task_ids=[0,1]
    vec = _FakeVec(num_envs=2, horizon=3, full_record_fn=_full_record)
    extractors = [_FakeExtractor(), _FakeExtractor()]
    runner._vectorized_cotrain_rollout(
        vec=vec, extractors=extractors, replay=_Replay(),
        num_envs=2, total_env_steps=12, episode_horizon=3,
        action_steps=1, image_size=64, task_ids=[0, 1],
    )
    # 12 env-steps / (2 envs × 3 horizon) = 2 episodes per slot = 4 episodes
    assert len(captured) == 4
    # every captured episode has exactly `horizon` transitions with required keys
    for ep in captured:
        assert len(ep) == 3
        assert {"image", "obs_embedding", "wm_action", "reward", "done", "is_terminal"} <= set(ep[0])
        assert float(ep[-1]["is_terminal"]) == 1.0 and all(float(s["is_terminal"]) == 0.0 for s in ep[:-1])
    # extractor.reset called once per finished episode per slot (2 each)
    assert extractors[0].resets == 2 and extractors[1].resets == 2
```

- [ ] **Step 2: Run, verify it fails** — `… -k vectorized_rollout`; Expected FAIL (`_vectorized_cotrain_rollout` missing).

- [ ] **Step 3: Implement `_vectorized_cotrain_rollout`** following the Design loop. Keep training-burst
  invocation behind an injected/overridable hook so the fake test (no WM/optimizer) skips it
  (e.g. `self._maybe_train_burst` is a no-op when `replay` reports not ready / when a test monkeypatches
  it). Use `extractor_obs_from_record`, `pop_open_loop_action`, `build_cotrain_replay_transition`.

- [ ] **Step 4: Run, verify it passes** — same as Step 2. Expected PASS.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py tests/unit_tests/test_cotrain_vec_rollout.py
git commit -s -m "feat(cotrain): vectorized continuous rollout loop (per-slot queues/extractors)"
```

### Task 5: wire `num_envs>1` branch (env_vars, per-slot extractors, VecRolloutEnv)

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py` — in `_online_cotrain_loop`, branch on `num_envs`.

- [ ] **Step 1: Implement the branch** (after validation, Task 3):

```python
if num_envs > 1:
    env_vars = {
        k: os.environ[k]
        for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "DVLA_DATA_ROOT", "LIBERO_CONFIG_PATH")
        if k in os.environ
    }
    env_vars["MUJOCO_GL"] = render_backend
    env_vars["PYOPENGL_PLATFORM"] = render_backend
    main_extractor = self._oft_action_hidden_extractor
    extractors = [main_extractor] + [
        self._build_action_hidden_extractor(policy)        # mirror collect_parallel_rollouts.py:284–294
        for _ in range(num_envs - 1)
    ]
    vec = VecRolloutEnv(num_envs=num_envs, cfg_kwargs=self._env_cfg_kwargs(cfg), env_vars=env_vars)
    with vec:
        return self._vectorized_cotrain_rollout(
            vec=vec, extractors=extractors, replay=replay, num_envs=num_envs,
            total_env_steps=total_env_steps, episode_horizon=episode_horizon,
            action_steps=action_steps, image_size=int(cfg.env.image_size), task_ids=env_task_ids,
        )
# else: existing single-env loop, unchanged
```

`_build_action_hidden_extractor` / `_env_cfg_kwargs` extract the existing extractor-construction and
env-config-dict logic so both the single-env builder and the vectorized branch share one source
(no duplicated extractor kwargs).

- [ ] **Step 2: Verify legacy path untouched** — run the existing cotrain runner unit test:

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_collect_rollouts_runner.py -q`
Expected: PASS (no behavior change for `num_envs==1`).

- [ ] **Step 3: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py
git commit -s -m "feat(cotrain): route num_envs>1 to vectorized egl rollout, keep single-env fallback"
```

### Task 6: suite + lint green

- [ ] **Step 1: Run the full unit suite in dreamervla**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests -q`
Expected: all green; new file's tests included; baseline (582) + new tests pass.

- [ ] **Step 2: Lint changed files**

Run: `conda run -n dreamervla ruff check dreamervla/runners/online_cotrain_runner.py dreamervla/runners/vectorized_collect.py tests/unit_tests/test_cotrain_vec_rollout.py`
Expected: no errors.

### Task 7: GPU smoke (egl multi-env)

- [ ] **Step 1: Run a tiny cotrain on a free GPU** (pick an idle GPU via `nvidia-smi`):

Run:
```bash
conda run -n dreamervla env CUDA_VISIBLE_DEVICES=<free> python -m dreamervla.train \
  experiment=online_cotrain_pipeline_libero_goal task=libero_goal \
  debug=true online_rollout.num_envs=4 online_rollout.render_backend=egl
```
Expected: **no SIGABRT**; K child envs start under egl; rollout reaches a training burst; run exits clean.

- [ ] **Step 2: Sanity-compare wall-clock** vs `online_rollout.num_envs=1` (osmesa default for ==1) on the
  same debug budget → egl multi-env should be materially faster per env-step. Record both numbers in the
  PR/commit message.

- [ ] **Step 3 (regression guard): confirm single-env still works**

Run: same as Step 1 but `online_rollout.num_envs=1`. Expected: legacy osmesa path runs, no crash.

### Task 8: commit isolation + merge to main

> The working tree currently mixes **three** uncommitted efforts on `main`: (a) the 2026-06-23 cotrain
> shared-rollout + osmesa SIGABRT fix (`online_cotrain_runner.py`, `vectorized_collect.py`,
> `oft_collect_common.py`, `collect_parallel_rollouts.py`, cotrain config); (b) MEM-RL-01 micro-batch
> WMPO (`grpo.py`, `outcome.py`, `_base_wmpo_outcome.yaml`, `test_wmpo_*`); (c) backlog docs. **Do not
> sweep all of these into one commit.**

- [ ] **Step 1:** Confirm the current tree's suite is green *before* layering Option 1
  (`conda run -n dreamervla python -m pytest tests/unit_tests -q`). If (a) is unverified, commit it as its
  own conventional commit first (shared-rollout + osmesa fix), so Option 1 layers on a clean base.
- [ ] **Step 2:** Leave MEM-RL-01 (b) changes **unstaged** — they belong to a separate commit/effort
  (see `docs/plans/2026-06-22-mem-rl-01-microbatch-wmpo.md`). Stage only Option-1 files per the per-task
  commits above.
- [ ] **Step 3:** With Tasks 1–7 green, push `main` (this repo's mainline is `main`). Commit subjects must
  pass the hooks: `--signoff`, conventional, no `===`/`/` in the subject.

---

## Risks / breaking assumptions

- **egl GPU memory on the training GPU.** K egl contexts + WM/policy/classifier + burst share one GPU.
  Mitigation: `num_envs=4` default; tune down on OOM. (Collection used envs_per_gpu up to 32 but with no
  training on the same GPU.)
- **Extractor sharing.** Slot 0 reuses the main extractor; slots ≥1 MUST be fresh (isolated history) and
  `.reset()` on every episode start — covered by Task 4's `resets` assertion.
- **`wm_action` scale.** Assumes OFT action_hidden env action == wm_action (env `action_input="raw"`,
  `policy_action_to_env_action` = clip-only, `train_env.py:319–334`). Task 2's parity test + Task 7's smoke
  guard this. If a future config uses `action_input="normalized"`, the builder needs the unnormalize step.
- **Replay fills ~K× faster** → burst cadence/readiness reached sooner; defaults unchanged here, flagged
  for tuning (`min_replay`, `train_every`).
- **Distributed (DDP).** If cotrain runs multi-rank, each rank spawns its own K children; the global
  readiness reduce (`get_replay_task_stats_global`) is unchanged. Smoke is single-rank; multi-rank is
  out of scope for this plan.

## Out of scope (separate plans)

- **Option 2 — RLinf async pipeline (Ray backend):** GPU-infer ∥ CPU-step ∥ learner overlap, env actors,
  the known OFT-on-Ray blockers (online policy is RynnVLA, obs-dim alignment). Separate plan after Option 1
  merges.
- **`backbone_latent` vectorization** (per-slot WM latent) — asserted `num_envs==1` here.
- **Multi-rank DDP** vectorized rollout tuning.

## Verification quick-reference (dreamervla env)

```bash
# unit (fast, CPU)
conda run -n dreamervla python -m pytest tests/unit_tests/test_cotrain_vec_rollout.py -q
conda run -n dreamervla python -m pytest tests/unit_tests/test_collect_rollouts_runner.py -q
# full suite + lint
conda run -n dreamervla python -m pytest tests/unit_tests -q
conda run -n dreamervla ruff check dreamervla/runners/online_cotrain_runner.py dreamervla/runners/vectorized_collect.py
# GPU smoke (egl multi-env)
conda run -n dreamervla env CUDA_VISIBLE_DEVICES=<free> python -m dreamervla.train \
  experiment=online_cotrain_pipeline_libero_goal task=libero_goal \
  debug=true online_rollout.num_envs=4 online_rollout.render_backend=egl
```
