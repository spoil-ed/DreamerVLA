# RLinf-Alignment Worker Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve DreamerVLA's Ray workers from a tightly-coupled research prototype toward RLinf's "one responsibility per worker" design — without rewriting it into RLinf — by adopting RLinf's three engineering principles: **环境隔离 (env isolation)**, **职责拆分 (responsibility separation)**, **资源显式调度 (explicit resource scheduling)**.

**Architecture:** The work is staged in 6 phases (env isolation → swappable reward/verifier interfaces → WM-rollout extraction → version management → multi-worker scaling → multi-learner/FSDP). Each phase produces working, tested software on its own. This document gives **Phase 1 as DONE**, **Phase 2 in full TDD detail** (the actionable next step), and **Phases 3–6 as design specs** that each become their own plan when their predecessor lands.

**Tech Stack:** Python 3.11, PyTorch (bf16 AMP), Ray actors (`dreamervla/scheduler/`), Hydra config groups, `multiprocessing` spawn (egl isolation), pytest. Run everything in the `dreamervla` conda env (`conda run -n dreamervla …`).

---

## Architectural Assessment (read before planning the work)

The generic RL framing — "split `RewardWorker` and `CriticWorker` out of `LearnerWorker`" — does **not** map 1:1 onto DreamerVLA, and the plan corrects for it. Verified against the code:

1. **Reward and value are NOT inside `LearnerWorker`.** `LearnerWorker.update()` (`dreamervla/workers/actor/learner_worker.py:94-104`) dispatches to a registered **algorithm route**; the RL step `_dreamervla_rl_update_once` (`:230-262`) calls `dino_wmpo_outcome_step`. Reward (`_build_reward_tensor`, `dreamervla/algorithms/ppo/outcome.py:95-127`) and the value/success signal (`classifier.predict_success`, invoked inside `_imagine_and_score_slice`, `outcome.py:286-294`) already live in the **algorithms layer**, not the learner.

2. **There is no separate critic network on the outcome route.** DreamerVLA's `V(e_t)=P(future success)` **is** the `LatentSuccessClassifier` (`dreamervla/models/reward/latent_success_classifier.py`). A scalar critic (`TDMPCCritic`) exists *only* on the dense route (`registry.py:110`, `uses_critic=True`). The classifier is **light** and sits on the **imagination critical path** (imagine chunk → classify → reward → GRPO advantage, per RL update).

3. **Therefore: extract INTERFACES first, host in separate Ray WORKERS later.** Spinning up separate `RewardWorker`/`CriticWorker` *processes* now would add cross-process latency to the per-update imagination loop and buy no scaling (the classifier is not the bottleneck; CPU env physics is). The bottleneck-justified moment for a *process* split is when the verifier becomes a heavy separate model (e.g. a VLM verifier) — that is **Phase 5**.

This matches the original analysis's own wording: Step 2 says extract Reward/Critic as **逻辑接口 (logical interfaces)**, and lists swap targets (MLP / transformer / two-hot / ensemble critic; real / classifier / VLM / dense reward). Phase 2 below delivers exactly that — swappable interfaces, default numerics unchanged — and defers process-splitting to Phase 5.

---

## Roadmap (6 phases)

| Phase | Theme | Principle | Status | Deliverable |
|------|-------|-----------|--------|-------------|
| 1 | EnvWorker spawn isolation (egl crash fix) | env isolation | **DONE** | this doc §Phase 1 |
| 2 | Swappable Reward + Verifier interfaces | responsibility separation | **THIS PLAN** (TDD) | this doc §Phase 2 |
| 3 | Extract WM imagined-rollout interface | responsibility separation | design spec → own plan | this doc §Phase 3 |
| 4 | Model/policy version + staleness management | explicit scheduling | design spec → own plan | this doc §Phase 4 |
| 5 | Multi-EnvWorker + multi-PolicyWorker + hosted Reward/Verifier workers | explicit scheduling | design spec → own plan | this doc §Phase 5 |
| 6 | Multi-Learner / FSDP / Megatron | explicit scheduling | design spec → own plan | this doc §Phase 6 |

**Sequencing rule:** do not start a phase until its predecessor is merged and verified. Phases 3–6 get their own bite-sized plans (one per subsystem) authored from their design spec here, at the time they start.

---

## Phase 1 — EnvWorker spawn isolation (init FIXED; sustained-render resilience IN PROGRESS)

**Goal:** Eliminate the robosuite `read_pixels` SIGABRT that hit in-Ray-actor egl rendering at multi-worker concurrency, and make 16-worker init survive the cold-start storm.

**What shipped** (`dreamervla/workers/env/env_worker.py`, branch `feat/cotrain-smoke-and-multigpu-h100`):
- Each egl env runs in a **clean `multiprocessing.spawn` subprocess** (`_env_subprocess_main`, `:39-116`; `_init_spawn`, `:164-`) so its GL context initializes in a fresh interpreter — RLinf's per-env spawn-venv pattern. The osmesa / synthetic path stays in-process (`_init_inproc`).
- **Cold-start-storm fix** (this session): `WorkerGroup.launch` fires all 16 `init.remote()` at once, so 16 children cold-start LIBERO+robosuite+egl simultaneously. The fix de-peaks the storm + uses a generous, configurable init timeout:
  - per-rank stagger `time.sleep(local_rank * egl_spawn_stagger_s)` (default 3s) so later ranks ride the warmed page cache,
  - `egl_spawn_init_timeout_s` (default 900s, was a hardcoded 300s).

**Verification — what the init fix achieved (GPUs 4-7, `env.num_workers=16`):** 16 env workers built with **no init timeout**; GPUs 4-7 at **100%** util; rollout ran live egl rendering. Committed as `1f42a2e`.

**Verification — what a FULL run revealed (NOT done):** a `rollout.steps=600` run reached **step ~86/600** and then a spawn **child died silently** → parent `EOFError` in `_step_spawn`'s `_rpc` recv (3 actors hit it as the failure cascaded; driver exited). Root-cause evidence: **no** stderr in any Ray EnvWorker `.err` (mujoco `mju_error` would print → this is **not** the read_pixels mju_error abort), **no** apport report / core (apport ignores conda-python), **not** OOM (box has 2TB RAM / 128 cores). ⇒ a **silent native SIGSEGV** in the render path under sustained 16-way concurrent egl on 4 shared GPUs. **Spawn isolation fixed INIT fragility, not sustained-concurrency rendering.**

**Conclusion:** the `_init_spawn` timeout/stagger fix is correct and merged-ready on its own, but Phase 1 cannot be called done until the rollout **survives** an occasional egl child crash. That is Phase 1b below. (The deeper structural fix — fewer Ray actors each owning a `SubprocVecEnv` of N envs, which also reduces per-GPU egl context churn — is Phase 5.)

---

### Phase 1b: egl child-crash resilience (respawn + drop episode)

**Goal:** When a spawn env-child dies mid-rollout (silent native crash → `EOFError`/`OSError` on the pipe), the `EnvWorker` **recovers** — drop the partial episode, respawn a clean child, return an episode-boundary `done` — instead of propagating and killing the whole job. RLinf's env workers auto-recover from flaky env subprocesses; this is the same primitive. A per-worker respawn cap prevents infinite thrash on a persistently-crashing env.

> Caveat to verify empirically: if crashes are frequent, constant respawns (cold start ~60-120s each) will hurt throughput — in which case the real remedy is Phase 5 (lower per-GPU egl concurrency via `SubprocVecEnv`). Phase 1b makes the run *survive*; Phase 5 makes it *fast*.

**Files:**
- Modify: `dreamervla/workers/env/env_worker.py` (`__init__`, `init`/`_init_spawn` store the egl device id + respawn state; `_step_spawn` catch crash; new `_recover_from_child_death`)
- Test: `tests/unit_tests/test_env_worker_spawn_recovery.py`

- [ ] **Step 1: Write the failing test** (logic-level — no real subprocess; a stub `_conn` raises `EOFError`, `_init_spawn` is monkeypatched to set a fresh obs)

```python
# tests/unit_tests/test_env_worker_spawn_recovery.py
import numpy as np

from dreamervla.workers.env.env_worker import EnvWorker


class _DeadConn:
    def send(self, *_):  # send may succeed; recv reports the dead child
        pass

    def recv(self):
        raise EOFError


class _Proc:
    def is_alive(self):
        return False

    def terminate(self):
        pass


def _worker():
    w = EnvWorker(
        env_cfg={"egl_device_pool": [0], "egl_max_respawns": 2},
        task_id=0,
        replay=None,
    )
    w.local_rank = 0
    w._egl_device_id = 0
    w._proc = _Proc()
    w._conn = _DeadConn()
    w.obs = {"x": np.zeros(1)}
    w.episode = [{"dummy": 1}]
    return w


def test_step_recovers_on_child_death(monkeypatch):
    w = _worker()
    spawned = {"n": 0}

    def _fake_init_spawn(egl_device_id):
        spawned["n"] += 1
        w.obs = {"fresh": np.ones(1)}

    monkeypatch.setattr(w, "_init_spawn", _fake_init_spawn)

    obs, done, info = w.step(action=np.zeros(7), obs_embedding=np.zeros(4))
    assert done is True
    assert info.get("env_crash_recovered") is True
    assert spawned["n"] == 1            # respawned once
    assert w.episode == []              # partial episode dropped
    assert obs == {"fresh": np.ones(1).tolist()} or obs["fresh"].tolist() == [1.0]


def test_respawn_cap_eventually_raises(monkeypatch):
    w = _worker()
    monkeypatch.setattr(w, "_init_spawn", lambda egl_device_id: None)

    # egl_max_respawns=2 → 2 recoveries ok, 3rd raises
    w.step(action=np.zeros(7), obs_embedding=np.zeros(4))
    w._conn = _DeadConn()
    w.step(action=np.zeros(7), obs_embedding=np.zeros(4))
    w._conn = _DeadConn()
    try:
        w.step(action=np.zeros(7), obs_embedding=np.zeros(4))
    except RuntimeError as exc:
        assert "egl child died" in str(exc)
    else:
        raise AssertionError("expected RuntimeError after exceeding egl_max_respawns")
```

- [ ] **Step 2: Run test to verify it fails** — `conda run -n dreamervla python -m pytest tests/unit_tests/test_env_worker_spawn_recovery.py -q` → FAIL (no recovery; `EOFError` propagates).

- [ ] **Step 3: Implement** — in `env_worker.py`: store `self._egl_device_id`, `self._respawn_count = 0`, `self._max_respawns = int(env_cfg.get("egl_max_respawns", 5))` in `__init__`; have `_init_spawn` set `self._egl_device_id`; wrap the `_rpc` call in `_step_spawn` with `try/except (EOFError, OSError)` → `return self._recover_from_child_death()`; add `_recover_from_child_death` (increment + cap-check → raise; else drop episode, terminate dead proc, `_init_spawn(self._egl_device_id)`, bump `episode_id`, return `(self.obs, True, {"env_crash_recovered": True})`).

- [ ] **Step 4: Run test to verify it passes** — same command → PASS (2 passed).

- [ ] **Step 5: GPU re-run** — `experiment=online_cotrain_ray_oft_action_hidden … env.num_workers=16 rollout.steps=150` on GPUs 4-7. Expected: the run **survives** past step ~86 (a `env_crash_recovered` warning may appear) and reaches `[ray-cotrain] FINAL METRICS:` with `rollout/episodes>0` — no `EOFError` job kill.

- [ ] **Step 6: Commit** — `git commit --signoff -m "feat: respawn egl env child on crash so rollout survives a single env segfault"`.

---

**Remaining before Phase 1 merge:** Phase 1b green (unit + GPU survive-the-crash); keep osmesa path working (it never enters `_step_spawn`).

---

## Phase 2 — Swappable Reward + Verifier interfaces

**Goal:** Make the **reward definition** and the **success-verifier** (`V(e_t)=P(success)`) selectable via config behind small protocols + a registry mirroring `dreamervla/algorithms/registry.py`, with the **default numerics bit-for-bit unchanged** (default reward = `sparse_outcome` wrapping today's `_build_reward_tensor`; default verifier = `LatentSuccessClassifier`).

**Why this is the right Phase 2:** see Architectural Assessment. It delivers the user's "拆出逻辑接口" + swap targets, is fully in-process (no latency cost), changes no defaults, and is the foundation Phases 3 & 5 build on.

### File Structure

- `dreamervla/algorithms/reward/__init__.py` — re-export registry API; import `sparse_outcome` so the default registers on import.
- `dreamervla/algorithms/reward/protocol.py` — `RewardModel` Protocol.
- `dreamervla/algorithms/reward/registry.py` — `register_reward_model` / `get_reward_model` / `reward_model_names` (clone of the actor-update registry's shape).
- `dreamervla/algorithms/reward/sparse_outcome.py` — `SparseOutcomeReward` delegating to `_build_reward_tensor`.
- `dreamervla/algorithms/verifier/__init__.py` — re-export the protocol.
- `dreamervla/algorithms/verifier/protocol.py` — `SuccessVerifier` Protocol (typing contract; selection stays via the classifier component's Hydra `_target_`, so **no registry needed**).
- Modify `dreamervla/algorithms/ppo/outcome.py` — `dino_wmpo_outcome_step` resolves the reward model from `algorithm_cfg.wmpo.reward_model` (default `"sparse_outcome"`) via a **function-local** import (breaks the import cycle, since `sparse_outcome` imports `_build_reward_tensor` from this module).
- Tests: `tests/unit_tests/test_reward_registry.py`, `tests/unit_tests/test_reward_sparse_outcome.py`, `tests/unit_tests/test_success_verifier_protocol.py`.

> Run all test commands with `conda run -n dreamervla python -m pytest …` from repo root `/mnt/data/spoil/workspace/DreamerVLA`.

---

### Task 1: Reward model protocol

**Files:**
- Create: `dreamervla/algorithms/reward/protocol.py`
- Test: `tests/unit_tests/test_reward_registry.py` (created here, extended in Task 2)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/test_reward_registry.py
import torch

from dreamervla.algorithms.reward.protocol import RewardModel


def test_reward_model_protocol_runtime_checkable():
    class _Stub:
        name = "stub"

        def build_reward(self, *, batch, max_steps, chunk_size, finish_step, complete, device):
            return torch.zeros((batch, max_steps), device=device)

    assert isinstance(_Stub(), RewardModel)

    class _NotAModel:
        name = "x"

    assert not isinstance(_NotAModel(), RewardModel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_reward_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: dreamervla.algorithms.reward`

- [ ] **Step 3: Write minimal implementation**

```python
# dreamervla/algorithms/reward/protocol.py
"""Protocol for swappable WMPO reward definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RewardModel(Protocol):
    """Maps an imagined rollout's success outcome to a per-step reward tensor.

    The verifier emits ``(complete, finish_step)``; a ``RewardModel`` turns that
    into the ``[batch, max_steps]`` reward the WMPO advantage consumes. The default
    sparse-outcome form places ``float(complete)`` at ``finish_step``; dense /
    verifier-shaped forms may return a per-step signal instead.
    """

    name: str

    def build_reward(
        self,
        *,
        batch: int,
        max_steps: int,
        chunk_size: int,
        finish_step: torch.Tensor,
        complete: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a ``[batch, max_steps]`` float32 reward tensor on ``device``."""
        ...
```

Also create an empty package marker so the import resolves:

```python
# dreamervla/algorithms/reward/__init__.py
"""Swappable WMPO reward definitions (protocol + registry)."""

from dreamervla.algorithms.reward.protocol import RewardModel

__all__ = ["RewardModel"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_reward_registry.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add dreamervla/algorithms/reward/__init__.py dreamervla/algorithms/reward/protocol.py tests/unit_tests/test_reward_registry.py
git commit --signoff -m "feat: add RewardModel protocol for swappable WMPO reward"
```

---

### Task 2: Reward registry

**Files:**
- Create: `dreamervla/algorithms/reward/registry.py`
- Test: `tests/unit_tests/test_reward_registry.py` (extend)

- [ ] **Step 1: Write the failing test (append to the file from Task 1)**

```python
def test_register_and_get_roundtrip():
    from dreamervla.algorithms.reward.registry import (
        get_reward_model,
        register_reward_model,
        reward_model_names,
    )

    class _Stub:
        name = "stub_route"

        def build_reward(self, *, batch, max_steps, chunk_size, finish_step, complete, device):
            return torch.zeros((batch, max_steps), device=device)

    stub = _Stub()
    register_reward_model(stub, aliases=("stub_alias",))
    assert get_reward_model("stub_route") is stub
    assert get_reward_model("STUB-ALIAS") is stub  # normalised lookup
    assert "stub_route" in reward_model_names()


def test_get_unknown_raises():
    from dreamervla.algorithms.reward.registry import get_reward_model

    try:
        get_reward_model("does_not_exist")
    except ValueError as exc:
        assert "Unknown reward model" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_reward_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: …reward.registry`

- [ ] **Step 3: Write minimal implementation**

```python
# dreamervla/algorithms/reward/registry.py
"""Registry for swappable WMPO reward models (mirrors the actor-update registry)."""

from __future__ import annotations

from collections.abc import Iterable

from dreamervla.algorithms.reward.protocol import RewardModel

_REWARD_MODELS: dict[str, RewardModel] = {}


def _normalise_name(name: str) -> str:
    normalised = name.strip().lower().replace("-", "_")
    if not normalised:
        raise ValueError("Reward model name must be non-empty.")
    return normalised


def register_reward_model(model: RewardModel, *, aliases: Iterable[str] = ()) -> RewardModel:
    """Register a reward model and aliases."""

    keys = [_normalise_name(model.name), *(_normalise_name(a) for a in aliases)]
    for key in keys:
        existing = _REWARD_MODELS.get(key)
        if existing is not None and existing is not model:
            raise ValueError(
                f"Reward model `{key}` is already registered to `{existing.name}`."
            )
    for key in keys:
        _REWARD_MODELS[key] = model
    return model


def get_reward_model(name: str) -> RewardModel:
    """Return a registered reward model by canonical name or alias."""

    key = _normalise_name(name)
    try:
        return _REWARD_MODELS[key]
    except KeyError as exc:
        known = ", ".join(reward_model_names())
        raise ValueError(
            f"Unknown reward model `{name}`. Available reward models: {known}."
        ) from exc


def reward_model_names() -> tuple[str, ...]:
    """Return canonical registered reward-model names."""

    return tuple(sorted({m.name for m in _REWARD_MODELS.values()}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_reward_registry.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dreamervla/algorithms/reward/registry.py tests/unit_tests/test_reward_registry.py
git commit --signoff -m "feat: add reward-model registry"
```

---

### Task 3: SparseOutcomeReward (default, numerics-preserving)

**Files:**
- Create: `dreamervla/algorithms/reward/sparse_outcome.py`
- Modify: `dreamervla/algorithms/reward/__init__.py` (import to auto-register)
- Test: `tests/unit_tests/test_reward_sparse_outcome.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/test_reward_sparse_outcome.py
import torch

from dreamervla.algorithms.ppo.outcome import _build_reward_tensor
from dreamervla.algorithms.reward import get_reward_model


def test_sparse_outcome_matches_build_reward_tensor_bitforbit():
    batch, max_steps, K = 5, 12, 3
    finish_step = torch.tensor([0, 4, 11, 7, 3])
    complete = torch.tensor([True, False, True, True, False])

    model = get_reward_model("sparse_outcome")
    got = model.build_reward(
        batch=batch,
        max_steps=max_steps,
        chunk_size=K,
        finish_step=finish_step,
        complete=complete,
        device=torch.device("cpu"),
    )
    expected = _build_reward_tensor(
        batch=batch,
        max_steps=max_steps,
        chunk_size=K,
        finish_step=finish_step,
        complete=complete,
    )
    assert torch.equal(got, expected)
    # sparse 0/1: one positive per complete row at its finish column
    assert got.sum().item() == 3.0
    assert got[0, 0].item() == 1.0 and got[1].sum().item() == 0.0


def test_sparse_outcome_aliases_resolve():
    assert get_reward_model("outcome").name == "sparse_outcome"
    assert get_reward_model("sparse").name == "sparse_outcome"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_reward_sparse_outcome.py -q`
Expected: FAIL — `ValueError: Unknown reward model \`sparse_outcome\``

- [ ] **Step 3: Write minimal implementation**

```python
# dreamervla/algorithms/reward/sparse_outcome.py
"""Default sparse outcome reward: float(complete) at finish_step, else 0."""

from __future__ import annotations

import torch

from dreamervla.algorithms.ppo.outcome import _build_reward_tensor
from dreamervla.algorithms.reward.registry import register_reward_model


class SparseOutcomeReward:
    """Wraps the canonical ``_build_reward_tensor`` so the default WMPO numerics are
    bit-for-bit unchanged; exists so the reward DEFINITION is selectable via
    ``algorithm.wmpo.reward_model`` alongside future dense / verifier-shaped forms.
    """

    name = "sparse_outcome"

    def build_reward(
        self,
        *,
        batch: int,
        max_steps: int,
        chunk_size: int,
        finish_step: torch.Tensor,
        complete: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        return _build_reward_tensor(
            batch=batch,
            max_steps=max_steps,
            chunk_size=chunk_size,
            finish_step=finish_step,
            complete=complete,
        ).to(device)


register_reward_model(SparseOutcomeReward(), aliases=("outcome", "sparse"))
```

Wire auto-registration in the package init:

```python
# dreamervla/algorithms/reward/__init__.py
"""Swappable WMPO reward definitions (protocol + registry)."""

from dreamervla.algorithms.reward.protocol import RewardModel
from dreamervla.algorithms.reward.registry import (
    get_reward_model,
    register_reward_model,
    reward_model_names,
)

# Import implementations for their registration side effect.
from dreamervla.algorithms.reward import sparse_outcome as _sparse_outcome  # noqa: F401

__all__ = [
    "RewardModel",
    "get_reward_model",
    "register_reward_model",
    "reward_model_names",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_reward_sparse_outcome.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dreamervla/algorithms/reward/sparse_outcome.py dreamervla/algorithms/reward/__init__.py tests/unit_tests/test_reward_sparse_outcome.py
git commit --signoff -m "feat: SparseOutcomeReward default reward model (numerics-preserving)"
```

---

### Task 4: Route the outcome step through the reward registry

**Files:**
- Modify: `dreamervla/algorithms/ppo/outcome.py:521-527` (the `_build_reward_tensor(...)` call inside `dino_wmpo_outcome_step`)
- Test: `tests/unit_tests/test_outcome_reward_model_selection.py`

- [ ] **Step 1: Write the failing test**

This test drives the step's reward selection without running the full WM/policy: it monkeypatches the registry lookup to assert the step requests the configured model and uses its output. (Keep it light — no model construction.)

```python
# tests/unit_tests/test_outcome_reward_model_selection.py
import torch

import dreamervla.algorithms.reward as reward_pkg


def test_outcome_step_resolves_reward_model_from_cfg(monkeypatch):
    seen = {}

    class _Spy:
        name = "spy"

        def build_reward(self, *, batch, max_steps, chunk_size, finish_step, complete, device):
            seen["called_with"] = (batch, max_steps, chunk_size)
            return torch.zeros((batch, max_steps), device=device)

    def _fake_get(name):
        seen["name"] = name
        return _Spy()

    monkeypatch.setattr(reward_pkg, "get_reward_model", _fake_get)

    # The helper under test: a thin reward-resolution seam extracted from the step,
    # exercised directly so we don't need a real WM/policy/classifier here.
    from dreamervla.algorithms.ppo.outcome import _resolve_reward_tensor

    finish_step = torch.tensor([0, 1])
    complete = torch.tensor([True, False])
    out = _resolve_reward_tensor(
        wmpo_cfg={"reward_model": "spy"},
        batch=2,
        max_steps=4,
        chunk_size=2,
        finish_step=finish_step,
        complete=complete,
        device=torch.device("cpu"),
    )
    assert seen["name"] == "spy"
    assert seen["called_with"] == (2, 4, 2)
    assert out.shape == (2, 4)


def test_outcome_default_reward_model_is_sparse_outcome(monkeypatch):
    captured = {}
    real_get = reward_pkg.get_reward_model

    def _capturing_get(name):
        captured["name"] = name
        return real_get(name)

    monkeypatch.setattr(reward_pkg, "get_reward_model", _capturing_get)

    from dreamervla.algorithms.ppo.outcome import _resolve_reward_tensor

    _resolve_reward_tensor(
        wmpo_cfg={},
        batch=2,
        max_steps=4,
        chunk_size=2,
        finish_step=torch.tensor([0, 1]),
        complete=torch.tensor([True, False]),
        device=torch.device("cpu"),
    )
    assert captured["name"] == "sparse_outcome"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_outcome_reward_model_selection.py -q`
Expected: FAIL — `ImportError: cannot import name '_resolve_reward_tensor'`

- [ ] **Step 3: Write minimal implementation**

Add the resolution seam to `outcome.py` (function-local import of the package breaks the `sparse_outcome → outcome` import cycle), then call it from `dino_wmpo_outcome_step`.

Add near the other module-level helpers in `dreamervla/algorithms/ppo/outcome.py` (e.g. after `_build_reward_tensor`, ~line 128):

```python
def _resolve_reward_tensor(
    *,
    wmpo_cfg: Any,
    batch: int,
    max_steps: int,
    chunk_size: int,
    finish_step: torch.Tensor,
    complete: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build the reward via the config-selected reward model (default sparse_outcome).

    Imported lazily: ``dreamervla.algorithms.reward.sparse_outcome`` imports
    ``_build_reward_tensor`` from THIS module, so a top-level import would cycle.
    """
    import dreamervla.algorithms.reward as reward_pkg

    name = str(wmpo_cfg.get("reward_model", "sparse_outcome"))
    model = reward_pkg.get_reward_model(name)
    return model.build_reward(
        batch=batch,
        max_steps=max_steps,
        chunk_size=chunk_size,
        finish_step=finish_step,
        complete=complete,
        device=device,
    )
```

Then replace the hardcoded call in `dino_wmpo_outcome_step` (`outcome.py:521-527`):

```python
    reward_tensor = _resolve_reward_tensor(
        wmpo_cfg=wmpo_cfg,
        batch=B_eff,
        max_steps=T_max,
        chunk_size=K,
        finish_step=finish_step,
        complete=complete,
        device=device,
    )
```

> Note: the test monkeypatches `reward_pkg.get_reward_model`, so `_resolve_reward_tensor` must reference it as `reward_pkg.get_reward_model` (attribute lookup at call time), **not** `from … import get_reward_model`. The code above does this correctly.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_outcome_reward_model_selection.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the broader algorithm + learner suite to confirm no regression**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_algorithm_registry.py tests/e2e_tests/test_s4_learner_worker.py -q`
Expected: PASS (unchanged counts)

- [ ] **Step 6: Commit**

```bash
git add dreamervla/algorithms/ppo/outcome.py tests/unit_tests/test_outcome_reward_model_selection.py
git commit --signoff -m "feat: select WMPO reward via algorithm.wmpo.reward_model (default unchanged)"
```

---

### Task 5: SuccessVerifier protocol (the value/critic contract)

**Files:**
- Create: `dreamervla/algorithms/verifier/protocol.py`
- Create: `dreamervla/algorithms/verifier/__init__.py`
- Test: `tests/unit_tests/test_success_verifier_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/test_success_verifier_protocol.py
import torch

from dreamervla.algorithms.verifier import SuccessVerifier


def test_stub_satisfies_verifier_protocol():
    class _Stub:
        def predict_success(self, latent_video, *, threshold, stride=1, min_steps=1, **kwargs):
            b = latent_video.shape[0]
            return {
                "complete": torch.zeros(b, dtype=torch.bool),
                "finish_step": torch.zeros(b, dtype=torch.long),
            }

    assert isinstance(_Stub(), SuccessVerifier)


def test_latent_success_classifier_declares_predict_success():
    # Contract smoke test: the default verifier exposes the method the WMPO loop
    # calls (outcome.py:286). We assert the attribute exists without constructing
    # the (heavyweight) model so the test stays a fast unit test.
    from dreamervla.models.reward.latent_success_classifier import (
        LatentSuccessClassifier,
    )

    assert hasattr(LatentSuccessClassifier, "predict_success")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_success_verifier_protocol.py -q`
Expected: FAIL — `ModuleNotFoundError: dreamervla.algorithms.verifier`

- [ ] **Step 3: Write minimal implementation**

```python
# dreamervla/algorithms/verifier/protocol.py
"""Protocol for the WMPO success verifier — DreamerVLA's value source."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class SuccessVerifier(Protocol):
    """Scores an imagined latent video and returns per-rollout success.

    This is DreamerVLA's ``V(e_t)=P(future success)``. ``LatentSuccessClassifier``
    satisfies it today; an MLP / transformer / two-hot / ensemble / calibrated
    critic can be swapped in via the ``classifier`` component's Hydra ``_target_``
    as long as it implements this method with this return contract.
    """

    def predict_success(
        self,
        latent_video: torch.Tensor,
        *,
        threshold: float,
        stride: int = 1,
        min_steps: int = 1,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Return ``{"complete": [B] bool, "finish_step": [B] long}``."""
        ...
```

```python
# dreamervla/algorithms/verifier/__init__.py
"""WMPO success-verifier contract (the value source)."""

from dreamervla.algorithms.verifier.protocol import SuccessVerifier

__all__ = ["SuccessVerifier"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_success_verifier_protocol.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add dreamervla/algorithms/verifier/ tests/unit_tests/test_success_verifier_protocol.py
git commit --signoff -m "feat: add SuccessVerifier protocol (WMPO value-source contract)"
```

---

### Task 6: Phase-2 regression gate + docs note

**Files:**
- Modify: `AGENTS.md` (or `docs/HISTORY.md`) — one-line pointer that reward is selectable via `algorithm.wmpo.reward_model` and the verifier contract is `SuccessVerifier`.

- [ ] **Step 1: Run the full unit suite + a ray cotrain smoke**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests -q`
Expected: PASS (baseline count + the new tests; no new failures vs the clean baseline of 582 passed in the `dreamervla` env).

Run: `conda run -n dreamervla python -m pytest tests/e2e_tests/test_scheduler_ray_smoke.py -q`
Expected: PASS.

- [ ] **Step 2: Lint changed files**

Run: `conda run -n dreamervla ruff check dreamervla/algorithms/reward dreamervla/algorithms/verifier dreamervla/algorithms/ppo/outcome.py`
Expected: no errors.

- [ ] **Step 3: Add the docs pointer + commit**

Add to `AGENTS.md` near the algorithms/registry guidance: "Reward definition is selectable via `algorithm.wmpo.reward_model` (registry: `dreamervla/algorithms/reward/`); the success verifier (value source) must satisfy `dreamervla.algorithms.verifier.SuccessVerifier`."

```bash
git add AGENTS.md
git commit --signoff -m "docs: note reward_model selector + SuccessVerifier contract"
```

---

## Phase 3 — Extract the WM imagined-rollout interface (design spec → own plan)

**Goal:** Give the world-model imagination a named interface so the rollout-and-score step is a swappable component, decoupling "how we imagine + score a trajectory" from "how we turn it into a PPO loss."

**Current coupling:** `_imagine_and_score_slice` (`outcome.py:154-311`) interleaves (a) WM chunk rollout, (b) policy sampling + ref-KL, (c) classifier scoring, and (d) host-buffer layout for the multi-epoch PPO re-eval. It returns PPO-shaped buffers (`actor_feats`, `old_log_probs`, `ref_kls`, `complete`, `finish_step`), not a clean trajectory.

**Approach:** Define an `ImaginedRollout` dataclass (latents/actions/old_log_probs/ref_kls + verifier outputs) and an `Imaginer` protocol `imagine(current, policy, world_model, verifier, cfg) -> ImaginedRollout`. Keep `_imagine_and_score_slice` as the default `WMPOImaginer` implementation. The MEM-RL-01 slice/micro-batch logic stays inside it.

**Why deferred:** higher blast radius (touches the PPO re-eval contract + numerics) and depends on Phase 2's verifier protocol existing. Author its own TDD plan once Phase 2 merges. **Verification target:** golden-value test that the refactored imaginer yields bit-identical `complete`/`finish_step`/`old_log_probs` on a fixed seed vs. the pre-refactor path.

---

## Phase 4 — Model/policy version + staleness management (design spec → own plan)

**Goal:** Make off-policy staleness explicit and bounded, instead of best-effort weight pulls.

**Current state:** `policy_version` is incremented in `online_cotrain_ray_runner.py` (`:291/318/370`), pushed via `ObjectStoreWeightSyncer.push(key, state, version)` and pulled via `pull(key, model, local_version)` (`dreamervla/hybrid_engines/weight_syncer/objectstore.py`) only when `version > local_version`. There is **no staleness bound** and **no per-trajectory version stamp** — a rollout may mix actions from several policy versions with no record.

**Approach (RLinf off-policy correctness layer — the deferred "Stage 2"):** stamp each collected transition with the inference worker's `policy_version`; expose `staleness_threshold` so the learner can drop/down-weight trajectories older than N versions; optionally add off-policy log-prob interpolation (`alpha`) at update time. Wire counters into the `time/` + `rollout/` metric namespaces.

**Why deferred:** correctness-sensitive; needs Phase 2/3 interfaces to attach reward/verifier versioning cleanly. **Verification target:** unit test that stale trajectories beyond `staleness_threshold` are dropped; metric `rollout/policy_version_lag` reported.

---

## Phase 5 — Multi-EnvWorker + multi-PolicyWorker + hosted Reward/Verifier workers (design spec → own plan)

**Goal:** Scale rollout throughput and, **only when justified**, host the reward/verifier behind their own Ray workers.

**Two independent pieces:**
1. **Rollout scaling (the throughput win):** the egl16 run is CPU-physics-bound (~0.1 step/s). Adopt RLinf's structure — fewer Ray workers each owning a `SubprocVecEnv` of N spawn-isolated envs — instead of 16 Ray actors × 1 spawn child. This is the proper successor to Phase 1's per-actor spawn and the real fix for env throughput. Add multiple `PolicyWorker`/inference replicas behind the existing `inference.worker_target` seam.
2. **Hosted Reward/VerifierWorker (only if heavy):** wrap the Phase 2 `RewardModel` / `SuccessVerifier` in `Worker` subclasses with their own `placement`, following the `RolloutInferenceWorker` pattern + `ObjectStoreWeightSyncer`. **Trigger:** a heavy verifier (e.g. VLM) that would contend with inference/learner on the shared GPU pool. For the light `LatentSuccessClassifier`, keep it in-process (Phase 2) — process-splitting it only adds latency.

**Why deferred:** largest scope; the rollout-scaling half is independently valuable and should likely be its own plan ahead of the hosted-worker half. **Verification target:** throughput (step/s) up materially vs egl16 baseline at equal GPU budget; FINAL METRICS unchanged in distribution.

---

## Phase 6 — Multi-Learner / FSDP / Megatron (design spec → own plan)

**Goal:** Scale the learner past one GPU when the learner becomes heavy (large-model RL).

**Approach:** the YAML already documents the **disaggregate** alternative (`configs/dreamervla/ray_online_cotrain_oft_action_hidden.yaml:159-161`): give the learner its own GPU range + `num_gpus_per_worker>1` (FSDP). Phase 6 implements multi-learner data/model parallelism behind `learner.placement`. Megatron is a separate backend behind an explicit Hydra experiment per the RLinf Alignment Snapshot in `CLAUDE.md`.

**Why deferred (and last):** only pays off once the learner is the bottleneck, which it is not today (`time/learner_wait_s≈3.6s` vs `time/env_step_wait_s≈1405s`). **Verification target:** learner step time scales with learner GPUs; outcome metrics unchanged.

---

## Self-Review

**Spec coverage:** the original analysis's 6 steps map to Phases 1–6 one-to-one; its 3 principles head the roadmap; its "swap targets" (MLP/transformer/ensemble critic; real/classifier/VLM/dense reward) are the `SuccessVerifier` + `RewardModel` extension points in Phase 2/5. The "改造1 env 隔离" = Phase 1 (done); "改造2 critic"/"改造3 reward" = Phase 2 interfaces + Phase 5 hosting.

**Placeholder scan:** Phase 2 tasks contain complete code + exact run commands + expected output; Phases 3–6 are explicitly marked design-spec → own plan (not bite-sized tasks), with a concrete verification target each.

**Type consistency:** `RewardModel.build_reward(*, batch, max_steps, chunk_size, finish_step, complete, device)` is identical across protocol/registry/impl/seam/tests; `_resolve_reward_tensor` keyword args match `dino_wmpo_outcome_step`'s locals (`B_eff`, `T_max`, `K`, `finish_step`, `complete`, `device`); `SuccessVerifier.predict_success` signature matches the call at `outcome.py:286-292` (`threshold`, `stride`, `min_steps`, `pre_pooled` via `**kwargs`).

**Correction surfaced:** the "split into separate workers now" framing is replaced with "extract interfaces now (Phase 2), host as workers when heavy (Phase 5)" — with the bottleneck evidence (`learner_wait≈3.6s` vs `env_step_wait≈1405s`) justifying the ordering.
