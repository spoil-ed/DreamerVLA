# Pipeline Progress Reporter + Cotrain Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one uniform, RLinf-style progress line to every pipeline loop (training, cotrain, collect, eval, preprocess) — replacing all tqdm — and give `online_cotrain` real resume via the inherited base checkpoint machinery.

**Architecture:** A pure formatter `format_progress_line` in `dreamervla/utils/console.py`, a stateful `ProgressReporter` (injected clock/sink, wall-time throttle) in a new `dreamervla/utils/progress.py`, and a `BaseRunner.console_progress(...)` hook that caches one reporter per `desc` in the existing `_console_state`. Cotrain resume retires the bespoke `_save_cotrain_ckpt()` and uses the inherited `save_checkpoint`/`load_checkpoint`/`resume`, capturing optimizer state and calling `self.resume()` at `run()` start.

**Tech Stack:** Python, OmegaConf/Hydra, pytest. Runs in the `dreamervla` conda env: `conda run -n dreamervla python -m pytest tests/unit_tests -q`.

**Spec:** `docs/plans/2026-06-21-pipeline-progress-and-cotrain-resume.md`

**Builds on:** the console family from `docs/plans/2026-06-20-train-console-*` — `console_banner`/`console_record_success`/`console_metrics` on `BaseRunner`, backed by `dreamervla/utils/console.py`. `console_progress` is the next member of that family.

**Commit rules (this repo):** every commit uses `git commit --signoff`; commit descriptions must not contain `===` or `/`; ruff runs on changed Python (no unused imports / trailing whitespace). Append the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- **Modify** `dreamervla/utils/console.py` — add pure `format_progress_line` + `_fmt_duration` (rendering only, no I/O).
- **Create** `dreamervla/utils/progress.py` — stateful `ProgressReporter` (timing, throttle, sink); depends only on the formatter.
- **Modify** `dreamervla/runners/base_runner.py` — add `console_progress`, a `"progress"` slot + `progress_every_s` in `_console_state_get`, and close cached reporters in `teardown`.
- **Modify** `dreamervla/runners/online_cotrain_runner.py` — R1 resume: `include_keys`/`exclude_keys`, `_save_checkpoint_sidecars` override, retire `_save_cotrain_ckpt` torch path, call `self.resume()`.
- **Modify** the loop runners + standalone preprocess scripts (Task 6 map) — call `console_progress` / use `ProgressReporter`; delete tqdm.
- **Create/extend** `tests/unit_tests/test_console.py`, new `tests/unit_tests/test_progress.py`, `tests/unit_tests/test_base_runner_console.py`, new `tests/unit_tests/test_cotrain_resume.py`.

---

## Task 1: Pure `format_progress_line` + `_fmt_duration`

**Files:**
- Modify: `dreamervla/utils/console.py`
- Test: `tests/unit_tests/test_console.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit_tests/test_console.py`)

```python
from dreamervla.utils.console import format_progress_line


def test_format_progress_line_with_total():
    s = format_progress_line(
        "pretokenize", 12800, 50000, elapsed_s=201.0, eta_s=585.0, rate=63.7
    )
    assert s == "pretokenize 12800/50000 (26%) · 03:21<09:45 · 63.7 it/s"


def test_format_progress_line_open_ended():
    s = format_progress_line(
        "collect", 812, None, elapsed_s=201.0, eta_s=None, rate=4.0, unit="ep"
    )
    assert s == "collect 812 · 03:21 · 4.0 ep/s"


def test_format_progress_line_hour_duration_and_zero_total():
    s = format_progress_line("train", 0, 0, elapsed_s=3725.0, eta_s=0.0, rate=0.0)
    # total<=0 is treated as open-ended (no pct/eta); duration rolls to h:mm:ss
    assert s == "train 0 · 1:02:05 · 0.0 it/s"
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_console.py -k format_progress_line -q`
Expected: FAIL — `ImportError: cannot import name 'format_progress_line'`.

- [ ] **Step 3: Implement** (append to `dreamervla/utils/console.py`)

```python
def _fmt_duration(seconds: float) -> str:
    """Format a duration as mm:ss, or h:mm:ss past an hour."""
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def format_progress_line(
    desc: str,
    current: int,
    total: int | None,
    *,
    elapsed_s: float,
    eta_s: float | None,
    rate: float,
    unit: str = "it",
) -> str:
    """RLinf-style one-line progress string.

    total>0  -> "desc cur/total (pct%) · elapsed<eta · rate unit/s"
    total in (None, <=0) -> "desc cur · elapsed · rate unit/s" (open-ended).
    """
    head = f"{desc} {current}"
    if total and total > 0:
        pct = int(round(100.0 * current / total))
        timing = f"{_fmt_duration(elapsed_s)}<{_fmt_duration(eta_s or 0.0)}"
        head = f"{desc} {current}/{total} ({pct}%)"
    else:
        timing = _fmt_duration(elapsed_s)
    return f"{head} · {timing} · {rate:.1f} {unit}/s"
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_console.py -q`
Expected: PASS (new + existing console tests).

- [ ] **Step 5: Commit**

```bash
git add dreamervla/utils/console.py tests/unit_tests/test_console.py
git commit --signoff -m "feat(console): add pure format_progress_line renderer"
```

---

## Task 2: Stateful `ProgressReporter`

**Files:**
- Create: `dreamervla/utils/progress.py`
- Test: `tests/unit_tests/test_progress.py`

- [ ] **Step 1: Write the failing test** (`tests/unit_tests/test_progress.py`)

```python
from dreamervla.utils.progress import ProgressReporter


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _reporter(total=100, **kw):
    clk = _Clock()
    out = []
    r = ProgressReporter(
        total, "train", clock=clk, sink=out.append, min_interval_s=5.0, **kw
    )
    return r, clk, out


def test_first_update_prints_then_throttled_by_walltime():
    r, clk, out = _reporter()
    r.update()                 # first tick always prints
    assert len(out) == 1 and out[0].startswith("train 1/100")
    clk.t += 2.0
    r.update()                 # 2s < 5s -> suppressed
    assert len(out) == 1
    clk.t += 4.0
    r.update()                 # 6s since last print -> prints
    assert len(out) == 2 and out[1].startswith("train 3/100")


def test_close_always_prints_final_summary():
    r, clk, out = _reporter()
    r.update()                 # prints (first)
    clk.t += 1.0
    r.set(100)                 # throttled
    r.close()                  # always prints final
    assert out[-1].startswith("train 100/100")


def test_disabled_is_silent():
    out = []
    r = ProgressReporter(10, "x", enabled=False, sink=out.append, clock=_Clock())
    r.update(); r.set(5); r.close()
    assert out == []


def test_open_ended_total_none_has_no_pct():
    clk = _Clock()
    out = []
    r = ProgressReporter(None, "collect", clock=clk, sink=out.append, unit="ep")
    r.update()
    assert out[0].startswith("collect 1 ·") and "%" not in out[0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_progress.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreamervla.utils.progress'`.

- [ ] **Step 3: Implement** (`dreamervla/utils/progress.py`)

```python
"""Stateful, wall-time-throttled progress reporter.

One uniform progress line for every pipeline loop. No tqdm: prints plain lines
via an injected sink so output is clean in log files, nohup, and Ray worker
logs. clock/sink are injectable for deterministic tests.
"""

from __future__ import annotations

import time
from typing import Callable

from dreamervla.utils.console import format_progress_line


class ProgressReporter:
    def __init__(
        self,
        total: int | None,
        desc: str,
        *,
        enabled: bool = True,
        min_interval_s: float = 5.0,
        unit: str = "it",
        clock: Callable[[], float] = time.monotonic,
        sink: Callable[[str], None] = print,
    ) -> None:
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.min_interval_s = float(min_interval_s)
        self.unit = unit
        self._clock = clock
        self._sink = sink
        self._current = 0
        self._start_t = clock()
        self._last_print_t: float | None = None

    def update(self, n: int = 1) -> None:
        self.set(self._current + n)

    def set(self, current: int) -> None:
        self._current = int(current)
        if not self.enabled:
            return
        now = self._clock()
        if self._last_print_t is None or (now - self._last_print_t) >= self.min_interval_s:
            self._emit(now)

    def close(self) -> None:
        if not self.enabled:
            return
        self._emit(self._clock())

    def _emit(self, now: float) -> None:
        elapsed = max(1e-9, now - self._start_t)
        rate = self._current / elapsed
        eta = None
        if self.total and self.total > 0 and rate > 0:
            eta = max(0.0, (self.total - self._current) / rate)
        self._sink(
            format_progress_line(
                self.desc,
                self._current,
                self.total,
                elapsed_s=elapsed,
                eta_s=eta,
                rate=rate,
                unit=self.unit,
            )
        )
        self._last_print_t = now

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_progress.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add dreamervla/utils/progress.py tests/unit_tests/test_progress.py
git commit --signoff -m "feat(progress): add wall-time-throttled ProgressReporter"
```

---

## Task 3: `BaseRunner.console_progress` + config knob

**Files:**
- Modify: `dreamervla/runners/base_runner.py` (`_console_state_get` add `"progress"` + `progress_every_s`; new `console_progress`; close reporters in `teardown`)
- Test: `tests/unit_tests/test_base_runner_console.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit_tests/test_base_runner_console.py`)

```python
def _progress_runner(cfg, *, main=True):
    obj = types.SimpleNamespace()
    obj.cfg = cfg
    obj.is_main_process = main
    for name in ("console_progress", "_console_state_get"):
        setattr(obj, name, types.MethodType(getattr(BaseRunner, name), obj))
    return obj


def test_console_progress_prints_and_caches_per_desc(capsys):
    cfg = OmegaConf.create({"console": {"progress_every_s": 0.0}})
    r = _progress_runner(cfg)
    r.console_progress(1, 10, "train")
    r.console_progress(2, 10, "train")
    out = capsys.readouterr().out
    assert "train 1/10" in out and "train 2/10" in out
    # one cached reporter per desc
    assert set(r._console_state["progress"].keys()) == {"train"}


def test_console_progress_guarded_on_non_main(capsys):
    cfg = OmegaConf.create({"console": {"progress_every_s": 0.0}})
    _progress_runner(cfg, main=False).console_progress(1, 10, "train")
    assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_base_runner_console.py -k progress -q`
Expected: FAIL — `AttributeError: ... 'console_progress'` / KeyError `"progress"`.

- [ ] **Step 3: Implement** in `dreamervla/runners/base_runner.py`

Add the import near the existing console import:
```python
from dreamervla.utils.progress import ProgressReporter
```

In `_console_state_get`, add two entries to the `st = {...}` dict:
```python
                "progress_every_s": float(
                    OmegaConf.select(self.cfg, "console.progress_every_s", default=5.0)
                ),
                "progress": {},
```

Add the method (next to `console_metrics`):
```python
    def console_progress(self, current: int, total: int | None, desc: str, *, unit: str = "it") -> None:
        st = self._console_state_get()
        reporters = st["progress"]
        rep = reporters.get(desc)
        if rep is None:
            rep = ProgressReporter(
                total,
                desc,
                enabled=self.is_main_process,
                min_interval_s=st["progress_every_s"],
                unit=unit,
            )
            reporters[desc] = rep
        rep.set(current)
```

In `teardown`, close any cached reporters before `finish_metric_logger()`:
```python
    def teardown(self) -> None:
        """Optional lifecycle hook after execution."""
        st = getattr(self, "_console_state", None)
        if st is not None:
            for rep in st.get("progress", {}).values():
                rep.close()
        self.finish_metric_logger()
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_base_runner_console.py tests/unit_tests/test_console.py tests/unit_tests/test_progress.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/base_runner.py tests/unit_tests/test_base_runner_console.py
git commit --signoff -m "feat(base_runner): add console_progress hook and progress_every_s knob"
```

---

## Task 4: Cotrain resume (R1 — reuse base machinery)

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py`
- Test: `tests/unit_tests/test_cotrain_resume.py`

**Read-time step (unavoidable, like the prior console plan's per-runner step):**
List `OnlineCotrainRunner`'s instance attributes that expose `state_dict`
(modules + optimizers + any frozen/aux modules such as `encoder`, `ref_policy`,
`_unwrapped_world_model`). The four trainable modules + four `*_optimizer`
attributes must be **kept**; frozen/auxiliary modules must go into `exclude_keys`.
Confirm whether `DreamerVLARunner` already overrides
`_state_dict_for_checkpoint` for DDP unwrap; if not, add the override here.

- [ ] **Step 1: Write the failing test** (`tests/unit_tests/test_cotrain_resume.py`)

```python
import torch
from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner


class _Mini(BaseRunner):
    """Minimal runner exercising base save/load round-trip with optimizer + scalars."""
    include_keys = ("global_step", "classifier_threshold")
    exclude_keys = ("frozen",)

    def __init__(self, cfg, tmp):
        self.cfg = cfg
        self._out = tmp
        self.global_step = 0
        self.classifier_threshold = 0.5
        self.policy = torch.nn.Linear(3, 2)
        self.frozen = torch.nn.Linear(3, 2)  # must NOT be checkpointed
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def get_checkpoint_path(self, tag="latest", *, prefer_existing=False):
        import pathlib
        return pathlib.Path(self._out) / f"{tag}.ckpt"

    def run(self):  # abstract
        return None


def test_base_checkpoint_roundtrips_optimizer_and_scalars(tmp_path):
    cfg = OmegaConf.create({})
    a = _Mini(cfg, tmp_path)
    # take an optimizer step so momentum buffers are non-empty
    loss = a.policy(torch.ones(1, 3)).sum()
    loss.backward(); a.policy_optimizer.step()
    a.global_step = 7
    a.classifier_threshold = 0.73
    path = a.save_checkpoint()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "policy" in payload["state_dicts"]
    assert "policy_optimizer" in payload["state_dicts"]
    assert "frozen" not in payload["state_dicts"]   # exclude_keys honored

    b = _Mini(cfg, tmp_path)
    b.load_checkpoint(path=path)
    assert b.global_step == 7
    assert abs(b.classifier_threshold - 0.73) < 1e-9
    assert b.policy_optimizer.state_dict()["state"]  # momentum restored
```

- [ ] **Step 2: Run to verify it fails / proves the contract**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_cotrain_resume.py -q`
Expected: this test passes against the *existing* base machinery once
`include_keys`/`exclude_keys` are set on the subclass — it pins the behavior R1
relies on. If it fails, the base capture semantics differ and Task 4 must adapt.

- [ ] **Step 3: Implement R1 in `online_cotrain_runner.py`**

1. Class-level (top of `OnlineCotrainRunner`):
```python
    include_keys = ("global_step", "classifier_threshold")
    exclude_keys = (<frozen/aux attrs from the read-time step, e.g. "encoder", "ref_policy", "_unwrapped_world_model">)
```
2. Override the HF sidecar hook (move the `checkpoint_save_hf()` block out of
   `_save_cotrain_ckpt`):
```python
    def _save_checkpoint_sidecars(self, path, payload):
        if not self.checkpoint_save_hf():
            return
        ckpt_dir = path.parent
        for name, module, cfg_key in (
            ("world_model", self.world_model, "world_model"),
            ("policy", self.policy, "policy"),
            ("critic", self.critic, "critic"),
        ):
            blk = OmegaConf.to_container(OmegaConf.select(self.cfg, cfg_key), resolve=True)
            target = blk.pop("_target_")
            if name == "policy":
                blk.pop("init_action_head_ckpt", None)
            save_module_pretrained(_unwrap(module), str(ckpt_dir / f"{path.stem}_hf_{name}"),
                                   target=target, init_args=blk)
        save_module_pretrained(_unwrap(self.classifier), str(ckpt_dir / f"{path.stem}_hf_classifier"),
                               target="dreamervla.models.reward.latent_success_classifier.LatentSuccessClassifier",
                               init_args=getattr(self, "_classifier_cls_kwargs", {}))
```
3. Replace the body of `_save_cotrain_ckpt` with a call to the inherited saver:
```python
    def _save_cotrain_ckpt(self) -> None:
        path = self.save_checkpoint()            # torch payload + HF sidecars via hook
        print(f"[online-cotrain] ckpt -> {path}", flush=True)
```
4. If DDP unwrap is needed (read-time step), add:
```python
    def _state_dict_for_checkpoint(self, key, value):
        return _unwrap(value).state_dict()
    def _load_state_dict_from_checkpoint(self, key, value, state_dict, **kwargs):
        _unwrap(value).load_state_dict(state_dict, **kwargs)
```
5. In `run()`, after `self._build_components(cfg)` and before `_online_cotrain_loop`, call:
```python
        self.resume()
```

- [ ] **Step 4: Verify**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_cotrain_resume.py tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -q`
Run: `conda run -n dreamervla python -m py_compile dreamervla/runners/online_cotrain_runner.py`
Grep: `grep -n "torch.save" dreamervla/runners/online_cotrain_runner.py` → no raw torch.save in `_save_cotrain_ckpt`.
Expected: tests pass; py_compile clean.

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py tests/unit_tests/test_cotrain_resume.py
git commit --signoff -m "feat(cotrain): real resume via base save, load, resume with optimizer state"
```

---

## Task 5: Wire cotrain loop progress

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py` (env-step loop ~`:635`)

- [ ] **Step 1:** In `_online_cotrain_loop`, just before/after `console_metrics`, add the uniform progress line:
```python
                self.console_progress(env_step, total_env_steps, "cotrain", unit="env")
```
Keep the existing `console_metrics` box (different cadence/purpose).

- [ ] **Step 2: Verify**

Run: `conda run -n dreamervla python -m pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py
git commit --signoff -m "feat(cotrain): uniform progress line over env steps"
```

---

## Task 6: Wire all remaining flows + delete tqdm (shared recipe)

**Recipe per loop:**
1. Read the loop region (entry method + loop line from the map).
2. Add one call: runner loops → `self.console_progress(current, total, desc, unit=...)`; standalone scripts (no runner) → wrap the loop in `with ProgressReporter(total, desc, unit=...) as pbar:` and call `pbar.update()` / `pbar.set(...)`.
3. Delete any tqdm import + `tqdm(...)` wrapping in that file; the `ProgressReporter` replaces it (no mixed schemes).
4. `total` from the map; open-ended loops pass `total=None`.
5. Verify: `conda run -n dreamervla python -m py_compile <file>`; run that file's tests if any; `grep -n "tqdm" <file>` → none.
6. Commit `--signoff` (plain description).

| Sub | File | Loop | total | tqdm? |
| --- | --- | --- | --- | --- |
| 6a | `dreamerv3_pixel_runner.py` | epoch loop @~526 / inner @577 | num_epochs · steps | yes → remove |
| 6b | `dreamerv3_token_runner.py` | epoch loop @~432 | num_epochs | no |
| 6c | `dreamervla_runner.py` | epoch loop @~1684 | num_epochs | no |
| 6d | `latent_wm_runner.py` | train loop | max steps | check |
| 6e | `backbone_dreamerv3_wm_runner.py` | train loop | max steps | check |
| 6f | `pretokenize_vla_runner.py` (covers VLASFT) | epoch/train @~792 | num_epochs / len(loader) | yes → remove |
| 6g | `embodied_eval_runner.py` | episode loop @~1918 | n_episodes | no |
| 6h | `rlinf_libero_rollout.py` | rollout loop | episode/task count | check |
| 6i | `collect_parallel_rollouts.py` | rollout loop | target_episodes | no |
| 6j | `vectorized_collect.py` | rollout loop | target_episodes | no |
| 6k | `cold_start_ray_collect_runner.py` | while loop @293 | target_episodes | yes → remove |
| 6l | `collect_online_rollouts_for_classifier.py` | rollout loop | target_episodes | check |
| 6m | `collect_rollouts_runner.py` | rollout loop @~80 | target_episodes | check |
| 6n | `preprocess_oft_action_hidden.py` | item loop | len(items) | yes → remove |
| 6o | `preprocess_rynn_pixel_hidden.py` | item loop | len(items) | yes → remove |
| 6p | `preprocess_remaining_steps_reward.py` | item loop | len(items) | yes → remove |
| 6q | `pre_tokenize_action_local.py` / `pre_tokenize_action_state_local.py` | item loop | len(items) | yes → remove |
| 6r | libero regen scripts (`preprocess/libero_utils/regenerate_*`) | demo loop | len(demos) | yes → remove |

Per-loop exact diffs are deferred to read-time by necessity (each loop differs); the recipe + map fixes the contract. One commit per sub-task or per small group.

---

## Task 7: Full suite + cotrain smoke

- [ ] **Step 1:** `conda run -n dreamervla python -m pytest tests/unit_tests -q` → all green.
- [ ] **Step 2:** Run the cotrain smoke profile a few steps → confirm a `checkpoints/latest.ckpt` appears, then relaunch with `training.resume=true` and confirm `global_step` continues (log line "Resuming from checkpoint …") and a progress line prints.
- [ ] **Step 3:** `grep -rn "from tqdm" dreamervla | grep -v test` → only intentionally-kept (ideally none in pipeline loops).
- [ ] **Step 4: Commit** any smoke-config touch-ups `--signoff`.

---

## Self-Review

- **Spec coverage:** Part 1 progress core → Tasks 1–3; integration map (every loop) → Tasks 5–6; Part 2 cotrain resume R1 → Task 4; testing → Tasks 1–4 unit + Task 7 smoke. ✓
- **Placeholder scan:** Tasks 1–3 carry literal code + tests. Task 4 has one explicit read-time step (exact `exclude_keys` / DDP-override decision) — unavoidable and flagged, mirroring the prior console plan. Task 6 is recipe+map pattern application (each loop differs) — flagged. ✓
- **Type consistency:** `format_progress_line(desc,current,total,*,elapsed_s,eta_s,rate,unit)`, `ProgressReporter(total,desc,*,enabled,min_interval_s,unit,clock,sink).update/set/close`, `BaseRunner.console_progress(current,total,desc,*,unit)`, `console.progress_every_s`, `include_keys/exclude_keys` — used identically across tasks. ✓
- **Testability:** core is fully unit-tested with injected clock/sink; resume pinned by `test_cotrain_resume`; loop wiring (GPU/LIBERO) verified via py_compile + existing tests + grep, same inherent gap as the prior plans.
