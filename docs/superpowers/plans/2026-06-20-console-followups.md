# Console API Follow-up Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `force` param to `console_metrics`, add `console_success_rate()` helper, tidy reach-ins in three runner files, and add four unit tests.

**Architecture:** All base changes go in `dreamervla/runners/base_runner.py`. Three runner files get surgical edits (reach-in removal + `force=True`). Tests go in the existing `tests/unit_tests/test_base_runner_console.py` file, reusing the `_runner` helper.

**Tech Stack:** Python 3.10+, pytest, omegaconf

## Global Constraints

- Branch: `feat/console-followups` — do NOT switch or create another branch.
- `git commit --signoff` required on every commit.
- Commit message must be: `feat(console): add force flag and console_success_rate helper; tidy reach-ins`
- Conventional-commit hook rejects `===` and `/` in the DESCRIPTION — the message above is already safe.
- ruff: no unused imports, no unused locals, no trailing whitespace.
- Only touch lines that trace to the spec. Do not improve adjacent code.

---

### Task 1: Add `force` param to `console_metrics` and `console_success_rate()` to BaseRunner

**Files:**
- Modify: `dreamervla/runners/base_runner.py:974-991` (`console_metrics`) and insert `console_success_rate` after it

**Interfaces:**
- Produces: `console_metrics(self, header: str, metrics: dict, *, force: bool = False) -> None`
- Produces: `console_success_rate(self) -> float`

- [ ] **Step 1: Write failing tests first (in test file)**

Add to `tests/unit_tests/test_base_runner_console.py` (append after the last existing test):

```python
def test_console_metrics_force_bypasses_throttle(capsys):
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 5, "success_window": 4}})
    r = _runner(cfg)
    r.console_metrics("step 1", {"train/loss": 0.5}, force=True)  # call #1, log_every=5 → should print
    out = capsys.readouterr().out
    assert out.strip() != "", "force=True must print regardless of throttle"
    r.console_metrics("step 2", {"train/loss": 0.4})  # call #2, not a multiple of 5 → no print
    assert capsys.readouterr().out == ""


def test_console_success_rate_zero_before_any_record():
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 1, "success_window": 4}})
    r = _runner(cfg)
    assert r.console_success_rate() == 0.0


def test_console_success_rate_after_records():
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 1, "success_window": 4}})
    r = _runner(cfg)
    for s in (True, True, False, True):  # 3 successes / 4 total → rate = 0.75
        r.console_record_success(s)
    assert abs(r.console_success_rate() - 0.75) < 1e-9


def test_console_record_success_non_main():
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 1, "success_window": 4}})
    r = _runner(cfg, main=False)
    for s in (True, True, False):
        r.console_record_success(s)
    assert abs(r.console_success_rate() - (2 / 3)) < 1e-9


def test_group_metric_rows_skip_success_drops_rollout_success_rate():
    from dreamervla.runners.base_runner import _group_metric_rows
    rows = _group_metric_rows(
        {"rollout/success_rate": 0.5, "train/loss": 0.1},
        skip_success=True,
    )
    joined = "\n".join(rows)
    assert "success_rate" not in joined
    assert "loss=0.1" in joined
```

- [ ] **Step 2: Extend the `_runner` helper to bind `console_success_rate`**

In `tests/unit_tests/test_base_runner_console.py`, change line 12–13:

```python
    for name in ("console_banner", "console_record_success", "console_metrics", "_console_state_get"):
        setattr(obj, name, types.MethodType(getattr(BaseRunner, name), obj))
```

to:

```python
    for name in ("console_banner", "console_record_success", "console_metrics",
                 "_console_state_get", "console_success_rate"):
        setattr(obj, name, types.MethodType(getattr(BaseRunner, name), obj))
```

- [ ] **Step 3: Run tests to confirm they fail (AttributeError / TypeError expected)**

```bash
cd /mnt/data/spoil/workspace/DreamerVLA && python -m pytest tests/unit_tests/test_base_runner_console.py -v 2>&1 | tail -20
```

Expected: failures on `test_console_metrics_force_bypasses_throttle` (unexpected keyword `force`), `test_console_success_rate_*` (AttributeError on `console_success_rate`).

- [ ] **Step 4: Implement `force` param in `console_metrics` in `base_runner.py`**

Current signature at line 974:
```python
    def console_metrics(self, header: str, metrics: dict) -> None:
```

Change to:
```python
    def console_metrics(self, header: str, metrics: dict, *, force: bool = False) -> None:
```

Current throttle check at line 979:
```python
        if st["counter"] % st["log_every"] != 0:
            return
```

Change to:
```python
        if not force and st["counter"] % st["log_every"] != 0:
            return
```

- [ ] **Step 5: Add `console_success_rate()` to `base_runner.py` immediately after `console_metrics`**

Insert after line 991 (after `if tr is not None: tr.mark_printed()`):

```python
    def console_success_rate(self) -> float:
        st = self._console_state_get()
        tr = st["tracker"]
        return tr.rate() if tr is not None else 0.0
```

- [ ] **Step 6: Run tests to confirm all pass**

```bash
cd /mnt/data/spoil/workspace/DreamerVLA && python -m pytest tests/unit_tests/test_base_runner_console.py tests/unit_tests/test_console.py tests/unit_tests/test_base_runner_config_gate.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

---

### Task 2: Tidy reach-ins in `cold_start_ray_collect_runner.py` and add `force=True`

**Files:**
- Modify: `dreamervla/runners/cold_start_ray_collect_runner.py` — two sites (~lines 343-359 and ~535-551)

**Interfaces:**
- Consumes: `self.console_success_rate() -> float` (from Task 1)
- Consumes: `self.console_metrics(..., force=True)` (from Task 1)

- [ ] **Step 1: Replace reach-in at site 1 (~lines 343-345)**

Current code:
```python
        _st = self._console_state_get()
        _tr = _st["tracker"]
        succ_rate = _tr.rate() if _tr is not None and len(_tr) > 0 else 0.0
```

Replace with:
```python
        succ_rate = self.console_success_rate()
```

- [ ] **Step 2: Add `force=True` to `console_metrics` at site 1 (~line 351)**

Current code:
```python
        self.console_metrics(
            "collect",
            {
                "collect/episodes": episodes,
                "collect/steps": int(steps),
                "collect/success_rate": succ_rate,
                "env/num_env_workers": int(num_envs),
            },
        )
```

Change to:
```python
        self.console_metrics(
            "collect",
            {
                "collect/episodes": episodes,
                "collect/steps": int(steps),
                "collect/success_rate": succ_rate,
                "env/num_env_workers": int(num_envs),
            },
            force=True,
        )
```

- [ ] **Step 3: Replace reach-in at site 2 (~lines 535-537)**

Current code:
```python
        _st = self._console_state_get()
        _tr = _st["tracker"]
        succ_rate = _tr.rate() if _tr is not None and len(_tr) > 0 else 0.0
```

Replace with:
```python
        succ_rate = self.console_success_rate()
```

- [ ] **Step 4: Add `force=True` to `console_metrics` at site 2 (~line 543)**

Current code:
```python
        self.console_metrics(
            "collect",
            {
                "collect/episodes": episodes,
                "collect/steps": int(steps),
                "collect/success_rate": succ_rate,
                "env/num_env_workers": int(num_envs),
            },
        )
```

Change to:
```python
        self.console_metrics(
            "collect",
            {
                "collect/episodes": episodes,
                "collect/steps": int(steps),
                "collect/success_rate": succ_rate,
                "env/num_env_workers": int(num_envs),
            },
            force=True,
        )
```

- [ ] **Step 5: Verify no reach-ins remain and py_compile passes**

```bash
cd /mnt/data/spoil/workspace/DreamerVLA && grep -n "_console_state_get\|\[.tracker.\]" dreamervla/runners/cold_start_ray_collect_runner.py
python -m py_compile dreamervla/runners/cold_start_ray_collect_runner.py && echo "OK"
```

Expected: `grep` returns nothing. `py_compile` exits 0.

---

### Task 3: Add `force=True` to summary `console_metrics` in `collect_rollouts_runner.py` and `embodied_eval_runner.py`

**Files:**
- Modify: `dreamervla/runners/collect_rollouts_runner.py` (~line 95)
- Modify: `dreamervla/runners/embodied_eval_runner.py` (~lines 145 and 385)

**Interfaces:**
- Consumes: `self.console_metrics(..., force=True)` (from Task 1)

- [ ] **Step 1: Add `force=True` in `collect_rollouts_runner.py` (~line 95)**

Current code:
```python
        self.console_metrics(
            "collect",
            {
                "collect/episodes": len(successes),
                "collect/success_rate": succ_rate,
            },
        )
```

Change to:
```python
        self.console_metrics(
            "collect",
            {
                "collect/episodes": len(successes),
                "collect/success_rate": succ_rate,
            },
            force=True,
        )
```

- [ ] **Step 2: Add `force=True` to first eval summary in `embodied_eval_runner.py` (~line 145)**

Current code:
```python
        self.console_metrics(
            "eval",
            {
                "eval/success_rate": eval_rate,
                "eval/episodes": float(metrics.get("eval_total_episodes", 0.0)),
                "eval/successes": float(metrics.get("eval_total_successes", 0.0)),
            },
        )
```

Change to:
```python
        self.console_metrics(
            "eval",
            {
                "eval/success_rate": eval_rate,
                "eval/episodes": float(metrics.get("eval_total_episodes", 0.0)),
                "eval/successes": float(metrics.get("eval_total_successes", 0.0)),
            },
            force=True,
        )
```

- [ ] **Step 3: Add `force=True` to second eval summary in `embodied_eval_runner.py` (~line 385)**

Current code:
```python
        self.console_metrics(
            "eval",
            {
                "eval/success_rate": dreamer_eval_rate,
                "eval/episodes": float(metrics.get("eval_total_episodes", 0.0)),
                "eval/successes": float(metrics.get("eval_total_successes", 0.0)),
            },
        )
```

Change to:
```python
        self.console_metrics(
            "eval",
            {
                "eval/success_rate": dreamer_eval_rate,
                "eval/episodes": float(metrics.get("eval_total_episodes", 0.0)),
                "eval/successes": float(metrics.get("eval_total_successes", 0.0)),
            },
            force=True,
        )
```

- [ ] **Step 4: Verify py_compile and grep for 5 force flags total**

```bash
cd /mnt/data/spoil/workspace/DreamerVLA && python -m py_compile dreamervla/runners/base_runner.py dreamervla/runners/cold_start_ray_collect_runner.py dreamervla/runners/collect_rollouts_runner.py dreamervla/runners/embodied_eval_runner.py && echo "py_compile OK"
grep -n "force=True" dreamervla/runners/collect_rollouts_runner.py dreamervla/runners/cold_start_ray_collect_runner.py dreamervla/runners/embodied_eval_runner.py
```

Expected: `py_compile OK`, and `grep` shows exactly 5 lines (1 in collect_rollouts, 2 in cold_start_ray, 2 in embodied_eval).

---

### Task 4: Final verification + write report + commit

**Files:**
- Create: `.superpowers/sdd/followups-report.md`

- [ ] **Step 1: Run full test suite**

```bash
cd /mnt/data/spoil/workspace/DreamerVLA && python -m pytest tests/unit_tests/test_base_runner_console.py tests/unit_tests/test_console.py tests/unit_tests/test_base_runner_config_gate.py -v 2>&1
```

Expected: all tests pass.

- [ ] **Step 2: Run all verifications from spec**

```bash
cd /mnt/data/spoil/workspace/DreamerVLA
python -m py_compile dreamervla/runners/base_runner.py dreamervla/runners/cold_start_ray_collect_runner.py dreamervla/runners/collect_rollouts_runner.py dreamervla/runners/embodied_eval_runner.py && echo "py_compile OK"
grep -n "_console_state_get\|\[.tracker.\]" dreamervla/runners/cold_start_ray_collect_runner.py
grep -n "force=True" dreamervla/runners/collect_rollouts_runner.py dreamervla/runners/cold_start_ray_collect_runner.py dreamervla/runners/embodied_eval_runner.py
```

Expected:
- `py_compile OK`
- reach-in grep returns nothing
- force grep shows 5 lines

- [ ] **Step 3: Write report to `.superpowers/sdd/followups-report.md`**

Create the directory and file with full findings.

- [ ] **Step 4: Commit**

```bash
cd /mnt/data/spoil/workspace/DreamerVLA
git add dreamervla/runners/base_runner.py \
        dreamervla/runners/cold_start_ray_collect_runner.py \
        dreamervla/runners/collect_rollouts_runner.py \
        dreamervla/runners/embodied_eval_runner.py \
        tests/unit_tests/test_base_runner_console.py
git commit --signoff -m "feat(console): add force flag and console_success_rate helper; tidy reach-ins"
```
