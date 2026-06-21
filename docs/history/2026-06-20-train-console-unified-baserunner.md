# Unified BaseRunner Console API + Tier-2/3 Runner Wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Move the phase-banner / metric-box / success-tracking plumbing into three unified `BaseRunner` methods, refactor the existing cotrain path onto them, then wire every active runner's loop to call them.

**Architecture:** `BaseRunner` gains `console_banner`, `console_record_success`, `console_metrics` (backed by the existing `dreamervla/utils/console.py` helpers + `SuccessTracker`). Each runner calls these at its phase boundaries and per-loop; the box auto-groups metrics by namespace and prepends a VLA success row when a tracker is active. Single implementation, uniform output.

**Tech Stack:** Python, OmegaConf/Hydra, pytest. Config knobs via `OmegaConf.select(cfg, "console.*", default=...)`.

**Builds on:** `docs/history/2026-06-20-train-console-output-vla-signal.md` (Tier 0/1, merged). Helpers `phase_banner`/`metric_box`/`fmt_value`/`count_trainable` (console.py) and `SuccessTracker` (online_utils.py) already exist.

**Commit rules (this repo):** every commit uses `git commit --signoff`; commit *descriptions* must not contain `===` or `/`; ruff runs on changed Python (no unused imports / trailing whitespace).

---

## Shared: the BaseRunner console API contract

All wiring tasks (3–13) use exactly these methods (implemented in Task 1):

- `self.console_banner(title: str, *, subtitle: str | None = None, done: bool = False) -> None`
  Main-process-guarded. Prints `phase_banner(...)` at `console.banner_width`. Call at each phase boundary (start, and `done=True` at end).
- `self.console_record_success(success: bool) -> None`
  Feeds a lazily-created base-owned `SuccessTracker` (window = `console.success_window`). Call once per finished episode where a real success/return signal exists. Offline/loss-only runners never call it (→ no VLA row).
- `self.console_metrics(header: str, metrics: dict) -> None`
  Main-process-guarded; throttled by `console.log_every` (base-owned counter). Renders a `metric_box`: a leading `VLA succ@N=… (Δ … · best …)` row iff the tracker has data, then the runner's `metrics` auto-grouped by namespace prefix (`train/`, `rollout/`, `eval/`, `time/`, …). Pass your existing namespaced per-loop metrics dict. Call once per logged iteration.

The reference usage after refactor lives in `online_cotrain_runner.py` / `online_cotrain_pipeline_runner.py` (Task 2) — mirror it.

---

## Task 1: Unified console methods on BaseRunner

**Files:**
- Modify: `dreamervla/runners/base_runner.py` (add a module-level `_group_metric_rows` helper + three methods + lazy state)
- Test: `tests/unit_tests/test_base_runner_console.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/test_base_runner_console.py
import types

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner, _group_metric_rows


def _runner(cfg, *, main=True):
    obj = types.SimpleNamespace()
    obj.cfg = cfg
    obj.is_main_process = main
    for name in ("console_banner", "console_record_success", "console_metrics", "_console_state_get"):
        setattr(obj, name, types.MethodType(getattr(BaseRunner, name), obj))
    return obj


def test_group_metric_rows_groups_by_namespace_and_skips_meta():
    rows = _group_metric_rows({"train/wm_loss": 0.182, "train/actor_loss": 0.226,
                               "rollout/success_rate": 0.55, "global_step": 5, "phase": "cotrain"})
    joined = "\n".join(rows)
    assert any(r.startswith("train") for r in rows)
    assert "wm_loss=0.182" in joined and "actor_loss=0.226" in joined
    assert "global_step" not in joined and "phase" not in joined


def test_console_banner_guarded(capsys):
    cfg = OmegaConf.create({"console": {"banner_width": 65}})
    _runner(cfg).console_banner("[1/3] WM WARMUP", subtitle="256 steps")
    out = capsys.readouterr().out
    assert "WM WARMUP" in out and len(out.strip()) == 65
    _runner(cfg, main=False).console_banner("X")
    assert capsys.readouterr().out == ""


def test_console_metrics_throttle_and_vla_row(capsys):
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 2, "success_window": 4}})
    r = _runner(cfg)
    for s in (True, False, True, False):
        r.console_record_success(s)
    r.console_metrics("cotrain · step 1", {"train/wm_loss": 0.18})  # counter=1, log_every=2 -> no print
    assert capsys.readouterr().out == ""
    r.console_metrics("cotrain · step 2", {"train/wm_loss": 0.18})  # counter=2 -> print
    out = capsys.readouterr().out
    assert "VLA" in out and "succ@4=" in out and "wm_loss=0.18" in out
    assert all(len(ln) == 65 for ln in out.strip().splitlines())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit_tests/test_base_runner_console.py -v`
Expected: FAIL — `ImportError: cannot import name '_group_metric_rows'` / missing methods.

- [ ] **Step 3: Write the implementation**

In `dreamervla/runners/base_runner.py`, add imports (merge with existing console import if present):
```python
from dreamervla.utils.console import fmt_value, metric_box, phase_banner
from dreamervla.runners.online_utils import SuccessTracker
```

Add a module-level helper (near the top, after imports):
```python
def _group_metric_rows(metrics: dict, *, skip_success: bool = False) -> list[str]:
    """Group namespaced metrics into one row per prefix, dropping meta keys."""
    meta = {"global_step", "step", "epoch", "ts", "phase"}
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for k, v in metrics.items():
        if k in meta or isinstance(v, str):
            continue
        if skip_success and k.startswith("rollout/success_rate"):
            continue
        prefix, _, name = k.partition("/")
        if not name:
            prefix, name = "metrics", k
        if prefix not in groups:
            groups[prefix] = []
            order.append(prefix)
        groups[prefix].append(f"{name}={fmt_value(v)}")
    return [f"{p:<7} " + "  ".join(groups[p]) for p in order]
```

Add to the `BaseRunner` class (anywhere among its methods):
```python
    def _console_state_get(self) -> dict:
        st = getattr(self, "_console_state", None)
        if st is None:
            st = {
                "width": int(OmegaConf.select(self.cfg, "console.banner_width", default=65)),
                "log_every": max(1, int(OmegaConf.select(self.cfg, "console.log_every", default=1))),
                "window": int(OmegaConf.select(self.cfg, "console.success_window", default=50)),
                "counter": 0,
                "tracker": None,
            }
            self._console_state = st
        return st

    def console_banner(self, title: str, *, subtitle: str | None = None, done: bool = False) -> None:
        if not self.is_main_process:
            return
        st = self._console_state_get()
        print(phase_banner(title, subtitle=subtitle, done=done, width=st["width"]), flush=True)

    def console_record_success(self, success: bool) -> None:
        st = self._console_state_get()
        if st["tracker"] is None:
            st["tracker"] = SuccessTracker(window=st["window"])
        st["tracker"].update(bool(success))

    def console_metrics(self, header: str, metrics: dict) -> None:
        if not self.is_main_process:
            return
        st = self._console_state_get()
        st["counter"] += 1
        if st["counter"] % st["log_every"] != 0:
            return
        tr = st["tracker"]
        rows: list[str] = []
        if tr is not None and len(tr) > 0:
            rows.append(
                f"VLA     succ@{st['window']}={fmt_value(tr.rate())} "
                f"(Δ {tr.delta():+.3f} · best {tr.best:.3f})"
            )
        rows.extend(_group_metric_rows(metrics, skip_success=tr is not None))
        print(metric_box(header, rows, width=st["width"]), flush=True)
        if tr is not None:
            tr.mark_printed()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit_tests/test_base_runner_console.py tests/unit_tests/test_console.py tests/unit_tests/test_base_runner_config_gate.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/base_runner.py tests/unit_tests/test_base_runner_console.py
git commit --signoff -m "feat(base_runner): unified console_banner, console_record_success, console_metrics"
```

---

## Task 2: Refactor the cotrain path onto the unified API

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py`, `dreamervla/runners/online_cotrain_pipeline_runner.py`

**Goal:** replace the inline `phase_banner(...)`, `metric_box(...)`, and the local `SuccessTracker` usage (added in Tier-1) with the new `self.console_*` methods. Behavior stays equivalent (banners at the same boundaries; box per `log_every`; VLA row from the tracker). Remove now-unused imports.

- [ ] **Step 1: Refactor `online_cotrain_runner.py`**
  - Delete the local tracker setup (`success_window`, `tracker = SuccessTracker(...)`, `log_every`, `banner_width`, `update_idx`) from the counter-init block; keep `n_episodes`/`n_success`.
  - In the episode-done block, replace `tracker.update(success)` with `self.console_record_success(success)`.
  - Remove `"rollout/success_rate_windowed": tracker.rate()` from the metrics dict (the VLA row now comes from the base tracker).
  - Replace the whole `if self.distributed.is_main_process:` box-print block (the `update_idx % log_every` gate + `metric_box(...)` + `mark_printed()`) with a single call, keeping `log_metrics` every update:
    ```python
                self.console_metrics(f"{metrics['phase']} · step {self.global_step}", metrics)
                if self.distributed.is_main_process:
                    self.log_metrics(metrics, step=int(self.global_step))
    ```
  - Remove now-unused imports: `metric_box`, `fmt_value` from the console import; `SuccessTracker` from the online_utils import (leave other symbols).

- [ ] **Step 2: Refactor `online_cotrain_pipeline_runner.py`**
  - Replace each `print(phase_banner(...), flush=True)` (the six occurrences) with `self.console_banner(...)` (drop the explicit `is_main_process` guard around banner-only lines — `console_banner` guards internally; keep the guard where it also wraps `_save_*`). Keep the `[ok] model ready` line as-is.
  - Remove the `phase_banner` import (keep `count_trainable`).

- [ ] **Step 3: Verify**

Run: `python -m py_compile dreamervla/runners/online_cotrain_runner.py dreamervla/runners/online_cotrain_pipeline_runner.py`
Run: `python -m pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py tests/unit_tests/test_console.py tests/unit_tests/test_base_runner_console.py -q`
Expected: py_compile clean; launcher 35 + console tests pass.
Grep checks: `grep -n "metric_box\|phase_banner\|SuccessTracker(" dreamervla/runners/online_cotrain_runner.py` → no matches (all via base now); pipeline runner has no `phase_banner(` (uses `self.console_banner`).

- [ ] **Step 4: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py dreamervla/runners/online_cotrain_pipeline_runner.py
git commit --signoff -m "refactor(cotrain): use unified BaseRunner console methods"
```

---

## Tasks 3–13: per-runner wiring (shared recipe)

**Recipe for every wiring task:**
1. Read the runner's loop region (entry method + loop line from the map below).
2. At each phase boundary, add `self.console_banner(title, subtitle=..., [done=True])`. Replace any existing ad-hoc "starting/finished phase" prints; keep genuine `[load]`/`[ok]`/error lines.
3. Per logged iteration, call `self.console_metrics(header, metrics)` with the runner's existing namespaced metrics dict. If the runner currently prints a raw per-step/per-epoch metrics line, replace it with this call; keep `self.log_metrics(...)` (TensorBoard/JSON) unchanged.
4. If the map marks a success signal, call `self.console_record_success(bool(success))` once per finished episode/eval rollout.
5. Do NOT print the full config or param dumps (Tier-0 already suppresses config; drop any remaining verbose dumps to terminal).
6. Verify: `python -m py_compile <file>`; run that runner's tests if any exist; confirm `python -m pytest tests/unit_tests/test_base_runner_console.py -q` still green; grep to confirm banners/metrics calls added.
7. Commit `--signoff` with a plain description (no `===`, no `/`).

Banners use the runner's natural phases. Header for `console_metrics`: `f"{phase} · epoch {self.epoch}"` or `f"{phase} · step {self.global_step}"` as fits.

| Task | File (runner) | Loop / phases (from map) | console_record_success? |
| --- | --- | --- | --- |
| 3 | `dreamerv3_pixel_runner.py` (DreamerV3PixelRunner — base; also covers LatentWM/Backbone via inheritance) | `while self.epoch < num_epochs` @526; per-epoch train; tqdm @577 | No (loss-only) |
| 4 | `dreamerv3_token_runner.py` (DreamerV3TokenRunner) | `while self.epoch < num_epochs` @432; per-epoch train | No |
| 5 | `chameleon_latent_action_wm_runner.py` (ChameleonLatentActionWMRunner) | `while self.epoch < num_epochs` @372+; per-epoch train; prints @385-392 | No |
| 6 | `dreamervla_runner.py` (DreamerVLARunner) | `while self.epoch < num_epochs` @1684; Phase1 WM pretrain @1707 → Phase2 AC imagination @1749 → optional eval; epoch metrics @1882 | No (offline; returns/loss only) |
| 7 | `latent_classifier_runner.py` (LatentClassifierRunner) | `while self.epoch < num_epochs` @266; prints @243-251 | No (loss/acc) |
| 8 | `openvla_oft_runner.py` (OpenVLAOFTTrainingRunner) | `while self.epoch < num_epochs` @182; per-step metrics @210-222 | No |
| 9 | `pretokenize_vla_runner.py` (PretokenizeVLARunner — base; covers VLASFTRunner) | `while self.epoch < num_epochs` @792; train @801 → optional val → optional LIBERO eval @853; eval_success_rate @856 | **Yes** — feed from the per-episode LIBERO eval result when eval runs |
| 10 | `embodied_eval_runner.py` (EmbodiedEvalRunner) | `for episode_idx in range(n_eps)` @1918; per-task eval; success_rate @1912 | **Yes** — per eval episode success |
| 11 | `cold_start_ray_collect_runner.py` (ColdStartRayCollectRunner) | `while env_ids and steps < max_steps` @293; ray rollout + async collect; summary @119 | **Yes** — per-episode success/return from env info |
| 12 | `online_cotrain_ray_runner.py` (OnlineCotrainRayRunner) | `for step in range(rollout_steps)` @192 + async train @434 | **Yes** — rollout episode success |
| 13 | `collect_rollouts_runner.py` (CollectRolloutsRunner) | sequential rollout loop @80+ | **Yes** — episode success if env provides it |

---

## Self-Review

- **Coverage:** Task 1 builds the unified API; Task 2 makes the existing cotrain path the reference consumer; Tasks 3–13 cover all 11 active runners (with inheritance covering LatentWM/Backbone/VLASFT). The flagged legacy/helper runners are intentionally excluded (per user scope decision). ✓
- **DRY:** all banner/box/tracker logic lives once in BaseRunner; runners only call it. ✓
- **Placeholders:** Tasks 1–2 carry exact code. Tasks 3–13 are pattern-application against a fixed, tested API with per-runner loop coordinates from the map — the implementer reads the specific loop (unavoidable: each loop differs) and applies the recipe. This is the one place exact diffs are deferred to read-time, by necessity. Flagged here.
- **Types/consistency:** method names `console_banner` / `console_record_success` / `console_metrics` and `_group_metric_rows` are used identically across all tasks. ✓
- **Testability:** Task 1 is fully unit-tested; the runner loops need GPU/LIBERO so Tasks 2–13 verify via py_compile + existing tests + grep (inherent gap, same as Tier-1).
