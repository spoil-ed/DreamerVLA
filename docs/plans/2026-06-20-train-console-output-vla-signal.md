# Training Console Output Layering + Per-Loop VLA Signal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate training terminal output into three layers (config→files, normal runtime logs, `===` phase banners + interval metric box) and add a windowed per-loop VLA-improvement line, reusing the cotrain rollout success signal.

**Architecture:** Two new pure, unit-tested helpers (`console.py` rendering, `SuccessTracker`) plus a config-suppression gate and a model-summary writer in `BaseRunner`. Tier 0 (base) auto-applies config suppression to every runner; Tier 1 wires banners/box/VLA-line into the real cotrain path (`OnlineCotrainPipelineRunner` + `OnlineCotrainRunner`). Tier 2/3 (other runners) is a mechanical rollout deferred to its own plan.

**Tech Stack:** Python, PyTorch, OmegaConf/Hydra, pytest. Config knobs read via the repo's existing `OmegaConf.select(cfg, "...", default=...)` pattern (e.g. `online_cotrain_pipeline_runner.py:161`), overridable as Hydra args (`console.*`, `training.print_config`).

**Spec:** `docs/specs/2026-06-20-train-console-output-vla-signal-design.md`

**Commit note:** this repo's pre-commit requires a sign-off trailer. Every commit command below uses `git commit --signoff`.

---

## File Structure

- **Create** `dreamervla/utils/console.py` — pure rendering: `fmt_value`, `phase_banner`, `metric_box`. One responsibility: turn data into deterministic terminal strings.
- **Modify** `dreamervla/runners/online_utils.py` — add `SuccessTracker` (windowed success rate + best + delta-since-last-print).
- **Modify** `dreamervla/runners/base_runner.py` — gate `print_config` on `training.print_config` (default false); add `append_model_summary` (writes runtime model info into the existing `run_manifest.json`); add `count_trainable` import use.
- **Modify** `dreamervla/runners/online_cotrain_pipeline_runner.py` — phase banners around WM/classifier warmup and the online-cotrain start/skip; `[ok] model ready` line + model-summary write after `_build_components`; warmup helpers return last loss/acc for the "done" banner.
- **Modify** `dreamervla/runners/online_cotrain_runner.py` — feed `SuccessTracker` at episode-done; add windowed success metric; replace the ad-hoc per-update print with `metric_box`; drop the shape print and relocate the freeze summary into the model summary.
- **Create** `tests/unit_tests/test_console.py` — rendering + tracker tests.
- **Create** `tests/unit_tests/test_base_runner_config_gate.py` — print_config gate + manifest model-summary tests.

---

## Task 1: Console rendering helpers

**Files:**
- Create: `dreamervla/utils/console.py`
- Test: `tests/unit_tests/test_console.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/test_console.py
from dreamervla.utils import console


def test_fmt_value_thresholds():
    assert console.fmt_value(0.0) == "0"
    assert console.fmt_value(2) == "2"
    assert console.fmt_value(0.12345) == "0.123"
    assert console.fmt_value(0.0009) == "9.00e-04"
    assert console.fmt_value(123456.0) == "1.23e+05"
    assert console.fmt_value("warmup") == "warmup"


def test_phase_banner_start_and_done_are_symmetric_width():
    start = console.phase_banner("[1/3] WM WARMUP", subtitle="256 steps", width=65)
    done = console.phase_banner("[1/3] WM WARMUP", subtitle="wm_loss 0.012", done=True, width=65)
    assert start.startswith("=") and start.endswith("=")
    assert len(start) == 65 and len(done) == 65
    assert "WM WARMUP" in start
    assert "done" in done


def test_metric_box_renders_header_and_rows():
    box = console.metric_box(
        "cotrain · env_step 1600/8000 · 20%",
        ["VLA    succ@50=0.62 (d +0.08 best 0.66)", "train  wm=0.182 actor=0.226"],
        width=65,
    )
    lines = box.splitlines()
    assert lines[0].startswith("╭") and lines[0].endswith("╮")   # top corners
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")  # bottom corners
    assert all(len(ln) == 65 for ln in lines)
    assert any("succ@50" in ln for ln in lines)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit_tests/test_console.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dreamervla.utils.console'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreamervla/utils/console.py
"""Deterministic terminal-rendering helpers for training output.

Pure functions only — no I/O — so they are unit-testable. Value formatting
mirrors the threshold rules used by RLinf's print_metrics_table.
"""

from __future__ import annotations


def fmt_value(v: object) -> str:
    """Format a metric value compactly and deterministically."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0:
            return "0"
        a = abs(v)
        if a < 0.001 or a >= 100000:
            return f"{v:.2e}"
        if a < 0.01:
            return f"{v:.4f}"
        return f"{v:.3f}"
    return str(v)


def phase_banner(
    title: str, *, subtitle: str | None = None, done: bool = False, width: int = 65
) -> str:
    """Return a single `===` banner line of exactly `width` chars."""
    label = title if not done else f"{title} — done"
    if subtitle:
        label = f"{label} · {subtitle}"
    label = f" {label} "
    if len(label) >= width - 2:
        label = label[: width - 2]
    pad = width - len(label)
    left = pad // 2
    right = pad - left
    return ("=" * left) + label + ("=" * right)


def metric_box(header: str, rows: list[str], *, width: int = 65) -> str:
    """Return a box-drawn metric panel; every line is exactly `width` chars."""
    inner = width - 2

    def _fit(text: str) -> str:
        if len(text) <= inner:
            return text + (" " * (inner - len(text)))
        return text[: inner - 1] + "…"

    head = f" {header} "
    if len(head) > inner:
        head = head[:inner]
    pad = inner - len(head)
    top = "╭" + ("─" * (pad // 2)) + head + ("─" * (pad - pad // 2)) + "╮"
    body = ["│" + _fit(r) + "│" for r in rows]
    bottom = "╰" + ("─" * inner) + "╯"
    return "\n".join([top, *body, bottom])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit_tests/test_console.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add dreamervla/utils/console.py tests/unit_tests/test_console.py
git commit --signoff -m "feat(console): add deterministic banner/box rendering helpers"
```

---

## Task 2: SuccessTracker (windowed VLA signal)

**Files:**
- Modify: `dreamervla/runners/online_utils.py` (append class at end of file)
- Test: `tests/unit_tests/test_console.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit_tests/test_console.py
from dreamervla.runners.online_utils import SuccessTracker


def test_success_tracker_window_best_and_delta():
    t = SuccessTracker(window=4)
    assert t.rate() == 0.0 and len(t) == 0
    for s in (True, False, True, False):   # 2/4 = 0.5 over window
        t.update(s)
    assert t.rate() == 0.5
    assert t.best == 0.5
    # delta is vs last marked print; nothing marked yet -> 0.0
    assert t.delta() == 0.0
    t.mark_printed()
    t.update(True)  # window now (F,T,F,T) -> 0.5 still; then drops oldest True
    t.update(True)  # window (F,T,T,T) -> 0.75
    assert t.rate() == 0.75
    assert round(t.delta(), 3) == 0.25
    assert t.best == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit_tests/test_console.py::test_success_tracker_window_best_and_delta -v`
Expected: FAIL with `ImportError: cannot import name 'SuccessTracker'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to dreamervla/runners/online_utils.py
from collections import deque


class SuccessTracker:
    """Windowed episode success rate with best-so-far and delta-since-last-print.

    Cumulative success rate hides improvement (early failures sit in the
    denominator forever); a moving window over recent episodes reflects current
    policy quality. `delta()` is measured against the last `mark_printed()` so
    each printed box shows the change since the previous box.
    """

    def __init__(self, window: int) -> None:
        self._buf: deque[float] = deque(maxlen=max(1, int(window)))
        self._best: float = 0.0
        self._last_printed: float | None = None

    def update(self, success: bool) -> None:
        self._buf.append(1.0 if success else 0.0)
        r = self.rate()
        if r > self._best:
            self._best = r

    def rate(self) -> float:
        return (sum(self._buf) / len(self._buf)) if self._buf else 0.0

    @property
    def best(self) -> float:
        return self._best

    def delta(self) -> float:
        if self._last_printed is None:
            return 0.0
        return self.rate() - self._last_printed

    def mark_printed(self) -> None:
        self._last_printed = self.rate()

    def __len__(self) -> int:
        return len(self._buf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit_tests/test_console.py -v`
Expected: PASS (4 tests total)

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/online_utils.py tests/unit_tests/test_console.py
git commit --signoff -m "feat(console): add windowed SuccessTracker for VLA-improvement signal"
```

---

## Task 3: BaseRunner config-suppression gate + model-summary writer

**Files:**
- Modify: `dreamervla/runners/base_runner.py:128-130` (print_config), and add `append_model_summary` after `write_run_artifacts` (after `base_runner.py:151`)
- Test: `tests/unit_tests/test_base_runner_config_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit_tests/test_base_runner_config_gate.py
import json
import types

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner


def _fake(cfg, manifest_path):
    """Minimal object exposing the two methods under test without full init."""
    obj = types.SimpleNamespace()
    obj.cfg = cfg
    obj.config = cfg
    obj.is_main_process = True
    obj.print_config = types.MethodType(BaseRunner.print_config, obj)
    obj.append_model_summary = types.MethodType(BaseRunner.append_model_summary, obj)
    obj.get_run_manifest_path = lambda: manifest_path
    return obj


def test_print_config_suppressed_by_default(capsys):
    cfg = OmegaConf.create({"a": 1})
    _fake(cfg, None).print_config()
    assert capsys.readouterr().out == ""


def test_print_config_emitted_when_enabled(capsys):
    cfg = OmegaConf.create({"a": 1, "training": {"print_config": True}})
    _fake(cfg, None).print_config()
    assert "'a': 1" in capsys.readouterr().out


def test_append_model_summary_updates_manifest(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")
    cfg = OmegaConf.create({})
    _fake(cfg, path).append_model_summary({"total_trainable": 12_300_000})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["model"]["total_trainable"] == 12_300_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit_tests/test_base_runner_config_gate.py -v`
Expected: FAIL — `test_print_config_suppressed_by_default` fails (config still printed) and `append_model_summary` raises `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Replace `base_runner.py:128-130`:

```python
    def print_config(self) -> None:
        # Config dump — suppressed by default; the resolved config is always
        # persisted to resolved_config.yaml + .hydra/, so nothing is lost.
        if not bool(OmegaConf.select(self.cfg, "training.print_config", default=False)):
            return
        pprint(OmegaConf.to_container(self.config, resolve=True))
```

Add after `write_run_artifacts` (after `base_runner.py:151`):

```python
    def append_model_summary(self, summary: dict[str, Any]) -> None:
        """Write runtime-derived model info (param counts, freeze flags) into
        the existing run manifest. These are not in any config because they are
        computed after model instantiation."""
        if not self.is_main_process:
            return
        path = self.get_run_manifest_path()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}
        manifest["model"] = summary
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit_tests/test_base_runner_config_gate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/base_runner.py tests/unit_tests/test_base_runner_config_gate.py
git commit --signoff -m "feat(base_runner): gate config dump on training.print_config; model summary to manifest"
```

---

## Task 4: count_trainable helper + model-summary wiring + `[ok]` line

**Files:**
- Modify: `dreamervla/utils/console.py` (add `count_trainable`)
- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py:98` (after `_build_components(cfg)`)
- Test: `tests/unit_tests/test_console.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit_tests/test_console.py
import torch

from dreamervla.utils.console import count_trainable


def test_count_trainable_counts_only_grad_params():
    m = torch.nn.Linear(4, 3)            # 4*3 + 3 = 15 params
    assert count_trainable(m) == 15
    for p in m.parameters():
        p.requires_grad_(False)
    assert count_trainable(m) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit_tests/test_console.py::test_count_trainable_counts_only_grad_params -v`
Expected: FAIL with `ImportError: cannot import name 'count_trainable'`

- [ ] **Step 3: Write minimal implementation**

Append to `dreamervla/utils/console.py`:

```python
def count_trainable(module) -> int:
    """Number of trainable (requires_grad) parameters in a torch module."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
```

Then wire it in `online_cotrain_pipeline_runner.py` immediately after
`self._build_components(cfg)` (line 98). Add the import near the top of the file
(`from dreamervla.utils.console import count_trainable, phase_banner`) and insert:

```python
        self._build_components(cfg)
        if self.distributed.is_main_process:
            trainable = {
                "world_model": count_trainable(self.world_model),
                "policy": count_trainable(self.policy),
                "critic": count_trainable(self.critic),
                "classifier": count_trainable(self.classifier),
            }
            total = sum(trainable.values())
            self.append_model_summary(
                {"total_trainable": total, "trainable_params": trainable}
            )
            print(f"[ok] model ready · {total/1e6:.1f}M trainable", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit_tests/test_console.py -v`
Expected: PASS (count_trainable test green; others still green)

- [ ] **Step 5: Commit**

```bash
git add dreamervla/utils/console.py dreamervla/runners/online_cotrain_pipeline_runner.py tests/unit_tests/test_console.py
git commit --signoff -m "feat(cotrain): write model summary to manifest + one-line [ok] model ready"
```

---

## Task 5: Phase banners in the pipeline runner

**Files:**
- Modify: `dreamervla/runners/online_cotrain_pipeline_runner.py:33-48` and `:50-64` (warmup helpers return last loss/acc), `:137-166` (banners around phases)

- [ ] **Step 1: Make warmup helpers return their last metric.**

In `_offline_warmup_wm` (`:33-48`), track and return the last loss:

```python
    def _offline_warmup_wm(self, replay, *, steps: int, batch_size: int, optim_cfg) -> float:
        self.world_model.train()
        last = 0.0
        for i in range(int(steps)):
            wm_batch = self._build_wm_pretrain_batch(replay.sample(batch_size))
            if wm_batch is None:
                continue
            m = world_model_pretrain_step(
                policy=self.policy,
                world_model=self.world_model,
                optimizer=self.world_model_optimizer,
                batch=wm_batch,
                device=self.device,
                optim_cfg=optim_cfg,
            )
            last = float(m.get("loss", 0.0))
            if i % 50 == 0:
                print(f"[pipeline][wm-warmup] step={i}/{steps} loss={last:.4f}", flush=True)
        return last
```

In `_offline_warmup_classifier` (`:50-64`), return the last accuracy:

```python
    def _offline_warmup_classifier(
        self, replay, *, steps: int, batch_size: int, early_neg_stride: int, grad_clip: float
    ) -> float:
        last_acc = 0.0
        for i in range(int(steps)):
            m = online_classifier_update_step(
                classifier=self.classifier,
                optimizer=self.classifier_optimizer,
                replay=replay,
                device=self.device,
                batch_size=batch_size,
                early_neg_stride=early_neg_stride,
                grad_clip=grad_clip,
            )
            last_acc = float(m["acc"])
            if i % 50 == 0:
                print(f"[pipeline][cls-warmup] step={i}/{steps} loss={float(m['loss']):.4f} acc={last_acc:.3f}", flush=True)
        return last_acc
```

- [ ] **Step 2: Add banners around the phases.**

Ensure the import line added in Task 4 includes `phase_banner`. Wrap phase [1]
(`:137-144`):

```python
        if need_wm:
            if self.distributed.is_main_process:
                print(phase_banner("[1/3] WM WARMUP", subtitle=f"{wm_steps} steps"), flush=True)
            wm_last = self._offline_warmup_wm(warmup_replay, steps=wm_steps, batch_size=bs, optim_cfg=optim_cfg)
            if self.distributed.is_main_process:
                self._save_wm_warmup()
                print(phase_banner("[1/3] WM WARMUP", subtitle=f"wm_loss {wm_last:.3f}", done=True), flush=True)
        else:
            payload = torch.load(self._wm_warmup_ckpt(), map_location="cpu", weights_only=False)
            _unwrap(self.world_model).load_state_dict(payload["world_model"])
```

Wrap phase [2] (`:145-153`):

```python
        if need_cls:
            if self.distributed.is_main_process:
                print(phase_banner("[2/3] CLASSIFIER WARMUP", subtitle=f"{cls_steps} steps"), flush=True)
            cls_last = self._offline_warmup_classifier(warmup_replay, steps=cls_steps, batch_size=cls_bs,
                                            early_neg_stride=early_neg_stride, grad_clip=grad_clip)
            if self.distributed.is_main_process:
                self._save_cls_warmup()
                print(phase_banner("[2/3] CLASSIFIER WARMUP", subtitle=f"acc {cls_last:.3f}", done=True), flush=True)
        else:
            payload = torch.load(self._cls_warmup_ckpt(), map_location="cpu", weights_only=False)
            _unwrap(self.classifier).load_state_dict(payload["classifier"])
            self.classifier_threshold = float(payload.get("classifier_threshold", self.classifier_threshold))
```

Phase [3] start / skip (`:161-166`):

```python
        total_env_steps = int(OmegaConf.select(cfg, "online_rollout.total_env_steps", default=0))
        if total_env_steps <= 0:
            if self.distributed.is_main_process:
                print(phase_banner("[3/3] ONLINE COTRAIN", subtitle="skipped · total_env_steps=0", done=True), flush=True)
            return []
        if self.distributed.is_main_process:
            print(phase_banner("[3/3] ONLINE COTRAIN", subtitle=f"{total_env_steps} env steps"), flush=True)
        return self._online_cotrain_loop(cfg)
```

- [ ] **Step 3: Run the existing launcher test to confirm no regression.**

Run: `python -m pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -v`
Expected: PASS (unchanged behavior — banners are additive prints)

- [ ] **Step 4: Commit**

```bash
git add dreamervla/runners/online_cotrain_pipeline_runner.py
git commit --signoff -m "feat(cotrain): === phase banners for WM/classifier warmup and online cotrain"
```

---

## Task 6: Cotrain loop — windowed VLA line + metric box

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py:411-414` (init tracker), `:432-438` (drop shape print), `:441-445` (feed tracker), `:467-552` (windowed metric + metric_box)

- [ ] **Step 1: Initialize the tracker and drop the shape print.**

Add imports near the top: `from dreamervla.utils.console import metric_box, fmt_value` and `from dreamervla.runners.online_utils import SuccessTracker` (if not already importing from online_utils).

Replace the counter init (`:411-414`):

```python
        n_episodes = 0
        n_success = 0
        stop = False
        success_window = int(OmegaConf.select(cfg, "console.success_window", default=50))
        tracker = SuccessTracker(window=success_window)
        log_every = int(OmegaConf.select(cfg, "console.log_every", default=1))
        banner_width = int(OmegaConf.select(cfg, "console.banner_width", default=65))
        update_idx = 0
```

Delete the shape-print block (`:432-438`) entirely (it is debug noise; param/shape detail lives in the manifest model summary). Remove the now-unused `printed_shapes` variable from `:414`.

- [ ] **Step 2: Feed the tracker at episode-done.**

Replace `:441-445`:

```python
            if done:
                rec = replay.add_episode(episode)
                if rec is not None:
                    n_episodes += 1
                    success = bool(rec["success"])
                    n_success += int(success)
                    tracker.update(success)
                episode = []
                obs, _info = env.reset()
                latent, prev_action = None, None
```

- [ ] **Step 3: Add the windowed metric and replace the per-update print with a box.**

In the metrics dict (`:468-473`), add the windowed rate alongside the cumulative one:

```python
                metrics: dict[str, float | str | int] = {
                    "global_step": int(self.global_step),
                    "phase": "warmup" if in_warmup else "cotrain",
                    "rollout/success_rate": (n_success / n_episodes) if n_episodes else 0.0,
                    "rollout/success_rate_windowed": tracker.rate(),
                    "buffer/size": float(replay.num_transitions),
                }
```

Replace the print block (`:542-550`) — keep `log_metrics`, swap the line for a
phase-aware box printed every `log_every` updates:

```python
                if self.distributed.is_main_process:
                    update_idx += 1
                    if update_idx % log_every == 0:
                        rows = []
                        if not in_warmup:
                            rows.append(
                                f"VLA    succ@{success_window}={fmt_value(tracker.rate())} "
                                f"(Δ {tracker.delta():+.3f} · best {tracker.best:.3f})   "
                                f"return={fmt_value(metrics.get('rl/returns_mean', 0.0))}"
                            )
                        rows.append(
                            f"train  wm={fmt_value(metrics.get('wm/loss', float('nan')))}  "
                            f"actor={fmt_value(metrics.get('rl/actor_loss', float('nan')))}  "
                            f"cls_acc={fmt_value(metrics.get('cls/acc', float('nan')))}"
                        )
                        rows.append(
                            f"data   buf={fmt_value(metrics['buffer/size'])}  "
                            f"ep={n_episodes}  cum_succ={fmt_value(metrics['rollout/success_rate'])}"
                        )
                        header = f"{metrics['phase']} · step {self.global_step}"
                        print(metric_box(header, rows, width=banner_width), flush=True)
                        tracker.mark_printed()
                    self.log_metrics(metrics, step=int(self.global_step))
```

- [ ] **Step 4: Run the existing launcher test to confirm no regression.**

Run: `python -m pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py
git commit --signoff -m "feat(cotrain): windowed VLA success line + metric box; drop shape print"
```

---

## Task 7: Relocate the freeze summary into the model summary

**Files:**
- Modify: `dreamervla/runners/online_cotrain_runner.py:~130-155` (the classifier warm-start / freeze-summary print block)

- [ ] **Step 1: Read the block at `online_cotrain_runner.py:130-155`** to identify the freeze-summary prints (trainable-name list, freeze flags). These were confirmed to print a multi-line freeze summary to stdout.

- [ ] **Step 2: Replace the multi-line stdout freeze summary with a manifest write.**

Collect the same freeze information into a dict and pass it to
`self.append_model_summary({... , "freeze": <freeze_dict>})` (merge with the
summary written in Task 4 by reading the manifest, which `append_model_summary`
already does). Keep at most one short terminal line if a "loaded/frozen" signal
is genuinely useful (Layer 2), e.g. `[ok] classifier warm-started`. Remove the
verbose per-name listing from stdout.

Concretely, where the block currently does `print(...freeze summary...)`, change
to build a `freeze = {"trainable_modules": [...], ...}` dict and call
`self.append_model_summary({"freeze": freeze})` (guarded by
`self.distributed.is_main_process`). Retain any genuine `[load]`/`[ok]` one-liner.

- [ ] **Step 3: Run the existing launcher test.**

Run: `python -m pytest tests/unit_tests/test_coldstart_warmup_cotrain_launcher.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add dreamervla/runners/online_cotrain_runner.py
git commit --signoff -m "refactor(cotrain): move freeze summary off terminal into run manifest"
```

---

## Task 8: Warmup-only stdout smoke test

**Files:**
- Create/extend: `tests/unit_tests/test_console.py` (or a new `tests/unit_tests/test_cotrain_output_smoke.py` if env construction is too heavy)

- [ ] **Step 1: Decide test depth.** A full runner smoke test needs LIBERO/GPU and is not a unit test. Instead, assert the *contract* the runner relies on with a lightweight test that exercises `phase_banner` for the warmup-only skip path and the box for a synthetic cotrain step — proving the strings the runner emits are well-formed.

```python
# append to tests/unit_tests/test_console.py
from dreamervla.utils import console
from dreamervla.runners.online_utils import SuccessTracker


def test_cotrain_box_strings_are_wellformed_for_a_synthetic_step():
    tr = SuccessTracker(window=50)
    for s in (True, False, True):
        tr.update(s)
    rows = [
        f"VLA    succ@50={console.fmt_value(tr.rate())} (Δ {tr.delta():+.3f} · best {tr.best:.3f})   return=0.71",
        "train  wm=0.182  actor=0.226  cls_acc=0.95",
        "data   buf=10000  ep=3  cum_succ=0.667",
    ]
    box = console.metric_box("cotrain · step 1600", rows, width=65)
    assert all(len(ln) == 65 for ln in box.splitlines())
    skip = console.phase_banner("[3/3] ONLINE COTRAIN", subtitle="skipped · total_env_steps=0", done=True)
    assert "skipped" in skip and len(skip) == 65
```

- [ ] **Step 2: Run the full unit suite for the feature.**

Run: `python -m pytest tests/unit_tests/test_console.py tests/unit_tests/test_base_runner_config_gate.py -v`
Expected: PASS (all)

- [ ] **Step 3: Commit**

```bash
git add tests/unit_tests/test_console.py
git commit --signoff -m "test(console): well-formed cotrain box + warmup-only skip banner"
```

---

## Task 9 (appendix): Tier 2/3 rollout — separate plan

Tier 0 (config suppression via `BaseRunner.print_config`) already applies to
**every** runner automatically once Tasks 1–8 land. The remaining runners
(`dreamervla_runner`, `dreamerv3_pixel_runner`, `dreamerv3_token_runner`,
`online_dreamervla`(+`_multiproc`), `frozen_wm_actor_critic`, `vla_sft_runner`,
`latent_wm_runner`, `latent_classifier_runner`, the `collect_*`/eval/`*_ray_runner`
families) need their own phase banners + metric box wired at each runner's
specific loop boundaries.

This is mechanical but per-runner (each has distinct phase boundaries and
metric keys), so it is **not** enumerated to the line here — that would be
fabricated detail. After Tasks 1–8 prove the helper API, run `writing-plans`
again with the brief: "wire `phase_banner` + `metric_box` + (where a rollout/eval
success signal exists) `SuccessTracker` into runners X, Y, Z, mirroring the
pattern in `online_cotrain_runner.py`." One task per runner, same TDD shape.

---

## Self-Review

- **Spec coverage:**
  - Layer 1 (config→files): Task 3 (gate) + Task 4 (manifest model summary) + Task 7 (freeze summary). ✓
  - Layer 2 (normal output stays): preserved — only verbose dumps removed (Tasks 6, 7); `[load]/[ok]` one-liners kept. ✓
  - Layer 3 (`===` banners + interval box): Task 5 (banners) + Task 6 (box). ✓
  - VLA windowed signal: Task 2 (tracker) + Task 6 (wiring). ✓
  - Hydra knobs (`training.print_config`, `console.*`): Tasks 3, 6 via `OmegaConf.select` defaults. ✓
  - All-runners scope: Task 3 covers all for config suppression; Task 9 sequences the rest. ✓
  - Tests: Tasks 1–4, 8. ✓
- **Placeholder scan:** Task 7 Step 1 reads a block before editing (the freeze block was not quoted by exploration line-for-line) and Step 2 describes the transform rather than a literal diff — this is the one place a concrete diff is deferred to read-time because the exact lines were not captured. Acceptable and explicitly flagged; all other steps carry literal code.
- **Type consistency:** `SuccessTracker.update/rate/best/delta/mark_printed/__len__`, `console.fmt_value/phase_banner/metric_box/count_trainable`, `BaseRunner.print_config/append_model_summary` — names used identically across Tasks 1–8. ✓
