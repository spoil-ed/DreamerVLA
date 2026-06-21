# Pipeline progress reporter + cotrain resume — design

Date: 2026-06-21
Status: approved (A + R1)

## Goal

Two related gaps in run observability and recoverability:

1. **Unified progress output across the whole pipeline.** Every long-running
   loop (training, cotrain, collect/rollout, eval, preprocess) should print a
   consistent, RLinf-style progress line. One mechanism everywhere — no mix of
   tqdm in some places and ad-hoc prints in others. Not necessarily tqdm; just a
   single, uniform reporter.
2. **Real resume for `online_cotrain`.** Today the cotrain loop never reloads a
   checkpoint: `global_step` restarts at 0, optimizer state is never saved, and
   `_save_cotrain_ckpt()` is a parallel, lossy path that bypasses the base
   checkpoint machinery.

The two share a delivery because both are "make every flow behave the same way"
cleanups.

## Non-goals

- No new logger backends or metric-namespace changes (reuse `time/`/`train/`).
- No tqdm dependency. The whole-codebase cleanup review is a separate follow-on
  deliverable, not part of this spec.

## Part 1 — Unified progress reporter

### 1a. Pure formatter — `dreamervla/utils/console.py`

`console.py` is deliberately I/O-free and unit-testable; the formatter stays
there next to `phase_banner`/`metric_box`/`fmt_value`.

```python
def format_progress_line(desc, current, total, *, elapsed_s, eta_s, rate, unit="it") -> str
```

- `total` int  → `"pretokenize 12800/50000 (26%) · 03:21<09:45 · 63.7 it/s"`
- `total=None` → `"collect 812 · 03:21 · 4.0 ep/s"` (open-ended: count + elapsed + rate, no %/ETA)
- Private `_fmt_duration(s)` → `mm:ss`, or `h:mm:ss` past an hour.
- `·`-separated to match the existing console aesthetic; percent is integer-rounded.

### 1b. Stateful reporter — new `dreamervla/utils/progress.py`

Timing + throttling + sink live outside `console.py` (which must stay pure).
Depends only on the formatter.

```python
class ProgressReporter:
    def __init__(self, total, desc, *, enabled=True, min_interval_s=5.0,
                 unit="it", clock=time.monotonic, sink=print): ...
    def update(self, n=1): ...     # advance (training / preprocess)
    def set(self, current): ...    # absolute value (collect polls dump.size())
    def close(self): ...           # always prints a final summary line
    def __enter__(self) / __exit__(self): ...
```

- Throttled by **wall-clock** `min_interval_s` (default 5s), independent of
  `console.log_every` (which gates the metric box). Always prints the first tick
  and the final `close()`.
- `clock` and `sink` are injectable so unit tests are deterministic — no real
  sleeping, no tqdm carriage-return behaviour (clean in log files / `nohup` /
  Ray worker logs).
- `enabled=False` makes every method a no-op (non-main ranks, or progress
  disabled).
- Rate is computed from `(current - start_current) / (now - start_time)`; ETA
  from rate and remaining when `total` is known.

### 1c. Runner hook — `BaseRunner.console_progress(...)`

```python
def console_progress(self, current, total, desc, *, unit="it") -> None
```

- Lazily builds and caches one `ProgressReporter` per `desc` inside
  `_console_state` (new `"progress": {}` dict), `enabled=self.is_main_process`,
  `min_interval_s=console.progress_every_s`.
- Each call does `reporter.set(current)`. `teardown()` closes any cached
  reporters (final summary line per flow).
- Standalone preprocess scripts that have no runner import `ProgressReporter`
  directly and use it as a context manager.

### 1d. Config knob

`console.progress_every_s` (float, default `5.0`); `0` disables progress output
entirely. Read via `OmegaConf.select(..., default=5.0)` in `_console_state_get`,
alongside the existing `console.*` knobs. No config file change required.

### 1e. Integration map (every loop, one mechanism)

| Category   | Loops                                                                                             | `total` source        |
|------------|--------------------------------------------------------------------------------------------------|-----------------------|
| Training   | `dreamervla`, `latent_wm`, `backbone_dreamerv3_wm`, `dreamerv3_pixel`, `dreamerv3_token`, `vla_sft` | max update steps (cfg) |
| Cotrain    | `online_cotrain` (warmup + RL)                                                                    | `total_env_steps`     |
| Collect    | `collect_parallel_rollouts`, `vectorized_collect`, `cold_start_ray_collect`*, `collect_online_rollouts_for_classifier` | `target_episodes`     |
| Eval       | `embodied_eval`, `rlinf_libero_rollout`                                                           | episode / task count  |
| Preprocess | `pretokenize_vla`*, `preprocess_oft_action_hidden`, `preprocess_rynn_pixel_hidden`, `preprocess_remaining_steps_reward`, `pre_tokenize_action_local`, `pre_tokenize_action_state_local`, libero regen scripts | `len(dataset)` / item count |

`*` = currently uses tqdm → replaced by `ProgressReporter` for one uniform
format. Loops with no clean bound pass `total=None`. Exact call sites are
enumerated in the implementation plan.

## Part 2 — Cotrain resume (R1: reuse base machinery)

`OnlineCotrainRunner(DreamerVLARunner)` already inherits
`BaseRunner.save_checkpoint`/`load_checkpoint`/`resume`. R1 wires it in and
retires the bespoke path.

1. **Save** — replace the torch branch of `_save_cotrain_ckpt()` with the
   inherited `save_checkpoint()`. The base machinery auto-captures every
   `self.__dict__` attribute that has `state_dict`/`load_state_dict`: the four
   modules (`world_model`, `policy`, `critic`, `classifier`) **and** their four
   optimizers (`*_optimizer`) into canonical `checkpoints/latest.ckpt`. Optimizer
   momentum — the thing currently lost — is captured.
2. **Scalars** — set `include_keys = ("global_step", "classifier_threshold")` on
   the class so the two scalars round-trip through the pickle path.
3. **`exclude_keys`** — the base captures *everything* with a `state_dict`, which
   would also pull in frozen/auxiliary attributes (e.g. `encoder`, `ref_policy`).
   Set `exclude_keys` to drop those so the checkpoint stays the four trainable
   components + optimizers + the two scalars — parity with the old bespoke path.
   The exact attribute set is enumerated and verified in the implementation plan.
4. **HF sidecars** — move the existing
   `latest_hf_{world_model,policy,critic,classifier}` export into a
   `_save_checkpoint_sidecars()` override. Both torch and HF are still emitted,
   both through the base save entry point.
5. **DDP unwrap** — current path uses `_unwrap(m).state_dict()`. If
   `DreamerVLARunner` does not already override the state-dict hooks for DDP, the
   cotrain runner overrides `_state_dict_for_checkpoint` /
   `_load_state_dict_from_checkpoint` to unwrap DDP-wrapped modules, matching
   existing `_unwrap` behaviour. Verified during implementation.
6. **Resume** — call `self.resume()` at the start of `run()` (after
   `_build_components`). It honours `training.resume` / `training.resume_dir`,
   finds canonical `checkpoints/latest.ckpt` (or `latest_hf`), restores
   `global_step`, optimizer state, and module weights; the env-step loop
   continues from the restored `global_step`.

Canonical `checkpoints/` is the output location (RLinf alignment); the legacy
`output_dir/ckpt/latest.ckpt` remains resume-only compatibility.

## Testing

- `format_progress_line` (pure): percent rounding, `total=None` branch, duration
  formatting, rate — same style as existing `tests/unit_tests/test_console.py`.
- `ProgressReporter`: injected fake clock + captured sink; assert throttling
  (prints only at interval boundaries + first tick + `close()`), `enabled=False`
  is silent, `total=None` path. No real sleeping.
- `console_progress`: main-process guard; per-`desc` reporter caching.
- Cotrain resume: `save_checkpoint` → `load_checkpoint` round-trips
  `global_step`, `classifier_threshold`, and optimizer state; `resume()` picks up
  canonical `checkpoints/latest.ckpt` and `global_step` continues; HF sidecars
  still written; `exclude_keys` keeps frozen modules out of the payload.
- Smoke: existing cotrain smoke/e2e config runs a few steps → checkpoint →
  resume → continue (AGENTS.md low-cost smoke requirement).

Tests run in the `dreamervla` conda env:
`conda run -n dreamervla python -m pytest tests/unit_tests -q`.

## Follow-on (separate deliverable)

Whole-codebase cleanup review delivered as an item-by-item Markdown audit,
oriented toward the same unification goal. Tracked separately; not part of this
spec.
