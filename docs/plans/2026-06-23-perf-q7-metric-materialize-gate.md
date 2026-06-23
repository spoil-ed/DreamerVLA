# PERF-Q7 — gate offline DreamerV3 metric materialization behind `log_every`

## Problem (audit Q7)
The two offline DreamerV3 world-model runners build a per-step `row` dict that
materializes every loss/metric scalar to host on **every** training step, even
though `row` is consumed only at logging boundaries. The host pulls force a
GPU→host sync each step, serializing the training loop.

Sites (at base `4aa4346`):

- `dreamervla/runners/dreamerv3_pixel_runner.py:394-406` — the `row` build:
  - `:398` `"grad_norm": float(grad_norm)` (`grad_norm` is the tensor returned
    by `clip_grad_norm_`; `float(...)` syncs).
  - `:401-404` `**self._reduce_metrics({k: float(v.detach().float().mean().cpu())
    for k, v in out.items() if k != "_loss"})` — one `.cpu()` + `float()` per
    metric tensor, every step. (`_reduce_metrics` at `:262-278` then adds a
    `.detach().cpu().tolist()` D2H, but it runs only as part of this `row`
    build.)
- `dreamervla/runners/dreamerv3_token_runner.py:325-335` — the same pattern:
  - `:330` `"grad_norm": float(grad_norm)`.
  - `:331-334` `**{k: float(v.detach().float().mean().cpu()) for k, v in
    out.items() if k != "_loss"}`.

## Materialization-site classification
| Site | Loc | Class | Reason |
|---|---|---|---|
| `float(grad_norm)` | pixel:398, token:330 | log-only → gate | only stored into `row` |
| metric dict-comp `float(v...cpu())` | pixel:401-404, token:331-334 | log-only → gate | only stored into `row` |
| `_reduce_metrics` `.cpu().tolist()` | pixel:277 | log-only → gate | runs only inside the `row` build |
| `loss = out["_loss"].mean()` | pixel:386, token:317 | leave | feeds `loss.backward()`; never pulled to host (`_loss` is excluded from the row comp) |
| `_tensor_to_pil` `.cpu()` | pixel:141 | leave | viz path, already gated by `viz.every_n_steps`; out of Q7 scope |

**Critical every-step check (confirmed safe to gate):**
- `loss` is used for `backward()` but is NOT materialized to host and is NOT in
  `row` (`if k != "_loss"`). Untouched.
- `grad_norm` is used only inside `row`; there is no NaN / early-stop / scheduler
  use of it.
- The LR warmup scheduler (`pixel:368-373`, `token:301-306`) keys off the integer
  `self.global_step`, not any metric scalar.
- `console_progress(self.global_step, total, "train")` (`pixel:407`,
  `token:336`) takes integers only — it never consumes `row` or a metric tensor.
- `console_metrics` / `log_metrics` consume `row`, but only inside the existing
  log gate. No accumulator / running mean over the synced scalars exists.

So the entire `row` build is purely-for-logging and may move inside the log gate.

## The log gate being reused (already in the runners)
- Pixel (`:408-417`):
  ```python
  if (self.is_main_process and log_handle is not None
          and log_every > 0 and self.global_step % log_every == 0):
      log_handle.write(json.dumps(row) + "\n"); log_handle.flush()
      self.log_metrics(row, step=self.global_step)
      self.console_metrics(...)
  ```
- Token (`:337`): `if log_every > 0 and self.global_step % log_every == 0:`
  (no main-process / handle guard; this runner is single-process).

## The change (scope: the two runners only)
Hoist the gate predicate to a local `should_log` computed BEFORE the `row` build,
then build `row` (the only consumer of the materialized scalars) only when
`should_log`. `console_progress(...)` still runs every step (it needs no `row`).

Pixel:
```python
should_log = (
    self.is_main_process
    and log_handle is not None
    and log_every > 0
    and self.global_step % log_every == 0
)
self.console_progress(self.global_step, total_steps, "train")
if should_log:
    row = { ... unchanged body ... }
    log_handle.write(json.dumps(row) + "\n"); log_handle.flush()
    self.log_metrics(row, step=self.global_step)
    self.console_metrics(f"train · epoch {self.epoch}", {f"train/{k}": v for k, v in row.items()})
```

Token (mirror, with its single-process gate):
```python
should_log = log_every > 0 and self.global_step % log_every == 0
self.console_progress(self.global_step, progress_total, "train")
if should_log:
    row = { ... unchanged body ... }
    log_handle.write(json.dumps(row) + "\n"); log_handle.flush()
    self.log_metrics(row, step=self.global_step)
    self.console_metrics(f"train · epoch {self.epoch}", {f"train/{k}": v for k, v in row.items()})
```

The `row` body (keys, `float(...)`, `_reduce_metrics`, dict-comp) is byte-for-byte
unchanged — only its *invocation* moves behind the gate that already governs its
sole consumers. `_maybe_save_viz` and `_save_ckpt` calls are unchanged.

## Equivalence
On logging steps the `row` dict is identical to today (same code, same inputs).
On non-logging steps `row` was previously built and then discarded — nothing read
it — so skipping its construction changes no observable behavior. The only removed
work is the per-step host sync. LOGGED values are byte-identical.

## TDD test design (`tests/unit_tests/test_dreamerv3_metric_log_gate.py`)
Drive the REAL `run()` loop on CPU. The runner is constructed via `__new__` to
bypass the CUDA-defaulting `__init__`; only the attributes `run()` reads are set,
and the heavy collaborators are stubbed so a few CPU steps execute:

- `hydra.utils.instantiate` → returns a fake dataset, then a fake model whose
  `__call__` returns `{"_loss": tensor, <metric>: SyncCountingTensor}`.
- `_make_loader` → a tiny list of fake CPU batches (loop runs N>log_every steps).
- `_maybe_resume` → `False`; `_save_ckpt`, `_maybe_save_viz`, `console_*`,
  `log_metrics` → no-op / capture.
- `_reduce_metrics` → identity (single process), so the count of host syncs is
  the dict-comp's `.cpu()` calls.

`SyncCountingTensor` is a `torch.Tensor` subclass that increments a module-level
counter inside `.cpu()`. Assertions:
1. **Gating (RED before fix):** the metric tensor's `.cpu()` is invoked ONLY on
   log-boundary steps — count == number of steps with `global_step % log_every
   == 0`. On base code it fires every step → RED.
2. **Equivalence:** captured logged `row` values on the gated path equal the
   eager-reference values for the same step (means/grad_norm unchanged).

If real-loop wiring proves infeasible at unit scale, fall back to extracting the
gate predicate, but the production change above is preferred (it is minimal and
the test exercises the genuine loop).

## Test / ruff gate
- `conda run -n dreamervla python -m pytest tests/unit_tests/test_dreamerv3_metric_log_gate.py -q` → green.
- `conda run -n dreamervla python -m pytest tests/unit_tests/ -q -k "dreamerv3 or metric or log"` → no regressions vs `4aa4346`.
- `conda run -n dreamervla ruff check dreamervla/runners/dreamerv3_pixel_runner.py dreamervla/runners/dreamerv3_token_runner.py tests/unit_tests/test_dreamerv3_metric_log_gate.py` → clean.

## Out of scope
`_dreamer_runner_common.py`, `dreamervla_runner.py`, any other runner; the viz
`.cpu()` (already gated); the `_reduce_metrics` collective batching (that is Q8).
