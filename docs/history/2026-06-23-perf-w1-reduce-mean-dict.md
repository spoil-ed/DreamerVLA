# PERF-W1 (Q8) — batch `reduce_mean_dict` into a single all_reduce

## Problem (audit Q8 / roadmap W1 / §A)
`NopretokenizeSFTDistributedHelper.reduce_mean_dict` at
`dreamervla/runners/distributed.py:212-213` currently does:

```python
def reduce_mean_dict(self, metrics: dict[str, float | int]) -> dict[str, float]:
    return {key: self.reduce_mean(value) for key, value in metrics.items()}
```

Each key fans out to `reduce_mean`, which issues **one `dist.all_reduce`** and
**one blocking `.item()`** per key. The DreamerVLA runners assemble large metric
dicts (`dreamervla_runner.py` builds 80+ keys) and reduce them every sync, so a
single log/sync step launches dozens of tiny collectives plus dozens of
device→host syncs on the hot path. Each `.item()` forces a GPU sync, serializing
the loop.

## Existing pattern to REUSE (do NOT invent)
The correct batched reduction already lives in the repo at
`dreamervla/runners/dreamerv3_pixel_runner.py:262-278` (`_reduce_metrics`):

```python
def _reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
    if not self.use_ddp:
        return metrics
    keys = list(metrics.keys())
    if not keys:
        return metrics
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        device=self.device,
        dtype=torch.float32,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(self.world_size)
    return {
        key: float(value)
        for key, value in zip(keys, values.detach().cpu().tolist(), strict=True)
    }
```

Approach: **stack all values into one float32 tensor → single `all_reduce(SUM)` →
divide by `world_size` → split back into the dict (one D2H via `.tolist()`)**.

## Exact change (scope: `distributed.py` only)
Rewrite `reduce_mean_dict` (currently lines 212-213) to mirror the pixel-runner
pattern, adapted to this helper's fields:

- Use `self.is_distributed` (this helper's guard) instead of `use_ddp`.
- Use `self._reduce_device()` (this helper's device resolver, the same one
  `reduce_mean` uses) instead of `self.device`.
- Empty dict → `{}`.
- **Always materialize the float32 tensor** (do NOT short-circuit the
  non-distributed path with raw Python doubles): `reduce_mean` round-trips every
  value — even when `world_size == 1` — through a `torch.float32` tensor, so its
  output for e.g. `3e-4` is `0.0003000000142492354`, not `0.0003`. To stay
  *numerically identical*, the batched path must round-trip through float32 too;
  it just skips the `all_reduce` + `/= world_size` when not distributed.

Resulting body:

```python
def reduce_mean_dict(self, metrics: dict[str, float | int]) -> dict[str, float]:
    keys = list(metrics.keys())
    if not keys:
        return {}
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        device=self._reduce_device(),
        dtype=torch.float32,
    )
    if self.is_distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= float(self.world_size)
    return dict(zip(keys, values.detach().cpu().tolist(), strict=True))
```

No other code is touched. `reduce_mean` / `reduce_sum` stay as-is (single-scalar
reducers, not the hot loop the audit flagged).

## Equivalence (numerically identical to the per-key path)
- Per-key today: `float(value)` → SUM all_reduce → `/ world_size` → `float`.
- Batched: `float(value)` for every key → SUM all_reduce on the stacked vector →
  `/ world_size` (same scalar divisor, same float32 dtype/device) → per-key
  `float`. Element-wise the arithmetic is identical; stacking only changes the
  number of collectives, not the values.
- Non-distributed path: today `reduce_mean` returns
  `float(torch.tensor(float(value), float32).item())`; batched path returns the
  same value via `tensor(..., float32).tolist()`. Identical (incl. the float32
  rounding — see the "always materialize the float32 tensor" note above).

## TDD steps (each verifiable)
RED-driver: the distributed all_reduce cannot be exercised without spawning
ranks, so the test drives RED on the load-bearing behavioral change — **the
number of collectives** — plus pins the means on the unit-testable
`world_size == 1` path.

1. RED — add `tests/unit_tests/test_reduce_mean_dict_batched.py`:
   - `world_size == 1`: `reduce_mean_dict` returns `{key: float(value)}` for the
     input — means unchanged, every value a `float`.
   - Collective count: build a helper with `world_size > 1`, monkeypatch
     `_reduce_device` → CPU and `dist.all_reduce` → a counting/identity stub,
     and assert `all_reduce` is called **exactly once** for a multi-key dict.
     Old per-key code calls it N times → this assertion FAILS (RED). New batched
     code calls it once → PASSES (GREEN).
   - Equivalence: with the stubbed `all_reduce` as identity (single-rank sum) and
     `world_size == 1`, assert the batched result equals a reference per-key
     reduction `{k: float(v) for ...}` exactly (dict equality + float type).
   - Empty dict → `{}`.
2. Run → MUST FAIL on the collective-count assertion (old code = N calls).
3. Implement the rewrite above → GREEN.
4. `conda run -n dreamervla python -m pytest tests/unit_tests/test_reduce_mean_dict_batched.py -q`
   then `conda run -n dreamervla ruff check dreamervla/runners/distributed.py tests/unit_tests/test_reduce_mean_dict_batched.py`.

## Equivalence gate
Means on `world_size == 1` are unchanged (pins values), the distributed divisor /
dtype / device match the per-key path (arithmetic identity), and exactly one
`all_reduce` is issued per call (the behavioral win, asserted by the
mocked-collective count test).

## Out of scope
`reduce_mean` / `reduce_sum` single-scalar reducers; any call-site changes;
gating dict assembly behind `log_every` (a separate audit item, not Q8).
