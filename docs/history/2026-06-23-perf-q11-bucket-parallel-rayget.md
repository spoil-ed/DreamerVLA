# PERF-Q11 — batch bucketed `ray.get` into a single call (weight syncer)

## Problem (audit Q11 / §J / §3.7)
`dreamervla/hybrid_engines/weight_syncer/bucket.py` serializes Ray transfers by
calling `ray.get` *inside* the per-bucket loop:
- `push` (line ~80): one blocking `ray.get(self._store.set.remote(...))` per bucket.
- `pull` (line ~95): one blocking `ray.get(self._store.get.remote(...))` per bucket.

Each iteration waits for its own round-trip before issuing the next remote call,
so N buckets cost N serial round-trips. The audit prescribes: issue ALL bucket
refs first, then a single `ray.get([...])` so the per-bucket remote work overlaps.

## Goal
Behavior-identical refactor that batches the independent per-bucket gets:
- Same buckets synced, same final effect (same `set`/`get` keys, same merged
  state, same returned version).
- `ray.get` called ONCE for the bucket batch instead of once-per-bucket.
- Preserve ordering/dependencies: the `set.remote(...)` calls are issued in the
  same bucket order (the `_WeightStore` actor serializes calls in submission
  order, so meta-after-buckets ordering is preserved). No reordering of any
  dependent step.

## Why this is safe
- The `_WeightStore` is a Ray actor; method calls execute serially in the order
  their `.remote()` was *submitted*, regardless of when their refs are awaited.
  Submitting all bucket `set.remote(...)` in the existing loop order, then a
  single `ray.get([...])`, preserves both ordering and final state.
- `push`: bucket `set` calls are mutually independent (distinct keys). The meta
  `set` stays a separate call submitted AFTER the bucket batch, preserving the
  current "meta records num_buckets after buckets are set" sequencing.
- `pull`: bucket `get` reads are mutually independent (distinct keys, no
  cross-bucket dependency). The `None`-check + `merged.update` ordering over
  `range(num_buckets)` is preserved by iterating the resolved list in index
  order.

## Plan
1. Read audit Q11/§J/§3.7 + MEM-RL-01 STYLE. DONE
2. Write this plan doc. DONE
3. TDD (RED to GREEN) — new test file
   `tests/unit_tests/test_weight_syncer_bucket_parallel.py`:
   - Monkeypatch the module-level `ray` in `bucket.py` with a fake that records
     every `ray.get` call and resolves a fake store actor's `set`/`get`.
   - `push`: assert the same per-bucket `set` keys/values land in the fake store
     AND `ray.get` is invoked ONCE for the bucket batch (not once-per-bucket) —
     the call-count assertion is the RED driver.
   - `pull`: round-trip a multi-bucket state_dict through `push`+`pull` (same
     fake store), assert the model loads the identical merged state AND `ray.get`
     batches the bucket reads into one call (call-count RED driver).
   - RED: current per-bucket `ray.get` causes the call-count assertion to fail.
4. Implement: collect bucket refs, one `ray.get([...])` per method. To GREEN.
5. `conda run -n dreamervla python -m pytest <new test> + existing bucket test -q`;
   `conda run -n dreamervla ruff check bucket.py + test`.

## Out of scope
Other audit items (`env_worker.py:85` fire-and-forget, etc.). Strictly
`bucket.py` + new test + this plan doc.
