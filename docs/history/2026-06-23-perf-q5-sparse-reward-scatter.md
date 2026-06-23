# PERF-Q5 — vectorize the outcome sparse-reward build with `scatter_`

## Problem (perf audit item Q5 / §H / §3.3)
`_build_reward_tensor` in `dreamervla/algorithms/ppo/outcome.py` (~lines 117-125)
builds the `[B_eff, T_max]` sparse outcome-reward tensor with a Python `for i in
range(batch)` loop that calls `.item()` twice per rollout (`comp[i].item()` and
`finish[i].item()`). For `B_eff` in the hundreds this is hundreds of host-side
scalar reads + a Python-level loop on a hot RL-update path. The audit flags it as
"Med impact, ==已核验==": replace with a single vectorized assignment.

## Exact current behavior (the contract to preserve, bit-for-bit)
```python
reward = torch.zeros((batch, max_steps), dtype=torch.float32)   # CPU
if max_steps <= 0:
    return reward
finish = finish_step.detach().cpu().long().clamp(min=0, max=max_steps - 1)
comp = complete.detach().cpu().bool()
for i in range(batch):
    if comp[i].item():
        reward[i, finish[i].item()] = 1.0
return reward
```
Semantics: for every rollout `i` with `complete[i] == True`, write `1.0` at column
`finish[i]` (a clamped env-step index); leave everything else `0.0`. Output shape
`[batch, max_steps]`, dtype `float32`, on CPU. Rows with `complete[i] == False`
stay all-zero regardless of their `finish[i]`.

## Exact change (scope: ONLY this construction)
Replace the loop with a single `scatter_` along dim 1. The value written is
`float(complete)` at the finish column, so encode the per-row write value as
`comp.float()` and scatter it into a zero tensor at `finish` (unsqueezed to a
column index). Rows where `comp` is False scatter `0.0`, which is a no-op against
the zero base — identical to skipping them in the loop.

```python
reward = torch.zeros((batch, max_steps), dtype=torch.float32)
if max_steps <= 0:
    return reward
finish = finish_step.detach().cpu().long().clamp(min=0, max=max_steps - 1)
comp = complete.detach().cpu().bool()
reward.scatter_(1, finish.unsqueeze(1), comp.float().unsqueeze(1))
return reward
```
Why this is numerically identical:
- `scatter_(1, index[:,0:1], src[:,0:1])` writes `src[i,0]` into `reward[i,
  finish[i]]` for every row — exactly one column per row, the same column the loop
  used (`finish[i]`).
- `src = comp.float()`: `1.0` where complete, `0.0` otherwise. Writing `0.0` into a
  zero base leaves the row all-zero, matching the loop's `if comp[i]` skip.
- No row has two writes to the same column (one index per row), so there is no
  scatter accumulation/order ambiguity. Output is bit-for-bit equal (atol=0).

Nothing else in `outcome.py` changes — micro-batch slicing, advantage, PPO loss,
and metrics are untouched.

## TDD steps
1. **RED** — add `tests/unit_tests/test_outcome_sparse_reward_scatter.py` with an
   independent reference oracle (the ORIGINAL loop, kept inline in the test) and
   assert `_build_reward_tensor(...)` equals it with `torch.equal` (atol=0) across
   a representative input: mixed complete/incomplete rows, varied finish indices
   incl. boundary `0` and `max_steps-1`, an out-of-range finish that must clamp,
   and an incomplete row whose finish is non-zero (must stay all-zero). Also cover
   the `max_steps <= 0` early-return. Run FIRST against the current loop to confirm
   the oracle matches the loop (sanity); then this same test is the equivalence
   gate the `scatter_` rewrite must keep green.
2. **GREEN** — replace the loop with `scatter_`. Re-run → pass.
3. **REGRESSION** — run the two existing outcome.py equivalence suites
   (`test_wmpo_microbatch_equivalence.py`, `test_wmpo_slice_latent.py`) plus the
   new test; they exercise `_build_reward_tensor` through `dino_wmpo_outcome_step`
   and must stay green.

## Equivalence gate
`conda run -n dreamervla python -m pytest
tests/unit_tests/test_wmpo_microbatch_equivalence.py
tests/unit_tests/test_wmpo_slice_latent.py
tests/unit_tests/test_outcome_sparse_reward_scatter.py -q`
then `conda run -n dreamervla ruff check dreamervla/algorithms/ppo/outcome.py
tests/unit_tests/test_outcome_sparse_reward_scatter.py`.

## Out of scope
Every other `outcome.py` `.item()`/loop (audit rows A, BC, ratio-list) and all
MEM-RL-01 micro-batch logic. This change is the sparse-reward construction only.
