# PERF-Q10 — vectorize the `bin_centers` action decode with fancy indexing

## Problem (audit item Q10, §H, §3.5)
Two VLA embodiment variants decode discretized action tokens into normalized
action values with a Python list-comprehension over the batch dimension:

```python
normalized_actions = np.asarray(
    [self.bin_centers[da] for da in discretized_actions]
)  # [B, dim]
```

- `dreamervla/models/embodiment/openvla/openvla_action_model.py:699-701`
- `dreamervla/models/embodiment/openvla_oft/dreamervla/openvla_oft_action_model.py:397-399`

The list-comp iterates over the rows of `discretized_actions` and re-indexes
`bin_centers` once per row, then `np.asarray` re-stacks. This is a redundant
Python loop where a single NumPy fancy-index does the same work.

The **official** OFT variant
(`openvla_oft/official/openvla_oft_action_model.py:404`) already does the
correct, vectorized form — it is the reference:

```python
normalized_actions = self.bin_centers[discretized_actions]  # [B, dim]
```

## Goal
Replace the list-comp with `self.bin_centers[discretized_actions]` in the two
non-official variants, **numerically identical** (same values, same dtype,
same shape, same row/column ordering).

## Key facts that make it correct (verified in `dreamervla` env)
- `self.bins = np.linspace(-1, 1, n_action_bins)` (1-D float64);
  `self.bin_centers = (bins[:-1] + bins[1:]) / 2.0` → 1-D `float64`, shape
  `(n_action_bins - 1,)`.
- `discretized_actions` is a 2-D **integer** array of shape `[B, dim]` after
  `np.clip(discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0]-1)`,
  so every index is in-bounds for `bin_centers`.
- List-comp semantics: iterating `for da in discretized_actions` yields each
  **row** `da` (shape `[dim]`); `bin_centers[da]` is itself fancy indexing →
  `[dim]`; `np.asarray([...])` stacks the rows → `[B, dim]`, `float64`.
- Fancy-index semantics: indexing a 1-D array with a 2-D int index array,
  `bin_centers[discretized_actions]`, produces an array of the index array's
  shape `[B, dim]`, `float64`, with `out[i, j] = bin_centers[discretized_actions[i, j]]`.
- These are mathematically identical element-by-element, with identical dtype
  and shape — confirmed by a standalone numpy check (incl. edge bins 0 and
  `bin_centers.shape[0]-1`): `np.array_equal` True, dtypes equal, shapes equal.

### OFT-variant reshape note
The OFT dreamervla variant follows the list-comp with
`normalized_actions = normalized_actions.reshape(-1, self.action_dim)`. Because
`discretized_actions` there is already `chunk_action_tokens.reshape(-1, action_dim)`
(shape `[B, action_dim]`), the list-comp already returns `[B, action_dim]` and
the subsequent reshape is a no-op identity. The fancy-index form returns the
same `[B, action_dim]`, so the **existing reshape line stays untouched** and the
result is unchanged. Scope: only the list-comp line is replaced; the reshape and
everything else are left exactly as-is.

## Test-import constraint (worktree/local-env artifact)
Both target modules import the vendored `prismatic` tree, which is **not
installed** in this worktree env:
`ModuleNotFoundError: No module named 'prismatic'`. Per the task rules this is a
known-ignorable import failure. The decode is the same pure NumPy mapping in all
three variants, so — per the rules' fallback — the equivalence is tested at the
lowest reachable level: a standalone test that builds `bin_centers` exactly as
the models do and asserts the vectorized fancy-index EQUALS the list-comp
reference for representative `discretized_actions` (incl. edge bins 0 and max),
`atol=0`, same dtype, same shape. This directly guards the exact substitution.

## Steps (each verifiable)
1. TDD RED: add `tests/unit_tests/test_q10_bin_centers_vectorize.py` asserting
   `bin_centers[discretized_actions]` == `np.asarray([bin_centers[da] for da in
   discretized_actions])` with `np.array_equal` (atol=0), equal dtype, equal
   shape, over representative 2-D int `discretized_actions` including edge bins
   `0` and `bin_centers.shape[0]-1`, and over the real `n_action_bins` (256).
   The test compares the reference (list-comp) and the new form (fancy-index)
   head-to-head. → verify: run the test (passes immediately because both forms
   are pure numpy; this is a SAFETY gate proving the substitution is exact, not
   a behavior-change driver — the change is a perf/no-op refactor).
2. Implement: replace the list-comp with `self.bin_centers[discretized_actions]`
   in both target files; leave the trailing `[B, dim]` comment / OFT reshape and
   all surrounding code untouched. → verify: `git diff` shows only the two
   decode lines changed.
3. Run the test in `dreamervla` env → GREEN.
4. `ruff check` the changed files in `dreamervla` env → clean.
5. One commit (plan + code + test), conventional + signoff, no `===` or `/` in
   subject. Do NOT push.

## Out of scope
The HF `generate()` per-token path, the single-forward logic, the two `-inf`
writes before cross-entropy (a separate §3.5 row), the roadmap/backlog docs, the
official variant (already correct), and every other audit item.
