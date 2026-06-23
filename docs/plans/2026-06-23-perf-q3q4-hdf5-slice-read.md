# PERF-Q3/Q4 — slice-read HDF5 actions per `__getitem__` (drop whole-segment read)

## Problem (audit §1.1 Q3/Q4, §3.6, §F — all ==已核验==)
Two map-style HDF5 datasets read the ENTIRE per-demo `actions` array on every
`__getitem__`, then slice/gather only the few rows they need. With stride-1
windows and per-frame samples, the same demo is read once per window/frame, so
total IO is O(num_windows × T) while the needed data is O(horizon).

- **Q3** `dataset/vla_sft_hdf5_dataset.py:210` (`_action_chunk`):
  `raw = np.asarray(demo["actions"], …)` reads all `T` rows, then gathers
  `raw[min(arange(index, index+H), T-1)]` (last-frame padding for the tail that
  runs past the episode end).
- **Q4** `dataset/pixel_sequence_dataset.py:176,183`:
  `raw_actions = np.asarray(demo["actions"], …)` reads all `T` rows, builds
  `prev_actions` from `raw_actions[start:end-1]`, and
  `current_actions = torch.from_numpy(raw_actions[start:end].copy())`. The
  `.copy()` is needed only because `[start:end]` is a *view* into the
  whole-read array; a direct h5py slice read returns a fresh owned array.

## Goal
Read only the needed `[index:index+H]` / `[start:end]` rows directly from the
HDF5 dataset (`demo["actions"][lo:hi]`), and drop the now-redundant `.copy()`.
Output MUST be **byte-identical** to the current code: same slice, dtype, shape,
last-frame padding at episode boundaries, and the same downstream transforms.

## Why the slice read is equivalent

### Q3 boundary/padding
- `index` ranges over `range(length)` where `length = demo["actions"].shape[0] = T`,
  so `0 <= index < T` always.
- Original: `indices = min(arange(index, index+H), T-1)`; rows `index..min(index+H,T)-1`
  are the real (unclamped) rows, and any tail past `T-1` repeats row `T-1`.
- Slice read: `chunk = demo["actions"][index:index+H]` returns the real rows
  (h5py clamps the upper bound to `T`, so `chunk` has `min(H, T-index)` rows).
  Pad the tail to `H` rows by repeating the LAST row (`chunk[-1]`) — identical
  to clamping indices to `T-1`. `T` is read from `.shape[0]` (HDF5 metadata, no
  data read).
- The padded `(H, A)` array is then fed through the SAME
  `_libero_oft_action_transform` + `_normalize_bounds_q99(...)` as today.

### Q4 boundary/padding + `.copy()`
- Windows are only built when `episode_length >= sequence_length`, and
  `end = start + L <= episode_length`, so `raw_actions[start:end]` and
  `raw_actions[start:end-1]` are fully in-bounds — there is NO clamping/padding.
- `window = demo["actions"][start:end]` (a fresh, h5py-owned `(L, A)` array).
  - `prev_actions[1:] = window[:-1]` (copies into the freshly allocated
    `prev_actions`, exactly as before).
  - `current_actions = torch.from_numpy(window)` — `window` is already a private
    owned array (h5py slice read), so the `.copy()` is redundant and removed.
    `prev_actions` does not alias `window` (it is a separate `np.zeros`), so no
    aliasing assumption is broken.

## Exact change per file
1. `dataset/vla_sft_hdf5_dataset.py` — `_action_chunk`: replace the whole-read +
   `np.minimum`-gather with a slice read `demo["actions"][index:index+H]` plus
   last-row edge padding to `H`, then the unchanged transform/normalize.
2. `dataset/pixel_sequence_dataset.py` — `__getitem__`: replace
   `raw_actions = np.asarray(demo["actions"], …)` with
   `window = np.asarray(demo["actions"][start:end], dtype=np.float32)`; build
   `prev_actions[1:] = window[:-1]`; `current_actions = torch.from_numpy(window)`
   (drop `.copy()`).

## TDD (failing-first equivalence gate)
New test file `tests/unit_tests/test_hdf5_action_slice_read.py`, reusing the
existing tiny-HDF5 fixture pattern (`h5py.File(... 'w')` with `data/demo_*`).

The tests capture the OLD whole-read-then-slice semantics as a reference
function and assert the NEW `__getitem__` / `_action_chunk` output equals it
exactly (`np.array_equal`), at the FIRST window/frame, an interior one, and the
LAST (boundary/padding) one.

- RED first: write the reference + assertions; before editing, they pass against
  current code (this change is correctness-preserving, so the test is a SAFETY
  gate — like `test_wmpo_microbatch_equivalence.py`). To get a genuine RED, the
  test ALSO asserts the implementation no longer materializes the whole array:
  it patches the demo's `actions` dataset access to fail on a full read and
  succeed on a bounded slice, proving only `[lo:hi]` rows are read.

## Equivalence gate / verification
- `conda run -n dreamervla python -m pytest tests/unit_tests/test_hdf5_action_slice_read.py -q`
- Re-run the two existing dataset tests
  (`test_openvla_oft_hdf5_sft.py`, `test_dataset_public_api.py`).
- `conda run -n dreamervla ruff check` on the two changed files + the test.

## Out of scope
Other §3.6/§F items (pretokenize pickle index, replay layout, manifest reuse).
This change touches ONLY the two `actions` reads + the redundant `.copy()`.
