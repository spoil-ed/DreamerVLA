# PERF W3 + W4 — pretokenize dataset IO (manifest-first index + per-worker frame cache)

Scope: `dreamervla/dataset/pretokenize_dataset.py` only (+ tests + this plan).
Source audit: `docs/plans/performance_optimization_audit.md` §F and §3.6
(`pretokenize_dataset.py:300-343` = W3, `:374-424` = W4).

Hard invariant for BOTH items: `__init__` index and every `__getitem__` item must be
**byte-identical** to the current code — same frames, same order, same tensors. The
optimizations are pure IO-shape changes (fewer / cached `pickle.load`s), never a
semantic change.

## Schema verified first (on real LIBERO manifest)

`*-record.jsonl` record keys: `file, len, id, meta, reward, next_obs`. The ONLY image-path
field anywhere in a record is `next_obs.image` (also mirrored at `meta.next_obs.image`).
Measured on `libero_spatial_..._record.jsonl`:

- The pkl payload `image` is the **current** frame (`imgs_third_view/image_N.png`).
- The manifest `next_obs.image` is the **next** frame: `next_obs_idx − current_idx` is
  `1` for ~99% of records but `0` at terminal frames (effective_horizon collapse). It is
  NOT a fixed offset and NOT equal to `effective_horizon`.

Consequence: `_index_sequence_records` (W3) keys on the pkl's **current** `image` and then
derives `frame_index`, `action_path` (`action_N.npy`), `reward_path` (`reward_N.npy`).
`_getitem_sequence` emits `frame_index` into `meta_seq` and loads those sibling
action/reward `.npy` files. So the current-frame index N is load-bearing for the output.
The manifest cannot supply N: only `next_obs.image` (N+delta, delta∈{0,1}) is present.

### W3 design decision (surfaced, not silent)

Because the existing manifest lacks the **current** observation image path, a manifest-first
index can only be byte-identical when the manifest carries a current-obs image. We therefore:

1. Add a small helper `_record_current_image(record)` that returns the current-frame image
   list from the manifest IF present, preferring `meta.next_obs.image` / `next_obs.image`
   exactly like `one_trajectory_pretokenize_dataset.py:88`. The manifest-first index is taken
   ONLY when that image parses to a `(task_name, trajectory_key, frame_index)` whose derived
   sibling `action_N.npy` exists on disk — i.e. when it is provably the current frame.
2. When the manifest field is absent OR does not match the current-frame layout (sibling
   action `.npy` missing), fall back to the existing per-pkl `pickle.load` scan (current
   behavior, unchanged).

This keeps the hard byte-identity invariant absolute: the manifest path is taken only when it
reproduces the exact `_FrameRecord` the pickle scan would build (verified by the equality
test). For the shipped LIBERO manifests (which store the *next* frame) the scan path still
runs; emitting a current-obs image into the manifest is a preprocess follow-up (out of scope —
do not touch the writer or roadmap here).

> Net effect this commit: the *mechanism* + a guarded, byte-identical manifest-first path that
> the W3 test exercises with a fixture whose manifest carries the current-frame image, plus the
> always-safe fallback. No behavior change on existing data.

## W4 design (per-worker bounded frame-payload cache)

`_getitem_sequence` `pickle.load`s every frame in the window; stride=1 ⇒ adjacent windows
reload nearly the same frames. Add a per-worker, bounded LRU keyed by frame file path that
caches the loaded payload dict.

- Mirror the repo's per-worker handle-cache idiom (`base_dataset.cached_hdf5_file` + the plain
  `self._file_cache: dict = {}` instance attr in `pixel_sequence_dataset`): a plain instance
  attribute, so the DataLoader fork/spawn gives each worker its own copy (per-worker, never
  shared across processes).
- Bounded: `collections.OrderedDict` LRU with a small cap (`_frame_cache_capacity`, default 64)
  so it cannot grow unbounded; `move_to_end` on hit, `popitem(last=False)` on overflow.
- Byte-identical: cache returns the SAME payload object the `pickle.load` would; no payload
  mutation downstream (existing code copies via `list(...)`/`dict(...)` before use).
- Only `_getitem_sequence` is wrapped (W4 target lines). `PretokenizeActionChunkDataset`
  already has its own `_record_payload_cache` (untouched).

## STYLE (from 2026-06-22-mem-rl-01 plan + repo memory)

- No hardcoded values that decide behavior; the cache cap is a named instance field.
- Surgical: touch only `_index_sequence_records` / `_getitem_sequence` and add small helpers.
- TDD: failing-first tests on a tiny synthetic dataset (pkl frames + manifest under tmp_path).

## Steps (each verifiable)

1. ✅ Read W3/W4 §F/§3.6, `one_trajectory:88`, MEM-RL-01 STYLE. Schema verified (above).
2. RED tests (`tests/unit_tests/test_pretokenize_io_w3w4.py`):
   - W3a: manifest-with-current-image index `==` pickle-scan index (same `_FrameRecord`s,
     same windows, same `__getitem__` items).
   - W3b: spy on `pickle.load`; `__init__` must NOT call it when the manifest carries the
     current-frame image (call-count RED driver) — fallback path still may.
   - W4a: two overlapping windows return identical frames/tensors.
   - W4b: spy on `pickle.load`; a frame shared by two windows is loaded from disk ONCE
     (call-count RED driver).
3. GREEN: implement guarded manifest-first index + per-worker bounded LRU.
4. `conda run -n dreamervla python -m pytest tests/unit_tests/test_pretokenize_io_w3w4.py
   tests/unit_tests/test_one_trajectory_vla_sft_dataset.py tests/unit_tests/test_dataset_public_api.py -q`
   and full suite spot-check; `ruff check` the two files.

## Out of scope
`wm_replay_classifier_dataset.py`; the preprocess writer / roadmap doc; emitting a
current-obs image path into the manifest (preprocess follow-up).
