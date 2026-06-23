# Perf H9 — Cache Chameleon `_update_causal_mask`

- Status: implemented
- Owner: perf-audit H9
- Date: 2026-06-23
- Scope: `dreamervla/models/embodiment/chameleon_model/chameleon/modeling_chameleon.py`,
  `tests/unit_tests/test_chameleon_mask_cache.py`, this plan doc. Nothing else.
- Constraint: CPU-only correctness proof; the runtime win is realized on GPU and is not measured here.

## Problem

`ChameleonModel._update_causal_mask(...)` is called once per forward (call site:
`modeling_chameleon.py:1533`; definition: `modeling_chameleon.py:1607-1722`, marked
`# Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask`).

For the eager attention path it materializes an `O(L^2)` float causal mask every
forward via `torch.full((sequence_length, target_length), ...)`, `torch.triu`, an
`arange > cache_position` comparison, and a `[None, None] -> expand(batch, 1, -1, -1)`
broadcast (`modeling_chameleon.py:1668-1681`). For Chameleon's long image-token
sequences `L` is large, so this rebuild is a per-forward hotspot.

H9: cache the constructed mask and reuse it when every input that determines it is
unchanged; otherwise build fresh (the original code path).

## Exact files and lines

- Build/return: `dreamervla/models/embodiment/chameleon_model/chameleon/modeling_chameleon.py:1607-1722`
- Call site (per forward): `…/modeling_chameleon.py:1533-1539`
- `cache_position` construction (when `None`): `…/modeling_chameleon.py:1520-1528`
- SDPA attention consumer (uses the returned mask, supports the `None`/`is_causal`
  fast path): `…/modeling_chameleon.py:666-691`
- `ChameleonModel.__init__` (where the cache slots are added): `…/modeling_chameleon.py:1410-1435`

## Exact set of mask-determining inputs

The returned object is a pure function of the following (read directly from the
body of `_update_causal_mask`):

1. `self.config._attn_implementation` — selects `flash_attention_2` (1621),
   `sdpa` (1636/1710), or eager. Different branches, different outputs.
2. `attention_mask` — presence, `dim()`, AND content. Used by the FA2 early return
   (`0.0 in attention_mask`, 1622), by `_ignore_causal_mask_sdpa` (1640), by the 4D
   inverted-mask check (`attention_mask.max()`, 1662), by `target_length`
   (`attention_mask.shape[-1]`, 1655), and by the padding fill (1683-1707).
3. `input_tensor.dtype` — `min_dtype = finfo(dtype).min` (1649) and the mask dtype (1671).
4. `input_tensor.device` — the mask device (1672) and the cuda-only
   `_unmask_unattended` branch (1712).
5. `input_tensor.shape[1]` = `sequence_length` (1650, 1669).
6. `input_tensor.shape[0]` = batch size — `expand(input_tensor.shape[0], 1, -1, -1)` (1680).
7. `past_key_values` — `get_seq_length()` -> `past_seen_tokens` (1629),
   `isinstance(..., StaticCache)` (1632), `get_max_length()` -> `target_length` (1652).
   This object is **mutable**: its length grows across decode steps, so the cache
   MUST read `past_seen_tokens`/`using_static_cache`/`max_length` freshly every call.
8. `cache_position` — its CONTENT, via `arange(target_length) > cache_position.reshape(-1,1)` (1676-1678).
9. `output_attentions` — guards the sdpa branches (1638, 1713).
10. `self.training` — passed to `_ignore_causal_mask_sdpa` (1644).

## Cached vs fall-through

To make a stale/incorrect hit **structurally impossible**, the cache is consulted
ONLY when `attention_mask is None`. Whenever a non-`None` `attention_mask` is
present the function takes the original, uncached path (build fresh, return, store
nothing). This is the "common all-ones / None case" the H9 brief calls acceptable
and still the dominant case: training forwards and same-length rollout forwards on
this model pass `attention_mask=None`, and for those the padding-fill block
(1683-1707), the 4D-mask branch (1660-1666), the FA2 `0.0 in attention_mask` content
test (1622), and the cuda `_unmask_unattended` (1709-1720, requires
`attention_mask is not None`) are all skipped — so there is no padding content to
fingerprint, and the result is determined entirely by cheap scalars plus
`cache_position` content.

With `attention_mask is None`, the determining inputs reduce to:

- `self.config._attn_implementation`
- `input_tensor.dtype`, `str(input_tensor.device)`
- `sequence_length` (`input_tensor.shape[1]`), batch size (`input_tensor.shape[0]`)
- `past_seen_tokens`, `using_static_cache`, and `target_length`
  (`get_max_length()` when static)
- `output_attentions`, `self.training`
- `cache_position` CONTENT

## Cache-key design — complete and cheaper than the build

Single-slot instance cache, initialised in `__init__`:
`self._causal_mask_cache_key = None`, `self._causal_mask_cache_value = None`,
`self._causal_mask_cache_position = None`.

On each call, after the `attention_mask is None` guard, build a cheap scalar tuple
key:

```
key = (
    self.config._attn_implementation,   # branch selector
    str(dtype),                         # dtype -> min_dtype + mask dtype
    str(device),                        # device + cuda branch
    sequence_length,                    # O(1)
    batch_size,                         # O(1) expand size
    past_seen_tokens,                   # O(1) from cache
    using_static_cache,                 # O(1)
    target_length,                      # O(1)
    bool(output_attentions),            # O(1)
    bool(self.training),                # O(1)
    int(cache_position.shape[0]),       # O(1)
)
```

`cache_position` CONTENT is verified separately. On a scalar-key match we confirm
`torch.equal(cache_position, self._causal_mask_cache_position)` — an `O(L)` integer
comparison, far cheaper than the `O(L^2)` `torch.full`/`triu`/`expand` build it
replaces. Only when BOTH the scalar key matches AND `cache_position` is equal do we
return the stored value (which may be the `None` from the sdpa/FA2 ignore path, or
the built float mask). Otherwise we run the ORIGINAL build path unchanged and store
`(key, result, cache_position)`.

Why complete: every determining input is in the key (scalars) or content-verified
(`cache_position`), and the only inputs the key omits — the `attention_mask`
content / dim / 4D-inversion and the padding-fill — cannot affect the result because
the cache is only entered when `attention_mask is None`. `past_key_values`
mutation is captured because `past_seen_tokens`/`using_static_cache`/`target_length`
are recomputed from the live object on every call before the key is formed.

Why cheaper: the key is ~11 Python scalars (each `O(1)`) plus one `O(L)`
`torch.equal`; the build it short-circuits is `O(L^2)` allocation + `triu` +
broadcast-expand + (eager path) a clone. For large `L` this is a strict win, and for
small `L` it is negligible overhead.

## Byte-identity argument

- `attention_mask is not None` -> never consulted, original code runs -> identical.
- `attention_mask is None` and any determining input changed -> scalar key or
  `cache_position` differs -> rebuild via original path -> identical.
- `attention_mask is None` and all determining inputs identical -> the original
  function is deterministic and side-effect-free given those inputs (it allocates
  fresh tensors from `dtype/device/shape/cache_position` only), so the previously
  built object is bit-for-bit what a rebuild would produce -> returning the stored
  object is byte-identical. The `None` ignore-path return is stored and returned as
  `None` identically.

The build logic, dtype, fill value, and the eager/sdpa/FA2 branch decisions are NOT
altered; only the rebuild is short-circuited.

## TDD test (`tests/unit_tests/test_chameleon_mask_cache.py`, CPU-only)

A tiny `ChameleonModel` subclass overrides `__init__` to set only the fields
`_update_causal_mask` touches (`self.config`, `self.training`) plus the three cache
slots, and counts builds by wrapping `torch.full`/`torch.triu` via a spy, so we can
assert a second identical call does NOT rebuild. To compare against "the original
build", a sibling stub that does NOT cache (or a pre-change reference invocation)
produces golden tensors compared with `torch.equal`.

Branches exercised (eager unless noted):

- (a) repeated identical call (`attention_mask=None`): result `torch.equal` to a
  fresh build AND the build spy shows the 2nd call did NOT rebuild. **This is the
  RED assertion before the change** (uncached code rebuilds every call).
- (b) changed `sequence_length` -> rebuilds, byte-identical to fresh build.
- (c) changed `dtype` -> rebuilds, byte-identical.
- (d) non-`None` padding `attention_mask` with changed content -> NOT served from a
  stale all-`None` hit; each call equals its own fresh build (the guard makes a
  stale padding hit impossible because the cache is never consulted here).
- (e) `None`-return ignore path: exercised via the `flash_attention_2` branch
  (1621-1624), which returns `None` for an all-ones / `None` mask without calling
  `_ignore_causal_mask_sdpa`; assert the cached call also returns `None`. (The
  `sdpa` `None`-return at 1640 is NOT cleanly reachable on the installed
  transformers 4.40.1 fork: the vendored call passes `is_training=`, which that
  version's `_ignore_causal_mask_sdpa` signature rejects with `TypeError`. This is a
  pre-existing vendored-tree quirk, independent of H9; the cache still stores/returns
  whatever that branch produces byte-identically.)

## Gate

- `conda run -n dreamervla python -m pytest tests/unit_tests/test_chameleon_mask_cache.py -q` green.
- `conda run -n dreamervla python -m pytest tests/unit_tests/ -q -k "chameleon or mask or causal"` green
  (pre-existing failures, e.g. `test_openvla_oft_*`, checked on clean `1bb5a25`).
- `conda run -n dreamervla ruff check` clean on the two touched code/test files.
- One commit, conventional `perf(chameleon): ...`, `--signoff`, no `===` and no `/` in the subject.
