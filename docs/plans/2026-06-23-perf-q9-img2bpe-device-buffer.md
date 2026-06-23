# PERF-Q9 — keep the `img2bpe` mapping tensor on the input's device

## Problem (audit item Q9)
In the vendored HuggingFace Chameleon code,
`dreamervla/models/embodiment/chameleon_model/chameleon/modeling_chameleon.py`,
the image-token → BPE-token conversion forces a host round-trip on every call:

```python
@cached_property
def img2bpe_mapping_tensor(self):                                   # line 1254-1259
    mapping = torch.zeros(max(self.img2bpe.keys()) + 1, dtype=torch.int)
    for k, v in self.img2bpe.items():
        mapping[k] = v
    return mapping

def convert_img2bpe(self, img_batch: torch.Tensor) -> torch.Tensor:  # line 1261-1264
    device = img_batch.device
    img_tokens = self.img2bpe_mapping_tensor[img_batch.to("cpu")]    # D2H copy of img_batch
    return img_tokens.to(device)                                     # H2D copy of result
```

The `@cached_property` builds the mapping tensor once **on CPU**. Every
`convert_img2bpe` call therefore:

1. copies `img_batch` Device→Host (`.to("cpu")`) — a per-call host sync,
2. gathers on CPU,
3. copies the gathered result Host→Device (`.to(device)`).

This sits on the hot image-token path (`ChameleonModel.get_image_tokens`,
line 1452: `bpe_toks = self.vocabulary_mapping.convert_img2bpe(image_toks)`),
so every batch of images pays two synchronizing transfers plus a CPU gather.

## Exact files / lines
- Target: `dreamervla/models/embodiment/chameleon_model/chameleon/modeling_chameleon.py`
  - `ChameleonImageVocabularyMapping` class — definition at line 1214.
  - `img2bpe_mapping_tensor` `@cached_property` — lines 1254-1259.
  - `convert_img2bpe` — lines 1261-1264.
- Only caller (whole repo): `modeling_chameleon.py:1452`
  (`ChameleonModel.get_image_tokens`).
- Instantiation: `modeling_chameleon.py:1415`
  (`self.vocabulary_mapping = ChameleonImageVocabularyMapping(config.vocabulary_map)`).
- A **separate** implementation lives in
  `chameleon_model/chameleon_vae_ori/vocab.py:107` — out of scope (Q9 is the
  `modeling_chameleon.py` mapping only), and it already indexes without a
  forced CPU copy.

## DEVIATION from the prescribed `register_buffer` approach (read this)
The Q9 task text prescribes registering the mapping as a **non-persistent
buffer** (`self.register_buffer("img2bpe_mapping_tensor", mapping,
persistent=False)`) and asserting `state_dict().keys()` are unchanged.

**That prescription is infeasible here**, because the enclosing class
`ChameleonImageVocabularyMapping` (line 1214) is a **plain Python class — it
does NOT inherit from `torch.nn.Module`**:

```python
class ChameleonImageVocabularyMapping:          # no nn.Module base
    def __init__(self, vocab_map):
        self.vocab_map = vocab_map
        self.image_token_id = vocab_map.get("<image>")
```

Consequences:
- `register_buffer(...)` and `state_dict()` are `nn.Module` APIs and **do not
  exist** on this class — calling `register_buffer` would `AttributeError`.
- The owning `ChameleonModel` holds it as a **plain attribute**
  (`self.vocabulary_mapping = ChameleonImageVocabularyMapping(...)`), which is
  **not** a registered submodule. So even if a buffer existed on the mapping,
  `model.to(device)` would never move it (it is outside the module tree), and
  it would never enter `ChameleonModel.state_dict()`.
- Therefore the mapping tensor is **already absent from every `state_dict`
  today** — the `persistent=False` concern (avoid changing checkpoint keys)
  is automatically satisfied because there is nothing in `state_dict` to
  change, and the fix introduces nothing into `state_dict` either.

Rather than re-architect the vendored class into an `nn.Module` (a large,
numerics-risky divergence from upstream that the minimal-divergence rule
forbids), the fix achieves the **actual Q9 goal — eliminate the per-call
D2H/H2D host round-trip — with the smallest possible edit** by keeping the
mapping tensor on the *input's own device* and gathering there.

## The fix (minimal, device-resident, per-device cache)
Replace the CPU-pinned `@cached_property` + forced `.to("cpu")` with a tiny
helper that builds the mapping tensor **on the requested device** and caches
one tensor per device, then index on the input's device:

```python
def _img2bpe_mapping_tensor(self, device):
    cached = self._img2bpe_mapping_cache.get(device)
    if cached is None:
        cached = torch.zeros(max(self.img2bpe.keys()) + 1, dtype=torch.int, device=device)
        for k, v in self.img2bpe.items():
            cached[k] = v
        self._img2bpe_mapping_cache[device] = cached
    return cached

def convert_img2bpe(self, img_batch: torch.Tensor) -> torch.Tensor:
    return self._img2bpe_mapping_tensor(img_batch.device)[img_batch]
```

- The per-device cache dict `self._img2bpe_mapping_cache = {}` is created in
  `__init__`. The build loop is unchanged byte-for-byte except for the
  `device=` kwarg, so the tensor's **values and `dtype=torch.int` are
  identical** to upstream.
- `convert_img2bpe` now gathers with `img_batch` on its own device and
  returns the result already on that device — **no `.to("cpu")`, no
  `.to(device)` round-trip**. For a CPU `img_batch` the device is `cpu`, the
  cached tensor is built on `cpu`, and the gather is bit-for-bit the same
  fancy-index as before. For a CUDA `img_batch` the gather runs on-device.
- The old `@cached_property` is removed (a property and a same-named regular
  attribute cannot coexist on one class); the only reader of the mapping
  tensor is `convert_img2bpe` (verified repo-wide), so removing the property
  is safe.

## Numeric-identity argument
The gather is `mapping[img_batch]`, an integer fancy-index, identical in both
old and new code:
- Same mapping construction (`torch.zeros(max(keys)+1, dtype=torch.int)` then
  `mapping[k] = v` for every `(k, v)` in `self.img2bpe`) → identical values
  and dtype; only the storage device differs.
- Same index tensor `img_batch` (its `dtype`/values are untouched).
- Same output dtype (`torch.int`) and shape.
The **only** behavioural change is *where* the gather runs (input's device vs.
CPU) and *where* the tensor lives. On CPU the result is byte-identical; on any
device the integer values are identical (an integer gather is exact and
device-independent).

## `state_dict`-unchanged guarantee
Because `ChameleonImageVocabularyMapping` is not an `nn.Module` and is held as
a plain attribute of `ChameleonModel`, neither the old `@cached_property` nor
the new per-device cache appears in `ChameleonModel.state_dict()`. The fix adds
no buffers/parameters/submodules to any `nn.Module`, so `state_dict().keys()`
are unchanged vs. upstream. The test asserts this directly on the real
`ChameleonModel.state_dict()` (built before vs. after exercising the path).

## TDD test (`tests/unit_tests/test_img2bpe_device_buffer.py`, CPU-only)
Exercise the **real** `ChameleonImageVocabularyMapping.convert_img2bpe`:
1. **Byte-identical tokens** — build a reference CPU mapping the upstream way
   (`torch.zeros(max(keys)+1, dtype=torch.int)`; fill) and assert
   `torch.equal(convert_img2bpe(img_batch), reference[img_batch])` and equal
   dtype.
2. **No per-call D2H** — spy on `torch.Tensor.to`; assert `convert_img2bpe`
   does **not** call `.to("cpu")` on the input (the pre-fix code calls it every
   time → this assertion is RED before the fix), and assert the returned
   tensor stays on `img_batch.device`.
3. **`state_dict` keys unchanged** — assert the mapping is not an `nn.Module`
   and that a real `ChameleonModel`'s `state_dict().keys()` are identical
   before and after running `get_image_tokens`/`convert_img2bpe` (i.e. the
   per-device cache never leaks into `state_dict`). A tiny `ChameleonConfig`
   keeps it CPU-cheap; if model construction is too heavy for a unit test we
   assert against a freshly built `ChameleonImageVocabularyMapping`'s absent
   `state_dict` surface (the class has no such method, proving it cannot
   contribute keys).

## Gate
- `conda run -n dreamervla python -m pytest tests/unit_tests/test_img2bpe_device_buffer.py -q` — green.
- `conda run -n dreamervla python -m pytest tests/unit_tests/ -q -k "chameleon or img2bpe or token"` — green (pre-existing failures, if any, confirmed on clean `e792323`).
- `conda run -n dreamervla ruff check <target> <test>` — clean.
- One commit, conventional `perf(chameleon): ...`, `--signoff`, no `===`, no slash in subject. No push.
