# H5 — switchable SDPA path for the DINO-WM chunk attention

## Problem
The DINO-WM chunk world-model attention in
`dreamervla/models/world_model/dino_wm_chunk.py` (`_DinoStyleAttention.forward`,
lines 55-73 at commit 016b900) is a hand-rolled QK^T softmax:

```python
dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # self.scale = dim_head ** -0.5
if mask is not None:
    dots = dots + mask.to(device=dots.device, dtype=dots.dtype)[None, None]
attn = F.softmax(dots, dim=-1)
attn = self.dropout(attn)
out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
return self.to_out(out)
```

This materialises the full `[B, H, S, S]` score tensor and forgoes the fused
flash / memory-efficient SDPA kernels available on GPU. H5 offers a
`torch.nn.functional.scaled_dot_product_attention` (SDPA) path so the WM forward
can use those kernels on GPU, cutting attention memory and time.

## Exact files / lines touched (at 016b900)
- `dreamervla/models/world_model/dino_wm_chunk.py`
  - `_DinoStyleAttention.__init__` (33-53) — add `attn_impl` param + validation + store.
  - `_DinoStyleAttention.forward` (55-73) — add the `sdpa` branch; the `manual`
    branch is left byte-for-byte unchanged.
  - `_DinoStyleTransformer.__init__` (77-105) — add `attn_impl` param, forward it
    to each `_DinoStyleAttention`.
  - `ChunkAwareDinoWMWorldModel.__init__` (130-227) — add `attn_impl` named param
    (popped before `super().__init__`, like `dim_head`), store it, pass it to
    `_DinoStyleTransformer` at the `self.predictor = _DinoStyleTransformer(...)`
    construction (216-223).
- `tests/unit_tests/test_dino_wm_sdpa_equivalence.py` — NEW CPU-only equivalence test.
- `docs/plans/2026-06-23-perf-h5-dino-wm-sdpa.md` — this plan.

No other source files are touched. The parent `DinoWMWorldModel.__init__`
(`dreamervla/models/world_model/dino_wm.py`) takes explicit named params and **no**
`**kwargs`, so `attn_impl` MUST be consumed as an explicit named param on the
chunk subclass before `super().__init__(*args_list, **kwargs)` is called — otherwise
an unknown `attn_impl` kwarg would raise `TypeError`. That is the only reason the
flag is added as a named param rather than left to flow through `**kwargs`.

## The switchable flag + default = manual contract
- New parameter `attn_impl: str = "manual"` on all three classes above.
- `"manual"` ⇒ the EXISTING code path runs verbatim. The manual branch is not
  edited at all, so any existing run, checkpoint, or test that does not set
  `attn_impl` is **numerically identical** to pre-H5 behaviour.
- `"sdpa"` ⇒ route through `F.scaled_dot_product_attention`.
- Any other value raises `ValueError` in `_DinoStyleAttention.__init__`.
- The flag is read from the Hydra WM config the same way `depth` / `heads` /
  `dim_head` already are: `world_model.attn_impl: sdpa` in a `worldmodel/*.yaml`
  arrives as a kwarg to `ChunkAwareDinoWMWorldModel.__init__`. No config file is
  changed by this commit; the default `"manual"` preserves current behaviour, and
  enabling SDPA later is a one-line config (or caller kwarg) change — to be
  validated on GPU.

## scale / mask / dropout mapping (why the math matches)
- **scale.** The manual path multiplies `QK^T` by `self.scale = dim_head ** -0.5`.
  SDPA's default scale is `1/sqrt(q.size(-1)) = 1/sqrt(dim_head)`, which is the
  same value. We nonetheless pass `scale=self.scale` **explicitly** to SDPA so the
  match does not depend on SDPA's default ever changing.
- **mask.** The manual path adds `mask[None, None]` (an additive float bias
  broadcast over batch and head) to the pre-softmax scores. SDPA accepts an
  additive float `attn_mask` that is added to the scaled scores before softmax —
  identical role. We pass the same `mask` broadcast to `[1, 1, S, S]`
  (`mask[None, None]`), cast to the q dtype/device, so SDPA receives exactly the
  bias the manual path applies. `mask=None` ⇒ `attn_mask=None`.
- **dropout.** The manual path applies `nn.Dropout(dropout)` to the post-softmax
  weights. SDPA applies attention-weight dropout via `dropout_p`. We pass
  `dropout_p = (self.dropout.p if self.training else 0.0)` so in `eval()` (or with
  `dropout=0`) both paths apply zero dropout. With a non-zero `dropout` in
  training the two paths draw DIFFERENT RNG masks and will NOT match — expected,
  and out of scope for the equivalence test.
- After SDPA, the output is `.transpose(1, 2).reshape(bsz, seq_len, -1)` then
  `self.to_out(...)` — the identical tail of the manual path, sharing weights.

## Equivalence test design (`test_dino_wm_sdpa_equivalence.py`, CPU-only)
1. Construct `_DinoStyleAttention(..., dropout=0.0, attn_impl="manual")` and
   `_DinoStyleAttention(..., dropout=0.0, attn_impl="sdpa")`; copy the manual
   module's `state_dict` into the sdpa module so weights are identical; call
   `.eval()` on both.
2. **No-mask case:** feed a fixed random `x`, assert
   `torch.allclose(out_manual, out_sdpa, atol=1e-5, rtol=1e-4)`. SDPA and the
   manual matmul use different reduction orders, so they are close-but-not-bit-equal;
   `torch.equal` would be wrong here — a float tolerance is required.
3. **Additive-mask case:** build a small additive float mask of shape `[S, S]`
   (e.g. a causal `-inf`/`0` or random-bias mask), feed it to both, assert
   `torch.allclose(..., atol=1e-5, rtol=1e-4)`.
4. **Manual-is-unchanged guard:** capture a golden output from a manual module via
   the literal pre-change formula (recomputed inline in the test from the same
   weights/inputs) and assert the `attn_impl="manual"` module's output is
   `torch.equal` to that golden — proving the manual branch is byte-for-byte the
   old code.
5. Determinism: seed torch before constructing modules / inputs so the test is
   reproducible on CPU.

## Config plumbing actually threaded
Only the constructor chain inside `dino_wm_chunk.py`:
`ChunkAwareDinoWMWorldModel(attn_impl=...)` → `_DinoStyleTransformer(attn_impl=...)`
→ `_DinoStyleAttention(attn_impl=...)`. The chunk subclass reads `attn_impl` from
its kwargs (Hydra `world_model.attn_impl`), defaulting to `"manual"`. No new YAML
key is added in this commit; the capability lands and is enabled later from config.

## Gate (CPU only — no GPU)
- `conda run -n dreamervla python -m pytest tests/unit_tests/test_dino_wm_sdpa_equivalence.py -q` green.
- `conda run -n dreamervla python -m pytest tests/unit_tests/ -q -k "dino or wm or sdpa or attention"` — no new failures vs clean 016b900.
- `conda run -n dreamervla ruff check dreamervla/models/world_model/dino_wm_chunk.py tests/unit_tests/test_dino_wm_sdpa_equivalence.py` clean.
- ONE commit, `perf(wm): ...`, `--signoff`, no `===` and no `/` in the subject. No push.
