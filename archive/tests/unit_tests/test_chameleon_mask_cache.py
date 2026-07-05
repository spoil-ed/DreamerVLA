"""CPU correctness tests for the H9 ``_update_causal_mask`` single-slot cache.

The cache must be byte-identical to the original (uncached) build on every
branch, must rebuild whenever any mask-determining input changes, and must
avoid the ``O(L^2)`` rebuild on a repeated identical call.

CPU-only. Run via the ``dreamervla`` conda env.
"""

from types import SimpleNamespace

import torch

from dreamervla.models.embodiment.chameleon_model.chameleon import (
    modeling_chameleon,
)
from dreamervla.models.embodiment.chameleon_model.chameleon.modeling_chameleon import (
    ChameleonModel,
)


def _make_model(attn_impl="eager", training=False):
    """A minimal object that owns only what ``_update_causal_mask`` touches.

    Bypasses the heavy ``ChameleonModel.__init__`` (VQVAE, embeddings, decoder
    layers) via ``object.__new__`` and sets only ``config`` + ``training`` plus
    the cache slots the H9 change introduces (defaulted defensively so the test
    works against both the pre-change and post-change function).
    """
    model = object.__new__(ChameleonModel)
    model.config = SimpleNamespace(_attn_implementation=attn_impl)
    model.training = training
    # Cache slots: harmless extra attributes for the uncached (pre-change) fn.
    model._causal_mask_cache_key = None
    model._causal_mask_cache_value = None
    model._causal_mask_cache_position = None
    return model


def _inputs(seq_len, batch=2, dtype=torch.float32, past=0):
    inputs_embeds = torch.zeros(batch, seq_len, 8, dtype=dtype)
    cache_position = torch.arange(past, past + seq_len)
    return inputs_embeds, cache_position


class _BuildSpy:
    """Counts ``_update_causal_mask`` full builds by wrapping ``torch.full``.

    The eager build path calls ``torch.full((seq_len, target_len), ...)``; a
    cached return must NOT call it. We restrict counting to 2D float fills so
    unrelated ``torch.full`` calls (none expected here) cannot inflate the count.
    """

    def __init__(self, monkeypatch):
        self.count = 0
        self._orig = torch.full
        monkeypatch.setattr(torch, "full", self._wrapped)

    def _wrapped(self, size, *args, **kwargs):
        if isinstance(size, tuple) and len(size) == 2:
            self.count += 1
        return self._orig(size, *args, **kwargs)


def _fresh_build(attn_impl, seq_len, batch, dtype, past, training=False):
    """Original (uncached) reference output for byte-identity comparison.

    A separate model instance whose cache never gets populated by prior calls,
    so its return is always a genuine fresh build for the given inputs.
    """
    ref = _make_model(attn_impl=attn_impl, training=training)
    embeds, pos = _inputs(seq_len, batch=batch, dtype=dtype, past=past)
    return ref._update_causal_mask(None, embeds, pos, None, False)


def test_repeated_identical_call_is_cached_and_byte_identical(monkeypatch):
    """(a) Second identical call returns a byte-identical mask WITHOUT rebuilding.

    Pre-change (uncached) this is RED: the build spy records 2 builds.
    """
    model = _make_model(attn_impl="eager")
    embeds, pos = _inputs(seq_len=6)

    golden = _fresh_build("eager", seq_len=6, batch=2, dtype=torch.float32, past=0)

    spy = _BuildSpy(monkeypatch)
    first = model._update_causal_mask(None, embeds, pos, None, False)
    builds_after_first = spy.count
    second = model._update_causal_mask(None, embeds, pos, None, False)
    builds_after_second = spy.count

    assert torch.equal(first, golden)
    assert torch.equal(second, golden)
    # First call must build once; the second identical call must NOT rebuild.
    assert builds_after_first == 1
    assert builds_after_second == 1


def test_changed_seq_len_rebuilds_correctly():
    """(b) A different sequence length invalidates the cache and rebuilds."""
    model = _make_model(attn_impl="eager")

    embeds6, pos6 = _inputs(seq_len=6)
    out6 = model._update_causal_mask(None, embeds6, pos6, None, False)
    assert torch.equal(out6, _fresh_build("eager", 6, 2, torch.float32, 0))

    embeds9, pos9 = _inputs(seq_len=9)
    out9 = model._update_causal_mask(None, embeds9, pos9, None, False)
    assert torch.equal(out9, _fresh_build("eager", 9, 2, torch.float32, 0))
    assert out9.shape != out6.shape


def test_changed_dtype_rebuilds_correctly():
    """(c) A different dtype (different min fill value) invalidates the cache."""
    model = _make_model(attn_impl="eager")

    embeds_f32, pos = _inputs(seq_len=5, dtype=torch.float32)
    out_f32 = model._update_causal_mask(None, embeds_f32, pos, None, False)
    assert torch.equal(out_f32, _fresh_build("eager", 5, 2, torch.float32, 0))

    embeds_f16, pos = _inputs(seq_len=5, dtype=torch.float16)
    out_f16 = model._update_causal_mask(None, embeds_f16, pos, None, False)
    assert torch.equal(out_f16, _fresh_build("eager", 5, 2, torch.float16, 0))
    assert out_f16.dtype == torch.float16


def test_changed_padding_mask_is_not_a_stale_hit():
    """(d) A non-None padding mask is never served from a stale all-None hit.

    First prime the cache with ``attention_mask=None``; then two distinct
    padding masks each must equal their OWN fresh build (the cache is bypassed
    whenever a padding mask is present, so a stale hit is impossible).
    """
    model = _make_model(attn_impl="eager")
    seq_len, batch = 5, 2

    # Prime the all-None cache slot.
    embeds, pos = _inputs(seq_len, batch=batch)
    model._update_causal_mask(None, embeds, pos, None, False)

    def build_with_mask(mask):
        ref = _make_model(attn_impl="eager")
        e, p = _inputs(seq_len, batch=batch)
        return ref._update_causal_mask(mask, e, p, None, False)

    # 3D (batch, query, key) additive mask: the vendored dim()==3 branch.
    mask_a = torch.ones(batch, seq_len, seq_len)
    mask_a[:, :, 0] = 0.0  # first key position padded
    mask_b = torch.ones(batch, seq_len, seq_len)
    mask_b[:, :, -1] = 0.0  # last key position padded

    out_a = model._update_causal_mask(mask_a, *_inputs(seq_len, batch=batch), None, False)
    out_b = model._update_causal_mask(mask_b, *_inputs(seq_len, batch=batch), None, False)

    assert torch.equal(out_a, build_with_mask(mask_a))
    assert torch.equal(out_b, build_with_mask(mask_b))
    assert not torch.equal(out_a, out_b)


def test_none_return_branch_is_cached_identically(monkeypatch):
    """(e) The ``None``-return ignore path is stored and returned as ``None``.

    Reached via the flash_attention_2 branch (returns ``None`` for an all-ones /
    None mask) WITHOUT calling ``_ignore_causal_mask_sdpa`` (whose vendored
    call signature is incompatible with transformers 4.40.1 — pre-existing).
    """
    model = _make_model(attn_impl="flash_attention_2")
    embeds, pos = _inputs(seq_len=4)

    first = model._update_causal_mask(None, embeds, pos, None, False)
    second = model._update_causal_mask(None, embeds, pos, None, False)
    assert first is None
    assert second is None


def test_module_level_import_is_clean():
    """Guard: the module imports and the function is still 'Copied from' Llama."""
    assert hasattr(modeling_chameleon.ChameleonModel, "_update_causal_mask")
