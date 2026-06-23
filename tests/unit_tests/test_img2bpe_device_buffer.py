"""PERF-Q9 — `convert_img2bpe` must gather on the input's device.

The vendored Chameleon `ChameleonImageVocabularyMapping.convert_img2bpe`
previously forced `img_batch.to("cpu")` on every call (a D2H copy), gathered on
CPU, then copied the result back (`.to(device)`). Q9 keeps the mapping tensor on
the input's own device so the gather runs there with no per-call host round-trip.

These tests are CPU-only and exercise the real method.
"""

from __future__ import annotations

import torch

from dreamervla.models.embodiment.chameleon_model.chameleon.modeling_chameleon import (
    ChameleonImageVocabularyMapping,
)


def _make_mapping():
    """Build a tiny real mapping: IMGIMG<A..>Z names map digits 0..4 -> ids 100..104."""
    vocab_map = {"<image>": 4}
    for i in range(5):
        vocab_map[f"IMGIMG{chr(ord('A') + i)}Z"] = 100 + i
    return ChameleonImageVocabularyMapping(vocab_map)


def _reference_mapping_tensor(mapping):
    """Upstream CPU build of the mapping tensor, used as the byte-identity oracle."""
    ref = torch.zeros(max(mapping.img2bpe.keys()) + 1, dtype=torch.int)
    for k, v in mapping.img2bpe.items():
        ref[k] = v
    return ref


def test_convert_img2bpe_byte_identical_to_reference_gather():
    mapping = _make_mapping()
    ref = _reference_mapping_tensor(mapping)
    img_batch = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)

    out = mapping.convert_img2bpe(img_batch)
    expected = ref[img_batch]

    assert out.dtype == expected.dtype == torch.int32
    assert out.shape == expected.shape
    assert torch.equal(out, expected)


def test_convert_img2bpe_does_not_force_input_to_cpu():
    """The fix must not do a per-call `.to("cpu")` D2H copy of the input."""
    mapping = _make_mapping()
    img_batch = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    orig_to = torch.Tensor.to
    to_targets: list[object] = []

    def spy_to(self, *args, **kwargs):  # noqa: ANN001
        if args:
            to_targets.append(args[0])
        elif "device" in kwargs:
            to_targets.append(kwargs["device"])
        return orig_to(self, *args, **kwargs)

    torch.Tensor.to = spy_to  # type: ignore[method-assign]
    try:
        out = mapping.convert_img2bpe(img_batch)
    finally:
        torch.Tensor.to = orig_to  # type: ignore[method-assign]

    # No `.to("cpu")` (string or torch.device) should appear on the hot path.
    assert "cpu" not in [str(t) for t in to_targets], (
        f"convert_img2bpe forced a device copy: .to targets were {to_targets}"
    )
    # Result stays on the input's own device.
    assert out.device == img_batch.device


def test_mapping_tensor_lives_on_input_device():
    """The gathered mapping tensor must be built on the input's device (no CPU pin)."""
    mapping = _make_mapping()
    img_batch = torch.tensor([0, 1, 2], dtype=torch.long, device="cpu")
    out = mapping.convert_img2bpe(img_batch)
    assert out.device == img_batch.device


def test_mapping_class_contributes_no_state_dict_keys():
    """`ChameleonImageVocabularyMapping` is a plain class, not an nn.Module.

    It is held by `ChameleonModel` as a plain attribute (not a submodule), so it
    cannot contribute any `state_dict` keys. The per-device cache the fix adds
    must therefore never leak into any checkpoint. We assert the class is not an
    nn.Module and exposes no `state_dict`, which is the structural guarantee that
    `state_dict().keys()` are unchanged vs. upstream.
    """
    mapping = _make_mapping()
    assert not isinstance(mapping, torch.nn.Module)
    assert not hasattr(mapping, "state_dict")
    # Exercising the path must not turn the mapping into a Module or add a
    # `state_dict` surface.
    mapping.convert_img2bpe(torch.tensor([0, 1], dtype=torch.long))
    assert not isinstance(mapping, torch.nn.Module)
    assert not hasattr(mapping, "state_dict")
