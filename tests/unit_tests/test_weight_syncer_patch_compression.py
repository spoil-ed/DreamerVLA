"""Ray-free tests for patch and compression weight-sync helpers."""

from __future__ import annotations

import torch

from dreamervla.hybrid_engines.weight_syncer.compression import DTypeTensorCompressor
from dreamervla.hybrid_engines.weight_syncer.patch import state_dict_delta


def test_state_dict_delta_only_keeps_changed_tensors() -> None:
    previous = {
        "same": torch.ones(2),
        "changed": torch.zeros(2),
    }
    current = {
        "same": torch.ones(2),
        "changed": torch.ones(2),
        "new": torch.arange(2),
    }

    delta = state_dict_delta(previous, current)

    assert list(delta) == ["changed", "new"]
    assert torch.equal(delta["changed"], current["changed"])
    assert torch.equal(delta["new"], current["new"])


def test_dtype_compressor_round_trips_float_tensors_and_preserves_ints() -> None:
    state = {
        "float": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        "int": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    compressor = DTypeTensorCompressor(transport_dtype="fp16")

    packed = compressor.compress_state_dict(state)
    assert packed["float"].tensor.dtype is torch.float16
    assert packed["float"].original_dtype is torch.float32
    assert packed["int"].tensor.dtype is torch.int64

    restored = compressor.decompress_state_dict(packed)

    assert restored["float"].dtype is torch.float32
    assert restored["int"].dtype is torch.int64
    assert torch.equal(restored["float"], state["float"])
    assert torch.equal(restored["int"], state["int"])
