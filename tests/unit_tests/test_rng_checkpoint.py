"""RNG state capture/restore for bit-exact runner resume."""

import random

import numpy as np
import pytest
import torch


def test_set_seed_reproduces_python_numpy_and_torch_draws():
    from dreamervla.utils.seed import set_seed

    set_seed(4242)
    expected = (random.random(), np.random.random(), torch.rand(()))
    set_seed(4242)
    actual = (random.random(), np.random.random(), torch.rand(()))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_set_seed_and_restore_rng_state_reproduce_all_cpu_draws():
    from dreamervla.utils.seed import capture_rng_state, restore_rng_state, set_seed

    set_seed(20260715)
    state = capture_rng_state()
    ref = (
        [random.random() for _ in range(4)],
        np.random.random(4),
        torch.rand(4),
    )

    set_seed(123456)

    restore_rng_state(state, strict=True)
    got = (
        [random.random() for _ in range(4)],
        np.random.random(4),
        torch.rand(4),
    )

    assert got[0] == ref[0]
    np.testing.assert_array_equal(got[1], ref[1])
    torch.testing.assert_close(got[2], ref[2], rtol=0, atol=0)


def test_restore_rng_state_tolerates_missing_or_none_payload():
    from dreamervla.utils.seed import restore_rng_state

    # Backward compatibility: old checkpoints have no "rng" key.
    restore_rng_state(None)
    restore_rng_state({})  # partial / empty payloads must not raise


@pytest.mark.parametrize("missing", ["python", "numpy", "torch", "cuda"])
def test_restore_rng_state_strict_rejects_each_missing_key(missing):
    from dreamervla.utils.seed import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    state.pop(missing)

    with pytest.raises(RuntimeError, match="missing keys"):
        restore_rng_state(state, strict=True)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("python", "invalid"),
        ("numpy", "invalid"),
        ("torch", "invalid"),
        ("cuda", "invalid"),
    ],
)
def test_restore_rng_state_strict_rejects_invalid_types(key, bad_value):
    from dreamervla.utils.seed import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    state[key] = bad_value

    with pytest.raises(RuntimeError, match=key):
        restore_rng_state(state, strict=True)


@pytest.mark.parametrize("key", ["python", "numpy"])
def test_restore_rng_state_strict_wraps_malformed_tuple_state(key):
    from dreamervla.utils.seed import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    state[key] = ()

    with pytest.raises(RuntimeError, match=key):
        restore_rng_state(state, strict=True)


def test_restore_rng_state_strict_rejects_cuda_topology_mismatch(monkeypatch):
    from dreamervla.utils.seed import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    state["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="topology mismatch"):
        restore_rng_state(state, strict=True)


def test_select_rank_rng_state_handles_rank_lists_single_mapping_and_invalid_inputs():
    from dreamervla.utils.seed import capture_rng_state, select_rank_rng_state

    rank_zero = capture_rng_state()
    rank_one = capture_rng_state()

    assert select_rank_rng_state([rank_zero, rank_one], 1) is rank_one
    assert select_rank_rng_state([rank_zero], 1) is None
    assert select_rank_rng_state([rank_zero], -1) is None
    assert select_rank_rng_state(["invalid"], 0) is None
    assert select_rank_rng_state(rank_zero, 7) is rank_zero
    assert select_rank_rng_state("invalid", 0) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_restore_rng_state_reproduces_cuda_draws():
    from dreamervla.utils.seed import capture_rng_state, restore_rng_state, set_seed

    set_seed(314159)
    state = capture_rng_state()
    try:
        expected = [
            torch.rand(4, device=f"cuda:{index}").cpu()
            for index in range(torch.cuda.device_count())
        ]
        set_seed(271828)
        restore_rng_state(state, strict=True)
        actual = [
            torch.rand(4, device=f"cuda:{index}").cpu()
            for index in range(torch.cuda.device_count())
        ]
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            pytest.skip("CUDA is visible but has no free memory")
        raise

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
