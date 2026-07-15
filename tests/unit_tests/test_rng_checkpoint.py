"""RNG state capture/restore for bit-exact runner resume."""

import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dreamervla.runtime.world_model_training_utils import DreamerCkptResumeMixin


class _MiniDreamer(DreamerCkptResumeMixin):
    """Smallest mixin host exercising _save_ckpt / _maybe_resume."""

    def __init__(self, tmp, cfg):
        self.is_main_process = True
        self.global_step = 5
        self.epoch = 2
        self.cfg = cfg
        self.ckpt_dir = tmp


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


def test_dreamerv3_save_ckpt_routes_rng_through_shared_helper(tmp_path):
    # Unifying DreamerV3 onto the shared helper means its payload now also
    # carries python `random` state (previously torch+cuda only).
    runner = _MiniDreamer(tmp_path, OmegaConf.create({}))
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters())
    path = tmp_path / "latest.ckpt"

    runner._save_ckpt(model, opt, path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "python" in payload["rng"]
    assert "numpy" in payload["rng"]


def test_dreamerv3_resume_restores_python_numpy_and_torch_rng_bit_exact(tmp_path):
    cfg = OmegaConf.create({"training": {"resume": True}})
    path = tmp_path / "latest.ckpt"

    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    _MiniDreamer(tmp_path, cfg)._save_ckpt(
        torch.nn.Linear(3, 2), torch.optim.Adam(torch.nn.Linear(3, 2).parameters()), path
    )
    ref = (
        [torch.rand(()).item() for _ in range(3)],
        [random.random() for _ in range(3)],
        np.random.random(3),
    )

    torch.manual_seed(999)
    random.seed(999)
    np.random.seed(999)
    resumed = _MiniDreamer(tmp_path, cfg)._maybe_resume(
        torch.nn.Linear(3, 2), torch.optim.Adam(torch.nn.Linear(3, 2).parameters())
    )
    got = (
        [torch.rand(()).item() for _ in range(3)],
        [random.random() for _ in range(3)],
        np.random.random(3),
    )

    assert resumed is True
    assert got[0] == ref[0]
    assert got[1] == ref[1]
    np.testing.assert_array_equal(got[2], ref[2])
