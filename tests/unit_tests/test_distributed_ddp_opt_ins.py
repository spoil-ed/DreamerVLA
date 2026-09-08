"""NopretokenizeSFTDistributedHelper DDP option contracts.

DDP / the process group cannot be constructed on a single CPU process, so these
tests follow the established pattern in ``test_reduce_mean_dict_batched.py``:
build the helper directly with ``world_size > 1`` and monkeypatch the
``dreamervla.utils.training.distributed`` module globals (``DDP`` / ``dist`` /
``torch.cuda``) to capture what the helper *would* construct.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch

from dreamervla.utils.training.distributed import NopretokenizeSFTDistributedHelper


def _make_helper(
    world_size: int,
    *,
    backend: str = "nccl",
) -> NopretokenizeSFTDistributedHelper:
    return NopretokenizeSFTDistributedHelper(
        rank=0,
        local_rank=0,
        world_size=world_size,
        strategy="ddp",
        fsdp_mixed_precision="bf16",
        enable_activation_checkpointing=False,
        backend=backend,
    )


class _FakeDDP(torch.nn.Module):
    """Captures the kwargs the helper passes to DDP without needing a PG.

    Subclasses ``nn.Module`` (like the real DDP) so ``wrap_world_model`` can
    ``setattr`` it back onto a parent module.
    """

    def __init__(self, module, **kwargs):  # noqa: ANN001
        super().__init__()
        self.module = module
        self.kwargs = kwargs


def _patch_ddp(monkeypatch) -> None:
    monkeypatch.setattr("dreamervla.utils.training.distributed.DDP", _FakeDDP)


def test_fsdp_uses_local_rank_when_current_device_changes(monkeypatch):
    """FSDP must follow the rank-owned model device, not mutable CUDA state."""
    captured: dict = {}

    class _FakeFSDP(torch.nn.Module):
        def __init__(self, module, **kwargs):  # noqa: ANN001
            super().__init__()
            self.module = module
            captured.update(kwargs)

    helper = NopretokenizeSFTDistributedHelper(
        rank=3,
        local_rank=3,
        world_size=4,
        strategy="fsdp",
        fsdp_mixed_precision="bf16",
        enable_activation_checkpointing=False,
    )
    monkeypatch.setattr("dreamervla.utils.training.distributed.FSDP", _FakeFSDP)
    monkeypatch.setattr(
        "dreamervla.utils.training.distributed.torch.cuda.current_device", lambda: 0
    )
    monkeypatch.setattr(
        "dreamervla.utils.training.distributed.torch.cuda.synchronize", lambda: None
    )

    helper.wrap_trainable_module(torch.nn.Linear(2, 2))

    assert captured["device_id"] == torch.device("cuda:3")


# ── wrap_trainable_module: default-off must stay byte-identical ───────────────


def test_wrap_trainable_module_default_kwargs_are_byte_identical(monkeypatch):
    """No opt-in args -> the exact DDP kwargs used by the OFT-caller contract."""
    _patch_ddp(monkeypatch)
    helper = _make_helper(world_size=2)

    wrapped = helper.wrap_trainable_module(torch.nn.Linear(2, 2))

    assert wrapped.kwargs == {
        "device_ids": [0],
        "output_device": 0,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }


def test_wrap_trainable_module_find_unused_parameters_opt_in(monkeypatch):
    _patch_ddp(monkeypatch)
    helper = _make_helper(world_size=2)

    wrapped = helper.wrap_trainable_module(torch.nn.Linear(2, 2), find_unused_parameters=True)

    assert wrapped.kwargs["find_unused_parameters"] is True
    # the other opt-in stays at its default
    assert wrapped.kwargs["broadcast_buffers"] is False


def test_wrap_trainable_module_broadcast_buffers_opt_in(monkeypatch):
    _patch_ddp(monkeypatch)
    helper = _make_helper(world_size=2)

    wrapped = helper.wrap_trainable_module(torch.nn.Linear(2, 2), broadcast_buffers=True)

    assert wrapped.kwargs["broadcast_buffers"] is True
    assert wrapped.kwargs["find_unused_parameters"] is False


def test_wrap_trainable_module_both_opt_ins_match_online_wm_contract(monkeypatch):
    """The online WM/policy/critic route can opt into both DDP flags."""
    _patch_ddp(monkeypatch)
    helper = _make_helper(world_size=2)

    wrapped = helper.wrap_trainable_module(
        torch.nn.Linear(2, 2),
        find_unused_parameters=True,
        broadcast_buffers=True,
    )

    assert wrapped.kwargs == {
        "device_ids": [0],
        "output_device": 0,
        "broadcast_buffers": True,
        "find_unused_parameters": True,
    }


def test_wrap_trainable_module_static_graph_optimizations_are_opt_in(monkeypatch):
    """Static-graph/bucket-view flags must reach DDP only when Hydra selects them."""
    _patch_ddp(monkeypatch)
    helper = _make_helper(world_size=2)

    wrapped = helper.wrap_trainable_module(
        torch.nn.Linear(2, 2),
        find_unused_parameters=False,
        broadcast_buffers=False,
        static_graph=True,
        gradient_as_bucket_view=True,
    )

    assert wrapped.kwargs == {
        "device_ids": [0],
        "output_device": 0,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "static_graph": True,
        "gradient_as_bucket_view": True,
    }


def test_wrap_trainable_module_init_sync_is_opt_in(monkeypatch):
    _patch_ddp(monkeypatch)
    helper = _make_helper(world_size=2)

    wrapped = helper.wrap_trainable_module(torch.nn.Linear(2, 2), init_sync=False)

    assert wrapped.kwargs["init_sync"] is False


# ── wrap_world_model: untouched, must keep the hardcoded OFT defaults ─────────


def test_wrap_world_model_still_uses_hardcoded_defaults(monkeypatch):
    _patch_ddp(monkeypatch)
    helper = _make_helper(world_size=2)

    class _WM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)

    wm = _WM()
    helper.wrap_world_model(wm)

    assert isinstance(wm.encoder, _FakeDDP)
    assert wm.encoder.kwargs == {
        "device_ids": [0],
        "output_device": 0,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }


# ── initialize: NCCL timeout opt-in (default-off) ────────────────────────────


def _patch_init(monkeypatch, captured: dict) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.is_available", lambda: True)
    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.is_initialized", lambda: False)
    monkeypatch.setattr(
        "dreamervla.utils.training.distributed.torch.cuda.is_available", lambda: False
    )

    def _fake_init(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.init_process_group", _fake_init)


def test_initialize_passes_nccl_timeout_when_set(monkeypatch):
    captured: dict = {}
    _patch_init(monkeypatch, captured)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)

    NopretokenizeSFTDistributedHelper.initialize(nccl_timeout_seconds=1234)

    assert captured.get("backend") == "nccl"
    assert captured.get("timeout") == timedelta(seconds=1234)
    assert os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "1234"


def test_initialize_omits_timeout_by_default(monkeypatch):
    """Default-off: no timeout kwarg → byte-identical to today's init."""
    captured: dict = {}
    _patch_init(monkeypatch, captured)

    NopretokenizeSFTDistributedHelper.initialize()

    assert captured.get("backend") == "nccl"
    assert "timeout" not in captured


def test_initialize_supports_gloo_without_nccl_heartbeat(monkeypatch):
    captured: dict = {}
    _patch_init(monkeypatch, captured)
    monkeypatch.delenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", raising=False)

    helper = NopretokenizeSFTDistributedHelper.initialize(
        backend="gloo",
        nccl_timeout_seconds=1234,
    )

    assert captured["backend"] == "gloo"
    assert helper.backend == "gloo"
    assert "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" not in os.environ


def test_initialize_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported distributed backend"):
        NopretokenizeSFTDistributedHelper.initialize(backend="mpi")


def test_initialize_treats_single_process_strategy_as_unwrapped_ddp(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")

    helper = NopretokenizeSFTDistributedHelper.initialize(strategy="single")

    assert helper.strategy == "ddp"
    assert helper.world_size == 1
    assert helper.is_distributed is False


def test_object_barrier_uses_cpu_collective_group(monkeypatch):
    helper = _make_helper(world_size=4)
    object_group = object()
    helper.object_group = object_group
    captured: dict = {}

    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.is_available", lambda: True)
    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.is_initialized", lambda: True)

    def _fake_barrier(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.barrier", _fake_barrier)

    helper.object_barrier()

    assert captured == {"group": object_group}


def test_gloo_barrier_does_not_pass_cuda_device_ids(monkeypatch):
    helper = _make_helper(world_size=4, backend="gloo")
    captured: dict = {}
    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.is_available", lambda: True)
    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "dreamervla.utils.training.distributed.torch.cuda.is_available", lambda: True
    )

    def _fake_barrier(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    monkeypatch.setattr("dreamervla.utils.training.distributed.dist.barrier", _fake_barrier)

    helper.barrier()

    assert captured == {}
