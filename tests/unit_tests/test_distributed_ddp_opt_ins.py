"""NopretokenizeSFTDistributedHelper DDP option contracts.

DDP / the process group cannot be constructed on a single CPU process, so these
tests follow the established pattern in ``test_reduce_mean_dict_batched.py``:
build the helper directly with ``world_size > 1`` and monkeypatch the
``dreamervla.runtime.distributed`` module globals (``DDP`` / ``dist`` /
``torch.cuda``) to capture what the helper *would* construct.
"""

from __future__ import annotations

from datetime import timedelta

import torch

from dreamervla.runtime.distributed import NopretokenizeSFTDistributedHelper


def _make_helper(world_size: int) -> NopretokenizeSFTDistributedHelper:
    return NopretokenizeSFTDistributedHelper(
        rank=0,
        local_rank=0,
        world_size=world_size,
        strategy="ddp",
        fsdp_mixed_precision="bf16",
        enable_activation_checkpointing=False,
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
    monkeypatch.setattr("dreamervla.runtime.distributed.DDP", _FakeDDP)


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
    monkeypatch.setattr("dreamervla.runtime.distributed.dist.is_available", lambda: True)
    monkeypatch.setattr("dreamervla.runtime.distributed.dist.is_initialized", lambda: False)
    monkeypatch.setattr("dreamervla.runtime.distributed.torch.cuda.is_available", lambda: False)

    def _fake_init(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    monkeypatch.setattr("dreamervla.runtime.distributed.dist.init_process_group", _fake_init)


def test_initialize_passes_nccl_timeout_when_set(monkeypatch):
    captured: dict = {}
    _patch_init(monkeypatch, captured)

    NopretokenizeSFTDistributedHelper.initialize(nccl_timeout_seconds=1234)

    assert captured.get("backend") == "nccl"
    assert captured.get("timeout") == timedelta(seconds=1234)


def test_initialize_omits_timeout_by_default(monkeypatch):
    """Default-off: no timeout kwarg → byte-identical to today's init."""
    captured: dict = {}
    _patch_init(monkeypatch, captured)

    NopretokenizeSFTDistributedHelper.initialize()

    assert captured.get("backend") == "nccl"
    assert "timeout" not in captured
