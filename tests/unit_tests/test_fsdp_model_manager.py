from __future__ import annotations

import torch


class _Checkpointable(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing_enabled = True


def test_fsdp_model_manager_noops_without_distributed() -> None:
    from dreamervla.hybrid_engines.fsdp import FSDPModelManager

    model = _Checkpointable()
    manager = FSDPModelManager(
        strategy="none",
        precision="bf16",
        activation_checkpointing=True,
    )

    wrapped = manager.prepare_model(model)

    assert wrapped is model
    assert model.checkpointing_enabled is True
    assert manager.param_dtype is torch.bfloat16


def test_fsdp_model_manager_rejects_auto_precision() -> None:
    from dreamervla.hybrid_engines.fsdp import FSDPModelManager

    try:
        FSDPModelManager(precision="auto")
    except ValueError as exc:
        assert "precision" in str(exc)
    else:
        raise AssertionError("precision='auto' must be rejected")


def test_fsdp_model_manager_initializes_single_node_process_group(monkeypatch) -> None:
    import torch.distributed as dist

    from dreamervla.hybrid_engines.fsdp import FSDPModelManager

    calls: list[dict[str, object]] = []
    state = {"initialized": False}

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29529")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: state["initialized"])

    def fake_init_process_group(*, backend: str, rank: int, world_size: int) -> None:
        calls.append({"backend": backend, "rank": rank, "world_size": world_size})
        state["initialized"] = True

    monkeypatch.setattr(dist, "init_process_group", fake_init_process_group)

    manager = FSDPModelManager(strategy="fsdp", precision="bf16", backend="gloo")

    assert manager.ensure_process_group() is True
    assert calls == [{"backend": "gloo", "rank": 1, "world_size": 2}]


def test_fsdp_model_manager_requires_rendezvous_env_for_multi_worker(monkeypatch) -> None:
    import pytest
    import torch.distributed as dist

    from dreamervla.hybrid_engines.fsdp import FSDPModelManager

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    manager = FSDPModelManager(strategy="fsdp")

    with pytest.raises(RuntimeError, match="MASTER_ADDR"):
        manager.ensure_process_group()


def test_manager_phase_offload_moves_parameters_and_optimizer_state() -> None:
    from dreamervla.hybrid_engines.fsdp import FSDPModelManager

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    manager = FSDPModelManager(strategy="none")

    manager.offload_param_and_grad(model, offload_grad=True)
    manager.offload_optimizer(optimizer)
    manager.onload_param_and_grad(model, torch.device("cpu"), onload_grad=True)
    manager.onload_optimizer(optimizer, torch.device("cpu"))

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(
        value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
