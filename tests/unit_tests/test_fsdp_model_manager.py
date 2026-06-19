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
