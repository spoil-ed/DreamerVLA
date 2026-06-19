"""Unit tests for the pluggable FSDP strategy subtree (single-rank, ray-free)."""

from __future__ import annotations

import pytest

from dreamervla.hybrid_engines.fsdp import FSDPModelManager
from dreamervla.hybrid_engines.fsdp.strategy import (
    FSDP2Strategy,
    FSDPStrategy,
    FSDPStrategyBase,
    NoShardStrategy,
)
from dreamervla.workers.actor._test_models import TinyCheckpointPolicy


def test_create_routes_strategy_names() -> None:
    assert isinstance(FSDPStrategyBase.create("none"), NoShardStrategy)
    assert isinstance(FSDPStrategyBase.create("ddp"), NoShardStrategy)
    assert isinstance(FSDPStrategyBase.create("fsdp"), FSDPStrategy)
    assert isinstance(FSDPStrategyBase.create("fsdp1"), FSDPStrategy)
    assert isinstance(FSDPStrategyBase.create("fsdp2"), FSDP2Strategy)


def test_create_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="strategy"):
        FSDPStrategyBase.create("bogus")


def test_fsdp_version_tags() -> None:
    assert FSDPStrategyBase.create("fsdp").fsdp_version() == "fsdp1"
    assert FSDPStrategyBase.create("fsdp2").fsdp_version() == "fsdp2"
    assert FSDPStrategyBase.create("none").fsdp_version() == "none"


def test_single_rank_wrap_is_passthrough_with_checkpointing() -> None:
    # WORLD_SIZE unset/1: no real sharding, model returned as-is, checkpointing on.
    for name in ("none", "fsdp", "fsdp2"):
        model = TinyCheckpointPolicy()
        wrapped = FSDPStrategyBase.create(name, activation_checkpointing=True).wrap_model(model)
        assert wrapped is model
        assert int(model.checkpoint_flag.item()) == 1


def test_invalid_precision_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="precision"):
        FSDPStrategyBase.create("fsdp", precision="tf32")


def test_manager_delegates_to_strategy_and_supports_fsdp2() -> None:
    manager = FSDPModelManager(strategy="fsdp2", activation_checkpointing=True)
    assert isinstance(manager.make_strategy(), FSDP2Strategy)

    model = TinyCheckpointPolicy()
    wrapped = manager.prepare_model(model)
    assert wrapped is model
    assert int(model.checkpoint_flag.item()) == 1
