"""Early-validation tests for the learner FSDP block (ray-free)."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg


def _cfg(fsdp: dict) -> OmegaConf:
    return OmegaConf.create({"learner": {"train_cfg": {"fsdp": fsdp}}})


def test_validate_cfg_rejects_unknown_fsdp_strategy() -> None:
    with pytest.raises(ValueError, match="fsdp.*strategy"):
        validate_cfg(_cfg({"strategy": "bogus"}))


def test_validate_cfg_rejects_invalid_fsdp_precision() -> None:
    with pytest.raises(ValueError, match="fsdp.*precision"):
        validate_cfg(_cfg({"strategy": "fsdp", "precision": "tf32"}))


def test_validate_cfg_accepts_supported_fsdp_strategies() -> None:
    for strategy in ("none", "ddp", "fsdp", "fsdp1", "fsdp2"):
        validate_cfg(_cfg({"strategy": strategy, "precision": "bf16", "cpu_offload": True}))


def test_validate_cfg_without_fsdp_block_is_noop() -> None:
    validate_cfg(OmegaConf.create({"learner": {"train_cfg": {"batch_size": 2}}}))
