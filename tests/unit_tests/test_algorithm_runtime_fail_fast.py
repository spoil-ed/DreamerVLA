from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner
from dreamervla.runtime.libero_vla_evaluation_base import LIBEROVLAEvaluationBase
from dreamervla.runtime.world_model_training_base import WorldModelTrainingBase


class _SpatialWorldModel:
    spatial_codec = True
    io_mode = "hidden"


def test_spatial_world_model_requires_encoder_for_image_token_mapping() -> None:
    subject = SimpleNamespace(
        _unwrapped_world_model=_SpatialWorldModel(),
        world_model=None,
        encoder=None,
        distributed=SimpleNamespace(is_main_process=False),
    )

    with pytest.raises(RuntimeError, match="requires an encoder"):
        WorldModelTrainingBase._attach_image_token_mapping(subject)


def test_spatial_world_model_propagates_mapping_attachment_failure() -> None:
    subject = SimpleNamespace(
        _unwrapped_world_model=_SpatialWorldModel(),
        world_model=None,
        encoder=SimpleNamespace(backbone=SimpleNamespace()),
        distributed=SimpleNamespace(is_main_process=False),
    )

    with pytest.raises(RuntimeError, match="failed to attach image-token mapping") as exc_info:
        WorldModelTrainingBase._attach_image_token_mapping(subject)

    assert isinstance(exc_info.value.__cause__, AttributeError)


class _ItemProcessor:
    def process_item(self, _conversation: object, *, training_mode: bool) -> list[int]:
        assert training_mode is False
        return [1, 2, 3]


class _FailingActionHead:
    config = SimpleNamespace(max_position_embeddings=32)

    def generate_action_head(self, _input_ids: torch.Tensor, _config: object) -> torch.Tensor:
        raise RuntimeError("incompatible action head")


class _EmptyActionHead:
    config = SimpleNamespace(max_position_embeddings=32)

    def generate_action_head(self, _input_ids: torch.Tensor, _config: object) -> torch.Tensor:
        return torch.empty((0, 7))


def _evaluation_runner() -> LIBEROVLAEvaluationRunner:
    runner = object.__new__(LIBEROVLAEvaluationRunner)
    runner.device = torch.device("cpu")
    return runner


def test_existing_action_head_failure_does_not_change_evaluation_algorithm() -> None:
    runner = _evaluation_runner()

    with pytest.raises(RuntimeError, match="generate_action_head failed") as exc_info:
        runner._generate_vla_actions_with_trace(
            _FailingActionHead(),
            _ItemProcessor(),
            [],
            np.zeros(7, dtype=np.float32),
            "pick up the object",
            1,
        )

    assert "incompatible action head" in str(exc_info.value.__cause__)


def test_action_head_must_return_at_least_one_action() -> None:
    runner = _evaluation_runner()
    runner._unnorm_actions = lambda actions: actions
    runner._write_policy_trace = lambda **_kwargs: None

    with pytest.raises(RuntimeError, match="returned no actions"):
        runner._generate_vla_actions_with_trace(
            _EmptyActionHead(),
            _ItemProcessor(),
            [],
            np.zeros(7, dtype=np.float32),
            "pick up the object",
            1,
        )


def test_checkpoint_without_action_head_uses_capability_fallback(monkeypatch) -> None:
    runner = _evaluation_runner()
    expected = [np.ones(7, dtype=np.float32)]

    def fallback(*_args: object, **_kwargs: object) -> list[np.ndarray]:
        return expected

    monkeypatch.setattr(LIBEROVLAEvaluationBase, "_generate_actions", fallback)
    backbone = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32))

    actual = runner._generate_vla_actions_with_trace(
        backbone,
        _ItemProcessor(),
        [],
        np.zeros(7, dtype=np.float32),
        "pick up the object",
        1,
    )

    assert actual is expected
