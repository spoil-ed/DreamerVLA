import pytest
import torch

from dreamervla.algorithms.reward.protocol import RewardModel


def test_reward_model_protocol_runtime_checkable():
    class _Stub:
        name = "stub"

        def build_reward(
            self,
            *,
            batch,
            max_steps,
            chunk_size,
            finish_step,
            complete,
            device,
            score=None,
            score_step=None,
        ):
            del chunk_size, finish_step, complete, score, score_step
            return torch.zeros((batch, max_steps), device=device)

    assert isinstance(_Stub(), RewardModel)

    class _NotAModel:
        name = "x"

    assert not isinstance(_NotAModel(), RewardModel)


def test_register_and_get_roundtrip():
    from dreamervla.algorithms.reward.registry import (
        get_reward_model,
        register_reward_model,
        reward_model_names,
    )

    class _Stub:
        name = "stub_route"

        def build_reward(
            self,
            *,
            batch,
            max_steps,
            chunk_size,
            finish_step,
            complete,
            device,
            score=None,
            score_step=None,
        ):
            del chunk_size, finish_step, complete, score, score_step
            return torch.zeros((batch, max_steps), device=device)

    stub = _Stub()
    register_reward_model(stub, aliases=("stub_alias",))
    assert get_reward_model("stub_route") is stub
    assert get_reward_model("STUB-ALIAS") is stub  # normalised lookup
    assert "stub_route" in reward_model_names()


def test_get_unknown_raises():
    from dreamervla.algorithms.reward.registry import get_reward_model

    try:
        get_reward_model("does_not_exist")
    except ValueError as exc:
        assert "Unknown reward model" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_register_rejects_object_without_reward_protocol() -> None:
    from dreamervla.algorithms.reward.registry import register_reward_model

    class _Invalid:
        name = "invalid_reward_protocol"

    with pytest.raises(TypeError, match="RewardModel"):
        register_reward_model(_Invalid())  # type: ignore[arg-type]
