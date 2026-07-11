from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dreamervla.algorithms.actor import LatentToOpenVLAHiddenStateActor


def _tiny_mainline_actor() -> LatentToOpenVLAHiddenStateActor:
    return LatentToOpenVLAHiddenStateActor(
        source_token_count=256,
        source_token_dim=4096,
        hidden_state_dim=4096,
        action_dim=7,
        time_horizon=8,
        bridge_hidden_dim=4,
        num_bridge_layers=1,
        num_bridge_heads=2,
        vocab_size=32,
        action_token_bins=8,
        adapter_type="identity",
        init_lm_head_ckpt=None,
        freeze_lm_head=False,
        head_type="oft_discrete_token",
    )


def _input_tokens(batch_size: int = 2) -> torch.Tensor:
    return torch.zeros(batch_size, 256, 4096)


def test_hidden_state_actor_uses_canonical_input_token_boundary() -> None:
    actor = _tiny_mainline_actor()

    assert actor.source_token_count == 256
    assert actor.source_token_dim == 4096
    assert actor.action_token_count == 56
    assert actor.hidden_state_dim == 4096
    assert isinstance(actor.lm_head, nn.Linear)


def test_hidden_state_actor_rejects_removed_hidden_token_alias() -> None:
    with pytest.raises(TypeError, match="removed 56x1024 route"):
        LatentToOpenVLAHiddenStateActor(hidden_token_dim=1024)


def test_hidden_state_actor_rejects_noncanonical_source_shape() -> None:
    with pytest.raises(ValueError, match=r"requires source tokens \[256,4096\]"):
        LatentToOpenVLAHiddenStateActor(
            source_token_count=56,
            source_token_dim=1024,
        )


def test_hidden_state_actor_rejects_noncanonical_decoder_width() -> None:
    with pytest.raises(ValueError, match="hidden_state_dim is fixed to 4096"):
        LatentToOpenVLAHiddenStateActor(hidden_state_dim=1024)


def test_hidden_state_actor_rejects_noncanonical_action_geometry() -> None:
    with pytest.raises(ValueError, match=r"fixed to \[8,7\]"):
        LatentToOpenVLAHiddenStateActor(time_horizon=4)


def test_hidden_state_actor_rejects_flat_observation() -> None:
    actor = _tiny_mainline_actor()

    with pytest.raises(ValueError, match="flat observations are closed"):
        actor.reference_action_chunk(torch.zeros(1, 256 * 4096))


def test_hidden_state_actor_decodes_internal_action_slots() -> None:
    torch.manual_seed(0)
    actor = _tiny_mainline_actor()
    hidden = _input_tokens()

    action_chunk, log_prob, extra = actor(
        {"mode": "sample", "hidden": hidden, "return_chunk": True}
    )

    assert action_chunk.shape == (2, 8, 7)
    assert log_prob.shape == (2,)
    assert extra["action_token_ids"].shape == (2, 8, 7)
    assert extra["hidden_state"].shape == (2, 56, 4096)


def test_hidden_state_actor_token_level_logprobs_round_trip() -> None:
    torch.manual_seed(0)
    actor = _tiny_mainline_actor().eval()
    hidden = _input_tokens()

    action_chunk, sampled_log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": hidden,
            "return_chunk": True,
            "logprob_type": "token_level",
        }
    )
    evaluated_log_prob, entropy, _ = actor(
        {
            "mode": "evaluate",
            "hidden": hidden,
            "action": action_chunk,
            "action_token_ids": extra["action_token_ids"],
            "logprob_type": "token_level",
        }
    )

    assert sampled_log_prob.shape == (2, 8, 7)
    assert evaluated_log_prob.shape == (2, 8, 7)
    assert entropy.shape == (2, 8, 7)
    torch.testing.assert_close(evaluated_log_prob, sampled_log_prob)
