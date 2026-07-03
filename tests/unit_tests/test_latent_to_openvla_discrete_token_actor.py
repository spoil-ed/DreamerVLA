from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dreamervla.models.actor.latent_to_openvla_discrete_token_actor import (
    LatentToOpenVLADiscreteTokenActor,
)
from dreamervla.models.actor.latent_to_openvla_hidden_state_actor import (
    LatentToOpenVLAHiddenStateActor,
)


def _tiny_actor() -> LatentToOpenVLADiscreteTokenActor:
    # Tiny discrete bridge for the query_before (backbone/input-token) route.
    return LatentToOpenVLADiscreteTokenActor(
        source_token_count=5,
        source_token_dim=4,
        action_hidden_dim=4,
        action_dim=2,
        time_horizon=3,
        bridge_hidden_dim=4,
        num_bridge_layers=1,
        num_bridge_heads=2,
        vocab_size=32,
        action_token_bins=8,
        adapter_type="identity",
        init_lm_head_ckpt=None,
        head_type="oft_discrete_token",
    )


def _tiny_hidden_state_actor() -> LatentToOpenVLAHiddenStateActor:
    return LatentToOpenVLAHiddenStateActor(
        source_token_count=5,
        source_token_dim=4,
        hidden_state_dim=4,
        action_dim=2,
        time_horizon=3,
        bridge_hidden_dim=4,
        num_bridge_layers=1,
        num_bridge_heads=2,
        vocab_size=32,
        action_token_bins=8,
        adapter_type="identity",
        init_lm_head_ckpt=None,
        head_type="oft_discrete_token",
    )


def test_hidden_state_actor_uses_hidden_state_dim_only() -> None:
    actor = _tiny_hidden_state_actor()
    assert actor.hidden_state_dim == 4
    assert not hasattr(actor, "action_hidden_dim")


def test_hidden_state_actor_rejects_action_hidden_dim_alias() -> None:
    with pytest.raises(TypeError, match="action_hidden_dim"):
        LatentToOpenVLAHiddenStateActor(
            source_token_count=5,
            source_token_dim=4,
            hidden_state_dim=4,
            action_hidden_dim=4,
            action_dim=2,
            time_horizon=3,
            bridge_hidden_dim=4,
            num_bridge_layers=1,
            num_bridge_heads=2,
            vocab_size=32,
            action_token_bins=8,
            adapter_type="identity",
            init_lm_head_ckpt=None,
            head_type="oft_discrete_token",
        )


def test_hidden_state_actor_decodes_backbone_latent_to_action_chunk() -> None:
    torch.manual_seed(0)
    actor = _tiny_hidden_state_actor()
    hidden = torch.randn(2, 5, 4)

    chunk = actor.reference_action_chunk(hidden)
    assert chunk.shape == (2, actor.time_horizon, actor.action_dim)


def test_actor_decodes_backbone_latent_to_action_chunk() -> None:
    torch.manual_seed(0)
    actor = _tiny_actor()
    # Tokenized backbone latent [B, source_token_count, source_token_dim].
    hidden = torch.randn(2, 5, 4)

    chunk = actor.reference_action_chunk(hidden)
    assert chunk.shape == (2, actor.time_horizon, actor.action_dim)  # (2, 3, 2)

    action_chunk, log_prob, extra = actor(
        {"mode": "sample", "hidden": hidden, "return_chunk": True}
    )
    assert action_chunk.shape == (2, 3, 2)
    assert log_prob.shape == (2,)
    assert extra["action_token_ids"].shape == (2, 3, 2)


def test_actor_returns_token_level_logprobs_when_requested() -> None:
    torch.manual_seed(0)
    actor = _tiny_actor()
    actor.eval()
    hidden = torch.randn(2, 5, 4)

    action_chunk, log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": hidden,
            "return_chunk": True,
            "logprob_type": "token_level",
        }
    )
    eval_log_prob, entropy, _ = actor(
        {
            "mode": "evaluate",
            "hidden": hidden,
            "action": action_chunk,
            "action_token_ids": extra["action_token_ids"],
            "logprob_type": "token_level",
        }
    )

    assert log_prob.shape == (2, 3, 2)
    assert eval_log_prob.shape == (2, 3, 2)
    assert entropy.shape == (2, 3, 2)
    assert torch.allclose(eval_log_prob, log_prob)


def test_actor_accepts_flat_and_tokenized_latent() -> None:
    torch.manual_seed(0)
    actor = _tiny_actor()
    tokenized = torch.randn(2, 5, 4)
    flat = tokenized.reshape(2, 5 * 4)  # [B, source_token_count * source_token_dim]

    assert actor.reference_action_chunk(tokenized).shape == (2, 3, 2)
    assert actor.reference_action_chunk(flat).shape == (2, 3, 2)


def test_actor_is_discrete_with_no_l1_head() -> None:
    actor = _tiny_actor()
    # Discrete decode path: an OpenVLA LM head over the action-token vocabulary.
    assert actor.head_type == "oft_discrete_token"
    assert isinstance(actor.lm_head, nn.Linear)
    # No L1 action head is constructed on this route.
    assert not hasattr(actor, "action_head")
    assert not hasattr(actor, "output_projection")

