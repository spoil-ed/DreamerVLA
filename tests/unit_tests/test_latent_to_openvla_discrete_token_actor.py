from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from dreamervla.models.actor.latent_to_openvla_discrete_token_actor import (
    LatentToOpenVLADiscreteTokenActor,
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


def test_backbone_latent_online_route_wires_discrete_actor() -> None:
    # The query_before (backbone-latent) online WMPO route uses the discrete bridge,
    # the lean ~313M WM, and the online input-token obs source — i.e. it is wired
    # end-to-end and no longer raises NotImplementedError.
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "dreamervla=openvla_oft_input_token_wmpo_outcome",
                "task=OpenVLA_Onetraj_LIBERO",
            ],
        )

    assert cfg.policy._target_.endswith("LatentToOpenVLADiscreteTokenActor")
    assert cfg.policy.head_type == "oft_discrete_token"
    # Lean-debottlenecked query_before WM profile (~313M).
    assert cfg.world_model.latent_stage == "query_before"
    assert cfg.world_model.dim_head == 128
    assert cfg.world_model.mlp_dim == 2048

    # Online input-token obs source exists (replaces the old NotImplementedError).
    from dreamervla.runners.online_utils import obs_to_input_token_embedding

    assert callable(obs_to_input_token_embedding)
