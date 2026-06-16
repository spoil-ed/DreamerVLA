from __future__ import annotations

import torch

from dreamervla.models.world_model.dreamer_v3_pixel_backbone_world_model import (
    DreamerV3PixelBackboneWorldModel,
)
from dreamervla.models.world_model.dreamerv3_torch import (
    DreamerV3LatentState,
    DreamerV3RSSM,
)


def test_rssm_observe_next_matches_sequence_observe() -> None:
    rssm = DreamerV3RSSM(
        action_dim=2,
        deter=16,
        hidden=8,
        stoch=3,
        classes=5,
        blocks=4,
        imglayers=1,
        obslayers=1,
    )
    rssm.build_posterior(token_dim=7)
    tokens = torch.randn(2, 2, 7)
    actions = torch.randn(2, 2, 2)
    is_first = torch.tensor([[True, False], [True, False]])

    torch.manual_seed(0)
    seq = rssm.observe(tokens, actions, is_first)

    init = rssm.initial(batch_size=2, device=tokens.device, dtype=tokens.dtype)
    latent0 = DreamerV3LatentState(deter=init["deter"], stoch=init["stoch"])
    torch.manual_seed(0)
    step0 = rssm.observe_next(latent0, tokens[:, 0], actions[:, 0], is_first[:, 0])
    step1 = rssm.observe_next(step0, tokens[:, 1], actions[:, 1], is_first[:, 1])

    assert torch.allclose(step0.deter, seq["deter"][:, 0])
    assert torch.allclose(step0.stoch, seq["stoch"][:, 0])
    assert torch.allclose(step0.logits, seq["post_logits"][:, 0])
    assert torch.allclose(step1.deter, seq["deter"][:, 1])
    assert torch.allclose(step1.stoch, seq["stoch"][:, 1])
    assert torch.allclose(step1.logits, seq["post_logits"][:, 1])


def test_rynn_world_model_can_expose_rssm_feature_for_dreamer_actor() -> None:
    model = DreamerV3PixelBackboneWorldModel(
        obs_dim=12,
        action_dim=2,
        image_channels=3,
        image_size=64,
        encoder_hidden=16,
        encoder_layers=1,
        deter=16,
        hidden=8,
        stoch=3,
        classes=5,
        blocks=4,
        depth=4,
        actor_input_kind="feature",
    )
    latent = DreamerV3LatentState(
        deter=torch.zeros(2, 16),
        stoch=torch.zeros(2, 3, 5),
        logits=torch.zeros(2, 3, 5),
    )

    actor_input = model.actor_input(latent)

    assert actor_input.shape == (2, 31)


def test_rynn_world_model_default_encoder_uses_obs_embedding_directly() -> None:
    model = DreamerV3PixelBackboneWorldModel(
        obs_dim=12,
        action_dim=2,
        image_channels=3,
        image_size=64,
        encoder_hidden=16,
        encoder_layers=1,
        deter=16,
        hidden=8,
        stoch=3,
        classes=5,
        blocks=4,
        depth=4,
    )
    obs_embedding = torch.randn(2, 3, 12)

    enc = model._encode_obs_embedding(obs_embedding)

    assert enc.shape == obs_embedding.shape
    assert torch.allclose(enc, obs_embedding)
