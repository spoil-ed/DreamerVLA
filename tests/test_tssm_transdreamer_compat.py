from __future__ import annotations

import torch
from torch import nn

from src.models.world_model.dreamerv3_torch import _act
from src.models.world_model.tssm_torch import (
    TSSMDynamic,
    TSSMRynnBackboneWorldModel,
    TSSMTokenDynamic,
    TSSMTokenRynnBackboneWorldModel,
)


def test_dreamervla_activation_factory_supports_transdreamer_elu() -> None:
    assert isinstance(_act("elu"), nn.ELU)


def test_flat_tssm_observe_returns_transdreamer_t_minus_one_steps() -> None:
    model = TSSMDynamic(
        obs_emb_dim=6,
        action_dim=3,
        hidden=8,
        stoch=2,
        classes=3,
        n_layers=2,
        n_head=2,
        d_model=8,
        d_inner=4,
        d_ff_inner=16,
        dropout=0.0,
        dropatt=0.0,
        tssm_window=8,
        free_nats=0.0,
    )
    enc = torch.randn(2, 5, 6)
    actions = torch.randn(2, 5, 3)
    is_first = torch.zeros(2, 5, dtype=torch.bool)
    is_first[:, 0] = True

    seq = model.observe(enc, actions, is_first)

    assert seq["deter"].shape == (2, 4, 16)
    assert seq["stoch"].shape == (2, 4, 2, 3)
    assert seq["post_logits"].shape == (2, 4, 2, 3)
    assert seq["prior_logits"].shape == (2, 4, 2, 3)


def test_token_tssm_observe_returns_transdreamer_t_minus_one_steps() -> None:
    model = TSSMTokenDynamic(
        n_tokens=4,
        token_dim=6,
        action_dim=3,
        hidden=8,
        stoch=2,
        classes=3,
        n_layers=2,
        n_head=2,
        d_model=8,
        d_inner=4,
        d_ff_inner=16,
        dropout=0.0,
        dropatt=0.0,
        tssm_window=8,
        free_nats=0.0,
    )
    obs_tokens = torch.randn(2, 5, 4, 6)
    actions = torch.randn(2, 5, 3)
    is_first = torch.zeros(2, 5, dtype=torch.bool)
    is_first[:, 0] = True

    seq = model.observe(obs_tokens, actions, is_first)

    assert seq["deter"].shape == (2, 4, 4, 16)
    assert seq["stoch"].shape == (2, 4, 4, 2, 3)
    assert seq["post_logits"].shape == (2, 4, 4, 2, 3)
    assert seq["prior_logits"].shape == (2, 4, 4, 2, 3)


def _small_batch(obs_dim: int, action_dim: int) -> dict[str, torch.Tensor]:
    batch_size = 2
    steps = 5
    return {
        "images": torch.randint(0, 256, (batch_size, steps, 3, 64, 64), dtype=torch.uint8),
        "obs_embedding": torch.randn(batch_size, steps, obs_dim),
        "actions": torch.randn(batch_size, steps, action_dim),
        "rewards": torch.zeros(batch_size, steps),
        "dones": torch.zeros(batch_size, steps),
        "is_first": torch.tensor(
            [[True, False, False, False, False], [True, False, False, False, False]]
        ),
    }


def test_flat_tssm_world_model_loss_trims_targets_to_transdreamer_steps() -> None:
    model = TSSMRynnBackboneWorldModel(
        obs_dim=10,
        action_dim=3,
        image_channels=3,
        image_size=64,
        embed_dim=6,
        encoder_hidden=8,
        encoder_layers=1,
        hidden=8,
        stoch=2,
        classes=3,
        tssm_layers=2,
        tssm_nhead=2,
        tssm_d_model=8,
        tssm_d_inner=4,
        tssm_d_ff_inner=16,
        tssm_dropout=0.0,
        tssm_dropatt=0.0,
        tssm_window=8,
        depth=4,
        hidden_decoder_layers=1,
        hidden_decoder_units=8,
        rec_scale=0.0,
        rew_scale=0.0,
        con_scale=0.0,
        hidden_rec_scale=1.0,
    )

    loss = model.loss(_small_batch(obs_dim=10, action_dim=3))

    assert torch.isfinite(loss.loss)
    assert loss.metrics["hidden_rec_loss"].ndim == 0


def test_token_tssm_world_model_loss_trims_targets_to_transdreamer_steps() -> None:
    model = TSSMTokenRynnBackboneWorldModel(
        obs_dim=10,
        action_dim=3,
        image_channels=3,
        image_size=64,
        n_tokens=2,
        token_dim=5,
        hidden=8,
        stoch=2,
        classes=3,
        tssm_layers=2,
        tssm_nhead=2,
        tssm_d_model=8,
        tssm_d_inner=4,
        tssm_d_ff_inner=16,
        tssm_dropout=0.0,
        tssm_dropatt=0.0,
        tssm_window=8,
        depth=4,
        hidden_decoder_layers=1,
        hidden_decoder_units=8,
        rec_scale=0.0,
        rew_scale=0.0,
        con_scale=0.0,
        hidden_rec_scale=1.0,
    )

    loss = model.loss(_small_batch(obs_dim=10, action_dim=3))

    assert torch.isfinite(loss.loss)
    assert loss.metrics["hidden_rec_loss"].ndim == 0
