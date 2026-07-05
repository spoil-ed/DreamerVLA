from __future__ import annotations

import torch

from dreamervla.models.embodiment.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    RynnVLAActionHead,
)


def test_rynnvla_action_head_uses_one_token_per_action_dimension() -> None:
    head = RynnVLAActionHead(
        action_dim=7,
        time_horizon=5,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
    )
    hidden_states = torch.randn(2, 8, 4096)
    input_ids = torch.tensor(
        [
            [11, 12, 10004, 21, 22, 23, 24, 25],
            [31, 32, 33, 10004, 41, 42, 43, 44],
        ],
        dtype=torch.long,
    )

    actions, ok = head(hidden_states=hidden_states, input_ids=input_ids, eval=True)

    assert ok is True
    assert head.action_token_embeddings.weight.shape == (1, 5 * 7 * 4096)
    assert actions.shape == (2 * 5, 7)


def test_rynnvla_action_head_supports_attention_mask() -> None:
    head = RynnVLAActionHead(
        action_dim=7,
        time_horizon=3,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
    )
    hidden_states = torch.randn(1, 6, 4096)
    input_ids = torch.tensor([[1, 2, 3, 10004, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)

    actions, ok = head(
        hidden_states=hidden_states,
        input_ids=input_ids,
        attention_mask=attention_mask,
        eval=True,
    )

    assert ok is True
    assert actions.shape == (3, 7)


def test_rynnvla_action_head_accepts_bool_attention_mask() -> None:
    head = RynnVLAActionHead(
        action_dim=7,
        time_horizon=3,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
    )
    hidden_states = torch.randn(1, 6, 4096)
    input_ids = torch.tensor([[1, 2, 3, 10004, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[True, True, True, True, False, False]])

    actions, ok = head(
        hidden_states=hidden_states,
        input_ids=input_ids,
        attention_mask=attention_mask,
        eval=True,
    )

    assert ok is True
    assert actions.shape == (3, 7)


def test_rynnvla_action_head_exposes_observation_conditioned_action_hidden() -> (
    None
):
    head = RynnVLAActionHead(
        action_dim=4,
        time_horizon=3,
        hidden_size_factor=0.25,
        num_encoder_layers=1,
    )
    hidden_states = torch.randn(2, 4, 4096)
    input_ids = torch.tensor(
        [
            [10, 11, 10004, 0],
            [20, 21, 22, 10004],
        ],
        dtype=torch.long,
    )

    action_hidden, ok = head.extract_action_hidden(
        hidden_states=hidden_states,
        input_ids=input_ids,
        eval=True,
    )

    assert ok is True
    assert action_hidden.shape == (2, 3 * 4, head.reduced_hidden_size)
    shifted_hidden = hidden_states.clone()
    shifted_hidden[:, 0, :] += 1.0
    shifted_action_hidden, _ = head.extract_action_hidden(
        hidden_states=shifted_hidden,
        input_ids=input_ids,
        eval=True,
    )
    assert not torch.allclose(action_hidden, shifted_action_hidden)
