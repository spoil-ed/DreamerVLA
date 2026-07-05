from __future__ import annotations

import torch

from dreamervla.models.actor.rynnvla_action_hidden_actor import RynnVLAActionHiddenActor


def test_rynnvla_actor_samples_and_evaluates_full_action_chunks() -> None:
    torch.manual_seed(0)
    actor = RynnVLAActionHiddenActor(
        action_hidden_dim=4,
        action_dim=2,
        time_horizon=3,
        adapter_type="identity",
        freeze_output_projection=False,
        initial_log_std=-0.2,
    )
    hidden = torch.randn(5, actor.hidden_dim)

    action_chunk, log_prob, extra = actor(
        {
            "mode": "sample",
            "hidden": hidden,
            "deterministic": False,
            "return_chunk": True,
        }
    )

    assert action_chunk.shape == (5, 3, 2)
    assert log_prob.shape == (5,)
    assert not torch.allclose(action_chunk, extra["mean_chunk"])

    eval_log_prob, entropy, _ = actor(
        {
            "mode": "evaluate",
            "hidden": hidden,
            "action": action_chunk,
        }
    )

    assert eval_log_prob.shape == (5,)
    assert entropy.shape == (5,)
    assert torch.allclose(eval_log_prob, log_prob)
