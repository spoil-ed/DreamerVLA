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


def _hidden_token(batch_size: int = 2) -> torch.Tensor:
    return torch.zeros(batch_size, 256, 4096)


def test_hidden_state_actor_uses_canonical_hidden_token_boundary() -> None:
    actor = _tiny_mainline_actor()

    assert actor.source_token_count == 256
    assert actor.source_token_dim == 4096
    assert actor.action_token_count == 56
    assert actor.hidden_state_dim == 4096
    assert isinstance(actor.lm_head, nn.Linear)
    assert actor.lm_head.out_features == actor.action_token_bins


def test_hidden_state_actor_loads_legacy_full_vocabulary_lm_head() -> None:
    actor = _tiny_mainline_actor()
    state = actor.state_dict()
    full_weight = torch.arange(
        actor.vocab_size * actor.hidden_state_dim,
        dtype=torch.float32,
    ).reshape(actor.vocab_size, actor.hidden_state_dim)
    state["lm_head.weight"] = full_weight

    restored = _tiny_mainline_actor()
    restored.load_state_dict(state)

    torch.testing.assert_close(
        restored.lm_head.weight,
        full_weight[-actor.action_token_bins :],
    )


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_token_count": 0}, "source tokens"),
        ({"bridge_hidden_dim": 0}, "bridge_hidden_dim"),
        ({"num_bridge_layers": 0}, "num_bridge_layers"),
        ({"num_bridge_heads": 0}, "num_bridge_heads"),
        ({"bridge_dropout": 1.0}, "bridge_dropout"),
        ({"adapter_hidden_dim": 0}, "adapter_hidden_dim"),
        ({"min_action": 1.0, "max_action": 1.0}, "min_action"),
    ],
)
def test_hidden_state_actor_rejects_invalid_internal_geometry(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LatentToOpenVLAHiddenStateActor(**kwargs)


def test_hidden_state_actor_rejects_flat_observation() -> None:
    actor = _tiny_mainline_actor()

    with pytest.raises(ValueError, match="flat observations are closed"):
        actor.reference_action_chunk(torch.zeros(1, 256 * 4096))


def test_hidden_state_actor_decodes_internal_action_slots() -> None:
    torch.manual_seed(0)
    actor = _tiny_mainline_actor()
    hidden = _hidden_token()

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
    hidden = _hidden_token()

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


def test_hidden_state_actor_rejects_out_of_vocabulary_action_token_ids() -> None:
    actor = _tiny_mainline_actor().eval()
    hidden = _hidden_token(batch_size=1)
    action = torch.zeros(1, 8, 7)
    invalid_ids = torch.zeros(1, 8, 7, dtype=torch.long)

    with pytest.raises(ValueError, match="action_token_ids must be within"):
        actor(
            {
                "mode": "evaluate",
                "hidden": hidden,
                "action": action,
                "action_token_ids": invalid_ids,
            }
        )


def test_hidden_state_actor_requires_token_ids_for_exact_logprob() -> None:
    torch.manual_seed(0)
    actor = _tiny_mainline_actor().eval()
    hidden = _hidden_token()
    action, sampled_log_prob, _ = actor({"mode": "sample", "hidden": hidden})

    assert sampled_log_prob.shape == (2,)
    with pytest.raises(KeyError, match="action_token_ids"):
        actor({"mode": "evaluate", "hidden": hidden, "action": action})


def test_hidden_state_actor_rejects_misaligned_action_token_count() -> None:
    actor = _tiny_mainline_actor().eval()
    start = actor.vocab_size - actor.action_token_bins

    with pytest.raises(ValueError, match="token count"):
        actor(
            {
                "mode": "evaluate",
                "hidden": _hidden_token(batch_size=1),
                "action": torch.zeros(1, 8, 7),
                "action_token_ids": torch.full((1, 7), start, dtype=torch.long),
            }
        )


def test_hidden_state_actor_rejects_checkpoint_without_lm_head(tmp_path) -> None:
    checkpoint = tmp_path / "invalid.ckpt"
    torch.save({"state_dict": {"unrelated.weight": torch.ones(1)}}, checkpoint)

    with pytest.raises(KeyError, match="LM-head weight"):
        LatentToOpenVLAHiddenStateActor(
            bridge_hidden_dim=4,
            num_bridge_layers=1,
            num_bridge_heads=2,
            init_lm_head_ckpt=str(checkpoint),
        )
