from __future__ import annotations

import numpy as np
import torch

from dreamervla.workers.actor.embodied_fsdp_actor import EmbodiedFSDPActor
from dreamervla.workers.cotrain.messages import RealTrajectory, RealTrajectoryBatch


def _actor() -> EmbodiedFSDPActor:
    actor = EmbodiedFSDPActor(
        policy_cfg={
            "target": "dreamervla.workers.actor._test_models:TinyStagedVLAPolicy",
            "kwargs": {"num_bins": 3},
        },
        init_ckpt={},
        train_cfg={
            "device": "cpu",
            "fsdp": {"strategy": "none", "precision": "fp32"},
            "optimizers": {
                "policy": {"name": "adamw", "lr": 0.1},
                "encoder": {"name": "adamw", "lr": 0.1},
            },
            "encoder_sft": {"epochs": 2, "batch_size": 2},
        },
    )
    actor.init()
    actor.set_global_step(7)
    return actor


def _transition(value: int, *, decision: bool, token: int = 0) -> dict:
    record = {
        "image": np.full((2, 2, 3), value, dtype=np.uint8),
        "agentview_rgb": np.full((2, 2, 3), value, dtype=np.uint8),
        "state": np.asarray([value], dtype=np.float32),
        "task_description": "put the cup down",
        "policy_decision": bool(decision),
    }
    if decision:
        record["action_token_ids_chunk"] = np.asarray([[token]], dtype=np.int64)
    return record


def _batch(*, success: bool = True) -> RealTrajectoryBatch:
    return RealTrajectoryBatch(
        global_step=7,
        trajectories=(
            RealTrajectory(
                env_rank=0,
                slot_id=0,
                task_id=0,
                episode_id=3,
                global_step=7,
                success=success,
                transitions=(
                    _transition(1, decision=True, token=2),
                    _transition(2, decision=False),
                    _transition(3, decision=True, token=2),
                ),
            ),
        ),
    )


def _parameters(actor: EmbodiedFSDPActor, prefix: str) -> list[torch.Tensor]:
    return [
        value.detach().clone()
        for name, value in actor._policy().named_parameters()
        if name.startswith(prefix)
    ]


def test_encoder_sft_updates_only_encoder_from_successful_decisions() -> None:
    actor = _actor()
    encoder_before = _parameters(actor, "encoder.")
    action_before = _parameters(actor, "actor.")

    metrics = actor.encoder_sft(_batch())

    encoder_after = _parameters(actor, "encoder.")
    action_after = _parameters(actor, "actor.")
    assert metrics["actor/encoder_sft_skipped"] == 0.0
    assert metrics["actor/encoder_sft_trajectories"] == 1.0
    assert metrics["actor/encoder_sft_decisions"] == 2.0
    assert metrics["actor/encoder_sft_kl"] >= 0.0
    assert any(not torch.equal(a, b) for a, b in zip(encoder_before, encoder_after, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(action_before, action_after, strict=True))
    assert all(
        not parameter.requires_grad
        for name, parameter in actor._policy().named_parameters()
        if name.startswith("encoder.")
    )


def test_encoder_sft_uses_dedicated_cotrain_progress_name(monkeypatch) -> None:
    import dreamervla.workers.actor.embodied_fsdp_actor as actor_module

    names: list[str] = []

    class _Progress:
        def __init__(self, _total, name, **_kwargs) -> None:
            names.append(str(name))

        def set_status(self, _status: str) -> None:
            return None

        def set(self, _done: int, **_kwargs) -> None:
            return None

    monkeypatch.setattr(actor_module, "ProgressReporter", _Progress)
    actor = _actor()

    actor.encoder_sft(_batch())

    assert names == ["cotrain-vla-real-sft/00000007"]


def test_encoder_sft_skips_without_success_and_changes_no_parameters() -> None:
    actor = _actor()
    before = _parameters(actor, "")

    metrics = actor.encoder_sft(_batch(success=False))

    after = _parameters(actor, "")
    assert metrics["actor/encoder_sft_skipped"] == 1.0
    assert metrics["actor/encoder_sft_decisions"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_reencode_real_trajectories_updates_every_transition_and_version() -> None:
    actor = _actor()

    encoded = actor.reencode_real_trajectories(_batch())

    assert encoded is not _batch()
    assert encoded.global_step == 7
    transitions = encoded.trajectories[0].transitions
    assert len(transitions) == 3
    assert all(np.asarray(step["obs_embedding"]).shape == (1, 2) for step in transitions)
    assert all(np.asarray(step["lang_emb"]).shape == (1,) for step in transitions)
    assert all(step["encoder_version"] == 7 for step in transitions)
    assert "action_token_ids_chunk" not in transitions[1]


def test_policy_transaction_rolls_back_policy_and_both_optimizers() -> None:
    actor = _actor()
    before = _parameters(actor, "")
    actor.begin_policy_transaction()
    with torch.no_grad():
        for parameter in actor._policy().parameters():
            parameter.add_(1.0)

    metrics = actor.finalize_policy_transaction(observed_kl=0.2, max_kl=0.1)

    after = _parameters(actor, "")
    assert metrics["actor/kl_transaction_accepted"] == 0.0
    assert metrics["actor/kl_transaction_rolled_back"] == 1.0
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_policy_transaction_accepts_update_inside_budget() -> None:
    actor = _actor()
    actor.begin_policy_transaction()
    with torch.no_grad():
        next(actor._policy().parameters()).add_(1.0)

    metrics = actor.finalize_policy_transaction(observed_kl=0.05, max_kl=0.1)

    assert metrics["actor/kl_transaction_accepted"] == 1.0
    assert metrics["actor/kl_transaction_rolled_back"] == 0.0
