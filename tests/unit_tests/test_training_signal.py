from __future__ import annotations

import torch

from dreamervla.runners.cotrain_runner import CotrainRunner
from dreamervla.runtime.training_signal import evaluate_imagined_success_sft_signal


class _Ready:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value


class _ActorGroup:
    def __init__(self):
        self.weight = torch.zeros(1)

    def state_dict(self):
        return _Ready([{"weight": self.weight.clone()}])


def test_signal_policy_hash_can_refresh_without_writing_checkpoint() -> None:
    runner = object.__new__(CotrainRunner)
    runner._pending_manual_resume_payload = None
    actor_group = _ActorGroup()

    runner._initialize_policy_hashes(actor_group)
    initial_hash = runner._policy_initial_hash
    actor_group.weight.add_(1)
    runner._refresh_policy_final_hash(actor_group)

    assert runner._policy_initial_hash == initial_hash
    assert runner._policy_final_hash != initial_hash


def test_imagined_success_sft_signal_passes_only_for_a_committed_parameter_change() -> None:
    result = evaluate_imagined_success_sft_signal(
        {
            "actor/success_sft_trajectories": 3.0,
            "actor/success_sft_valid_samples": 12.0,
            "actor/success_sft_optimizer_steps": 1.0,
            "actor/success_sft_grad_norm": 0.25,
            "actor/success_sft_update_committed": 1.0,
            "actor/success_sft_skipped_no_success": 0.0,
        },
        training_mode="imagined_success_sft",
        policy_initial_hash="before",
        policy_final_hash="after",
        applied_policy_steps=1,
    )

    assert result.passed is True
    assert result.failures == ()


def test_imagined_success_sft_signal_reports_missing_success_and_unchanged_policy() -> None:
    result = evaluate_imagined_success_sft_signal(
        {},
        training_mode="imagined_success_sft",
        policy_initial_hash="same",
        policy_final_hash="same",
        applied_policy_steps=0,
    )

    assert result.passed is False
    assert any("no successful" in failure for failure in result.failures)
    assert any("did not change" in failure for failure in result.failures)
