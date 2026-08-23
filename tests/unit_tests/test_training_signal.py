from __future__ import annotations

from dreamervla.runtime.training_signal import evaluate_imagined_success_sft_signal


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
