from __future__ import annotations

import pytest

from dreamervla.runners.cotrain_runner import _classifier_checkpoint_threshold


def test_trajectory_threshold_comes_from_episode_f1_checkpoint_metadata() -> None:
    threshold = _classifier_checkpoint_threshold(
        {
            "classifier_threshold": 0.2,
            "classifier_threshold_metric": "episode_f1",
            "classifier_threshold_objective": "recall_at_zero_fp",
            "classifier_threshold_space": "logit",
            "classifier_success_aggregation": "terminal_streak",
            "classifier_calibrated_task_ids": [1, 2, 4, 8],
            "classifier_threshold_false_positives": 0,
            "classifier_success_consecutive_chunks": 2,
            "best_episode_f1": 0.8,
            "best_episode_selection_score": 0.6,
            "best_episode_threshold": 0.65,
        },
        configured_threshold=0.1,
        require_trajectory_threshold=True,
    )

    assert threshold == pytest.approx(0.65)


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "classifier_threshold": 0.95,
            "classifier_threshold_metric": "window_f1",
            "best_window_f1": 0.95,
        },
        {
            "classifier_threshold": 0.95,
            "best_episode_f1": -1.0,
        },
    ],
)
def test_trajectory_threshold_rejects_unsafe_checkpoint(
    metadata: dict[str, float | str],
) -> None:
    with pytest.raises(ValueError, match="zero false positives"):
        _classifier_checkpoint_threshold(
            metadata,
            configured_threshold=0.33,
            require_trajectory_threshold=True,
        )
