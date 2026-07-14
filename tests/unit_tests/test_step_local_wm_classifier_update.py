from __future__ import annotations

import numpy as np
import pytest
import torch

from dreamervla.workers.actor.learner_worker import _calibrate_binary_threshold
from dreamervla.workers.cotrain.messages import RealTrajectory, RealTrajectoryBatch
from dreamervla.workers.replay.replay_worker import ReplayWorker


def _episode(value: float, *, encoder_version: int, length: int = 4) -> list[dict]:
    return [
        {
            "obs_embedding": np.full((2, 3), value, dtype=np.float32),
            "lang_emb": np.full((2,), value, dtype=np.float32),
            "proprio": np.full((1,), value, dtype=np.float32),
            "action": np.full((1,), value, dtype=np.float32),
            "reward": float(index == length - 1),
            "done": bool(index == length - 1),
            "is_terminal": bool(index == length - 1),
            "is_last": bool(index == length - 1),
            "task_id": 0,
            "encoder_version": int(encoder_version),
        }
        for index in range(length)
    ]


def _batch(version: int) -> RealTrajectoryBatch:
    return RealTrajectoryBatch(
        global_step=version,
        trajectories=(
            RealTrajectory(
                env_rank=0,
                slot_id=0,
                task_id=0,
                episode_id=version,
                global_step=version,
                success=True,
                transitions=tuple(_episode(float(version), encoder_version=version)),
            ),
        ),
    )


def _worker() -> ReplayWorker:
    worker = ReplayWorker(
        {
            "capacity": 100,
            "sequence_length": 2,
            "task_ids": (0,),
            "rank": 0,
        }
    )
    worker.init()
    return worker


def test_step_local_load_replaces_old_replay_instead_of_appending() -> None:
    worker = _worker()
    worker.add_episode(_episode(1.0, encoder_version=1))

    metrics = worker.replace_real_trajectories(_batch(7))

    replay = worker._replay()
    assert len(replay.episodes) == 1
    assert replay.num_transitions == 4
    assert replay.episodes[0]["episode"][0]["encoder_version"] == 7
    assert metrics["replay_buffer/step_local_trajectories"] == 1.0
    assert metrics["replay_buffer/step_local_encoder_version"] == 7.0


def test_step_local_load_rejects_mixed_or_stale_encoder_versions() -> None:
    worker = _worker()
    bad = RealTrajectoryBatch(
        global_step=7,
        trajectories=(
            RealTrajectory(
                env_rank=0,
                slot_id=0,
                task_id=0,
                episode_id=0,
                global_step=7,
                success=False,
                transitions=tuple(_episode(0.0, encoder_version=6)),
            ),
        ),
    )

    with pytest.raises(ValueError, match="encoder_version"):
        worker.replace_real_trajectories(bad)

    assert worker.size() == 0


def test_frozen_real_load_appends_history_and_preserves_explicit_failure() -> None:
    worker = _worker()
    worker.add_episode(
        _episode(1.0, encoder_version=1),
        source="coldstart",
        success=False,
    )
    current = RealTrajectoryBatch(
        global_step=7,
        trajectories=(
            RealTrajectory(
                env_rank=0,
                slot_id=0,
                task_id=0,
                episode_id=7,
                global_step=7,
                # The copied transitions contain positive terminal aliases, so
                # this proves the trajectory outcome is the authoritative label.
                success=False,
                transitions=tuple(_episode(7.0, encoder_version=1)),
            ),
        ),
    )

    metrics = worker.append_real_trajectories(current)

    replay = worker._replay()
    assert len(replay.episodes) == 2
    assert [record["source"] for record in replay.episodes] == ["coldstart", "online"]
    assert [record["success"] for record in replay.episodes] == [False, False]
    assert worker.eligible_initial_condition_count("failed_episode_start") == 2
    assert metrics["replay_buffer/appended_trajectories"] == 1.0


def test_classifier_threshold_is_recalibrated_for_best_current_step_f1() -> None:
    threshold, metrics = _calibrate_binary_threshold(
        labels=torch.tensor([0, 1, 0, 1]),
        probabilities=torch.tensor([0.1, 0.4, 0.6, 0.9]),
        previous_threshold=0.5,
    )

    assert threshold == pytest.approx(0.4)
    assert metrics["cls/calibration_updated"] == 1.0
    assert metrics["cls/calibration_f1"] == pytest.approx(0.8)


def test_classifier_threshold_keeps_previous_value_for_single_class_step() -> None:
    threshold, metrics = _calibrate_binary_threshold(
        labels=torch.ones(4, dtype=torch.long),
        probabilities=torch.tensor([0.1, 0.4, 0.6, 0.9]),
        previous_threshold=0.45,
    )

    assert threshold == pytest.approx(0.45)
    assert metrics["cls/calibration_updated"] == 0.0
    assert metrics["cls/calibration_single_class"] == 1.0
