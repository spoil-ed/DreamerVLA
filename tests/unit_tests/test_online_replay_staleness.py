"""Phase 4: OnlineReplay stamps a rollout policy version and the sample staleness
gate drops over-stale episodes (with a fallback so the learner is never starved).
"""

from __future__ import annotations

import random

from test_online_replay_task_balanced import _episode

from dreamervla.runtime.online_replay import OnlineReplay


def test_add_episode_stamps_current_policy_version():
    replay = OnlineReplay(capacity=100, sequence_length=3)
    replay.set_policy_version(5)
    rec = replay.add_episode(_episode(task_id=2, length=10, success=True))
    assert rec["policy_version"] == 5
    replay.set_policy_version(8)
    rec2 = replay.add_episode(_episode(task_id=2, length=10, success=True))
    assert rec2["policy_version"] == 8


def test_sample_drops_stale_when_fresh_available():
    random.seed(0)
    replay = OnlineReplay(capacity=10_000, sequence_length=3, task_balanced=False)
    replay.set_policy_version(0)
    for _ in range(5):
        replay.add_episode(_episode(task_id=2, length=10, success=True))  # stale (v0)
    replay.set_policy_version(10)
    for _ in range(5):
        replay.add_episode(_episode(task_id=3, length=10, success=True))  # fresh (v10)

    # current=10, threshold=2 → v0 episodes (age 10) stale, v10 (age 0) fresh.
    batch = replay.sample(40, staleness_threshold=2)
    assert set(batch["task_ids"].tolist()) == {3}


def test_sample_falls_back_when_all_stale():
    random.seed(0)
    replay = OnlineReplay(capacity=10_000, sequence_length=3, task_balanced=False)
    replay.set_policy_version(0)
    replay.add_episode(_episode(task_id=2, length=10, success=True))
    replay.set_policy_version(100)  # the only episode (v0) is now far too stale

    # All stale → fallback to all valid so the learner is not starved.
    batch = replay.sample(8, staleness_threshold=2)
    assert set(batch["task_ids"].tolist()) == {2}


def test_sample_without_threshold_is_unchanged():
    random.seed(0)
    replay = OnlineReplay(capacity=10_000, sequence_length=3, task_balanced=False)
    replay.add_episode(_episode(task_id=2, length=10, success=True))
    batch = replay.sample(4)  # default path: no staleness arg
    assert len(batch["task_ids"]) == 4
