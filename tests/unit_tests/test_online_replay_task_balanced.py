from __future__ import annotations

import random

import numpy as np
import torch

from dreamervla.runners.online_replay import (
    OnlineReplay,
    pack_replay_task_stats_for_ddp,
    unpack_replay_task_stats_from_ddp,
)


def _step(task_id: int, t: int, *, success: bool = False, done: bool = False) -> dict:
    return {
        "image": np.full((1, 1, 3), t, dtype=np.uint8),
        "obs_embedding": np.full((2,), t, dtype=np.float32),
        "policy_action": np.zeros((1,), dtype=np.float32),
        "wm_action": np.full((1,), t, dtype=np.float32),
        "reward": np.float32(1.0 if success else 0.0),
        "done": np.float32(done or success),
        "is_first": t == 0,
        "is_terminal": np.float32(success),
        "is_last": np.float32(done or success),
        "task_id": task_id,
    }


def _episode(task_id: int, length: int, *, success: bool) -> list[dict]:
    return [
        _step(task_id, t, success=success and t == length - 1, done=t == length - 1)
        for t in range(length)
    ]


def _collect_schema_step(
    task_id: int, t: int, *, success: bool = False, done: bool = False
) -> dict:
    return {
        "obs_embedding": np.full((2,), t, dtype=np.float32),
        "actions": np.full((1,), t, dtype=np.float32),
        "rewards": np.float32(1.0 if success else 0.0),
        "sparse_rewards": np.uint8(1 if success else 0),
        "dones": np.uint8(1 if done or success else 0),
        "success": bool(success),
        "task_id": int(task_id),
    }


def _collect_schema_episode(task_id: int, length: int, *, success: bool) -> list[dict]:
    return [
        _collect_schema_step(
            task_id,
            t,
            success=success and t == length - 1,
            done=t == length - 1,
        )
        for t in range(length)
    ]


def test_online_replay_samples_failed_episodes_only_from_prefix() -> None:
    random.seed(0)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        failure_prefix_steps=4,
        task_balanced=True,
    )
    replay.add_episode(_episode(task_id=2, length=10, success=False))

    batch = replay.sample(16)

    assert set(batch["task_ids"].tolist()) == {2}
    assert batch["start_indices"].max().item() <= 1


def test_online_replay_balances_available_tasks() -> None:
    random.seed(1)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        failure_prefix_steps=4,
        task_balanced=True,
    )
    replay.add_episode(_episode(task_id=2, length=10, success=True))
    replay.add_episode(_episode(task_id=9, length=10, success=True))

    batch = replay.sample(6)

    assert batch["task_ids"].tolist().count(2) == 3
    assert batch["task_ids"].tolist().count(9) == 3


def test_online_replay_can_sample_without_images() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3, task_balanced=False)
    replay.add_episode(_episode(task_id=2, length=5, success=True))

    batch = replay.sample(2, include_images=False)

    assert "images" not in batch
    assert batch["obs_embedding"].dtype == torch.float32
    assert batch["obs_embedding"].shape == (2, 3, 2)


def test_online_replay_samples_collect_schema_steps_without_reward_aliases() -> None:
    random.seed(0)
    replay = OnlineReplay(capacity=100, sequence_length=3, task_balanced=False)
    replay.add_episode(_collect_schema_episode(task_id=2, length=5, success=True))

    batch = replay.sample(1)

    assert batch["rewards"].shape == (1, 3)
    assert batch["actions"].shape == (1, 3, 1)
    assert batch["current_actions"].shape == (1, 3, 1)
    assert "images" not in batch


def test_online_replay_state_dict_round_trips_records_and_cursors() -> None:
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_ids=(2, 9),
        rank=4,
        replay_sampling={"latest_online_required": True},
    )
    replay.set_policy_version(7)
    online = replay.add_episode(_episode(task_id=2, length=5, success=True), source="online")
    replay.add_episode(_episode(task_id=9, length=6, success=False), source="coldstart")

    restored = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_ids=(2, 9),
        rank=4,
        replay_sampling={"latest_online_required": True},
    )
    restored.load_state_dict(replay.state_dict())

    assert online is not None
    assert restored.num_transitions == replay.num_transitions
    assert restored.task_stats((2, 9)) == replay.task_stats((2, 9))
    assert restored._current_policy_version == 7
    assert restored._next_episode_id == replay._next_episode_id
    assert restored._next_collection_index == replay._next_collection_index
    assert restored._next_task_episode_index == replay._next_task_episode_index
    assert restored._pending_latest_online_episode_ids == {int(online["episode_id"])}

    next_record = restored.add_episode(
        _episode(task_id=2, length=5, success=False),
        source="online",
    )

    assert next_record is not None
    assert int(next_record["episode_id"]) == int(replay._next_episode_id)
    assert int(next_record["policy_version"]) == 7


def test_online_replay_samples_proprio_and_episode_language_sidecar() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3, task_balanced=False)
    episode = _episode(task_id=2, length=3, success=True)
    for idx, step in enumerate(episode):
        step["proprio"] = np.full((8,), float(idx), dtype=np.float32)
        step["lang_emb"] = np.arange(6, dtype=np.float32) + 0.25
    replay.add_episode(episode)

    batch = replay.sample(1, include_images=False)

    assert batch["proprio"].shape == (1, 3, 8)
    assert batch["proprio"].dtype == torch.float32
    assert batch["lang_emb"].shape == (1, 6)
    assert batch["lang_emb"].dtype == torch.float32
    assert torch.allclose(batch["proprio"][0, :, 0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(batch["lang_emb"][0], torch.arange(6, dtype=torch.float32) + 0.25)


def test_online_replay_classifier_windows_include_wm_proprio_language() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=4, task_balanced=False)
    episode = _episode(task_id=2, length=8, success=True)
    for idx, step in enumerate(episode):
        step["obs_embedding"] = np.full((2, 4), float(idx), dtype=np.float32)
        step["proprio"] = np.full((8,), float(idx), dtype=np.float32)
        step["lang_emb"] = np.arange(6, dtype=np.float32) + 0.25
    replay.add_episode(episode)

    batch = replay.sample_classifier_windows(
        1,
        window=2,
        chunk_size=2,
        chunk_pool="last",
        early_neg_stride=100,
    )

    assert batch["windows"].shape == (1, 2, 2, 4)
    assert batch["is_end_window"].item() is True
    assert batch["proprio"].shape == (1, 2, 8)
    assert batch["lang_emb"].shape == (1, 6)
    assert torch.allclose(batch["proprio"][0, :, 0], torch.tensor([5.0, 7.0]))
    assert torch.allclose(batch["lang_emb"][0], torch.arange(6, dtype=torch.float32) + 0.25)


def test_online_replay_classifier_windows_follow_wmpo_episode_protocol(monkeypatch) -> None:
    random.seed(3)
    monkeypatch.setattr(random, "random", lambda: 0.99)
    monkeypatch.setattr(random, "choice", lambda seq: list(seq)[0])

    def sample_one(success: bool) -> dict[str, torch.Tensor]:
        replay = OnlineReplay(capacity=100, sequence_length=4, task_balanced=False)
        episode = _episode(task_id=2, length=12, success=success)
        for idx, step in enumerate(episode):
            step["obs_embedding"] = np.full((1,), float(idx), dtype=np.float32)
        replay.add_episode(episode)
        return replay.sample_classifier_windows(
            2,
            window=2,
            chunk_size=2,
            chunk_pool="last",
            early_neg_stride=4,
        )

    batch = sample_one(success=True)
    assert batch["labels"].tolist() == [1, 0]
    assert batch["is_end_window"].tolist() == [True, False]
    assert batch["episode_ids"].tolist() == [0, 0]
    assert batch["finish_steps"].tolist() == [12, 12]
    assert batch["window_end_indices"].tolist() == [12, 8]
    assert batch["source_success"].tolist() == [True, True]
    assert torch.allclose(
        batch["windows"].squeeze(-1),
        torch.tensor(
            [
                [9.0, 11.0],
                [5.0, 7.0],
            ]
        ),
    )

    batch = sample_one(success=False)
    assert batch["labels"].tolist() == [0, 0]
    assert batch["is_end_window"].tolist() == [True, False]
    assert batch["episode_ids"].tolist() == [0, 0]
    assert batch["finish_steps"].tolist() == [12, 12]
    assert batch["window_end_indices"].tolist() == [12, 8]
    assert batch["source_success"].tolist() == [False, False]

    replay = OnlineReplay(capacity=100, sequence_length=4, task_balanced=False)
    episode = _episode(task_id=2, length=12, success=True)
    for idx, step in enumerate(episode):
        step["obs_embedding"] = np.full((1,), float(idx), dtype=np.float32)
    replay.add_episode(episode)

    first = replay.sample_classifier_windows(
        1,
        window=2,
        chunk_size=2,
        chunk_pool="last",
        early_neg_stride=4,
    )
    second = replay.sample_classifier_windows(
        1,
        window=2,
        chunk_size=2,
        chunk_pool="last",
        early_neg_stride=4,
    )

    assert first["labels"].tolist() == [1]
    assert first["is_end_window"].tolist() == [True]
    assert first["window_end_indices"].tolist() == [12]
    assert second["labels"].tolist() == [0]
    assert second["is_end_window"].tolist() == [False]
    assert second["window_end_indices"].tolist() == [8]


def test_online_replay_training_readiness_requires_each_task() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3)

    replay.add_episode(_episode(task_id=2, length=10, success=True))

    assert (
        replay.ready_for_training(
            min_transitions=3,
            task_ids=(2, 9),
            min_episodes_per_task=1,
        )
        is False
    )

    replay.add_episode(_episode(task_id=9, length=10, success=False))

    assert (
        replay.ready_for_training(
            min_transitions=3,
            task_ids=(2, 9),
            min_episodes_per_task=1,
        )
        is True
    )


def test_online_replay_reports_per_task_start_pool_stats() -> None:
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        failure_prefix_steps=4,
        failure_prefix_ratio=0.0,
    )

    replay.add_episode(_episode(task_id=2, length=10, success=True))
    replay.add_episode(_episode(task_id=2, length=10, success=False))
    replay.add_episode(_episode(task_id=9, length=5, success=False))

    stats = replay.task_stats(task_ids=(2, 9))

    assert stats["2"]["episodes"] == 2
    assert stats["2"]["successes"] == 1
    assert stats["2"]["failures"] == 1
    assert stats["2"]["sampleable_windows"] == 10
    assert stats["9"]["episodes"] == 1
    assert stats["9"]["successes"] == 0
    assert stats["9"]["failures"] == 1
    assert stats["9"]["sampleable_windows"] == 2


def test_online_replay_keeps_independent_capacity_per_requested_task() -> None:
    replay = OnlineReplay(
        capacity=12,
        sequence_length=3,
        task_ids=(0, 1),
        capacity_mode="per_task",
    )

    replay.add_episode(_episode(task_id=0, length=10, success=True))
    replay.add_episode(_episode(task_id=1, length=10, success=True))
    replay.add_episode(_episode(task_id=1, length=10, success=True))

    stats = replay.task_stats(task_ids=(0, 1))

    assert stats["0"]["episodes"] == 1
    assert stats["0"]["transitions"] == 10
    assert stats["1"]["episodes"] == 1
    assert stats["1"]["transitions"] == 10


def test_online_replay_can_report_global_ddp_task_stats() -> None:
    replay_rank0 = OnlineReplay(capacity=100, sequence_length=3, task_ids=(0, 1))
    replay_rank1 = OnlineReplay(capacity=100, sequence_length=3, task_ids=(0, 1))
    replay_rank0.add_episode(_episode(task_id=0, length=10, success=True))
    replay_rank1.add_episode(_episode(task_id=1, length=8, success=False))

    packed = pack_replay_task_stats_for_ddp(
        replay_rank0,
        task_ids=(0, 1),
        min_transitions=3,
        min_episodes_per_task=1,
    ) + pack_replay_task_stats_for_ddp(
        replay_rank1,
        task_ids=(0, 1),
        min_transitions=3,
        min_episodes_per_task=1,
    )

    stats, coverage_ready, all_ranks_ready = unpack_replay_task_stats_from_ddp(
        packed,
        task_ids=(0, 1),
        world_size=2,
        min_transitions=3,
        min_episodes_per_task=1,
    )

    assert coverage_ready is True
    assert all_ranks_ready is False
    assert stats["0"]["episodes"] == 1
    assert stats["0"]["successes"] == 1
    assert stats["1"]["episodes"] == 1
    assert stats["1"]["failures"] == 1


def test_online_replay_stamps_episode_source_and_returns_source_ids() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3, task_balanced=False)
    cold = replay.add_episode(_episode(task_id=0, length=6, success=True), source="coldstart")
    online = replay.add_episode(_episode(task_id=0, length=6, success=False))

    assert cold is not None and cold["source"] == "coldstart" and cold["source_id"] == 0
    assert online is not None and online["source"] == "online" and online["source_id"] == 1

    batch = replay.sample(8)

    assert set(batch["replay_source_ids"].tolist()) <= {0, 1}


def test_online_replay_three_pool_recent_samples_newest_online_only() -> None:
    random.seed(2)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_balanced=False,
        replay_sampling={
            "enabled": True,
            "recent_episode_count": 1,
            "mix": {
                "online_recent": 1.0,
                "online_replay": 0.0,
                "coldstart_anchor": 0.0,
            },
        },
    )
    replay.add_episode(_episode(task_id=0, length=6, success=True), source="coldstart")
    old_online = replay.add_episode(_episode(task_id=0, length=6, success=True), source="online")
    new_online = replay.add_episode(_episode(task_id=0, length=6, success=True), source="online")

    batch = replay.sample(12)

    assert old_online is not None and new_online is not None
    assert set(batch["episode_ids"].tolist()) == {int(new_online["episode_id"])}
    assert set(batch["replay_source_ids"].tolist()) == {1}


def test_online_replay_three_pool_can_sample_coldstart_anchor_only() -> None:
    random.seed(3)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_balanced=False,
        replay_sampling={
            "enabled": True,
            "recent_episode_count": 2,
            "mix": {
                "online_recent": 0.0,
                "online_replay": 0.0,
                "coldstart_anchor": 1.0,
            },
        },
    )
    cold = replay.add_episode(_episode(task_id=0, length=6, success=True), source="coldstart")
    replay.add_episode(_episode(task_id=0, length=6, success=True), source="online")

    batch = replay.sample(8)

    assert cold is not None
    assert set(batch["episode_ids"].tolist()) == {int(cold["episode_id"])}
    assert set(batch["replay_source_ids"].tolist()) == {0}


def test_online_replay_three_pool_mixed_weights_can_return_anchor_and_recent() -> None:
    random.seed(4)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_balanced=False,
        replay_sampling={
            "enabled": True,
            "recent_episode_count": 1,
            "mix": {
                "online_recent": 0.5,
                "online_replay": 0.0,
                "coldstart_anchor": 0.5,
            },
        },
    )
    replay.add_episode(_episode(task_id=0, length=6, success=True), source="coldstart")
    replay.add_episode(_episode(task_id=0, length=6, success=True), source="online")

    batch = replay.sample(64)

    assert {0, 1} <= set(batch["replay_source_ids"].tolist())


def test_online_replay_three_pool_falls_back_when_recent_pool_empty() -> None:
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_balanced=False,
        replay_sampling={
            "enabled": True,
            "recent_episode_count": 0,
            "mix": {
                "online_recent": 1.0,
                "online_replay": 0.0,
                "coldstart_anchor": 0.0,
            },
        },
    )
    replay.add_episode(_episode(task_id=0, length=6, success=True), source="coldstart")

    batch = replay.sample(4)

    assert set(batch["replay_source_ids"].tolist()) == {0}


def test_online_replay_three_pool_sampling_falls_back_to_available_pool() -> None:
    random.seed(7)
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_balanced=False,
        replay_sampling={
            "enabled": True,
            "recent_episode_count": 8,
            "mix": {
                "online_recent": 0.0,
                "online_replay": 0.0,
                "coldstart_anchor": 1.0,
            },
        },
    )
    online = replay.add_episode(_episode(task_id=0, length=6, success=True), source="online")

    batch = replay.sample(3)

    assert online is not None
    assert set(batch["episode_ids"].tolist()) == {int(online["episode_id"])}
    assert set(batch["replay_source_ids"].tolist()) == {1}


def test_online_replay_latest_online_required_samples_new_episode_first() -> None:
    replay = OnlineReplay(
        capacity=100,
        sequence_length=3,
        task_balanced=False,
        replay_sampling={
            "enabled": True,
            "recent_episode_count": 1,
            "latest_online_required": True,
            "mix": {
                "online_recent": 0.0,
                "online_replay": 0.0,
                "coldstart_anchor": 1.0,
            },
        },
    )
    replay.add_episode(_episode(task_id=0, length=6, success=True), source="coldstart")
    online = replay.add_episode(_episode(task_id=0, length=6, success=True), source="online")

    batch = replay.sample(1)

    assert online is not None
    assert int(batch["episode_ids"][0]) == int(online["episode_id"])
    assert int(batch["replay_source_ids"][0]) == 1


def test_online_replay_classifier_evidence_readiness_requires_pos_and_neg() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3)
    replay.add_episode(_episode(task_id=0, length=6, success=True))

    assert replay.ready_for_training(
        min_transitions=3,
        task_ids=(0,),
        min_episodes_per_task=1,
        require_classifier_evidence=True,
    ) is False

    replay.add_episode(_episode(task_id=0, length=6, success=False))

    assert replay.ready_for_training(
        min_transitions=3,
        task_ids=(0,),
        min_episodes_per_task=1,
        require_classifier_evidence=True,
    ) is True


def test_online_replay_readiness_requires_sampleable_window_budget() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3)
    replay.add_episode(_episode(task_id=0, length=6, success=True))

    assert replay.sampleable_window_count() == 4
    assert (
        replay.ready_for_training(
            min_transitions=3,
            task_ids=(0,),
            min_episodes_per_task=1,
            min_sampleable_windows=5,
        )
        is False
    )
    assert (
        replay.ready_for_training(
            min_transitions=3,
            task_ids=(0,),
            min_episodes_per_task=1,
            min_sampleable_windows=4,
        )
        is True
    )


def test_ddp_replay_readiness_packs_sampleable_window_budget() -> None:
    replay = OnlineReplay(capacity=100, sequence_length=3)
    replay.add_episode(_episode(task_id=0, length=6, success=True))

    packed = pack_replay_task_stats_for_ddp(
        replay,
        task_ids=(0,),
        min_transitions=3,
        min_episodes_per_task=1,
        min_sampleable_windows=5,
    )
    _stats, coverage_ready, all_ranks_ready = unpack_replay_task_stats_from_ddp(
        packed,
        task_ids=(0,),
        world_size=1,
        min_transitions=3,
        min_episodes_per_task=1,
        min_sampleable_windows=5,
    )

    assert coverage_ready is False
    assert all_ranks_ready is False
