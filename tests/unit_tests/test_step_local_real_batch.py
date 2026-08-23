from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from dreamervla.workers.cotrain.messages import RolloutResultMsg
from dreamervla.workers.env.evaluation_env_worker import EvaluationEnvironmentWorker
from dreamervla.workers.env.trajectory_env_worker import RealEnvWorker


class _ReplaySink:
    def __init__(self) -> None:
        self.episodes: list[list[dict]] = []

    def add_episode(self, episode, source="online"):
        del source
        self.episodes.append(list(episode))

    def set_policy_version(self, version):
        self.policy_version = int(version)


def _worker(*, horizon: int = 1, num_action_chunks: int = 1) -> RealEnvWorker:
    worker = RealEnvWorker(
        env_cfg={
            "target": "dreamervla.workers.env._test_envs:NoSidecarTrainEnv",
            "kwargs": {"horizon": horizon, "image_shape": [4, 4, 3]},
        },
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=num_action_chunks,
        num_action_chunks=num_action_chunks,
        replay=_ReplaySink(),
        request_final_bootstrap=False,
    )
    worker.init()
    worker.set_global_step(4)
    worker.bootstrap_obs()
    return worker


def test_real_task_schedule_balances_32_trajectories_over_selected_suite() -> None:
    worker = RealEnvWorker(
        env_cfg={
            "target": "dreamervla.workers.env._test_envs:NoSidecarTrainEnv",
            "kwargs": {"horizon": 1, "image_shape": [4, 4, 3]},
        },
        num_slots=8,
        rollout_epoch=4,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_ids=tuple(range(10)),
        request_final_bootstrap=False,
    )

    scheduled = [
        worker._scheduled_task_id(slot_id, episode_id=episode_id)
        for episode_id in range(4)
        for slot_id in range(8)
    ]
    counts = Counter(scheduled)

    assert len(scheduled) == 32
    assert set(counts) == set(range(10))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_real_trajectory_batch_drains_exactly_once_with_raw_and_tokens() -> None:
    worker = _worker()
    result = RolloutResultMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=0,
        actions=torch.zeros(1, 7),
        prev_logprobs=torch.zeros(1),
        prev_values=None,
        forward_inputs={
            "hidden": torch.ones(1, 2, 3),
            "lang_emb": torch.ones(1, 4),
            "action_token_ids": torch.arange(7).reshape(1, 1, 7),
            "input_ids": torch.tensor([[1, 9, 11]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "action": torch.zeros(1, 1, 7),
        },
        versions={"policy": 4, "global_step": 4},
    )

    worker.apply_rollout_result(result)
    batch = worker.drain_real_trajectories(global_step=4)

    assert batch.global_step == 4
    assert batch.num_trajectories == 1
    trajectory = batch.trajectories[0]
    assert trajectory.success is True
    assert trajectory.task_id == 0
    assert trajectory.episode_id == 0
    assert len(trajectory.transitions) == 1
    transition = trajectory.transitions[0]
    assert np.asarray(transition["image"]).shape == (4, 4, 3)
    assert np.asarray(transition["agentview_rgb"]).shape == (4, 4, 3)
    assert np.asarray(transition["state"]).shape == (2,)
    assert np.asarray(transition["policy_action"]).shape == (7,)
    assert np.asarray(transition["action_token_ids_chunk"]).shape == (1, 7)
    assert np.asarray(transition["input_ids"]).tolist() == [1, 9, 11]
    assert transition["policy_decision"] is True
    assert bool(transition["is_last"])

    drained_again = worker.drain_real_trajectories(global_step=4)
    assert drained_again.num_trajectories == 0


def test_eval_trajectory_batch_drains_once_without_replay_writes() -> None:
    worker = EvaluationEnvironmentWorker(
        env_cfg={
            "target": "dreamervla.workers.env._test_envs:NoSidecarTrainEnv",
            "kwargs": {"horizon": 1, "image_shape": [4, 4, 3]},
        },
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_ids=(0,),
    )
    worker.init()
    worker.set_global_step(6)
    worker.bootstrap_obs()
    result = RolloutResultMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=0,
        actions=torch.zeros(1, 7),
        prev_logprobs=torch.zeros(1),
        prev_values=None,
        forward_inputs={"hidden": torch.ones(1, 2, 3)},
        versions={"policy": 6, "global_step": 6},
    )

    worker.apply_rollout_result(result)
    batch = worker.drain_real_trajectories(global_step=6)

    assert worker.replay is None
    assert worker.replay_write_enabled is False
    assert batch.num_trajectories == 1
    assert batch.trajectories[0].global_step == 6
    assert worker.drain_real_trajectories(global_step=6).num_trajectories == 0


def test_drain_does_not_return_another_global_steps_trajectories() -> None:
    worker = _worker()

    batch = worker.drain_real_trajectories(global_step=5)

    assert batch.global_step == 5
    assert batch.num_trajectories == 0


def test_step_local_collection_discards_partial_episode_before_policy_changes() -> None:
    worker = _worker(horizon=3, num_action_chunks=3)
    partial = RolloutResultMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=0,
        actions=torch.zeros(1, 7),
        prev_logprobs=torch.zeros(1),
        prev_values=None,
        forward_inputs={"hidden": torch.ones(1, 2, 3)},
        versions={"policy": 4, "global_step": 4},
    )
    worker.apply_rollout_result(partial)

    worker.set_global_step(5)
    metrics = worker.begin_step_local_real_collection(global_step=5)

    assert metrics["env/real_env/discarded_partial_episodes"] == 1.0
    assert metrics["env/real_env/discarded_partial_transitions"] == 1.0

    current = RolloutResultMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=1,
        step=0,
        actions=torch.zeros(3, 7),
        prev_logprobs=torch.zeros(1),
        prev_values=None,
        forward_inputs={"hidden": torch.ones(1, 2, 3)},
        versions={"policy": 5, "global_step": 5},
    )
    worker.apply_rollout_result(current)
    batch = worker.drain_real_trajectories(global_step=5)

    assert batch.num_trajectories == 1
    assert batch.trajectories[0].episode_id == 1
    assert {int(step["global_step"]) for step in batch.trajectories[0].transitions} == {5}
    assert worker.drain_real_trajectories(global_step=4).num_trajectories == 0


def test_action_chunk_labels_only_the_first_physical_step_as_policy_decision() -> None:
    worker = _worker(horizon=2, num_action_chunks=2)
    result = RolloutResultMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=0,
        actions=torch.zeros(2, 7),
        prev_logprobs=torch.zeros(1),
        prev_values=None,
        forward_inputs={
            "hidden": torch.ones(1, 2, 3),
            "lang_emb": torch.ones(1, 4),
            "action_token_ids": torch.arange(14).reshape(1, 2, 7),
            "input_ids": torch.tensor([[1, 9, 11]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        versions={"policy": 4, "global_step": 4},
    )

    worker.apply_rollout_result(result)
    trajectory = worker.drain_real_trajectories(global_step=4).trajectories[0]

    assert len(trajectory.transitions) == 2
    first, second = trajectory.transitions
    assert first["policy_decision"] is True
    assert np.asarray(first["action_token_ids_chunk"]).shape == (2, 7)
    assert second["policy_decision"] is False
    assert "action_token_ids_chunk" not in second
    assert "input_ids" not in second
    assert "attention_mask" not in second
    assert "obs_embedding" not in second
