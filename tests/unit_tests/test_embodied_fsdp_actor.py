from __future__ import annotations

from dataclasses import replace
import uuid

import pytest
import ray
import torch

from dreamervla.hybrid_engines.weight_syncer import PatchWeightSyncer
from dreamervla.scheduler.cluster import Cluster
from dreamervla.workers.actor._test_models import TinyLumosPolicy
import dreamervla.workers.actor.embodied_fsdp_actor as embodied_fsdp_actor
from dreamervla.workers.actor.embodied_fsdp_actor import EmbodiedFSDPActor
from dreamervla.workers.cotrain.messages import TrajectoryShard, collate_trajectory_shards
from dreamervla.workers.env.trajectory_env_worker import _concat_trajectory_shards


def _actor_cfg(store_name: str | None = None) -> dict:
    train_cfg = {
        "device": "cpu",
        "lr": 1e-3,
        "fsdp": {"strategy": "none", "precision": "fp32"},
        "algorithm_cfg": {
            "group_size": 2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "clip_ratio_c": 3.0,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ppo_update_epochs": 1,
            "entropy_coef": 0.0,
        },
    }
    if store_name is not None:
        train_cfg["syncer"] = {"store_name": store_name}
    return {
        "policy_cfg": {
            "target": "dreamervla.workers.actor._test_models:TinyLumosPolicy",
            "kwargs": {"hidden_dim": 4, "action_dim": 3, "chunk_size": 2},
        },
        "init_ckpt": {},
        "train_cfg": train_cfg,
    }


def _shard(reward0: float, reward1: float) -> TrajectoryShard:
    actions = torch.zeros(2, 2, 2, 3)
    actions[:, 1].fill_(1.0)
    return TrajectoryShard(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_ids=[0, 1],
        actions=actions,
        rewards=torch.tensor(
            [
                [[reward0, 0.0], [reward1, 0.0]],
                [[reward0, 0.0], [reward1, 0.0]],
            ],
            dtype=torch.float32,
        ),
        dones=torch.zeros(2, 2, 2, dtype=torch.bool),
        prev_logprobs=torch.zeros(2, 2),
        prev_values=None,
        forward_inputs={
            "hidden": torch.ones(2, 2, 4),
            "action": actions.clone(),
        },
        versions={"policy": torch.zeros(2, 2, dtype=torch.long)},
    )


def _step_action_shard(reward0: float, reward1: float) -> TrajectoryShard:
    actions = torch.zeros(2, 2, 3)
    actions[:, 1].fill_(1.0)
    return TrajectoryShard(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_ids=[0, 1],
        actions=actions,
        rewards=torch.tensor(
            [[reward0, reward1], [reward0, reward1]],
            dtype=torch.float32,
        ),
        dones=torch.zeros(2, 2, dtype=torch.bool),
        prev_logprobs=torch.zeros(2, 2),
        prev_values=None,
        forward_inputs={"hidden": torch.ones(2, 2, 4)},
        versions={"policy": torch.zeros(2, 2, dtype=torch.long)},
    )


def _variable_length_shard(
    *,
    steps: int,
    slot_id: int,
    reward: float,
) -> TrajectoryShard:
    actions = torch.full((steps, 1, 2, 3), float(slot_id), dtype=torch.float32)
    return TrajectoryShard(
        env_rank=0,
        slot_id=int(slot_id),
        task_id=0,
        episode_ids=[int(slot_id)],
        actions=actions,
        rewards=torch.full((steps, 1, 2), float(reward), dtype=torch.float32),
        dones=torch.zeros(steps, 1, 2, dtype=torch.bool),
        prev_logprobs=torch.zeros(steps, 1),
        prev_values=None,
        forward_inputs={
            "hidden": torch.ones(steps, 1, 4),
            "action": actions.clone(),
        },
        versions={"policy": torch.zeros(steps, 1, dtype=torch.long)},
    )


class _MemoryActorChannel:
    def __init__(self, items: list[object]) -> None:
        self.items = list(items)
        self.get_batch_calls: list[tuple[int, str]] = []

    def get_batch(self, n: int, *, key: str = "default") -> list[object]:
        self.get_batch_calls.append((int(n), str(key)))
        out = self.items[: int(n)]
        del self.items[: int(n)]
        return out


def test_actor_group_computes_group_advantages_from_trajectory_rewards() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()

    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    metrics = actor.compute_advantages_and_returns()

    assert metrics["actor/trajectory_count"] == 2.0
    assert metrics["actor/advantage_std"] > 0.0


def test_actor_recv_rollout_trajectories_batches_channel_and_reports_timings(
    monkeypatch,
) -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    channel = _MemoryActorChannel([_shard(0.0, 1.0)])
    monkeypatch.setattr(
        embodied_fsdp_actor.Channel,
        "connect",
        staticmethod(lambda name: channel),
    )

    metrics = actor.recv_rollout_trajectories("actor", expected_shards=1)

    assert channel.get_batch_calls == [(1, "default")]
    assert metrics["actor/received_shards"] == 1.0
    assert metrics["actor/channel_get_batch_s"] >= 0.0
    assert metrics["actor/load_trajectory_shards_s"] >= 0.0
    assert actor.batch is not None


def test_actor_group_sums_chunk_rewards_per_trajectory() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()

    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    metrics = actor.compute_advantages_and_returns()

    assert metrics["actor/trajectory_count"] == 2.0
    assert actor.returns.shape == (2,)
    assert actor.returns.tolist() == [0.0, 2.0]


def test_collate_trajectory_shards_pads_variable_length_with_loss_mask() -> None:
    batch = collate_trajectory_shards(
        [
            _variable_length_shard(steps=1, slot_id=0, reward=1.0),
            _variable_length_shard(steps=3, slot_id=1, reward=2.0),
        ]
    )

    assert batch.actions.shape == (3, 2, 2, 3)
    assert batch.loss_mask.tolist() == [[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    assert batch.dones[1:, 0].all()


def test_buffered_slot_shard_keeps_loss_mask_after_episode_reset() -> None:
    terminal_episode = replace(
        _variable_length_shard(steps=1, slot_id=0, reward=1.0),
        episode_ids=[10],
        dones=torch.ones(1, 1, 2, dtype=torch.bool),
    )
    reset_episode = replace(
        _variable_length_shard(steps=1, slot_id=0, reward=2.0),
        episode_ids=[11],
        dones=torch.zeros(1, 1, 2, dtype=torch.bool),
    )

    batch = collate_trajectory_shards(
        [_concat_trajectory_shards([terminal_episode, reset_episode])]
    )

    assert batch.loss_mask.squeeze(1).tolist() == [1.0, 1.0]


def test_actor_run_training_masks_padded_variable_length_trajectories() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()

    actor.load_trajectory_shards(
        [
            _variable_length_shard(steps=1, slot_id=0, reward=0.0),
            _variable_length_shard(steps=3, slot_id=1, reward=1.0),
        ]
    )
    advantage_metrics = actor.compute_advantages_and_returns()
    train_metrics = actor.run_training()

    assert advantage_metrics["actor/trajectory_count"] == 2.0
    assert advantage_metrics["actor/loss_mask_sum"] == 4.0
    assert actor.returns is not None
    assert actor.returns.tolist() == [0.0, 6.0]
    assert train_metrics["actor/ppo_updates"] == 1.0


def test_actor_run_training_updates_policy_parameters() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    before = {key: value.clone() for key, value in actor.state_dict().items()}

    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    actor.compute_advantages_and_returns()
    metrics = actor.run_training()

    after = actor.state_dict()
    assert metrics["actor/ppo_updates"] == 1.0
    assert any(not torch.equal(before[key], after[key]) for key in before)


def test_actor_run_training_rejects_step_action_tensors_for_manual_cotrain() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()

    actor.load_trajectory_shards([_step_action_shard(0.0, 1.0)])
    actor.compute_advantages_and_returns()
    with pytest.raises(ValueError, match="chunk-level actions"):
        actor.run_training()


def test_sync_model_to_rollout_pushes_patch_and_returns_version_metric() -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        store_name = f"test-actor-rollout-patch-{uuid.uuid4().hex}"
        actor = EmbodiedFSDPActor(**_actor_cfg(store_name=store_name))
        actor.init()
        actor.set_global_step(7)
        actor.load_trajectory_shards([_shard(0.0, 1.0)])
        actor.compute_advantages_and_returns()
        actor.run_training()

        metrics = actor.sync_model_to_rollout()

        target = TinyLumosPolicy(hidden_dim=4, action_dim=3, chunk_size=2)
        pulled = PatchWeightSyncer(store_name=store_name).pull(
            "policy",
            target,
            local_version=0,
        )
        assert metrics["sync/policy_version"] == 7.0
        assert metrics["sync/policy_export_s"] >= 0.0
        assert metrics["sync/policy_push_s"] >= 0.0
        assert metrics["sync/policy_tensors"] > 0.0
        assert pulled == 7
        for name, value in actor.state_dict().items():
            assert torch.allclose(target.state_dict()[name], value)
    finally:
        cluster.shutdown()


def test_sync_model_to_rollout_nonzero_rank_participates_without_pushing_patch() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    actor.rank = 1
    state_calls = []
    pushes = []

    def fake_state_dict():
        state_calls.append("state_dict")
        return {"linear.weight": torch.ones(1, 1)}

    class _FakeSyncer:
        def push(self, key: str, state_dict: dict, version: int) -> None:
            pushes.append((key, state_dict, version))

    actor.state_dict = fake_state_dict  # type: ignore[method-assign]
    actor.syncer = _FakeSyncer()  # type: ignore[assignment]

    metrics = actor.sync_model_to_rollout("policy", version=9)

    assert state_calls == ["state_dict"]
    assert pushes == []
    assert metrics["sync/policy_version"] == 9.0
    assert metrics["sync/policy_export_s"] >= 0.0
    assert metrics["sync/policy_push_s"] == 0.0
    assert metrics["sync/policy_tensors"] > 0.0
