from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import replace

import pytest
import ray
import torch
from torch import nn

import dreamervla.workers.actor.embodied_fsdp_actor as embodied_fsdp_actor
from dreamervla.hybrid_engines.weight_syncer import PatchWeightSyncer
from dreamervla.scheduler.cluster import Cluster
from dreamervla.workers.actor._test_models import TinyLumosPolicy
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
        self.get_calls: list[str] = []
        self.get_batch_calls: list[tuple[int, str]] = []

    def get(self, *, key: str = "default") -> object:
        self.get_calls.append(str(key))
        return self.items.pop(0)

    def get_batch(self, n: int, *, key: str = "default") -> list[object]:
        self.get_batch_calls.append((int(n), str(key)))
        out = self.items[: int(n)]
        del self.items[: int(n)]
        return out


class _OutstandingGraphProbe:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.backward_calls = 0


class _FailIfPreviousGraphLive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, probe: _OutstandingGraphProbe) -> torch.Tensor:
        if probe.active:
            raise RuntimeError("previous actor step graph is still live")
        probe.active += 1
        probe.max_active = max(probe.max_active, probe.active)
        ctx.probe = probe
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        ctx.probe.active -= 1
        ctx.probe.backward_calls += 1
        return grad_output.clone(), None


class _GraphProbePolicy(nn.Module):
    def __init__(self, probe: _OutstandingGraphProbe) -> None:
        super().__init__()
        self.probe = probe
        self.logprob = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, None]:
        bsz = int(batch["hidden"].shape[0])
        logprob = _FailIfPreviousGraphLive.apply(
            self.logprob.expand(bsz),
            self.probe,
        )
        entropy = torch.zeros_like(logprob)
        return logprob, entropy, None


class _NoSyncProbePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.no_sync_calls = 0
        self.no_sync_active = False
        self.forward_sync_states: list[bool] = []

    @contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        self.no_sync_active = True
        try:
            yield
        finally:
            self.no_sync_active = False

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, None]:
        self.forward_sync_states.append(not self.no_sync_active)
        batch_size = int(batch["hidden"].shape[0])
        logprob = self.offset.expand(batch_size)
        return logprob, torch.zeros_like(logprob), None

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(self.parameters(), float(max_norm))


class _FSDPGradNormProbePolicy(_NoSyncProbePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.clip_max_norms: list[float] = []

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        self.clip_max_norms.append(float(max_norm))
        return torch.tensor(3.5)


class _TokenLevelPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.seen_logprob_types: list[object] = []

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        logprob_type = batch.get("logprob_type")
        self.seen_logprob_types.append(logprob_type)
        if logprob_type != "token_level":
            raise AssertionError(f"expected token_level logprob_type, got {logprob_type!r}")
        action = batch["action"].float()
        logprob = torch.zeros_like(action) + self.offset
        entropy = torch.ones_like(logprob) * 0.5
        return logprob, entropy, {}


def test_actor_group_computes_group_advantages_from_trajectory_rewards() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()

    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    metrics = actor.compute_advantages_and_returns()

    assert metrics["actor/trajectory_count"] == 2.0
    assert metrics["actor/advantage_std"] > 0.0


def test_actor_checkpoint_round_trips_policy_optimizer_state() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    actor.compute_advantages_and_returns()
    actor.run_training()

    cfg = _actor_cfg()
    cfg["init_ckpt"] = {
        "policy": actor.state_dict(),
        "policy_optimizer": actor.optimizer_state_dict(),
    }
    restored = EmbodiedFSDPActor(**cfg)
    restored.init()

    source_state = actor._optimizer().state_dict()
    restored_state = restored._optimizer().state_dict()
    assert restored_state["param_groups"] == source_state["param_groups"]
    assert restored_state["state"].keys() == source_state["state"].keys()
    assert all(
        torch.equal(restored_state["state"][key]["step"], value["step"])
        for key, value in source_state["state"].items()
    )


def test_actor_optimizer_uses_complete_hydra_adam_contract() -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["optimizers"] = {
        "policy": {
            "name": "adam",
            "lr": 2.0e-4,
            "betas": [0.8, 0.91],
            "eps": 3.0e-7,
            "weight_decay": 0.04,
        }
    }
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()

    group = actor._optimizer().param_groups[0]
    assert group["lr"] == 2.0e-4
    assert group["betas"] == (0.8, 0.91)
    assert group["eps"] == 3.0e-7
    assert group["weight_decay"] == 0.04


def test_actor_recv_rollout_trajectories_gets_shards_incrementally_and_reports_timings(
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

    assert channel.get_calls == ["default"]
    assert channel.get_batch_calls == []
    assert metrics["actor/received_shards"] == 1.0
    assert metrics["actor/channel_get_batch_s"] >= 0.0
    assert metrics["actor/load_trajectory_shards_s"] >= 0.0
    assert actor.batch is not None


def test_actor_recv_rollout_trajectories_gets_keyed_shards_incrementally(
    monkeypatch,
) -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    channel = _MemoryActorChannel([_shard(0.0, 1.0), _shard(2.0, 3.0)])
    monkeypatch.setattr(
        embodied_fsdp_actor.Channel,
        "connect",
        staticmethod(lambda name: channel),
    )

    metrics = actor.recv_rollout_trajectories(
        "actor",
        keyed_counts=[("wm_env", 2)],
    )

    assert channel.get_calls == ["wm_env", "wm_env"]
    assert channel.get_batch_calls == []
    assert metrics["actor/received_shards"] == 2.0
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


def test_single_shard_collate_reuses_hidden_storage() -> None:
    shard = _shard(0.0, 1.0)
    source_hidden = shard.forward_inputs["hidden"]

    batch = collate_trajectory_shards([shard])

    assert (
        batch.forward_inputs["hidden"].untyped_storage().data_ptr()
        == source_hidden.untyped_storage().data_ptr()
    )


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


def test_actor_run_training_backprops_each_step_before_next_forward() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    probe = _OutstandingGraphProbe()
    policy = _GraphProbePolicy(probe)
    actor.policy = policy
    actor.optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)

    actor.load_trajectory_shards(
        [
            _variable_length_shard(steps=3, slot_id=0, reward=0.0),
            _variable_length_shard(steps=3, slot_id=1, reward=1.0),
        ]
    )
    actor.compute_advantages_and_returns()

    metrics = actor.run_training()

    assert metrics["actor/ppo_updates"] == 1.0
    assert probe.max_active == 1
    assert probe.backward_calls == 3
    assert probe.active == 0


def test_actor_run_training_applies_behavior_kl_anchor_when_advantages_are_zero() -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["lr"] = 0.1
    cfg["train_cfg"]["algorithm_cfg"]["kl_coef"] = 1.0
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    probe = _OutstandingGraphProbe()
    policy = _GraphProbePolicy(probe)
    with torch.no_grad():
        policy.logprob.fill_(1.0)
    actor.policy = policy
    actor.optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    actor.load_trajectory_shards([_shard(0.0, 0.0)])
    actor.compute_advantages_and_returns()
    before = float(policy.logprob.detach().cpu())
    metrics = actor.run_training()

    assert float(policy.logprob.detach().cpu()) < before
    assert metrics["actor/behavior_kl_mean"] > 0.0
    assert metrics["actor/kl_coef"] == 1.0


def test_actor_run_training_backprops_zero_loss_for_global_padded_steps(
    monkeypatch,
) -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    probe = _OutstandingGraphProbe()
    policy = _GraphProbePolicy(probe)
    actor.policy = policy
    actor.optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)

    actor.load_trajectory_shards(
        [
            _variable_length_shard(steps=1, slot_id=0, reward=0.0),
            _variable_length_shard(steps=1, slot_id=1, reward=1.0),
        ]
    )
    actor.compute_advantages_and_returns()
    monkeypatch.setattr(
        embodied_fsdp_actor,
        "_distributed_max_int",
        lambda value, device: 3,
    )
    monkeypatch.setattr(
        embodied_fsdp_actor,
        "_distributed_sum_int",
        lambda value, device: int(value),
    )

    metrics = actor.run_training()

    assert metrics["actor/ppo_updates"] == 1.0
    assert metrics["actor/global_time_steps"] == 3.0
    assert metrics["actor/zero_loss_steps"] == 2.0
    assert probe.backward_calls == 3
    assert probe.active == 0


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


def test_actor_training_uses_rlinf_global_and_micro_batches(capsys) -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"].update(
        {
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "seed": 7,
            "progress_every_s": 0.0,
        }
    )
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    actor.set_global_step(3)
    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    actor.compute_advantages_and_returns()

    metrics = actor.run_training()
    output = capsys.readouterr().out

    assert metrics["actor/ppo_optimizer_steps"] == 2.0
    assert metrics["actor/ppo_updates"] == 2.0
    assert metrics["actor/ppo_forward_backward_steps"] == 4.0
    assert metrics["actor/ppo_progress_ops"] == 6.0
    assert metrics["actor/lr"] == pytest.approx(1.0e-3)
    assert 0.0 <= metrics["actor/clip_fraction"] <= 1.0
    assert torch.isfinite(torch.tensor(metrics["actor/approx_kl"]))
    assert "ppo/00000003" in output
    assert "6/6" in output
    assert "optimizer=2/2" in output


def test_actor_microbatch_matches_full_batch_update() -> None:
    # Two shards -> 4 rollouts * 2 chunks = 8 flattened samples. Both routes
    # use one global batch / optimizer step; only gradient accumulation differs.
    def _train(micro_batch_size: int) -> dict:
        cfg = _actor_cfg()
        cfg["train_cfg"]["global_batch_size"] = 8
        cfg["train_cfg"]["micro_batch_size"] = micro_batch_size
        torch.manual_seed(1234)
        actor = EmbodiedFSDPActor(**cfg)
        actor.init()
        actor.load_trajectory_shards([_shard(0.0, 1.0), _shard(2.0, 3.0)])
        actor.compute_advantages_and_returns()
        actor.run_training()
        return actor.state_dict()

    full = _train(8)
    micro = _train(2)

    assert set(full) == set(micro)
    for key in full:
        assert torch.allclose(full[key], micro[key], atol=1e-5), key


def test_actor_microbatch_slices_hidden_before_device_transfer(monkeypatch) -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["global_batch_size"] = 8
    cfg["train_cfg"]["micro_batch_size"] = 2
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    actor.load_trajectory_shards([_shard(0.0, 1.0), _shard(2.0, 3.0)])
    actor.compute_advantages_and_returns()
    assert actor.batch is not None
    transferred_batch_sizes: list[int] = []
    original_to = torch.Tensor.to

    def tracked_to(tensor, *args, **kwargs):
        if (
            tensor.ndim == 2
            and int(tensor.shape[-1]) == 4
            and tensor.device.type == "cpu"
        ):
            transferred_batch_sizes.append(int(tensor.shape[0]))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)

    actor.run_training()

    assert len(transferred_batch_sizes) >= 4
    assert set(transferred_batch_sizes) == {2}


def test_actor_hidden_transfer_uses_configured_fsdp_precision() -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["fsdp"]["precision"] = "bf16"
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    assert actor.batch is not None

    eval_batch = actor._eval_inputs_for_step(
        actor.batch,
        0,
        batch_slice=slice(0, 1),
    )

    assert eval_batch["hidden"].dtype == torch.bfloat16


def test_actor_fsdp_no_syncs_all_but_final_accumulated_backward(
    monkeypatch,
) -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["global_batch_size"] = 8
    cfg["train_cfg"]["micro_batch_size"] = 2
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    policy = _NoSyncProbePolicy()
    actor.policy = policy
    actor.optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    actor.load_trajectory_shards([_shard(0.0, 1.0), _shard(2.0, 3.0)])
    actor.compute_advantages_and_returns()
    monkeypatch.setattr(embodied_fsdp_actor, "_is_fsdp_module", lambda _policy: True)

    actor.run_training()

    assert policy.no_sync_calls == 3
    assert policy.forward_sync_states == [False, False, False, True]


def test_actor_training_avoids_per_microbatch_scalar_device_sync(monkeypatch) -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["global_batch_size"] = 8
    cfg["train_cfg"]["micro_batch_size"] = 2
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    actor.load_trajectory_shards([_shard(0.0, 1.0), _shard(2.0, 3.0)])
    actor.compute_advantages_and_returns()
    item_calls = 0
    original_item = torch.Tensor.item

    def tracked_item(tensor, *args, **kwargs):
        nonlocal item_calls
        item_calls += 1
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", tracked_item)

    actor.run_training()

    assert item_calls <= 8


def test_actor_uses_fsdp_global_grad_norm_clipping(monkeypatch) -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    policy = _FSDPGradNormProbePolicy()
    actor.policy = policy
    monkeypatch.setattr(embodied_fsdp_actor, "_is_fsdp_module", lambda _policy: True)

    grad_norm = actor._clip_or_measure_grad_norm({"grad_clip_norm": 1.25})

    assert grad_norm == 3.5
    assert policy.clip_max_norms == [1.25]


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


def test_sync_model_to_rollout_casts_snapshot_to_configured_bfloat16() -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["syncer"] = {"precision": "bf16"}
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    pushed: dict[str, torch.Tensor] = {}

    class _FakeSyncer:
        last_push_metrics: dict[str, float] = {}

        def push(self, key: str, state_dict: dict, version: int) -> None:
            del key, version
            pushed.update(state_dict)

    actor.syncer = _FakeSyncer()  # type: ignore[assignment]

    metrics = actor.sync_model_to_rollout("policy", version=3)

    floating_dtypes = {
        value.dtype for value in pushed.values() if value.is_floating_point()
    }
    expected_bytes = sum(
        value.numel() * value.element_size() for value in pushed.values()
    )
    assert floating_dtypes == {torch.bfloat16}
    assert metrics["sync/policy_bytes"] == float(expected_bytes)


def _filter_rewards_cfg() -> dict:
    cfg = _actor_cfg()
    ac = cfg["train_cfg"]["algorithm_cfg"]
    ac["filter_rewards"] = True
    ac["reward_coef"] = 1.0
    ac["rewards_lower_bound"] = 0.5
    ac["rewards_upper_bound"] = 4.5
    return cfg


def test_actor_filters_out_of_bound_reward_groups_when_enabled() -> None:
    actor = EmbodiedFSDPActor(**_filter_rewards_cfg())
    actor.init()

    # group_size=2: both rollouts fail -> group mean 0.0 < lower bound -> filtered.
    actor.load_trajectory_shards([_shard(0.0, 0.0)])
    metrics = actor.compute_advantages_and_returns()

    assert metrics["actor/reward_filtered_rollouts"] == 2.0
    assert actor.advantages is not None


def test_actor_skips_training_when_reward_filter_removes_every_group() -> None:
    actor = EmbodiedFSDPActor(**_filter_rewards_cfg())
    actor.init()

    actor.load_trajectory_shards([_shard(0.0, 0.0)])
    advantage_metrics = actor.compute_advantages_and_returns()
    train_metrics = actor.run_training()

    assert advantage_metrics["actor/loss_mask_sum"] == 4.0
    assert advantage_metrics["actor/reward_filtered_rollouts"] == 2.0
    assert train_metrics["actor/global_loss_mask_sum"] == 0.0
    assert train_metrics["actor/ppo_updates"] == 0.0
    assert train_metrics["actor/skipped_zero_valid_update"] == 1.0


def test_actor_keeps_in_bound_reward_groups_when_filter_enabled() -> None:
    actor = EmbodiedFSDPActor(**_filter_rewards_cfg())
    actor.init()

    # returns [0, 2] -> group mean 1.0 in [0.5, 4.5] -> kept.
    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    metrics = actor.compute_advantages_and_returns()

    assert metrics["actor/reward_filtered_rollouts"] == 0.0
    assert metrics["actor/advantage_std"] > 0.0


def test_actor_per_rollout_normalization_flag_reported() -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["algorithm_cfg"]["loss_normalization"] = "per_rollout"
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()

    actor.load_trajectory_shards(
        [
            _variable_length_shard(steps=1, slot_id=0, reward=0.0),
            _variable_length_shard(steps=3, slot_id=1, reward=1.0),
        ]
    )
    actor.compute_advantages_and_returns()
    train_metrics = actor.run_training()

    assert train_metrics["actor/loss_normalization_per_rollout"] == 1.0
    assert train_metrics["actor/ppo_updates"] == 1.0


def test_actor_default_normalization_flag_off() -> None:
    actor = EmbodiedFSDPActor(**_actor_cfg())
    actor.init()
    actor.load_trajectory_shards([_shard(0.0, 1.0)])
    actor.compute_advantages_and_returns()
    train_metrics = actor.run_training()

    assert train_metrics["actor/loss_normalization_per_rollout"] == 0.0


def test_actor_token_level_logprobs_are_not_collapsed_to_chunk_scalars() -> None:
    cfg = _actor_cfg()
    cfg["train_cfg"]["algorithm_cfg"].update(
        {
            "logprob_type": "token_level",
            "loss_agg_func": "token-mean",
            "loss_normalization": "global_valid_count",
            "clip_log_ratio": None,
        }
    )
    actor = EmbodiedFSDPActor(**cfg)
    actor.init()
    policy = _TokenLevelPolicy()
    actor.policy = policy
    actor.optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
    actions = torch.zeros(1, 2, 2, 3)
    shard = TrajectoryShard(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_ids=[0, 1],
        actions=actions,
        rewards=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        dones=torch.zeros(1, 2, dtype=torch.bool),
        prev_logprobs=torch.zeros(1, 2, 2, 3),
        prev_values=None,
        forward_inputs={
            "hidden": torch.ones(1, 2, 4),
            "action": actions.clone(),
        },
        versions={"policy": torch.zeros(1, 2, dtype=torch.long)},
    )

    actor.load_trajectory_shards([shard])
    actor.compute_advantages_and_returns()
    train_metrics = actor.run_training()

    assert policy.seen_logprob_types == ["token_level"]
    assert train_metrics["actor/logprob_type_token_level"] == 1.0
    assert train_metrics["actor/global_logprob_token_count"] == 12.0
