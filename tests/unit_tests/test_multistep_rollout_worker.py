from __future__ import annotations

import uuid

import numpy as np
import ray
import torch

from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.cluster import Cluster
from dreamervla.workers.cotrain.messages import ObservationMsg, RolloutResultMsg, StopMsg
from dreamervla.workers.rollout.multistep_rollout_worker import (
    MultiStepRolloutWorker,
    _obs_embedding_from_obs,
)


def _policy_cfg() -> dict:
    return {
        "target": "dreamervla.workers.actor._test_models:TinyLumosPolicy",
        "kwargs": {"hidden_dim": 4, "action_dim": 3, "chunk_size": 2},
    }


def _obs(
    step: int = 0,
    *,
    env_rank: int = 0,
    slot_id: int = 0,
) -> ObservationMsg:
    return ObservationMsg(
        env_rank=env_rank,
        slot_id=slot_id,
        task_id=0,
        episode_id=0,
        step=step,
        obs={"obs_embedding": np.ones(4, dtype=np.float32)},
        versions={"policy": 0},
    )


def _real_obs(step: int = 0, *, is_first: bool = False, seed: int = 5) -> ObservationMsg:
    return ObservationMsg(
        env_rank=0,
        slot_id=0,
        task_id=0,
        episode_id=0,
        step=step,
        obs={
            "seed": seed,
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "state": np.ones(2, dtype=np.float32),
            "task_description": "pick up the block",
            "is_first": bool(is_first),
        },
        versions={"policy": 0},
    )


def _encoder_cfg() -> dict:
    return {
        "target": "dreamervla.workers.inference._test_rollout_stub:StubRolloutBundle",
        "kwargs": {"action_dim": 3, "hidden_dim": 4, "emit_lang": True},
    }


def test_generate_once_accepts_obs_embedding_and_returns_forward_inputs() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )
    worker.init()

    out = worker.generate_once(_obs())

    assert isinstance(out, RolloutResultMsg)
    assert out.actions.shape == (2, 3)
    assert out.prev_logprobs.shape == (1,)
    assert out.forward_inputs["hidden"].shape == (1, 4)
    assert out.forward_inputs["action"].shape == (1, 2, 3)
    assert out.versions["policy"] == 0


def test_generate_once_encodes_real_env_observation_without_obs_embedding() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=_encoder_cfg(),
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )
    worker.init()

    out = worker.generate_once(_real_obs(is_first=True, seed=5))

    assert isinstance(out, RolloutResultMsg)
    assert out.actions.shape == (2, 3)
    assert out.forward_inputs["hidden"].shape == (1, 4)
    assert out.forward_inputs["hidden"].tolist() == [[5.0, 5.0, 5.0, 5.0]]
    assert out.forward_inputs["lang_emb"].tolist() == [5.5, 5.5]
    assert out.forward_inputs["action"].shape == (1, 2, 3)
    assert out.versions["policy"] == 0


def test_generate_once_preserves_lang_emb_and_marks_final_bootstrap() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )
    worker.init()
    obs = _obs()
    obs.obs["lang_emb"] = np.full((2,), 3.0, dtype=np.float32)
    obs.obs["_final_bootstrap"] = True

    out = worker.generate_once(obs)

    assert out.forward_inputs["hidden"].shape[0] == 1
    assert out.forward_inputs["lang_emb"].tolist() == [3.0, 3.0]
    assert out.versions["final_bootstrap"] == 1


def test_obs_embedding_from_obs_accepts_wm_latent() -> None:
    hidden = _obs_embedding_from_obs({"latent": np.zeros(4, dtype=np.float32)})

    assert hidden.shape == (4,)


def test_empty_encoder_cfg_is_treated_as_no_encoder() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg={},
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )

    worker.init()

    assert worker.encoder is None


def test_sync_model_from_actor_applies_patch_syncer() -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        worker = MultiStepRolloutWorker(
            policy_cfg=_policy_cfg(),
            encoder_cfg=None,
            init_ckpt={},
            train_cfg={
                "device": "cpu",
                "syncer": {"store_name": f"test-rollout-patch-{uuid.uuid4().hex}"},
            },
        )
        worker.init()
        state = worker.state_dict()
        changed = {key: value + 1.0 for key, value in state.items()}
        worker._syncer().push("policy", changed, 1)

        assert worker.sync_model_from_actor("policy", local_version=0) == 1
        synced = worker.state_dict()
        assert torch.allclose(next(iter(synced.values())), next(iter(changed.values())))
    finally:
        cluster.shutdown()


def test_generate_reads_channel_writes_results_and_stops() -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        input_name = f"test-rollout-in-{uuid.uuid4().hex}"
        output_name = f"test-rollout-out-{uuid.uuid4().hex}"
        input_channel = Channel.create(input_name)
        output_channel = Channel.create(output_name)
        input_channel.put(_obs(step=0))
        input_channel.put(_obs(step=1))
        input_channel.put(StopMsg(reason="unit-test"))

        worker = MultiStepRolloutWorker(
            policy_cfg=_policy_cfg(),
            encoder_cfg=None,
            init_ckpt={},
            train_cfg={"device": "cpu"},
        )
        worker.init()

        stats = worker.generate(input_name, output_name)

        assert stats == {"rollout/generated": 2.0}
        assert output_channel.qsize(key="0:0") == 2
        first = output_channel.get(key="0:0")
        second = output_channel.get(key="0:0")
        assert isinstance(first, RolloutResultMsg)
        assert isinstance(second, RolloutResultMsg)
        assert first.step == 0
        assert second.step == 1
        assert first.actions.shape == (2, 3)
    finally:
        cluster.shutdown()


def test_generate_reads_rank_slot_keyed_channels_when_num_slots_is_set() -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        input_name = f"test-rollout-keyed-in-{uuid.uuid4().hex}"
        output_name = f"test-rollout-keyed-out-{uuid.uuid4().hex}"
        input_channel = Channel.create(input_name)
        output_channel = Channel.create(output_name)
        input_channel.put(_obs(step=0, slot_id=0), key="0:0")
        input_channel.put(_obs(step=10, slot_id=1), key="0:1")
        input_channel.put(StopMsg(reason="unit-test"), key="0:0")
        input_channel.put(StopMsg(reason="unit-test"), key="0:1")

        worker = MultiStepRolloutWorker(
            policy_cfg=_policy_cfg(),
            encoder_cfg=None,
            init_ckpt={},
            train_cfg={"device": "cpu"},
        )
        worker.init()

        stats = worker.generate(input_name, output_name, num_slots=2)

        assert stats == {"rollout/generated": 2.0}
        assert output_channel.qsize(key="0:0") == 1
        assert output_channel.qsize(key="0:1") == 1
        first = output_channel.get(key="0:0")
        second = output_channel.get(key="0:1")
        assert isinstance(first, RolloutResultMsg)
        assert isinstance(second, RolloutResultMsg)
        assert first.step == 0
        assert second.step == 10
    finally:
        cluster.shutdown()
