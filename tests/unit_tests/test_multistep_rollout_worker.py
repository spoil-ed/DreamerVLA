from __future__ import annotations

import uuid
import warnings

import numpy as np
import pytest
import ray
import torch

import dreamervla.workers.rollout.multistep_rollout_worker as rollout_worker
from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.cluster import Cluster
from dreamervla.workers.cotrain.messages import (
    ObservationBatchMsg,
    ObservationMsg,
    RolloutResultBatchMsg,
    RolloutResultMsg,
    StopMsg,
    rollout_result_batch_to_messages,
)
from dreamervla.workers.inference.oft_rollout import _select_image_keys_for_policy
from dreamervla.workers.rollout.multistep_rollout_worker import (
    MultiStepRolloutWorker,
    _obs_embedding_from_obs,
    _to_cpu_tensor,
    _to_device_float_tensor,
)


def _policy_cfg() -> dict:
    return {
        "target": "dreamervla.workers.actor._test_models:TinyLumosPolicy",
        "kwargs": {"hidden_dim": 4, "action_dim": 3, "chunk_size": 2},
    }


def _counting_policy_cfg() -> dict:
    return {
        "target": "dreamervla.workers.actor._test_models:CountingTinyLumosPolicy",
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


def _obs_batch(*observations: ObservationMsg, env_rank: int = 0) -> ObservationBatchMsg:
    return ObservationBatchMsg(env_rank=int(env_rank), observations=list(observations))


class _InspectLogprobTypePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_logprob_types: list[object] = []

    def forward(self, batch):  # type: ignore[override]
        logprob_type = batch.get("logprob_type")
        self.seen_logprob_types.append(logprob_type)
        if logprob_type != "token_level":
            raise AssertionError(f"expected token_level logprob_type, got {logprob_type!r}")
        bsz = int(batch["hidden"].shape[0])
        action = torch.zeros(bsz, 2, 3)
        log_prob = torch.zeros(bsz, 2, 3)
        return action, log_prob, {"action_chunk": action}


class _InspectTokenGridPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_shapes: list[tuple[int, ...]] = []

    def forward(self, batch):  # type: ignore[override]
        hidden = batch["hidden"]
        self.hidden_shapes.append(tuple(int(dim) for dim in hidden.shape))
        batch_size = int(hidden.shape[0])
        action = torch.zeros(batch_size, 2, 3)
        log_prob = torch.zeros(batch_size, 1)
        return action, log_prob, {"action_chunk": action}


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
    assert out.versions["actor_policy_version"] == 0
    assert out.versions["rollout_policy_version"] == 0
    assert out.versions["global_step"] == 0


def test_generate_once_forwards_token_level_logprob_type() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu", "logprob_type": "token_level"},
    )
    worker.init()
    policy = _InspectLogprobTypePolicy()
    worker.policy = policy

    out = worker.generate_once(_obs())

    assert policy.seen_logprob_types == ["token_level"]
    assert out.prev_logprobs.shape == (2, 3)


def test_to_device_float_tensor_avoids_unconditional_numpy_copy(monkeypatch) -> None:
    array = np.ones((2, 4), dtype=np.float32)

    monkeypatch.setattr(
        rollout_worker.np,
        "array",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("numpy observation conversion should avoid np.array copy")
        ),
    )

    tensor = _to_device_float_tensor(array, torch.device("cpu"))

    assert tensor.shape == (2, 4)
    assert tensor.dtype == torch.float32


def test_to_device_float_tensor_copies_readonly_numpy_without_warning() -> None:
    array = np.ones((2, 4), dtype=np.float32)
    array.setflags(write=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensor = _to_device_float_tensor(array, torch.device("cpu"))

    assert tensor.shape == (2, 4)
    assert tensor.dtype == torch.float32
    assert not caught


def test_to_device_float_tensor_preserves_existing_tensor_dtype() -> None:
    tensor = torch.ones((2, 4), dtype=torch.bfloat16)

    out = _to_device_float_tensor(tensor, torch.device("cpu"))

    assert out.shape == (2, 4)
    assert out.dtype == torch.bfloat16


def test_generate_batch_batches_direct_hidden_conversion(monkeypatch) -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )
    worker.init()
    observations = [_obs(slot_id=0), _obs(slot_id=1)]

    monkeypatch.setattr(
        rollout_worker,
        "_to_device_float_tensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("batched rollout hidden conversion should not be per-row")
        ),
    )

    results = worker.generate_batch(observations)

    assert [result.slot_id for result in results] == [0, 1]
    assert all(result.forward_inputs["hidden"].shape == (1, 4) for result in results)


def test_generate_once_propagates_component_versions_from_env_message() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu", "policy_version": 5},
    )
    worker.init()
    worker.set_global_step(17)
    obs = _obs()
    obs.versions.update(
        {
            "global_step": 13,
            "world_model_version": 7,
            "wm_version": 7,
            "classifier_version": 9,
            "reward_or_classifier_version": 9,
        }
    )

    out = worker.generate_once(obs)

    assert out.versions["policy"] == 5
    assert out.versions["actor_policy_version"] == 5
    assert out.versions["rollout_policy_version"] == 5
    assert out.versions["global_step"] == 13
    assert out.versions["world_model_version"] == 7
    assert out.versions["wm_version"] == 7
    assert out.versions["classifier_version"] == 9
    assert out.versions["reward_or_classifier_version"] == 9


def test_generate_batch_uses_one_policy_forward_for_multiple_observations() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_counting_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )
    worker.init()

    results = worker.generate_batch(
        [
            _obs(step=0, slot_id=0),
            _obs(step=10, slot_id=1),
        ]
    )

    assert len(results) == 2
    assert int(worker._policy().forward_calls.item()) == 1
    assert [result.slot_id for result in results] == [0, 1]
    assert [result.step for result in results] == [0, 10]
    assert all(result.actions.shape == (2, 3) for result in results)
    assert all(result.forward_inputs["hidden"].shape == (1, 4) for result in results)
    assert all(result.forward_inputs["action"].shape == (1, 2, 3) for result in results)


def test_generate_batch_uses_batched_obs_hidden_payload() -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )
    worker.init()
    observations = [
        ObservationMsg(
            env_rank=0,
            slot_id=0,
            task_id=0,
            episode_id=10,
            step=2,
            obs={"task_description": "task 0"},
            versions={"policy": 0},
        ),
        ObservationMsg(
            env_rank=0,
            slot_id=1,
            task_id=0,
            episode_id=11,
            step=3,
            obs={"task_description": "task 0"},
            versions={"policy": 0},
        ),
    ]

    results = worker.generate_batch(
        observations,
        batched_obs={
            "latent": np.stack(
                [
                    np.ones(4, dtype=np.float32),
                    np.full(4, 2.0, dtype=np.float32),
                ],
                axis=0,
            ),
            "lang_emb": np.stack(
                [
                    np.full(2, 3.0, dtype=np.float32),
                    np.full(2, 4.0, dtype=np.float32),
                ],
                axis=0,
            ),
        },
    )

    assert [result.slot_id for result in results] == [0, 1]
    assert [result.step for result in results] == [2, 3]
    # The env already owns the hidden it sent in batched_obs; echoing it back
    # would only duplicate the largest tensor on the rollout->env channel.
    assert all("hidden" not in result.forward_inputs for result in results)
    assert results[0].forward_inputs["lang_emb"].tolist() == [3.0, 3.0]
    assert results[1].forward_inputs["lang_emb"].tolist() == [4.0, 4.0]


@pytest.mark.parametrize("use_batched_obs", [False, True])
def test_generate_batch_preserves_token_grid_for_policy(
    use_batched_obs: bool,
) -> None:
    worker = MultiStepRolloutWorker(
        policy_cfg=_policy_cfg(),
        encoder_cfg=None,
        init_ckpt={},
        train_cfg={"device": "cpu"},
    )
    worker.init()
    policy = _InspectTokenGridPolicy()
    worker.policy = policy
    grids = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    observations = [
        ObservationMsg(
            env_rank=0,
            slot_id=index,
            task_id=0,
            episode_id=index,
            step=0,
            obs=(
                {"task_description": "task 0"}
                if use_batched_obs
                else {"obs_embedding": grids[index]}
            ),
            versions={"policy": 0},
        )
        for index in range(2)
    ]

    worker.generate_result_batch(
        observations,
        batched_obs={"obs_embedding": grids} if use_batched_obs else None,
    )

    assert policy.hidden_shapes == [(2, 2, 3)]


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


def test_oft_rollout_rejects_non_mainline_image_counts() -> None:
    keys = ["agentview_rgb"]

    assert _select_image_keys_for_policy(keys, 1) == ["agentview_rgb"]
    with pytest.raises(ValueError, match="requires num_images_in_input=1"):
        _select_image_keys_for_policy(keys, 2)
    with pytest.raises(ValueError, match="requires num_images_in_input=1"):
        _select_image_keys_for_policy(keys, 3)
    with pytest.raises(ValueError, match="exactly one image key"):
        _select_image_keys_for_policy(
            ["agentview_rgb", "eye_in_hand_rgb"],
            1,
        )


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


def test_to_cpu_tensor_copies_readonly_numpy_without_warning() -> None:
    value = np.ones((2,), dtype=np.float32)
    value.setflags(write=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensor = _to_cpu_tensor(value)

    assert tensor.tolist() == [1.0, 1.0]
    assert caught == []


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

        sync_metrics = worker.sync_model_from_actor("policy", local_version=0)
        assert sync_metrics["sync/rollout_policy_version"] == 1.0
        assert sync_metrics["sync/rollout_policy_updated"] == 1.0
        assert sync_metrics["sync/rollout_policy_pull_s"] >= 0.0
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

        assert stats["rollout/generated"] == 2.0
        assert stats["rollout/channel_get_s"] >= 0.0
        assert stats["rollout/policy_forward_s"] >= 0.0
        assert stats["rollout/channel_put_s"] >= 0.0
        assert stats["rollout/loop_s"] >= 0.0
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


def test_generate_reads_rank_keyed_observation_batches_when_num_slots_is_set(monkeypatch) -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        traces: list[str] = []
        monkeypatch.setattr(
            "dreamervla.workers.rollout.multistep_rollout_worker._hs_trace",
            traces.append,
        )
        input_name = f"test-rollout-keyed-in-{uuid.uuid4().hex}"
        output_name = f"test-rollout-keyed-out-{uuid.uuid4().hex}"
        input_channel = Channel.create(input_name)
        output_channel = Channel.create(output_name)
        input_channel.put(
            _obs_batch(_obs(step=0, slot_id=0), _obs(step=10, slot_id=1)),
            key="0",
        )
        input_channel.put(StopMsg(reason="unit-test"), key="0")

        worker = MultiStepRolloutWorker(
            policy_cfg=_policy_cfg(),
            encoder_cfg=None,
            init_ckpt={},
            train_cfg={"device": "cpu"},
        )
        worker.init()

        stats = worker.generate(input_name, output_name, num_slots=2)

        assert stats["rollout/generated"] == 2.0
        assert stats["rollout/channel_get_s"] >= 0.0
        assert stats["rollout/policy_forward_s"] >= 0.0
        assert stats["rollout/channel_put_s"] >= 0.0
        assert stats["rollout/loop_s"] >= 0.0
        assert output_channel.qsize(key="0") == 1
        batch = output_channel.get(key="0")
        assert isinstance(batch, RolloutResultBatchMsg)
        assert batch.results == []
        first, second = rollout_result_batch_to_messages(batch)
        assert first.step == 0
        assert second.step == 10
        assert any(
            "[rollout rank=0] recv action request batch_size=2" in line
            for line in traces
        )
        assert any(
            "[rollout rank=0] send action response batch_size=2" in line
            for line in traces
        )
        assert any("[rollout rank=0] recv StopMsg key=0" in line for line in traces)
        assert any("[rollout rank=0] generate exit generated=2" in line for line in traces)
    finally:
        cluster.shutdown()


def test_generate_reads_rank_keyed_batch_hidden_payload() -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        input_name = f"test-rollout-batched-hidden-in-{uuid.uuid4().hex}"
        output_name = f"test-rollout-batched-hidden-out-{uuid.uuid4().hex}"
        input_channel = Channel.create(input_name)
        output_channel = Channel.create(output_name)
        input_channel.put(
            ObservationBatchMsg(
                env_rank=0,
                observations=[
                    ObservationMsg(
                        env_rank=0,
                        slot_id=0,
                        task_id=0,
                        episode_id=10,
                        step=2,
                        obs={"task_description": "task 0"},
                        versions={"policy": 0},
                    ),
                    ObservationMsg(
                        env_rank=0,
                        slot_id=1,
                        task_id=0,
                        episode_id=11,
                        step=3,
                        obs={"task_description": "task 0"},
                        versions={"policy": 0},
                    ),
                ],
                batched_obs={
                    "latent": np.stack(
                        [
                            np.ones(4, dtype=np.float32),
                            np.full(4, 2.0, dtype=np.float32),
                        ],
                        axis=0,
                    ),
                    "lang_emb": np.stack(
                        [
                            np.full(2, 3.0, dtype=np.float32),
                            np.full(2, 4.0, dtype=np.float32),
                        ],
                        axis=0,
                    ),
                },
            ),
            key="0",
        )
        input_channel.put(StopMsg(reason="unit-test"), key="0")

        worker = MultiStepRolloutWorker(
            policy_cfg=_policy_cfg(),
            encoder_cfg=None,
            init_ckpt={},
            train_cfg={"device": "cpu"},
        )
        worker.init()

        stats = worker.generate(input_name, output_name, num_slots=2)

        assert stats["rollout/generated"] == 2.0
        batch = output_channel.get(key="0")
        assert isinstance(batch, RolloutResultBatchMsg)
        assert batch.results == []
        assert batch.slot_ids == [0, 1]
        assert batch.task_ids == [0, 0]
        assert batch.episode_ids == [10, 11]
        assert batch.steps == [2, 3]
        assert torch.as_tensor(batch.actions).shape == (2, 2, 3)
        assert torch.as_tensor(batch.prev_logprobs).shape == (2, 1)
        assert batch.prev_values is None
        # Obs-provided hidden is not echoed back; the env worker re-attaches
        # its own copy when building trajectory shards.
        assert "hidden" not in batch.forward_inputs
        assert torch.as_tensor(batch.forward_inputs["lang_emb"]).tolist() == [
            [3.0, 3.0],
            [4.0, 4.0],
        ]
        assert torch.as_tensor(batch.forward_inputs["action"]).shape == (2, 2, 3)
        assert torch.as_tensor(batch.versions["policy"]).tolist() == [0, 0]
    finally:
        cluster.shutdown()


def test_generate_rank_keyed_batch_sends_direct_batched_payload(monkeypatch) -> None:
    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        input_name = f"test-rollout-direct-batch-in-{uuid.uuid4().hex}"
        output_name = f"test-rollout-direct-batch-out-{uuid.uuid4().hex}"
        input_channel = Channel.create(input_name)
        output_channel = Channel.create(output_name)
        input_channel.put(
            ObservationBatchMsg(
                env_rank=0,
                observations=[_obs(step=0, slot_id=0), _obs(step=1, slot_id=1)],
            ),
            key="0",
        )
        input_channel.put(StopMsg(reason="unit-test"), key="0")

        worker = MultiStepRolloutWorker(
            policy_cfg=_policy_cfg(),
            encoder_cfg=None,
            init_ckpt={},
            train_cfg={"device": "cpu"},
        )
        worker.init()

        worker.generate(input_name, output_name, num_slots=2)

        batch = output_channel.get(key="0")
        assert isinstance(batch, RolloutResultBatchMsg)
        assert batch.results == []
        assert batch.slot_ids == [0, 1]
        assert torch.as_tensor(batch.actions).shape == (2, 2, 3)
    finally:
        cluster.shutdown()


def test_generate_wraps_rank_slot_failures_with_channel_key() -> None:
    class FailingRolloutWorker(MultiStepRolloutWorker):
        def generate_result_batch(
            self,
            obs_msgs: list[ObservationMsg],
            *,
            batched_obs: dict[str, object] | None = None,
        ) -> RolloutResultBatchMsg:
            del obs_msgs
            del batched_obs
            raise ValueError("encoder failed")

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()
    try:
        input_name = f"test-rollout-keyed-fail-in-{uuid.uuid4().hex}"
        output_name = f"test-rollout-keyed-fail-out-{uuid.uuid4().hex}"
        input_channel = Channel.create(input_name)
        Channel.create(output_name)
        input_channel.put(_obs_batch(_obs(step=0, slot_id=0)), key="0")

        worker = FailingRolloutWorker(
            policy_cfg=_policy_cfg(),
            encoder_cfg=None,
            init_ckpt={},
            train_cfg={"device": "cpu"},
        )
        worker.init()

        with pytest.raises(
            RuntimeError,
            match=r"generate_result_batch failed rank=0.*keys=0:0.*encoder failed",
        ):
            worker.generate(input_name, output_name, num_slots=1)
    finally:
        cluster.shutdown()
