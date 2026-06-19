from __future__ import annotations

import numpy as np
import ray

from dreamervla.scheduler.cluster import Cluster
from dreamervla.scheduler.placement import NodePlacementStrategy
from dreamervla.scheduler.worker_group import WorkerGroup


def _cfg() -> dict:
    return {
        "encoder": {"target": "dreamervla.workers.inference._test_models:TinyEncoder"},
        "world_model": {
            "target": "dreamervla.workers.inference._test_models:TinyWorldModel",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        "policy": {
            "target": "dreamervla.workers.inference._test_models:TinyPolicy",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        "device": "cpu",
    }


def _obs(env_id: int, step: int, *, is_first: bool = False) -> dict:
    return {"env_id": env_id, "step": step, "is_first": is_first}


def test_inference_worker_keeps_per_env_state_and_resets() -> None:
    try:
        from dreamervla.workers.inference.inference_worker import InferenceWorker
    except ModuleNotFoundError as exc:
        raise AssertionError("InferenceWorker module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        group = WorkerGroup(InferenceWorker, _cfg(), {}, num_envs=2).launch(
            cluster, NodePlacementStrategy(1)
        )

        first = group.forward_batch([_obs(0, 0, is_first=True)], [0]).wait()[0]
        second = group.forward_batch([_obs(0, 1)], [0]).wait()[0]
        group.reset_states([0]).wait()
        first_after_reset = group.forward_batch([_obs(0, 0, is_first=True)], [0]).wait()[0]

        assert np.allclose(first["actions"][0], first_after_reset["actions"][0])
        assert not np.allclose(first["actions"][0], second["actions"][0])

        group.reset_states([0, 1]).wait()
        alone0_step0 = group.forward_batch([_obs(0, 0, is_first=True)], [0]).wait()[0]
        alone0_step1 = group.forward_batch([_obs(0, 1)], [0]).wait()[0]
        group.reset_states([0, 1]).wait()
        mixed0 = group.forward_batch(
            [_obs(0, 0, is_first=True), _obs(1, 10, is_first=True)], [0, 1]
        ).wait()[0]
        mixed1 = group.forward_batch([_obs(0, 1), _obs(1, 11)], [0, 1]).wait()[0]

        assert np.allclose(alone0_step0["actions"][0], mixed0["actions"][0])
        assert np.allclose(alone0_step1["actions"][0], mixed1["actions"][0])
        assert not np.allclose(mixed1["actions"][0], mixed1["actions"][1])
    finally:
        cluster.shutdown()


def test_inference_worker_update_weights_changes_policy_output() -> None:
    try:
        from dreamervla.workers.inference.inference_worker import InferenceWorker
    except ModuleNotFoundError as exc:
        raise AssertionError("InferenceWorker module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        group = WorkerGroup(InferenceWorker, _cfg(), {}, num_envs=1).launch(
            cluster, NodePlacementStrategy(1)
        )
        before = group.forward_batch([_obs(0, 0, is_first=True)], [0]).wait()[0]["actions"][0]
        state = group.state_dicts().wait()[0]
        state["policy"]["bias"] += 3.0
        group.update_weights(policy_sd=state["policy"]).wait()
        group.reset_states([0]).wait()
        after = group.forward_batch([_obs(0, 0, is_first=True)], [0]).wait()[0]["actions"][0]

        assert np.allclose(after - before, np.full_like(before, 3.0))
    finally:
        cluster.shutdown()


def test_inference_worker_handles_dict_latent_state() -> None:
    try:
        from dreamervla.workers.inference.inference_worker import InferenceWorker
    except ModuleNotFoundError as exc:
        raise AssertionError("InferenceWorker module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        cfg = _cfg()
        cfg["world_model"] = {
            "target": "dreamervla.workers.inference._test_models:TinyDictWorldModel",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        }
        group = WorkerGroup(InferenceWorker, cfg, {}, num_envs=2).launch(
            cluster, NodePlacementStrategy(1)
        )

        first = group.forward_batch([_obs(0, 0, is_first=True), _obs(1, 10, is_first=True)], [0, 1]).wait()[0]
        second = group.forward_batch([_obs(0, 1), _obs(1, 11)], [0, 1]).wait()[0]
        group.reset_states([0]).wait()
        reset = group.forward_batch([_obs(0, 0, is_first=True)], [0]).wait()[0]

        assert len(first["actions"]) == 2
        assert len(second["actions"]) == 2
        assert not np.allclose(first["actions"][0], second["actions"][0])
        assert np.allclose(first["actions"][0], reset["actions"][0])
    finally:
        cluster.shutdown()


def test_encode_batch_uses_single_obs_encoder_fallback() -> None:
    from dreamervla.workers.inference.inference_worker import _encode_batch

    class _SingleObsEncoder:
        def encode(self, obs: dict) -> np.ndarray:
            return np.asarray(
                [
                    float(obs["step"]),
                    float(obs["env_id"]),
                    float(bool(obs.get("is_first", False))),
                    1.0,
                ],
                dtype=np.float32,
            )

    encoded = _encode_batch(
        _SingleObsEncoder(),
        [_obs(0, 3, is_first=True), _obs(1, 4, is_first=False)],
    )

    assert encoded.shape == (2, 4)
    assert np.allclose(encoded.numpy(), [[3.0, 0.0, 1.0, 1.0], [4.0, 1.0, 0.0, 1.0]])


def test_inference_worker_reports_stage_timing() -> None:
    from dreamervla.workers.inference.inference_worker import InferenceWorker

    worker = InferenceWorker(_cfg(), {}, num_envs=1)
    worker.init()

    out = worker.forward_batch([_obs(0, 0, is_first=True)], [0])

    assert "timing" in out
    for key in ("encode_s", "world_model_s", "policy_s"):
        assert key in out["timing"]
        assert out["timing"][key] >= 0.0
