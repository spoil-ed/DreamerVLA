from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dreamervla.envs.world_model.latent_world_model_env import LatentWorldModelEnv
from dreamervla.runtime.online_replay import OnlineReplay
from dreamervla.workers.env.trajectory_env_worker import WMEnvWorker


class _IdentityWM(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"latent": batch["latent"]}


class _ReplayWithInitialEmbeddings:
    def __init__(
        self,
        latents: np.ndarray,
        lang_embs: np.ndarray | None = None,
        proprios: np.ndarray | None = None,
    ) -> None:
        self.latents = np.asarray(latents, dtype=np.float32)
        self.lang_embs = None if lang_embs is None else np.asarray(lang_embs, dtype=np.float32)
        self.proprios = None if proprios is None else np.asarray(proprios, dtype=np.float32)

    def size(self) -> int:
        return int(self.latents.shape[0])

    def sample_initial_obs_embeddings(
        self,
        batch_size: int,
        *,
        task_id: int | None = None,
        key: str = "obs_embedding",
    ) -> np.ndarray:
        del task_id
        source = self.latents
        if key == "lang_emb":
            if self.lang_embs is None:
                raise KeyError("lang_emb")
            source = self.lang_embs
        elif key == "proprio":
            if self.proprios is None:
                raise KeyError("proprio")
            source = self.proprios
        elif key != "obs_embedding":
            raise KeyError(key)
        values = [
            source[index % source.shape[0]]
            for index in range(int(batch_size))
        ]
        return np.stack(values, axis=0)


class _ReplayWithInitialConditions(_ReplayWithInitialEmbeddings):
    def sample_initial_conditions(self, batch_size: int, *, task_ids=None, keys=()):
        del task_ids
        assert int(batch_size) == 2
        assert tuple(keys) == ("obs_embedding", "lang_emb", "proprio")
        return {
            "task_ids": np.array([3, 7], dtype=np.int64),
            "obs_embedding": self.latents,
            "lang_emb": self.lang_embs,
            "proprio": self.proprios,
        }


class _CyclingGroupedInitialConditions(_ReplayWithInitialEmbeddings):
    def __init__(self) -> None:
        super().__init__(np.array([[0.0, 0.0]], dtype=np.float32))
        self.calls = 0

    def sample_initial_conditions(self, batch_size: int, *, task_ids=None, keys=()):
        del task_ids
        assert int(batch_size) == 1
        assert tuple(keys) == ("obs_embedding", "lang_emb", "proprio")
        value = 3 if self.calls == 0 else 7
        self.calls += 1
        return {
            "task_ids": np.array([value], dtype=np.int64),
            "obs_embedding": np.array([[float(value), float(value)]], dtype=np.float32),
            "lang_emb": np.array([[100.0 + value] * 3], dtype=np.float32),
            "proprio": np.array([[200.0 + value] * 2], dtype=np.float32),
        }


class _SelectorInitialConditions(_CyclingGroupedInitialConditions):
    def __init__(self) -> None:
        super().__init__()
        self.selectors: list[str] = []

    def sample_initial_conditions(
        self,
        batch_size: int,
        *,
        task_ids=None,
        keys=(),
        selector: str = "episode_start",
    ):
        self.selectors.append(str(selector))
        return super().sample_initial_conditions(
            batch_size,
            task_ids=task_ids,
            keys=keys,
        )


def test_online_replay_samples_initial_obs_embeddings() -> None:
    replay = OnlineReplay(capacity=10, sequence_length=1, task_ids=(0,))
    replay.add_episode(
        [
            {
                "task_id": 0,
                "obs_embedding": np.array([1.0, 2.0], dtype=np.float32),
                "action": np.zeros(1, dtype=np.float32),
                "reward": 0.0,
                "done": False,
            }
        ],
        source="coldstart",
    )

    latents = replay.sample_initial_obs_embeddings(3, task_id=0)

    assert latents.shape == (3, 2)
    assert latents.tolist() == [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]


def test_latent_world_model_env_uses_per_slot_initial_latents() -> None:
    env = LatentWorldModelEnv(
        world_model=_IdentityWM(),
        classifier=None,
        latent_dim=2,
        action_dim=1,
        num_envs=2,
    )
    env.set_initial_latents(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )

    obs0, _ = env.reset_slot(0, task_id=0, episode_id=0)
    obs1, _ = env.reset_slot(1, task_id=0, episode_id=1)

    assert obs0["latent"].tolist() == [1.0, 2.0]
    assert obs1["latent"].tolist() == [3.0, 4.0]


def test_wm_env_worker_bootstraps_initial_latents_from_replay() -> None:
    worker = WMEnvWorker(
        env_cfg={
            "target": (
                "dreamervla.envs.world_model.latent_world_model_env:"
                "LatentWorldModelEnv"
            ),
            "kwargs": {
                "world_model": {
                    "target": (
                        "dreamervla.workers.actor._test_models:"
                        "TinyLumosWorldModel"
                    ),
                    "kwargs": {"hidden_dim": 2, "action_dim": 1},
                },
                "classifier": None,
                "latent_dim": 2,
                "action_dim": 1,
                "num_envs": 2,
                "device": "cpu",
            },
        },
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
        replay=_ReplayWithInitialEmbeddings(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        ),
    )
    try:
        worker.init()
        messages = worker.bootstrap_obs()

        assert messages[0].obs["latent"].tolist() == [5.0, 6.0]
        assert messages[1].obs["latent"].tolist() == [7.0, 8.0]
    finally:
        worker.close()


def test_wm_env_worker_bootstraps_aligned_multi_task_conditions() -> None:
    worker = WMEnvWorker(
        env_cfg={
            "target": (
                "dreamervla.envs.world_model.latent_world_model_env:"
                "LatentWorldModelEnv"
            ),
            "kwargs": {
                "world_model": {
                    "target": (
                        "dreamervla.workers.actor._test_models:"
                        "TinyLumosWorldModel"
                    ),
                    "kwargs": {"hidden_dim": 2, "action_dim": 1},
                },
                "classifier": None,
                "latent_dim": 2,
                "action_dim": 1,
                "lang_dim": 3,
                "proprio_dim": 2,
                "num_envs": 2,
                "device": "cpu",
            },
        },
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
        replay=_ReplayWithInitialConditions(
            np.array([[3.0, 3.0], [7.0, 7.0]], dtype=np.float32),
            lang_embs=np.array([[103.0] * 3, [107.0] * 3], dtype=np.float32),
            proprios=np.array([[203.0] * 2, [207.0] * 2], dtype=np.float32),
        ),
    )
    try:
        worker.init()
        messages = worker.bootstrap_obs()

        assert [message.task_id for message in messages] == [3, 7]
        assert [message.obs["latent"][0].item() for message in messages] == [3.0, 7.0]
        assert [message.obs["lang_emb"][0].item() for message in messages] == [
            103.0,
            107.0,
        ]
        assert [message.obs["proprio"][0].item() for message in messages] == [
            203.0,
            207.0,
        ]
    finally:
        worker.close()


def test_wm_env_worker_repeats_one_aligned_condition_for_each_policy_group() -> None:
    replay = _CyclingGroupedInitialConditions()
    worker = WMEnvWorker(
        env_cfg={
            "target": (
                "dreamervla.envs.world_model.latent_world_model_env:"
                "LatentWorldModelEnv"
            ),
            "bootstrap_group_size": 4,
            "defer_initial_condition_bootstrap": True,
            "kwargs": {
                "world_model": {
                    "target": (
                        "dreamervla.workers.actor._test_models:"
                        "TinyLumosWorldModel"
                    ),
                    "kwargs": {"hidden_dim": 2, "action_dim": 1},
                },
                "classifier": None,
                "latent_dim": 2,
                "action_dim": 1,
                "lang_dim": 3,
                "proprio_dim": 2,
                "num_envs": 4,
                "device": "cpu",
            },
        },
        num_slots=4,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
        replay=replay,
    )
    try:
        worker.init()
        assert replay.calls == 0
        worker.refresh_wm_initial_conditions()
        first = worker.bootstrap_obs()

        assert [message.task_id for message in first] == [3, 3, 3, 3]
        assert [message.obs["latent"][0].item() for message in first] == [3.0] * 4
        assert [message.obs["lang_emb"][0].item() for message in first] == [103.0] * 4
        assert [message.obs["proprio"][0].item() for message in first] == [203.0] * 4

        worker.refresh_wm_initial_conditions()
        refreshed = worker.bootstrap_obs()

        assert [message.task_id for message in refreshed] == [7, 7, 7, 7]
        assert [message.obs["latent"][0].item() for message in refreshed] == [7.0] * 4
    finally:
        worker.close()


def test_wm_env_worker_forwards_failure_initial_condition_selector() -> None:
    replay = _SelectorInitialConditions()
    worker = WMEnvWorker(
        env_cfg={
            "target": (
                "dreamervla.envs.world_model.latent_world_model_env:"
                "LatentWorldModelEnv"
            ),
            "bootstrap_group_size": 4,
            "defer_initial_condition_bootstrap": True,
            "initial_condition_selector": "failed_episode_start",
            "kwargs": {
                "world_model": {
                    "target": (
                        "dreamervla.workers.actor._test_models:"
                        "TinyLumosWorldModel"
                    ),
                    "kwargs": {"hidden_dim": 2, "action_dim": 1},
                },
                "classifier": None,
                "latent_dim": 2,
                "action_dim": 1,
                "lang_dim": 3,
                "proprio_dim": 2,
                "num_envs": 4,
                "device": "cpu",
            },
        },
        num_slots=4,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
        replay=replay,
    )
    try:
        worker.init()
        worker.refresh_wm_initial_conditions()
        messages = worker.bootstrap_obs()

        assert replay.selectors == ["failed_episode_start"]
        assert [message.task_id for message in messages] == [3, 3, 3, 3]
    finally:
        worker.close()


def test_wm_env_worker_bootstraps_initial_lang_embs_from_replay() -> None:
    worker = WMEnvWorker(
        env_cfg={
            "target": (
                "dreamervla.envs.world_model.latent_world_model_env:"
                "LatentWorldModelEnv"
            ),
            "kwargs": {
                "world_model": {
                    "target": (
                        "dreamervla.workers.actor._test_models:"
                        "TinyLumosWorldModel"
                    ),
                    "kwargs": {"hidden_dim": 2, "action_dim": 1},
                },
                "classifier": None,
                "latent_dim": 2,
                "action_dim": 1,
                "lang_dim": 3,
                "num_envs": 2,
                "device": "cpu",
            },
        },
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
        replay=_ReplayWithInitialEmbeddings(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            lang_embs=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        ),
    )
    try:
        worker.init()
        messages = worker.bootstrap_obs()

        assert messages[0].obs["lang_emb"].tolist() == [1.0, 2.0, 3.0]
        assert messages[1].obs["lang_emb"].tolist() == [4.0, 5.0, 6.0]
    finally:
        worker.close()


def test_wm_env_worker_bootstraps_initial_proprios_from_replay() -> None:
    worker = WMEnvWorker(
        env_cfg={
            "target": (
                "dreamervla.envs.world_model.latent_world_model_env:"
                "LatentWorldModelEnv"
            ),
            "kwargs": {
                "world_model": {
                    "target": (
                        "dreamervla.workers.actor._test_models:"
                        "TinyLumosWorldModel"
                    ),
                    "kwargs": {"hidden_dim": 2, "action_dim": 1},
                },
                "classifier": None,
                "latent_dim": 2,
                "action_dim": 1,
                "proprio_dim": 2,
                "num_envs": 2,
                "device": "cpu",
            },
        },
        num_slots=2,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=2,
        num_action_chunks=1,
        task_id=0,
        replay=_ReplayWithInitialEmbeddings(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            proprios=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ),
    )
    try:
        worker.init()
        messages = worker.bootstrap_obs()

        assert messages[0].obs["proprio"].tolist() == [1.0, 2.0]
        assert messages[1].obs["proprio"].tolist() == [3.0, 4.0]
    finally:
        worker.close()
