from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dreamervla.envs.world_model.latent_world_model_env import LatentWorldModelEnv
from dreamervla.runners.online_replay import OnlineReplay
from dreamervla.workers.env.trajectory_env_worker import WMEnvWorker


class _IdentityWM(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"latent": batch["latent"]}


class _ReplayWithInitialEmbeddings:
    def __init__(self, latents: np.ndarray) -> None:
        self.latents = np.asarray(latents, dtype=np.float32)

    def size(self) -> int:
        return int(self.latents.shape[0])

    def sample_initial_obs_embeddings(
        self,
        batch_size: int,
        *,
        task_id: int | None = None,
        key: str = "obs_embedding",
    ) -> np.ndarray:
        del task_id, key
        values = [
            self.latents[index % self.latents.shape[0]]
            for index in range(int(batch_size))
        ]
        return np.stack(values, axis=0)


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
