import numpy as np
import torch

from dreamervla.envs.world_model.latent_world_model_env import LatentWorldModelEnv


class _TinyWM(torch.nn.Module):
    def forward(self, batch):
        latent = batch["latent"]
        action = batch["action"]
        return latent + action[..., : latent.shape[-1]]


class _TinyClassifier(torch.nn.Module):
    def forward(self, latent):
        return latent.sum(dim=-1, keepdim=True)


def test_latent_world_model_env_step_returns_env_tuple():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=7,
        success_threshold=0.5,
    )

    obs, info = env.reset(task_id=1, episode_id=2)
    assert "latent" in obs
    assert info["task_id"] == 1

    next_obs, reward, terminated, truncated, info = env.step(
        np.ones(7, dtype=np.float32)
    )

    assert "latent" in next_obs
    assert reward > 0.0
    assert terminated is True
    assert truncated is False
    assert info["wm_version"] == 0
    assert info["classifier_version"] == 0


def test_latent_world_model_env_loads_independent_versions():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=7,
    )

    env.load_world_model_state({}, version=5)
    env.load_classifier_state({}, version=7)

    assert env.wm_version == 5
    assert env.classifier_version == 7


def test_latent_world_model_env_config_modules_make_replay_transition():
    env = LatentWorldModelEnv(
        world_model={
            "target": "dreamervla.workers.actor._test_models:TinyLumosWorldModel",
            "kwargs": {"hidden_dim": 4, "action_dim": 7},
        },
        classifier={
            "target": "dreamervla.workers.actor._test_models:TinySuccessClassifier",
            "kwargs": {"hidden_dim": 4, "window": 3},
        },
        latent_dim=4,
        action_dim=7,
        image_shape=(4, 4, 3),
        max_episode_steps=3,
    )

    obs, _info = env.reset(task_id=3, episode_id=4)
    action = np.zeros(7, dtype=np.float32)
    _next_obs, reward, terminated, truncated, info = env.step(action)
    transition = env.make_transition(obs, action, reward, terminated, truncated, info)

    assert transition["obs_embedding"].shape == (4,)
    assert transition["wm_action"].shape == (7,)
    assert transition["image"].shape == (4, 4, 3)
    assert transition["task_id"] == 3
    assert transition["episode_id"] == 4
