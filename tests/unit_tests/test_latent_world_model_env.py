import numpy as np
import pytest
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


class _ZeroTwoClassClassifier(torch.nn.Module):
    def forward(self, latent):
        return torch.zeros(latent.shape[0], 2, device=latent.device)


class _MalformedDictClassifier(torch.nn.Module):
    def forward(self, latent):
        return {"unknown": torch.zeros(latent.shape[0], 1, device=latent.device)}


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


def test_latent_world_model_env_converts_two_class_logits_to_success_probability():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_ZeroTwoClassClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=0.5,
    )

    env.reset()
    _next_obs, reward, terminated, truncated, info = env.step(
        np.zeros(2, dtype=np.float32)
    )

    assert reward == pytest.approx(0.5)
    assert info["success_score"] == pytest.approx(0.5)
    assert terminated is True
    assert truncated is False


def test_latent_world_model_env_rejects_unknown_classifier_dict_output():
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_MalformedDictClassifier(),
        latent_dim=2,
        action_dim=2,
    )

    env.reset()
    with pytest.raises(ValueError, match="classifier output dict"):
        env.step(np.zeros(2, dtype=np.float32))


def test_latent_world_model_env_batches_independent_slots() -> None:
    env = LatentWorldModelEnv(
        world_model=_TinyWM(),
        classifier=_TinyClassifier(),
        latent_dim=2,
        action_dim=2,
        success_threshold=10.0,
        num_envs=3,
    )

    obs0, _ = env.reset_slot(0, task_id=0, episode_id=0)
    obs1, _ = env.reset_slot(1, task_id=1, episode_id=11)
    obs2, _ = env.reset_slot(2, task_id=2, episode_id=22)

    assert obs0["latent"].shape == (2,)
    assert obs1["episode_id"] == 11
    assert obs2["task_id"] == 2

    next_obs0, reward0, done0, truncated0, info0 = env.step_slot(
        0, np.array([1.0, 0.0], dtype=np.float32)
    )
    next_obs1, reward1, done1, truncated1, info1 = env.step_slot(
        1, np.array([0.0, 2.0], dtype=np.float32)
    )

    assert next_obs0["latent"].tolist() == [1.0, 0.0]
    assert next_obs1["latent"].tolist() == [0.0, 2.0]
    assert reward0 == 1.0
    assert reward1 == 2.0
    assert done0 is False
    assert done1 is False
    assert truncated0 is False
    assert truncated1 is False
    assert info0["slot_id"] == 0
    assert info1["slot_id"] == 1


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
