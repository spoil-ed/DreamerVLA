from types import SimpleNamespace

import numpy as np
import torch

from dreamervla.envs.world_model.base_world_model_env import WorldModelEnvProtocol
from dreamervla.envs.world_model.latent_world_model_env import LatentWorldModelEnv
from dreamervla.utils.frozen_components import state_dict_sha256


class _StubWorldEnv:
    wm_version = 3
    classifier_version = 4

    def reset(self, *, task_id=0, episode_id=0):
        return {"latent": np.zeros(2, dtype=np.float32)}, {"task_id": task_id}

    def step(self, action):
        return (
            {"latent": np.ones(2, dtype=np.float32)},
            1.0,
            True,
            False,
            {"wm_version": self.wm_version, "classifier_version": self.classifier_version},
        )

    def load_world_model_state(self, state_dict, version):
        self.wm_version = int(version)

    def load_classifier_state(self, state_dict, version):
        self.classifier_version = int(version)


def test_world_model_env_protocol_runtime_checkable():
    assert isinstance(_StubWorldEnv(), WorldModelEnvProtocol)


def test_latent_world_model_env_builds_flat_hydra_configs_and_freezes_components():
    env = LatentWorldModelEnv(
        world_model={
            "_target_": (
                "dreamervla.workers.actor._test_models.TinyLumosWorldModel"
            ),
            "hidden_dim": 2,
            "action_dim": 1,
        },
        classifier={
            "_target_": (
                "dreamervla.workers.actor._test_models.TinySuccessClassifier"
            ),
            "hidden_dim": 2,
        },
        latent_dim=2,
        action_dim=1,
        freeze_components=True,
    )

    assert env.world_model.training is False
    assert env.classifier is not None
    assert env.classifier.training is False
    assert all(not parameter.requires_grad for parameter in env.world_model.parameters())
    assert all(not parameter.requires_grad for parameter in env.classifier.parameters())

    before = env.component_state_hashes()
    env.reset_slot(0)
    env.step_slot(0, [0.0])
    after = env.component_state_hashes()

    assert after == before


def test_latent_world_model_env_preserves_world_model_checkpoint_dtype_on_load():
    source = torch.nn.Linear(2, 2).to(torch.bfloat16).state_dict()
    env = LatentWorldModelEnv(
        world_model=torch.nn.Linear(2, 2),
        classifier=None,
        latent_dim=2,
        action_dim=1,
        freeze_components=True,
    )

    env.load_world_model_state(source, version=1)

    assert next(env.world_model.parameters()).dtype == torch.bfloat16
    assert env.component_state_hashes()["world_model"] == state_dict_sha256(source)


def test_latent_world_model_env_preserves_classifier_checkpoint_dtype_on_load():
    source = torch.nn.Linear(2, 1).to(torch.bfloat16).state_dict()
    env = LatentWorldModelEnv(
        world_model=torch.nn.Linear(2, 2),
        classifier=torch.nn.Linear(2, 1),
        latent_dim=2,
        action_dim=1,
        freeze_components=True,
    )

    env.load_classifier_state(source, version=1)

    assert env.classifier is not None
    assert next(env.classifier.parameters()).dtype == torch.bfloat16
    assert env.component_state_hashes()["classifier"] == state_dict_sha256(source)


def test_latent_world_model_env_uses_predict_next_mode_for_wm_step():
    class _PredictNextWM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_batch = None

        def forward(self, batch):
            self.last_batch = batch
            assert batch["mode"] == "predict_next"
            assert tuple(batch["actions"].shape) == (1, 1, 1)
            return {"hidden": batch["latent"].reshape(1, 2) + 1.0}

    wm = _PredictNextWM()
    env = LatentWorldModelEnv(
        wm,
        classifier=None,
        latent_dim=2,
        action_dim=1,
        initial_latent=np.zeros(2, dtype=np.float32),
        num_envs=1,
    )
    env.reset_slot(0)

    obs, reward, terminated, truncated, _info = env.step_slot(0, [0.5])

    assert wm.last_batch is not None
    assert np.allclose(obs["latent"], np.ones(2, dtype=np.float32))
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    metrics = env.get_metrics()
    assert metrics["model_forwards"] == 1.0
    assert metrics["wm_forward_calls"] == 1.0
    assert metrics["wm_forward_time_s"] >= 0.0


def test_latent_world_model_env_reports_step_batch_sizes():
    class _BatchWM(torch.nn.Module):
        def forward(self, batch):
            return {"hidden": batch["latent"] + 1.0}

    env = LatentWorldModelEnv(
        _BatchWM(),
        classifier=None,
        latent_dim=2,
        action_dim=1,
        initial_latent=np.zeros(2, dtype=np.float32),
        num_envs=4,
    )
    for slot_id in range(4):
        env.reset_slot(slot_id)

    env.step_batch(np.zeros((3, 1), dtype=np.float32), env_ids=[0, 2, 3])
    env.step_slot(1, [0.0])

    metrics = env.get_metrics()

    assert metrics["wm_forward_calls"] == 2.0
    assert metrics["batch_size_sum"] == 4.0
    assert metrics["batch_size_avg"] == 2.0
    assert metrics["batch_size_min"] == 1.0
    assert metrics["batch_size_max"] == 3.0


def test_latent_world_model_env_passes_lang_emb_to_world_model():
    class _LanguageConditionedWM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_batch = None

        def forward(self, batch):
            self.last_batch = batch
            assert tuple(batch["lang_emb"].shape) == (1, 3)
            return {
                "hidden": batch["latent"].reshape(1, 2) + 1.0,
                "lang": batch["lang_emb"],
            }

    wm = _LanguageConditionedWM()
    env = LatentWorldModelEnv(
        wm,
        classifier=None,
        latent_dim=2,
        action_dim=1,
        lang_dim=3,
        initial_latent=np.zeros(2, dtype=np.float32),
        initial_lang_emb=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        num_envs=1,
    )
    reset_obs, _ = env.reset_slot(0)

    obs, _reward, _terminated, _truncated, _info = env.step_slot(0, [0.5])

    assert wm.last_batch is not None
    assert reset_obs["lang_emb"].tolist() == [1.0, 2.0, 3.0]
    assert obs["lang_emb"].tolist() == [1.0, 2.0, 3.0]


def test_latent_world_model_env_passes_proprio_to_world_model():
    class _ProprioConditionedWM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_batch = None

        def forward(self, batch):
            self.last_batch = batch
            assert tuple(batch["proprio"].shape) == (1, 2)
            return {
                "hidden": batch["latent"].reshape(1, 2) + 1.0,
                "proprio": batch["proprio"] + 1.0,
            }

    wm = _ProprioConditionedWM()
    env = LatentWorldModelEnv(
        wm,
        classifier=None,
        latent_dim=2,
        action_dim=1,
        proprio_dim=2,
        initial_latent=np.zeros(2, dtype=np.float32),
        initial_proprio=np.array([5.0, 6.0], dtype=np.float32),
        num_envs=1,
    )
    reset_obs, _ = env.reset_slot(0)

    obs, _reward, _terminated, _truncated, _info = env.step_slot(0, [0.5])
    transition = env.make_transition(
        obs,
        np.array([0.5], dtype=np.float32),
        0.0,
        False,
        False,
        {},
    )

    assert wm.last_batch is not None
    assert reset_obs["proprio"].tolist() == [5.0, 6.0]
    assert obs["proprio"].tolist() == [6.0, 7.0]
    assert transition["proprio"].tolist() == [6.0, 7.0]


def test_latent_world_model_env_strips_internal_proprio_token_width():
    class _InternalTokenWM(torch.nn.Module):
        def forward(self, batch):
            del batch
            return {
                "hidden": torch.tensor(
                    [[[1.0, 2.0, 90.0], [3.0, 4.0, 91.0]]],
                    dtype=torch.float32,
                ),
                "proprio": torch.tensor([[5.0]], dtype=torch.float32),
            }

    env = LatentWorldModelEnv(
        _InternalTokenWM(),
        classifier=None,
        latent_dim=4,
        action_dim=1,
        proprio_dim=1,
        initial_latent=np.zeros(4, dtype=np.float32),
        initial_proprio=np.array([0.0], dtype=np.float32),
        num_envs=1,
    )
    env.reset_slot(0)

    obs, _reward, _terminated, _truncated, _info = env.step_slot(0, [0.5])

    assert obs["latent"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert obs["proprio"].tolist() == [5.0]


def test_latent_world_model_env_scores_classifier_with_window_sidecars():
    class _IdentityWM(torch.nn.Module):
        def forward(self, batch):
            return {
                "hidden": batch["latent"].reshape(1, 4),
                "proprio": batch["proprio"],
                "lang": batch["lang_emb"],
            }

    class _WindowClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(window=3)
            self.last = None

        def forward(self, latent_window, *, task_ids=None, proprio=None, lang_emb=None):
            self.last = (latent_window, task_ids, proprio, lang_emb)
            assert tuple(latent_window.shape) == (1, 3, 4)
            assert tuple(proprio.shape) == (1, 3, 2)
            assert tuple(lang_emb.shape) == (1, 5)
            assert task_ids.tolist() == [7]
            return torch.tensor([[0.0, 2.0]], dtype=torch.float32)

    classifier = _WindowClassifier()
    env = LatentWorldModelEnv(
        _IdentityWM(),
        classifier=classifier,
        latent_dim=4,
        action_dim=1,
        lang_dim=5,
        proprio_dim=2,
        initial_latent=np.ones(4, dtype=np.float32),
        initial_lang_emb=np.arange(5, dtype=np.float32),
        initial_proprio=np.array([1.0, 2.0], dtype=np.float32),
        success_threshold=0.99,
        num_envs=1,
    )
    env.reset_slot(0, task_id=7)

    _obs, reward, terminated, _truncated, _info = env.step_slot(0, [0.5])

    assert classifier.last is not None
    assert np.isclose(reward, float(torch.sigmoid(torch.tensor(2.0))))
    assert terminated is False
