import numpy as np

from dreamervla.envs.world_model.base_world_model_env import WorldModelEnvProtocol


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
