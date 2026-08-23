"""ReconfigureSubprocEnv: spawn workers + in-child env rebuild."""

import numpy as np

from dreamervla.envs.libero.venv import ReconfigureSubprocEnv


class _TagEnv:
    """Minimal gym-like env; step echoes its construction tag."""

    def __init__(self, tag):
        self.tag = tag

    def reset(self):
        return {"tag": self.tag}

    def step(self, action):
        return {"tag": self.tag}, 0.0, False, {"tag": self.tag}

    def seed(self, seed):
        return [seed]

    def close(self):
        pass


def _make_tag_env(tag="a"):
    return _TagEnv(tag)


def _make_tag_env_a():
    return _make_tag_env("a")


def _make_tag_env_b():
    return _make_tag_env("b")


def test_reconfigure_rebuilds_child_env():
    env = ReconfigureSubprocEnv([_make_tag_env_a, _make_tag_env_a])
    try:
        env.reset()
        _, _, _, infos = env.step(np.zeros((2, 7)))
        assert [i["tag"] for i in infos] == ["a", "a"]
        env.reconfigure_env_fns([_make_tag_env_b], id=[1])
        env.reset(id=[1])
        _, _, _, infos = env.step(np.zeros((2, 7)))
        assert [i["tag"] for i in infos] == ["a", "b"]
    finally:
        env.close()
