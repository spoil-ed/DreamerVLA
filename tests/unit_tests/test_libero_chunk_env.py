"""LiberoChunkEnv (RLinf LiberoEnv port): enumeration, reset, chunk_step."""

import numpy as np
from omegaconf import OmegaConf

from dreamervla.envs.libero_chunk_env import LiberoChunkEnv


class _FakeSuite:
    """3 tasks x 4 init states."""

    def __init__(self, n_tasks=3, n_trials=4):
        self._n_tasks = n_tasks
        self._n_trials = n_trials

    def get_num_tasks(self):
        return self._n_tasks

    def get_task_init_states(self, task_id):
        return [f"init-{task_id}-{k}" for k in range(self._n_trials)]

    def get_task(self, task_id):
        class _T:
            problem_folder = "fake"
            bddl_file = f"task_{task_id}.bddl"
            language = f"do task {task_id}"

        return _T()


class _FakeVecEnv:
    """Stands in for ReconfigureSubprocEnv: obs echo + scripted terminations."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.reconfigured = []
        self.reset_ids = []
        self.init_states = []
        self.terminate_at = {}  # env_id -> step count at which done=True
        self._steps = np.zeros(num_envs, dtype=int)

    def reconfigure_env_fns(self, env_fns, id=None):
        self.reconfigured.append(list(id))

    def seed(self, seed):
        pass

    def reset(self, id=None):
        ids = list(id) if id is not None else list(range(self.num_envs))
        self.reset_ids.append(ids)
        for e in ids:
            self._steps[e] = 0
        return [self._obs() for _ in ids]

    def set_init_state(self, init_state=None, id=None):
        self.init_states.append((list(id), list(init_state)))
        return [self._obs() for _ in id]

    def step(self, actions, id=None):
        ids = list(id) if id is not None else list(range(self.num_envs))
        obs, rews, dones, infos = [], [], [], []
        for e in ids:
            self._steps[e] += 1
            done = self._steps[e] >= self.terminate_at.get(e, 10**9)
            obs.append(self._obs())
            rews.append(0.0)
            dones.append(done)
            infos.append({})
        return obs, np.array(rews), np.array(dones), infos

    def close(self):
        pass

    @staticmethod
    def _obs():
        return {
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }


def _make_env(num_envs=4, **overrides):
    cfg = OmegaConf.create(
        {
            "task_suite_name": "fake",
            "seed": 0,
            "group_size": 1,
            "is_eval": True,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_episode_steps": 6,
            "reset_wait_steps": 2,
            "reset_gripper_open": True,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "task_id_filter": None,
            "max_trials_per_task": None,
            "specific_reset_id": None,
            "init_params": {"camera_heights": 4, "camera_widths": 4},
            **overrides,
        }
    )
    fake_vec = _FakeVecEnv(num_envs)

    class _TestChunkEnv(LiberoChunkEnv):
        def _load_task_suite(self):
            return _FakeSuite()

        def _init_env(self):
            self.env = fake_vec

        def _make_env_fns(self, env_idx=None):
            if env_idx is None:
                env_idx = np.arange(self.num_envs)
            for env_id in env_idx:
                task = self.task_suite.get_task(int(self.task_ids[env_id]))
                self.task_descriptions[env_id] = task.language
            return [None] * len(env_idx)

    env = _TestChunkEnv(cfg, num_envs=num_envs)
    return env, fake_vec


def test_ordered_eval_enumeration_is_task_major_grid():
    env, _ = _make_env(num_envs=4)
    # 3 tasks x 4 trials, eval: no shuffle -> ids 0..11 task-major
    assert env.reset_state_ids.tolist() == [0, 1, 2, 3]
    env.update_reset_state_ids()
    assert env.reset_state_ids.tolist() == [4, 5, 6, 7]
    env.update_reset_state_ids()
    assert env.reset_state_ids.tolist() == [8, 9, 10, 11]
    env.update_reset_state_ids()  # wraps: restarts enumeration
    assert env.reset_state_ids.tolist() == [0, 1, 2, 3]


def test_max_trials_per_task_caps_enumeration():
    env, _ = _make_env(num_envs=6, max_trials_per_task=2)
    # 3 tasks x 2 trials = 6 ids total, one block
    assert env.reset_state_ids.tolist() == [0, 1, 2, 3, 4, 5]
    tids, trids = env._get_task_and_trial_ids_from_reset_state_ids(env.reset_state_ids)
    assert tids.tolist() == [0, 0, 1, 1, 2, 2]
    assert trids.tolist() == [0, 1, 0, 1, 0, 1]


def test_reset_applies_init_states_and_warmup():
    env, vec = _make_env(num_envs=4)
    obs, infos = env.reset()
    # init states applied for the 4 ordered ids (task 0 trials 0..3)
    ids, states = vec.init_states[-1]
    assert ids == [0, 1, 2, 3]
    assert states == ["init-0-0", "init-0-1", "init-0-2", "init-0-3"]
    # reset_wait_steps=2 zero-action warmup steps ran
    assert vec._steps.tolist() == [2, 2, 2, 2]
    assert obs["main_images"].shape == (4, 4, 4, 3)
    assert obs["states"].shape == (4, 8)
    assert len(obs["task_descriptions"]) == 4


def test_eval_reconfigures_only_on_task_change():
    env, vec = _make_env(num_envs=4)
    env.reset()
    n_reconf_initial = len(vec.reconfigured)
    # next ordered block is trials of task 1 -> task changed -> reconfigure
    env.update_reset_state_ids()
    env.is_start = True
    env.reset()
    assert len(vec.reconfigured) > n_reconf_initial
    # resetting the same block again (same tasks) -> no reconfigure in eval mode
    n_after_switch = len(vec.reconfigured)
    env.is_start = True
    env.reset(reset_state_ids=env.reset_state_ids)
    assert len(vec.reconfigured) == n_after_switch
