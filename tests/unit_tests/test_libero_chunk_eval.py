"""RLinf-style chunk eval driver: prev-done dedup, rolling blocks, per-task SR."""

import numpy as np

from dreamervla.runners.libero_chunk_eval import ChunkEvalTally, run_rlinf_chunk_eval


class _ScriptedChunkEnv:
    """LiberoChunkEnv stand-in: (task, trial) grid of 2 tasks x 3 trials,
    success iff (task_id + trial_id) is even, episodes end after 1 chunk."""

    def __init__(self, num_envs, auto_reset=False):
        self.num_envs = num_envs
        self.auto_reset = auto_reset
        self.is_start = True
        self.n_tasks, self.n_trials = 2, 3
        self.total = self.n_tasks * self.n_trials
        self._block_start = 0
        self.reset_state_ids = None
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        ids = [(self._block_start + k) % self.total for k in range(self.num_envs)]
        self._block_start = (self._block_start + self.num_envs) % self.total
        self.reset_state_ids = np.array(ids)

    def reset(self, env_idx=None, reset_state_ids=None):
        self.is_start = False
        return self._obs(), {}

    def _obs(self):
        return {"task_descriptions": ["t"] * self.num_envs}

    def _episode_info(self):
        task_ids = self.reset_state_ids // self.n_trials
        trial_ids = self.reset_state_ids % self.n_trials
        return {
            "success_once": (task_ids + trial_ids) % 2 == 0,
            "task_id": task_ids,
            "reset_state_id": self.reset_state_ids.copy(),
        }

    def chunk_step(self, chunk_actions):
        n, c = chunk_actions.shape[0], chunk_actions.shape[1]
        terms = np.zeros((n, c), dtype=bool)
        truncs = np.zeros((n, c), dtype=bool)
        truncs[:, -1] = True  # every episode ends after one chunk
        infos = {"episode": self._episode_info()}
        if self.auto_reset:
            infos = {"final_info": {"episode": self._episode_info()}}
            self.update_reset_state_ids()
        return (
            [self._obs()] * c,
            np.zeros((n, c)),
            terms,
            truncs,
            [infos] * c,
        )


def _policy(obs):
    return np.zeros((len(obs["task_descriptions"]), 2, 7))


def test_tally_dedups_wrapped_reset_ids():
    tally = ChunkEvalTally()
    ep = {
        "success_once": np.array([True, False]),
        "task_id": np.array([0, 0]),
        "reset_state_id": np.array([0, 1]),
    }
    tally.add(ep, np.array([True, True]))
    # same reset ids come around again with flipped success: first result wins
    ep2 = {
        "success_once": np.array([False, True]),
        "task_id": np.array([0, 0]),
        "reset_state_id": np.array([0, 1]),
    }
    tally.add(ep2, np.array([True, True]))
    assert tally.num_episodes == 2
    assert tally.records() == {0: {0: True, 1: False}}


def test_driver_covers_grid_with_rolling_blocks():
    # 6 episodes, 4 envs -> 2 epochs, last block wraps (dedup keeps 6)
    env = _ScriptedChunkEnv(num_envs=4, auto_reset=False)
    tally = run_rlinf_chunk_eval(
        env, _policy, n_chunk_steps=3, num_epochs=2, total_episodes=6
    )
    assert tally.num_episodes == 6
    metrics = tally.summarize(episodes_per_task=3)
    # success iff (task+trial) even: task0 -> trials 0,2 succeed (2/3); task1 -> trial 1 (1/3)
    assert metrics["eval_task_0_success_rate"] == 2 / 3
    assert metrics["eval_task_1_success_rate"] == 1 / 3
    assert abs(metrics["eval_success_rate"] - 0.5) < 1e-9


def test_driver_auto_reset_counts_newly_done_each_chunk():
    env = _ScriptedChunkEnv(num_envs=3, auto_reset=True)
    tally = run_rlinf_chunk_eval(
        env, _policy, n_chunk_steps=2, num_epochs=1, total_episodes=6
    )
    assert tally.num_episodes == 6
