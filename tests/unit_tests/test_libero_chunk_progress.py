"""Progress callback for the dependency-light RLinf chunk driver."""

import numpy as np

from dreamervla.runtime.libero_chunk_eval import run_rlinf_chunk_eval


class _OneChunkEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.auto_reset = False
        self.is_start = True

    def reset(self):
        return {"task_descriptions": ["a", "b"]}, {}

    def chunk_step(self, chunk_actions):
        steps = int(chunk_actions.shape[1])
        terms = np.zeros((2, steps), dtype=bool)
        truncs = np.zeros((2, steps), dtype=bool)
        truncs[:, -1] = True
        info = {
            "episode": {
                "success_once": np.asarray([True, False]),
                "task_id": np.asarray([0, 0]),
                "reset_state_id": np.asarray([0, 1]),
            }
        }
        return (
            [{"task_descriptions": ["a", "b"]}] * steps,
            np.zeros((2, steps)),
            terms,
            truncs,
            [info] * steps,
        )

    def update_reset_state_ids(self):
        return None


def test_driver_reports_completed_episode_progress() -> None:
    completed = []

    tally = run_rlinf_chunk_eval(
        _OneChunkEnv(),
        lambda _obs: np.zeros((2, 2, 7)),
        n_chunk_steps=2,
        num_epochs=1,
        total_episodes=2,
        on_progress=lambda current: completed.append(current),
    )

    assert tally.num_episodes == 2
    assert completed == [2]
