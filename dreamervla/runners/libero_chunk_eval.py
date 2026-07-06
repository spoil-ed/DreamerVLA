"""Single-process port of RLinf's env eval loop (env_worker.evaluate).

Drives a LiberoChunkEnv through epochs of [reset -> n_chunk_steps x
(policy_fn -> chunk_step)]. Metrics follow RLinf: an episode is counted when
its env is NEWLY done (prev-done mask when auto_reset is off, per-chunk dones
when auto_reset is on), reading ``episode.success_once``. Deviations: the
tally dedups by ``reset_state_id`` (rolling ordered blocks can wrap), epochs
may early-break once every env is done / the target episode count is reached
(RLinf steps a fixed lockstep schedule because of channel pairing), and the
driver advances the ordered enumeration between non-auto-reset epochs via
``env.update_reset_state_ids()`` (RLinf only advances inside auto-reset).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class ChunkEvalTally:
    """First-result-wins per-(reset_state_id) success record."""

    def __init__(self) -> None:
        self._by_reset_id: dict[int, tuple[int, bool]] = {}
        self.env_chunk_steps = 0
        self.env_action_steps = 0

    def add(self, episode_info: dict, newly_done: np.ndarray) -> None:
        for i in np.flatnonzero(np.asarray(newly_done)):
            reset_id = int(episode_info["reset_state_id"][i])
            if reset_id in self._by_reset_id:
                continue
            self._by_reset_id[reset_id] = (
                int(episode_info["task_id"][i]),
                bool(episode_info["success_once"][i]),
            )

    def record_chunk_step(self, chunk_actions: np.ndarray) -> None:
        actions = np.asarray(chunk_actions)
        if actions.ndim < 2:
            raise ValueError(
                "chunk eval policy_fn must return actions with shape "
                "[num_envs, chunk_steps, ...]"
            )
        self.env_chunk_steps += int(actions.shape[0])
        self.env_action_steps += int(actions.shape[0] * actions.shape[1])

    @property
    def num_episodes(self) -> int:
        return len(self._by_reset_id)

    def records(self) -> dict[int, dict[int, bool]]:
        out: dict[int, dict[int, bool]] = {}
        for reset_id in sorted(self._by_reset_id):
            task_id, success = self._by_reset_id[reset_id]
            out.setdefault(task_id, {})[reset_id] = success
        return out

    def summarize(self, *, episodes_per_task: int) -> dict[str, float]:
        from dreamervla.runners.eval_metrics import summarize_libero_task_success

        records = [
            {
                "task_id": task_id,
                "episodes": len(results),
                "successes": sum(results.values()),
            }
            for task_id, results in sorted(self.records().items())
        ]
        return summarize_libero_task_success(
            records, episodes_per_task=episodes_per_task
        )


def run_rlinf_chunk_eval(
    env: Any,
    policy_fn: Callable[[dict], np.ndarray],
    *,
    n_chunk_steps: int,
    num_epochs: int,
    total_episodes: int,
    on_epoch_start: Callable[[], None] | None = None,
) -> ChunkEvalTally:
    tally = ChunkEvalTally()
    for _epoch in range(int(num_epochs)):
        if tally.num_episodes >= total_episodes:
            break
        if on_epoch_start is not None:
            on_epoch_start()
        env.is_start = True
        prev_done = np.zeros(env.num_envs, dtype=bool)
        obs, _infos = env.reset()
        for _chunk in range(int(n_chunk_steps)):
            chunk_actions = policy_fn(obs)
            tally.record_chunk_step(chunk_actions)
            obs_list, _rewards, terms, truncs, infos_list = env.chunk_step(
                chunk_actions
            )
            obs = obs_list[-1]
            infos = infos_list[-1]
            current_dones = np.logical_or(terms, truncs)[:, -1]
            if env.auto_reset:
                newly_done = current_dones
            else:
                newly_done = current_dones & ~prev_done
                prev_done = prev_done | current_dones
            if newly_done.any():
                episode_info = (
                    infos["final_info"]["episode"]
                    if "final_info" in infos
                    else infos["episode"]
                )
                tally.add(episode_info, newly_done)
            if not env.auto_reset and prev_done.all():
                break
            if tally.num_episodes >= total_episodes:
                break
        if not env.auto_reset:
            env.update_reset_state_ids()
    return tally
