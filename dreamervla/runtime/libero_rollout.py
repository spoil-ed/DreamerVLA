"""Sink-agnostic vectorized LIBERO rollout core (shared by eval + collect).

Drives K VecRolloutEnv slots through a finite (task_id, episode_id) work-list:
gather K obs -> one batched infer_fn -> open-loop action per slot -> step all
active slots in parallel -> per-step callback -> on a slot's done: on_episode +
refill.  Mirrors RLinf's SubprocVectorEnv scatter/gather.  Ported from
`vectorized_collect.collect_vectorized`, which now delegates here.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from dreamervla.runtime.oft_collect import pop_open_loop_action


def _decode_action_chunk(result: Any) -> Any:
    if hasattr(result, "action_chunk"):
        return result.action_chunk
    return result[0]


@dataclass
class SlotStepContext:
    """Everything a sink needs for one executed slot-step."""
    slot: int
    task_id: int
    episode_id: int
    task_description: str
    record_before: Any
    out: Any                # infer_fn output for this slot (has .action_chunk / .hidden_state)
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    record_after: Any


def run_vectorized_rollout(
    vec_env: Any,
    extractors: Sequence[Any],
    infer_fn: Callable[[list[Any]], list[Any]],
    work_list: Sequence[tuple[int, int]],
    episode_horizon: int,
    *,
    obs_from_record: Callable[[Any], Any] = lambda r: r,
    on_step: Callable[[SlotStepContext], Any | None] | None = None,
    on_episode: Callable[[int, int, list[Any], bool], None] | None = None,
    action_steps: int = 1,
    pop_action: Callable[[Any, list, int], Any] = pop_open_loop_action,
) -> int:
    """Run the whole work_list across vec_env.num_envs slots. Returns episodes completed.

    ``pop_action(chunk, queue, action_steps) -> action`` refills ``queue`` from the
    decoded chunk and pops the next open-loop action. Defaults to
    ``pop_open_loop_action`` (OFT collect: gripper-post-processed). The VLA eval
    path injects a variant WITHOUT that gripper transform so the action fed to
    ``vec_env.step`` is byte-identical to the sequential eval (which does not
    gripper-process).
    """
    if len(extractors) != vec_env.num_envs:
        raise ValueError(
            f"need one extractor per env: {len(extractors)} extractors, {vec_env.num_envs} envs"
        )
    num_envs = vec_env.num_envs
    queue = list(work_list)
    next_idx = 0
    done_count = 0
    action_steps = max(1, int(action_steps))

    slot_task = [-1] * num_envs
    slot_desc = [""] * num_envs
    slot_rec: list[Any] = [None] * num_envs
    slot_steps: list[list[Any]] = [[] for _ in range(num_envs)]
    slot_t = [0] * num_envs
    slot_ep = [-1] * num_envs
    active = [False] * num_envs
    action_queues: list[list[Any]] = [[] for _ in range(num_envs)]

    def _start_slot(k: int) -> None:
        nonlocal next_idx
        if next_idx >= len(queue):
            active[k] = False
            return
        tid, ep = queue[next_idx]
        next_idx += 1
        if slot_task[k] != tid:
            slot_desc[k] = vec_env.set_task([tid], env_ids=[k])[0]
            slot_task[k] = tid
        slot_rec[k] = vec_env.reset([tid], [ep], env_ids=[k])[0]
        extractors[k].reset()
        slot_steps[k] = []
        slot_t[k] = 0
        slot_ep[k] = ep
        action_queues[k] = []
        active[k] = True

    for k in range(num_envs):
        _start_slot(k)

    while any(active):
        active_ids = [k for k in range(num_envs) if active[k]]
        preps = [
            extractors[k].prepare(obs_from_record(slot_rec[k]), slot_desc[k])
            for k in active_ids
        ]
        outs = infer_fn(preps)
        actions = [
            pop_action(_decode_action_chunk(outs[i]), action_queues[k], action_steps)
            for i, k in enumerate(active_ids)
        ]
        step_results = vec_env.step(actions, env_ids=active_ids)

        finished: list[int] = []
        for i, k in enumerate(active_ids):
            reward, terminated, truncated, info, rec_after = step_results[i]
            if on_step is not None:
                rec = on_step(SlotStepContext(
                    slot=k, task_id=slot_task[k], episode_id=slot_ep[k],
                    task_description=slot_desc[k], record_before=slot_rec[k],
                    out=outs[i], action=actions[i], reward=reward,
                    terminated=terminated, truncated=truncated, info=info,
                    record_after=rec_after,
                ))
                if rec is not None:
                    slot_steps[k].append(rec)
            slot_t[k] += 1
            slot_rec[k] = rec_after
            if bool(terminated or truncated) or slot_t[k] >= episode_horizon:
                success = bool(info.get("success", terminated))
                if on_episode is not None:
                    on_episode(slot_task[k], slot_ep[k], slot_steps[k], success)
                done_count += 1
                finished.append(k)

        for k in finished:
            _start_slot(k)

    return done_count


def build_grid_work_list(task_ids: Sequence[int], num_episodes_per_task: int) -> list[tuple[int, int]]:
    """Deterministic task-major (task_id, init_state_index) grid == the sequential eval order."""
    return [(int(t), e) for t in task_ids for e in range(int(num_episodes_per_task))]


class SuccessTally:
    """Per-task episode/success accumulator → macro-average SR via eval_metrics."""
    def __init__(self) -> None:
        self._episodes: dict[int, int] = {}
        self._successes: dict[int, int] = {}
        self._order: list[int] = []

    def on_episode(self, task_id: int, episode_id: int, steps: list[Any], success: bool) -> None:
        if task_id not in self._episodes:
            self._episodes[task_id] = 0
            self._successes[task_id] = 0
            self._order.append(task_id)
        self._episodes[task_id] += 1
        self._successes[task_id] += 1 if success else 0

    def summarize(self, *, episodes_per_task: int) -> dict[str, float]:
        from dreamervla.runtime.eval_metrics import summarize_libero_task_success
        records = [
            {"task_id": t, "episodes": self._episodes[t], "successes": self._successes[t]}
            for t in self._order
        ]
        return summarize_libero_task_success(records, episodes_per_task=episodes_per_task)
