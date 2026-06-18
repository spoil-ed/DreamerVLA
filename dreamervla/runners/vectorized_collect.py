"""Continuous-stepping vectorized rollout collection (migration §5.3).

Drives K env slots through a finite ``(task_id, episode_id)`` work-list:

    each tick:  gather K obs -> ONE batched VLA forward -> scatter 1 action/slot
                -> step all active slots in parallel -> append per-slot frame
                -> on a slot's done: finalize+write that demo, refill the slot

Batching constraint: ``rollout_hidden_extractor.batched_forward`` requires all preps in a
batch to share a prompt length, and the prompt is the task description.  So the loop
batches PER TASK — all active slots run the same task (same prompt -> batchable), which
is RLinf's ``group_size`` grouping.  Slots desync in time (different init_states ->
different episode lengths) but stay on the same task, so the batch is always valid.  When
a task's episodes are exhausted the slots drain and the loop advances to the next task.

Per-step record pairing mirrors ``collect_parallel_rollouts._run_episode``: the record at
tick t holds the pre-step ``full_record`` (state s_t), the ``obs_embedding`` computed from
that same observation, and the action executed at t (``info['wm_action']``, raw env scale).
``dones``/``sparse_rewards`` are filled on the terminal frame.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from dreamervla.runners.oft_collect_common import process_action

# Per-step proprio fed to the extractor: ee_pos(3) + ee_ori/axisangle(3) + gripper(2) = 8.
_PROPRIO_KEYS = ("ee_pos", "ee_ori", "gripper_states")
# obs sub-group written to the reward-dir HDF5.
_OBS_F64_KEYS = ("ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states")
_OBS_IMG_KEYS = ("agentview_rgb", "eye_in_hand_rgb")


def proprio_from_record(rec: dict[str, Any]) -> np.ndarray:
    """8-dim proprio state for the extractor (matches env._format_obs 'state')."""
    return np.concatenate([rec[k].astype(np.float32) for k in _PROPRIO_KEYS]).astype(np.float32)


def extractor_obs_from_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Build the extractor's obs dict (raw images + 8-dim proprio) from a full_record."""
    return {
        "agentview_rgb": rec["agentview_rgb"],
        "eye_in_hand_rgb": rec["eye_in_hand_rgb"],
        "state": proprio_from_record(rec),
    }


def build_step_record(
    rec: dict[str, Any],
    flat_hidden: Any,
    wm_action: Any,
) -> dict[str, Any]:
    """One per-step dict for ``RolloutDumpWriter.write_demo`` (schema per _run_episode)."""
    emb = flat_hidden.numpy() if hasattr(flat_hidden, "numpy") else np.asarray(flat_hidden)
    return {
        "actions": np.asarray(wm_action, dtype=np.float64),
        "rewards": np.float32(0.0),
        "sparse_rewards": np.uint8(0),  # set on the terminal frame
        "dones": np.uint8(0),           # set on the terminal frame
        "robot_states": rec["robot_states"].astype(np.float64),
        "states": rec["states"].astype(np.float64),
        "obs": {
            **{k: rec[k] for k in _OBS_IMG_KEYS},
            **{k: rec[k].astype(np.float64) for k in _OBS_F64_KEYS},
        },
        "obs_embedding": emb,
    }


def collect_vectorized(
    vec_env: Any,
    extractors: Sequence[Any],
    infer_fn: Callable[[list[dict[str, Any]]], list[tuple[list[Any], Any]]],
    writer: Any,
    work_list: Sequence[tuple[int, int]],
    episode_horizon: int,
    *,
    preprocess_config: dict[str, Any] | None = None,
    data_attrs: dict[str, Any] | None = None,
    start_demo_index: int = 0,
    rank: int = 0,
    on_episode: Callable[[int, int, int, bool], None] | None = None,
) -> int:
    """Collect the whole ``work_list`` across ``vec_env.num_envs`` slots in one continuous loop.

    Slots pull ``(task_id, episode_id)`` items from the work-list as they finish episodes, so
    different slots may run DIFFERENT tasks at the same tick.  ``infer_fn`` (batched_forward)
    left-pads mixed-length prompts, so a batch spanning tasks is valid — there is **no barrier
    between tasks and no tail idle**.  The work-list is kept in order (task-major from the
    caller), so a slot only reconfigures its env when it crosses a task boundary.

    Args:
        vec_env: a VecRolloutEnv-like object: ``num_envs``, ``set_task(task_ids, env_ids)``,
            ``reset(task_ids, episode_ids, env_ids)`` (returns full_record dicts),
            ``step(actions, env_ids)`` (returns per-env (reward, term, trunc, info, record)).
        extractors: list of length ``num_envs``; each has ``reset()`` and
            ``prepare(obs, task_description) -> prep``.  One per slot (isolated history).
        infer_fn: ``preps -> list[(action_chunk, flat_hidden)]`` (e.g. a closure over
            ``rollout_hidden_extractor.batched_forward``).  MAY receive mixed prompt lengths.
        writer: a RolloutDumpWriter-like object with
            ``write_demo(index, steps, preprocess_config=, data_attrs=)``.
        work_list: ``(task_id, episode_id)`` pairs to collect (consumed in the given order).
        episode_horizon: max steps per episode (truncation bound).
        preprocess_config / data_attrs: written once, on the first demo (pass None on
            non-rank-0 to skip — only rank 0 should write the shared sidecar config).
        start_demo_index: first demo index (for shard-local numbering).
        on_episode: optional callback ``(task_id, episode_id, n_steps, success)`` per demo.

    Returns:
        Number of demos written.
    """
    if len(extractors) != vec_env.num_envs:
        raise ValueError(
            f"need one extractor per env: {len(extractors)} extractors, {vec_env.num_envs} envs"
        )

    num_envs = vec_env.num_envs
    queue = list(work_list)
    next_idx = 0
    demo_index = start_demo_index
    pending_config = preprocess_config
    pending_attrs = data_attrs

    slot_task = [-1] * num_envs  # current task id per slot (skip redundant set_task)
    slot_desc = [""] * num_envs  # current task description per slot
    slot_rec: list[Any] = [None] * num_envs
    slot_steps: list[Any] = [None] * num_envs
    slot_t = [0] * num_envs
    slot_ep = [-1] * num_envs
    active = [False] * num_envs

    def _start_slot(k: int) -> None:
        nonlocal next_idx
        if next_idx >= len(queue):
            active[k] = False
            return
        tid, ep = queue[next_idx]
        next_idx += 1
        if slot_task[k] != tid:  # reconfigure the env only when crossing a task boundary
            slot_desc[k] = vec_env.set_task([tid], env_ids=[k])[0]
            slot_task[k] = tid
        rec = vec_env.reset([tid], [ep], env_ids=[k])[0]
        extractors[k].reset()
        slot_rec[k] = rec
        slot_steps[k] = []
        slot_t[k] = 0
        slot_ep[k] = ep
        active[k] = True

    for k in range(num_envs):
        _start_slot(k)

    while any(active):
        active_ids = [k for k in range(num_envs) if active[k]]
        preps = [
            extractors[k].prepare(extractor_obs_from_record(slot_rec[k]), slot_desc[k])
            for k in active_ids
        ]
        outs = infer_fn(preps)  # aligned with active_ids
        # receding-horizon: execute chunk[0] per slot, with gripper post-process
        # (required before env.step; also what gets recorded as wm_action below).
        actions = [process_action(outs[i][0][0]) for i in range(len(active_ids))]
        step_results = vec_env.step(actions, env_ids=active_ids)

        finished: list[int] = []
        for i, k in enumerate(active_ids):
            _action_chunk, flat_hidden = outs[i]
            reward, terminated, truncated, info, rec_after = step_results[i]
            wm_action = info.get("wm_action", info.get("env_action", actions[i]))
            slot_steps[k].append(build_step_record(slot_rec[k], flat_hidden, wm_action))
            slot_t[k] += 1
            slot_rec[k] = rec_after
            done = bool(terminated or truncated) or slot_t[k] >= episode_horizon
            if done:
                success = bool(info.get("success", terminated))
                steps = slot_steps[k]
                steps[-1]["dones"] = np.uint8(1)
                steps[-1]["sparse_rewards"] = np.uint8(1 if success else 0)
                writer.write_demo(
                    index=demo_index,
                    steps=steps,
                    preprocess_config=pending_config,
                    data_attrs=pending_attrs,
                    task_id=slot_task[k],
                    episode_id=slot_ep[k],
                    task_description=slot_desc[k],
                    episode_success=success,
                    episode_horizon=episode_horizon,
                )
                pending_config = None
                pending_attrs = None
                if on_episode is not None:
                    on_episode(slot_task[k], slot_ep[k], len(steps), success)
                demo_index += 1
                finished.append(k)

        for k in finished:
            _start_slot(k)

    return demo_index - start_demo_index
