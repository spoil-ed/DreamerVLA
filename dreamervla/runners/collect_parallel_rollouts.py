"""Multi-rank rollout collector for DreamerVLA (torchrun M-rank sharding).

Drives a base OFT VLA in LIBERO and dumps reward-dir-compatible HDF5 files
plus obs_embedding sidecars and preprocess_config.json that the offline WM
trainer (BalancedTerminalDataset) can consume zero-change.

Layer 1 — torchrun M-rank sharding (PRIMARY):
    Run under ``torchrun --nproc_per_node=M``; each rank collects its own slice
    of the (task_id, episode) work-list on its own GPU (LOCAL_RANK).  Each rank
    writes shard files prefixed with its rank:
        reward/r{rank}_shard_000.hdf5
        hidden/r{rank}_shard_000.hdf5
    BalancedTerminalDataset globs ``*.hdf5`` from the dir, so it picks up all
    shards from all ranks automatically.  preprocess_config.json (identical
    content on every rank) is written by rank 0 only to avoid races.

    Single-process path: if WORLD_SIZE is unset or 1 the script behaves
    exactly as the original single-process collector (shard prefix is empty,
    file is ``shard_000.hdf5`` as before).

Image rotation contract (explicit):
  - full_record() uses pixel_rotate_180=False (DreamerVLAOnlineTrainEnvConfig
    default) -> stored dump images are in raw camera orientation (NOT rotated).
  - The extractor receives those same raw images and applies rotate_images_180=True
    internally to match the offline sidecar protocol.
  - This matches exactly how test_inline_matches_offline_sidecar feeds frames:
    it reads obs_group[key][t] (raw/un-rotated reward-dir frames) into
    extractor.step() with rotate_images_180=True inside the extractor.

Entry point (pure Hydra; torchrun M-rank):
  torchrun --standalone --nproc_per_node=2 -m dreamervla.train \\
    experiment=collect_rollouts_onetraj task=openvla_onetraj_coldstart_libero \\
    collect.task_ids=all collect.episodes_per_task=2 collect.episode_horizon=64
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamervla.dataset.collection_manifest import (
    complete_episode_ids_per_task,
    next_shard_index,
)
from dreamervla.runners.oft_collect_common import (
    assert_policy_mode_matches,
    load_policy,
    make_preprocess_config,
    oft_open_loop_action,
    resolve_model_path,
    sidecar_to_numpy,
    vla_latent_spec,
)
from dreamervla.utils.paths import data_path, data_root
from dreamervla.utils.progress import AggregateProgress

_assert_policy_mode_matches = assert_policy_mode_matches
_load_policy = load_policy
_make_preprocess_config = make_preprocess_config
_resolve_model_path = resolve_model_path

# ---------------------------------------------------------------------------
# Torchrun rank helpers
# ---------------------------------------------------------------------------


def _get_dist_info() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank) from env; defaults to (0, 1, 0)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _resolve_task_ids(task_ids: Any, num_tasks: int) -> list[int]:
    if isinstance(task_ids, (list, tuple)):
        return [int(x) for x in task_ids]
    if isinstance(task_ids, int):
        return [task_ids]
    if str(task_ids).strip().lower() == "all":
        return list(range(num_tasks))
    return [int(x.strip()) for x in str(task_ids).split(",") if x.strip()]


def _shard_work(
    work_list: list[tuple[int, int]],
    rank: int,
    world_size: int,
) -> list[tuple[int, int]]:
    """Return the slice of work_list assigned to this rank (round-robin)."""
    return [item for i, item in enumerate(work_list) if i % world_size == rank]


def _build_work_list(
    task_ids: list[int],
    episodes_per_task: int,
    collected_per_task: dict[int, int],
) -> list[tuple[int, int]]:
    """Resume-aware ``(task_id, episode_id)`` work list.

    ``episode_id`` is the init_state selector (env reset uses
    ``init_state = episode_id % num_init_states``), so each task CONTINUES from the count
    already on disk instead of restarting at 0 — otherwise a resume re-collects the same
    init_states. Built globally (before round-robin rank sharding) so ranks stay disjoint.
    """
    return [
        (int(tid), int(collected_per_task.get(int(tid), 0)) + ep)
        for tid in task_ids
        for ep in range(int(episodes_per_task))
    ]


def _build_resume_work_list(
    task_ids: list[int],
    target_episodes_per_task: int,
    complete_episode_ids_by_task: dict[int, set[int]],
) -> list[tuple[int, int]]:
    """Return missing ``(task_id, episode_id)`` pairs below a per-task target.

    Unlike ``_build_work_list``, this preserves holes: if episode ids 0 and 2 are
    complete and the target is 4, only 1 and 3 are scheduled. Incomplete or absent
    ids are therefore re-collected, while complete ids are skipped.
    """
    target = int(target_episodes_per_task)
    return [
        (int(tid), int(ep))
        for tid in task_ids
        for ep in range(target)
        if ep not in complete_episode_ids_by_task.get(int(tid), set())
    ]


_REQUIRED_COLLECT_KEYS: tuple[str, ...] = (
    "model_path", "policy_mode", "unnorm_key", "task_suite_name", "task_ids",
    "episodes_per_task", "episode_horizon", "envs_per_gpu", "memory_fraction",
    "reward_dir", "hidden_dir", "image_keys", "expected_history", "num_images_in_input",
    "expected_action_head_type", "expected_include_state", "expected_obs_hidden_source",
    "expected_prompt_style", "expected_rotate_images_180", "time_horizon", "token_dim",
    "action_dim", "chunk_size", "resolution", "gpu_id",
)


def _require_keys(cfg: dict[str, Any]) -> None:
    """Fail fast (before any GPU/model work) if a required extraction key is absent."""
    missing = [k for k in _REQUIRED_COLLECT_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"collect_rollouts cfg missing required keys: {missing}")


def _assert_gpu_free_memory(gpu_id: int, min_free_gb: float, *, rank: int) -> None:
    """Fail fast if the rank's GPU lacks free memory for the OFT VLA (~16 GB).

    On a shared box another job may already hold the per-rank GPU; without this preflight
    the load OOMs silently and only the rank whose GPU happened to be free survives, which
    reads as "only one GPU is working". ``min_free_gb <= 0`` disables the check.
    """
    if min_free_gb <= 0:
        return
    free_gb = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"[collector rank={rank}] GPU {gpu_id} has only {free_gb:.1f} GB free but the "
            f"OFT VLA needs ~{min_free_gb:.0f} GB — it is likely occupied by another job. "
            f"Point collection at free GPUs via CUDA_VISIBLE_DEVICES, or lower "
            f"collect.min_free_gpu_gb (=0 disables this check)."
        )


def _make_dump_writer(
    reward_dir: Path,
    hidden_dir: Path,
    shard_name: str,
    *,
    shard_prefix: str,
    start_index: int,
    demos_per_shard: int,
) -> Any:
    """Single growing shard, or a writer that slices demos into N-sized shards.

    ``demos_per_shard <= 0`` returns the plain ``RolloutDumpWriter`` (one shard per
    rank, byte-identical to before); ``> 0`` returns a ``RotatingRolloutDumpWriter``
    that rolls a new ``{prefix}_{NNN}.hdf5`` shard every ``demos_per_shard`` demos
    so a long collect is sliced (and a crash only loses the last small shard).
    """
    from dreamervla.dataset.rollout_dump_writer import (
        RolloutDumpWriter,
        RotatingRolloutDumpWriter,
    )

    if demos_per_shard and int(demos_per_shard) > 0:
        return RotatingRolloutDumpWriter(
            reward_dir,
            hidden_dir,
            shard_prefix=shard_prefix,
            demos_per_shard=int(demos_per_shard),
            start_index=start_index,
        )
    return RolloutDumpWriter(reward_dir, hidden_dir, shard_name)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    env: Any,
    extractor: Any,
    task_description: str,
    episode_id: int,
    episode_horizon: int,
    task_id: int,
    rank: int = 0,
    action_steps: int = 1,
) -> list[dict[str, Any]]:
    """Run one episode and return a list of per-step dicts for the writer.

    Per-step dict schema (matches RolloutDumpWriter.write_demo expectations):
        actions         (7,)    float64   — raw env-scale action (wm_action)
        rewards         scalar  float32   — always 0.0 (collector convention)
        sparse_rewards  scalar  uint8     — 0 everywhere except terminal success=1
        dones           scalar  uint8     — 1 on the final step
        robot_states    (9,)    float64
        states          (S,)    float64
        obs:
            agentview_rgb    (256,256,3) uint8   — raw / non-rotated
            eye_in_hand_rgb  (256,256,3) uint8   — raw / non-rotated
            ee_pos           (3,)        float64
            ee_ori           (3,)        float64
            ee_states        (6,)        float64
            gripper_states   (2,)        float64
            joint_states     (7,)        float64
        obs_embedding   (229376,) float16

    Image rotation contract:
        full_record() is called with pixel_rotate_180=False (env default), so
        the images stored in obs["agentview_rgb"] / obs["eye_in_hand_rgb"] are
        in raw camera orientation (NOT rotated).  The same raw frames are passed
        to extractor.step(), which applies rotate_images_180=True internally.
        This exactly mirrors how test_inline_matches_offline_sidecar feeds frames.
    """
    obs, reset_info = env.reset(episode_id=episode_id, task_id=task_id)
    init_state_index = (
        reset_info.get("init_state_index") if isinstance(reset_info, dict) else None
    )
    extractor.reset()

    # Build the 8-dim proprio state for the extractor from full_record fields.
    # Layout: ee_pos(3) + ee_ori/axisangle(3) + gripper_states(2)
    # (matches env._format_obs "state" computation)
    def _proprio_from_rec(rec: dict[str, Any]) -> np.ndarray:
        return np.concatenate([
            rec["ee_pos"].astype(np.float32),
            rec["ee_ori"].astype(np.float32),     # already axisangle from full_record
            rec["gripper_states"].astype(np.float32),
        ]).astype(np.float32)

    steps: list[dict[str, Any]] = []
    success = False
    done = False
    t = 0
    action_queue: list[Any] = []
    action_steps = max(1, int(action_steps))
    while not done and t < episode_horizon:
        rec = env.full_record()

        extractor_obs = {
            "agentview_rgb": rec["agentview_rgb"],      # raw, extractor rotates internally
            "eye_in_hand_rgb": rec["eye_in_hand_rgb"],
            "state": _proprio_from_rec(rec),
        }
        # Open-loop OFT action + per-step obs_embedding — the SINGLE shared
        # implementation also used by the online cotrain rollout
        # (oft_open_loop_action). process_action (inside it) does the LIBERO
        # gripper binarize/invert required before env.step.
        step_out = oft_open_loop_action(
            extractor, extractor_obs, task_description, action_queue, action_steps
        )
        action, hidden_state = step_out

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        success = bool(info.get("success", terminated))

        step = {
            "actions": np.asarray(info.get("wm_action", info.get("env_action", action)), dtype=np.float64),
            "rewards": np.float32(0.0),
            "sparse_rewards": np.uint8(0),
            "dones": np.uint8(0),
            "robot_states": rec["robot_states"].astype(np.float64),
            "states": rec["states"].astype(np.float64),
            "obs": {
                "agentview_rgb": rec["agentview_rgb"],
                "eye_in_hand_rgb": rec["eye_in_hand_rgb"],
                "ee_pos": rec["ee_pos"].astype(np.float64),
                "ee_ori": rec["ee_ori"].astype(np.float64),
                "ee_states": rec["ee_states"].astype(np.float64),
                "gripper_states": rec["gripper_states"].astype(np.float64),
                "joint_states": rec["joint_states"].astype(np.float64),
            },
            "obs_embedding": sidecar_to_numpy(hidden_state),
        }
        if rec.get("init_state_index") is not None:
            step["init_state_index"] = int(rec["init_state_index"])
        elif init_state_index is not None:
            step["init_state_index"] = int(init_state_index)
        lang_emb = sidecar_to_numpy(getattr(step_out, "lang_emb", None), dtype=np.float32)
        if lang_emb is not None:
            step["lang_emb"] = lang_emb.reshape(-1)
        steps.append(step)
        t += 1

    # Post-episode: set dones and sparse_rewards on terminal step.
    # sparse_rewards[T-1] = 1 iff success (per collector convention).
    # dones[T-1] = 1 always (episode ended).
    if steps:
        steps[-1]["dones"] = np.uint8(1)
        steps[-1]["sparse_rewards"] = np.uint8(1 if success else 0)

    if rank == 0:
        print(
            f"  [rank={rank} episode {episode_id}] task_id={task_id} steps={len(steps)} "
            f"success={success}",
            flush=True,
        )
    return steps


# ---------------------------------------------------------------------------
# Layer-2: vectorized (K env subprocess) collection path
# ---------------------------------------------------------------------------

def _collect_vectorized_path(
    *,
    policy: Any,
    extractor: Any,
    unnorm_key: str,
    env_cfg_kwargs: dict[str, Any],
    num_envs: int,
    my_work: list[tuple[int, int]],
    episode_horizon: int,
    reward_dir: Path,
    hidden_dir: Path,
    shard_name: str,
    shard_prefix: str,
    shard_idx: int,
    demos_per_shard: int,
    preprocess_config: dict[str, Any],
    task_suite_name: str,
    rank: int,
    world_size: int,
    progress_dir: str | None,
    history: int,
    rotate_images_180: bool,
    image_keys: list[str],
    obs_hidden_source: str,
    on_episode: Callable[[int, int, int, bool], None] | None = None,
) -> int:
    """Drive K env subprocesses with batched VLA inference; returns demos written.

    K independent envs step in parallel (VecRolloutEnv, send-all/recv-all) while the K
    observations are batched through one VLA forward (batched_forward).  Work is batched
    per task so every batch shares a prompt length.  See the migration spec §5.
    """
    from dreamervla.runners.rollout_hidden_extractor import (
        OFTBatchedDecoder,
        OFTRolloutHiddenExtractor,
    )
    from dreamervla.runners.vec_rollout_env import VecRolloutEnv
    from dreamervla.runners.vectorized_collect import collect_vectorized

    # Children inherit the LIBERO/mujoco env vars (spawn does not carry runtime edits).
    env_vars = {
        k: os.environ[k]
        for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "DVLA_DATA_ROOT", "LIBERO_CONFIG_PATH")
        if k in os.environ
    }

    if rank == 0:
        print(
            f"[collector rank={rank}] Layer-2: spawning {num_envs} env subprocesses ...",
            flush=True,
        )
    vec_env = VecRolloutEnv(num_envs=num_envs, cfg_kwargs=env_cfg_kwargs, env_vars=env_vars)
    try:
        # One extractor per slot (isolated history); reuse the main one for slot 0.
        extractors = [extractor] + [
            OFTRolloutHiddenExtractor(
                policy,
                image_keys=list(image_keys),
                history=int(history),
                rotate_images_180=bool(rotate_images_180),
                center_crop=True,
                unnorm_key=unnorm_key,
                obs_hidden_source=str(obs_hidden_source),
            )
            for _ in range(num_envs - 1)
        ]

        first_desc = vec_env.set_task([my_work[0][0]], env_ids=[0])[0]
        data_attrs = {"task_suite_name": task_suite_name, "env_name": first_desc}

        # Build the batched decoder ONCE (resolves model handles / head mode a single time).
        decoder = OFTBatchedDecoder(
            policy,
            unnorm_key,
            obs_hidden_source=str(obs_hidden_source),
            image_keys=list(image_keys),
        )

        def infer_fn(preps: list[dict[str, Any]]) -> list[tuple[list[Any], Any]]:
            return decoder.predict_batch(preps)

        def _vec_on_episode(tid: int, ep: int, ns: int, ok: bool) -> None:
            if rank == 0:
                print(
                    f"  [rank={rank} vec] task={tid} ep={ep} steps={ns} success={ok}",
                    flush=True,
                )
            if on_episode is not None:
                on_episode(tid, ep, ns, ok)

        with _make_dump_writer(
            reward_dir,
            hidden_dir,
            shard_name,
            shard_prefix=shard_prefix,
            start_index=shard_idx,
            demos_per_shard=demos_per_shard,
        ) as writer:
            return collect_vectorized(
                vec_env,
                extractors,
                infer_fn,
                writer,
                my_work,
                episode_horizon,
                preprocess_config=(preprocess_config if rank == 0 else None),
                data_attrs=data_attrs,
                rank=rank,
                world_size=world_size,
                progress_dir=progress_dir,
                action_steps=int(preprocess_config["chunk_size"]),
                on_episode=_vec_on_episode,
            )
    finally:
        vec_env.close()


# ---------------------------------------------------------------------------
# Runner entry point
# ---------------------------------------------------------------------------

def collect_rollouts(
    cfg: dict[str, Any],
    rank: int,
    world_size: int,
    local_rank: int,
    on_episode: Callable[[int, int, int, bool], None] | None = None,
) -> int:
    """Collect rollouts for this rank's work-slice; returns demos written.

    All extraction parameters come from ``cfg`` (no hardcoded defaults); see
    _REQUIRED_COLLECT_KEYS.  Layer-1 sharding uses (rank, world_size); Layer-2
    within-rank K-env batching is enabled by cfg["envs_per_gpu"] > 1.

    on_episode: optional callback ``(task_id, episode_id, n_steps, success)``
        invoked after each episode completes; used by the runner console API.
    """
    _require_keys(cfg)
    cfg["_rank"] = rank
    cfg["_world_size"] = world_size
    cfg["_local_rank"] = local_rank
    is_distributed = world_size > 1

    if is_distributed:
        gpu_id = local_rank
    else:
        gpu_id = int(cfg["gpu_id"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_id = 0

    if rank == 0:
        print("[collector] config:", cfg, flush=True)
    print(f"[collector] rank={rank}/{world_size} local_rank={local_rank} gpu_id={gpu_id}", flush=True)

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    data_root()
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(data_path(".libero")))

    memory_fraction = float(cfg["memory_fraction"])
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError(f"collect.memory_fraction must be in (0, 1], got {memory_fraction}")
    torch.cuda.set_device(gpu_id)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=gpu_id)

    from dreamervla.envs.train_env import (
        DreamerVLAOnlineTrainEnv,
        DreamerVLAOnlineTrainEnvConfig,
    )
    from dreamervla.runners.rollout_hidden_extractor import OFTRolloutHiddenExtractor

    task_suite_name = cfg["task_suite_name"]
    episodes_per_task = int(cfg["episodes_per_task"])
    episode_horizon = int(cfg["episode_horizon"])
    unnorm_key = cfg["unnorm_key"]
    envs_per_gpu = int(cfg["envs_per_gpu"])
    resolution = int(cfg["resolution"])
    image_keys = list(cfg["image_keys"])
    history = int(cfg["expected_history"])
    rotate_images_180 = bool(cfg["expected_rotate_images_180"])

    reward_dir = Path(cfg["reward_dir"]).expanduser().resolve()
    hidden_dir = Path(cfg["hidden_dir"]).expanduser().resolve()
    reward_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # Append-aware: a fresh dir yields index 000 (byte-identical to the old fixed
    # name); a resumed collection writes the next index instead of overwriting.
    shard_prefix = f"r{rank}_shard" if is_distributed else "shard"
    shard_idx = next_shard_index(reward_dir, prefix=shard_prefix)
    shard_name = f"{shard_prefix}_{shard_idx:03d}.hdf5"
    # Optional slicing: >0 rolls a new shard every N demos (resumable: rotation
    # continues from shard_idx, so next_shard_index picks up where this run left off).
    demos_per_shard = int(cfg.get("demos_per_shard", 0))

    # Preflight the target GPU BEFORE loading the ~16 GB OFT VLA. On a shared box the
    # per-rank GPU may already be occupied by another job; without this check the load
    # silently OOMs and only the rank whose GPU happened to be free survives — looking
    # like "only one GPU is working". Fail fast with the rank + GPU named instead.
    _assert_gpu_free_memory(gpu_id, float(cfg.get("min_free_gpu_gb", 18.0)), rank=rank)

    policy = _load_policy(cfg, gpu_id)
    _assert_policy_mode_matches(cfg)
    if str(cfg["expected_obs_hidden_source"]) == "input_token_embedding":
        spec = vla_latent_spec(policy.vla, image_keys)
        cfg["token_count"] = int(spec["token_count"])
        cfg["hidden_dim"] = int(spec["flat_dim"])
        cfg["patches_per_image"] = int(spec["patches_per_image"])
        cfg["num_images_in_input"] = int(spec["num_images_in_input"])
    extractor = OFTRolloutHiddenExtractor(
        policy,
        image_keys=image_keys,
        history=history,
        rotate_images_180=rotate_images_180,
        center_crop=True,
        unnorm_key=unnorm_key,
        obs_hidden_source=str(cfg["expected_obs_hidden_source"]),
    )

    preprocess_config = _make_preprocess_config(cfg)

    env_cfg_kwargs: dict[str, Any] = dict(
        task_suite_name=task_suite_name,
        task_id=0,
        resolution=resolution,
        full_record=True,
        init_state_sampling="sequential",
        action_input="raw",
        pixel_rotate_180=False,
        vla_rotate_180=True,
    )
    env_cfg = DreamerVLAOnlineTrainEnvConfig(**env_cfg_kwargs)
    with DreamerVLAOnlineTrainEnv(env_cfg) as _tmp_env:
        num_tasks = _tmp_env.num_tasks

    task_ids = _resolve_task_ids(cfg["task_ids"], num_tasks)
    # Resume-aware work list: episode_id is the init_state selector. Complete
    # reward/hidden pairs are skipped; missing or incomplete ids below the per-task
    # target are re-collected into new shards.
    complete_ids = complete_episode_ids_per_task(reward_dir, hidden_dir)
    full_work = _build_resume_work_list(task_ids, episodes_per_task, complete_ids)
    my_work = _shard_work(full_work, rank, world_size)

    resume_note = (
        f" resume_complete={{{', '.join(f'{k}: {len(v)}' for k, v in sorted(complete_ids.items()))}}}"
        if complete_ids
        else ""
    )
    print(
        f"[collector rank={rank}] task_suite={task_suite_name} "
        f"total_work={len(full_work)} my_work={len(my_work)} shard={shard_name}{resume_note}",
        flush=True,
    )
    if not my_work:
        print(f"[collector rank={rank}] No work assigned. Exiting.", flush=True)
        return 0

    # Print WHERE the collected data lands before the progress bar starts (rank 0).
    # progress_dir (shared across ranks via the launcher's fixed out_dir) lets rank 0
    # render ONE aggregated bar over all ranks; absent it, each rank shows its own bar.
    progress_dir = cfg.get("progress_dir")
    if rank == 0:
        print("[collector] collected data ->", flush=True)
        print(f"  reward : {reward_dir}", flush=True)
        print(f"  hidden : {hidden_dir}", flush=True)

    t_collect_start = time.time()
    if envs_per_gpu > 1:
        demo_index = _collect_vectorized_path(
            policy=policy,
            extractor=extractor,
            unnorm_key=unnorm_key,
            env_cfg_kwargs=env_cfg_kwargs,
            num_envs=envs_per_gpu,
            my_work=my_work,
            episode_horizon=episode_horizon,
            reward_dir=reward_dir,
            hidden_dir=hidden_dir,
            shard_name=shard_name,
            shard_prefix=shard_prefix,
            shard_idx=shard_idx,
            demos_per_shard=demos_per_shard,
            preprocess_config=preprocess_config,
            task_suite_name=task_suite_name,
            rank=rank,
            world_size=world_size,
            progress_dir=progress_dir,
            history=history,
            rotate_images_180=rotate_images_180,
            image_keys=image_keys,
            obs_hidden_source=str(cfg["expected_obs_hidden_source"]),
            on_episode=on_episode,
        )
    else:
        demo_index = 0
        with DreamerVLAOnlineTrainEnv(env_cfg) as env:
            env.set_task(my_work[0][0])
            data_attrs: dict[str, Any] = {
                "task_suite_name": task_suite_name,
                "env_name": env.task_description,
            }
            with _make_dump_writer(
                        reward_dir,
                        hidden_dir,
                        shard_name,
                        shard_prefix=shard_prefix,
                        start_index=shard_idx,
                        demos_per_shard=demos_per_shard,
                    ) as writer, \
                    AggregateProgress(
                        len(my_work), "collect", rank=rank, world_size=world_size,
                        progress_dir=progress_dir, unit="ep",
                    ) as pbar:
                current_task_id = -1
                task_description = env.task_description
                for task_id, ep in my_work:
                    if task_id != current_task_id:
                        env.set_task(task_id)
                        current_task_id = task_id
                        task_description = env.task_description
                        if rank == 0:
                            print(
                                f"[collector rank={rank}] task_id={task_id} "
                                f"description={task_description!r}",
                                flush=True,
                            )
                    steps = _run_episode(
                        env=env,
                        extractor=extractor,
                        task_description=task_description,
                        episode_id=ep,
                        episode_horizon=episode_horizon,
                        task_id=task_id,
                        rank=rank,
                        action_steps=int(preprocess_config["chunk_size"]),
                    )
                    # Derive success from sparse_rewards on the terminal step;
                    # this matches the vectorized path and the downstream reader.
                    ep_success = bool(steps[-1]["sparse_rewards"]) if steps else False
                    if on_episode is not None:
                        on_episode(task_id, ep, len(steps), ep_success)
                    writer.write_demo(
                        index=demo_index,
                        steps=steps,
                        preprocess_config=preprocess_config if (demo_index == 0 and rank == 0) else None,
                        data_attrs=data_attrs if demo_index == 0 else None,
                        task_id=task_id,
                        episode_id=ep,
                        task_description=task_description,
                        episode_horizon=episode_horizon,
                    )
                    demo_index += 1
                    pbar.set(demo_index)

    t_collect = time.time() - t_collect_start
    # One concise summary line per rank (so multi-rank collect stays an aggregate, not a
    # per-episode flood); the path/GPU detail is rank-0 only.
    print(
        f"[collector rank={rank}] Done. {demo_index} demos written "
        f"in {t_collect:.1f}s ({t_collect / max(demo_index, 1):.1f}s/demo)",
        flush=True,
    )
    if rank == 0:
        print(f"  reward dir : {reward_dir}", flush=True)
        print(f"  hidden dir : {hidden_dir}", flush=True)
        mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"  GPU {gpu_id} mem: allocated={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB", flush=True)
    return demo_index
