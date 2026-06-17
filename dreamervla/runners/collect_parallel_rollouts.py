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
    experiment=collect_rollouts_onetraj task=OpenVLA_Onetraj_ColdStart_LIBERO \\
    collect.task_ids=all collect.episodes_per_task=2 collect.episode_horizon=64
  (or: bash scripts/run_collect_rollouts.sh ...)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def _load_policy(cfg: dict[str, Any], gpu_id: int) -> Any:
    """Load OpenVLAOFTPolicy from checkpoint on the specified GPU.

    Under torchrun, gpu_id = LOCAL_RANK; CUDA_VISIBLE_DEVICES is NOT
    overridden (torchrun sets it per-process already via LOCAL_RANK env).
    """
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()

    from dreamervla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy

    model_path = _resolve_model_path(cfg["model_path"])

    device = torch.device(f"cuda:{gpu_id}")

    # Auto-detect head mode (l1 vs discrete) from the checkpoint, mirroring the offline
    # preprocess (resolve_oft_policy_mode).  The one-trajectory cold-start ckpt is DISCRETE
    # (no action_head -> actions decoded from LM logits), and discrete implies no proprio.
    from dreamervla.preprocess.preprocess_oft_action_hidden import resolve_oft_policy_mode

    mode = resolve_oft_policy_mode(model_path, str(cfg["policy_mode"]))
    use_l1 = mode == "l1"
    use_proprio = use_l1
    cfg["_policy_mode"] = mode
    cfg["_use_proprio"] = use_proprio

    print(
        f"[collector rank={cfg['_rank']}] Loading OFT policy ({mode}) from {model_path} on {device} ...",
        flush=True,
    )
    t0 = time.time()
    policy = OpenVLAOFTPolicy(
        model_path=model_path,
        component_ckpt_dir=model_path,
        torch_dtype="bf16",
        num_images_in_input=int(cfg["num_images_in_input"]),
        use_lora=False,
        use_l1_regression=use_l1,
        use_diffusion=False,
        use_proprio=use_proprio,
        use_film=False,
        freeze_vla_backbone=True,
    )
    policy.eval()
    policy.to(device)

    # Load LIBERO-specific norm_stats from dataset_statistics.json
    stats_path = Path(model_path) / "dataset_statistics.json"
    with stats_path.open() as fh:
        policy.vla.norm_stats = json.load(fh)

    unnorm_key = cfg["unnorm_key"]
    assert unnorm_key in policy.vla.norm_stats, (
        f"{unnorm_key!r} not in norm_stats; found: {list(policy.vla.norm_stats)}"
    )

    # Cast proprio_projector to bfloat16 (matches sidecar-generation runtime)
    if policy.proprio_projector is not None:
        policy.proprio_projector.to(dtype=torch.bfloat16)

    print(f"[collector rank={cfg['_rank']}] Policy loaded in {time.time() - t0:.1f}s", flush=True)
    return policy


# ---------------------------------------------------------------------------
# Preprocess config
# ---------------------------------------------------------------------------

def _resolve_model_path(model_path: str) -> str:
    """Absolute path for a checkpoint dir; relative paths resolve against cwd."""
    p = Path(model_path)
    return str(p.expanduser().resolve() if p.is_absolute() else Path.cwd() / model_path)


def _make_preprocess_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build preprocess_config.json from cfg (no hardcoded extraction defaults).

    Matches BalancedTerminalDataset._validate_hidden_sidecar.  action_head_type
    and include_state reflect the DETECTED policy mode (cfg["_policy_mode"] /
    cfg["_use_proprio"], set by _load_policy and asserted == task expectation by
    _assert_policy_mode_matches); the remaining fields come straight from cfg.
    """
    mode = cfg["_policy_mode"]
    use_proprio = cfg["_use_proprio"]
    return {
        "action_head_type": "oft_l1_regression" if mode == "l1" else "oft_discrete_token",
        "obs_hidden_source": cfg["expected_obs_hidden_source"],
        "prompt_style": cfg["expected_prompt_style"],
        "history": int(cfg["expected_history"]),
        "include_state": bool(use_proprio),
        "rotate_images_180": bool(cfg["expected_rotate_images_180"]),
        "time_horizon": int(cfg["time_horizon"]),
        "token_dim": int(cfg["token_dim"]),
        "action_dim": int(cfg["action_dim"]),
        "num_images_in_input": int(cfg["num_images_in_input"]),
        "chunk_size": int(cfg["chunk_size"]),
        "hidden_key": "obs_embedding",
        "resolution": int(cfg["resolution"]),
        "model_path": _resolve_model_path(cfg["model_path"]),
        "unnorm_key": cfg["unnorm_key"],
        "center_crop": True,
        "task_suite_name": cfg["task_suite_name"],
    }


def _assert_policy_mode_matches(cfg: dict[str, Any]) -> None:
    """Early validation: ckpt-detected mode == task expected_* (RLinf-style)."""
    detected_head = (
        "oft_l1_regression" if cfg["_policy_mode"] == "l1" else "oft_discrete_token"
    )
    if detected_head != cfg["expected_action_head_type"]:
        raise ValueError(
            f"Detected OFT head {detected_head!r} from ckpt {cfg['model_path']!r} "
            f"!= task expected_action_head_type {cfg['expected_action_head_type']!r}. "
            "Point the cold-start task at a checkpoint matching the WM's expected head."
        )
    if bool(cfg["_use_proprio"]) != bool(cfg["expected_include_state"]):
        raise ValueError(
            f"Detected proprio={cfg['_use_proprio']!r} != task expected_include_state "
            f"{cfg['expected_include_state']!r} for ckpt {cfg['model_path']!r}."
        )


_REQUIRED_COLLECT_KEYS: tuple[str, ...] = (
    "model_path", "policy_mode", "unnorm_key", "task_suite_name", "task_ids",
    "episodes_per_task", "episode_horizon", "envs_per_gpu", "reward_dir", "hidden_dir",
    "image_keys", "expected_history", "num_images_in_input", "expected_action_head_type",
    "expected_include_state", "expected_obs_hidden_source", "expected_prompt_style",
    "expected_rotate_images_180", "time_horizon", "token_dim", "action_dim",
    "chunk_size", "resolution", "gpu_id",
)


def _require_keys(cfg: dict[str, Any]) -> None:
    """Fail fast (before any GPU/model work) if a required extraction key is absent."""
    missing = [k for k in _REQUIRED_COLLECT_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"collect_rollouts cfg missing required keys: {missing}")


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
    obs, _info = env.reset(episode_id=episode_id, task_id=task_id)
    extractor.reset()

    # Capture the post-reset full record (t=0): states == init_state per contract.
    rec0 = env.full_record()

    # Build the 8-dim proprio state for the extractor from full_record fields.
    # Layout: ee_pos(3) + ee_ori/axisangle(3) + gripper_states(2)
    # (matches env._format_obs "state" computation)
    def _proprio_from_rec(rec: dict[str, Any]) -> np.ndarray:
        return np.concatenate([
            rec["ee_pos"].astype(np.float32),
            rec["ee_ori"].astype(np.float32),     # already axisangle from full_record
            rec["gripper_states"].astype(np.float32),
        ]).astype(np.float32)

    # First step: use t=0 full_record for obs_embedding at step 0 (post-reset).
    extractor_obs0 = {
        "agentview_rgb": rec0["agentview_rgb"],     # raw, extractor rotates internally
        "eye_in_hand_rgb": rec0["eye_in_hand_rgb"],
        "state": _proprio_from_rec(rec0),
    }
    action_chunk0, flat_hidden0 = extractor.step(extractor_obs0, task_description)

    steps: list[dict[str, Any]] = []
    success = False
    done = False

    # Execute first action (from the t=0 inference)
    action = action_chunk0[0]
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    success = bool(info.get("success", terminated))

    # Record step 0: use rec0 obs fields + flat_hidden0
    steps.append({
        "actions": np.asarray(info.get("wm_action", info.get("env_action", action)), dtype=np.float64),
        "rewards": np.float32(0.0),
        "sparse_rewards": np.uint8(0),  # filled in post-episode
        "dones": np.uint8(0),           # filled in post-episode
        "robot_states": rec0["robot_states"].astype(np.float64),
        "states": rec0["states"].astype(np.float64),
        "obs": {
            "agentview_rgb": rec0["agentview_rgb"],
            "eye_in_hand_rgb": rec0["eye_in_hand_rgb"],
            "ee_pos": rec0["ee_pos"].astype(np.float64),
            "ee_ori": rec0["ee_ori"].astype(np.float64),
            "ee_states": rec0["ee_states"].astype(np.float64),
            "gripper_states": rec0["gripper_states"].astype(np.float64),
            "joint_states": rec0["joint_states"].astype(np.float64),
        },
        "obs_embedding": flat_hidden0.numpy(),  # (229376,) float16
    })

    # Remaining steps
    t = 1
    while not done and t < episode_horizon:
        # Get full_record for the current observation (post-step)
        rec = env.full_record()

        extractor_obs = {
            "agentview_rgb": rec["agentview_rgb"],      # raw, extractor rotates internally
            "eye_in_hand_rgb": rec["eye_in_hand_rgb"],
            "state": _proprio_from_rec(rec),
        }
        action_chunk, flat_hidden = extractor.step(extractor_obs, task_description)
        action = action_chunk[0]  # receding-horizon closed-loop: execute ONE action

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        success = bool(info.get("success", terminated))

        steps.append({
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
            "obs_embedding": flat_hidden.numpy(),
        })
        t += 1

    # Post-episode: set dones and sparse_rewards on terminal step.
    # sparse_rewards[T-1] = 1 iff success (per collector convention).
    # dones[T-1] = 1 always (episode ended).
    steps[-1]["dones"] = np.uint8(1)
    steps[-1]["sparse_rewards"] = np.uint8(1 if success else 0)

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
    preprocess_config: dict[str, Any],
    task_suite_name: str,
    rank: int,
    history: int,
    rotate_images_180: bool,
    image_keys: list[str],
) -> int:
    """Drive K env subprocesses with batched VLA inference; returns demos written.

    K independent envs step in parallel (VecRolloutEnv, send-all/recv-all) while the K
    observations are batched through one VLA forward (batched_forward).  Work is batched
    per task so every batch shares a prompt length.  See the migration spec §5.
    """
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
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
            )
            for _ in range(num_envs - 1)
        ]

        first_desc = vec_env.set_task([my_work[0][0]], env_ids=[0])[0]
        data_attrs = {"task_suite_name": task_suite_name, "env_name": first_desc}

        # Build the batched decoder ONCE (resolves model handles / head mode a single time).
        decoder = OFTBatchedDecoder(policy, unnorm_key)

        def infer_fn(preps: list[dict[str, Any]]) -> list[tuple[list[Any], Any]]:
            return decoder.predict_batch(preps)

        with RolloutDumpWriter(reward_dir, hidden_dir, shard_name) as writer:
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
                on_episode=lambda tid, ep, ns, ok: print(
                    f"  [rank={rank} vec] task={tid} ep={ep} steps={ns} success={ok}",
                    flush=True,
                ),
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
) -> int:
    """Collect rollouts for this rank's work-slice; returns demos written.

    All extraction parameters come from ``cfg`` (no hardcoded defaults); see
    _REQUIRED_COLLECT_KEYS.  Layer-1 sharding uses (rank, world_size); Layer-2
    within-rank K-env batching is enabled by cfg["envs_per_gpu"] > 1.
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
    project_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("DVLA_DATA_ROOT", str(project_root / "data"))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(project_root / "data" / ".libero"))

    torch.cuda.set_device(gpu_id)
    torch.cuda.set_per_process_memory_fraction(0.8, device=gpu_id)

    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
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

    shard_name = f"r{rank}_shard_000.hdf5" if is_distributed else "shard_000.hdf5"

    policy = _load_policy(cfg, gpu_id)
    _assert_policy_mode_matches(cfg)
    extractor = OFTRolloutHiddenExtractor(
        policy,
        image_keys=image_keys,
        history=history,
        rotate_images_180=rotate_images_180,
        center_crop=True,
        unnorm_key=unnorm_key,
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
    full_work: list[tuple[int, int]] = [
        (tid, ep) for tid in task_ids for ep in range(episodes_per_task)
    ]
    my_work = _shard_work(full_work, rank, world_size)

    print(
        f"[collector rank={rank}] task_suite={task_suite_name} "
        f"total_work={len(full_work)} my_work={len(my_work)} shard={shard_name}",
        flush=True,
    )
    if not my_work:
        print(f"[collector rank={rank}] No work assigned. Exiting.", flush=True)
        return 0

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
            preprocess_config=preprocess_config,
            task_suite_name=task_suite_name,
            rank=rank,
            history=history,
            rotate_images_180=rotate_images_180,
            image_keys=image_keys,
        )
    else:
        demo_index = 0
        with DreamerVLAOnlineTrainEnv(env_cfg) as env:
            env.set_task(my_work[0][0])
            data_attrs: dict[str, Any] = {
                "task_suite_name": task_suite_name,
                "env_name": env.task_description,
            }
            with RolloutDumpWriter(reward_dir, hidden_dir, shard_name) as writer:
                current_task_id = -1
                task_description = env.task_description
                for task_id, ep in my_work:
                    if task_id != current_task_id:
                        env.set_task(task_id)
                        current_task_id = task_id
                        task_description = env.task_description
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
                    )
                    # episode_success is intentionally omitted here: the single-env
                    # _run_episode does not surface a success flag. Downstream
                    # (offline_seed / WMPOAlignedLatentDataset) derives success from
                    # sparse_rewards, so it stays recoverable. The vectorized path,
                    # which has `success` in scope, passes episode_success explicitly.
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

    t_collect = time.time() - t_collect_start
    print(
        f"\n[collector rank={rank}] Done. {demo_index} demos written "
        f"in {t_collect:.1f}s ({t_collect / max(demo_index, 1):.1f}s/demo)",
        flush=True,
    )
    print(f"  shard      : {shard_name}", flush=True)
    print(f"  reward dir : {reward_dir}", flush=True)
    print(f"  hidden dir : {hidden_dir}", flush=True)
    mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
    print(f"  GPU {gpu_id} mem: allocated={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB", flush=True)
    return demo_index
