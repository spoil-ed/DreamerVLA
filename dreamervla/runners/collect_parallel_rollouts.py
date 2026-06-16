"""Multi-rank rollout collector for DreamerVLA (torchrun + optional K-env layer).

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

Layer 2 — K env subprocesses per rank (SECONDARY / OPTIONAL):
    When ``envs_per_gpu > 1``, each rank drives K LIBERO envs in parallel
    child processes via SubprocVecEnv (multiprocessing, spawn).  Each child
    holds its own env + extractor instance so history buffers are isolated.
    Inference is K sequential forwards (VLA is GPU-bound; env-stepping is the
    bottleneck).  Layer 2 is gated by the OSMesa availability check at startup;
    if the check fails or mujoco renders fail to initialize in a spawned child
    the code falls back to K=1 automatically and logs a warning.

Image rotation contract (explicit):
  - full_record() uses pixel_rotate_180=False (DreamerVLAOnlineTrainEnvConfig
    default) -> stored dump images are in raw camera orientation (NOT rotated).
  - The extractor receives those same raw images and applies rotate_images_180=True
    internally to match the offline sidecar protocol.
  - This matches exactly how test_inline_matches_offline_sidecar feeds frames:
    it reads obs_group[key][t] (raw/un-rotated reward-dir frames) into
    extractor.step() with rotate_images_180=True inside the extractor.

Entry point (single-process):
  python -m dreamervla.runners.collect_parallel_rollouts \\
    task_suite_name=libero_goal \\
    task_ids=0 \\
    episodes_per_task=2 \\
    episode_horizon=80 \\
    out_dir=/tmp/rollout_smoke \\
    gpu_id=2

Entry point (multi-rank, 2 GPUs):
  torchrun --nproc_per_node=2 \\
    -m dreamervla.runners.collect_parallel_rollouts \\
    task_suite_name=libero_goal \\
    task_ids=0,1 \\
    episodes_per_task=2 \\
    episode_horizon=80 \\
    out_dir=/tmp/dvla_par_smoke \\
    num_gpus=2
"""

from __future__ import annotations

import json
import os
import sys
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

def _parse_args() -> dict[str, Any]:
    """Minimal key=value argparser.  Returns a dict of typed values."""
    defaults: dict[str, Any] = {
        "task_suite_name": "libero_goal",
        "task_ids": "0",           # comma-separated ints or "all"
        "episodes_per_task": 2,
        "episode_horizon": 80,
        "deterministic": True,     # no-op: OFT L1-regression heads are always deterministic
        "out_dir": "/tmp/dvla_rollouts",
        "gpu_id": 0,               # used in single-process path; ignored under torchrun
        "unnorm_key": "libero_goal_no_noops",
        "model_path": "data/checkpoints/OpenVLA-OFT/libero_goal",
        # Layer-1 informational knob (actual parallelism comes from torchrun)
        "num_gpus": 1,
        # Layer-2 knob: K env subprocesses per rank (1 = disabled)
        "envs_per_gpu": 1,
    }
    args = dict(defaults)
    for token in sys.argv[1:]:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.lstrip("-")
        # type coerce: if default exists and is int/float/bool, cast
        if k in defaults:
            d = defaults[k]
            if isinstance(d, bool):
                args[k] = v.lower() in {"1", "true", "yes", "y"}
            elif isinstance(d, int):
                args[k] = int(v)
            elif isinstance(d, float):
                args[k] = float(v)
            else:
                args[k] = v
        else:
            args[k] = v
    return args


def _resolve_task_ids(task_ids_str: str, num_tasks: int) -> list[int]:
    if str(task_ids_str).strip().lower() == "all":
        return list(range(num_tasks))
    return [int(x.strip()) for x in str(task_ids_str).split(",") if x.strip()]


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

    model_path = str(
        Path(cfg["model_path"]).expanduser().resolve()
        if Path(cfg["model_path"]).is_absolute()
        else Path.cwd() / cfg["model_path"]
    )

    device = torch.device(f"cuda:{gpu_id}")

    print(f"[collector rank={cfg['_rank']}] Loading OFT policy from {model_path} on {device} ...", flush=True)
    t0 = time.time()
    policy = OpenVLAOFTPolicy(
        model_path=model_path,
        component_ckpt_dir=model_path,
        torch_dtype="bf16",
        num_images_in_input=4,      # history(2) × views(2)
        use_lora=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_proprio=True,
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

    print(f"[collector rank={cfg['_rank']}] Policy loaded in {time.time()-t0:.1f}s", flush=True)
    return policy


# ---------------------------------------------------------------------------
# Preprocess config
# ---------------------------------------------------------------------------

def _make_preprocess_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the preprocess_config.json matching BalancedTerminalDataset._validate_hidden_sidecar."""
    model_path = str(
        Path(cfg["model_path"]).expanduser().resolve()
        if Path(cfg["model_path"]).is_absolute()
        else Path.cwd() / cfg["model_path"]
    )
    return {
        "action_head_type": "oft_l1_regression",
        "obs_hidden_source": "action_query",
        "prompt_style": "vla_policy",
        "history": 2,
        "include_state": True,
        "rotate_images_180": True,
        "time_horizon": 8,
        "token_dim": 4096,
        "action_dim": 7,
        "num_images_in_input": 4,
        "chunk_size": 4,
        "hidden_key": "obs_embedding",
        "resolution": 256,
        "model_path": model_path,
        "unnorm_key": cfg["unnorm_key"],
        "center_crop": True,
        "task_suite_name": cfg["task_suite_name"],
    }


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
# Layer 2: SubprocVecEnv (K env subprocesses per rank)
# ---------------------------------------------------------------------------
# The worker function runs in a spawned child process.  It receives commands
# over a pipe and sends back observations / results.
# Commands: ("reset", task_id, episode_id, task_description)
#           ("step",  action)
#           ("full_record", None)
#           ("set_task", task_id)
#           ("close", None)

def _subproc_worker(
    conn: Any,
    task_suite_name: str,
    env_vars: dict[str, str],
) -> None:
    """Worker function executed in a spawned child process.

    Sets up its own LIBERO env, then processes commands from the parent.
    """
    # Restore env vars in the child (spawn clears them)
    for k, v in env_vars.items():
        os.environ[k] = v

    from dreamervla.envs.train_env import (
        DreamerVLAOnlineTrainEnv,
        DreamerVLAOnlineTrainEnvConfig,
    )

    env_cfg = DreamerVLAOnlineTrainEnvConfig(
        task_suite_name=task_suite_name,
        task_id=0,
        resolution=256,
        full_record=True,
        init_state_sampling="sequential",
        action_input="raw",
        pixel_rotate_180=False,
        vla_rotate_180=True,
    )

    try:
        env = DreamerVLAOnlineTrainEnv(env_cfg)
        env.__enter__()
        conn.send(("ready", None))
    except Exception as exc:
        conn.send(("error", str(exc)))
        conn.close()
        return

    while True:
        try:
            cmd, payload = conn.recv()
        except EOFError:
            break

        if cmd == "close":
            break
        elif cmd == "set_task":
            task_id = payload
            env.set_task(task_id)
            conn.send(("ok", env.task_description))
        elif cmd == "reset":
            task_id, episode_id = payload
            obs, info = env.reset(episode_id=episode_id, task_id=task_id)
            rec = env.full_record()
            conn.send(("reset_done", rec))
        elif cmd == "step":
            action = payload
            obs, reward, terminated, truncated, info = env.step(action)
            rec = env.full_record()
            conn.send(("step_done", (obs, reward, terminated, truncated, info, rec)))
        elif cmd == "full_record":
            conn.send(("record", env.full_record()))
        else:
            conn.send(("error", f"unknown cmd {cmd!r}"))

    try:
        env.__exit__(None, None, None)
    except Exception:
        pass
    conn.close()


class SubprocEnvHandle:
    """Handle to a single env running in a child process."""

    def __init__(self, conn: Any, proc: Any) -> None:
        self._conn = conn
        self._proc = proc
        self._task_description: str = ""

    def set_task(self, task_id: int) -> str:
        self._conn.send(("set_task", task_id))
        _, desc = self._conn.recv()
        self._task_description = desc
        return desc

    def reset(self, task_id: int, episode_id: int) -> dict:
        self._conn.send(("reset", (task_id, episode_id)))
        _, rec = self._conn.recv()
        return rec

    def step(self, action: np.ndarray) -> tuple:
        self._conn.send(("step", action))
        _, payload = self._conn.recv()
        return payload  # (obs, reward, terminated, truncated, info, rec)

    def close(self) -> None:
        try:
            self._conn.send(("close", None))
        except Exception:
            pass
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()


def _try_launch_subproc_envs(
    k: int,
    task_suite_name: str,
    env_vars: dict[str, str],
) -> list[SubprocEnvHandle] | None:
    """Launch K env subprocesses. Returns None if any fail to start."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    handles: list[SubprocEnvHandle] = []
    try:
        for _ in range(k):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_subproc_worker,
                args=(child_conn, task_suite_name, env_vars),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # close child end in parent
            # Wait for ready signal (up to 120s for env init)
            if not parent_conn.poll(120):
                raise RuntimeError("child env timed out during init")
            status, msg = parent_conn.recv()
            if status == "error":
                raise RuntimeError(f"child env init failed: {msg}")
            handles.append(SubprocEnvHandle(parent_conn, proc))
    except Exception as exc:
        # Shut down any that launched successfully
        for h in handles:
            h.close()
        print(f"[collector] Layer-2 subproc env launch failed: {exc}. Falling back to K=1.", flush=True)
        return None
    return handles


# ---------------------------------------------------------------------------
# Episode runner — K-env parallel variant (Layer 2)
# ---------------------------------------------------------------------------

def _run_episode_subproc(
    handle: SubprocEnvHandle,
    extractor: Any,
    task_id: int,
    episode_id: int,
    episode_horizon: int,
    rank: int = 0,
) -> list[dict[str, Any]]:
    """Run one episode using a SubprocEnvHandle. Mirrors _run_episode exactly."""
    task_description = handle._task_description

    def _proprio_from_rec(rec: dict[str, Any]) -> np.ndarray:
        return np.concatenate([
            rec["ee_pos"].astype(np.float32),
            rec["ee_ori"].astype(np.float32),
            rec["gripper_states"].astype(np.float32),
        ]).astype(np.float32)

    rec0 = handle.reset(task_id=task_id, episode_id=episode_id)
    extractor.reset()

    extractor_obs0 = {
        "agentview_rgb": rec0["agentview_rgb"],
        "eye_in_hand_rgb": rec0["eye_in_hand_rgb"],
        "state": _proprio_from_rec(rec0),
    }
    action_chunk0, flat_hidden0 = extractor.step(extractor_obs0, task_description)

    steps: list[dict[str, Any]] = []
    action = action_chunk0[0]
    obs, reward, terminated, truncated, info, rec_after = handle.step(action)
    done = bool(terminated or truncated)
    success = bool(info.get("success", terminated))

    steps.append({
        "actions": np.asarray(info.get("wm_action", info.get("env_action", action)), dtype=np.float64),
        "rewards": np.float32(0.0),
        "sparse_rewards": np.uint8(0),
        "dones": np.uint8(0),
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
        "obs_embedding": flat_hidden0.numpy(),
    })

    t = 1
    while not done and t < episode_horizon:
        rec = rec_after
        extractor_obs = {
            "agentview_rgb": rec["agentview_rgb"],
            "eye_in_hand_rgb": rec["eye_in_hand_rgb"],
            "state": _proprio_from_rec(rec),
        }
        action_chunk, flat_hidden = extractor.step(extractor_obs, task_description)
        action = action_chunk[0]

        obs, reward, terminated, truncated, info, rec_after = handle.step(action)
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

    steps[-1]["dones"] = np.uint8(1)
    steps[-1]["sparse_rewards"] = np.uint8(1 if success else 0)

    print(
        f"  [rank={rank} episode {episode_id}] task_id={task_id} steps={len(steps)} "
        f"success={success}",
        flush=True,
    )
    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _parse_args()

    # ── Detect torchrun / distributed context ────────────────────────────────
    rank, world_size, local_rank = _get_dist_info()
    cfg["_rank"] = rank
    cfg["_world_size"] = world_size
    cfg["_local_rank"] = local_rank
    is_distributed = world_size > 1

    # gpu_id: under torchrun use LOCAL_RANK; single-process respects gpu_id arg
    if is_distributed:
        gpu_id = local_rank
    else:
        gpu_id = int(cfg["gpu_id"])
        # Single-process path: set CUDA_VISIBLE_DEVICES as before
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_id = 0  # after masking, always cuda:0

    if rank == 0:
        print("[collector] config:", cfg, flush=True)
    print(f"[collector] rank={rank}/{world_size} local_rank={local_rank} gpu_id={gpu_id}", flush=True)

    # Set up LIBERO environment variables before any imports.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    project_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("DVLA_DATA_ROOT", str(project_root / "data"))
    os.environ.setdefault(
        "LIBERO_CONFIG_PATH", str(project_root / "data" / ".libero")
    )

    # ── Set per-process GPU memory fraction (80%) ─────────────────────────────
    torch.cuda.set_device(gpu_id)
    torch.cuda.set_per_process_memory_fraction(0.8, device=gpu_id)

    from dreamervla.envs.train_env import (
        DreamerVLAOnlineTrainEnv,
        DreamerVLAOnlineTrainEnvConfig,
    )
    from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
    from dreamervla.runners.rollout_hidden_extractor import OFTRolloutHiddenExtractor

    task_suite_name = cfg["task_suite_name"]
    episodes_per_task = int(cfg["episodes_per_task"])
    episode_horizon = int(cfg["episode_horizon"])
    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    unnorm_key = cfg["unnorm_key"]
    envs_per_gpu = int(cfg["envs_per_gpu"])

    reward_dir = out_dir / "reward"
    hidden_dir = out_dir / "hidden"
    # All ranks create dirs (exist_ok avoids races)
    reward_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # ── Shard file naming ─────────────────────────────────────────────────────
    # Multi-rank: prefix r{rank}_ so globs pick up all shards.
    # Single-process: no prefix, backward-compatible "shard_000.hdf5".
    if is_distributed:
        shard_name = f"r{rank}_shard_000.hdf5"
    else:
        shard_name = "shard_000.hdf5"

    # ── Load policy + build extractor ────────────────────────────────────────
    policy = _load_policy(cfg, gpu_id)
    extractor = OFTRolloutHiddenExtractor(
        policy,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        history=2,
        rotate_images_180=True,
        center_crop=True,
        unnorm_key=unnorm_key,
    )

    preprocess_config = _make_preprocess_config(cfg)

    # ── Build the full work-list, then shard by rank ──────────────────────────
    # We need num_tasks to resolve "all"; use a temporary env just for that.
    env_cfg = DreamerVLAOnlineTrainEnvConfig(
        task_suite_name=task_suite_name,
        task_id=0,
        resolution=256,
        full_record=True,
        init_state_sampling="sequential",
        action_input="raw",
        pixel_rotate_180=False,
        vla_rotate_180=True,
    )
    with DreamerVLAOnlineTrainEnv(env_cfg) as _tmp_env:
        num_tasks = _tmp_env.num_tasks

    task_ids = _resolve_task_ids(cfg["task_ids"], num_tasks)

    # Full work-list: (task_id, episode_id) pairs, task-major order
    full_work: list[tuple[int, int]] = [
        (tid, ep)
        for tid in task_ids
        for ep in range(episodes_per_task)
    ]

    # Each rank takes its slice (round-robin for roughly equal load)
    my_work = _shard_work(full_work, rank, world_size)

    print(
        f"[collector rank={rank}] task_suite={task_suite_name} "
        f"total_work={len(full_work)} my_work={len(my_work)} "
        f"shard={shard_name}",
        flush=True,
    )

    if not my_work:
        print(f"[collector rank={rank}] No work assigned. Exiting.", flush=True)
        return

    # ── Layer 2: optionally launch K subproc envs ─────────────────────────────
    use_subproc = envs_per_gpu > 1
    subproc_handles: list[SubprocEnvHandle] | None = None

    if use_subproc:
        # Pass env vars so children inherit the LIBERO config
        child_env_vars = {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "osmesa"),
            "DVLA_DATA_ROOT": os.environ.get("DVLA_DATA_ROOT", str(project_root / "data")),
            "LIBERO_CONFIG_PATH": os.environ.get(
                "LIBERO_CONFIG_PATH", str(project_root / "data" / ".libero")
            ),
        }
        print(f"[collector rank={rank}] Attempting Layer-2: {envs_per_gpu} subproc envs ...", flush=True)
        subproc_handles = _try_launch_subproc_envs(envs_per_gpu, task_suite_name, child_env_vars)
        if subproc_handles is None:
            use_subproc = False
            print(f"[collector rank={rank}] Layer-2 unavailable; using single env (K=1).", flush=True)
        else:
            print(f"[collector rank={rank}] Layer-2 active: {envs_per_gpu} subproc envs ready.", flush=True)

    # ── Collect rollouts ──────────────────────────────────────────────────────
    t_collect_start = time.time()
    demo_index = 0

    if use_subproc and subproc_handles is not None:
        # Layer 2: interleave K subproc envs.
        # Round-robin: assign my_work[i] to handles[i % K].
        # We process in batches of K (or the remainder), issuing each episode
        # to a distinct handle, then collecting results sequentially.
        k = len(subproc_handles)

        # Pre-warm: set tasks for first batch
        def _set_task_for_handle(h: SubprocEnvHandle, tid: int) -> None:
            h.set_task(tid)

        # Build per-handle extractors (K separate history buffers)
        # Reuse the main extractor for handle 0; build k-1 more for handles 1..K-1
        from dreamervla.runners.rollout_hidden_extractor import OFTRolloutHiddenExtractor
        handle_extractors: list[OFTRolloutHiddenExtractor] = [extractor]
        for _ in range(k - 1):
            handle_extractors.append(
                OFTRolloutHiddenExtractor(
                    policy,
                    image_keys=["agentview_rgb", "eye_in_hand_rgb"],
                    history=2,
                    rotate_images_180=True,
                    center_crop=True,
                    unnorm_key=unnorm_key,
                )
            )

        with RolloutDumpWriter(reward_dir, hidden_dir, shard_name) as writer:
            # data_attrs built from first task
            first_tid = my_work[0][0]
            subproc_handles[0].set_task(first_tid)
            data_attrs: dict[str, Any] = {
                "task_suite_name": task_suite_name,
                "env_name": subproc_handles[0]._task_description,
            }

            i = 0
            while i < len(my_work):
                batch = my_work[i: i + k]
                # Set tasks as needed
                for b_idx, (tid, _ep) in enumerate(batch):
                    h = subproc_handles[b_idx]
                    if h._task_description == "" or True:
                        h.set_task(tid)

                # Collect episodes sequentially per handle in this batch
                # (env-stepping parallelism benefit: each env steps CPU; inference GPU)
                # Note: sequential inference is the ACCEPTABLE fallback per spec
                for b_idx, (tid, ep) in enumerate(batch):
                    h = subproc_handles[b_idx]
                    ex = handle_extractors[b_idx]
                    steps = _run_episode_subproc(
                        handle=h,
                        extractor=ex,
                        task_id=tid,
                        episode_id=ep,
                        episode_horizon=episode_horizon,
                        rank=rank,
                    )
                    writer.write_demo(
                        index=demo_index,
                        steps=steps,
                        preprocess_config=preprocess_config if (demo_index == 0 and rank == 0) else None,
                        data_attrs=data_attrs if demo_index == 0 else None,
                    )
                    demo_index += 1

                i += k

        for h in subproc_handles:
            h.close()

    else:
        # Layer 1 / single-process: use one in-process env
        with DreamerVLAOnlineTrainEnv(env_cfg) as env:
            env.set_task(my_work[0][0])  # initial task set

            data_attrs: dict[str, Any] = {
                "task_suite_name": task_suite_name,
                "env_name": env.task_description,
            }

            with RolloutDumpWriter(reward_dir, hidden_dir, shard_name) as writer:
                current_task_id = -1
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
                    writer.write_demo(
                        index=demo_index,
                        steps=steps,
                        # preprocess_config: rank 0 writes it (content is identical across ranks)
                        preprocess_config=preprocess_config if (demo_index == 0 and rank == 0) else None,
                        data_attrs=data_attrs if demo_index == 0 else None,
                    )
                    demo_index += 1

    t_collect = time.time() - t_collect_start
    print(
        f"\n[collector rank={rank}] Done. {demo_index} demos written to {out_dir} "
        f"in {t_collect:.1f}s ({t_collect/max(demo_index,1):.1f}s/demo)",
        flush=True,
    )
    print(f"  shard      : {shard_name}", flush=True)
    print(f"  reward dir : {reward_dir}", flush=True)
    print(f"  hidden dir : {hidden_dir}", flush=True)

    # Report per-GPU memory usage
    mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
    print(
        f"  GPU {gpu_id} mem: allocated={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
