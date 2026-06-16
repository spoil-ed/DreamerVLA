"""Single-process cold-start rollout collector for DreamerVLA.

Drives a base OFT VLA in LIBERO and dumps reward-dir-compatible HDF5 files
plus obs_embedding sidecars and preprocess_config.json that the offline WM
trainer (BalancedTerminalDataset) can consume zero-change.

Phase 1 — single-process only (num_gpus=1, envs_per_gpu=1).
Parallelism is a later task.

Image rotation contract (explicit):
  - full_record() uses pixel_rotate_180=False (DreamerVLAOnlineTrainEnvConfig
    default) -> stored dump images are in raw camera orientation (NOT rotated).
  - The extractor receives those same raw images and applies rotate_images_180=True
    internally to match the offline sidecar protocol.
  - This matches exactly how test_inline_matches_offline_sidecar feeds frames:
    it reads obs_group[key][t] (raw/un-rotated reward-dir frames) into
    extractor.step() with rotate_images_180=True inside the extractor.

Entry point:
  python -m dreamervla.runners.collect_parallel_rollouts \\
    task_suite_name=libero_goal \\
    task_ids=0 \\
    episodes_per_task=2 \\
    episode_horizon=80 \\
    out_dir=/tmp/rollout_smoke \\
    gpu_id=2
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
        "gpu_id": 0,
        "unnorm_key": "libero_goal_no_noops",
        "model_path": "data/checkpoints/OpenVLA-OFT/libero_goal",
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


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def _load_policy(cfg: dict[str, Any]) -> Any:
    """Load OpenVLAOFTPolicy from checkpoint, matching the consistency gate."""
    from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path

    ensure_openvla_oft_on_path()

    from dreamervla.models.encoder.openvla_oft_policy import OpenVLAOFTPolicy

    gpu_id = int(cfg["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model_path = str(
        Path(cfg["model_path"]).expanduser().resolve()
        if Path(cfg["model_path"]).is_absolute()
        else Path.cwd() / cfg["model_path"]
    )

    device = torch.device("cuda:0")  # after CUDA_VISIBLE_DEVICES is set

    print(f"[collector] Loading OFT policy from {model_path} ...", flush=True)
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

    print(f"[collector] Policy loaded in {time.time()-t0:.1f}s", flush=True)
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
        f"  [episode {episode_id}] task_id={task_id} steps={len(steps)} "
        f"success={success}",
        flush=True,
    )
    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = _parse_args()
    print("[collector] config:", cfg, flush=True)

    # Set up LIBERO environment variables before any imports.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    project_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("DVLA_DATA_ROOT", str(project_root / "data"))
    os.environ.setdefault(
        "LIBERO_CONFIG_PATH", str(project_root / "data" / ".libero")
    )

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

    reward_dir = out_dir / "reward"
    hidden_dir = out_dir / "hidden"
    reward_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    # Load policy once (shared across all tasks/episodes)
    policy = _load_policy(cfg)

    # Build extractor
    extractor = OFTRolloutHiddenExtractor(
        policy,
        image_keys=["agentview_rgb", "eye_in_hand_rgb"],
        history=2,
        rotate_images_180=True,
        center_crop=True,
        unnorm_key=unnorm_key,
    )

    preprocess_config = _make_preprocess_config(cfg)

    # Build env (single-task env; we call set_task per task_id)
    # action_input="raw": predict_action already returns env-scale actions; "raw" passes
    # them through as-is.  "normalized" would unnormalize again, double-scaling actions.
    env_cfg = DreamerVLAOnlineTrainEnvConfig(
        task_suite_name=task_suite_name,
        task_id=0,       # initial; overridden per-task via reset(task_id=...)
        resolution=256,
        full_record=True,
        init_state_sampling="sequential",
        action_input="raw",           # OFT predict_action already returns env-scale
        pixel_rotate_180=False,       # CRITICAL: raw images in dump; extractor rotates internally
        vla_rotate_180=True,
    )
    with DreamerVLAOnlineTrainEnv(env_cfg) as env:
        num_tasks = env.num_tasks

        task_ids = _resolve_task_ids(cfg["task_ids"], num_tasks)
        print(f"[collector] task_suite={task_suite_name} task_ids={task_ids} "
              f"episodes_per_task={episodes_per_task} horizon={episode_horizon}", flush=True)

        shard_name = "shard_000.hdf5"
        demo_index = 0

        with RolloutDumpWriter(reward_dir, hidden_dir, shard_name) as writer:
            for task_id in task_ids:
                env.set_task(task_id)
                task_description = env.task_description

                # data_attrs: constructed after set_task so env_name reflects the actual task.
                if demo_index == 0:
                    data_attrs: dict[str, Any] = {
                        "task_suite_name": task_suite_name,
                        "env_name": env.task_description,
                    }

                print(
                    f"[collector] task_id={task_id} description={task_description!r}",
                    flush=True,
                )
                for ep in range(episodes_per_task):
                    steps = _run_episode(
                        env=env,
                        extractor=extractor,
                        task_description=task_description,
                        episode_id=ep,
                        episode_horizon=episode_horizon,
                        task_id=task_id,
                    )
                    writer.write_demo(
                        index=demo_index,
                        steps=steps,
                        preprocess_config=preprocess_config if demo_index == 0 else None,
                        data_attrs=data_attrs if demo_index == 0 else None,
                    )
                    demo_index += 1

        print(f"\n[collector] Done. {demo_index} demos written to {out_dir}", flush=True)
        print(f"  reward dir : {reward_dir}", flush=True)
        print(f"  hidden dir : {hidden_dir}", flush=True)


if __name__ == "__main__":
    main()
