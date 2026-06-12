#!/usr/bin/env python
"""Standalone deterministic eval for frozen-WM DreamerVLA actor checkpoints.

Loads a checkpoint saved by ``train_frozen_wm_actor_critic.py`` and rolls
each requested LIBERO task for N episodes with the policy in deterministic
mode.  One MP4 per episode is saved under ``<out-dir>/videos`` and a JSON
summary of per-task success counts is written to ``<out-dir>/eval_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from dreamer_vla.envs.train_env import DreamerVLAOnlineTrainEnv  # noqa: E402
from dreamer_vla.runners.online_utils import (  # noqa: E402
    build_encoder,
    load_world_model_state,
    obs_to_action_hidden,
)
from dreamer_vla.utils.paths import checkpoints_path  # noqa: E402
from dreamer_vla.utils.policy_chunk_queue import PolicyChunkActionQueue  # noqa: E402
from dreamer_vla.utils.seed import set_seed  # noqa: E402
from dreamer_vla.utils.torch_utils import freeze_module  # noqa: E402


def load_eval_checkpoint(
    ckpt_path: str | Path,
    *,
    world_model: torch.nn.Module,
    policy: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
) -> tuple[int, int]:
    """Load model weights for evaluation without restoring optimizer state."""
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"eval ckpt not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dicts = payload.get("state_dicts", {})
    modules = {
        "world_model": world_model,
        "policy": policy,
        "critic": critic,
        "target_critic": target_critic,
    }
    for key, module in modules.items():
        state = state_dicts.get(key)
        if state is None:
            continue
        missing, unexpected = module.load_state_dict(state, strict=True)
        if missing or unexpected:
            print(
                f"[eval] {key} missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
    env_step = int(payload.get("env_step", 0))
    update_step = int(payload.get("update_step", 0))
    print(
        f"[eval] loaded model weights from {path} env_step={env_step} update_step={update_step}",
        flush=True,
    )
    return env_step, update_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic LIBERO eval for frozen-WM actor"
    )
    p.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT / "configs/dreamervla_rynn_dino_wm_actor_critic.yaml"
        ),
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--resume-ckpt",
        default=None,
        help="Trained actor-critic checkpoint; if omitted, eval SFT-init policy from cfg.init.",
    )
    p.add_argument("--world-model-ckpt", required=True)
    p.add_argument(
        "--vla-ckpt-path",
        default=str(checkpoints_path("VLA_model_256", "libero_goal")),
    )
    p.add_argument(
        "--encoder-state-ckpt",
        default="",
    )
    p.add_argument(
        "--action-head-type", default="legacy", choices=["legacy"]
    )
    p.add_argument("--task-suite", default="libero_goal")
    p.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--episodes-per-task", type=int, default=10)
    p.add_argument("--episode-horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-token-id", type=int, default=10004)
    p.add_argument("--rssm-action-scale", default="env")
    p.add_argument("--imagination-horizon", type=int, default=10)
    p.add_argument(
        "--collect-chunk-steps",
        type=int,
        default=0,
        help="Number of sampled chunk actions to execute before resampling; <=0 uses the full chunk.",
    )
    p.add_argument("--bc-to-vla", type=float, default=0.0)
    p.add_argument("--policy-adapter-type", default="identity")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-frame-key", default="third_image")
    p.add_argument("--save-video", action="store_true", default=True)
    p.add_argument("--no-save-video", dest="save_video", action="store_false")
    return p.parse_args()


def _capture_frame(obs: dict[str, Any], key: str) -> np.ndarray:
    if key in obs:
        frame = obs[key]
    elif "agentview_rgb" in obs:
        frame = obs["agentview_rgb"]
    else:
        raise KeyError(f"obs missing video frame key {key!r} / 'agentview_rgb'")
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"video frame must be HWC uint8 RGB, got {tuple(arr.shape)}")
    return np.ascontiguousarray(arr)


def _save_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.init.vla_ckpt_path = args.vla_ckpt_path
    cfg.init.encoder_state_ckpt = args.encoder_state_ckpt
    cfg.init.world_model_state_ckpt = args.world_model_ckpt
    cfg.training.out_dir = str(out_dir)
    cfg.training.distributed_strategy = "single"
    cfg.algorithm.imagination_horizon = int(args.imagination_horizon)
    cfg.algorithm.rssm_action_scale = str(args.rssm_action_scale)
    cfg.algorithm.repval_loss = False
    cfg.algorithm.repval_scale = 0.0
    cfg.algorithm.actor_bc_to_vla_scale = float(args.bc_to_vla)
    if args.policy_adapter_type is not None:
        cfg.policy.adapter_type = str(args.policy_adapter_type)
    OmegaConf.save(cfg, out_dir / "eval_config.yaml", resolve=True)

    print(f"[eval] out_dir={out_dir}", flush=True)
    print(f"[eval] resume_ckpt={args.resume_ckpt}", flush=True)
    print(
        f"[eval] task_ids={args.task_ids} episodes_per_task={args.episodes_per_task} horizon={args.episode_horizon}",
        flush=True,
    )

    world_model = hydra.utils.instantiate(cfg.world_model).to(
        device=device, dtype=torch.bfloat16
    )
    load_world_model_state(world_model, args.world_model_ckpt, reset_reward_head=False)
    freeze_module(world_model)
    world_model.eval()

    policy = hydra.utils.instantiate(cfg.policy).to(device)
    critic = hydra.utils.instantiate(cfg.critic).to(device)
    target_critic = hydra.utils.instantiate(cfg.critic).to(device)
    target_critic.load_state_dict(critic.state_dict())
    freeze_module(target_critic)

    if args.resume_ckpt:
        load_eval_checkpoint(
            args.resume_ckpt,
            world_model=world_model,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
        )
        print(f"[eval] loaded resume_ckpt={args.resume_ckpt}", flush=True)
    else:
        print(
            "[eval] no --resume-ckpt: using SFT-init policy from cfg.init", flush=True
        )
    policy.eval()

    encoder = build_encoder(args, device)
    processor = encoder._build_processor(device)

    task_ids = tuple(
        int(item) for item in str(args.task_ids).split(",") if item.strip()
    )
    env = DreamerVLAOnlineTrainEnv(
        task_suite_name=args.task_suite,
        task_id=task_ids[0],
        task_ids=task_ids,
        seed=args.seed,
        max_steps=args.episode_horizon,
        action_input="normalized",
        task_sampling="sequential",
        init_state_sampling="sequential",
        history_length=2,
        include_state=True,
        vla_rotate_180=True,
        obs_hidden_source="action_query",
        action_head_type=args.action_head_type,
    )

    per_task: dict[int, dict[str, int]] = {
        tid: {"success": 0, "total": 0} for tid in task_ids
    }
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    episode_idx_global = 0

    for tid in task_ids:
        for ep in range(int(args.episodes_per_task)):
            episode_idx_global += 1
            obs, _info = env.reset(
                task_id=int(tid),
                episode_id=int(ep),
                seed=args.seed + int(tid) * 100 + int(ep),
            )
            latent = None
            prev_wm_action: torch.Tensor | None = None
            action_queue = PolicyChunkActionQueue(
                collect_chunk_steps=int(args.collect_chunk_steps)
            )
            frames: list[np.ndarray] = [_capture_frame(obs, args.video_frame_key)]
            ep_len = 0
            ep_return = 0.0
            ep_wm_reward_sum = 0.0
            ep_wm_reward_steps = 0
            terminated = False
            truncated = False

            for _ in range(int(args.episode_horizon)):
                with torch.no_grad():
                    obs_embedding = obs_to_action_hidden(
                        encoder, processor, obs, device, args.target_token_id
                    )
                    is_first = bool(obs.get("is_first", False)) or latent is None
                    if is_first:
                        latent = world_model(
                            {"mode": "encode_latent", "hidden": obs_embedding}
                        )
                    else:
                        assert prev_wm_action is not None
                        latent = world_model(
                            {
                                "mode": "observe_next",
                                "latent": latent,
                                "hidden": obs_embedding,
                                "actions": prev_wm_action,
                                "is_first": False,
                            }
                        )
                    feat = world_model(
                        {"mode": "actor_input", "latent": latent}
                    ).float()
                    wm_reward_pred = world_model({"mode": "reward", "latent": latent})
                    ep_wm_reward_sum += float(
                        wm_reward_pred.reshape(-1).mean().detach().cpu()
                    )
                    ep_wm_reward_steps += 1
                    policy_action = action_queue.next_action(
                        policy, hidden=feat, deterministic=True
                    )

                next_obs, reward, terminated, truncated, info = env.step(policy_action)
                ep_len += 1
                ep_return += float(reward)
                frames.append(_capture_frame(next_obs, args.video_frame_key))
                wm_action_np = np.asarray(info["wm_action"], dtype=np.float32).reshape(
                    -1
                )[:7]
                prev_wm_action = (
                    torch.from_numpy(wm_action_np)
                    .to(device=device, dtype=obs_embedding.dtype)
                    .unsqueeze(0)
                )
                obs = next_obs
                if terminated or truncated:
                    break

            success = bool(terminated)
            per_task[int(tid)]["success"] += int(success)
            per_task[int(tid)]["total"] += 1

            elapsed = time.time() - t0
            wm_reward_mean = (
                (ep_wm_reward_sum / ep_wm_reward_steps) if ep_wm_reward_steps else 0.0
            )
            row = {
                "task_id": int(tid),
                "episode": int(ep),
                "ep_len": int(ep_len),
                "ep_return": float(ep_return),
                "wm_reward_mean": float(wm_reward_mean),
                "wm_reward_sum": float(ep_wm_reward_sum),
                "success": bool(success),
                "elapsed_sec": float(elapsed),
            }
            rows.append(row)
            print(
                f"[eval] task={int(tid)} ep={int(ep)} len={ep_len:3d} return={ep_return:.3f} "
                f"wm_r={wm_reward_mean:.4f} success={success} "
                f"({episode_idx_global}/{len(task_ids) * int(args.episodes_per_task)})",
                flush=True,
            )

            if args.save_video:
                vid_path = (
                    out_dir
                    / "videos"
                    / f"task={int(tid):02d}_ep={int(ep):02d}_len={ep_len:04d}_success={int(success)}.mp4"
                )
                _save_mp4(frames, vid_path, args.video_fps)

    summary = {
        "ckpt": str(Path(args.resume_ckpt).resolve())
        if args.resume_ckpt
        else "SFT_INIT_NO_RESUME",
        "task_suite": args.task_suite,
        "wm_reward_overall_mean": float(
            sum(r["wm_reward_sum"] for r in rows)
            / max(sum(r["ep_len"] for r in rows), 1)
        ),
        "episodes_per_task": int(args.episodes_per_task),
        "episode_horizon": int(args.episode_horizon),
        "device": str(device),
        "total_elapsed_sec": float(time.time() - t0),
        "per_task": {int(tid): per_task[int(tid)] for tid in task_ids},
        "overall_success": float(
            sum(d["success"] for d in per_task.values())
            / max(sum(d["total"] for d in per_task.values()), 1)
        ),
        "rows": rows,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 60, flush=True)
    print(
        f"[eval] overall success: {summary['overall_success'] * 100:.1f}% "
        f"({sum(d['success'] for d in per_task.values())} / {sum(d['total'] for d in per_task.values())})",
        flush=True,
    )
    for tid in task_ids:
        d = per_task[int(tid)]
        rate = (d["success"] / d["total"]) if d["total"] else 0.0
        print(
            f"  task {int(tid):2d}: {d['success']:2d} / {d['total']:2d}  ({rate * 100:5.1f}%)",
            flush=True,
        )
    print(
        f"[eval] elapsed: {summary['total_elapsed_sec']:.1f}s — summary: {out_dir / 'eval_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
