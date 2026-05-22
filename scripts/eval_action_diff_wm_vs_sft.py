"""Side-by-side comparison: SFT actor vs WM-wrapped actor on the same observations.

At each env step:
  real_hidden = encoder(obs)                          # what SFT VLA sees
  latent      = wm.observe_next(real_hidden, prev_state)
  wm_hidden   = wm.actor_input(latent)                # WM-reconstructed approximation
  sft_action  = actor(real_hidden, deterministic)     # what SFT would do
  wm_action   = actor(wm_hidden,  deterministic)      # what WM-wrapped actor does

Step env with SFT action so the trajectory follows the known-good 97%-success path.
At each step log MSE/cosine on hidden and action, plus per-dim diffs on first action.

Usage (single GPU):
  python -m scripts.eval_action_diff_wm_vs_sft \\
    --config <v4b cfg> --world-model-ckpt <v4-B best> --out-dir <out> \\
    --task-ids 0 --num-episodes 3 --episode-horizon 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_online_pi0_action_hidden_dreamervla import (  # noqa: E402
    build_encoder,
    load_world_model_state,
    obs_to_action_hidden,
)
from src.env.train_env import DreamerVLAOnlineTrainEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--world-model-ckpt", required=True)
    p.add_argument("--vla-ckpt-path", default=str(PROJECT_ROOT / "data/ckpts/VLA_model_256/libero_goal"))
    p.add_argument("--encoder-state-ckpt", default="")
    p.add_argument("--task-suite", default="libero_goal")
    p.add_argument("--task-ids", default="0")
    p.add_argument("--num-episodes", type=int, default=3)
    p.add_argument("--episode-horizon", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--action-head-type", default="legacy", choices=["legacy", "pi0_query"])
    p.add_argument("--policy-adapter-type", default="identity")
    p.add_argument("--target-token-id", type=int, default=10004)
    p.add_argument("--rssm-action-scale", default="env")
    p.add_argument(
        "--step-with",
        choices=["sft", "wm"],
        default="sft",
        help="Which action to feed back into env. 'sft' follows the SFT trajectory (97%% baseline path).",
    )
    p.add_argument(
        "--action-strategy",
        choices=["recompute", "chunk_replay"],
        default="chunk_replay",
        help="recompute: call actor each step, take chunk[0]. chunk_replay: call actor once per "
             "chunk_size steps, replay chunk[0..chunk_size-1] sequentially. eval_libero uses replay.",
    )
    p.add_argument("--chunk-size", type=int, default=5, help="Actions per chunk (Pi0 time_horizon=5).")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "action_diff.jsonl"
    summary_path = out_dir / "action_diff_summary.json"

    cfg = OmegaConf.load(args.config)
    cfg.init.vla_ckpt_path = args.vla_ckpt_path
    cfg.init.encoder_state_ckpt = args.encoder_state_ckpt if args.encoder_state_ckpt else None
    cfg.init.world_model_state_ckpt = args.world_model_ckpt
    cfg.algorithm.rssm_action_scale = str(args.rssm_action_scale)
    cfg.policy.adapter_type = str(args.policy_adapter_type)

    print(f"[diff] device={device}  out_dir={out_dir}", flush=True)
    print(f"[diff] wm_ckpt={args.world_model_ckpt}", flush=True)
    print(f"[diff] step_with={args.step_with}", flush=True)

    # ---- world model ----
    world_model = hydra.utils.instantiate(cfg.world_model).to(device=device, dtype=torch.bfloat16)
    load_world_model_state(world_model, args.world_model_ckpt, reset_reward_head=False)
    world_model.eval()

    # ---- policy (SFT VLA output_projection auto-loaded from cfg.init.vla_ckpt_path) ----
    policy = hydra.utils.instantiate(cfg.policy).to(device)
    policy.eval()

    # ---- encoder (RynnVLA, legacy action_head_type) ----
    encoder = build_encoder(args, device)
    encoder.eval()
    processor = encoder._build_processor(device)

    # ---- env ----
    task_ids = tuple(int(item) for item in str(args.task_ids).split(",") if item.strip())
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
        action_head_type=str(args.action_head_type),
    )

    # ---- rollout loop ----
    successes = []
    per_step_stats: list[dict[str, float]] = []
    t0 = time.time()
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    for ep in range(int(args.num_episodes)):
        obs, _info = env.reset(seed=args.seed + ep)
        latent = None
        prev_wm_action: torch.Tensor | None = None
        ep_len = 0
        ep_ret = 0.0
        ep_sft_act_diff_l1: list[float] = []
        ep_hidden_cos: list[float] = []
        ep_hidden_mse: list[float] = []
        succ = False
        frames: list[np.ndarray] = []
        # Chunk-replay buffers: the chunks active for playback this round.
        # Refreshed every chunk_size env steps in chunk_replay mode.
        sft_playback: np.ndarray | None = None  # shape (chunk_size, 7)
        wm_playback: np.ndarray | None = None   # shape (chunk_size, 7)
        chunk_pos = 0
        for t in range(int(args.episode_horizon)):
            real_hidden = obs_to_action_hidden(encoder, processor, obs, device, args.target_token_id)

            is_first = bool(obs.get("is_first", False)) or latent is None
            if is_first:
                latent = world_model({"mode": "encode_latent", "hidden": real_hidden})
            else:
                assert prev_wm_action is not None
                latent = world_model({
                    "mode": "observe_next",
                    "latent": latent,
                    "hidden": real_hidden,
                    "actions": prev_wm_action,
                    "is_first": False,
                })
            wm_hidden = world_model({"mode": "actor_input", "latent": latent}).float()

            # SFT action (actor on real hidden)
            sft_chunk, _, _ = policy({
                "mode": "sample",
                "hidden": real_hidden.float(),
                "deterministic": True,
                "return_chunk": True,
            })
            # WM-wrapped action (actor on WM-reconstructed hidden)
            wm_chunk, _, _ = policy({
                "mode": "sample",
                "hidden": wm_hidden,
                "deterministic": True,
                "return_chunk": True,
            })

            # ---- diff stats (always on fresh chunks for measurement) ----
            sft_chunk_np = sft_chunk.reshape(-1, sft_chunk.shape[-1]).detach().float().cpu().numpy()  # (chunk_size, 7)
            wm_chunk_np = wm_chunk.reshape(-1, wm_chunk.shape[-1]).detach().float().cpu().numpy()
            sft_act = sft_chunk_np[0, :7]
            wm_act = wm_chunk_np[0, :7]
            sft_flat = sft_chunk.reshape(-1).detach().float().cpu().numpy()
            wm_flat = wm_chunk.reshape(-1).detach().float().cpu().numpy()
            real_flat = real_hidden.float().reshape(-1).detach().cpu().numpy()
            wm_h_flat = wm_hidden.reshape(-1).detach().cpu().numpy()
            denom = (np.linalg.norm(real_flat) * np.linalg.norm(wm_h_flat) + 1e-12)
            hidden_cos = float(np.dot(real_flat, wm_h_flat) / denom)
            hidden_mse = float(np.mean((real_flat - wm_h_flat) ** 2))
            chunk_l1 = float(np.mean(np.abs(sft_flat - wm_flat)))
            chunk_cos = float(np.dot(sft_flat, wm_flat) / (np.linalg.norm(sft_flat) * np.linalg.norm(wm_flat) + 1e-12))
            first_act_l1 = float(np.mean(np.abs(sft_act - wm_act)))
            per_dim = (sft_act - wm_act).tolist()

            per_step_stats.append({
                "ep": ep, "t": t,
                "hidden_cos": hidden_cos,
                "hidden_mse": hidden_mse,
                "chunk_l1": chunk_l1,
                "chunk_cos": chunk_cos,
                "first_act_l1": first_act_l1,
                "per_dim": per_dim,
                "sft_act": sft_act.tolist(),
                "wm_act": wm_act.tolist(),
            })
            ep_sft_act_diff_l1.append(first_act_l1)
            ep_hidden_cos.append(hidden_cos)
            ep_hidden_mse.append(hidden_mse)

            # ---- decide env action ----
            if args.action_strategy == "chunk_replay":
                # Refresh chunks at boundary (t % chunk_size == 0)
                if sft_playback is None or chunk_pos >= int(args.chunk_size):
                    sft_playback = sft_chunk_np[: int(args.chunk_size), :7].copy()
                    wm_playback = wm_chunk_np[: int(args.chunk_size), :7].copy()
                    chunk_pos = 0
                step_sft = sft_playback[chunk_pos]
                step_wm = wm_playback[chunk_pos]
                chunk_pos += 1
            else:  # recompute
                step_sft = sft_act
                step_wm = wm_act
            step_act = step_sft if args.step_with == "sft" else step_wm
            try:
                frames.append(env.render_frame(view="third", vla_aligned=False))
            except Exception:
                pass
            next_obs, reward, terminated, truncated, info = env.step(step_act)
            ep_ret += float(reward)
            ep_len += 1
            # For WM update next step, use the env_action that was actually applied (info["wm_action"])
            wm_action_np = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[:7]
            prev_wm_action = torch.from_numpy(wm_action_np).to(device=device, dtype=real_hidden.dtype).unsqueeze(0)
            obs = next_obs
            if bool(terminated) or bool(truncated):
                succ = bool(terminated)
                break

        successes.append(succ)
        print(
            f"[episode] ep={ep} task={task_ids[0]} step_with={args.step_with} "
            f"len={ep_len} return={ep_ret:.3f} success={succ}  "
            f"|  hidden_cos μ={np.mean(ep_hidden_cos):.4f}  "
            f"hidden_mse μ={np.mean(ep_hidden_mse):.4e}  "
            f"first_act_l1 μ={np.mean(ep_sft_act_diff_l1):.4f}",
            flush=True,
        )
        if frames:
            video_path = videos_dir / f"ep{ep:02d}_task{task_ids[0]}_step{args.step_with}_succ{int(succ)}_len{ep_len:03d}.mp4"
            try:
                imageio.mimsave(str(video_path), frames, fps=20)
                print(f"[video] saved {video_path}", flush=True)
            except Exception as e:
                print(f"[video] save failed: {e}", flush=True)

    env.close()

    # ---- dump ----
    with open(log_path, "w") as f:
        for row in per_step_stats:
            f.write(json.dumps(row) + "\n")
    s = {
        "num_episodes": len(successes),
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "hidden_cos_mean": float(np.mean([r["hidden_cos"] for r in per_step_stats])),
        "hidden_cos_p10": float(np.percentile([r["hidden_cos"] for r in per_step_stats], 10)),
        "hidden_mse_mean": float(np.mean([r["hidden_mse"] for r in per_step_stats])),
        "chunk_l1_mean": float(np.mean([r["chunk_l1"] for r in per_step_stats])),
        "chunk_cos_mean": float(np.mean([r["chunk_cos"] for r in per_step_stats])),
        "first_act_l1_mean": float(np.mean([r["first_act_l1"] for r in per_step_stats])),
        "first_act_l1_p90": float(np.percentile([r["first_act_l1"] for r in per_step_stats], 90)),
        "step_with": args.step_with,
        "wm_ckpt": args.world_model_ckpt,
        "wall_seconds": time.time() - t0,
    }
    with open(summary_path, "w") as f:
        json.dump(s, f, indent=2)
    print("===== summary =====", flush=True)
    print(json.dumps(s, indent=2), flush=True)


if __name__ == "__main__":
    main()
