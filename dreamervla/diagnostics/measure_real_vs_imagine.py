# ruff: noqa: E402
"""
THE correct fidelity test: do (actor in real LIBERO env) and (actor in WM imagine)
diverge?

For one task/episode:
  1. Reset env, get obs_0
  2. WM.encode_latent(obs_0_emb) → start_latent_0
  3. PATH A (real env):  loop actor + env.step → collect (real_action[t], real_reward[t], real_obs_emb[t])
  4. PATH B (imagine):   loop actor + WM.predict_next from start_latent_0
                         → collect (imag_action[t], imag_reward[t], recon_obs_emb[t])
  5. Compare per step: cos(real_emb, recon_emb), MAE(real_a, imag_a), real_r vs imag_r
     Compare terminal: success in real vs fire in imagine

Run on a single task/episode (small footprint), can be extended to N episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from dreamervla.envs.train_env import DreamerVLAOnlineTrainEnv
from dreamervla.models.world_model.dreamerv3_torch import DreamerV3LatentState
from dreamervla.runners.online_utils import (
    build_encoder,
    load_world_model_state,
    obs_to_action_hidden,
)
from dreamervla.utils.paths import checkpoints_path
from dreamervla.utils.seed import set_seed
from dreamervla.utils.torch_utils import freeze_module


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/dreamervla/rynnvla_wmpo_outcome.yaml"),
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Trained ckpt (provides cfg, WM state, policy state).",
    )
    p.add_argument(
        "--world-model-ckpt",
        required=True,
        help="Base WM ckpt used for cfg; the trained WM/policy in --ckpt override.",
    )
    p.add_argument(
        "--vla-ckpt-path",
        default=str(checkpoints_path("VLA_model_256", "libero_goal")),
    )
    p.add_argument(
        "--encoder-state-ckpt",
        default="",
    )
    p.add_argument("--task-suite", default="libero_goal")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episode-id", type=int, default=0)
    p.add_argument("--episode-horizon", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-token-id", type=int, default=10004)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--use-sft-init",
        action="store_true",
        help="Run with fresh SFT-init policy instead of trained policy in --ckpt.",
    )
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)
    cfg.init.vla_ckpt_path = args.vla_ckpt_path
    cfg.init.encoder_state_ckpt = args.encoder_state_ckpt
    cfg.init.world_model_state_ckpt = args.world_model_ckpt
    cfg.algorithm.rssm_action_scale = "env"
    cfg.policy.adapter_type = "identity"

    # Pull reward_head_type from ckpt cfg if present (binary vs twohot)
    print("[load] reading ckpt cfg first ...", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_cfg = OmegaConf.create(ckpt["cfg"])
    rwh = OmegaConf.select(ckpt_cfg, "world_model.reward_head_type", default=None)
    if rwh is not None:
        cfg.world_model.reward_head_type = rwh
        print(f"[load] reward_head_type overridden to {rwh}", flush=True)

    print("[load] building WM + policy ...", flush=True)
    world_model = hydra.utils.instantiate(cfg.world_model).to(
        device=device, dtype=torch.bfloat16
    )
    # Skip loading base WM ckpt's reward head (shape may differ from current head)
    load_world_model_state(
        world_model, args.world_model_ckpt, reset_reward_head=(rwh == "binary")
    )
    # Override WM with trained state from --ckpt
    world_model.load_state_dict(ckpt["state_dicts"]["world_model"], strict=False)
    freeze_module(world_model)
    world_model.eval()

    policy = hydra.utils.instantiate(cfg.policy).to(device)
    if not args.use_sft_init:
        policy.load_state_dict(ckpt["state_dicts"]["policy"], strict=True)
        print("[load] using TRAINED policy", flush=True)
    else:
        print("[load] using SFT-init policy (no RL training)", flush=True)
    freeze_module(policy)
    policy.eval()

    encoder = build_encoder(args, device)
    processor = encoder._build_processor(device)

    env = DreamerVLAOnlineTrainEnv(
        task_suite_name=args.task_suite,
        task_id=int(args.task_id),
        task_ids=(int(args.task_id),),
        seed=args.seed,
        max_steps=int(args.episode_horizon),
        action_input="normalized",
        task_sampling="sequential",
        init_state_sampling="sequential",
        history_length=2,
        include_state=True,
        vla_rotate_180=True,
        obs_hidden_source="action_query",
        action_head_type="legacy",
    )

    obs, _ = env.reset(
        task_id=int(args.task_id),
        episode_id=int(args.episode_id),
        seed=int(args.seed) + int(args.task_id) * 100 + int(args.episode_id),
    )

    # ── Storage ─────────────────────────────────────────────────────────────
    real_actions, real_rewards, real_obs_embs = [], [], []
    real_term, _ = False, False

    # ── PHASE 1: Real env rollout + record initial latent ──────────────────
    print(f"[real] task={args.task_id} ep={args.episode_id}", flush=True)
    start_latent: DreamerV3LatentState | None = None
    real_latents = []  # latents from real-env posterior (observe_next)
    prev_wm_action = None
    real_imag_rewards = []
    cur_latent: DreamerV3LatentState | None = None

    for _t in range(int(args.episode_horizon)):
        with torch.no_grad():
            obs_emb = obs_to_action_hidden(
                encoder, processor, obs, device, args.target_token_id
            )
            real_obs_embs.append(obs_emb.detach().float().cpu())
            is_first = bool(obs.get("is_first", False)) or cur_latent is None
            if is_first:
                cur_latent = world_model({"mode": "encode_latent", "hidden": obs_emb})
                if start_latent is None:
                    start_latent = cur_latent
            else:
                cur_latent = world_model(
                    {
                        "mode": "observe_next",
                        "latent": cur_latent,
                        "hidden": obs_emb,
                        "actions": prev_wm_action,
                        "is_first": False,
                    }
                )
            real_latents.append(cur_latent)
            real_imag_rewards.append(
                float(world_model.state_reward(cur_latent).float().cpu().item())
            )

            feat = world_model({"mode": "actor_input", "latent": cur_latent}).float()
            ach, _, _ = policy(
                {
                    "mode": "sample",
                    "hidden": feat,
                    "deterministic": True,
                    "return_chunk": True,
                }
            )
            policy_action = (
                ach.reshape(-1, ach.shape[-1])[0, :7].detach().cpu().float().numpy()
            )
        next_obs, reward, terminated, truncated, info = env.step(policy_action)
        real_actions.append(policy_action.tolist())
        real_rewards.append(float(reward))
        wm_a = np.asarray(info["wm_action"], dtype=np.float32).reshape(-1)[:7]
        prev_wm_action = (
            torch.from_numpy(wm_a).to(device=device, dtype=obs_emb.dtype).unsqueeze(0)
        )
        obs = next_obs
        if terminated or truncated:
            real_term, _ = terminated, truncated
            break

    T_real = len(real_actions)
    real_success = bool(real_term)
    print(
        f"[real] T={T_real}  success={real_success}  sum_reward={sum(real_rewards):.3f}",
        flush=True,
    )

    # ── PHASE 2: WM imagine from start_latent, same actor, no env ──────────
    print(f"[imag] same actor, WM-only rollout for {T_real} steps", flush=True)
    imag_actions, imag_rewards, recon_obs_embs = [], [], []
    cur = start_latent
    for _t in range(T_real):
        with torch.no_grad():
            # Reward head on this latent (state reward at step t before action)
            r = float(world_model.state_reward(cur).float().cpu().item())
            imag_rewards.append(r)
            # Recon obs_embedding at this step (for cosine vs real)
            recon = world_model.hidden_decoder(cur.feature().to(dtype=torch.bfloat16))
            recon_obs_embs.append(recon.detach().float().cpu().squeeze(0))
            # actor input + action
            feat = world_model.actor_input(cur).float()
            _, _, extra = policy(
                {
                    "mode": "sample",
                    "hidden": feat,
                    "deterministic": True,
                    "return_chunk": True,
                }
            )
            ach = extra["action_chunk"]
            a0 = ach.reshape(-1, ach.shape[-1])[0, :7].float()
            imag_actions.append(a0.detach().cpu().tolist())
            # WM predict_next
            cur = world_model(
                {
                    "mode": "predict_next",
                    "latent": cur,
                    "actions": a0.to(dtype=torch.bfloat16).unsqueeze(0),
                }
            )

    # ── Comparison ──────────────────────────────────────────────────────────
    real_obs_embs_t = torch.stack(real_obs_embs, dim=0).squeeze(1)  # [T, D]
    recon_obs_embs_t = torch.stack(recon_obs_embs, dim=0)  # [T, D]
    real_actions_t = torch.tensor(real_actions, dtype=torch.float32)
    imag_actions_t = torch.tensor(imag_actions, dtype=torch.float32)
    cos_emb = F.cosine_similarity(recon_obs_embs_t, real_obs_embs_t, dim=-1).numpy()
    action_mae_per_step = (real_actions_t - imag_actions_t).abs().mean(-1).numpy()
    action_max_per_step = (real_actions_t - imag_actions_t).abs().max(-1).values.numpy()

    print("\n=========== Real env vs WM imagine (same actor) ===========")
    print(
        f"{'t':>4} {'cos(emb)':>10} {'a_MAE':>8} {'a_max':>8} {'real_r':>8} {'imag_r':>8} {'real_postr':>11}"
    )
    idxs = sorted(
        set(
            [
                0,
                T_real // 8,
                T_real // 4,
                T_real // 2,
                3 * T_real // 4,
                T_real - 10,
                T_real - 5,
                T_real - 1,
            ]
        )
    )
    for t in idxs:
        if t < 0 or t >= T_real:
            continue
        print(
            f"{t:>4} {cos_emb[t]:>10.4f} {action_mae_per_step[t]:>8.4f} {action_max_per_step[t]:>8.4f} {real_rewards[t]:>8.4f} {imag_rewards[t]:>8.4f} {real_imag_rewards[t]:>11.4f}"
        )

    fire_imag = (
        int(np.argmax(np.array(imag_rewards) > 0.5))
        if (np.array(imag_rewards) > 0.5).any()
        else -1
    )
    fire_real_post = (
        int(np.argmax(np.array(real_imag_rewards) > 0.5))
        if (np.array(real_imag_rewards) > 0.5).any()
        else -1
    )
    print(f"\nimag fire@>0.5: step={fire_imag}  (final imag_r={imag_rewards[-1]:.4f})")
    print(
        f"real-posterior fire@>0.5: step={fire_real_post}  (final={real_imag_rewards[-1]:.4f})"
    )
    print(f"real env success={real_success}  (sum_real_reward={sum(real_rewards):.3f})")
    print(
        f"cos(real,recon): mean={cos_emb.mean():.4f}  min={cos_emb.min():.4f}  last10_mean={cos_emb[-10:].mean():.4f}"
    )
    print(
        f"action MAE:       mean={action_mae_per_step.mean():.4f}  max={action_mae_per_step.max():.4f}"
    )

    out = {
        "ckpt": args.ckpt,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "T": int(T_real),
        "real_success": bool(real_success),
        "real_actions": real_actions,
        "real_rewards": real_rewards,
        "imag_actions": imag_actions,
        "imag_rewards": imag_rewards,
        "real_posterior_imag_rewards": real_imag_rewards,
        "cos_real_vs_recon_obs_emb": cos_emb.tolist(),
        "action_mae_per_step": action_mae_per_step.tolist(),
        "fire_imag": fire_imag,
        "fire_real_post": fire_real_post,
    }
    out_json = (
        Path(args.out_json)
        if args.out_json
        else Path(args.ckpt).parent.parent / "real_vs_imagine.json"
    )
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_json}")


if __name__ == "__main__":
    main()
