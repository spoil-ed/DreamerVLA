"""Closed-loop imagination eval for WM-only checkpoints.

For each demo + start_step, runs WM forward T-start-1 steps in three modes:
  - env_actions       : teacher-force with true env actions (open-loop fidelity)
  - sft_open_loop     : SFT actor picks action from POSTERIOR latent at each t
  - closed_loop_sft   : SFT actor picks action from IMAGINED latent (true closed-loop)
Compares imagined hidden vs posterior hidden (cos/mse) and imagined reward
vs WM-posterior reward + sparse ground-truth.

Usage:
  python -m dreamer_vla.diagnostics.measure_wm_closed_loop \
    --ckpt <wm_ckpt> \
    --hidden-hdf5-dir <dir> --reward-hdf5-dir <dir> \
    --actor-cfg configs/dreamervla_rynn_dino_wm_actor_critic.yaml \
    --out-json <out.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hidden-hdf5-dir", required=True)
    p.add_argument("--reward-hdf5-dir", required=True)
    p.add_argument(
        "--actor-cfg",
        required=True,
        help="Path to actor-critic yaml that defines `policy:` and `init.vla_ckpt_path`.",
    )
    p.add_argument(
        "--files",
        nargs="*",
        default=[
            "open_the_middle_drawer_of_the_cabinet_demo.hdf5",
            "open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5",
            "push_the_plate_to_the_front_of_the_stove_demo.hdf5",
            "put_the_bowl_on_the_plate_demo.hdf5",
            "put_the_bowl_on_the_stove_demo.hdf5",
        ],
    )
    p.add_argument("--demo-key", default="demo_0")
    p.add_argument("--start-steps", type=int, nargs="*", default=[0, 20, 40, 60])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def slice_dict_latent(
    latent: dict[str, torch.Tensor], t: int
) -> dict[str, torch.Tensor]:
    return {k: v[:, t] if v.ndim >= 2 else v for k, v in latent.items()}


@torch.no_grad()
def imagine_path(
    world_model,
    actor_or_none,
    posterior_latent,
    demo_actions,
    start_step: int,
    T: int,
    mode: str,
):
    """mode in {'env','sft_open_loop','closed_loop'}.
    Returns (imag_hidden [L,D], imag_reward [L]) where L = T - start_step."""
    cur = slice_dict_latent(posterior_latent, start_step)
    hiddens = [cur["hidden"][0].float().cpu()]
    rewards = [float(world_model.reward_from_latent(cur).float().cpu().item())]
    for t in range(start_step, T - 1):
        if mode == "env":
            a = demo_actions[:, t, :].to(dtype=cur["hidden"].dtype)
        elif mode == "sft_open_loop":
            post_slice = slice_dict_latent(posterior_latent, t)
            hidden_in = world_model.actor_input(post_slice).float()
            _, _, extra = actor_or_none(
                {"mode": "sample", "hidden": hidden_in, "deterministic": True}
            )
            a = extra["action_chunk"][:, 0, :].to(dtype=cur["hidden"].dtype)
        elif mode == "closed_loop":
            hidden_in = world_model.actor_input(cur).float()
            _, _, extra = actor_or_none(
                {"mode": "sample", "hidden": hidden_in, "deterministic": True}
            )
            a = extra["action_chunk"][:, 0, :].to(dtype=cur["hidden"].dtype)
        else:
            raise ValueError(mode)
        cur = world_model.predict_next(cur, a)
        hiddens.append(cur["hidden"][0].float().cpu())
        rewards.append(float(world_model.reward_from_latent(cur).float().cpu().item()))
    return torch.stack(hiddens, dim=0), np.array(rewards, dtype=np.float64)


def stats_vs_post(
    imag_hidden: torch.Tensor,
    post_hidden: torch.Tensor,
    imag_reward: np.ndarray,
    sparse: np.ndarray,
) -> dict[str, float]:
    L = imag_hidden.shape[0]
    post = post_hidden[:L]
    cos = F.cosine_similarity(imag_hidden, post, dim=-1)
    mse = ((imag_hidden - post) ** 2).mean(dim=-1)
    fire = int(np.argmax(imag_reward > 0.5)) if (imag_reward > 0.5).any() else -1
    fire_gt = int(np.argmax(sparse[:L] > 0.5)) if (sparse[:L] > 0.5).any() else -1
    # Reward classification metrics against sparse ground truth.
    sp = sparse[:L].astype(np.float64)
    pr = imag_reward.astype(np.float64)
    pred = (pr > 0.5).astype(np.float64)
    acc = float((pred == sp).mean()) if L > 0 else float("nan")
    return {
        "cos_mean": float(cos.mean().item()),
        "cos_min": float(cos.min().item()),
        "cos_last": float(cos[-1].item()),
        "mse_mean": float(mse.mean().item()),
        "mse_last": float(mse[-1].item()),
        "reward_acc": acc,
        "reward_pred_pos_mean": float(pr[sp > 0.5].mean())
        if (sp > 0.5).any()
        else float("nan"),
        "reward_pred_neg_mean": float(pr[sp <= 0.5].mean())
        if (sp <= 0.5).any()
        else float("nan"),
        "reward_last": float(pr[-1]),
        "fire_pred": fire,
        "fire_gt": fire_gt,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    step = ckpt.get("global_step", "NA")
    print(f"[load] ckpt step={step}  model_dim={cfg.world_model.get('model_dim')}")

    wm = hydra.utils.instantiate(cfg.world_model).to(
        device=device, dtype=torch.bfloat16
    )
    missing, unexpected = wm.load_state_dict(ckpt["model"], strict=False)
    print(
        f"[load] wm tensors={len(ckpt['model'])}  missing={len(missing)}  unexpected={len(unexpected)}"
    )
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    ac_cfg = OmegaConf.load(args.actor_cfg)
    actor = hydra.utils.instantiate(ac_cfg.policy).to(device)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False
    print(f"[load] sft actor from {args.actor_cfg}")

    all_records = []
    for fname in args.files:
        h_path = Path(args.hidden_hdf5_dir) / fname
        r_path = Path(args.reward_hdf5_dir) / fname
        with h5py.File(h_path, "r") as fh, h5py.File(r_path, "r") as fr:
            obs_emb = fh["data"][args.demo_key]["obs_embedding"][:]
            actions = fr["data"][args.demo_key]["actions"][:]
            sparse = fr["data"][args.demo_key]["sparse_rewards"][:].astype(np.float32)
        T = int(obs_emb.shape[0])
        max_T = int(getattr(wm, "max_seq_len", T))
        if T > max_T:
            obs_emb, actions, sparse = obs_emb[:max_T], actions[:max_T], sparse[:max_T]
            T = max_T
        obs_t = (
            torch.from_numpy(obs_emb)
            .unsqueeze(0)
            .to(device=device, dtype=torch.bfloat16)
        )
        act_t = (
            torch.from_numpy(actions)
            .unsqueeze(0)
            .to(device=device, dtype=torch.bfloat16)
        )
        is_first = torch.zeros(1, T, dtype=torch.bool, device=device)
        is_first[0, 0] = True

        with torch.no_grad():
            observed = wm.observe_sequence(
                {
                    "obs_embedding": obs_t,
                    "actions": act_t,
                    "is_first": is_first,
                }
            )
        post_latent = observed["latent"]
        post_hidden = post_latent["hidden"][0].float().cpu()  # [T, D]

        for start in args.start_steps:
            if start >= T - 1:
                continue
            rec: dict[str, Any] = {
                "file": fname,
                "start": start,
                "T": T,
                "horizon": T - start - 1,
            }
            for mode in ["env", "sft_open_loop", "closed_loop"]:
                imag_h, imag_r = imagine_path(
                    wm, actor, post_latent, act_t, start, T, mode
                )
                rec[mode] = stats_vs_post(
                    imag_h, post_hidden[start:], imag_r, sparse[start:]
                )
            print(
                f"  {fname}  start={start}  closed_loop cos_mean={rec['closed_loop']['cos_mean']:.4f}  "
                f"cos_min={rec['closed_loop']['cos_min']:.4f}  rew_acc={rec['closed_loop']['reward_acc']:.3f}  "
                f"fire_pred={rec['closed_loop']['fire_pred']}  fire_gt={rec['closed_loop']['fire_gt']}"
            )
            all_records.append(rec)

    out = {
        "ckpt": args.ckpt,
        "ckpt_step": step,
        "model_dim": cfg.world_model.get("model_dim"),
        "files": list(args.files),
        "starts": list(args.start_steps),
        "records": all_records,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[save] {args.out_json}")


if __name__ == "__main__":
    main()
