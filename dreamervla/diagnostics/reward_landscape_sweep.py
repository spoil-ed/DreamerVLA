#!/usr/bin/env python
# ruff: noqa: E402
"""Sweep small Gaussian perturbations to the SFT policy's output_projection
weights and measure WM-imagined raw reward at each perturbed point.

Goal: test whether SFT init sits on a local peak in the WM reward landscape.
If the SFT raw reward is higher than every perturbation, REINFORCE will see
"every direction looks worse" and drift via noise.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import argparse
import json

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dreamervla.algorithms.dreamervla import imagine_actor_critic_step
from dreamervla.models.critic.twohot_critic import ReturnPercentileTracker
from dreamervla.runners.frozen_wm_actor_critic import (
    actor_critic_obs,
    build_offline_loader,
)
from dreamervla.utils.optim import build_optimizer
from dreamervla.utils.seed import set_seed
from dreamervla.utils.torch_utils import freeze_module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/dreamervla/rynnvla_wmpo_outcome.yaml"),
    )
    p.add_argument("--world-model-ckpt", required=True)
    p.add_argument("--n-perturbations", type=int, default=40)
    p.add_argument("--sigmas", default="0.001,0.003,0.01,0.03,0.1")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sequence-length", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--imagination-horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="average reward across this many fixed offline batches",
    )
    p.add_argument(
        "--reward-head-type",
        default=None,
        help="override cfg.world_model.reward_head_type (e.g. 'binary' for per_window ckpts)",
    )
    return p.parse_args()


def perturb_output_projection(
    policy: torch.nn.Module, sft_state: dict, sigma: float, seed: int
) -> None:
    """Restore SFT params, then add seeded Gaussian noise to every
    output_projection tensor (in-place on policy)."""
    policy.load_state_dict(sft_state)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if "output_projection" not in name:
                continue
            noise = torch.randn(p.shape, generator=gen, dtype=torch.float32).to(
                p.device
            )
            # scale noise by sigma * ||p|| / sqrt(numel)  → relative perturbation
            scale = sigma * (p.detach().abs().mean().item() + 1e-9)
            p.add_(noise.to(p.dtype) * scale)


def measure_imagined_reward(
    *,
    policy,
    world_model,
    critic,
    target_critic,
    policy_optimizer,
    critic_optimizer,
    return_tracker,
    batches,
    device,
    algorithm_cfg,
    optim_cfg,
) -> dict:
    """Run imagine_actor_critic_step on a list of fixed batches and average
    raw-reward metrics."""
    rs_mean, rs_min, rs_max = [], [], []
    rs_p10, rs_p50, rs_p90 = [], [], []
    for obs in batches:
        m = imagine_actor_critic_step(
            policy=policy,
            world_model=world_model,
            critic=critic,
            target_critic=target_critic,
            actor_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            return_tracker=return_tracker,
            obs=obs,
            device=device,
            algorithm_cfg=algorithm_cfg,
            optim_cfg=optim_cfg,
            ref_policy=None,  # disable KL/BC so we measure pure WM reward landscape
        )
        rs_mean.append(m["reward_raw_mean"])
        rs_min.append(m["reward_raw_min"])
        rs_max.append(m["reward_raw_max"])
        rs_p10.append(m["reward_raw_p10"])
        rs_p50.append(m["reward_raw_p50"])
        rs_p90.append(m["reward_raw_p90"])
    return {
        "mean": float(np.mean(rs_mean)),
        "p10": float(np.mean(rs_p10)),
        "p50": float(np.mean(rs_p50)),
        "p90": float(np.mean(rs_p90)),
        "min": float(np.mean(rs_min)),
        "max": float(np.mean(rs_max)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.init.world_model_state_ckpt = args.world_model_ckpt
    cfg.algorithm.imagination_horizon = int(args.imagination_horizon)
    cfg.algorithm.actor_bc_to_vla_scale = 0.0
    cfg.algorithm.actor_bc_to_ref_scale = 0.0
    cfg.algorithm.kl_coef = 0.0  # we want raw WM reward only
    if args.reward_head_type is not None:
        cfg.world_model.reward_head_type = str(args.reward_head_type)
        print(
            f"[landscape] cfg.world_model.reward_head_type → {cfg.world_model.reward_head_type}",
            flush=True,
        )

    print(f"[landscape] out_dir={out_dir}", flush=True)

    world_model = hydra.utils.instantiate(cfg.world_model).to(
        device=device, dtype=torch.bfloat16
    )
    # Try standard format first (state_dicts.world_model / model), then fall back to
    # our finetune format (top-level "world_model" key).
    payload = torch.load(args.world_model_ckpt, map_location="cpu", weights_only=False)
    state = (
        payload.get("state_dicts", {}).get("world_model")
        or payload.get("model")
        or payload.get("world_model")
    )
    if state is None:
        raise RuntimeError(f"no usable state dict in {args.world_model_ckpt}")
    cleaned = {}
    dtype = next(world_model.parameters()).dtype
    for raw_key, value in state.items():
        key = str(raw_key).removeprefix("module.")
        target = world_model.state_dict().get(key)
        if target is None or tuple(value.shape) != tuple(target.shape):
            continue
        cleaned[key] = (
            value.to(dtype=dtype) if torch.is_floating_point(value) else value
        )
    missing, unexpected = world_model.load_state_dict(cleaned, strict=False)
    print(
        f"[landscape] WM loaded: kept={len(cleaned)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    freeze_module(world_model)
    world_model.eval()

    policy = hydra.utils.instantiate(cfg.policy).to(device)
    critic = hydra.utils.instantiate(cfg.critic).to(device)
    target_critic = hydra.utils.instantiate(cfg.critic).to(device)
    target_critic.load_state_dict(critic.state_dict())
    freeze_module(target_critic)
    policy_optimizer = build_optimizer(policy, cfg.optim.policy)
    critic_optimizer = build_optimizer(critic, cfg.optim.critic)
    return_tracker = ReturnPercentileTracker(
        decay=float(
            OmegaConf.select(cfg, "algorithm.return_tracker.decay", default=0.99)
        ),
        low=float(OmegaConf.select(cfg, "algorithm.return_tracker.low", default=0.05)),
        high=float(
            OmegaConf.select(cfg, "algorithm.return_tracker.high", default=0.95)
        ),
    )

    # Build offline loader and draw a small fixed batch set we'll reuse.
    class _MiniArgs:
        batch_size = args.batch_size
        sequence_length = args.sequence_length
        num_workers = args.num_workers
        max_offline_windows = None

    loader, infinite = build_offline_loader(cfg, _MiniArgs())
    print(f"[landscape] offline_windows={len(loader.dataset)}", flush=True)

    fixed_obs = []
    for _ in range(int(args.num_batches)):
        b = next(infinite)
        fixed_obs.append(actor_critic_obs(b))
    print(
        f"[landscape] fixed {len(fixed_obs)} batches  (B={args.batch_size}, T={args.sequence_length})",
        flush=True,
    )

    # Snapshot SFT policy state
    sft_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    # Baseline reward at SFT
    policy.load_state_dict(sft_state)
    sft_metrics = measure_imagined_reward(
        policy=policy,
        world_model=world_model,
        critic=critic,
        target_critic=target_critic,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        return_tracker=return_tracker,
        batches=fixed_obs,
        device=device,
        algorithm_cfg=cfg.algorithm,
        optim_cfg=cfg.optim,
    )
    print(
        f"[landscape] SFT baseline: mean={sft_metrics['mean']:.4f}  p50={sft_metrics['p50']:.4f}",
        flush=True,
    )

    sigmas = [float(s) for s in str(args.sigmas).split(",") if s.strip()]
    rows = []
    rng_master = np.random.RandomState(args.seed)
    for sigma in sigmas:
        for k in range(int(args.n_perturbations)):
            seed_k = int(rng_master.randint(0, 10**8))
            perturb_output_projection(policy, sft_state, sigma=sigma, seed=seed_k)
            m = measure_imagined_reward(
                policy=policy,
                world_model=world_model,
                critic=critic,
                target_critic=target_critic,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                return_tracker=return_tracker,
                batches=fixed_obs,
                device=device,
                algorithm_cfg=cfg.algorithm,
                optim_cfg=cfg.optim,
            )
            row = {"sigma": float(sigma), "seed": int(seed_k), **m}
            rows.append(row)
            print(
                f"[landscape] sigma={sigma:.4f} k={k:02d} mean={m['mean']:.4f}  Δ={m['mean'] - sft_metrics['mean']:+.4f}",
                flush=True,
            )

    # Restore SFT before exit
    policy.load_state_dict(sft_state)

    # Persist summary
    summary = {
        "sft": sft_metrics,
        "perturbations": rows,
        "sigmas": sigmas,
        "n_perturbations": int(args.n_perturbations),
        "batch_size": int(args.batch_size),
        "seq_len": int(args.sequence_length),
        "imag_h": int(args.imagination_horizon),
        "num_batches_avg": int(args.num_batches),
    }
    (out_dir / "landscape_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    by_sigma = {s: [r["mean"] for r in rows if r["sigma"] == s] for s in sigmas}
    # Left: histogram of perturbed mean rewards
    for s, vals in by_sigma.items():
        axes[0].hist(vals, bins=15, alpha=0.45, label=f"σ={s}")
    axes[0].axvline(
        sft_metrics["mean"], color="k", linestyle="--", linewidth=2, label="SFT"
    )
    axes[0].set_xlabel("imagined raw reward (mean)")
    axes[0].set_ylabel("count")
    axes[0].set_title("WM raw reward at SFT vs random perturbations")
    axes[0].legend()
    # Right: reward delta vs sigma
    deltas_by_sigma = {s: np.array(by_sigma[s]) - sft_metrics["mean"] for s in sigmas}
    box_data = [deltas_by_sigma[s] for s in sigmas]
    axes[1].boxplot(box_data, labels=[f"{s}" for s in sigmas])
    axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("perturbation σ (relative to |w|)")
    axes[1].set_ylabel("Δ reward vs SFT")
    axes[1].set_title("Reward change under random output_projection perturbations")

    plt.tight_layout()
    plot_path = out_dir / "reward_landscape.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\n[landscape] saved → {plot_path}", flush=True)
    print(f"[landscape] summary → {out_dir / 'landscape_summary.json'}", flush=True)

    # Quick verdict
    all_means = [r["mean"] for r in rows]
    sft_m = sft_metrics["mean"]
    n_above = sum(1 for m in all_means if m > sft_m)
    n_below = sum(1 for m in all_means if m < sft_m)
    pct_below = 100.0 * n_below / len(all_means)
    print(
        f"\n[verdict] of {len(all_means)} perturbations:  {n_below} below SFT ({pct_below:.1f}%),  {n_above} above SFT",
        flush=True,
    )


if __name__ == "__main__":
    main()
