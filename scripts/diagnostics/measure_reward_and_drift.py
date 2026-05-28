"""
Three diagnostics on a trained DreamerVLA ckpt:

  A. Reward curve along a successful expert demo (whole episode).
  B. Whether SFT-direction action gets the highest reward_head value
     compared to random perturbations of the same action.
  C. Action drift: trained_policy vs SFT-init policy on the same posterior
     latents (true MSE / max-abs-diff, since drift_raw metric is broken
     for adapter_type=identity).

Usage:
    python scripts/measure_reward_and_drift.py \
        --ckpt /path/to/cotrain_perwindow_gpu4_.../checkpoints/latest.ckpt \
        --hidden-hdf5 /path/to/hidden/put_the_bowl_on_the_plate_demo.hdf5 \
        --reward-hdf5 /path/to/rewards/put_the_bowl_on_the_plate_demo.hdf5 \
        --demo-key demo_0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.world_model.dreamerv3_torch import DreamerV3LatentState


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--hidden-hdf5", required=True, help="sidecar HDF5 with obs_embedding"
    )
    p.add_argument(
        "--reward-hdf5", required=True, help="env HDF5 with actions/sparse_rewards"
    )
    p.add_argument("--demo-key", default="demo_0")
    p.add_argument(
        "--probe-steps",
        type=int,
        nargs="*",
        default=None,
        help="Mid-episode steps to probe for SFT-direction reward (default = 5 evenly spaced)",
    )
    p.add_argument("--num-perturb", type=int, default=8)
    p.add_argument("--perturb-scales", type=float, nargs="*", default=[0.1, 0.5, 1.0])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def slice_latent(latent: DreamerV3LatentState, t: int) -> DreamerV3LatentState:
    return DreamerV3LatentState(
        deter=latent.deter[:, t],
        stoch=latent.stoch[:, t],
        logits=latent.logits[:, t],
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load checkpoint and reconstruct cfg
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"[load] ckpt env_step={ckpt['env_step']} update_step={ckpt['update_step']}")

    cfg = OmegaConf.create(ckpt["cfg"])

    # 2. Instantiate WM + load trained state
    world_model = hydra.utils.instantiate(cfg.world_model).to(
        device=device, dtype=torch.bfloat16
    )
    missing, unexpected = world_model.load_state_dict(
        ckpt["state_dicts"]["world_model"], strict=False
    )
    if missing or unexpected:
        print(f"[wm] missing={len(missing)} unexpected={len(unexpected)}")
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad = False

    # 3. Instantiate SFT-init policy (fresh = SFT weights) and trained policy
    sft_policy = hydra.utils.instantiate(cfg.policy).to(device)
    trained_policy = hydra.utils.instantiate(cfg.policy).to(device)
    trained_policy.load_state_dict(ckpt["state_dicts"]["policy"], strict=True)
    sft_policy.eval()
    trained_policy.eval()
    for p in sft_policy.parameters():
        p.requires_grad = False
    for p in trained_policy.parameters():
        p.requires_grad = False

    # 4. Load one full demo
    with h5py.File(args.hidden_hdf5, "r") as fh, h5py.File(args.reward_hdf5, "r") as fr:
        obs_embedding = fh["data"][args.demo_key]["obs_embedding"][:]  # (T, 5120)
        actions = fr["data"][args.demo_key]["actions"][:]  # (T, 7)
        sparse_rewards = fr["data"][args.demo_key]["sparse_rewards"][:]  # (T,)
        dense_rewards = fr["data"][args.demo_key]["rewards"][
            :
        ]  # (T,) progress in [0,1]
    T = int(obs_embedding.shape[0])
    print(f"[demo] {args.demo_key} T={T} success={bool(sparse_rewards[-1])}")

    obs_emb_t = (
        torch.from_numpy(obs_embedding)
        .unsqueeze(0)
        .to(device=device, dtype=torch.bfloat16)
    )
    actions_t = (
        torch.from_numpy(actions).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    )
    is_first = torch.zeros(1, T, dtype=torch.bool, device=device)
    is_first[0, 0] = True

    # 5. Run posterior over full episode
    with torch.no_grad():
        observed = world_model(
            {
                "mode": "observe_sequence",
                "obs_embedding": obs_emb_t,
                "actions": actions_t,
                "is_first": is_first,
            }
        )
    latent_seq = observed["latent"]  # batch=1, time=T
    feat_seq = latent_seq.feature()  # [1, T, D]

    # ── A. Reward curve over the whole episode ──────────────────────────────
    with torch.no_grad():
        flat_feat = feat_seq.reshape(T, -1)
        if hasattr(world_model.reward_head, "pred"):
            logits = world_model.reward_head(flat_feat)
            reward_curve = world_model.reward_head.pred(logits).float().cpu().numpy()
        else:
            from src.models.world_model.dreamerv3_torch import _reward_pred

            pred = world_model.reward_head(flat_feat)
            reward_curve = (
                _reward_pred(world_model.reward_head, pred)
                .squeeze(-1)
                .float()
                .cpu()
                .numpy()
            )

    print(f"\n========== A. Reward curve along demo (T={T}) ==========")
    print(f"  reward_head_type: {type(world_model.reward_head).__name__}")
    print(f"  reward[t=0]      = {reward_curve[0]:.4f}")
    print(f"  reward[T/4]      = {reward_curve[T // 4]:.4f}")
    print(f"  reward[T/2]      = {reward_curve[T // 2]:.4f}")
    print(f"  reward[3T/4]     = {reward_curve[3 * T // 4]:.4f}")
    print(f"  reward[T-8]      = {reward_curve[T - 8]:.4f}")
    print(f"  reward[T-4]      = {reward_curve[T - 4]:.4f}")
    print(f"  reward[T-2]      = {reward_curve[T - 2]:.4f}")
    print(f"  reward[T-1]      = {reward_curve[T - 1]:.4f}")
    # ASCII sparkline at 40 columns
    cols = 60
    idxs = np.linspace(0, T - 1, cols).astype(int)
    sample = reward_curve[idxs]
    norm = (sample - sample.min()) / (sample.max() - sample.min() + 1e-9)
    bars = " ▁▂▃▄▅▆▇█"
    line = "".join(bars[int(v * (len(bars) - 1))] for v in norm)
    print(f"  curve: |{line}|  (min={sample.min():.3f} max={sample.max():.3f})")

    # ── B + C. Pick probe steps and compare actions ─────────────────────────
    probe_steps = args.probe_steps or list(np.linspace(8, T - 2, 5).astype(int))
    print(
        f"\n========== B+C. Probe SFT vs Trained action at steps {list(probe_steps)} =========="
    )
    bc_records = []
    for t in probe_steps:
        feat_t = feat_seq[:, t].float()  # [1, feat_dim]
        latent_t = slice_latent(latent_seq, t)
        # Policy expects hidden_decoder output ([1, 5120]), not raw feature
        with torch.no_grad():
            actor_hidden = world_model.actor_input(latent_t).float()  # [1, 5120]
        feat_t = actor_hidden
        # SFT and trained action chunks (deterministic for clean comparison)
        with torch.no_grad():
            _, _, sft_extra = sft_policy(
                {"mode": "sample", "hidden": feat_t, "deterministic": True}
            )
            _, _, tr_extra = trained_policy(
                {"mode": "sample", "hidden": feat_t, "deterministic": True}
            )
        sft_chunk = sft_extra["action_chunk"].float()  # [1, time_horizon, 7]
        tr_chunk = tr_extra["action_chunk"].float()
        drift_mse = (sft_chunk - tr_chunk).square().mean().item()
        drift_mae = (sft_chunk - tr_chunk).abs().mean().item()
        drift_max = (sft_chunk - tr_chunk).abs().max().item()

        # Use first action in chunk for predict_next
        sft_a0 = sft_chunk[:, 0, :]
        tr_a0 = tr_chunk[:, 0, :]
        # B. Reward of next-latent for SFT vs trained vs perturbations
        with torch.no_grad():
            ln_sft = world_model(
                {
                    "mode": "predict_next",
                    "latent": latent_t,
                    "actions": sft_a0.to(dtype=torch.bfloat16),
                }
            )
            ln_tr = world_model(
                {
                    "mode": "predict_next",
                    "latent": latent_t,
                    "actions": tr_a0.to(dtype=torch.bfloat16),
                }
            )
            r_sft = world_model.state_reward(ln_sft).float().cpu().item()
            r_tr = world_model.state_reward(ln_tr).float().cpu().item()

            # Perturbations around SFT
            K = args.num_perturb
            perturb_rewards = {}
            for scale in args.perturb_scales:
                noise = torch.randn(K, 7, device=device, dtype=torch.float32) * scale
                noisy_a = (sft_a0.expand(K, -1) + noise).to(dtype=torch.bfloat16)
                # broadcast latent to K
                latent_k = DreamerV3LatentState(
                    deter=latent_t.deter.expand(K, -1).contiguous(),
                    stoch=latent_t.stoch.expand(K, -1, -1).contiguous(),
                    logits=latent_t.logits.expand(K, -1, -1).contiguous(),
                )
                ln_k = world_model(
                    {"mode": "predict_next", "latent": latent_k, "actions": noisy_a}
                )
                r_k = world_model.state_reward(ln_k).float().cpu().numpy()
                perturb_rewards[scale] = {
                    "min": float(r_k.min()),
                    "p50": float(np.median(r_k)),
                    "max": float(r_k.max()),
                    "sft_beats_max": bool(r_sft >= r_k.max()),
                    "sft_rank": int((r_k > r_sft).sum()),  # 0 = SFT highest
                }
        rec = {
            "t": int(t),
            "drift_mse": drift_mse,
            "drift_mae": drift_mae,
            "drift_max_abs": drift_max,
            "r_sft_action": r_sft,
            "r_trained_action": r_tr,
            "r_diff_trained_minus_sft": r_tr - r_sft,
            "perturb": perturb_rewards,
            "demo_reward_at_t": float(reward_curve[t]),
        }
        bc_records.append(rec)
        # Print compact summary
        print(f"\n  t={t} (demo reward here = {reward_curve[t]:.3f})")
        print(
            f"    drift  MSE={drift_mse:.5g}  MAE={drift_mae:.5g}  max|Δ|={drift_max:.5g}"
        )
        print(f"    reward(next | SFT  action) = {r_sft:.4f}")
        print(f"    reward(next | TRND action) = {r_tr:.4f}   Δ={r_tr - r_sft:+.4f}")
        for scale, stats in perturb_rewards.items():
            verdict = (
                "✓ SFT wins"
                if stats["sft_rank"] == 0
                else f"✗ SFT rank {stats['sft_rank']}/{args.num_perturb}"
            )
            print(
                f"    perturb σ={scale}: min={stats['min']:.4f} p50={stats['p50']:.4f} max={stats['max']:.4f}  {verdict}"
            )

    # ── Save JSON ──────────────────────────────────────────────────────────
    out = {
        "ckpt": str(ckpt_path),
        "env_step": int(ckpt["env_step"]),
        "update_step": int(ckpt["update_step"]),
        "demo": {"hidden": args.hidden_hdf5, "key": args.demo_key, "T": T},
        "reward_head": type(world_model.reward_head).__name__,
        "reward_curve": reward_curve.tolist(),
        "demo_dense_rewards": dense_rewards.tolist(),
        "demo_sparse_rewards": sparse_rewards.tolist(),
        "probes": bc_records,
    }
    out_json = (
        Path(args.out_json)
        if args.out_json
        else ckpt_path.parent / "measure_reward_and_drift.json"
    )
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_json}")


if __name__ == "__main__":
    main()
