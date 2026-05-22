"""
Measure two things on a successful demo trajectory:

A. hidden_decoder reconstruction quality
    real_obs_embedding[t]  vs  hidden_decoder(posterior_latent[t])
    → MSE, cosine, max-abs-diff per step, and aggregated.

B. SFT actor sensitivity to recon
    a_real  = sft_policy(real_obs_embedding[t])
    a_recon = sft_policy(hidden_decoder(posterior_latent[t]))
    → action_chunk MSE / MAE / max-abs-diff per step, and aggregated.

Also compare action_chunk against demo's actual action[t] for both inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.world_model.dreamerv3_torch import DreamerV3LatentState


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hidden-hdf5", required=True)
    p.add_argument("--reward-hdf5", required=True)
    p.add_argument("--demo-key", default="demo_0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    print(f"[load] {Path(args.ckpt).parent.parent.name} step={ckpt['update_step']}")

    world_model = hydra.utils.instantiate(cfg.world_model).to(device=device, dtype=torch.bfloat16)
    world_model.load_state_dict(ckpt["state_dicts"]["world_model"], strict=False)
    world_model.eval()
    for p in world_model.parameters(): p.requires_grad = False

    sft_policy = hydra.utils.instantiate(cfg.policy).to(device)
    sft_policy.eval()
    for p in sft_policy.parameters(): p.requires_grad = False

    with h5py.File(args.hidden_hdf5, "r") as fh, h5py.File(args.reward_hdf5, "r") as fr:
        obs_emb = fh["data"][args.demo_key]["obs_embedding"][:]   # (T, 5120)
        actions = fr["data"][args.demo_key]["actions"][:]         # (T, 7)
    T = int(obs_emb.shape[0])
    print(f"[demo] T={T}")

    obs_t = torch.from_numpy(obs_emb).unsqueeze(0).to(device=device, dtype=torch.bfloat16)   # [1,T,5120]
    act_t = torch.from_numpy(actions).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    demo_a_chunk = act_t                                                                       # [1,T,7] reference
    is_first = torch.zeros(1, T, dtype=torch.bool, device=device); is_first[0, 0] = True

    # Posterior over the full episode
    with torch.no_grad():
        observed = world_model({"mode": "observe_sequence",
                                "obs_embedding": obs_t, "actions": act_t, "is_first": is_first})
    post_latent = observed["latent"]
    post_feat = post_latent.feature().float()                       # [1, T, feat_dim]

    # ── A. hidden_decoder reconstruction quality ────────────────────────────
    with torch.no_grad():
        recon = world_model.hidden_decoder(post_feat.reshape(T, -1).to(dtype=torch.bfloat16))   # [T, 5120]
    real = obs_t[0].float()                                          # [T, 5120]
    recon = recon.float()
    diff = recon - real
    per_step_mse = diff.square().mean(-1).cpu().numpy()              # [T]
    per_step_cos = F.cosine_similarity(recon, real, dim=-1).cpu().numpy()
    per_step_max_abs = diff.abs().max(-1).values.cpu().numpy()
    real_norm = real.norm(dim=-1).cpu().numpy()
    recon_norm = recon.norm(dim=-1).cpu().numpy()
    rel_err = (diff.norm(dim=-1) / (real.norm(dim=-1) + 1e-9)).cpu().numpy()

    print("\n=========== A. hidden_decoder recon vs real obs_embedding ===========")
    print(f"global per-step MSE:    mean={per_step_mse.mean():.4g}  p50={np.median(per_step_mse):.4g}  p90={np.percentile(per_step_mse,90):.4g}  max={per_step_mse.max():.4g}")
    print(f"global cosine sim:      mean={per_step_cos.mean():.4f}  p50={np.median(per_step_cos):.4f}  p10={np.percentile(per_step_cos,10):.4f}  min={per_step_cos.min():.4f}")
    print(f"global max-abs-diff:    mean={per_step_max_abs.mean():.3f}  max={per_step_max_abs.max():.3f}")
    print(f"relative L2 error:      mean={rel_err.mean():.4f}  p50={np.median(rel_err):.4f}  max={rel_err.max():.4f}")
    print(f"real obs_emb L2 norm:   mean={real_norm.mean():.2f}  recon L2 norm: mean={recon_norm.mean():.2f}")

    # ── B. SFT actor: real vs recon input → action delta ────────────────────
    with torch.no_grad():
        # All T steps in one batch each
        _, _, sft_real_extra  = sft_policy({"mode": "sample", "hidden": real.to(dtype=torch.bfloat16), "deterministic": True})
        _, _, sft_recon_extra = sft_policy({"mode": "sample", "hidden": recon.to(dtype=torch.bfloat16), "deterministic": True})
    a_real  = sft_real_extra["action_chunk"].float()    # [T, time_horizon, 7]
    a_recon = sft_recon_extra["action_chunk"].float()

    diff_a = a_real - a_recon
    per_step_action_mse  = diff_a.square().mean(dim=(-1,-2)).cpu().numpy()
    per_step_action_mae  = diff_a.abs().mean(dim=(-1,-2)).cpu().numpy()
    per_step_action_max  = diff_a.abs().reshape(T, -1).max(-1).values.cpu().numpy()

    print("\n=========== B. SFT actor: action_chunk(real) vs action_chunk(recon) ===========")
    print(f"per-step MSE:     mean={per_step_action_mse.mean():.5g}  p50={np.median(per_step_action_mse):.5g}  p90={np.percentile(per_step_action_mse,90):.5g}  max={per_step_action_mse.max():.5g}")
    print(f"per-step MAE:     mean={per_step_action_mae.mean():.5g}  p50={np.median(per_step_action_mae):.5g}  max={per_step_action_mae.max():.5g}")
    print(f"per-step max|Δ|:  mean={per_step_action_max.mean():.4f}  max={per_step_action_max.max():.4f}")

    # ── C. How close are SFT actions to demo? (sanity check) ────────────────
    demo_chunk_first = demo_a_chunk[0].float()                                # [T, 7] only first action
    # action_chunk has shape [T, time_horizon=5, 7]; the [t, 0, :] should match demo[t]
    a_real_first  = a_real[:, 0, :]    # [T, 7]
    a_recon_first = a_recon[:, 0, :]

    diff_real_vs_demo  = a_real_first  - demo_chunk_first
    diff_recon_vs_demo = a_recon_first - demo_chunk_first

    print("\n=========== C. SFT action[t,0] vs demo action[t] ===========")
    print(f"SFT(real)  vs demo:  MAE mean={diff_real_vs_demo.abs().mean().item():.5g}  max|Δ| mean={diff_real_vs_demo.abs().max(-1).values.mean().item():.4f}")
    print(f"SFT(recon) vs demo:  MAE mean={diff_recon_vs_demo.abs().mean().item():.5g}  max|Δ| mean={diff_recon_vs_demo.abs().max(-1).values.mean().item():.4f}")

    # ── Per-step printout at a few representative timesteps ────────────────
    print("\n=========== Per-step samples ===========")
    print(f"{'t':>4} {'recon_mse':>10} {'recon_cos':>10} {'a_real-recon MAE':>17} {'a_real-demo MAE':>16} {'a_recon-demo MAE':>17}")
    for t in [0, 10, 20, 40, 60, 80, T-5, T-1]:
        if t >= T: continue
        ar_demo = (a_real_first[t]  - demo_chunk_first[t]).abs().mean().item()
        ac_demo = (a_recon_first[t] - demo_chunk_first[t]).abs().mean().item()
        ar_ac   = (a_real_first[t]  - a_recon_first[t]).abs().mean().item()
        print(f"{t:>4} {per_step_mse[t]:>10.4g} {per_step_cos[t]:>10.4f} {ar_ac:>17.5g} {ar_demo:>16.5g} {ac_demo:>17.5g}")

    out = {
        "ckpt": args.ckpt, "T": T,
        "recon_mse_mean": float(per_step_mse.mean()),
        "recon_cos_mean": float(per_step_cos.mean()),
        "recon_rel_err_mean": float(rel_err.mean()),
        "action_real_vs_recon_MAE_mean": float(per_step_action_mae.mean()),
        "action_real_vs_demo_MAE_mean": float(diff_real_vs_demo.abs().mean().item()),
        "action_recon_vs_demo_MAE_mean": float(diff_recon_vs_demo.abs().mean().item()),
    }
    out_json = Path(args.out_json) if args.out_json else Path(args.ckpt).parent.parent / "recon_and_action_delta.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_json}")


if __name__ == "__main__":
    main()
