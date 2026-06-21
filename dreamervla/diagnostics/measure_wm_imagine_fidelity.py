"""
Can the WM imagine a full SFT-demo trajectory faithfully?

For one successful demo:
  1. Get full posterior trajectory (latent_post[0..T-1]) via observe_sequence.
  2. From posterior latent[0], imagine forward T-1 steps using the demo's
     true actions (i.e., the actor's true SFT actions), via repeated
     world_model.predict_next.  This produces latent_imag[1..T-1].
  3. Compare:
     - feature MSE / cosine between imagined and posterior at each t
     - reward_head(imag) vs reward_head(post) at each t
     - Can imagined-at-t=T-1 still light up the reward head?

Also: do this from multiple start steps (t=0, t=20, t=40, t=70) to see
how horizon×start interacts.
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

from dreamervla.diagnostics._common import resolve_device
from dreamervla.models.world_model.dreamerv3_torch import DreamerV3LatentState
from dreamervla.utils.latent import reward_of, slice_latent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hidden-hdf5", required=True)
    p.add_argument("--reward-hdf5", required=True)
    p.add_argument("--demo-key", default="demo_0")
    p.add_argument("--start-steps", type=int, nargs="*", default=[0, 20, 40, 60, 70])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def feat_of(latent: DreamerV3LatentState) -> torch.Tensor:
    return latent.feature()


def prepare_world_model_state_dict(world_model, state_dict):
    target = world_model.state_dict()
    converted = {}
    mismatched = []
    for raw_key, value in state_dict.items():
        key = raw_key.removeprefix("module.") if isinstance(raw_key, str) else raw_key
        if key in target and isinstance(value, torch.Tensor):
            if tuple(value.shape) != tuple(target[key].shape):
                mismatched.append((key, tuple(value.shape), tuple(target[key].shape)))
                continue
            if torch.is_floating_point(value):
                value = value.to(dtype=target[key].dtype)
        converted[key] = value
    return converted, mismatched


def main():
    args = parse_args()
    device = resolve_device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    step = ckpt.get("update_step", ckpt.get("epoch", "NA"))
    print(f"[load] {Path(args.ckpt).parent.parent.name} step={step}")

    world_model = hydra.utils.instantiate(cfg.world_model).to(
        device=device, dtype=torch.bfloat16
    )
    state_dict, mismatched = prepare_world_model_state_dict(
        world_model, ckpt["state_dicts"]["world_model"]
    )
    missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
    print(
        f"[load] tensors={len(state_dict)} missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}"
    )
    if missing:
        print(f"[load] missing sample: {missing[:5]}")
    if unexpected:
        print(f"[load] unexpected sample: {unexpected[:5]}")
    if mismatched:
        print(f"[load] mismatched sample: {mismatched[:5]}")
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad = False

    with h5py.File(args.hidden_hdf5, "r") as fh, h5py.File(args.reward_hdf5, "r") as fr:
        obs_emb = fh["data"][args.demo_key]["obs_embedding"][:]
        actions = fr["data"][args.demo_key]["actions"][:]
        sparse = fr["data"][args.demo_key]["sparse_rewards"][:]
    T = int(obs_emb.shape[0])
    print(f"[demo] T={T} success={bool(sparse[-1])}")

    obs_t = (
        torch.from_numpy(obs_emb).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    )
    act_t = (
        torch.from_numpy(actions).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    )
    is_first = torch.zeros(1, T, dtype=torch.bool, device=device)
    is_first[0, 0] = True

    # 1. Full posterior
    with torch.no_grad():
        observed = world_model(
            {
                "mode": "observe_sequence",
                "obs_embedding": obs_t,
                "actions": act_t,
                "is_first": is_first,
            }
        )
    post_latent = observed["latent"]  # [1, T, ...]
    post_feat = post_latent.feature().float()  # [1, T, D]
    with torch.no_grad():
        post_reward = world_model.reward_head(
            post_feat.reshape(T, -1).to(dtype=torch.bfloat16)
        )
        from dreamervla.models.world_model.dreamerv3_torch import _reward_pred

        post_reward = (
            _reward_pred(world_model.reward_head, post_reward)
            .squeeze(-1)
            .float()
            .cpu()
            .numpy()
        )

    print(
        f"\n[posterior reward curve]  t=0:{post_reward[0]:.4f}  T/2:{post_reward[T // 2]:.4f}  T-8:{post_reward[T - 8]:.4f}  T-1:{post_reward[T - 1]:.4f}"
    )

    # 2. Imagine from each start step using demo actions
    records = []
    for start in args.start_steps:
        if start >= T - 1:
            continue
        # Seed at posterior latent[start]
        cur_latent = slice_latent(post_latent, start)
        imag_feats = [cur_latent.feature().float().cpu()]
        imag_rewards = [reward_of(world_model, cur_latent)]
        for t in range(start, T - 1):
            a_t = act_t[:, t, :].to(dtype=cur_latent.deter.dtype)
            with torch.no_grad():
                cur_latent = world_model(
                    {"mode": "predict_next", "latent": cur_latent, "actions": a_t}
                )
            imag_feats.append(cur_latent.feature().float().cpu())
            imag_rewards.append(reward_of(world_model, cur_latent))
        imag_feats_t = torch.cat(imag_feats, dim=0)  # [T-start, D]
        post_feats_for_window = post_feat[0, start:T, :].cpu()  # [T-start, D]
        # Divergence: per-step feature MSE and cosine
        mse_per_step = (imag_feats_t - post_feats_for_window).square().mean(-1).numpy()
        cos_per_step = F.cosine_similarity(
            imag_feats_t, post_feats_for_window, dim=-1
        ).numpy()
        imag_rewards = np.array(imag_rewards)
        post_rewards_window = post_reward[start:T]

        print(
            f"\n──────────── imagine start={start} (horizon={T - start - 1} steps) ────────────"
        )
        # Sample every ~15% of horizon
        L = imag_rewards.shape[0]
        idxs = sorted(set([0, L // 8, L // 4, L // 2, 3 * L // 4, L - 8, L - 4, L - 1]))
        print(
            f"{'step_in_traj':>14} {'h':>4} {'feat_mse':>10} {'cosine':>8} {'imag_r':>8} {'post_r':>8}  Δr"
        )
        for i in idxs:
            if i < 0 or i >= L:
                continue
            print(
                f"{start + i:>14} {i:>4} {mse_per_step[i]:>10.3g} {cos_per_step[i]:>8.4f} "
                f"{imag_rewards[i]:>8.4f} {post_rewards_window[i]:>8.4f}  {imag_rewards[i] - post_rewards_window[i]:+.4f}"
            )
        # Saturation point: where does imagine reward "fire"?
        try:
            fire_imag = (
                int(np.argmax(imag_rewards > 0.5)) if (imag_rewards > 0.5).any() else -1
            )
            fire_post = (
                int(np.argmax(post_rewards_window > 0.5))
                if (post_rewards_window > 0.5).any()
                else -1
            )
        except Exception:
            fire_imag = -1
            fire_post = -1
        print(
            f"  fire_step (r>0.5): imagine={fire_imag}  posterior={fire_post}  (diff={fire_imag - fire_post if fire_imag >= 0 and fire_post >= 0 else 'NA'})"
        )
        records.append(
            {
                "start": int(start),
                "horizon": int(T - start - 1),
                "feat_mse": mse_per_step.tolist(),
                "cosine": cos_per_step.tolist(),
                "imag_reward": imag_rewards.tolist(),
                "post_reward": post_rewards_window.tolist(),
                "fire_step_imag": fire_imag,
                "fire_step_post": fire_post,
            }
        )

    out = {
        "ckpt": args.ckpt,
        "T": T,
        "demo": args.demo_key,
        "post_reward_curve": post_reward.tolist(),
        "starts": records,
    }
    out_json = (
        Path(args.out_json)
        if args.out_json
        else Path(args.ckpt).parent.parent / "wm_imagine_fidelity.json"
    )
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_json}")


if __name__ == "__main__":
    main()
