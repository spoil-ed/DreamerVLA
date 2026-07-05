"""
Like measure_wm_imagine_fidelity.py but imagine using the *trained* policy's
own action chunks (drifted from SFT), or the SFT-init policy's actions, or
demo actions, side-by-side. Tests whether WM imagine can still reach
reward-fire region under OOD actor outputs.
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


def prepare_state_dict(module, state_dict):
    target = module.state_dict()
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


def imagine_path(
    world_model, policy_or_None, demo_actions, start_latent, start_step, T, device
):
    """If policy_or_None is None, use demo actions. Else, query policy each step."""
    cur = start_latent
    rewards = [reward_of(world_model, cur)]
    feats = [cur.feature().float().cpu()]
    for t in range(start_step, T - 1):
        if policy_or_None is None:
            a = demo_actions[:, t, :].to(dtype=cur.deter.dtype)
        else:
            with torch.no_grad():
                hidden = world_model.actor_input(cur).float()
                _, _, extra = policy_or_None(
                    {
                        "mode": "sample",
                        "hidden": hidden,
                        "deterministic": True,
                    }
                )
                a = extra["action_chunk"][:, 0, :].to(dtype=cur.deter.dtype)
        with torch.no_grad():
            cur = world_model({"mode": "predict_next", "latent": cur, "actions": a})
        rewards.append(reward_of(world_model, cur))
        feats.append(cur.feature().float().cpu())
    return np.array(rewards), torch.cat(feats, dim=0)


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
    wm_state, wm_mismatched = prepare_state_dict(
        world_model, ckpt["state_dicts"]["world_model"]
    )
    wm_missing, wm_unexpected = world_model.load_state_dict(wm_state, strict=False)
    print(
        f"[load] world_model tensors={len(wm_state)} missing={len(wm_missing)} "
        f"unexpected={len(wm_unexpected)} mismatched={len(wm_mismatched)}"
    )
    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad = False

    sft_policy = hydra.utils.instantiate(cfg.policy).to(device)
    trained_policy = hydra.utils.instantiate(cfg.policy).to(device)
    policy_state, policy_mismatched = prepare_state_dict(
        trained_policy, ckpt["state_dicts"]["policy"]
    )
    policy_missing, policy_unexpected = trained_policy.load_state_dict(
        policy_state, strict=True
    )
    print(
        f"[load] policy tensors={len(policy_state)} missing={len(policy_missing)} "
        f"unexpected={len(policy_unexpected)} mismatched={len(policy_mismatched)}"
    )
    sft_policy.eval()
    trained_policy.eval()
    for p in sft_policy.parameters():
        p.requires_grad = False
    for p in trained_policy.parameters():
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

    with torch.no_grad():
        observed = world_model(
            {
                "mode": "observe_sequence",
                "obs_embedding": obs_t,
                "actions": act_t,
                "is_first": is_first,
            }
        )
    post_latent = observed["latent"]
    post_feat = post_latent.feature().float()
    with torch.no_grad():
        from dreamervla.models.world_model.dreamerv3_torch import _reward_pred

        pred = world_model.reward_head(
            post_feat.reshape(T, -1).to(dtype=torch.bfloat16)
        )
        post_reward = (
            _reward_pred(world_model.reward_head, pred)
            .squeeze(-1)
            .float()
            .cpu()
            .numpy()
        )

    records = []
    for start in args.start_steps:
        if start >= T - 1:
            continue
        start_lat = slice_latent(post_latent, start)
        # 3 policies: demo (in-dist), SFT actor, trained actor
        r_demo, f_demo = imagine_path(
            world_model, None, act_t, start_lat, start, T, device
        )
        r_sft, f_sft = imagine_path(
            world_model, sft_policy, act_t, start_lat, start, T, device
        )
        r_tr, f_tr = imagine_path(
            world_model, trained_policy, act_t, start_lat, start, T, device
        )
        L = len(r_demo)
        post_r_window = post_reward[start : start + L]

        def fire(r):
            return int(np.argmax(r > 0.5)) if (r > 0.5).any() else -1

        print(f"\n──────────── start={start}  horizon={T - start - 1} ────────────")
        print(
            f"{'h':>4} {'post':>7} {'demoA':>7} {'sftA':>7} {'trnA':>7}    cos(sft,post)  cos(trn,post)"
        )
        idxs = sorted(set([0, L // 4, L // 2, 3 * L // 4, L - 8, L - 1]))
        for i in idxs:
            if i >= L:
                continue
            cs = F.cosine_similarity(
                f_sft[i : i + 1], post_feat[0, start + i : start + i + 1].cpu(), dim=-1
            ).item()
            ct = F.cosine_similarity(
                f_tr[i : i + 1], post_feat[0, start + i : start + i + 1].cpu(), dim=-1
            ).item()
            print(
                f"{i:>4} {post_r_window[i]:>7.3f} {r_demo[i]:>7.3f} {r_sft[i]:>7.3f} {r_tr[i]:>7.3f}    {cs:>13.4f}  {ct:>13.4f}"
            )
        print(
            f"  fire (r>0.5): demo={fire(r_demo)}  sft={fire(r_sft)}  trained={fire(r_tr)}  post={fire(post_r_window)}"
        )
        # End reward (last imagine step)
        print(
            f"  final r:    demo={r_demo[-1]:.4f}  sft={r_sft[-1]:.4f}  trained={r_tr[-1]:.4f}  post={post_r_window[-1]:.4f}"
        )
        records.append(
            {
                "start": start,
                "horizon": T - start - 1,
                "imag_r_demo": r_demo.tolist(),
                "imag_r_sft": r_sft.tolist(),
                "imag_r_trained": r_tr.tolist(),
                "post_r": post_r_window.tolist(),
                "fire_demo": fire(r_demo),
                "fire_sft": fire(r_sft),
                "fire_trained": fire(r_tr),
                "fire_post": fire(post_r_window),
            }
        )

    out = {
        "ckpt": args.ckpt,
        "T": T,
        "post_reward_curve": post_reward.tolist(),
        "starts": records,
    }
    out_json = (
        Path(args.out_json)
        if args.out_json
        else Path(args.ckpt).parent.parent / "wm_imagine_actor.json"
    )
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_json}")


if __name__ == "__main__":
    main()
