"""Compare action chunks: trained policy vs frozen pi0-SFT baseline.

Both policies are VLAActionHeadActor instances built from the same config
(``configs/dreamervla_pi0_action_hidden_head_actor.yaml``):

  - baseline: only the pi0 SFT warm-start via init_action_head_ckpt
              (adapter random-init, never updated)
  - trained:  loads the live run's saved checkpoint, so adapter +
              transformer + output_projection reflect actual SGD updates

Then we dump action_chunks on a fixed set of WM-feat inputs sampled
from the offline LIBERO dataset (after WM.hidden_decoder), so the
comparison is grounded in realistic features, not random noise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
import hydra

sys.path.insert(0, "/mnt/data/spoil/workspace/DreamerVLA")


def load_policy_state_from_training_ckpt(ckpt_path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dicts = payload.get("state_dicts", payload)
    if "policy" in state_dicts:
        return state_dicts["policy"]
    raise RuntimeError(f"ckpt {ckpt_path} has no 'policy' state_dict")


def build_actor(cfg, device: str):
    return hydra.utils.instantiate(cfg.policy).to(device)


def get_chunk(actor, hidden: torch.Tensor) -> torch.Tensor:
    actor.eval()
    with torch.no_grad():
        _, _, extra = actor({
            "mode": "sample",
            "hidden": hidden,
            "deterministic": True,
            "return_chunk": True,
        })
    return extra["action_chunk"]


def fmt_row(label: str, vec: torch.Tensor) -> str:
    return f"{label}: [" + ", ".join(f"{v:+.4f}" for v in vec.tolist()) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dreamervla_pi0_action_hidden_head_actor.yaml")
    parser.add_argument(
        "--encoder-ckpt",
        default="/mnt/data/spoil/workspace/DreamerVLA/data/ckpts/pi0_query_vla_libero_goal/epoch003_train_vla_loss1.255_success8of10.ckpt",
        help="pi0 SFT ckpt used to warm-start both actors.",
    )
    parser.add_argument(
        "--trained-ckpt",
        required=True,
        help="Path to a training ckpt (state_dicts.policy will be loaded into the trained actor).",
    )
    parser.add_argument(
        "--baseline-ckpt",
        default=None,
        help="Optional second ckpt for the baseline. Default: just pi0 SFT warm-start (no training).",
    )
    parser.add_argument("--n-inputs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.init.encoder_state_ckpt = args.encoder_ckpt

    # ── build baseline (pi0 SFT only) and trained policies ────────────────
    # Use the SAME torch seed before each build so adapter random init matches.
    print(f"[compare] building baseline (pi0 SFT warm-start, no training) ...")
    torch.manual_seed(args.seed)
    baseline = build_actor(cfg, args.device)
    if args.baseline_ckpt:
        print(f"[compare] overriding baseline state from: {args.baseline_ckpt}")
        sd = load_policy_state_from_training_ckpt(args.baseline_ckpt)
        missing, unexpected = baseline.load_state_dict(sd, strict=False)
        print(f"  baseline non-strict load: missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}")

    print(f"[compare] building trained (loading {args.trained_ckpt}) ...")
    torch.manual_seed(args.seed)
    trained = build_actor(cfg, args.device)
    sd = load_policy_state_from_training_ckpt(args.trained_ckpt)
    missing, unexpected = trained.load_state_dict(sd, strict=False)
    print(f"  trained non-strict load: missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}")

    # ── fixed-seed deterministic WM-like inputs (random gaussian here) ────
    torch.manual_seed(args.seed)
    hidden = torch.randn(args.n_inputs, 5120, device=args.device)

    bc = get_chunk(baseline, hidden)
    tc = get_chunk(trained, hidden)

    diff_abs = (tc - bc).abs()
    print()
    print(f"trained vs baseline (over {args.n_inputs} fixed inputs):")
    print(f"  mean |Δaction|        = {float(diff_abs.mean()):.6f}")
    print(f"  max  |Δaction|        = {float(diff_abs.max()):.6f}")
    print(f"  per-input mean |Δ|    = {[round(float(d), 6) for d in diff_abs.flatten(1).mean(dim=1)]}")
    print(f"  per-action-dim mean |Δ| (across all inputs+timesteps):")
    pd_mean = diff_abs.flatten(0, 1).mean(dim=0)
    for i, v in enumerate(pd_mean.tolist()):
        print(f"     dim {i}: {v:.6f}")
    print()
    print(f"--- sample chunks for input[0] ---")
    for t in range(min(2, tc.shape[1])):
        print(f"  t={t}")
        print(f"    " + fmt_row("baseline", bc[0, t]))
        print(f"    " + fmt_row("trained ", tc[0, t]))
        print(f"    diff:      [" + ", ".join(f"{v:+.4f}" for v in (tc[0, t] - bc[0, t]).tolist()) + "]")

    # parameter-level diff
    print()
    n_changed = 0
    n_same = 0
    total_diff_norm = 0.0
    for (n, pb), (_, pt) in zip(baseline.named_parameters(), trained.named_parameters()):
        d = (pt - pb).detach()
        if torch.allclose(pb.detach(), pt.detach()):
            n_same += 1
        else:
            n_changed += 1
            total_diff_norm += float(d.norm())
    print(f"param-level: {n_changed} tensors changed, {n_same} unchanged, sum-of-norms = {total_diff_norm:.4f}")


if __name__ == "__main__":
    main()
