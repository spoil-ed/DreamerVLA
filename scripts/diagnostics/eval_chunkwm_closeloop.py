"""Closed-loop vs open-loop rollout evaluation for the chunk-aware WM.

For each real demo we feed the first H real frames + K real chunk actions
into the WM, then iterate forward N chunks. Two protocols:

  Open-loop (teacher-forced): every chunk's input history is the REAL
    obs frames at that env-step. Predictions are scored vs real obs but
    the model never sees its own predictions as input — measures the
    one-chunk prediction quality only.

  Close-loop (autoregressive): only the first chunk gets real history;
    afterwards the model's own predicted hiddens are rolled forward as
    the next chunk's history. Errors compound over chunks; measures the
    drift behavior under the actual PPO outcome rollout protocol.

Reports cos sim + MSE per env-step (averaged over demos) for both modes.

Usage:
    python scripts/diagnostics/eval_chunkwm_closeloop.py \
        --ckpt data/outputs/worldmodel/dinowm_chunk/<run>/ckpt/latest.ckpt \
        --num-demos 16 --num-chunks 20 --device cuda:4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dreamer_vla.dataset.wm_replay_classifier_dataset import _find_demo_pairs
from dreamer_vla.models.world_model.rynn_dino_wm_chunk import ChunkAwareRynnDinoWMWorldModel


def load_chunk_wm(
    ckpt_path: str, device: torch.device
) -> ChunkAwareRynnDinoWMWorldModel:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    wm_cfg_blob = sd.get("cfg", {}).get("world_model", {})
    chunk_size = int(wm_cfg_blob.get("chunk_size", 5))
    kwargs = {}
    for k in (
        "obs_dim",
        "action_dim",
        "token_count",
        "token_dim",
        "model_dim",
        "depth",
        "heads",
        "mlp_dim",
        "dropout",
        "num_hist",
        "num_pred",
        "max_seq_len",
        "hidden_loss_scale",
        "cosine_loss_scale",
        "rollout_loss_scale",
        "rollout_horizon",
        "rollout_context",
        "reward_head_type",
        "reward_loss_scale",
        "reward_hidden_dim",
        "reward_init_logit",
        "reward_pos_weight",
        "return_predictions",
    ):
        if k in wm_cfg_blob:
            kwargs[k] = wm_cfg_blob[k]
    wm = ChunkAwareRynnDinoWMWorldModel(chunk_size=chunk_size, **kwargs)
    missing, unexpected = wm.load_state_dict(sd["model"], strict=False)
    print(
        f"[load] global_step={sd.get('global_step')} epoch={sd.get('epoch')}"
        f" missing={missing} unexpected={unexpected}"
    )
    return wm.eval().to(device)


def load_demo(
    raw_p: Path, hid_p: Path, demo_key: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (obs[T, obs_dim], actions[T, action_dim]) for one demo, or None."""
    with h5py.File(str(hid_p), "r") as hh:
        if f"{demo_key}/obs_embedding" not in hh:
            return None
        obs = np.asarray(hh[f"{demo_key}/obs_embedding"][...], dtype=np.float32)
    with h5py.File(str(raw_p), "r") as fr:
        grp = fr[demo_key]
        if "actions" not in grp:
            return None
        actions = np.asarray(grp["actions"][...], dtype=np.float32)
    T = min(obs.shape[0], actions.shape[0])
    return obs[:T].reshape(T, -1), actions[:T]


@torch.no_grad()
def rollout(
    wm: ChunkAwareRynnDinoWMWorldModel,
    obs: torch.Tensor,  # [T, obs_dim]
    actions: torch.Tensor,  # [T, action_dim]
    num_chunks: int,
    mode: str,  # "open" or "close"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pred_seq[N*K, obs_dim], target_seq[N*K, obs_dim])."""
    H = wm.num_hist
    K = wm.chunk_size
    T = int(obs.shape[0])
    N = min(num_chunks, (T - H) // K)
    if N < 1:
        raise ValueError(f"demo too short: T={T} needs >= H+K = {H + K}")

    history = obs[:H].unsqueeze(0)  # [1, H, obs_dim]
    action_history = torch.zeros(
        1, H, wm.action_dim, device=obs.device, dtype=obs.dtype
    )
    if H > 1:
        action_history[:, : H - 1] = actions[: H - 1].unsqueeze(0)

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    cur_latent = {
        "hidden": history[:, -1],
        "history": history,
        "actions": action_history,
    }

    for c in range(N):
        chunk_actions = actions[H - 1 + c * K : H - 1 + c * K + K].unsqueeze(
            0
        )  # [1, K, A]
        out = wm.predict_next_chunk(cur_latent, chunk_actions)
        pred = out["hidden_seq"][0]  # [K, obs_dim]
        target = obs[H + c * K : H + (c + 1) * K]  # [K, obs_dim]
        preds.append(pred)
        targets.append(target)

        if mode == "open":
            # Teacher forcing: NEXT chunk's input is real obs aligned to where
            # chunk c just finished.  Chunk c predicted obs[H+c*K : H+(c+1)*K];
            # the next chunk needs history ending at obs[H+(c+1)*K - 1].
            #   h_0 (for chunk c+1) = obs[H + (c+1)*K - 1]
            #   history             = obs[(c+1)*K : (c+1)*K + H]
            #   action_history      = actions at frames [history start .. h_0 - 1]
            #                         (last slot stays 0, predict_next_chunk
            #                          overwrites with next chunk's a_0)
            start = (c + 1) * K
            end = start + H  # exclusive
            new_history = obs[start:end].unsqueeze(0)
            new_action_history = torch.zeros(
                1, H, wm.action_dim, device=obs.device, dtype=obs.dtype
            )
            if H > 1:
                new_action_history[:, : H - 1] = actions[
                    start : start + H - 1
                ].unsqueeze(0)
            cur_latent = {
                "hidden": new_history[:, -1],
                "history": new_history,
                "actions": new_action_history,
            }
        else:  # close
            cur_latent = {
                "history": out["history"],
                "actions": out["actions"],
                "hidden": out["hidden"],
            }

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)  # both [N*K, obs_dim]


def per_step_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-env-step cos sim + per-dim MSE; returns numpy arrays [T_pred]."""
    cos = F.cosine_similarity(pred.float(), target.float(), dim=-1)  # [T]
    mse = ((pred.float() - target.float()) ** 2).mean(dim=-1)  # [T]
    rel_l2 = (pred.float() - target.float()).norm(dim=-1) / target.float().norm(
        dim=-1
    ).clamp_min(1e-8)
    return {
        "cos": cos.cpu().numpy(),
        "mse": mse.cpu().numpy(),
        "rel_l2": rel_l2.cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--success-dir-raw",
        default="/mnt/data/spoil/workspace/DreamerVLA/data/processed_data/libero_goal_no_noops_t_256_pi06_remaining_reward",
    )
    parser.add_argument(
        "--success-dir-hidden",
        default="/mnt/data/spoil/workspace/DreamerVLA/data/processed_data/libero_goal_no_noops_t_256_pi0_legacy_action_hidden_vla_policy_h2",
    )
    parser.add_argument("--num-demos", type=int, default=16)
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="Optional JSON dump path.")
    args = parser.parse_args()

    device = torch.device(args.device)
    wm = load_chunk_wm(args.ckpt, device)
    H, K = wm.num_hist, wm.chunk_size
    print(
        f"[wm] num_hist={H} chunk_size={K} action_dim={wm.action_dim} obs_dim={wm.obs_dim}"
    )

    pairs = _find_demo_pairs(args.success_dir_raw, args.success_dir_hidden)[
        : args.num_demos
    ]
    print(f"[data] using {len(pairs)} demos")

    per_demo_open: list[dict] = []
    per_demo_close: list[dict] = []
    for i, (raw_p, hid_p, demo_key) in enumerate(pairs):
        rec = load_demo(Path(raw_p), Path(hid_p), demo_key)
        if rec is None:
            continue
        obs_np, act_np = rec
        T = int(obs_np.shape[0])
        if T < H + K:
            continue
        obs_t = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
        act_t = torch.from_numpy(act_np).to(device=device, dtype=torch.float32)

        pred_o, tgt_o = rollout(wm, obs_t, act_t, args.num_chunks, mode="open")
        pred_c, tgt_c = rollout(wm, obs_t, act_t, args.num_chunks, mode="close")
        per_demo_open.append(per_step_metrics(pred_o, tgt_o))
        per_demo_close.append(per_step_metrics(pred_c, tgt_c))
        print(
            f"  [{i + 1}/{len(pairs)}] {Path(raw_p).stem}/{demo_key} T={T} "
            f"open cos[mean/min]={per_demo_open[-1]['cos'].mean():.4f}/{per_demo_open[-1]['cos'].min():.4f} "
            f"close cos[mean/min]={per_demo_close[-1]['cos'].mean():.4f}/{per_demo_close[-1]['cos'].min():.4f}",
            flush=True,
        )

    # Stack and average across demos along env-step axis.
    def aggregate(per_demo: list[dict]) -> dict:
        min_T = min(d["cos"].shape[0] for d in per_demo)
        cos = np.stack([d["cos"][:min_T] for d in per_demo], axis=0)  # [D, T]
        mse = np.stack([d["mse"][:min_T] for d in per_demo], axis=0)
        rl2 = np.stack([d["rel_l2"][:min_T] for d in per_demo], axis=0)
        return {
            "cos_mean": cos.mean(0),
            "cos_std": cos.std(0),
            "mse_mean": mse.mean(0),
            "mse_std": mse.std(0),
            "rel_l2_mean": rl2.mean(0),
            "rel_l2_std": rl2.std(0),
            "T_pred": int(min_T),
            "n_demos": int(cos.shape[0]),
        }

    open_agg = aggregate(per_demo_open)
    close_agg = aggregate(per_demo_close)

    print("\n=== per-CHUNK summary (mean over demos) ===")
    print(
        f"{'chunk':>5} {'env-step':>8}   {'open cos':>10}  {'close cos':>10}   "
        f"{'open mse':>10}  {'close mse':>10}   {'open rL2':>10}  {'close rL2':>10}"
    )
    T_show = open_agg["T_pred"]
    for c in range(0, T_show // K):
        env_step = (c + 1) * K - 1  # chunk-end env-step (0-indexed)
        # average within the chunk
        sl = slice(c * K, (c + 1) * K)
        o_cos = open_agg["cos_mean"][sl].mean()
        c_cos = close_agg["cos_mean"][sl].mean()
        o_mse = open_agg["mse_mean"][sl].mean()
        c_mse = close_agg["mse_mean"][sl].mean()
        o_rl2 = open_agg["rel_l2_mean"][sl].mean()
        c_rl2 = close_agg["rel_l2_mean"][sl].mean()
        print(
            f"{c:>5} {env_step:>8}   {o_cos:>10.4f}  {c_cos:>10.4f}   "
            f"{o_mse:>10.4f}  {c_mse:>10.4f}   {o_rl2:>10.4f}  {c_rl2:>10.4f}"
        )

    print("\n=== aggregate over all predicted env-steps ===")
    print(
        f"  open : cos {open_agg['cos_mean'].mean():.4f}  mse {open_agg['mse_mean'].mean():.4f}  "
        f"rel_l2 {open_agg['rel_l2_mean'].mean():.4f}  ({open_agg['n_demos']} demos x {open_agg['T_pred']} steps)"
    )
    print(
        f"  close: cos {close_agg['cos_mean'].mean():.4f}  mse {close_agg['mse_mean'].mean():.4f}  "
        f"rel_l2 {close_agg['rel_l2_mean'].mean():.4f}"
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "ckpt": args.ckpt,
                    "num_demos": open_agg["n_demos"],
                    "T_pred": open_agg["T_pred"],
                    "open": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in open_agg.items()
                    },
                    "close": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in close_agg.items()
                    },
                }
            )
        )
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
