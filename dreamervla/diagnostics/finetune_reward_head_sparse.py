#!/usr/bin/env python
# ruff: noqa: E402
"""Fine-tune ONLY the WM reward head as a terminal-success classifier.

WMPO-style recipe:
  * Each training batch is 50/50 positive (window ending at terminal of a
    successful demo) and negative (window NOT touching terminal).
  * Targets are sparse_rewards (0 everywhere except a single 1 at the terminal
    step in positive windows; all zeros in negative windows).
  * Everything except the reward head is frozen.
  * Optionally swap the existing SymexpTwoHot reward head for a binary
    BernoulliRewardHead to match the {0,1} target shape directly.

Outputs a new WM checkpoint with the fine-tuned reward head; the actor/critic
training script can then resume from it via --world-model-ckpt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dreamervla.dataset.libero_balanced_terminal_dataset import (
    BalancedTerminalSampler,
    LIBEROBalancedTerminalDataset,
)
from dreamervla.models.world_model.dreamerv3_torch import BinaryRewardHead
from dreamervla.runners.online_utils import (
    load_world_model_state,
)
from dreamervla.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/dreamervla/rynnvla_wmpo_outcome.yaml"),
    )
    p.add_argument("--world-model-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sequence-length", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--swap-binary-head",
        action="store_true",
        help="Replace the existing SymexpTwoHot reward head with a fresh BinaryRewardHead",
    )
    p.add_argument("--binary-init-logit", type=float, default=-5.0)
    p.add_argument(
        "--binary-pos-weight",
        type=float,
        default=10.0,
        help="positive class weight for BCE (compensates within-window 1:31 imbalance)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--positive-ratio", type=float, default=0.5)
    p.add_argument(
        "--loss-mode",
        choices=("per_step_sparse", "per_window", "reward_diffusion"),
        default="per_window",
        help=(
            "per_step_sparse: original (sparse 0/1 per step, collapses); "
            "per_window: WMPO-style single label at last latent (BinaryRewardHead, BCE); "
            "reward_diffusion: gamma^(W-1-t) decay on positive windows, 0 on negatives."
        ),
    )
    p.add_argument("--diffusion-gamma", type=float, default=0.95)
    return p.parse_args()


def reward_head_param_names(world_model: torch.nn.Module) -> list[str]:
    return [n for n, _ in world_model.named_parameters() if "reward_head" in n]


def maybe_swap_binary_head(
    world_model: torch.nn.Module,
    cfg: Any,
    init_logit: float,
    pos_weight: float,
) -> None:
    """Replace world_model.reward_head with a fresh BinaryRewardHead.

    Inherits feat_dim / hidden / act from the existing head's MLP if accessible;
    otherwise falls back to standard values.
    """
    head = world_model.reward_head
    feat_dim = None
    units = 1024
    # Try to introspect existing head's MLP for feat_dim/hidden.
    try:
        mlp_net = head.net.net  # SymexpTwoHotHead.net (MLPHead).net (Sequential)
        for layer in mlp_net:
            if isinstance(layer, torch.nn.Linear):
                if feat_dim is None:
                    feat_dim = layer.in_features
                units = (
                    layer.out_features
                )  # last linear out_features will be bins / 1; we use mid-layer width
        # Take MLP hidden width = the FIRST Linear's out_features
        for layer in mlp_net:
            if isinstance(layer, torch.nn.Linear):
                units = layer.out_features
                break
    except AttributeError:
        feat_dim = getattr(world_model, "obs_dim", None)
        if feat_dim is None:
            feat_dim = OmegaConf.select(cfg, "world_model.obs_dim", default=None)
    if feat_dim is None:
        raise ValueError("Could not infer reward head feature dimension from model or config")
    new_head = BinaryRewardHead(
        int(feat_dim),
        layers=1,
        units=int(units),
        act="silu",
        init_logit=float(init_logit),
        pos_weight=float(pos_weight),
    )
    device = next(head.parameters()).device
    dtype = next(head.parameters()).dtype
    world_model.reward_head = new_head.to(device=device, dtype=dtype)
    print(
        f"[swap-head] replaced reward_head with BinaryRewardHead(feat_dim={feat_dim}, units={units}, "
        f"init_logit={init_logit}, pos_weight={pos_weight})",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.init.world_model_state_ckpt = args.world_model_ckpt
    # Force dataset → balanced-terminal variant; reuse existing dataset target's args.
    cfg.dataset._target_ = (
        "dreamervla.dataset.libero_balanced_terminal_dataset.LIBEROBalancedTerminalDataset"
    )
    cfg.dataset.sequence_length = int(args.sequence_length)

    print(f"[finetune] out_dir={out_dir}", flush=True)
    print(f"[finetune] wm_ckpt={args.world_model_ckpt}", flush=True)
    OmegaConf.save(cfg, out_dir / "finetune_config.yaml", resolve=True)

    # Build dataset
    dataset: LIBEROBalancedTerminalDataset = hydra.utils.instantiate(cfg.dataset)
    sampler = BalancedTerminalSampler(
        dataset,
        num_samples=int(args.max_steps) * int(args.batch_size),
        positive_ratio=float(args.positive_ratio),
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=bool(int(args.num_workers) > 0),
    )

    # Build WM
    world_model = hydra.utils.instantiate(cfg.world_model).to(
        device=device, dtype=torch.bfloat16
    )
    load_world_model_state(world_model, args.world_model_ckpt, reset_reward_head=False)

    if args.swap_binary_head:
        maybe_swap_binary_head(
            world_model, cfg, args.binary_init_logit, args.binary_pos_weight
        )

    # Freeze all except reward_head
    n_train, n_total = 0, 0
    for name, p in world_model.named_parameters():
        n_total += p.numel()
        if "reward_head" in name:
            p.requires_grad = True
            n_train += p.numel()
        else:
            p.requires_grad = False
    print(
        f"[finetune] trainable params = {n_train:,} / total = {n_total:,}", flush=True
    )

    trainable = [p for p in world_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    world_model.train()
    # Force frozen modules to eval to disable BN/dropout drift if present.
    for name, m in world_model.named_modules():
        if "reward_head" not in name and not any(
            "reward_head" in n for n, _ in m.named_parameters(recurse=False)
        ):
            m.eval()

    loss_mode = str(args.loss_mode)
    seq_len = int(args.sequence_length)
    print(f"[finetune] loss_mode = {loss_mode}", flush=True)

    if loss_mode == "per_window":
        from dreamervla.models.world_model.dreamerv3_torch import BinaryRewardHead as _BRH

        if not isinstance(world_model.reward_head, _BRH):
            raise RuntimeError(
                "loss_mode=per_window requires --swap-binary-head (BinaryRewardHead)"
            )
    if loss_mode == "reward_diffusion":
        gamma = float(args.diffusion_gamma)
        # decay[t] = gamma^(W-1-t)  → decay[W-1]=1, decay[0]=gamma^(W-1)
        decay_template = torch.tensor(
            [gamma ** (seq_len - 1 - t) for t in range(seq_len)],
            dtype=torch.float32,
            device=device,
        )  # [W]
        print(
            f"[finetune] reward_diffusion gamma={gamma}  "
            f"decay=[{float(decay_template[0]):.3f} … {float(decay_template[-1]):.3f}]",
            flush=True,
        )

    step = 0
    t0 = time.time()
    metrics_log: list[dict[str, Any]] = []
    for batch in loader:
        flat = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                flat[k] = v.to(device)
            else:
                flat[k] = v

        is_pos = flat.get("is_positive_window")
        if is_pos is not None and not isinstance(is_pos, torch.Tensor):
            is_pos = torch.as_tensor(is_pos)
        if is_pos is not None:
            is_pos = is_pos.to(device=device, dtype=torch.float32)

        if loss_mode == "per_window":
            # Bypass WM.loss: compute encoder + RSSM in no_grad, only run reward_head with grad.
            # NOTE: This WM is DreamerV3PixelRynnBackboneWorldModel — its encoder consumes
            # the precomputed `obs_embedding`, NOT raw `images`.
            obs_emb = flat["obs_embedding"]
            actions = flat["actions"]
            is_first = flat["is_first"]
            with torch.no_grad():
                enc = world_model.encoder(obs_emb)
                seq = world_model.rssm.observe(enc, actions, is_first)
                feat_full = world_model.feature(seq)  # [B, T, D]
            last_feat = feat_full[:, -1, :].detach()  # [B, D]
            reward_logits = world_model.reward_head(last_feat)  # [B, 1]
            loss = world_model.reward_head.loss(reward_logits, is_pos).mean()
            reward_loss = float(loss.detach().cpu())
            with torch.no_grad():
                preds = torch.sigmoid(reward_logits.squeeze(-1).float())
                reward_pred = float(preds.mean().cpu())
                pred_pos = float((preds >= 0.5).float().mean().cpu())
        elif loss_mode == "reward_diffusion":
            # Rewrite rewards: positive windows → decay_template; negative → zeros.
            B = is_pos.shape[0]
            rewards = decay_template.unsqueeze(0).expand(B, -1) * is_pos.unsqueeze(-1)
            flat["rewards"] = rewards
            out = world_model(flat)
            loss = out.get("_loss", out.get("loss"))
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError("world_model did not return loss tensor")
            reward_loss = float(out.get("reward_loss", torch.zeros(())).detach().cpu())
            reward_pred = float(
                out.get("reward_pred_mean", torch.zeros(())).detach().cpu()
            )
            pred_pos = -1.0
        else:  # per_step_sparse: original behavior
            out = world_model(flat)
            loss = out.get("_loss", out.get("loss"))
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError("world_model did not return loss tensor")
            reward_loss = float(out.get("reward_loss", torch.zeros(())).detach().cpu())
            reward_pred = float(
                out.get("reward_pred_mean", torch.zeros(())).detach().cpu()
            )
            pred_pos = -1.0

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            trainable, max_norm=float(args.grad_clip)
        )
        optimizer.step()

        # Diagnostics
        with torch.no_grad():
            pos_count = int(is_pos.sum()) if is_pos is not None else -1
            label_sum = (
                float(flat["rewards"].sum().cpu()) if "rewards" in flat else -1.0
            )
        if step % int(args.log_every) == 0:
            elapsed = max(time.time() - t0, 1e-6)
            print(
                f"[finetune] step={step:5d}  loss={float(loss.detach().cpu()):.4f}  "
                f"reward_loss={reward_loss:.4f}  reward_pred_mean={reward_pred:.4f}  "
                f"pred_pos_frac={pred_pos:.3f}  "
                f"pos_in_batch={pos_count}/{int(args.batch_size)}  label_sum={label_sum:.2f}  "
                f"gnorm={float(torch.as_tensor(gnorm).detach().cpu()):.3f}  fps={step / elapsed:.2f}",
                flush=True,
            )
        metrics_log.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "reward_loss": reward_loss,
                "reward_pred_mean": reward_pred,
                "pred_pos_frac": pred_pos,
                "pos_in_batch": pos_count,
                "label_sum": label_sum,
            }
        )

        if step > 0 and step % int(args.save_every) == 0:
            ckpt_path = out_dir / f"reward_head_step{step:06d}.ckpt"
            torch.save(
                {
                    "world_model": world_model.state_dict(),
                    "step": step,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[finetune] ckpt → {ckpt_path}", flush=True)

        step += 1
        if step >= int(args.max_steps):
            break

    final_ckpt = out_dir / "reward_head_final.ckpt"
    torch.save(
        {"world_model": world_model.state_dict(), "step": step, "args": vars(args)},
        final_ckpt,
    )
    print(f"\n[finetune] FINAL ckpt → {final_ckpt}", flush=True)

    (out_dir / "finetune_metrics.json").write_text(json.dumps(metrics_log, indent=2))
    print(f"[finetune] metrics → {out_dir / 'finetune_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
