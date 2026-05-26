#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _abs(path: str | None) -> str | None:
    if not path:
        return path
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def _strip_prefix(key: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "module."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _load_state(module: torch.nn.Module, state_dict: dict[str, Any], *, dtype: torch.dtype | None = None) -> None:
    converted = {}
    target = module.state_dict()
    for key, value in state_dict.items():
        key = _strip_prefix(key)
        if key.startswith("reward_head.net.") and not key.startswith("reward_head.net.net."):
            candidate = key.replace("reward_head.net.", "reward_head.net.net.", 1)
            if candidate in target:
                key = candidate
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value) and dtype is not None:
            value = value.to(dtype=dtype)
        converted[key] = value
    missing, unexpected = module.load_state_dict(converted, strict=False)
    if missing:
        print(f"[load] missing={len(missing)} first={missing[:5]}")
    if unexpected:
        print(f"[load] unexpected={len(unexpected)} first={unexpected[:5]}")


def _flatten_time(x: torch.Tensor, keep: int | None = None) -> torch.Tensor:
    if keep is not None:
        x = x[:, :keep]
    return x.reshape(-1, *x.shape[2:])


def _action_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    diff = pred - target
    denom = target.square().mean().sqrt().clamp_min(1e-8)
    gripper_pred = pred[..., -1]
    gripper_target = target[..., -1]
    return {
        "mse": float(diff.square().mean().detach().cpu()),
        "rmse": float(diff.square().mean().sqrt().detach().cpu()),
        "mae": float(diff.abs().mean().detach().cpu()),
        "rel_rmse": float((diff.square().mean().sqrt() / denom).detach().cpu()),
        "xyz_mae": float(diff[..., :3].abs().mean().detach().cpu()),
        "rot_mae": float(diff[..., 3:6].abs().mean().detach().cpu()),
        "gripper_mae": float((gripper_pred - gripper_target).abs().mean().detach().cpu()),
        "gripper_sign_acc": float(((gripper_pred >= 0) == (gripper_target >= 0)).float().mean().detach().cpu()),
        "cosine_loss": float((1.0 - F.cosine_similarity(pred, target, dim=-1).mean()).detach().cpu()),
    }


def _hidden_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    diff = pred - target
    return {
        "mse": float(diff.square().mean().detach().cpu()),
        "rmse": float(diff.square().mean().sqrt().detach().cpu()),
        "mae": float(diff.abs().mean().detach().cpu()),
        "cosine_loss": float((1.0 - F.cosine_similarity(pred, target, dim=-1).mean()).detach().cpu()),
        "pred_norm": float(pred.norm(dim=-1).mean().detach().cpu()),
        "target_norm": float(target.norm(dim=-1).mean().detach().cpu()),
        "norm_ratio": float((pred.norm(dim=-1).mean() / target.norm(dim=-1).mean().clamp_min(1e-8)).detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--hidden-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else torch.float32
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    cfg = ckpt["cfg"]
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
    if args.hidden_dir:
        with open_dict(dataset_cfg):
            dataset_cfg.hidden_dir = args.hidden_dir
    dataset = hydra.utils.instantiate(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    wm_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world_model, resolve=True))
    world_model = hydra.utils.instantiate(wm_cfg).to(device=device, dtype=dtype)
    _load_state(world_model, ckpt["state_dicts"]["world_model"], dtype=dtype)
    world_model.eval()

    policy_cfg = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=True))
    with open_dict(policy_cfg):
        policy_cfg.init_action_head_ckpt = _abs(str(policy_cfg.init_action_head_ckpt))
    policy_init = hydra.utils.instantiate(policy_cfg).to(device=device, dtype=dtype).eval()
    policy_ckpt = hydra.utils.instantiate(policy_cfg).to(device=device, dtype=dtype).eval()
    _load_state(policy_ckpt, ckpt["state_dicts"]["policy"], dtype=dtype)

    rows = []
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if bidx >= args.num_batches:
                break
            model_batch = {}
            for key in ("images", "obs_embedding", "actions", "rewards", "dones", "is_first"):
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    value = value.to(device=device)
                    if key not in {"is_first"} and torch.is_floating_point(value):
                        value = value.to(dtype=dtype if key != "images" else torch.float32)
                model_batch[key] = value

            observed = world_model.observe_sequence(model_batch)
            pred_hidden = world_model.actor_input(observed["latent"])
            real_hidden = model_batch["obs_embedding"].to(device=device, dtype=pred_hidden.dtype)

            # Dataset actions use previous-action convention: actions[t+1] is raw action at t.
            steps = min(int(policy_cfg.time_horizon), real_hidden.shape[1] - 1)
            real_flat = _flatten_time(real_hidden, keep=steps)
            pred_flat = _flatten_time(pred_hidden, keep=steps)
            gt_first = _flatten_time(model_batch["actions"][:, 1 : steps + 1].to(dtype=torch.float32))

            real_chunk_init = policy_init({"mode": "sample", "hidden": real_flat, "deterministic": True, "return_chunk": True})[2]["action_chunk"].float()
            pred_chunk_init = policy_init({"mode": "sample", "hidden": pred_flat, "deterministic": True, "return_chunk": True})[2]["action_chunk"].float()
            real_chunk_ckpt = policy_ckpt({"mode": "sample", "hidden": real_flat, "deterministic": True, "return_chunk": True})[2]["action_chunk"].float()

            rows.append(
                {
                    "hidden_pred_vs_real": _hidden_metrics(pred_flat, real_flat),
                    "init_action_pred_hidden_vs_real_hidden_first": _action_metrics(pred_chunk_init[:, 0], real_chunk_init[:, 0]),
                    "init_action_real_hidden_vs_dataset_first": _action_metrics(real_chunk_init[:, 0], gt_first),
                    "init_action_pred_hidden_vs_dataset_first": _action_metrics(pred_chunk_init[:, 0], gt_first),
                    "ckpt_action_real_hidden_vs_dataset_first": _action_metrics(real_chunk_ckpt[:, 0], gt_first),
                    "ckpt_action_vs_init_action_on_real_hidden_first": _action_metrics(real_chunk_ckpt[:, 0], real_chunk_init[:, 0]),
                }
            )

    def mean_nested(key: str) -> dict[str, float]:
        names = rows[0][key].keys()
        return {name: sum(row[key][name] for row in rows) / len(rows) for name in names}

    result = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "dataset_hidden_dir": str(Path(str(dataset_cfg.hidden_dir)).resolve()),
        "num_batches": len(rows),
        "batch_size": args.batch_size,
        "sequence_length": int(dataset_cfg.sequence_length),
        "time_horizon": int(policy_cfg.time_horizon),
        "metrics": {key: mean_nested(key) for key in rows[0].keys()},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
