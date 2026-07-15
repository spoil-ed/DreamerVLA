"""Evaluate DINO token one-step prediction against persistence on fixed windows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from dreamervla.diagnostics.eval_chunkwm_closeloop import (
    _load_world_model_config,
    _load_world_model_state,
)
from dreamervla.models.embodiment.world_model import DinoTokenWorldModel
from dreamervla.utils.run_config import load_run_config


def deterministic_window_starts(
    *,
    length: int,
    window: int,
    max_windows: int,
) -> list[int]:
    """Choose fixed, evenly spaced valid windows over one trajectory."""

    count = int(length) - int(window) + 1
    if count <= 0:
        return []
    limit = int(max_windows)
    if limit <= 0 or limit >= count:
        return list(range(count))
    return sorted(
        set(np.linspace(0, count - 1, num=limit, dtype=np.int64).tolist())
    )


@torch.no_grad()
def one_step_token_predictions(
    world_model: DinoTokenWorldModel,
    *,
    tokens: torch.Tensor,
    proprio: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return model, target, and copy-previous tensors for identical DINO shifts."""

    expected = int(world_model.num_hist + world_model.num_pred)
    if tokens.ndim != 4 or int(tokens.shape[1]) != expected:
        raise ValueError(
            f"tokens must be [B,{expected},N,D], got {tuple(tokens.shape)}"
        )
    latent = world_model.encode(
        {"visual": tokens, "proprio": proprio},
        actions,
    )
    source = latent[:, : world_model.num_hist]
    predicted = world_model.predict(source)[..., : world_model.token_dim]
    normalized = latent[..., : world_model.token_dim]
    target = normalized[:, world_model.num_pred :]
    persistence = normalized[:, : world_model.num_hist]
    return predicted, target, persistence


def token_prediction_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Compute cosine, MSE, and relative L2 for every predicted frame."""

    if predicted.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {predicted.shape} != {target.shape}"
        )
    flat_pred = predicted.float().reshape(-1, int(np.prod(predicted.shape[-2:])))
    flat_target = target.float().reshape(-1, int(np.prod(target.shape[-2:])))
    cosine = F.cosine_similarity(flat_pred, flat_target, dim=-1)
    mse = (flat_pred - flat_target).square().mean(dim=-1)
    relative_l2 = (flat_pred - flat_target).norm(dim=-1) / flat_target.norm(
        dim=-1
    ).clamp_min(1.0e-8)
    return {
        "cos": cosine.cpu().numpy(),
        "mse": mse.cpu().numpy(),
        "rel_l2": relative_l2.cpu().numpy(),
    }


def load_dino_token_world_model(
    checkpoint: str,
    *,
    device: torch.device,
    config_path: str | None = None,
) -> DinoTokenWorldModel:
    """Load a split warmup or regular checkpoint using its persisted Hydra config."""

    checkpoint_path = Path(checkpoint).expanduser()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model_cfg = _load_world_model_config(
        payload,
        checkpoint_path,
        config_path,
    )
    config = (
        world_model_cfg
        if OmegaConf.is_config(world_model_cfg)
        else OmegaConf.create(world_model_cfg)
    )
    model = hydra.utils.instantiate(config)
    if not isinstance(model, DinoTokenWorldModel):
        raise TypeError(
            "DINO token evaluation requires DinoTokenWorldModel, got "
            f"{type(model).__name__}"
        )
    model.load_state_dict(_load_world_model_state(payload, checkpoint_path), strict=True)
    print(
        f"[load] global_step={payload.get('global_step')} "
        f"epoch={payload.get('epoch')} target={config.get('_target_')}",
        flush=True,
    )
    return model.to(device=device, dtype=torch.float32).eval()


def _payload_runner_config(value: object, *, key: str) -> DictConfig | None:
    if value is None:
        return None
    if isinstance(value, DictConfig):
        return value
    if isinstance(value, Mapping):
        config = OmegaConf.create(dict(value))
        if isinstance(config, DictConfig):
            return config
    raise TypeError(f"checkpoint payload {key} must be a mapping")


def _runner_config_from_checkpoint(
    payload: dict,
    checkpoint_path: Path,
    config_path: str | None,
) -> DictConfig:
    for key in ("cfg", "config"):
        config = _payload_runner_config(payload.get(key), key=key)
        if config is not None and OmegaConf.select(config, "dataset.valid") is not None:
            return config
    try:
        config = load_run_config(config_path or checkpoint_path)
    except FileNotFoundError as exc:
        raise ValueError(
            "DINO token evaluation requires a run config containing dataset.valid so "
            "frameskip, action concatenation, split, and normalization match training"
        ) from exc
    if OmegaConf.select(config, "dataset.valid") is not None:
        return config
    raise ValueError(
        "DINO token evaluation requires the run config's dataset.valid section so "
        "frameskip, action concatenation, split, and normalization match training"
    )


def load_dino_token_checkpoint(
    checkpoint: str,
    *,
    device: torch.device,
    config_path: str | None = None,
) -> tuple[DinoTokenWorldModel, object]:
    """Load the world model and its complete runner/data config once."""

    checkpoint_path = Path(checkpoint).expanduser()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model_cfg = _load_world_model_config(payload, checkpoint_path, config_path)
    model_cfg = (
        world_model_cfg
        if OmegaConf.is_config(world_model_cfg)
        else OmegaConf.create(world_model_cfg)
    )
    model = hydra.utils.instantiate(model_cfg)
    if not isinstance(model, DinoTokenWorldModel):
        raise TypeError(
            "DINO token evaluation requires DinoTokenWorldModel, got "
            f"{type(model).__name__}"
        )
    model.load_state_dict(
        _load_world_model_state(payload, checkpoint_path),
        strict=True,
    )
    runner_cfg = _runner_config_from_checkpoint(
        payload,
        checkpoint_path,
        config_path,
    )
    return model.to(device=device, dtype=torch.float32).eval(), runner_cfg


def _summary(metrics: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    if not metrics:
        raise RuntimeError("evaluation produced no valid prediction windows")
    merged = {
        key: np.concatenate([item[key] for item in metrics], axis=0)
        for key in ("cos", "mse", "rel_l2")
    }
    return {
        "cos": float(merged["cos"].mean()),
        "cos_std": float(merged["cos"].std()),
        "mse": float(merged["mse"].mean()),
        "mse_std": float(merged["mse"].std()),
        "rel_l2": float(merged["rel_l2"].mean()),
        "rel_l2_std": float(merged["rel_l2"].std()),
        "predicted_frames": int(merged["cos"].shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-demos", type=int, default=16)
    parser.add_argument("--windows-per-demo", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    model, runner_cfg = load_dino_token_checkpoint(
        args.ckpt,
        device=device,
        config_path=args.config,
    )
    dataset = hydra.utils.instantiate(OmegaConf.select(runner_cfg, "dataset.valid"))
    selected_indices = dataset.evaluation_indices(
        max_trajectories=int(args.num_demos),
        windows_per_trajectory=int(args.windows_per_demo),
    )
    dataloader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )
    print(
        f"[data] split=valid trajectories="
        f"{min(len(dataset.trajectory_indices), int(args.num_demos))} "
        f"windows={len(selected_indices)} frameskip={dataset.frameskip}",
        flush=True,
    )

    model_metrics: list[dict[str, np.ndarray]] = []
    persistence_metrics: list[dict[str, np.ndarray]] = []
    evaluated_windows = 0
    for batch in dataloader:
        predicted, target, persistence = one_step_token_predictions(
            model,
            tokens=batch["obs_embedding"].to(device=device, dtype=torch.float32),
            proprio=batch["proprio"].to(device=device, dtype=torch.float32),
            actions=batch["current_actions"].to(device=device, dtype=torch.float32),
        )
        model_metrics.append(token_prediction_metrics(predicted, target))
        persistence_metrics.append(token_prediction_metrics(persistence, target))
        evaluated_windows += int(batch["obs_embedding"].shape[0])

    model_summary = _summary(model_metrics)
    persistence_summary = _summary(persistence_metrics)
    result = {
        "checkpoint": str(Path(args.ckpt).expanduser()),
        "num_demos": min(len(dataset.trajectory_indices), int(args.num_demos)),
        "evaluated_windows": int(evaluated_windows),
        "frameskip": int(dataset.frameskip),
        "teacher_forced_predictions_per_window": int(model.num_hist),
        "model": model_summary,
        "persistence": persistence_summary,
        "model_minus_persistence": {
            "cos": float(model_summary["cos"] - persistence_summary["cos"]),
            "mse": float(model_summary["mse"] - persistence_summary["mse"]),
            "rel_l2": float(
                model_summary["rel_l2"] - persistence_summary["rel_l2"]
            ),
        },
    }
    print("\n=== DINO token one-step vs persistence ===")
    print(
        f"model       cos={model_summary['cos']:.6f} "
        f"mse={model_summary['mse']:.6f} rel_l2={model_summary['rel_l2']:.6f}"
    )
    print(
        f"persistence cos={persistence_summary['cos']:.6f} "
        f"mse={persistence_summary['mse']:.6f} "
        f"rel_l2={persistence_summary['rel_l2']:.6f}"
    )
    print(
        "delta(model-persistence) "
        f"cos={result['model_minus_persistence']['cos']:+.6f} "
        f"mse={result['model_minus_persistence']['mse']:+.6f}"
    )
    if args.out:
        output = Path(args.out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[saved] {output}")


if __name__ == "__main__":
    main()
