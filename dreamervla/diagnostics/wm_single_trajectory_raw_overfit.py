#!/usr/bin/env python3
"""Overfit a small raw-state world model on one LIBERO trajectory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from dreamervla.utils.paths import data_path


@dataclass(frozen=True)
class RawEpisode:
    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        if self.states.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("states and actions must be rank-2 arrays")
        if self.states.shape[0] != self.actions.shape[0]:
            raise ValueError("states and actions must have equal length")
        if self.states.shape[0] < 2:
            raise ValueError("episode must contain at least two frames")


class RawStateWorldModel(nn.Module):
    """Tiny MLP dynamics model for a direct raw-state sanity check."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        width = max(64, min(512, input_dim * 4))
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def load_raw_episode(path: Path, demo_key: str) -> RawEpisode:
    """Load proprioceptive state and actions directly from raw LIBERO HDF5."""

    if not path.is_file():
        raise FileNotFoundError(f"raw HDF5 not found: {path}")
    with h5py.File(path, "r") as hdf5:
        demo = hdf5["data"][demo_key]
        states = np.concatenate(
            [
                np.asarray(demo["obs"]["ee_pos"], dtype=np.float32),
                np.asarray(demo["obs"]["ee_ori"], dtype=np.float32),
                np.asarray(demo["obs"]["gripper_states"], dtype=np.float32),
            ],
            axis=-1,
        )
        actions = np.asarray(demo["actions"], dtype=np.float32)
    return RawEpisode(states=states, actions=actions)


def _windows(
    episode: RawEpisode,
    history: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = episode.states.shape[0] - history
    if count <= 0:
        raise ValueError(f"episode is too short for history={history}")
    states = torch.as_tensor(episode.states, device=device)
    actions = torch.as_tensor(episode.actions, device=device)
    inputs = torch.cat(
        [
            torch.stack([states[i : i + history].flatten() for i in range(count)]),
            torch.stack([actions[i : i + history].flatten() for i in range(count)]),
        ],
        dim=-1,
    )
    targets = states[history:]
    return inputs, targets


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> dict[str, float]:
    prediction_raw = prediction * state_std + state_mean
    target_raw = target * state_std + state_mean
    mse = torch.nn.functional.mse_loss(prediction_raw, target_raw)
    cosine = torch.nn.functional.cosine_similarity(
        prediction_raw,
        target_raw,
        dim=-1,
    ).mean()
    normalized_mse = torch.nn.functional.mse_loss(prediction, target)
    return {
        "mse": float(mse.detach().cpu()),
        "normalized_mse": float(normalized_mse.detach().cpu()),
        "cosine_similarity": float(cosine.detach().cpu()),
    }


def run_overfit(
    *,
    episode: RawEpisode,
    out_dir: Path,
    device: torch.device,
    history: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    mse_threshold: float,
    cosine_threshold: float,
    required_passes: int,
    seed: int,
) -> dict[str, Any]:
    """Train and evaluate the raw-state model on every window of one episode."""

    if history <= 0 or max_epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("history, max_epochs, batch_size, and lr must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs, targets_raw = _windows(episode, history, device)
    state_mean = targets_raw.mean(dim=0)
    state_std = targets_raw.std(dim=0).clamp_min(1.0e-4)
    targets = (targets_raw - state_mean) / state_std
    model = RawStateWorldModel(inputs.shape[-1], targets.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    best: dict[str, float] | None = None
    streak = 0
    status = "not_converged"
    final_metrics: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(inputs.shape[0])
        losses: list[float] = []
        for offset in range(0, len(order), batch_size):
            index = torch.as_tensor(order[offset : offset + batch_size], device=device)
            prediction = model(inputs.index_select(0, index))
            loss = torch.nn.functional.mse_loss(prediction, targets.index_select(0, index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            prediction = model(inputs)
            final_metrics = _metrics(prediction, targets, state_mean, state_std)
        passed = (
            final_metrics["normalized_mse"] <= mse_threshold
            and final_metrics["cosine_similarity"] >= cosine_threshold
        )
        streak = streak + 1 if passed else 0
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "success_streak": streak,
            **final_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"[wm-raw-overfit] epoch={epoch}/{max_epochs} "
            f"mse={final_metrics['normalized_mse']:.6f} "
            f"cos={final_metrics['cosine_similarity']:.6f} "
            f"streak={streak}/{required_passes}",
            flush=True,
        )
        if best is None or final_metrics["normalized_mse"] < best["normalized_mse"]:
            best = dict(final_metrics)
            torch.save(
                {
                    "model": model.state_dict(),
                    "state_mean": state_mean.cpu(),
                    "state_std": state_std.cpu(),
                    "history": history,
                    "metrics": best,
                },
                out_dir / "raw_wm.ckpt",
            )
        if streak >= required_passes:
            status = "converged"
            break

    summary = {
        "status": status,
        "epochs_completed": epoch,
        "history": history,
        "num_windows": int(inputs.shape[0]),
        "best": best,
        "final": final_metrics,
        "mse_threshold": mse_threshold,
        "cosine_threshold": cosine_threshold,
        "required_passes": required_passes,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--raw-hdf5",
        type=Path,
        default=data_path(
            "processed_data/OpenVLA_Onetraj_LIBERO_libero_goal/"
            "no_noops_t_256_remaining_reward/"
            "open_the_middle_drawer_of_the_cabinet_demo.hdf5"
        ),
    )
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_path("outputs/world_model_probe/single_trajectory_raw_overfit"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--mse-threshold", type=float, default=0.03)
    parser.add_argument("--cosine-threshold", type=float, default=0.95)
    parser.add_argument("--required-passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    episode = load_raw_episode(args.raw_hdf5, args.demo_key)
    plan = {
        "initialization": "random",
        "raw_hdf5": str(args.raw_hdf5),
        "demo_key": args.demo_key,
        "history": args.history,
        "max_epochs": args.max_epochs,
        "device": args.device,
    }
    if not args.run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    summary = run_overfit(
        episode=episode,
        out_dir=args.out_dir,
        device=device,
        history=args.history,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mse_threshold=args.mse_threshold,
        cosine_threshold=args.cosine_threshold,
        required_passes=args.required_passes,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
