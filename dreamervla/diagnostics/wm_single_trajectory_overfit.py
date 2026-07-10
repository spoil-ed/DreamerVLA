#!/usr/bin/env python3
"""Random-init world-model overfit check on one LIBERO trajectory."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamervla.utils.paths import data_path

DEFAULT_HDF5_FILENAME = "open_the_middle_drawer_of_the_cabinet_demo.hdf5"


@dataclass(frozen=True)
class EpisodeArrays:
    """Aligned arrays loaded from one hidden/raw LIBERO demo."""

    hidden: np.ndarray
    lang: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    proprio: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            int(self.hidden.shape[0]),
            int(self.actions.shape[0]),
            int(self.rewards.shape[0]),
            int(self.proprio.shape[0]),
        }
        if len(lengths) != 1:
            raise ValueError("episode arrays must have the same leading length")

    @property
    def episode_len(self) -> int:
        """Return the aligned episode length."""

        return int(self.hidden.shape[0])


@dataclass(frozen=True)
class RunSettings:
    """Optimization and convergence settings for one overfit run."""

    max_epochs: int = 200
    batch_size: int = 8
    lr: float = 1.0e-4
    grad_clip: float = 1.0
    eval_every: int = 5
    mse_threshold: float = 0.03
    cosine_threshold: float = 0.95
    required_passes: int = 3
    seed: int = 23

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lr < 0.0:
            raise ValueError("lr must be non-negative")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.mse_threshold < 0.0:
            raise ValueError("mse_threshold must be non-negative")
        if not -1.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be in [-1, 1]")
        if self.required_passes <= 0:
            raise ValueError("required_passes must be positive")


@dataclass
class ConvergenceTracker:
    """Track consecutive full-evaluation threshold passes."""

    mse_threshold: float
    cosine_threshold: float
    required_passes: int
    streak: int = 0

    def __post_init__(self) -> None:
        if self.mse_threshold < 0.0:
            raise ValueError("mse_threshold must be non-negative")
        if not -1.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be in [-1, 1]")
        if self.required_passes <= 0:
            raise ValueError("required_passes must be positive")

    def observe(self, *, mse: float, cosine_similarity: float) -> bool:
        """Record one evaluation and return whether convergence is confirmed."""

        passed = (
            mse <= self.mse_threshold
            and cosine_similarity >= self.cosine_threshold
        )
        self.streak = self.streak + 1 if passed else 0
        return self.streak >= self.required_passes


def sliding_window_starts(*, episode_len: int, sequence_len: int) -> np.ndarray:
    """Return every valid sliding-window start for one episode."""

    count = episode_len - sequence_len + 1
    if count <= 0:
        raise ValueError(
            f"episode length {episode_len} is shorter than sequence length "
            f"{sequence_len}"
        )
    return np.arange(count, dtype=np.int64)


def iter_epoch_batches(
    starts: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield a shuffled epoch in batches, visiting each start exactly once."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    shuffled = rng.permutation(starts)
    for offset in range(0, len(shuffled), batch_size):
        yield shuffled[offset : offset + batch_size]


def _append_json(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def _episode_tensors(
    episode: EpisodeArrays,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "hidden": torch.as_tensor(episode.hidden, device=device),
        "lang": torch.as_tensor(episode.lang, device=device),
        "actions": torch.as_tensor(episode.actions, device=device),
        "rewards": torch.as_tensor(episode.rewards, device=device),
        "proprio": torch.as_tensor(episode.proprio, device=device),
    }


def _make_batch(
    episode: dict[str, torch.Tensor],
    batch_starts: np.ndarray,
    *,
    sequence_len: int,
) -> dict[str, torch.Tensor]:
    hidden = episode["hidden"]
    actions = episode["actions"]
    rewards = episode["rewards"]
    proprio = episode["proprio"]
    lang = episode["lang"]
    device = hidden.device
    starts = torch.as_tensor(batch_starts, device=device, dtype=torch.long)
    offsets = torch.arange(sequence_len, device=device, dtype=torch.long)
    frame_indices = starts[:, None] + offsets[None]
    flat_indices = frame_indices.reshape(-1)
    batch_size = len(batch_starts)
    return {
        "obs_embedding": hidden.index_select(0, flat_indices).reshape(
            batch_size,
            sequence_len,
            *hidden.shape[1:],
        ),
        "current_actions": actions.index_select(0, flat_indices).reshape(
            batch_size,
            sequence_len,
            actions.shape[-1],
        ),
        "actions": torch.zeros(
            batch_size,
            sequence_len,
            actions.shape[-1],
            device=device,
        ),
        "proprio": proprio.index_select(0, flat_indices).reshape(
            batch_size,
            sequence_len,
            proprio.shape[-1],
        ),
        "rewards": rewards.index_select(0, flat_indices).reshape(
            batch_size,
            sequence_len,
        ),
        "lang_emb": lang[None].expand(batch_size, -1),
    }


def _autocast(device: torch.device) -> Any:
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def evaluate_all_windows(
    model: torch.nn.Module,
    episode: dict[str, torch.Tensor],
    starts: np.ndarray,
    *,
    sequence_len: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate mean WM metrics over every window in the selected demo."""

    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    weighted_mse = 0.0
    weighted_cosine_loss = 0.0
    sample_count = 0
    with torch.inference_mode(), _autocast(device):
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            batch = _make_batch(
                episode,
                batch_starts,
                sequence_len=sequence_len,
            )
            output = model(batch)
            count = len(batch_starts)
            weighted_loss += float(output["_loss"].float().cpu()) * count
            weighted_mse += float(output["hidden_mse"].float().cpu()) * count
            weighted_cosine_loss += (
                float(output["hidden_cosine_loss"].float().cpu()) * count
            )
            sample_count += count
    model.train(was_training)
    return {
        "loss": weighted_loss / sample_count,
        "hidden_mse": weighted_mse / sample_count,
        "cosine_similarity": 1.0 - weighted_cosine_loss / sample_count,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "world_model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "initialization": "random",
        },
        path,
    )


def run_overfit(
    *,
    model: torch.nn.Module,
    episode: EpisodeArrays,
    settings: RunSettings,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train a world model on one trajectory until thresholds or epoch limit."""

    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    model = model.to(device).train()
    episode_tensors = _episode_tensors(episode, device)
    history = int(model.num_hist)
    chunk = int(model.chunk_size)
    sequence_len = history + chunk
    starts = sliding_window_starts(
        episode_len=episode.episode_len,
        sequence_len=sequence_len,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings.lr,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    tracker = ConvergenceTracker(
        mse_threshold=settings.mse_threshold,
        cosine_threshold=settings.cosine_threshold,
        required_passes=settings.required_passes,
    )

    baseline = evaluate_all_windows(
        model,
        episode_tensors,
        starts,
        sequence_len=sequence_len,
        batch_size=settings.batch_size,
        device=device,
    )
    _append_json(metrics_path, {"event": "eval", "epoch": 0, **baseline})
    best = dict(baseline)
    best_epoch = 0
    _save_checkpoint(
        checkpoints_dir / "best.ckpt",
        model=model,
        epoch=0,
        metrics=best,
    )

    status = "not_converged"
    final_eval = dict(baseline)
    epochs_completed = 0
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        weighted_loss = 0.0
        weighted_grad_norm = 0.0
        sample_count = 0
        for batch_starts in iter_epoch_batches(
            starts,
            batch_size=settings.batch_size,
            rng=rng,
        ):
            batch = _make_batch(
                episode_tensors,
                batch_starts,
                sequence_len=sequence_len,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output = model(batch)
                loss = output["_loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=settings.grad_clip,
            )
            optimizer.step()
            count = len(batch_starts)
            weighted_loss += float(loss.detach().float().cpu()) * count
            weighted_grad_norm += float(torch.as_tensor(grad_norm).float().cpu()) * count
            sample_count += count

        epochs_completed = epoch
        train_record = {
            "event": "train_epoch",
            "epoch": epoch,
            "train_loss": weighted_loss / sample_count,
            "grad_norm": weighted_grad_norm / sample_count,
        }
        _append_json(metrics_path, train_record)
        print(
            f"[wm-overfit] epoch={epoch}/{settings.max_epochs} "
            f"train_loss={train_record['train_loss']:.6f}",
            flush=True,
        )

        should_evaluate = (
            epoch % settings.eval_every == 0 or epoch == settings.max_epochs
        )
        if not should_evaluate:
            continue
        final_eval = evaluate_all_windows(
            model,
            episode_tensors,
            starts,
            sequence_len=sequence_len,
            batch_size=settings.batch_size,
            device=device,
        )
        converged = tracker.observe(
            mse=final_eval["hidden_mse"],
            cosine_similarity=final_eval["cosine_similarity"],
        )
        eval_record = {
            "event": "eval",
            "epoch": epoch,
            "success_streak": tracker.streak,
            **final_eval,
        }
        _append_json(metrics_path, eval_record)
        print(
            f"[wm-overfit] eval epoch={epoch} "
            f"mse={final_eval['hidden_mse']:.6f} "
            f"cos={final_eval['cosine_similarity']:.6f} "
            f"streak={tracker.streak}/{settings.required_passes}",
            flush=True,
        )
        candidate = (
            final_eval["hidden_mse"],
            -final_eval["cosine_similarity"],
        )
        incumbent = (best["hidden_mse"], -best["cosine_similarity"])
        if candidate < incumbent:
            best = dict(final_eval)
            best_epoch = epoch
            _save_checkpoint(
                checkpoints_dir / "best.ckpt",
                model=model,
                epoch=epoch,
                metrics=best,
            )
        if converged:
            status = "converged"
            break

    _save_checkpoint(
        checkpoints_dir / "final.ckpt",
        model=model,
        epoch=epochs_completed,
        metrics=final_eval,
    )
    summary: dict[str, Any] = {
        "status": status,
        "initialization": "random",
        "episode_len": episode.episode_len,
        "num_windows": int(len(starts)),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "baseline_hidden_mse": baseline["hidden_mse"],
        "baseline_cosine_similarity": baseline["cosine_similarity"],
        "best_hidden_mse": best["hidden_mse"],
        "best_cosine_similarity": best["cosine_similarity"],
        "final_hidden_mse": final_eval["hidden_mse"],
        "final_cosine_similarity": final_eval["cosine_similarity"],
        "mse_threshold": settings.mse_threshold,
        "cosine_threshold": settings.cosine_threshold,
        "required_passes": settings.required_passes,
        "success_streak": tracker.streak,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the one-command overfit diagnostic arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--task", default="openvla_onetraj_libero")
    parser.add_argument("--hdf5-filename", default=DEFAULT_HDF5_FILENAME)
    parser.add_argument("--hidden-hdf5", type=Path, default=None)
    parser.add_argument("--raw-hdf5", type=Path, default=None)
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_path("outputs/world_model_probe/single_trajectory_overfit"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--mse-threshold", type=float, default=0.03)
    parser.add_argument("--cosine-threshold", type=float, default=0.95)
    parser.add_argument("--required-passes", type=int, default=3)
    return parser.parse_args(argv)
