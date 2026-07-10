from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dreamervla.diagnostics import wm_single_trajectory_overfit as diag


def test_epoch_batches_visit_each_window_once() -> None:
    starts = diag.sliding_window_starts(episode_len=12, sequence_len=5)
    batches = list(
        diag.iter_epoch_batches(
            starts,
            batch_size=3,
            rng=np.random.default_rng(7),
        )
    )

    visited = np.concatenate(batches)
    assert sorted(visited.tolist()) == starts.tolist()
    assert len(set(visited.tolist())) == len(starts)


def test_convergence_requires_consecutive_threshold_passes() -> None:
    tracker = diag.ConvergenceTracker(
        mse_threshold=0.03,
        cosine_threshold=0.95,
        required_passes=3,
    )

    assert tracker.observe(mse=0.02, cosine_similarity=0.96) is False
    assert tracker.observe(mse=0.04, cosine_similarity=0.97) is False
    assert tracker.streak == 0
    assert tracker.observe(mse=0.02, cosine_similarity=0.96) is False
    assert tracker.observe(mse=0.01, cosine_similarity=0.97) is False
    assert tracker.observe(mse=0.01, cosine_similarity=0.98) is True


class _TinyWorldModel(torch.nn.Module):
    num_hist = 1
    chunk_size = 1

    def __init__(self) -> None:
        super().__init__()
        self.prediction = torch.nn.Parameter(torch.zeros(2))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target = batch["obs_embedding"][:, 1, 0, :]
        prediction = self.prediction[None].expand_as(target)
        mse = torch.nn.functional.mse_loss(prediction, target)
        cosine = torch.nn.functional.cosine_similarity(
            prediction,
            target,
            dim=-1,
        ).mean()
        return {
            "_loss": mse,
            "hidden_mse": mse.detach(),
            "hidden_cosine_loss": (1.0 - cosine).detach(),
        }


def _tiny_episode() -> diag.EpisodeArrays:
    return diag.EpisodeArrays(
        hidden=np.ones((6, 1, 2), dtype=np.float32),
        lang=np.zeros((4,), dtype=np.float32),
        actions=np.zeros((6, 7), dtype=np.float32),
        rewards=np.zeros((6,), dtype=np.float32),
        proprio=np.zeros((6, 8), dtype=np.float32),
    )


def test_run_overfit_converges_and_writes_checkpoints(tmp_path: Path) -> None:
    settings = diag.RunSettings(
        max_epochs=20,
        batch_size=2,
        lr=0.2,
        grad_clip=1.0,
        eval_every=1,
        mse_threshold=0.03,
        cosine_threshold=0.95,
        required_passes=2,
        seed=3,
    )

    summary = diag.run_overfit(
        model=_TinyWorldModel(),
        episode=_tiny_episode(),
        settings=settings,
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert summary["status"] == "converged"
    assert summary["best_hidden_mse"] <= 0.03
    assert summary["best_cosine_similarity"] >= 0.95
    assert (tmp_path / "checkpoints" / "best.ckpt").is_file()
    assert (tmp_path / "checkpoints" / "final.ckpt").is_file()


def test_run_overfit_reports_not_converged_at_epoch_limit(tmp_path: Path) -> None:
    settings = diag.RunSettings(
        max_epochs=1,
        batch_size=2,
        lr=0.0,
        grad_clip=1.0,
        eval_every=1,
        mse_threshold=0.001,
        cosine_threshold=0.99,
        required_passes=2,
        seed=3,
    )

    summary = diag.run_overfit(
        model=_TinyWorldModel(),
        episode=_tiny_episode(),
        settings=settings,
        out_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert summary["status"] == "not_converged"
    assert summary["epochs_completed"] == 1
    assert summary["success_streak"] == 0
