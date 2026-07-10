from __future__ import annotations

import numpy as np

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
