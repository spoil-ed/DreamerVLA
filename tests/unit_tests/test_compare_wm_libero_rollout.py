from __future__ import annotations

import numpy as np
import pytest

from dreamervla.diagnostics.compare_wm_libero_rollout import (
    _comparison_frame,
    _per_frame_pixel_metrics,
    _pixel_metrics_by_view,
    _select_trajectory_rows,
    _source_panel,
)


def test_select_trajectory_rows_is_outcome_aligned_and_deterministic() -> None:
    rows = [
        {"task_id": 0, "episode_id": 7, "horizon": 120, "success": True, "file": "b.h5"},
        {"task_id": 0, "episode_id": 3, "horizon": 120, "success": True, "file": "a.h5"},
        {"task_id": 0, "episode_id": 4, "horizon": 120, "success": False, "file": "c.h5"},
        {"task_id": 1, "episode_id": 0, "horizon": 120, "success": False, "file": "d.h5"},
    ]

    selected = _select_trajectory_rows(
        rows,
        task_id=0,
        outcomes=("success", "failure"),
        min_length=103,
    )

    assert [row["episode_id"] for row in selected] == [3, 4]


def test_select_trajectory_rows_requires_requested_horizon() -> None:
    rows = [{"task_id": 0, "episode_id": 0, "horizon": 20, "success": True}]
    with pytest.raises(RuntimeError, match="at least 103 frames"):
        _select_trajectory_rows(rows, task_id=0, outcomes=("success",), min_length=103)


def test_video_panels_have_even_codec_safe_dimensions() -> None:
    views = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    panel = _source_panel(views, source_label="source", step=0, phase="test")
    comparison = _comparison_frame(views, views, views, views, step=3, warmup_frames=3)

    assert panel.shape == (92, 32, 3)
    assert comparison.shape == (92, 128, 3)
    assert panel.shape[0] % 2 == panel.shape[1] % 2 == 0
    assert comparison.shape[0] % 2 == comparison.shape[1] % 2 == 0


def test_decoder_attribution_metrics_report_joint_views_and_time() -> None:
    target = np.zeros((2, 2, 4, 4, 3), dtype=np.uint8)
    oracle = target.copy()
    imagined = target.copy()
    oracle[:, 1] = 51
    imagined[:, 0] = 102
    imagined[:, 1] = 51

    summary = _pixel_metrics_by_view(oracle, target)
    per_frame = _per_frame_pixel_metrics(oracle, imagined, target)

    assert summary["by_view"]["base"]["mae"] == pytest.approx(0.0)
    assert summary["by_view"]["wrist"]["mae"] == pytest.approx(0.2)
    assert len(per_frame) == 2
    assert per_frame[0]["decoder_only"]["base"]["mae"] == pytest.approx(0.0)
    assert per_frame[0]["decoder_only"]["wrist"]["mae"] == pytest.approx(0.2)
    assert per_frame[1]["wm_decoder"]["base"]["mae"] == pytest.approx(0.4)
