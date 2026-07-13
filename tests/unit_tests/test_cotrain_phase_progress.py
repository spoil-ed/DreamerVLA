from __future__ import annotations

import json

from dreamervla.runners.cotrain_runner import (
    _read_manual_cotrain_progress_snapshot,
)
from dreamervla.workers.actor.learner_worker import LearnerWorker


def test_env_progress_snapshot_filters_real_and_imagined_roles(tmp_path) -> None:
    (tmp_path / "real.json").write_text(
        json.dumps(
            {
                "role": "real_env",
                "env_rank": 0,
                "done": 8,
                "total": 8,
                "finished": True,
                "global_step": 1,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "wm.json").write_text(
        json.dumps(
            {
                "role": "wm_env",
                "env_rank": 1,
                "done": 128,
                "total": 256,
                "finished": False,
                "global_step": 1,
            }
        ),
        encoding="utf-8",
    )

    real = _read_manual_cotrain_progress_snapshot(
        tmp_path,
        roles={"real_env"},
    )
    imagined = _read_manual_cotrain_progress_snapshot(
        tmp_path,
        roles={"wm_env"},
    )

    assert (real.done, real.total) == (8, 8)
    assert (imagined.done, imagined.total) == (128, 256)
    assert "wm_env" not in str(real.status)
    assert "real_env" not in str(imagined.status)


def test_step_local_wmcls_update_persists_monotonic_progress(tmp_path) -> None:
    worker = object.__new__(LearnerWorker)
    worker.train_cfg = {"mode": "wm_classifier_only", "early_stop_min_delta": 0.0}
    worker._train_progress_path = tmp_path / "learner.json"
    worker._train_progress = {}
    worker._dreamervla_wm_update_once = lambda: {"wm/loss": 1.0}
    worker._dreamervla_classifier_update_once = lambda: {"cls/loss": 1.0}
    worker._calibrate_classifier_threshold = lambda: {}

    metrics = worker.update_current_step("cotrain", 3, 3)
    payload = json.loads(worker._train_progress_path.read_text(encoding="utf-8"))

    assert metrics["learner/updates"] == 3.0
    assert payload["active"] is False
    assert payload["phase"] == "done"
    assert payload["train_step"] == 3
    assert payload["total_train_steps"] == 3
    assert payload["wm_step"] == 3
    assert payload["cls_step"] == 3
