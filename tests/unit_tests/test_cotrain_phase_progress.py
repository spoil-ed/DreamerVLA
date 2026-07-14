from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dreamervla.runners.cotrain_runner import (
    _EvaluationProgressMonitor,
    _read_manual_cotrain_progress_snapshot,
)
from dreamervla.workers.actor.learner_worker import LearnerWorker
from dreamervla.workers.env import trajectory_env_worker
from dreamervla.workers.env.evaluation_env_worker import EvaluationEnvironmentWorker


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


def test_eval_progress_rate_uses_completed_episodes(tmp_path) -> None:
    (tmp_path / "eval.json").write_text(
        json.dumps(
            {
                "role": "eval_env",
                "env_rank": 0,
                "done": 570,
                "total": 3800,
                "episodes_completed": 17,
                "episodes_successful": 16,
                "finished": False,
            }
        ),
        encoding="utf-8",
    )
    reports: list[tuple[int, int, str]] = []

    monitor = _EvaluationProgressMonitor(
        tmp_path,
        lambda done, total, _desc, **kwargs: reports.append((done, total, str(kwargs["status"]))),
        desc="eval/00000000",
        total_episodes=100,
    )
    monitor.report(force=True)

    assert reports == [(17, 100, "completed=17 successes=16 success_rate=0.941 chunks=570/3800")]


def test_real_rollout_progress_aggregates_completed_trajectories(tmp_path) -> None:
    for rank, completed, successes, chunks in ((0, 3, 2, 40), (1, 4, 3, 50)):
        (tmp_path / f"real_{rank}.json").write_text(
            json.dumps(
                {
                    "role": "real_env",
                    "env_rank": rank,
                    "done": chunks,
                    "total": 100,
                    "episodes_completed": completed,
                    "episodes_successful": successes,
                    "finished": False,
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "wm.json").write_text(
        json.dumps(
            {
                "role": "wm_env",
                "env_rank": 0,
                "done": 1000,
                "total": 2000,
                "episodes_completed": 100,
                "episodes_successful": 100,
            }
        ),
        encoding="utf-8",
    )
    reports: list[tuple[int, int, str, str]] = []
    monitor = _EvaluationProgressMonitor(
        tmp_path,
        lambda done, total, _desc, **kwargs: reports.append(
            (done, total, str(kwargs["unit"]), str(kwargs["status"]))
        ),
        desc="cotrain-real-rollout/00000001",
        total_episodes=10,
        roles={"real_env"},
        unit="trajectory",
    )

    monitor.report(force=True)

    assert reports == [
        (
            7,
            10,
            "trajectory",
            "completed=7 successes=5 success_rate=0.714 chunks=90/200",
        )
    ]


def test_eval_env_pins_osmesa_before_env_build(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    worker = EvaluationEnvironmentWorker(
        env_cfg={"target": "unused", "render_backend": "osmesa"},
        num_slots=1,
        rollout_epoch=1,
        max_steps_per_rollout_epoch=1,
        num_action_chunks=1,
        task_ids=[0],
    )
    worker.local_rank = 0

    monkeypatch.setattr(
        trajectory_env_worker,
        "apply_libero_render_regime",
        lambda backend, shard_id, gpu_pool: events.append(
            ("helper", backend, int(shard_id), list(gpu_pool))
        ),
    )
    monkeypatch.setattr(
        trajectory_env_worker,
        "_build_env_from_cfg",
        lambda cfg: events.append(("build", cfg.get("render_backend"))) or object(),
    )

    worker.init()

    assert events[:2] == [("helper", "osmesa", 0, []), ("build", "osmesa")]


def test_eval_env_runs_osmesa_slots_in_spawned_processes(monkeypatch) -> None:
    events: list[tuple[Any, ...]] = []
    worker = EvaluationEnvironmentWorker(
        env_cfg={
            "target": "unused",
            "render_backend": "osmesa",
            "spawn_env_slots": True,
        },
        num_slots=25,
        rollout_epoch=4,
        max_steps_per_rollout_epoch=304,
        num_action_chunks=8,
        task_ids=list(range(10)),
    )

    monkeypatch.setattr(
        worker,
        "_init_spawned_env_slots",
        lambda: events.append(("spawn", worker.num_slots)),
    )
    monkeypatch.setattr(
        worker,
        "_bootstrap_wm_initial_latents_from_replay",
        lambda: events.append(("bootstrap",)),
    )
    monkeypatch.setattr(
        worker,
        "_apply_pending_component_states",
        lambda: events.append(("state",)),
    )

    worker.init()

    assert events == [("spawn", 25), ("bootstrap",), ("state",)]


def test_cotrain_eval_defaults_to_25_parallel_osmesa_slots() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        project_root / "configs" / "dreamervla" / "openvla_onetraj_libero_cotrain.yaml"
    )

    assert cfg.manual_cotrain.eval_protocol.render_backend == "osmesa"
    assert cfg.manual_cotrain.eval_protocol.num_envs == 25
    assert cfg.manual_cotrain.eval_protocol.spawn_env_slots is True
