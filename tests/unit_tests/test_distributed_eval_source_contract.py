"""Static integration contract for distributed standalone evaluation wiring."""

from pathlib import Path


def test_eval_runner_wires_task_sharding_progress_and_global_aggregation() -> None:
    root = Path(__file__).resolve().parents[2]
    base = (root / "dreamervla/runtime/libero_vla_evaluation_base.py").read_text(encoding="utf-8")
    runner = (root / "dreamervla/runners/libero_vla_evaluation_runner.py").read_text(
        encoding="utf-8"
    )

    assert "shard_libero_eval_tasks" in base
    assert "allocate_divisible_worker_budget" in base
    assert "AggregateProgress" in base
    assert "merge_libero_eval_rank_payloads" in base
    assert "all_gather_objects" in base
    assert "self._cotrain_eval_distributed" in runner
    assert "metrics_from_rank_payloads" in runner
    assert "must run on a single process" not in runner
