"""Shared evaluation metric summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def shard_libero_eval_tasks(
    task_ids: Iterable[int],
    *,
    rank: int,
    world_size: int,
) -> list[int]:
    """Return one deterministic, disjoint task shard for a distributed eval rank."""

    tasks = [int(task_id) for task_id in task_ids]
    ranks = int(world_size)
    rank_id = int(rank)
    if ranks <= 0 or rank_id < 0 or rank_id >= ranks:
        raise ValueError(f"invalid distributed rank {rank_id}/{ranks}")
    return tasks[rank_id::ranks]


def allocate_divisible_worker_budget(
    episode_totals: Iterable[int],
    *,
    total_workers: int,
) -> list[int]:
    """Allocate workers without dropping an incomplete ordered-reset block.

    Every rank receives a positive divisor of its local episode count. Extra
    workers are assigned greedily to the rank with the highest remaining
    episodes-per-worker load, while respecting the configured global budget.
    """

    episodes = [int(total) for total in episode_totals]
    if not episodes or any(total <= 0 for total in episodes):
        raise ValueError("distributed eval episode totals must be positive")
    budget = int(total_workers)
    if budget < len(episodes):
        raise ValueError(
            f"global worker budget ({budget}) must be at least world_size ({len(episodes)})"
        )

    divisors = [
        [candidate for candidate in range(1, total + 1) if total % candidate == 0]
        for total in episodes
    ]
    allocations = [1] * len(episodes)
    remaining = budget - len(episodes)
    while remaining > 0:
        candidates: list[tuple[float, int, int]] = []
        for rank, (total, choices, current) in enumerate(
            zip(episodes, divisors, allocations, strict=True)
        ):
            next_workers = next((choice for choice in choices if choice > current), None)
            if next_workers is None or next_workers - current > remaining:
                continue
            candidates.append((float(total / current), -rank, next_workers))
        if not candidates:
            break
        _load, negative_rank, next_workers = max(candidates)
        rank = -negative_rank
        remaining -= next_workers - allocations[rank]
        allocations[rank] = next_workers
    return allocations


def merge_libero_eval_rank_payloads(
    payloads: Iterable[Mapping[str, Any]],
    *,
    episodes_per_task: int,
) -> dict[str, float]:
    """Merge raw rank episode records and additive rollout-work counters."""

    merged_records: dict[int, dict[int, bool]] = {}
    env_chunk_steps = 0
    env_action_steps = 0
    elapsed_seconds = 0.0
    for rank, payload in enumerate(payloads):
        local_records = dict(payload["records"])
        local_count = sum(len(dict(results)) for results in local_records.values())
        expected = int(payload.get("expected_episodes", local_count))
        if local_count != expected:
            raise ValueError(
                f"eval rank {rank} episode count mismatch: expected {expected}, got {local_count}"
            )
        for raw_task_id, raw_results in local_records.items():
            task_id = int(raw_task_id)
            task_results = merged_records.setdefault(task_id, {})
            for raw_reset_id, raw_success in dict(raw_results).items():
                reset_id = int(raw_reset_id)
                success = bool(raw_success)
                if reset_id in task_results:
                    raise ValueError(
                        f"duplicate eval result for task={task_id} reset_state={reset_id}"
                    )
                task_results[reset_id] = success
        env_chunk_steps += int(payload.get("env_chunk_steps", 0))
        env_action_steps += int(payload.get("env_action_steps", 0))
        elapsed_seconds = max(elapsed_seconds, float(payload.get("elapsed_seconds", 0.0)))

    metrics = summarize_libero_task_success(
        (
            {
                "task_id": task_id,
                "episodes": len(results),
                "successes": sum(results.values()),
            }
            for task_id, results in sorted(merged_records.items())
        ),
        episodes_per_task=episodes_per_task,
    )
    metrics["eval/env_chunk_steps"] = float(env_chunk_steps)
    metrics["eval/env_action_steps"] = float(env_action_steps)
    metrics["eval/elapsed_seconds"] = float(elapsed_seconds)
    metrics["eval/env_chunk_per_s"] = (
        float(env_chunk_steps) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    metrics["eval/env_action_step_per_s"] = (
        float(env_action_steps) / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    return metrics


def summarize_libero_task_success(
    task_records: Iterable[Mapping[str, int | float]],
    *,
    episodes_per_task: int,
) -> dict[str, float]:
    """Summarize LIBERO eval as a macro-average over task success rates."""
    records = [dict(record) for record in task_records]
    total_episodes = float(sum(int(record["episodes"]) for record in records))
    total_successes = float(sum(int(record["successes"]) for record in records))
    task_rates: list[tuple[int, float]] = []
    metrics: dict[str, float] = {}

    for record in records:
        task_id = int(record["task_id"])
        episodes = int(record["episodes"])
        successes = int(record["successes"])
        task_rate = float(successes / episodes) if episodes > 0 else 0.0
        task_rates.append((task_id, task_rate))
        metrics[f"eval_task_{task_id}_success_rate"] = task_rate
        metrics[f"eval_task_{task_id}_episodes"] = float(episodes)
        metrics[f"eval_task_{task_id}_successes"] = float(successes)

    macro_success_rate = (
        float(sum(rate for _, rate in task_rates) / len(task_rates)) if task_rates else 0.0
    )
    episode_weighted_rate = float(total_successes / total_episodes) if total_episodes > 0 else 0.0
    metrics.update(
        {
            "eval_success_rate": macro_success_rate,
            "eval_total_episodes": total_episodes,
            "eval_total_successes": total_successes,
            "eval_tasks": float(len(task_rates)),
            "eval_episodes_per_task": float(int(episodes_per_task)),
            "eval_episode_weighted_success_rate": episode_weighted_rate,
            "results/total_success_rate": macro_success_rate,
            "results/total_episodes": total_episodes,
            "results/total_successes": total_successes,
            "results/task_macro_success_rate": macro_success_rate,
            "results/episode_weighted_success_rate": episode_weighted_rate,
        }
    )
    return metrics
