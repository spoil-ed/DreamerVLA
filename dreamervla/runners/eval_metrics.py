"""Shared evaluation metric summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


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
        float(sum(rate for _, rate in task_rates) / len(task_rates))
        if task_rates
        else 0.0
    )
    episode_weighted_rate = (
        float(total_successes / total_episodes) if total_episodes > 0 else 0.0
    )
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
