from __future__ import annotations

import pytest

from dreamervla.preprocess.build_lerobot_task_subset import select_episode_records


def test_select_episode_records_balances_tasks_and_reindexes() -> None:
    tasks = [
        {"task_index": 0, "task": "task zero"},
        {"task_index": 1, "task": "task one"},
    ]
    episodes = [
        {"episode_index": 4, "tasks": ["task one"], "length": 14},
        {"episode_index": 2, "tasks": ["task zero"], "length": 12},
        {"episode_index": 1, "tasks": ["task one"], "length": 11},
        {"episode_index": 0, "tasks": ["task zero"], "length": 10},
        {"episode_index": 3, "tasks": ["task zero"], "length": 13},
    ]

    selected = select_episode_records(
        tasks=tasks,
        episodes=episodes,
        episodes_per_task=2,
    )

    assert [item.source_episode_index for item in selected] == [0, 2, 1, 4]
    assert [item.subset_episode_index for item in selected] == [0, 1, 2, 3]
    assert [item.task_index for item in selected] == [0, 0, 1, 1]


def test_select_episode_records_rejects_underfilled_task() -> None:
    with pytest.raises(ValueError, match="task 0 has only 1 episodes; 2 required"):
        select_episode_records(
            tasks=[{"task_index": 0, "task": "task zero"}],
            episodes=[{"episode_index": 0, "tasks": ["task zero"], "length": 10}],
            episodes_per_task=2,
        )
