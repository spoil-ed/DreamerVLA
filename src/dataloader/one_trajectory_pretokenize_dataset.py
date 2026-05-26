from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

from src.dataloader.pretokenize_dataset import PretokenizeActionChunkDataset


_TRAJECTORY_RE = re.compile(r"^trj_(\d+)$")


def _trajectory_sort_key(trajectory_key: str) -> tuple[int, int, str]:
    name = trajectory_key.rsplit("/", 1)[-1]
    match = _TRAJECTORY_RE.fullmatch(name)
    if match is None:
        return (1, 0, name)
    return (0, int(match.group(1)), name)


class OneTrajectoryPretokenizeActionChunkDataset(PretokenizeActionChunkDataset):
    """Pretokenized VLA SFT dataset restricted to a few trajectories per task.

    LIBERO pretokenized manifests are frame-level records.  The parent dataset
    first groups those records into contiguous action chunks; this subclass then
    keeps only chunks whose source trajectory belongs to the selected
    ``task/trj_*`` set.  With the default ``trajectories_per_task=1`` this gives
    the paper-style one-trajectory SFT split: one demonstration trajectory for
    every task in the suite.
    """

    def __init__(
        self,
        config_path: str | Path,
        action_horizon: int = 10,
        history: int | None = None,
        batch_length: int | None = None,
        replay_context: int | None = None,
        sequence_length: int | None = None,
        stride: int | None = None,
        sequence_next_obs_source: str | None = None,
        trajectories_per_task: int = 1,
        trajectory_offset: int = 0,
        strict: bool = True,
    ) -> None:
        if int(trajectories_per_task) < 1:
            raise ValueError("trajectories_per_task must be >= 1")
        if int(trajectory_offset) < 0:
            raise ValueError("trajectory_offset must be >= 0")

        self.trajectories_per_task = int(trajectories_per_task)
        self.trajectory_offset = int(trajectory_offset)
        self.strict = bool(strict)
        self.selected_trajectory_keys: tuple[str, ...] = ()

        super().__init__(
            config_path=config_path,
            action_horizon=action_horizon,
            history=history,
            batch_length=batch_length,
            replay_context=replay_context,
            sequence_length=sequence_length,
            stride=stride,
            sequence_next_obs_source=sequence_next_obs_source,
        )
        self._filter_chunk_windows_to_selected_trajectories()
        self._data_spec = replace(
            self._data_spec,
            num_samples=len(self._chunk_windows),
            one_trajectory_sft=True,
            trajectories_per_task=self.trajectories_per_task,
            trajectory_offset=self.trajectory_offset,
            selected_trajectory_keys=self.selected_trajectory_keys,
        )

    def _window_trajectory_info(self, record_indices: tuple[int, ...]) -> tuple[str, str] | None:
        if not record_indices:
            return None
        payload = self._load_payload_by_index(int(record_indices[0]))
        image_path = self._select_current_third_view(payload.get("image", []))
        parsed = self._parse_image_path(image_path)
        if parsed is None:
            return None
        task_name, trajectory_key, _frame_index = parsed
        return task_name, trajectory_key

    def _select_trajectory_keys(self) -> tuple[str, ...]:
        trajectories_by_task: OrderedDict[str, dict[str, None]] = OrderedDict()
        for record_indices in self._chunk_windows:
            info = self._window_trajectory_info(record_indices)
            if info is None:
                continue
            task_name, trajectory_key = info
            trajectories_by_task.setdefault(task_name, OrderedDict()).setdefault(trajectory_key, None)

        if self.strict and not trajectories_by_task:
            raise ValueError("One-trajectory SFT found no valid trajectory windows.")

        selected: list[str] = []
        for task_name, trajectory_map in trajectories_by_task.items():
            ordered = sorted(trajectory_map, key=_trajectory_sort_key)
            start = self.trajectory_offset
            stop = start + self.trajectories_per_task
            if self.strict and len(ordered) < stop:
                raise ValueError(
                    f"Task {task_name!r} has only {len(ordered)} trajectories; "
                    f"cannot select offset={start} count={self.trajectories_per_task}."
                )
            selected.extend(ordered[start:stop])
        return tuple(selected)

    def _filter_chunk_windows_to_selected_trajectories(self) -> None:
        selected_keys = self._select_trajectory_keys()
        selected_set = set(selected_keys)
        filtered: list[tuple[int, ...]] = []
        for record_indices in self._chunk_windows:
            info = self._window_trajectory_info(record_indices)
            if info is None:
                continue
            _task_name, trajectory_key = info
            if trajectory_key in selected_set:
                filtered.append(record_indices)

        if self.strict and selected_keys and not filtered:
            raise ValueError(
                "One-trajectory SFT selected trajectories but produced no action chunks; "
                "check action_horizon and contiguous frames."
            )

        self.selected_trajectory_keys = selected_keys
        self._chunk_windows = filtered

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_indices = self._chunk_windows[int(index)]
        info = self._window_trajectory_info(record_indices)
        item = super().__getitem__(index)
        if info is None:
            return item
        task_name, trajectory_key = info
        meta = dict(item.get("meta", {}))
        meta.update(
            {
                "one_trajectory_sft": True,
                "task_name": task_name,
                "trajectory_key": trajectory_key,
                "trajectories_per_task": self.trajectories_per_task,
                "trajectory_offset": self.trajectory_offset,
                "selected_trajectory_keys": self.selected_trajectory_keys,
            }
        )
        item["meta"] = meta
        return item


__all__ = ["OneTrajectoryPretokenizeActionChunkDataset"]
