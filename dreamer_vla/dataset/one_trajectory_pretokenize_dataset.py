from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

from dreamer_vla.dataset.pretokenize_dataset import PretokenizeActionChunkDataset


_TRAJECTORY_RE = re.compile(r"^trj_(\d+)$")


def _trajectory_sort_key(trajectory_key: str) -> tuple[int, int, str]:
    name = trajectory_key.rsplit("/", 1)[-1]
    match = _TRAJECTORY_RE.fullmatch(name)
    if match is None:
        return (1, 0, name)
    return (0, int(match.group(1)), name)


class OneTrajectoryPretokenizeActionChunkDataset(PretokenizeActionChunkDataset):
    """Pretokenized VLA SFT dataset restricted to the first trajectory.

    LIBERO pretokenized manifests are frame-level records.  The parent dataset
    groups records into contiguous action chunks.  This subclass narrows that
    grouping to selected ``task/trj_*`` keys before opening payload files, so
    one-trajectory SFT does not scan every demonstration pkl.
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
        selection_scope: str = "per_task",
        strict: bool = True,
    ) -> None:
        if int(trajectories_per_task) < 1:
            raise ValueError("trajectories_per_task must be >= 1")
        if int(trajectory_offset) < 0:
            raise ValueError("trajectory_offset must be >= 0")
        if selection_scope not in {"global", "per_task"}:
            raise ValueError("selection_scope must be 'global' or 'per_task'")

        self.trajectories_per_task = int(trajectories_per_task)
        self.trajectory_offset = int(trajectory_offset)
        self.selection_scope = str(selection_scope)
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
        self._data_spec = replace(
            self._data_spec,
            num_samples=len(self._chunk_windows),
            one_trajectory_sft=True,
            trajectories_per_task=self.trajectories_per_task,
            trajectory_offset=self.trajectory_offset,
            selected_trajectory_keys=self.selected_trajectory_keys,
        )

    def _record_frame_info_from_manifest(
        self, record_index: int
    ) -> tuple[str, str, int] | None:
        if record_index < 0 or record_index >= len(self.records):
            return None
        record = self.records[record_index]
        if not isinstance(record, dict):
            return None

        candidates: list[Any] = []
        meta = record.get("meta")
        if isinstance(meta, dict):
            next_obs = meta.get("next_obs")
            if isinstance(next_obs, dict):
                candidates.append(next_obs.get("image"))
        next_obs = record.get("next_obs")
        if isinstance(next_obs, dict):
            candidates.append(next_obs.get("image"))

        for images in candidates:
            image_path = self._select_current_third_view(images)
            parsed = self._parse_image_path(image_path)
            if parsed is None:
                continue
            return parsed

        try:
            payload = self._load_payload_by_index(record_index)
        except Exception:
            return None
        image_path = self._select_current_third_view(payload.get("image", []))
        return self._parse_image_path(image_path)

    def _record_trajectory_info_from_manifest(
        self, record_index: int
    ) -> tuple[str, str] | None:
        parsed = self._record_frame_info_from_manifest(record_index)
        if parsed is None:
            return None
        task_name, trajectory_key, _frame_index = parsed
        return task_name, trajectory_key

    def _build_chunk_windows(self) -> None:
        record_infos: list[tuple[int, str, str, int]] = []
        trajectories_by_task: OrderedDict[str, dict[str, None]] = OrderedDict()
        for record_index, _record in enumerate(self.records):
            parsed = self._record_frame_info_from_manifest(record_index)
            if parsed is None:
                continue
            task_name, trajectory_key, frame_index = parsed
            trajectories_by_task.setdefault(task_name, OrderedDict()).setdefault(
                trajectory_key, None
            )
            record_infos.append((record_index, task_name, trajectory_key, frame_index))

        start = self.trajectory_offset
        stop = start + self.trajectories_per_task
        if self.selection_scope == "global":
            trajectory_map: OrderedDict[str, None] = OrderedDict()
            for _record_index, _task_name, trajectory_key, _frame_index in record_infos:
                trajectory_map.setdefault(trajectory_key, None)
            ordered = list(trajectory_map)
            if self.strict and len(ordered) < stop:
                raise ValueError(
                    f"Manifest has only {len(ordered)} trajectories; "
                    f"cannot select offset={start} count={self.trajectories_per_task}."
                )
            selected = ordered[start:stop]
        else:
            selected = []
            for task_name, trajectory_map in trajectories_by_task.items():
                ordered = sorted(trajectory_map, key=_trajectory_sort_key)
                if self.strict and len(ordered) < stop:
                    raise ValueError(
                        f"Task {task_name!r} has only {len(ordered)} trajectories; "
                        f"cannot select offset={start} count={self.trajectories_per_task}."
                    )
                selected.extend(ordered[start:stop])

        selected_set = set(selected)
        frames_by_key: dict[str, dict[int, int]] = {}
        for record_index, _task_name, trajectory_key, frame_index in record_infos:
            if trajectory_key not in selected_set:
                continue
            frame_map = frames_by_key.setdefault(trajectory_key, {})
            frame_map.setdefault(frame_index, record_index)

        for frame_map in frames_by_key.values():
            frame_set = set(frame_map)
            for start in sorted(frame_map):
                wanted = range(start, start + self.action_horizon)
                if all(frame in frame_set for frame in wanted):
                    self._chunk_windows.append(
                        tuple(frame_map[frame] for frame in wanted)
                    )

        if self.strict and trajectories_by_task and not self._chunk_windows:
            raise ValueError(
                "One-trajectory SFT selected trajectories but produced no action chunks; "
                "check action_horizon and contiguous frames."
            )
        self.selected_trajectory_keys = tuple(selected)

    def _window_trajectory_info(
        self, record_indices: tuple[int, ...]
    ) -> tuple[str, str] | None:
        if not record_indices:
            return None
        record_index = int(record_indices[0])
        manifest_info = self._record_trajectory_info_from_manifest(record_index)
        if manifest_info is not None:
            return manifest_info

        payload = self._load_payload_by_index(record_index)
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
            trajectories_by_task.setdefault(task_name, OrderedDict()).setdefault(
                trajectory_key, None
            )

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
