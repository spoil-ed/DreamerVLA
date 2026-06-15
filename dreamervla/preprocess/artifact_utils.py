"""Shared artifact validation and work planning for preprocessing jobs."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py

from dreamervla.preprocess.sidecar_schema import required_demo_datasets_from_config


@dataclass(frozen=True)
class Hdf5SourceStat:
    """Low-cost size summary used for preprocessing task scheduling."""

    file: str
    demos: int
    frames: int


@dataclass(frozen=True)
class Hdf5PreprocessTask:
    """One source HDF5 file and its expected preprocessing cost."""

    source_path: Path
    demos: int
    frames: int


@dataclass(frozen=True)
class Hdf5TaskPlan:
    """Rank-local view of a deterministic global preprocessing plan."""

    assigned: list[Hdf5PreprocessTask]
    pending: list[Hdf5PreprocessTask]
    skipped: list[Hdf5PreprocessTask]
    repaired: list[Hdf5PreprocessTask]
    loads_by_rank: list[int]


def source_hdf5_stat(path: Path) -> Hdf5SourceStat:
    """Return demo and frame counts for a LIBERO-style source HDF5 file."""

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise RuntimeError(f"missing data group: {path}")
        demos = 0
        frames = 0
        for demo in data_group.values():
            demos += 1
            if "actions" in demo:
                frames += int(demo["actions"].shape[0])
                continue
            obs_group = demo.get("obs")
            if obs_group is not None and obs_group.keys():
                first_key = next(iter(obs_group.keys()))
                frames += int(obs_group[first_key].shape[0])
                continue
            raise RuntimeError(f"cannot infer frame count for demo in {path}")
    return Hdf5SourceStat(file=path.name, demos=demos, frames=frames)


def _demo_lengths(handle: h5py.File, path: Path) -> dict[str, int]:
    data_group = handle.get("data")
    if data_group is None or not data_group.keys():
        raise RuntimeError(f"missing non-empty data group: {path}")
    lengths: dict[str, int] = {}
    for demo_key, demo in data_group.items():
        if "actions" in demo:
            lengths[str(demo_key)] = int(demo["actions"].shape[0])
            continue
        obs_group = demo.get("obs")
        if obs_group is not None and obs_group.keys():
            first_key = next(iter(obs_group.keys()))
            lengths[str(demo_key)] = int(obs_group[first_key].shape[0])
            continue
        first_dataset = next(
            (value for value in demo.values() if isinstance(value, h5py.Dataset)),
            None,
        )
        if first_dataset is not None and len(first_dataset.shape) > 0:
            lengths[str(demo_key)] = int(first_dataset.shape[0])
            continue
        raise RuntimeError(f"cannot infer demo length for {path}:{demo_key}")
    return lengths


def is_complete_hdf5_artifact(
    path: Path,
    *,
    reference_path: Path | None = None,
    required_demo_datasets: Sequence[str] = (),
    require_complete_attr: bool = True,
) -> bool:
    """Return whether one generated HDF5 artifact is structurally complete.

    When ``reference_path`` is supplied, the output must contain the same demo
    keys as the source and every required per-demo dataset must have the same
    leading dimension as the source demo.
    """

    if not path.is_file():
        return False
    try:
        reference_lengths: dict[str, int] | None = None
        if reference_path is not None:
            with h5py.File(reference_path, "r") as reference:
                reference_lengths = _demo_lengths(reference, reference_path)
        with h5py.File(path, "r") as handle:
            if require_complete_attr and not bool(handle.attrs.get("complete", False)):
                return False
            data_group = handle.get("data")
            if data_group is None or not data_group.keys():
                return False
            if reference_lengths is not None and set(data_group.keys()) != set(reference_lengths):
                return False
            for demo_key, demo in data_group.items():
                for dataset in required_demo_datasets:
                    if dataset not in demo:
                        return False
                    if reference_lengths is not None:
                        value = demo[dataset]
                        if not isinstance(value, h5py.Dataset) or not value.shape:
                            return False
                        if int(value.shape[0]) != int(reference_lengths[str(demo_key)]):
                            return False
            return True
    except (OSError, RuntimeError):
        return False


def _required_demo_datasets_for_output(
    path: Path,
    explicit: Sequence[str] | None,
) -> Sequence[str]:
    if explicit is not None:
        return explicit
    config_path = path.parent / "preprocess_config.json"
    if not config_path.is_file():
        return ()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return ("__invalid_preprocess_config__",)
        return required_demo_datasets_from_config(config)
    except (OSError, ValueError):
        return ("__invalid_preprocess_config__",)


def assign_tasks_by_frames(
    tasks: Sequence[Hdf5PreprocessTask],
    *,
    world_size: int,
) -> list[list[Hdf5PreprocessTask]]:
    """Greedily balance tasks by estimated frame count across ranks."""

    world_size = max(1, int(world_size))
    buckets: list[list[Hdf5PreprocessTask]] = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    ordered = sorted(tasks, key=lambda task: (-task.frames, -task.demos, task.source_path.name))
    for task in ordered:
        rank = min(range(world_size), key=lambda idx: (loads[idx], len(buckets[idx]), idx))
        buckets[rank].append(task)
        loads[rank] += int(task.frames)
    for bucket in buckets:
        bucket.sort(key=lambda task: task.source_path.name)
    return buckets


def plan_hdf5_preprocess_tasks(
    source_files: Iterable[Path],
    *,
    rank: int,
    world_size: int,
    output_paths: Callable[[Path], Sequence[Path]],
    required_demo_datasets: Mapping[Path, Sequence[str]] | None = None,
    overwrite: bool = False,
    repair: bool = True,
) -> Hdf5TaskPlan:
    """Validate existing outputs and assign only pending source files to ranks.

    Complete outputs are excluded from the work queue. Existing incomplete
    outputs are repaired by default at file granularity: the bad outputs for
    that source file are removed, then the source file is put back into the
    pending queue. Set ``repair=False`` to fail fast instead.
    """

    required_demo_datasets = required_demo_datasets or {}
    pending: list[Hdf5PreprocessTask] = []
    skipped: list[Hdf5PreprocessTask] = []
    repaired: list[Hdf5PreprocessTask] = []

    for source_path in sorted((Path(path) for path in source_files), key=lambda path: path.name):
        stat = source_hdf5_stat(source_path)
        task = Hdf5PreprocessTask(
            source_path=source_path,
            demos=stat.demos,
            frames=stat.frames,
        )
        paths = [Path(path) for path in output_paths(source_path)]
        existing = [path for path in paths if path.exists()]
        if not paths:
            pending.append(task)
            continue
        if overwrite:
            pending.append(task)
            continue
        if not existing:
            pending.append(task)
            continue
        complete = all(
            is_complete_hdf5_artifact(
                path,
                reference_path=source_path,
                required_demo_datasets=_required_demo_datasets_for_output(
                    path,
                    required_demo_datasets.get(path),
                ),
                require_complete_attr=True,
            )
            for path in paths
        )
        if complete:
            skipped.append(task)
            continue
        if repair:
            for path in existing:
                path.unlink(missing_ok=True)
            repaired.append(task)
            pending.append(task)
            continue
        raise RuntimeError(
            "Refusing to use incomplete preprocessing artifact without --overwrite: "
            + ", ".join(str(path) for path in paths)
        )

    assignments = assign_tasks_by_frames(pending, world_size=world_size)
    loads_by_rank = [sum(int(task.frames) for task in bucket) for bucket in assignments]
    rank_index = int(rank)
    if rank_index < 0 or rank_index >= len(assignments):
        raise RuntimeError(f"rank {rank} outside world_size {world_size}")
    return Hdf5TaskPlan(
        assigned=list(assignments[rank_index]),
        pending=pending,
        skipped=skipped,
        repaired=repaired,
        loads_by_rank=loads_by_rank,
    )
