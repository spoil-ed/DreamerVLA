"""Process orchestration and metadata commits for parallel LIBERO replay."""

from __future__ import annotations

import json
import multiprocessing
import tempfile
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ReplayTaskResult:
    """Complete, parent-consumable result from one task-exclusive replay worker."""

    task_id: int
    task_description: str
    output_path: str
    metadata: JsonDict
    num_replays: int
    num_successes: int
    num_noops: int


@dataclass
class ReplayTotals:
    """Aggregate replay counts without worker-side output."""

    num_replays: int = 0
    num_successes: int = 0
    num_noops: int = 0

    def add(self, result: ReplayTaskResult) -> None:
        self.num_replays += int(result.num_replays)
        self.num_successes += int(result.num_successes)
        self.num_noops += int(result.num_noops)

    def summary(self) -> str:
        rate = 0.0 if self.num_replays == 0 else self.num_successes / self.num_replays * 100.0
        return (
            f"Total # episodes replayed: {self.num_replays}, "
            f"Total # successes: {self.num_successes} ({rate:.1f} %), "
            f"Total # no-op actions filtered out: {self.num_noops}"
        )


def iter_task_results(
    requests: Sequence[RequestT],
    *,
    num_workers: int,
    worker: Callable[[RequestT], ResultT],
) -> Iterator[ResultT]:
    """Yield task results sequentially or from a bounded spawn-based process pool."""

    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if num_workers == 1 or len(requests) <= 1:
        for request in requests:
            yield worker(request)
        return

    worker_count = min(int(num_workers), len(requests))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [executor.submit(worker, request) for request in requests]
        for future in as_completed(futures):
            yield future.result()


def _merge_metadata(destination: JsonDict, source: JsonDict) -> None:
    for task_key, task_value in source.items():
        if not isinstance(task_value, dict):
            raise ValueError(f"task metadata must be an object: {task_key}")
        current = destination.setdefault(str(task_key), {})
        if not isinstance(current, dict):
            raise ValueError(f"existing task metadata must be an object: {task_key}")
        current.update(task_value)


def _load_json_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metadata JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"metadata JSON must be an object: {path}")
    return value


def atomic_write_json(path: str | Path, value: JsonDict) -> None:
    """Atomically replace one JSON document on the destination filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def task_metadata_shard(shard_dir: str | Path, task_id: int) -> Path:
    return Path(shard_dir) / f"task_{int(task_id):03d}.json"


def write_task_metadata_shard(
    shard_dir: str | Path,
    *,
    task_id: int,
    metadata: JsonDict,
) -> Path:
    """Checkpoint one worker's metadata without sharing a writable file."""

    path = task_metadata_shard(shard_dir, task_id)
    atomic_write_json(path, metadata)
    return path


def load_resume_metadata(canonical_path: str | Path, shard_dir: str | Path) -> JsonDict:
    """Load canonical metadata and recover any uncommitted worker shards."""

    canonical = Path(canonical_path)
    metadata: JsonDict = _load_json_object(canonical) if canonical.is_file() else {}
    shards = Path(shard_dir)
    if shards.is_dir():
        for shard in sorted(shards.glob("task_*.json")):
            _merge_metadata(metadata, _load_json_object(shard))
    return metadata


def commit_task_result(
    canonical_path: str | Path,
    shard_dir: str | Path,
    result: ReplayTaskResult,
    accumulated: JsonDict,
) -> None:
    """Merge one completed task, commit canonical JSON, then retire its shard."""

    _merge_metadata(accumulated, result.metadata)
    atomic_write_json(canonical_path, accumulated)
    task_metadata_shard(shard_dir, result.task_id).unlink(missing_ok=True)


__all__ = [
    "ReplayTaskResult",
    "ReplayTotals",
    "atomic_write_json",
    "commit_task_result",
    "iter_task_results",
    "load_resume_metadata",
    "task_metadata_shard",
    "write_task_metadata_shard",
]
