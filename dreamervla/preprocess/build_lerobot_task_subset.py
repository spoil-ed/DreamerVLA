"""Build a deterministic, task-balanced subset of a LeRobot v2 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SelectedEpisode:
    """Mapping from one source episode to its contiguous subset identity."""

    task_index: int
    task: str
    source_episode_index: int
    subset_episode_index: int
    length: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_episode_records(
    *,
    tasks: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
    episodes_per_task: int,
) -> tuple[SelectedEpisode, ...]:
    """Select the lowest source episode indices for every task."""

    count = int(episodes_per_task)
    if count < 1:
        raise ValueError("episodes_per_task must be at least 1")

    ordered_tasks = sorted(tasks, key=lambda item: int(item["task_index"]))
    expected_indices = list(range(len(ordered_tasks)))
    actual_indices = [int(item["task_index"]) for item in ordered_tasks]
    if actual_indices != expected_indices:
        raise ValueError(
            f"task_index values must be unique and contiguous from zero; got {actual_indices}"
        )

    task_index_by_text = {str(item["task"]): int(item["task_index"]) for item in ordered_tasks}
    if len(task_index_by_text) != len(ordered_tasks):
        raise ValueError("task strings must be unique")

    buckets: dict[int, list[dict[str, Any]]] = {index: [] for index in expected_indices}
    for episode in sorted(episodes, key=lambda item: int(item["episode_index"])):
        episode_tasks = episode.get("tasks")
        if not isinstance(episode_tasks, list) or len(episode_tasks) != 1:
            raise ValueError(
                "each LIBERO episode must contain exactly one task string; "
                f"episode {episode.get('episode_index')} has {episode_tasks!r}"
            )
        task = str(episode_tasks[0])
        if task not in task_index_by_text:
            raise ValueError(
                f"episode {episode.get('episode_index')} references unknown task {task!r}"
            )
        task_index = task_index_by_text[task]
        if len(buckets[task_index]) < count:
            buckets[task_index].append(episode)

    selections: list[SelectedEpisode] = []
    for task_item in ordered_tasks:
        task_index = int(task_item["task_index"])
        task = str(task_item["task"])
        selected = buckets[task_index]
        if len(selected) != count:
            raise ValueError(
                f"task {task_index} has only {len(selected)} episodes; {count} required"
            )
        for episode in selected:
            selections.append(
                SelectedEpisode(
                    task_index=task_index,
                    task=task,
                    source_episode_index=int(episode["episode_index"]),
                    subset_episode_index=len(selections),
                    length=int(episode["length"]),
                )
            )
    return tuple(selections)


def build_lerobot_task_subset(
    *,
    source: str | Path,
    output: str | Path,
    episodes_per_task: int,
) -> dict[str, Any]:
    """Write a new task-balanced LeRobot v2 dataset without changing the source."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("source and output must be different directories")
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    meta_dir = source_path / "meta"
    required = [
        meta_dir / "info.json",
        meta_dir / "episodes.jsonl",
        meta_dir / "tasks.jsonl",
        meta_dir / "stats.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source dataset is incomplete; missing {missing}")

    with (meta_dir / "info.json").open(encoding="utf-8") as stream:
        source_info = json.load(stream)
    if source_info.get("codebase_version") != "v2.0":
        raise ValueError(
            f"only LeRobot v2.0 datasets are supported; got {source_info.get('codebase_version')!r}"
        )

    tasks = _read_jsonl(meta_dir / "tasks.jsonl")
    episodes = _read_jsonl(meta_dir / "episodes.jsonl")
    selections = select_episode_records(
        tasks=tasks,
        episodes=episodes,
        episodes_per_task=episodes_per_task,
    )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("building a LeRobot subset requires pyarrow") from exc

    building_path = output_path.parent / f".{output_path.name}.building-{os.getpid()}"
    if building_path.exists():
        raise FileExistsError(f"temporary output already exists: {building_path}")
    building_path.mkdir(parents=True)

    chunks_size = int(source_info["chunks_size"])
    data_template = str(source_info["data_path"])
    total_frames = sum(item.length for item in selections)
    try:
        output_meta = building_path / "meta"
        output_meta.mkdir()
        shutil.copy2(meta_dir / "tasks.jsonl", output_meta / "tasks.jsonl")
        # Keep the full-dataset normalization contract for controlled comparisons.
        shutil.copy2(meta_dir / "stats.json", output_meta / "stats.json")

        info = dict(source_info)
        info.update(
            total_episodes=len(selections),
            total_frames=total_frames,
            total_tasks=len(tasks),
            total_chunks=(len(selections) + chunks_size - 1) // chunks_size,
            splits={"train": f"0:{len(selections)}"},
        )
        with (output_meta / "info.json").open("w", encoding="utf-8") as stream:
            json.dump(info, stream, indent=2)
            stream.write("\n")

        global_frame_index = 0
        episode_records: list[dict[str, Any]] = []
        for item in selections:
            source_chunk = item.source_episode_index // chunks_size
            source_episode = source_path / data_template.format(
                episode_chunk=source_chunk,
                episode_index=item.source_episode_index,
            )
            if not source_episode.is_file():
                raise FileNotFoundError(f"missing source episode: {source_episode}")

            table = pq.read_table(source_episode)
            if table.num_rows != item.length:
                raise ValueError(
                    f"episode {item.source_episode_index} metadata length {item.length} "
                    f"does not match parquet rows {table.num_rows}"
                )
            for column_name in ("episode_index", "index", "task_index", "frame_index"):
                if column_name not in table.column_names:
                    raise ValueError(f"{source_episode} is missing column {column_name!r}")

            episode_field = table.schema.field("episode_index")
            index_field = table.schema.field("index")
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                episode_field,
                pa.array([item.subset_episode_index] * item.length, type=episode_field.type),
            )
            table = table.set_column(
                table.schema.get_field_index("index"),
                index_field,
                pa.array(
                    range(global_frame_index, global_frame_index + item.length),
                    type=index_field.type,
                ),
            )

            subset_chunk = item.subset_episode_index // chunks_size
            destination = building_path / data_template.format(
                episode_chunk=subset_chunk,
                episode_index=item.subset_episode_index,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination, compression="zstd")
            episode_records.append(
                {
                    "episode_index": item.subset_episode_index,
                    "tasks": [item.task],
                    "length": item.length,
                }
            )
            global_frame_index += item.length

        with (output_meta / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for record in episode_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        manifest = {
            "source_dataset": str(source_path),
            "selection_rule": (
                f"lowest {int(episodes_per_task)} source episode_index values for each task_index"
            ),
            "normalization_stats": "copied from the full dataset for controlled comparison",
            "episodes_per_task": int(episodes_per_task),
            "total_tasks": len(tasks),
            "total_episodes": len(selections),
            "total_frames": total_frames,
            "episodes": [asdict(item) for item in selections],
        }
        with (building_path / "subset_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(building_path, output_path)
    except BaseException:
        shutil.rmtree(building_path, ignore_errors=True)
        raise
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = _parse_args(argv)
    manifest = build_lerobot_task_subset(
        source=args.source,
        output=args.output,
        episodes_per_task=args.episodes_per_task,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "total_tasks": manifest["total_tasks"],
                "total_episodes": manifest["total_episodes"],
                "total_frames": manifest["total_frames"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
