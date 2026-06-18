"""Ray actor that writes cold-start rollout HDF5 shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dreamervla.dataset.rollout_dump_writer import RolloutDumpWriter
from dreamervla.scheduler.worker import Worker


class RolloutDumpWorker(Worker):
    """Collect completed episodes and write reward HDF5 plus hidden sidecars."""

    def __init__(
        self,
        reward_dir: str,
        hidden_dir: str,
        shard_name: str = "ray_shard_000.hdf5",
        preprocess_config: dict[str, Any] | None = None,
        data_attrs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.reward_dir = str(reward_dir)
        self.hidden_dir = str(hidden_dir)
        self.shard_name = str(shard_name)
        self.preprocess_config = dict(preprocess_config or {})
        self.data_attrs = dict(data_attrs or {})
        self.writer: RolloutDumpWriter | None = None
        self.num_episodes = 0

    def init(self) -> None:
        self.writer = RolloutDumpWriter(
            Path(self.reward_dir),
            Path(self.hidden_dir),
            self.shard_name,
        )

    def add_episode(self, episode: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not episode:
            return None
        first = episode[0]
        index = int(self.num_episodes)
        self._writer().write_demo(
            index=index,
            steps=episode,
            preprocess_config=self.preprocess_config if index == 0 else None,
            data_attrs=self.data_attrs if index == 0 else None,
            task_id=_optional_int(first.get("task_id")),
            episode_id=_optional_int(first.get("episode_id")),
            task_description=first.get("task_description"),
            episode_success=bool(episode[-1].get("success", False)),
            episode_horizon=len(episode),
        )
        self.num_episodes += 1
        return {"episode_index": index, "length": len(episode)}

    def size(self) -> int:
        return int(self.num_episodes)

    def close(self) -> None:
        writer = self.writer
        if writer is not None:
            writer.close()
        self.writer = None

    def _writer(self) -> RolloutDumpWriter:
        if self.writer is None:
            raise RuntimeError("RolloutDumpWorker.init() has not been called")
        return self.writer


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
