"""Ray actor wrapper around OnlineReplay."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from dreamervla.scheduler.worker import Worker


def _online_replay_cls() -> type:
    path = Path(__file__).resolve().parents[2] / "runners" / "online_replay.py"
    spec = importlib.util.spec_from_file_location("dreamervla_online_replay_for_ray", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load OnlineReplay from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OnlineReplay


class ReplayWorker(Worker):
    """Expose OnlineReplay through a Ray actor."""

    def __init__(self, replay_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.replay_cfg = dict(replay_cfg)
        self.replay: Any | None = None

    def init(self) -> None:
        self.replay = _online_replay_cls()(**self.replay_cfg)

    def add_episode(self, episode: list[dict[str, Any]]) -> dict[str, Any] | None:
        return self._replay().add_episode(episode)

    def sample(self, batch_size: int) -> dict[str, Any]:
        return self._replay().sample(int(batch_size))

    def size(self) -> int:
        return len(self._replay().episodes)

    def num_transitions(self) -> int:
        return int(self._replay().num_transitions)

    def ready(self, min_episodes: int) -> bool:
        return len(self._replay()._valid_records()) >= int(min_episodes)

    def task_stats(self, task_ids: tuple[int, ...] | None = None) -> dict[str, dict[str, int]]:
        return self._replay().task_stats(task_ids)

    def _replay(self) -> Any:
        if self.replay is None:
            raise RuntimeError("ReplayWorker.init() has not been called")
        return self.replay
