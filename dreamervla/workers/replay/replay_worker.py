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

    def add_episode(
        self, episode: list[dict[str, Any]], source: str = "online"
    ) -> dict[str, Any] | None:
        return self._replay().add_episode(episode, source=str(source))

    def set_policy_version(self, version: int) -> None:
        self._replay().set_policy_version(int(version))

    def sample(
        self,
        batch_size: int,
        staleness_threshold: int | None = None,
        include_images: bool = True,
    ) -> dict[str, Any]:
        return self._replay().sample(
            int(batch_size),
            staleness_threshold=staleness_threshold,
            include_images=bool(include_images),
        )

    def sample_initial_obs_embeddings(
        self,
        batch_size: int,
        *,
        task_id: int | None = None,
        key: str = "obs_embedding",
    ) -> Any:
        return self._replay().sample_initial_obs_embeddings(
            int(batch_size),
            task_id=task_id,
            key=str(key),
        )

    def sample_classifier_windows(
        self,
        batch_size: int,
        *,
        window: int,
        chunk_size: int,
        chunk_pool: str,
        early_neg_stride: int,
        sampling_protocol: str = "lumos",
        balance_batches: bool = False,
    ) -> dict[str, Any]:
        return self._replay().sample_classifier_windows(
            int(batch_size),
            window=int(window),
            chunk_size=int(chunk_size),
            chunk_pool=str(chunk_pool),
            early_neg_stride=int(early_neg_stride),
            sampling_protocol=str(sampling_protocol),
            balance_batches=bool(balance_batches),
        )

    def classifier_window_count(self, *, window: int, chunk_size: int) -> int:
        return int(
            self._replay().classifier_window_count(
                window=int(window),
                chunk_size=int(chunk_size),
            )
        )

    def size(self) -> int:
        return len(self._replay().episodes)

    def num_transitions(self) -> int:
        return int(self._replay().num_transitions)

    def ready(
        self,
        min_episodes_per_task: int,
        *,
        min_transitions: int = 0,
        task_ids: tuple[int, ...] | None = None,
        min_sampleable_windows: int = 0,
        require_classifier_evidence: bool = False,
    ) -> bool:
        replay = self._replay()
        selected_task_ids = (
            tuple(int(task_id) for task_id in task_ids)
            if task_ids is not None
            else tuple(int(task_id) for task_id in (replay.task_ids or (0,)))
        )
        return replay.ready_for_training(
            min_transitions=int(min_transitions),
            task_ids=selected_task_ids,
            min_episodes_per_task=int(min_episodes_per_task),
            min_sampleable_windows=int(min_sampleable_windows),
            require_classifier_evidence=bool(require_classifier_evidence),
        )

    def task_stats(self, task_ids: tuple[int, ...] | None = None) -> dict[str, dict[str, int]]:
        return self._replay().task_stats(task_ids)

    def state_dict(self) -> dict[str, Any]:
        return self._replay().state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._replay().load_state_dict(state)

    def _replay(self) -> Any:
        if self.replay is None:
            raise RuntimeError("ReplayWorker.init() has not been called")
        return self.replay
