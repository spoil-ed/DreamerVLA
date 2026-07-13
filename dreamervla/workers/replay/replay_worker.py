"""Ray actor wrapper around OnlineReplay."""

from __future__ import annotations

from typing import Any

from dreamervla.runtime.offline_seed import seed_replay_from_offline
from dreamervla.runtime.online_replay import OnlineReplay
from dreamervla.scheduler.worker import Worker
from dreamervla.workers.cotrain.messages import RealTrajectoryBatch


class ReplayWorker(Worker):
    """Expose OnlineReplay through a Ray actor."""

    def __init__(self, replay_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.replay_cfg = dict(replay_cfg)
        self.replay: Any | None = None

    def init(self) -> None:
        self.replay = OnlineReplay(**self.replay_cfg)

    def add_episode(
        self, episode: list[dict[str, Any]], source: str = "online"
    ) -> dict[str, Any] | None:
        return self._replay().add_episode(episode, source=str(source))

    def set_policy_version(self, version: int) -> None:
        self._replay().set_policy_version(int(version))

    def replace_real_trajectories(
        self,
        batch: RealTrajectoryBatch,
    ) -> dict[str, float]:
        """Replace replay with one re-encoded global-step real batch."""

        expected = int(batch.global_step)
        episodes: list[list[dict[str, Any]]] = []
        for trajectory in batch.trajectories:
            if int(trajectory.global_step) != expected:
                raise ValueError(
                    "step-local trajectory global_step does not match batch"
                )
            episode = [dict(transition) for transition in trajectory.transitions]
            versions = {
                int(transition.get("encoder_version", -1))
                for transition in episode
            }
            if versions != {expected}:
                raise ValueError(
                    "step-local replay requires every transition encoder_version "
                    f"to equal {expected}; got {sorted(versions)}"
                )
            episodes.append(episode)
        added = self._replay().replace_episodes(
            episodes,
            policy_version=expected,
            source="online",
        )
        return {
            "replay_buffer/step_local_trajectories": float(added),
            "replay_buffer/step_local_transitions": float(
                self._replay().num_transitions
            ),
            "replay_buffer/step_local_encoder_version": float(expected),
        }

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

    def classifier_endpoint_windows(
        self,
        *,
        window: int,
        chunk_size: int,
        chunk_pool: str,
    ) -> dict[str, Any]:
        """Return deterministic current-step endpoint windows for calibration."""

        return self._replay().classifier_endpoint_windows(
            window=int(window),
            chunk_size=int(chunk_size),
            chunk_pool=str(chunk_pool),
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

    def sample_initial_conditions(
        self,
        batch_size: int,
        *,
        task_ids: tuple[int, ...] | None = None,
        keys: tuple[str, ...] = ("obs_embedding", "lang_emb", "proprio"),
    ) -> Any:
        return self._replay().sample_initial_conditions(
            int(batch_size),
            task_ids=(
                tuple(int(task_id) for task_id in task_ids)
                if task_ids is not None
                else None
            ),
            keys=tuple(str(key) for key in keys),
        )

    def sampling_state_dict(self) -> dict[str, int]:
        """Return cursor-only replay state suitable for small checkpoints."""

        return self._replay().sampling_state_dict()

    def load_sampling_state_dict(self, state: dict[str, Any]) -> None:
        """Restore cursor-only replay state after deterministic offline seeding."""

        self._replay().load_sampling_state_dict(state)

    def seed_from_offline(self, seed_cfg: dict[str, Any]) -> dict[str, float]:
        """Seed official/collected HDF5 through the shared replay loader."""

        cfg = dict(seed_cfg)
        allowed = {
            "data_dir",
            "hidden_dir",
            "task_id",
            "default_task_id",
            "infer_task_id_from_shard",
            "task_ids",
            "max_episodes_per_task",
            "require_reference_complete",
        }
        unsupported = sorted(set(cfg) - allowed)
        if unsupported:
            raise ValueError(
                "unsupported replay seed options: " + ", ".join(unsupported)
            )
        task_ids = tuple(int(task_id) for task_id in cfg.pop("task_ids", ()) or ())
        default_task_id = cfg.pop("task_id", cfg.pop("default_task_id", None))
        added = seed_replay_from_offline(
            self._replay(),
            data_dir=cfg.pop("data_dir"),
            hidden_dir=cfg.pop("hidden_dir"),
            default_task_id=(
                None if default_task_id is None else int(default_task_id)
            ),
            infer_task_id_from_shard=bool(
                cfg.pop("infer_task_id_from_shard", False)
            ),
            max_episodes_per_task=cfg.pop("max_episodes_per_task", None),
            require_reference_complete=bool(
                cfg.pop("require_reference_complete", True)
            ),
        )
        stats = self._replay().task_stats(task_ids or None)
        if task_ids:
            missing = [
                task_id
                for task_id in task_ids
                if int(stats.get(str(task_id), {}).get("episodes", 0)) <= 0
            ]
            if missing:
                raise RuntimeError(
                    "offline replay seed did not retain episodes for task IDs "
                    f"{missing}"
                )
        return {
            "replay_buffer/seeded_episodes": float(added),
            "replay_buffer/seeded_transitions": float(self._replay().num_transitions),
            "replay_buffer/seeded_task_count": float(
                sum(
                    int(item.get("episodes", 0)) > 0
                    for item in stats.values()
                )
            ),
        }

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
