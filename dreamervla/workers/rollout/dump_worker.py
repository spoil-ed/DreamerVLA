"""Ray actor that writes cold-start rollout HDF5 shards."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dreamervla.dataset.collection_manifest import (
    append_episode_index_record,
    record_online_rollout_episode,
)
from dreamervla.dataset.collection_manifest import (
    read_online_rollout_manifest as read_online_rollout_manifest,
)
from dreamervla.dataset.rollout_dump_writer import (
    RolloutDumpWriter,
    per_trajectory_shard_name,
)
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
        demos_per_shard: int = 0,
        start_shard_index: int = 0,
        manifest_root: str | None = None,
        keep_last_global_steps: int = 0,
    ) -> None:
        super().__init__()
        self.reward_dir = str(reward_dir)
        self.hidden_dir = str(hidden_dir)
        self.shard_name = str(shard_name)
        self.preprocess_config = dict(preprocess_config or {})
        self.data_attrs = dict(data_attrs or {})
        self.demos_per_shard = int(demos_per_shard)
        self.writer: RolloutDumpWriter | None = None
        self.num_episodes = 0
        self.manifest_root = (
            str(manifest_root)
            if manifest_root not in (None, "")
            else str(Path(self.reward_dir).expanduser().parent)
        )
        self.keep_last_global_steps = int(keep_last_global_steps)
        # Resume-aware: start rotation at the next free index so a relaunch appends new
        # shards instead of overwriting ``ray_shard_000``.
        self._shard_idx = int(start_shard_index)
        self._shard_demos = 0

    def _shard_name(self, idx: int) -> str:
        stem = self.shard_name[:-5] if self.shard_name.endswith(".hdf5") else self.shard_name
        base = re.sub(r"_\d+$", "", stem)
        return f"{base}_{idx:03d}.hdf5"

    def init(self) -> None:
        first = self.shard_name if self.demos_per_shard <= 0 else self._shard_name(self._shard_idx)
        self.writer = RolloutDumpWriter(Path(self.reward_dir), Path(self.hidden_dir), first)

    def add_episode(self, episode: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not episode:
            return None
        if self.writer is None:
            self.writer = RolloutDumpWriter(
                Path(self.reward_dir),
                Path(self.hidden_dir),
                self._shard_name(self._shard_idx),
            )
        elif self.demos_per_shard > 0 and self._shard_demos >= self.demos_per_shard:
            self._writer().close()
            self._shard_idx += 1
            self._shard_demos = 0
            self.writer = RolloutDumpWriter(
                Path(self.reward_dir), Path(self.hidden_dir), self._shard_name(self._shard_idx)
            )
        first = episode[0]
        index = self._shard_demos if self.demos_per_shard > 0 else int(self.num_episodes)
        self._writer().write_demo(
            index=index,
            steps=episode,
            preprocess_config=self.preprocess_config if index == 0 else None,
            data_attrs=self.data_attrs if index == 0 else None,
            task_id=_optional_int(first.get("task_id")),
            episode_id=_optional_int(first.get("episode_id")),
            init_state_index=_optional_int(first.get("init_state_index")),
            task_description=first.get("task_description"),
            episode_success=bool(episode[-1].get("success", False)),
            episode_horizon=len(episode),
            episode_metadata=episode[-1].get("episode_metadata"),
        )
        shard_name = str(self._writer().shard_name)
        self._shard_demos += 1
        self.num_episodes += 1
        completed_shard_name = shard_name
        if self.demos_per_shard > 0 and self._shard_demos >= self.demos_per_shard:
            self._writer().close()
            if self.demos_per_shard == 1:
                completed_shard_name = self._rename_completed_episode_shard(
                    shard_name,
                    episode,
                    self._shard_idx,
                )
            self.writer = None
            self._shard_idx += 1
            self._shard_demos = 0
        if (
            self.demos_per_shard == 1
            and self.keep_last_global_steps <= 0
            and (Path(self.reward_dir) / completed_shard_name).exists()
        ):
            append_episode_index_record(
                Path(self.reward_dir),
                {
                    "file": completed_shard_name,
                    "task_id": _optional_int(first.get("task_id")),
                    "episode_id": _optional_int(first.get("episode_id")),
                    "init_state_index": _optional_int(first.get("init_state_index")),
                    "success": bool(episode[-1].get("success", False)),
                    "horizon": len(episode),
                },
            )
        manifest_entry = None
        if self.keep_last_global_steps > 0:
            manifest_entry = self._record_manifest_entry(episode, completed_shard_name)
        return {
            "episode_index": int(self.num_episodes - 1),
            "length": len(episode),
            "shard_name": completed_shard_name,
            "manifest_entry": manifest_entry,
        }

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

    def _rename_completed_episode_shard(
        self,
        shard_name: str,
        episode: list[dict[str, Any]],
        shard_idx: int,
    ) -> str:
        target_name = self._completed_episode_shard_name(episode, shard_idx)
        if target_name == shard_name:
            return shard_name
        # Identity-only names (coldstart, no global_step) may legitimately
        # collide on re-collection: the fresh episode replaces the stale one.
        metadata = dict(episode[-1].get("episode_metadata") or {})
        allow_overwrite = _metadata_int(metadata, "global_step") is None
        pairs = [
            (Path(self.reward_dir) / shard_name, Path(self.reward_dir) / target_name),
            (Path(self.hidden_dir) / shard_name, Path(self.hidden_dir) / target_name),
        ]
        existing = [(src, dst) for src, dst in pairs if src.exists()]
        if not existing:
            return shard_name
        for _src, dst in existing:
            if dst.exists() and not allow_overwrite:
                raise FileExistsError(f"rollout dump shard already exists: {dst}")
        for src, dst in existing:
            src.replace(dst)
        return target_name

    def _completed_episode_shard_name(
        self,
        episode: list[dict[str, Any]],
        shard_idx: int,
    ) -> str:
        stem = self.shard_name[:-5] if self.shard_name.endswith(".hdf5") else self.shard_name
        base = re.sub(r"_\d+$", "", stem)
        metadata = dict(episode[-1].get("episode_metadata") or {})
        global_step = _metadata_int(metadata, "global_step")
        step_token = f"gs{int(global_step):06d}" if global_step is not None else "gsunknown"
        outcome = "success" if bool(episode[-1].get("success", False)) else "fail"
        first = episode[0]
        task_id = _optional_int(first.get("task_id"))
        episode_id = _optional_int(first.get("episode_id"))
        if task_id is None or episode_id is None:
            return f"{base}_{step_token}_{outcome}_{int(shard_idx):03d}.hdf5"
        if global_step is None:
            # Coldstart collect: identity-only naming via
            # per_trajectory_shard_name so numbering matches the globally
            # assigned work list.
            return per_trajectory_shard_name("traj", task_id, episode_id)
        return f"{base}_{step_token}_t{int(task_id):02d}_ep{int(episode_id):06d}_{outcome}.hdf5"

    def _record_manifest_entry(
        self,
        episode: list[dict[str, Any]],
        shard_name: str,
    ) -> dict[str, Any]:
        first = episode[0]
        metadata = dict(episode[-1].get("episode_metadata") or {})
        global_step = _required_metadata_int(metadata, "global_step")
        env_step = _required_metadata_int(metadata, "env_step")
        task_id = _required_int(first.get("task_id"), "task_id")
        episode_id = _required_int(first.get("episode_id"), "episode_id")
        init_state_index = _optional_int(first.get("init_state_index"))
        return record_online_rollout_episode(
            self.manifest_root,
            reward_path=Path(self.reward_dir) / shard_name,
            hidden_path=Path(self.hidden_dir) / shard_name,
            task_id=task_id,
            episode_id=episode_id,
            init_state_index=init_state_index,
            success=bool(episode[-1].get("success", False)),
            complete=True,
            global_step=global_step,
            env_step=env_step,
            keep_last_global_steps=self.keep_last_global_steps,
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _required_metadata_int(metadata: dict[str, Any], key: str) -> int:
    value = _metadata_int(metadata, key)
    if value is None:
        raise ValueError(f"online rollout manifest requires episode_metadata.{key}")
    return int(value)


def _required_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"online rollout manifest requires {name}")
    return int(value)
