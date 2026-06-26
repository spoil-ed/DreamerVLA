"""Online replay buffer and distributed readiness helpers."""

from __future__ import annotations

import random
from collections import Counter, deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


class OnlineReplay:
    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        *,
        task_ids: tuple[int, ...] | None = None,
        capacity_mode: str = "per_task",
        failure_prefix_steps: int = 40,
        failure_prefix_ratio: float = 0.2,
        task_balanced: bool = True,
        rank: int = 0,
        replay_sampling: Mapping[str, Any] | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.task_ids = (
            tuple(int(task_id) for task_id in task_ids)
            if task_ids is not None
            else None
        )
        self.capacity_mode = str(capacity_mode)
        if self.capacity_mode not in {"per_task", "total_sharded"}:
            raise ValueError(
                "capacity_mode must be one of {'per_task', 'total_sharded'}"
            )
        self.failure_prefix_steps = int(failure_prefix_steps)
        self.failure_prefix_ratio = float(failure_prefix_ratio)
        self.task_balanced = bool(task_balanced)
        self.rank = int(rank)
        sampling_cfg = dict(replay_sampling or {})
        mix_cfg = dict(sampling_cfg.get("mix", {}) or {})
        self.replay_sampling_enabled = bool(sampling_cfg.get("enabled", False))
        self.recent_episode_count = int(sampling_cfg.get("recent_episode_count", 8))
        self.replay_sampling_mix = {
            "online_recent": float(mix_cfg.get("online_recent", 0.5)),
            "online_replay": float(mix_cfg.get("online_replay", 0.3)),
            "coldstart_anchor": float(mix_cfg.get("coldstart_anchor", 0.2)),
        }
        self.latest_online_required = bool(
            sampling_cfg.get("latest_online_required", False)
        )
        self.episodes_by_task: dict[int, deque[dict[str, Any]]] = {}
        self._transitions_by_task: Counter[int] = Counter()
        # Off-policy staleness (Phase 4): episodes are stamped with the rollout
        # policy version current at add time; the learner can gate stale samples.
        self._current_policy_version = 0
        self._next_episode_id = 0
        self._next_collection_index = 0
        self._next_task_episode_index: Counter[int] = Counter()
        self._pending_latest_online_episode_ids: set[int] = set()

    @property
    def episodes(self) -> list[dict[str, Any]]:
        return [
            record
            for task_id in sorted(self.episodes_by_task)
            for record in self.episodes_by_task[task_id]
        ]

    @property
    def num_transitions(self) -> int:
        return int(sum(self._transitions_by_task.values()))

    def set_policy_version(self, version: int) -> None:
        """Set the rollout policy version stamped onto subsequently added episodes."""
        self._current_policy_version = int(version)

    def _capacity_for_task(self, task_id: int) -> int:
        del task_id
        if self.capacity_mode == "per_task":
            return int(self.capacity)
        denom = max(1, len(self.task_ids or ()))
        return max(int(self.sequence_length), int(self.capacity) // denom)

    def add_episode(
        self, episode: list[dict[str, Any]], *, source: str = "online"
    ) -> dict[str, Any] | None:
        if len(episode) < self.sequence_length:
            return None
        task_id = self._episode_task_id(episode)
        success = self._episode_success(episode)
        finish_step = self._episode_finish_step(episode)
        episode_id = int(self._next_episode_id)
        collection_index = int(self._next_collection_index)
        task_episode_index = int(self._next_task_episode_index[int(task_id)])
        self._next_episode_id += 1
        self._next_collection_index += 1
        self._next_task_episode_index[int(task_id)] += 1
        record = {
            "episode": episode,
            "episode_id": episode_id,
            "collection_index": collection_index,
            "task_episode_index": task_episode_index,
            "rank": self.rank,
            "task_id": task_id,
            "success": success,
            "source": str(source),
            "source_id": self._source_id(str(source)),
            "length": len(episode),
            "finish_step": finish_step,
            "policy_version": int(self._current_policy_version),
        }
        bucket = self.episodes_by_task.setdefault(int(task_id), deque())
        bucket.append(record)
        self._transitions_by_task[int(task_id)] += len(episode)
        capacity = self._capacity_for_task(int(task_id))
        while self._transitions_by_task[int(task_id)] > capacity and bucket:
            old = bucket.popleft()
            self._transitions_by_task[int(task_id)] -= len(old["episode"])
        if not bucket:
            self.episodes_by_task.pop(int(task_id), None)
            self._transitions_by_task.pop(int(task_id), None)
        if str(source) == "online":
            self._pending_latest_online_episode_ids.add(int(episode_id))
        return record

    @staticmethod
    def _source_id(source: str) -> int:
        if source == "coldstart":
            return 0
        if source == "online":
            return 1
        return 2

    @staticmethod
    def _episode_success(episode: list[dict[str, Any]]) -> bool:
        return any(
            bool(step.get("success", False))
            or float(step.get("is_terminal", 0.0)) > 0.5
            or float(step.get("reward", 0.0)) > 0.0
            for step in episode
        )

    @staticmethod
    def _episode_finish_step(episode: list[dict[str, Any]]) -> int:
        for idx, step in enumerate(episode):
            if (
                bool(step.get("success", False))
                or float(step.get("is_terminal", 0.0)) > 0.5
                or float(step.get("reward", 0.0)) > 0.0
            ):
                return int(idx) + 1
        return int(len(episode))

    @staticmethod
    def _episode_task_id(episode: list[dict[str, Any]]) -> int:
        for step in episode:
            if "task_id" in step:
                return int(step["task_id"])
        return -1

    def _sample_limit(self, record: dict[str, Any]) -> int:
        episode = record["episode"]
        length = len(episode)
        if bool(record["success"]):
            return length
        limits: list[int] = []
        if self.failure_prefix_steps > 0:
            limits.append(self.failure_prefix_steps)
        if self.failure_prefix_ratio > 0.0:
            limits.append(max(1, int(round(length * self.failure_prefix_ratio))))
        if not limits:
            return length
        return min(length, max(self.sequence_length, min(limits)))

    def _valid_records(self) -> list[dict[str, Any]]:
        return [
            record
            for records in self.episodes_by_task.values()
            for record in records
            if self._sample_limit(record) >= self.sequence_length
        ]

    def _sampleable_windows(self, record: dict[str, Any]) -> int:
        limit = self._sample_limit(record)
        return max(0, int(limit) - self.sequence_length + 1)

    def sampleable_window_count(self) -> int:
        return sum(self._sampleable_windows(record) for record in self._valid_records())

    def task_episode_counts(self) -> Counter[int]:
        return Counter(int(record["task_id"]) for record in self._valid_records())

    def task_stats(
        self, task_ids: tuple[int, ...] | None = None
    ) -> dict[str, dict[str, int]]:
        requested = (
            set(int(task_id) for task_id in task_ids) if task_ids is not None else None
        )
        stats: dict[int, dict[str, int]] = {}
        for record in self._valid_records():
            task_id = int(record["task_id"])
            if requested is not None and task_id not in requested:
                continue
            entry = stats.setdefault(
                task_id,
                {
                    "episodes": 0,
                    "successes": 0,
                    "failures": 0,
                    "transitions": 0,
                    "sampleable_windows": 0,
                },
            )
            length = len(record["episode"])
            success = bool(record["success"])
            entry["episodes"] += 1
            entry["successes"] += int(success)
            entry["failures"] += int(not success)
            entry["transitions"] += int(length)
            entry["sampleable_windows"] += int(self._sampleable_windows(record))

        if requested is not None:
            for task_id in requested:
                stats.setdefault(
                    int(task_id),
                    {
                        "episodes": 0,
                        "successes": 0,
                        "failures": 0,
                        "transitions": 0,
                        "sampleable_windows": 0,
                    },
                )
        return {str(task_id): value for task_id, value in sorted(stats.items())}

    def ready_for_training(
        self,
        *,
        min_transitions: int,
        task_ids: tuple[int, ...],
        min_episodes_per_task: int,
        min_sampleable_windows: int = 0,
        require_classifier_evidence: bool = False,
    ) -> bool:
        if self.num_transitions < int(min_transitions):
            return False
        if int(min_sampleable_windows) > 0:
            if self.sampleable_window_count() < int(min_sampleable_windows):
                return False
        min_eps = int(min_episodes_per_task)
        if min_eps <= 0:
            ready = bool(self._valid_records())
        else:
            counts = self.task_episode_counts()
            ready = all(counts[int(task_id)] >= min_eps for task_id in task_ids)
        if not ready:
            return False
        if require_classifier_evidence and not self.classifier_ready(task_ids=task_ids):
            return False
        return True

    def classifier_ready(self, *, task_ids: tuple[int, ...] | None = None) -> bool:
        stats = self.task_stats(task_ids)
        totals = {
            "successes": sum(int(item["successes"]) for item in stats.values()),
            "failures": sum(int(item["failures"]) for item in stats.values()),
        }
        return totals["successes"] > 0 and totals["failures"] > 0

    def _fresh_valid_records(
        self, *, staleness_threshold: int | None = None
    ) -> list[dict[str, Any]]:
        valid = self._valid_records()
        if staleness_threshold is not None:
            fresh = [
                record
                for record in valid
                if not _is_stale(
                    int(record.get("policy_version", 0)),
                    self._current_policy_version,
                    int(staleness_threshold),
                )
            ]
            # Fall back to all valid records if gating empties the pool, so the
            # learner is never starved (e.g. OFT, where the fixed-base rollout
            # version stays 0 while the learner version climbs → all "stale").
            if fresh:
                valid = fresh
        return valid

    def _choose_one_task_balanced(
        self, records: list[dict[str, Any]], sample_idx: int = 0
    ) -> dict[str, Any]:
        if not records:
            raise RuntimeError("online replay has no full sequences")
        if not self.task_balanced:
            return random.choice(records)
        by_task: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            by_task.setdefault(int(record["task_id"]), []).append(record)
        task_ids = sorted(by_task)
        if not task_ids:
            raise RuntimeError("online replay has no task-indexed full sequences")
        return random.choice(by_task[task_ids[int(sample_idx) % len(task_ids)]])

    def _records_by_sampling_pool(
        self, valid: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        online = [record for record in valid if record.get("source", "online") == "online"]
        coldstart = [
            record for record in valid if record.get("source", "online") == "coldstart"
        ]
        recent_n = max(0, int(self.recent_episode_count))
        recent = sorted(
            online,
            key=lambda record: int(record.get("collection_index", 0)),
            reverse=True,
        )[:recent_n] if recent_n > 0 else []
        return {
            "online_recent": recent,
            "online_replay": online,
            "coldstart_anchor": coldstart,
        }

    def _choose_pool_name(self, pools: dict[str, list[dict[str, Any]]]) -> str:
        available = [
            (name, max(0.0, float(self.replay_sampling_mix.get(name, 0.0))))
            for name, records in pools.items()
            if records
        ]
        if not available:
            raise RuntimeError("online replay has no full sequences")
        total = sum(weight for _name, weight in available)
        if total <= 0.0:
            return random.choice([name for name, _weight in available])
        pick = random.random() * total
        acc = 0.0
        for name, weight in available:
            acc += weight
            if pick <= acc:
                return name
        return available[-1][0]

    def _pop_pending_latest_online(
        self, valid: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not self.latest_online_required or not self._pending_latest_online_episode_ids:
            return None
        by_id = {int(record["episode_id"]): record for record in valid}
        for episode_id in sorted(self._pending_latest_online_episode_ids, reverse=True):
            record = by_id.get(int(episode_id))
            if record is not None:
                self._pending_latest_online_episode_ids.discard(int(episode_id))
                return record
            self._pending_latest_online_episode_ids.discard(int(episode_id))
        return None

    def _choose_records(
        self, batch_size: int, *, staleness_threshold: int | None = None
    ) -> list[dict[str, Any]]:
        valid = self._fresh_valid_records(staleness_threshold=staleness_threshold)
        if not valid:
            raise RuntimeError("online replay has no full sequences")
        chosen: list[dict[str, Any]] = []
        pending = self._pop_pending_latest_online(valid)
        if pending is not None:
            chosen.append(pending)
        if self.replay_sampling_enabled:
            pools = self._records_by_sampling_pool(valid)
            while len(chosen) < int(batch_size):
                pool_name = self._choose_pool_name(pools)
                chosen.append(
                    self._choose_one_task_balanced(pools[pool_name], len(chosen))
                )
            return chosen
        while len(chosen) < int(batch_size):
            chosen.append(self._choose_one_task_balanced(valid, len(chosen)))
        return chosen

    def sample(
        self,
        batch_size: int,
        *,
        staleness_threshold: int | None = None,
        include_images: bool = True,
    ) -> dict[str, torch.Tensor]:
        windows = []
        task_ids: list[int] = []
        successes: list[bool] = []
        start_indices: list[int] = []
        episode_ids: list[int] = []
        collection_indices: list[int] = []
        task_episode_indices: list[int] = []
        episode_lengths: list[int] = []
        sample_limits: list[int] = []
        source_ranks: list[int] = []
        replay_source_ids: list[int] = []
        for record in self._choose_records(
            int(batch_size), staleness_threshold=staleness_threshold
        ):
            episode = record["episode"]
            limit = self._sample_limit(record)
            start = random.randint(0, limit - self.sequence_length)
            windows.append(episode[start : start + self.sequence_length])
            task_ids.append(int(record["task_id"]))
            successes.append(bool(record["success"]))
            start_indices.append(int(start))
            episode_ids.append(int(record["episode_id"]))
            collection_indices.append(int(record["collection_index"]))
            task_episode_indices.append(int(record["task_episode_index"]))
            episode_lengths.append(int(record["length"]))
            sample_limits.append(int(limit))
            source_ranks.append(int(record["rank"]))
            replay_source_ids.append(int(record.get("source_id", self._source_id("online"))))

        obs_embedding = np.stack(
            [[step["obs_embedding"] for step in window] for window in windows], axis=0
        )
        rewards = np.stack(
            [[step["reward"] for step in window] for window in windows], axis=0
        )
        dones = np.stack(
            [[step["done"] for step in window] for window in windows], axis=0
        )
        is_terminal = np.stack(
            [[step["is_terminal"] for step in window] for window in windows], axis=0
        )
        is_last = np.stack(
            [[step["is_last"] for step in window] for window in windows], axis=0
        )
        action_dim = int(
            np.asarray(windows[0][0]["wm_action"], dtype=np.float32)
            .reshape(-1)
            .shape[0]
        )
        actions = np.zeros(
            (len(windows), self.sequence_length, action_dim), dtype=np.float32
        )
        current_actions = np.zeros_like(actions)
        for batch_idx, window in enumerate(windows):
            for time_idx in range(self.sequence_length):
                current_actions[batch_idx, time_idx] = np.asarray(
                    window[time_idx]["wm_action"], dtype=np.float32
                )
            for time_idx in range(1, self.sequence_length):
                actions[batch_idx, time_idx] = np.asarray(
                    window[time_idx - 1]["wm_action"], dtype=np.float32
                )
        is_first = np.zeros((len(windows), self.sequence_length), dtype=np.bool_)
        is_first[:, 0] = True
        batch = {
            "obs_embedding": torch.from_numpy(obs_embedding),
            "actions": torch.from_numpy(actions),
            "current_actions": torch.from_numpy(current_actions),
            "rewards": torch.from_numpy(rewards.astype(np.float32, copy=False)),
            "dones": torch.from_numpy(dones.astype(np.float32, copy=False)),
            "is_terminal": torch.from_numpy(is_terminal.astype(np.float32, copy=False)),
            "is_last": torch.from_numpy(is_last.astype(np.float32, copy=False)),
            "is_first": torch.from_numpy(is_first),
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
            "episode_success": torch.tensor(successes, dtype=torch.bool),
            "start_indices": torch.tensor(start_indices, dtype=torch.long),
            "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
            "collection_indices": torch.tensor(collection_indices, dtype=torch.long),
            "task_episode_indices": torch.tensor(
                task_episode_indices, dtype=torch.long
            ),
            "episode_lengths": torch.tensor(episode_lengths, dtype=torch.long),
            "sample_limits": torch.tensor(sample_limits, dtype=torch.long),
            "source_ranks": torch.tensor(source_ranks, dtype=torch.long),
            "replay_source_ids": torch.tensor(replay_source_ids, dtype=torch.long),
        }
        if bool(include_images):
            images = np.stack(
                [[step["image"] for step in window] for window in windows], axis=0
            )
            batch["images"] = torch.from_numpy(images).to(torch.float32)
        return batch

    def sample_classifier_windows(
        self,
        batch_size: int,
        *,
        window: int,
        chunk_size: int,
        chunk_pool: str,
        early_neg_stride: int,
    ) -> dict[str, torch.Tensor]:
        window = int(window)
        chunk_size = int(chunk_size)
        window_env = window * chunk_size
        stride = max(1, int(early_neg_stride))
        candidates = [
            record
            for record in self._valid_records()
            if int(record.get("finish_step", len(record["episode"]))) >= window_env
        ]
        if not candidates:
            raise RuntimeError(
                f"online replay has no classifier windows with finish_step >= {window_env}"
            )

        windows: list[np.ndarray] = []
        labels: list[int] = []
        episode_ids: list[int] = []
        collection_indices: list[int] = []
        task_episode_indices: list[int] = []
        task_ids: list[int] = []
        source_ranks: list[int] = []
        replay_source_ids: list[int] = []
        source_successes: list[bool] = []
        episode_lengths: list[int] = []
        finish_steps: list[int] = []
        window_end_indices: list[int] = []
        is_end_window: list[bool] = []

        for _ in range(int(batch_size)):
            record = random.choice(candidates)
            episode = record["episode"]
            finish_step = int(record.get("finish_step", len(episode)))
            use_end = random.random() < 0.5
            if use_end:
                end = finish_step
                label = int(bool(record["success"]))
            else:
                valid_ends = list(range(finish_step - stride, window_env - 1, -stride))
                if valid_ends:
                    end = int(random.choice(valid_ends))
                    label = 0
                else:
                    end = finish_step
                    label = int(bool(record["success"]))
                    use_end = True

            env_window = np.stack(
                [step["obs_embedding"] for step in episode[end - window_env : end]],
                axis=0,
            ).astype(np.float32, copy=False)
            if chunk_size > 1:
                trailing_shape = env_window.shape[1:]
                reshaped = env_window.reshape(window, chunk_size, *trailing_shape)
                if chunk_pool == "last":
                    pooled = reshaped[:, -1]
                elif chunk_pool == "first":
                    pooled = reshaped[:, 0]
                elif chunk_pool == "mean":
                    pooled = reshaped.mean(axis=1)
                else:
                    raise ValueError(f"unknown chunk_pool={chunk_pool!r}")
            else:
                pooled = env_window

            windows.append(np.ascontiguousarray(pooled, dtype=np.float32))
            labels.append(int(label))
            episode_ids.append(int(record["episode_id"]))
            collection_indices.append(int(record["collection_index"]))
            task_episode_indices.append(int(record["task_episode_index"]))
            task_ids.append(int(record["task_id"]))
            source_ranks.append(int(record["rank"]))
            replay_source_ids.append(int(record.get("source_id", self._source_id("online"))))
            source_successes.append(bool(record["success"]))
            episode_lengths.append(int(record["length"]))
            finish_steps.append(int(finish_step))
            window_end_indices.append(int(end))
            is_end_window.append(bool(use_end))

        return {
            "windows": torch.from_numpy(np.stack(windows, axis=0)).to(torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
            "collection_indices": torch.tensor(collection_indices, dtype=torch.long),
            "task_episode_indices": torch.tensor(
                task_episode_indices, dtype=torch.long
            ),
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
            "source_ranks": torch.tensor(source_ranks, dtype=torch.long),
            "replay_source_ids": torch.tensor(replay_source_ids, dtype=torch.long),
            "source_success": torch.tensor(source_successes, dtype=torch.bool),
            "episode_lengths": torch.tensor(episode_lengths, dtype=torch.long),
            "finish_steps": torch.tensor(finish_steps, dtype=torch.long),
            "window_end_indices": torch.tensor(window_end_indices, dtype=torch.long),
            "is_end_window": torch.tensor(is_end_window, dtype=torch.bool),
        }

    def classifier_window_count(self, *, window: int, chunk_size: int) -> int:
        window_env = int(window) * int(chunk_size)
        return sum(
            1
            for record in self._valid_records()
            if int(record.get("finish_step", len(record["episode"]))) >= window_env
        )


_REPLAY_DDP_COLUMNS = (
    "episodes",
    "successes",
    "failures",
    "transitions",
    "sampleable_windows",
)


def pack_replay_task_stats_for_ddp(
    replay: OnlineReplay,
    *,
    task_ids: tuple[int, ...],
    min_transitions: int,
    min_episodes_per_task: int,
    min_sampleable_windows: int = 0,
    require_classifier_evidence: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Pack local replay stats into a tensor that can be all-reduced by SUM."""
    stats = replay.task_stats(task_ids)
    rows: list[list[float]] = []
    for task_id in task_ids:
        item = stats[str(int(task_id))]
        rows.append([float(item[key]) for key in _REPLAY_DDP_COLUMNS])
    local_ready = float(
        replay.ready_for_training(
            min_transitions=int(min_transitions),
            task_ids=task_ids,
            min_episodes_per_task=int(min_episodes_per_task),
            min_sampleable_windows=int(min_sampleable_windows),
            require_classifier_evidence=bool(require_classifier_evidence),
        )
    )
    rows.append([float(replay.num_transitions), local_ready, 0.0, 0.0, 0.0])
    return torch.tensor(rows, dtype=torch.float64, device=device)


def unpack_replay_task_stats_from_ddp(
    packed: torch.Tensor,
    *,
    task_ids: tuple[int, ...],
    world_size: int,
    min_transitions: int = 0,
    min_episodes_per_task: int = 1,
    min_sampleable_windows: int = 0,
) -> tuple[dict[str, dict[str, int]], bool, bool]:
    """Unpack SUM-reduced replay stats and compute global per-task readiness."""
    cpu = packed.detach().cpu()
    stats: dict[str, dict[str, int]] = {}
    for row_idx, task_id in enumerate(task_ids):
        stats[str(int(task_id))] = {
            key: int(round(float(cpu[row_idx, col_idx].item())))
            for col_idx, key in enumerate(_REPLAY_DDP_COLUMNS)
        }
    total_transitions = int(round(float(cpu[len(task_ids), 0].item())))
    local_ready_sum = int(round(float(cpu[len(task_ids), 1].item())))
    all_ranks_ready = local_ready_sum >= int(world_size)
    global_task_ready = all(
        stats[str(int(task_id))]["episodes"] >= int(min_episodes_per_task)
        for task_id in task_ids
    )
    total_sampleable_windows = sum(
        stats[str(int(task_id))]["sampleable_windows"] for task_id in task_ids
    )
    global_coverage_ready = (
        total_transitions >= int(min_transitions)
        and total_sampleable_windows >= int(min_sampleable_windows)
        and global_task_ready
    )
    return stats, bool(global_coverage_ready), bool(all_ranks_ready)


def get_replay_task_stats_global(
    replay: OnlineReplay,
    *,
    task_ids: tuple[int, ...],
    min_transitions: int,
    min_episodes_per_task: int,
    device: torch.device,
    is_dist: bool,
    world_size: int,
    min_sampleable_windows: int = 0,
    require_classifier_evidence: bool = False,
) -> tuple[dict[str, dict[str, int]], bool, bool]:
    packed = pack_replay_task_stats_for_ddp(
        replay,
        task_ids=task_ids,
        min_transitions=min_transitions,
        min_episodes_per_task=min_episodes_per_task,
        min_sampleable_windows=min_sampleable_windows,
        require_classifier_evidence=require_classifier_evidence,
        device=device if is_dist else None,
    )
    if is_dist:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return unpack_replay_task_stats_from_ddp(
        packed,
        task_ids=task_ids,
        world_size=world_size,
        min_transitions=min_transitions,
        min_episodes_per_task=min_episodes_per_task,
        min_sampleable_windows=min_sampleable_windows,
    )


def _is_stale(record_version: int, current_version: int, threshold: int) -> bool:
    if int(threshold) < 0:
        return False
    return max(0, int(current_version) - int(record_version)) > int(threshold)
