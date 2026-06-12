"""Online replay buffer and distributed readiness helpers."""

from __future__ import annotations

import random
from collections import Counter, deque
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
        self.episodes_by_task: dict[int, deque[dict[str, Any]]] = {}
        self._transitions_by_task: Counter[int] = Counter()
        self._next_episode_id = 0
        self._next_collection_index = 0
        self._next_task_episode_index: Counter[int] = Counter()

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

    def _capacity_for_task(self, task_id: int) -> int:
        del task_id
        if self.capacity_mode == "per_task":
            return int(self.capacity)
        denom = max(1, len(self.task_ids or ()))
        return max(int(self.sequence_length), int(self.capacity) // denom)

    def add_episode(self, episode: list[dict[str, Any]]) -> dict[str, Any] | None:
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
            "length": len(episode),
            "finish_step": finish_step,
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
        return record

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
    ) -> bool:
        if self.num_transitions < int(min_transitions):
            return False
        min_eps = int(min_episodes_per_task)
        if min_eps <= 0:
            return bool(self._valid_records())
        counts = self.task_episode_counts()
        return all(counts[int(task_id)] >= min_eps for task_id in task_ids)

    def _choose_records(self, batch_size: int) -> list[dict[str, Any]]:
        valid = self._valid_records()
        if not valid:
            raise RuntimeError("online replay has no full sequences")
        if not self.task_balanced:
            return [random.choice(valid) for _ in range(int(batch_size))]

        by_task: dict[int, list[dict[str, Any]]] = {}
        for record in valid:
            by_task.setdefault(int(record["task_id"]), []).append(record)
        task_ids = sorted(by_task)
        if not task_ids:
            raise RuntimeError("online replay has no task-indexed full sequences")
        offset = random.randrange(len(task_ids))
        ordered_tasks = task_ids[offset:] + task_ids[:offset]
        return [
            random.choice(by_task[ordered_tasks[idx % len(ordered_tasks)]])
            for idx in range(int(batch_size))
        ]

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
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
        for record in self._choose_records(int(batch_size)):
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

        images = np.stack(
            [[step["image"] for step in window] for window in windows], axis=0
        )
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
        return {
            "images": torch.from_numpy(images).to(torch.float32),
            "obs_embedding": torch.from_numpy(obs_embedding).to(torch.float32),
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
        }

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
    global_coverage_ready = (
        total_transitions >= int(min_transitions) and global_task_ready
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
) -> tuple[dict[str, dict[str, int]], bool, bool]:
    packed = pack_replay_task_stats_for_ddp(
        replay,
        task_ids=task_ids,
        min_transitions=min_transitions,
        min_episodes_per_task=min_episodes_per_task,
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
    )

