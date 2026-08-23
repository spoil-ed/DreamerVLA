"""Online replay buffer and distributed readiness helpers."""

from __future__ import annotations

import copy
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
            tuple(int(task_id) for task_id in task_ids) if task_ids is not None else None
        )
        self.capacity_mode = str(capacity_mode)
        if self.capacity_mode not in {"per_task", "total_sharded"}:
            raise ValueError("capacity_mode must be one of {'per_task', 'total_sharded'}")
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
        self.latest_online_required = bool(sampling_cfg.get("latest_online_required", False))
        self.episodes_by_task: dict[int, deque[dict[str, Any]]] = {}
        self._transitions_by_task: Counter[int] = Counter()
        # Off-policy staleness (Phase 4): episodes are stamped with the rollout
        # policy version current at add time; the learner can gate stale samples.
        self._current_policy_version = 0
        self._next_episode_id = 0
        self._next_collection_index = 0
        self._next_task_episode_index: Counter[int] = Counter()
        self._task_sample_cursor = 0
        self._initial_condition_cursor = 0
        self._pending_latest_online_episode_ids: set[int] = set()
        self._classifier_pending_windows: deque[tuple[Mapping[str, Any], int, int, bool]] = deque()
        self._classifier_pending_key: tuple[Any, ...] | None = None

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

    @property
    def task_sample_cursor(self) -> int:
        """Return the cross-batch cursor used by task-balanced sampling."""

        return int(self._task_sample_cursor)

    def _capacity_for_task(self, task_id: int) -> int:
        del task_id
        if self.capacity_mode == "per_task":
            return int(self.capacity)
        denom = max(1, len(self.task_ids or ()))
        return max(int(self.sequence_length), int(self.capacity) // denom)

    def add_episode(
        self,
        episode: list[dict[str, Any]],
        *,
        source: str = "online",
        success: bool | None = None,
    ) -> dict[str, Any] | None:
        if len(episode) < self.sequence_length:
            return None
        task_id = self._episode_task_id(episode)
        episode_success = self._episode_success(episode) if success is None else bool(success)
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
            "success": episode_success,
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

    def replace_episodes(
        self,
        episodes: list[list[dict[str, Any]]],
        *,
        policy_version: int,
        source: str = "online",
    ) -> int:
        """Atomically replace replay contents with one current-step dataset."""

        copied = [copy.deepcopy(list(episode)) for episode in episodes]
        self.episodes_by_task = {}
        self._transitions_by_task = Counter()
        self._next_episode_id = 0
        self._next_collection_index = 0
        self._next_task_episode_index = Counter()
        self._task_sample_cursor = 0
        self._initial_condition_cursor = 0
        self._pending_latest_online_episode_ids.clear()
        self._classifier_pending_windows.clear()
        self._classifier_pending_key = None
        self.set_policy_version(int(policy_version))
        added = 0
        for episode in copied:
            if self.add_episode(episode, source=str(source)) is not None:
                added += 1
        return int(added)

    def sample_initial_obs_embeddings(
        self,
        batch_size: int,
        *,
        task_id: int | None = None,
        key: str = "obs_embedding",
    ) -> np.ndarray:
        """Sample first-step hidden observations for WMEnv bootstrap."""

        records = self._valid_records()
        if task_id is not None:
            records = [record for record in records if int(record["task_id"]) == int(task_id)]
        if not records:
            raise RuntimeError("online replay has no records for WMEnv bootstrap")
        latents = []
        for index in range(int(batch_size)):
            record = records[index % len(records)]
            episode = record["episode"]
            if not episode:
                raise RuntimeError("online replay record has an empty episode")
            first = episode[0]
            if key not in first:
                raise KeyError(f"replay bootstrap step missing {key!r}")
            latents.append(np.asarray(first[key], dtype=np.float32))
        return np.stack(latents, axis=0)

    def sample_initial_conditions(
        self,
        batch_size: int,
        *,
        task_ids: tuple[int, ...] | None = None,
        keys: tuple[str, ...] = ("obs_embedding", "lang_emb", "proprio"),
        selector: str = "episode_start",
    ) -> dict[str, np.ndarray]:
        """Return aligned episode-start conditions with balanced task coverage.

        ``failed_episode_start`` filters to explicitly failed episode records and
        samples them with replacement. It never falls back to successful records.
        """

        count = int(batch_size)
        if count <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        requested_keys = tuple(str(key) for key in keys)
        if not requested_keys:
            raise ValueError("initial-condition keys must not be empty")

        normalized_selector = self._validate_initial_condition_selector(selector)
        records_by_task: dict[int, list[dict[str, Any]]] = {}
        for record in self._eligible_initial_condition_records(normalized_selector):
            records_by_task.setdefault(int(record["task_id"]), []).append(record)
        requested_tasks = tuple(
            int(task_id)
            for task_id in (
                task_ids
                if task_ids is not None
                else (self.task_ids or tuple(sorted(records_by_task)))
            )
        )
        selected_tasks = tuple(
            task_id for task_id in requested_tasks if records_by_task.get(task_id)
        )
        if not selected_tasks:
            if normalized_selector == "failed_episode_start":
                raise RuntimeError("online replay has no failed episodes for WMEnv bootstrap")
            raise RuntimeError("online replay has no records for WMEnv bootstrap")
        missing_tasks = [task_id for task_id in requested_tasks if not records_by_task.get(task_id)]
        if missing_tasks:
            if normalized_selector == "episode_start":
                raise RuntimeError(
                    f"online replay has no WMEnv bootstrap records for task IDs {missing_tasks}"
                )

        cursor = int(self._initial_condition_cursor)
        task_count = len(selected_tasks)
        selected_records: list[dict[str, Any]] = []
        sampled_task_ids: list[int] = []
        for offset in range(count):
            absolute_index = cursor + offset
            task_id = selected_tasks[absolute_index % task_count]
            task_records = records_by_task[task_id]
            task_cycle = absolute_index // task_count
            selected_records.append(task_records[task_cycle % len(task_records)])
            sampled_task_ids.append(int(task_id))
        self._initial_condition_cursor = cursor + count

        output: dict[str, np.ndarray] = {
            "task_ids": np.asarray(sampled_task_ids, dtype=np.int64),
            "anchor_step": np.zeros((count,), dtype=np.int64),
            "is_failure_anchor": np.full(
                (count,),
                normalized_selector == "failed_episode_start",
                dtype=np.bool_,
            ),
        }
        for key in requested_keys:
            values: list[np.ndarray] = []
            for record in selected_records:
                episode = record["episode"]
                if not episode:
                    raise RuntimeError("online replay record has an empty episode")
                first = episode[0]
                if key not in first:
                    raise KeyError(f"replay bootstrap step missing {key!r}")
                values.append(np.asarray(first[key], dtype=np.float32))
            output[key] = np.stack(values, axis=0)
        return output

    def eligible_initial_condition_count(
        self,
        selector: str = "episode_start",
        *,
        task_ids: tuple[int, ...] | None = None,
    ) -> int:
        """Return the number of replay episodes eligible for ``selector``."""

        normalized_selector = self._validate_initial_condition_selector(selector)
        requested = None if task_ids is None else {int(value) for value in task_ids}
        return sum(
            1
            for record in self._eligible_initial_condition_records(normalized_selector)
            if requested is None or int(record["task_id"]) in requested
        )

    @staticmethod
    def _validate_initial_condition_selector(selector: str) -> str:
        normalized = str(selector).strip().lower()
        allowed = {"episode_start", "failed_episode_start"}
        if normalized not in allowed:
            raise ValueError(
                f"initial-condition selector must be one of {sorted(allowed)}, got {selector!r}"
            )
        return normalized

    def _eligible_initial_condition_records(
        self,
        selector: str,
    ) -> list[dict[str, Any]]:
        records = self._valid_records()
        if selector == "failed_episode_start":
            return [record for record in records if not bool(record["success"])]
        return records

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
            or float(step.get("sparse_rewards", 0.0)) > 0.5
            or float(step.get("rewards", 0.0)) > 0.0
            for step in episode
        )

    @staticmethod
    def _episode_finish_step(episode: list[dict[str, Any]]) -> int:
        for idx, step in enumerate(episode):
            if (
                bool(step.get("success", False))
                or float(step.get("is_terminal", 0.0)) > 0.5
                or float(step.get("reward", 0.0)) > 0.0
                or float(step.get("sparse_rewards", 0.0)) > 0.5
                or float(step.get("rewards", 0.0)) > 0.0
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

    def task_stats(self, task_ids: tuple[int, ...] | None = None) -> dict[str, dict[str, int]]:
        requested = set(int(task_id) for task_id in task_ids) if task_ids is not None else None
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
        coldstart = [record for record in valid if record.get("source", "online") == "coldstart"]
        recent_n = max(0, int(self.recent_episode_count))
        recent = (
            sorted(
                online,
                key=lambda record: int(record.get("collection_index", 0)),
                reverse=True,
            )[:recent_n]
            if recent_n > 0
            else []
        )
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

    def _pop_pending_latest_online(self, valid: list[dict[str, Any]]) -> dict[str, Any] | None:
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
        balanced_count = 0

        def choose_balanced(records: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal balanced_count
            record = self._choose_one_task_balanced(
                records,
                self._task_sample_cursor + balanced_count,
            )
            if self.task_balanced:
                balanced_count += 1
            return record

        if self.replay_sampling_enabled:
            pools = self._records_by_sampling_pool(valid)
            while len(chosen) < int(batch_size):
                pool_name = self._choose_pool_name(pools)
                chosen.append(choose_balanced(pools[pool_name]))
            self._task_sample_cursor += balanced_count
            return chosen
        while len(chosen) < int(batch_size):
            chosen.append(choose_balanced(valid))
        self._task_sample_cursor += balanced_count
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
        rewards = np.stack([[_step_reward(step) for step in window] for window in windows], axis=0)
        dones = np.stack([[_step_done(step) for step in window] for window in windows], axis=0)
        is_terminal = np.stack(
            [[_step_is_terminal(step) for step in window] for window in windows], axis=0
        )
        is_last = np.stack([[_step_is_last(step) for step in window] for window in windows], axis=0)
        action_dim = int(_step_action(windows[0][0]).reshape(-1).shape[0])
        actions = np.zeros((len(windows), self.sequence_length, action_dim), dtype=np.float32)
        current_actions = np.zeros_like(actions)
        for batch_idx, window in enumerate(windows):
            for time_idx in range(self.sequence_length):
                current_actions[batch_idx, time_idx] = _step_action(window[time_idx])
            for time_idx in range(1, self.sequence_length):
                actions[batch_idx, time_idx] = _step_action(window[time_idx - 1])
        is_first = np.zeros((len(windows), self.sequence_length), dtype=np.bool_)
        is_first[:, 0] = True
        proprio = None
        if all("proprio" in step for window in windows for step in window):
            proprio = np.stack(
                [
                    [np.asarray(step["proprio"], dtype=np.float32).reshape(-1) for step in window]
                    for window in windows
                ],
                axis=0,
            )
        lang_emb = None
        if all("lang_emb" in step for window in windows for step in window):
            lang_emb = np.stack(
                [
                    np.asarray(window[0]["lang_emb"], dtype=np.float32).reshape(-1)
                    for window in windows
                ],
                axis=0,
            )
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
            "task_episode_indices": torch.tensor(task_episode_indices, dtype=torch.long),
            "episode_lengths": torch.tensor(episode_lengths, dtype=torch.long),
            "sample_limits": torch.tensor(sample_limits, dtype=torch.long),
            "source_ranks": torch.tensor(source_ranks, dtype=torch.long),
            "replay_source_ids": torch.tensor(replay_source_ids, dtype=torch.long),
        }
        if proprio is not None:
            batch["proprio"] = torch.from_numpy(proprio.astype(np.float32, copy=False))
        if lang_emb is not None:
            batch["lang_emb"] = torch.from_numpy(lang_emb.astype(np.float32, copy=False))
        if bool(include_images) and all("image" in step for window in windows for step in window):
            images = np.stack([[step["image"] for step in window] for window in windows], axis=0)
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
        sampling_protocol: str = "lumos",
        balance_batches: bool = False,
    ) -> dict[str, torch.Tensor]:
        window = int(window)
        chunk_size = int(chunk_size)
        window_env = window * chunk_size
        stride = max(1, int(early_neg_stride))
        sampling_protocol = str(sampling_protocol).lower()
        if sampling_protocol not in {"lumos", "wmpo"}:
            raise ValueError(
                f"sampling_protocol must be one of {{'lumos', 'wmpo'}}, got {sampling_protocol!r}"
            )
        balance_batches = bool(balance_batches)
        if balance_batches and int(batch_size) % 2:
            raise ValueError("balanced classifier batches require an even batch_size")
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
        proprio_windows: list[np.ndarray] = []
        lang_embs: list[np.ndarray] = []
        pending_key = (
            window_env,
            chunk_size,
            str(chunk_pool),
            stride,
            sampling_protocol,
            balance_batches,
        )
        if self._classifier_pending_key != pending_key:
            self._classifier_pending_windows.clear()
            self._classifier_pending_key = pending_key

        def _append_window(
            record: Mapping[str, Any],
            *,
            end: int,
            label: int,
            end_window: bool,
        ) -> None:
            episode = record["episode"]
            finish_step = int(record.get("finish_step", len(episode)))

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

            selected_steps = episode[end - window_env : end]
            if all("proprio" in step for step in selected_steps):
                env_proprio = np.stack(
                    [
                        np.asarray(step["proprio"], dtype=np.float32).reshape(-1)
                        for step in selected_steps
                    ],
                    axis=0,
                )
                if chunk_size > 1:
                    trailing_shape = env_proprio.shape[1:]
                    reshaped_proprio = env_proprio.reshape(window, chunk_size, *trailing_shape)
                    if chunk_pool == "last":
                        pooled_proprio = reshaped_proprio[:, -1]
                    elif chunk_pool == "first":
                        pooled_proprio = reshaped_proprio[:, 0]
                    elif chunk_pool == "mean":
                        pooled_proprio = reshaped_proprio.mean(axis=1)
                    else:
                        raise ValueError(f"unknown chunk_pool={chunk_pool!r}")
                else:
                    pooled_proprio = env_proprio
                proprio_windows.append(np.ascontiguousarray(pooled_proprio, dtype=np.float32))
            if all("lang_emb" in step for step in selected_steps):
                lang_embs.append(
                    np.asarray(selected_steps[0]["lang_emb"], dtype=np.float32).reshape(-1)
                )

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
            is_end_window.append(bool(end_window))

        target_batch = int(batch_size)
        if sampling_protocol == "wmpo":
            positive_specs: list[tuple[Mapping[str, Any], int, int, bool]] = []
            negative_specs: list[tuple[Mapping[str, Any], int, int, bool]] = []
            for record in candidates:
                finish_step = int(record.get("finish_step", len(record["episode"])))
                success = bool(record["success"])
                if success:
                    positive_specs.append((record, finish_step, 1, True))
                    max_negative_end = int(finish_step) - int(window_env)
                    if max_negative_end >= window_env:
                        negative_ends = list(range(max_negative_end, window_env - 1, -stride))
                        if not negative_ends:
                            negative_ends = [max_negative_end]
                        for end in negative_ends:
                            negative_specs.append((record, int(end), 0, False))
                else:
                    negative_ends = list(range(finish_step, window_env - 1, -stride))
                    if not negative_ends:
                        negative_ends = [finish_step]
                    for end in negative_ends:
                        negative_specs.append((record, int(end), 0, int(end) == int(finish_step)))
            if not positive_specs:
                raise RuntimeError("online replay has no WMPO positive classifier windows")
            if not negative_specs:
                raise RuntimeError("online replay has no WMPO negative classifier windows")
            if balance_batches:
                for index in range(target_batch):
                    specs = positive_specs if index % 2 == 0 else negative_specs
                    record, end, label, end_window = random.choice(specs)
                    _append_window(
                        record,
                        end=int(end),
                        label=int(label),
                        end_window=bool(end_window),
                    )
            else:
                specs = positive_specs + negative_specs
                while len(windows) < target_batch:
                    record, end, label, end_window = random.choice(specs)
                    _append_window(
                        record,
                        end=int(end),
                        label=int(label),
                        end_window=bool(end_window),
                    )
        else:
            while len(windows) < target_batch:
                if self._classifier_pending_windows:
                    pending_record, end, label, end_window = (
                        self._classifier_pending_windows.popleft()
                    )
                    _append_window(
                        pending_record,
                        end=end,
                        label=label,
                        end_window=end_window,
                    )
                    continue

                record = random.choice(candidates)
                episode = record["episode"]
                finish_step = int(record.get("finish_step", len(episode)))
                sample_specs = [(finish_step, int(bool(record["success"])), True)]
                if finish_step - stride >= window_env:
                    valid_ends = list(range(finish_step - stride, window_env - 1, -stride))
                    valid_ends = valid_ends or list(range(finish_step - 1, window_env - 1, -1))
                    if valid_ends:
                        sample_specs.append((int(random.choice(valid_ends)), 0, False))
                for end, label, end_window in sample_specs:
                    if len(windows) >= target_batch:
                        self._classifier_pending_windows.append(
                            (record, int(end), int(label), bool(end_window))
                        )
                        continue
                    _append_window(record, end=end, label=label, end_window=end_window)

        batch = {
            "windows": torch.from_numpy(np.stack(windows, axis=0)).to(torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
            "collection_indices": torch.tensor(collection_indices, dtype=torch.long),
            "task_episode_indices": torch.tensor(task_episode_indices, dtype=torch.long),
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
            "source_ranks": torch.tensor(source_ranks, dtype=torch.long),
            "replay_source_ids": torch.tensor(replay_source_ids, dtype=torch.long),
            "source_success": torch.tensor(source_successes, dtype=torch.bool),
            "episode_lengths": torch.tensor(episode_lengths, dtype=torch.long),
            "finish_steps": torch.tensor(finish_steps, dtype=torch.long),
            "window_end_indices": torch.tensor(window_end_indices, dtype=torch.long),
            "is_end_window": torch.tensor(is_end_window, dtype=torch.bool),
        }
        if len(proprio_windows) == len(windows):
            batch["proprio"] = torch.from_numpy(
                np.stack(proprio_windows, axis=0).astype(np.float32, copy=False)
            )
        if len(lang_embs) == len(windows):
            batch["lang_emb"] = torch.from_numpy(
                np.stack(lang_embs, axis=0).astype(np.float32, copy=False)
            )
        return batch

    def classifier_window_count(self, *, window: int, chunk_size: int) -> int:
        window_env = int(window) * int(chunk_size)
        return sum(
            1
            for record in self._valid_records()
            if int(record.get("finish_step", len(record["episode"]))) >= window_env
        )

    def classifier_endpoint_windows(
        self,
        *,
        window: int,
        chunk_size: int,
        chunk_pool: str,
    ) -> dict[str, torch.Tensor]:
        """Return one deterministic terminal window for every current episode."""

        window = int(window)
        chunk_size = int(chunk_size)
        window_env = window * chunk_size
        records = [
            record
            for record in self._valid_records()
            if int(record.get("finish_step", len(record["episode"]))) >= window_env
        ]
        if not records:
            raise RuntimeError(
                f"online replay has no classifier endpoint windows with finish_step >= {window_env}"
            )

        windows: list[np.ndarray] = []
        labels: list[int] = []
        task_ids: list[int] = []
        episode_ids: list[int] = []
        proprio_windows: list[np.ndarray] = []
        lang_embs: list[np.ndarray] = []
        all_have_proprio = True
        all_have_language = True
        for record in sorted(records, key=lambda item: int(item["episode_id"])):
            episode = record["episode"]
            end = int(record.get("finish_step", len(episode)))
            selected = episode[end - window_env : end]
            env_window = np.stack(
                [step["obs_embedding"] for step in selected],
                axis=0,
            ).astype(np.float32, copy=False)
            windows.append(
                np.ascontiguousarray(
                    _pool_classifier_steps(
                        env_window,
                        window=window,
                        chunk_size=chunk_size,
                        chunk_pool=str(chunk_pool),
                    ),
                    dtype=np.float32,
                )
            )
            labels.append(int(bool(record["success"])))
            task_ids.append(int(record["task_id"]))
            episode_ids.append(int(record["episode_id"]))

            if all("proprio" in step for step in selected):
                proprio = np.stack(
                    [
                        np.asarray(step["proprio"], dtype=np.float32).reshape(-1)
                        for step in selected
                    ],
                    axis=0,
                )
                proprio_windows.append(
                    np.ascontiguousarray(
                        _pool_classifier_steps(
                            proprio,
                            window=window,
                            chunk_size=chunk_size,
                            chunk_pool=str(chunk_pool),
                        ),
                        dtype=np.float32,
                    )
                )
            else:
                all_have_proprio = False
            if all("lang_emb" in step for step in selected):
                lang_embs.append(np.asarray(selected[0]["lang_emb"], dtype=np.float32).reshape(-1))
            else:
                all_have_language = False

        batch = {
            "windows": torch.from_numpy(np.stack(windows, axis=0)).float(),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task_ids": torch.tensor(task_ids, dtype=torch.long),
            "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
        }
        if all_have_proprio and len(proprio_windows) == len(windows):
            batch["proprio"] = torch.from_numpy(np.stack(proprio_windows, axis=0)).float()
        if all_have_language and len(lang_embs) == len(windows):
            batch["lang_emb"] = torch.from_numpy(np.stack(lang_embs, axis=0)).float()
        return batch


def _pool_classifier_steps(
    values: np.ndarray,
    *,
    window: int,
    chunk_size: int,
    chunk_pool: str,
) -> np.ndarray:
    if int(chunk_size) == 1:
        return values
    reshaped = values.reshape(int(window), int(chunk_size), *values.shape[1:])
    if str(chunk_pool) == "last":
        return reshaped[:, -1]
    if str(chunk_pool) == "first":
        return reshaped[:, 0]
    if str(chunk_pool) == "mean":
        return reshaped.mean(axis=1)
    raise ValueError(f"unknown chunk_pool={chunk_pool!r}")


def _step_reward(step: Mapping[str, Any]) -> float:
    if "reward" in step:
        return float(step["reward"])
    if "rewards" in step:
        return float(step["rewards"])
    return float(step.get("sparse_rewards", 0.0))


def _step_done(step: Mapping[str, Any]) -> float:
    if "done" in step:
        return float(step["done"])
    if "dones" in step:
        return float(step["dones"])
    return float(step.get("is_last", 0.0))


def _step_is_terminal(step: Mapping[str, Any]) -> float:
    if "is_terminal" in step:
        return float(step["is_terminal"])
    if "sparse_rewards" in step:
        return float(step["sparse_rewards"])
    return 1.0 if _step_reward(step) > 0.0 else 0.0


def _step_is_last(step: Mapping[str, Any]) -> float:
    if "is_last" in step:
        return float(step["is_last"])
    return _step_done(step)


def _step_action(step: Mapping[str, Any]) -> np.ndarray:
    if "wm_action" in step:
        return np.asarray(step["wm_action"], dtype=np.float32).reshape(-1)
    if "actions" in step:
        return np.asarray(step["actions"], dtype=np.float32).reshape(-1)
    if "action" in step:
        return np.asarray(step["action"], dtype=np.float32).reshape(-1)
    raise KeyError("step is missing wm_action/actions/action")


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
        stats[str(int(task_id))]["episodes"] >= int(min_episodes_per_task) for task_id in task_ids
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
