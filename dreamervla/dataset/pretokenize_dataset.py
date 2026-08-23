from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamervla.dataset import _pretokenize_helpers as _h
from dreamervla.dataset.base_dataset import BaseDataset


@dataclass(frozen=True)
class PretokenizeDataSpec:
    config_path: str
    manifest_path: str
    num_samples: int
    max_token_length: int
    history: int = 1
    batch_length: int | None = None
    replay_context: int | None = None
    sequence_length: int | None = None
    stride: int | None = None
    sequence_next_obs_source: str = "current_obs"
    prompt_text: str | None = None
    one_trajectory_sft: bool = False
    trajectories_per_task: int | None = None
    trajectory_offset: int | None = None
    selected_trajectory_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FrameRecord:
    file: str
    task_name: str
    trajectory_key: str
    frame_index: int
    image_path: str
    action_path: str
    reward_path: str
    record_index: int
    record_meta: dict[str, Any]


@dataclass(frozen=True)
class _Window:
    records: tuple[_FrameRecord, ...]


class PretokenizeDataset(BaseDataset):
    """Loads pretokenized samples.

    Default mode returns the original flat pretokenized sample format.  Passing
    ``batch_length``/``replay_context`` enables contiguous WM sequence windows;
    ``batch_length=1, replay_context=1`` gives T=2 and matches the old
    single-transition WM semantics.
    """

    def __init__(
        self,
        config_path: str | Path,
        history: int | None = None,
        batch_length: int | None = None,
        replay_context: int | None = None,
        sequence_length: int | None = None,
        stride: int | None = None,
        sequence_next_obs_source: str | None = None,
    ) -> None:
        super().__init__()
        self.config_path = self.resolve_project_path(config_path)
        config = self._load_config(self.config_path)

        meta_entries = config.get("META")
        if not isinstance(meta_entries, list) or not meta_entries:
            raise ValueError(
                f"PretokenizeDataset expects META to be a non-empty list in {self.config_path}"
            )
        manifest_value = meta_entries[0].get("path")
        if manifest_value is None:
            raise ValueError(f"PretokenizeDataset META[0] is missing 'path' in {self.config_path}")

        self.manifest_path = self.resolve_project_path(
            manifest_value, base_dir=self.config_path.parent
        )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = json.load(handle)
        if not isinstance(self.records, list):
            raise ValueError(f"Pretokenize manifest must be a list: {self.manifest_path}")

        self.max_token_length = max((int(item.get("len", 0)) for item in self.records), default=0)
        seq_cfg = config.get("sequence", {}) or {}
        self.history = max(
            int(
                history if history is not None else seq_cfg.get("history", config.get("history", 1))
            ),
            1,
        )
        seq_window_cfg = {key: value for key, value in seq_cfg.items() if key != "history"}
        self.sequence_mode = (
            batch_length is not None
            or replay_context is not None
            or sequence_length is not None
            or stride is not None
            or bool(seq_window_cfg)
        )
        self.batch_length: int | None = None
        self.replay_context: int | None = None
        self.sequence_length: int | None = None
        self.stride: int | None = None
        self.sequence_next_obs_source = str(
            sequence_next_obs_source
            if sequence_next_obs_source is not None
            else seq_cfg.get(
                "next_obs_source", config.get("sequence_next_obs_source", "current_obs")
            )
        )
        if self.sequence_next_obs_source not in {"current_obs", "flat_next_obs"}:
            raise ValueError(
                "sequence_next_obs_source must be 'current_obs' or "
                f"'flat_next_obs', got {self.sequence_next_obs_source!r}"
            )
        self._frames_by_key: dict[str, dict[int, _FrameRecord]] = {}
        self._sequence_records: list[_FrameRecord] = []
        self._windows: list[_Window] = []
        self.action_dim = 0
        # Per-worker bounded LRU of loaded frame payloads (W4): stride=1 makes
        # adjacent sequence windows reload nearly the same frames, so cache the
        # decoded pickle keyed by frame path. A plain instance attribute means
        # each DataLoader worker process gets its own copy (mirrors the
        # ``cached_hdf5_file`` per-worker handle-cache idiom; never shared).
        self._frame_payload_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._frame_cache_capacity = 64
        num_samples = len(self.records)

        if self.sequence_mode:
            self._index_sequence_records()

        if self.sequence_mode:
            self.batch_length = int(
                batch_length
                if batch_length is not None
                else seq_cfg.get("batch_length", config.get("batch_length", 64))
            )
            self.replay_context = int(
                replay_context
                if replay_context is not None
                else seq_cfg.get("replay_context", config.get("replay_context", 1))
            )
            default_sequence_length = self.batch_length + self.replay_context
            self.sequence_length = int(
                sequence_length
                if sequence_length is not None
                else seq_cfg.get(
                    "sequence_length",
                    config.get("sequence_length", default_sequence_length),
                )
            )
            self.stride = max(
                int(
                    stride if stride is not None else seq_cfg.get("stride", config.get("stride", 1))
                ),
                1,
            )
            if self.sequence_length < 2:
                raise ValueError("sequence_length must be >= 2")
            self._build_sequence_windows()
            num_samples = len(self._windows)

        self._data_spec = PretokenizeDataSpec(
            config_path=str(self.config_path),
            manifest_path=str(self.manifest_path),
            num_samples=num_samples,
            max_token_length=self.max_token_length,
            history=self.history,
            batch_length=self.batch_length,
            replay_context=self.replay_context,
            sequence_length=self.sequence_length,
            stride=self.stride,
            sequence_next_obs_source=self.sequence_next_obs_source,
            prompt_text=config.get("prompt_text"),
        )

    _load_config = staticmethod(_h.load_config)

    @property
    def data_spec(self) -> PretokenizeDataSpec:
        return self._data_spec

    def get_normalizer(self) -> dict[str, dict[str, torch.Tensor]]:
        if self.sequence_mode:
            return {
                "action": {
                    "mean": torch.zeros(self.action_dim, dtype=torch.float32),
                    "std": torch.ones(self.action_dim, dtype=torch.float32),
                }
            }
        return {}

    def __len__(self) -> int:
        if self.sequence_mode:
            return len(self._windows)
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.sequence_mode:
            return self._getitem_sequence(index)
        record = self.records[index]
        file_path = self.resolve_project_path(record["file"], base_dir=self.manifest_path.parent)
        with file_path.open("rb") as handle:
            payload = pickle.load(handle)
        return self._flat_item_from_payload(record, file_path, payload, index)

    def _flat_item_from_payload(
        self,
        record: dict[str, Any],
        file_path: Path,
        payload: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        input_ids = list(payload["token"])
        labels = list(payload["label"])
        meta = dict(record.get("meta", {}))
        if isinstance(payload, dict) and "meta" in payload and isinstance(payload["meta"], dict):
            meta.update(payload["meta"])

        image = list(payload.get("image", [])) if isinstance(payload, dict) else []
        action = list(payload.get("action", [])) if isinstance(payload, dict) else []
        state = list(payload.get("state", [])) if isinstance(payload, dict) else []
        next_obs = (
            dict(payload.get("next_obs", {}))
            if isinstance(payload, dict) and isinstance(payload.get("next_obs"), dict)
            else {}
        )
        reward_value = payload.get("reward") if isinstance(payload, dict) else None
        if reward_value is None:
            reward_value = meta.get("reward", 0.0)
        task_name = payload.get("task_name") if isinstance(payload, dict) else None
        if task_name is None:
            task_name = meta.get("task_name", "")
        wm_obs_input_ids = payload.get("wm_obs_input_ids") if isinstance(payload, dict) else None
        if not isinstance(wm_obs_input_ids, list):
            wm_obs_input_ids = list(input_ids)
        wm_next_obs_input_ids = (
            payload.get("wm_next_obs_input_ids") if isinstance(payload, dict) else None
        )
        if not isinstance(wm_next_obs_input_ids, list):
            wm_next_obs_input_ids = list(wm_obs_input_ids)

        wm_action = self._load_action_sequence(action)

        # EOT padding mask produced by pretokenize: first `effective_horizon`
        # action steps are real, the rest are padded (action tensor still has
        # values there, but WM should treat them as no-op).  Older pkls without
        # this field fall back to "all real".
        raw_mask = payload.get("wm_action_mask") if isinstance(payload, dict) else None
        if isinstance(raw_mask, list) and raw_mask:
            wm_action_mask = torch.tensor([bool(x) for x in raw_mask], dtype=torch.bool)
        else:
            wm_action_mask = torch.ones(
                int(wm_action.shape[0]) if wm_action.ndim >= 1 else 0, dtype=torch.bool
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": int(len(input_ids)),
            "image": image,
            "action": action,
            "state": state,
            "next_obs": next_obs,
            "reward": float(reward_value) if reward_value is not None else 0.0,
            "task_name": str(task_name),
            "wm_obs_input_ids": [int(x) for x in wm_obs_input_ids],
            "wm_next_obs_input_ids": [int(x) for x in wm_next_obs_input_ids],
            "wm_action": wm_action,
            "wm_action_mask": wm_action_mask,
            "meta": meta,
            "file": str(file_path),
            "id": int(payload.get("id", record.get("id", index))),
        }

    _load_action_sequence = staticmethod(_h.load_action_sequence)

    @staticmethod
    def _record_current_image(record: dict[str, Any]) -> Any:
        """Current-frame image list carried by the manifest, if any.

        Mirrors ``one_trajectory_pretokenize_dataset.py`` by preferring the
        ``meta.next_obs.image`` / ``next_obs.image`` field. The W3 manifest-first
        index is only taken when this image parses to the same current-frame
        layout the pickle scan would build (validated by the caller), so it stays
        byte-identical to ``payload['image']``.
        """
        meta = record.get("meta")
        if isinstance(meta, dict):
            next_obs = meta.get("next_obs")
            if isinstance(next_obs, dict) and next_obs.get("image"):
                return next_obs.get("image")
        next_obs = record.get("next_obs")
        if isinstance(next_obs, dict) and next_obs.get("image"):
            return next_obs.get("image")
        return None

    def _frame_from_manifest(
        self, record_index: int, record: dict[str, Any]
    ) -> _FrameRecord | None:
        """Build a frame record from the manifest without unpickling (W3).

        Returns ``None`` (caller falls back to the pickle scan) when the manifest
        lacks a current-frame image or it does not match the on-disk current-frame
        layout (its sibling ``action`` file is absent), keeping byte-identity.
        """
        images = self._record_current_image(record)
        if images is None:
            return None
        image_path = self._select_current_third_view(images)
        parsed = self._parse_image_path(image_path)
        if parsed is None:
            return None
        task_name, trajectory_key, frame_index = parsed
        action_path = self._sibling_step_path(image_path, "action", "action", ".npy")
        # Guard: only trust the manifest image when it is provably the current
        # frame (its sibling action file exists on disk). Manifests that store
        # the *next* observation will miss this and fall back to the scan.
        if not action_path or not Path(action_path).is_file():
            return None
        file_path = self.resolve_project_path(record["file"], base_dir=self.manifest_path.parent)
        manifest_task = (
            record.get("meta", {}).get("task_name")
            if isinstance(record.get("meta"), dict)
            else None
        )
        return _FrameRecord(
            file=str(file_path),
            task_name=str(manifest_task or task_name),
            trajectory_key=trajectory_key,
            frame_index=frame_index,
            image_path=image_path,
            action_path=action_path,
            reward_path=self._sibling_step_path(image_path, "reward", "reward", ".npy"),
            record_index=record_index,
            record_meta=dict(record.get("meta", {})),
        )

    def _frame_from_pickle(self, record_index: int, record: dict[str, Any]) -> _FrameRecord | None:
        file_path = self.resolve_project_path(record["file"], base_dir=self.manifest_path.parent)
        try:
            with file_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        image_path = self._select_current_third_view(payload.get("image", []))
        parsed = self._parse_image_path(image_path)
        if parsed is None:
            return None
        task_name, trajectory_key, frame_index = parsed
        return _FrameRecord(
            file=str(file_path),
            task_name=str(payload.get("task_name") or task_name),
            trajectory_key=trajectory_key,
            frame_index=frame_index,
            image_path=image_path,
            action_path=self._sibling_step_path(image_path, "action", "action", ".npy"),
            reward_path=self._sibling_step_path(image_path, "reward", "reward", ".npy"),
            record_index=record_index,
            record_meta=dict(record.get("meta", {})),
        )

    def _index_sequence_records(self) -> None:
        for record_index, record in enumerate(self.records):
            if not isinstance(record, dict) or "file" not in record:
                continue
            # W3: prefer the manifest field over unpickling every frame; fall
            # back to the pickle scan when the manifest can't supply the current
            # frame. Both paths build the identical _FrameRecord.
            frame = self._frame_from_manifest(record_index, record)
            if frame is None:
                frame = self._frame_from_pickle(record_index, record)
            if frame is None:
                continue
            trajectory_key = frame.trajectory_key
            frame_index = frame.frame_index
            # Prefer the first record for each frame. In mixed manifests this is
            # normally the action sample; its wm_obs_input_ids are the same image
            # observation we need for sequence WM training.
            frame_map = self._frames_by_key.setdefault(trajectory_key, {})
            if frame_index not in frame_map:
                frame_map[frame_index] = frame
                self._sequence_records.append(frame)
            if self.action_dim == 0 and Path(frame.action_path).is_file():
                action = np.asarray(np.load(frame.action_path), dtype=np.float32)
                self.action_dim = int(action.reshape(-1).shape[0])

    def _build_sequence_windows(self) -> None:
        if self.sequence_length is None or self.stride is None:
            return
        # New his=1 data stores one current observation per pkl.  WM sequences
        # are therefore built from unique contiguous observation frames in a
        # single trajectory.  ``history`` expands the temporal window, but does
        # not concatenate token embeddings inside each timestep: every element
        # of wm_obs_input_ids_seq is still one current-frame observation.
        obs_count = self.sequence_length + self.history - 1
        for frame_map in self._frames_by_key.values():
            frames = sorted(frame_map)
            frame_set = set(frames)
            for start in frames[:: self.stride]:
                wanted = range(start, start + obs_count)
                if not all(idx in frame_set for idx in wanted):
                    continue
                records = tuple(frame_map[idx] for idx in wanted)
                # Actions align as [zero, a_start, a_start+1, ...], so only the
                # first T-1 frames need transition actions.
                if not all(Path(rec.action_path).is_file() for rec in records[:-1]):
                    continue
                self._windows.append(_Window(records=records))

    _select_current_third_view = staticmethod(_h.select_current_third_view)
    _parse_image_path = staticmethod(_h.parse_image_path)
    _sibling_step_path = staticmethod(_h.sibling_step_path)
    _load_step_action = staticmethod(_h.load_step_action)
    _load_step_reward = staticmethod(_h.load_step_reward)

    def _load_frame_payload(self, file_path: str) -> dict[str, Any]:
        """Load a frame payload via a per-worker bounded LRU (W4).

        Overlapping stride=1 windows reload the same frame; cache the decoded
        pickle keyed by path so a shared frame is read from disk once. Returns
        the same dict ``pickle.load`` would; downstream code copies before use,
        so a cached object is byte-identical.
        """
        cache = self._frame_payload_cache
        cached = cache.get(file_path)
        if cached is not None:
            cache.move_to_end(file_path)
            return cached
        with Path(file_path).open("rb") as handle:
            payload = pickle.load(handle)
        cache[file_path] = payload
        if len(cache) > self._frame_cache_capacity:
            cache.popitem(last=False)
        return payload

    def _getitem_sequence(self, index: int) -> dict[str, Any]:
        window = self._windows[index]
        token_seq: list[list[int]] = []
        metas_seq: list[dict[str, Any]] = []
        actions = [torch.zeros(self.action_dim, dtype=torch.float32)]
        rewards = [0.0]
        dones = [0.0]
        first_flat: dict[str, Any] | None = None
        previous_payload: dict[str, Any] | None = None

        for idx, frame in enumerate(window.records):
            frame_path = Path(frame.file)
            payload = self._load_frame_payload(frame.file)
            obs_ids = payload.get("wm_obs_input_ids")
            if not isinstance(obs_ids, list):
                raise KeyError(f"missing wm_obs_input_ids in {frame.file}")
            token_ids = obs_ids
            if idx > 0 and self.sequence_next_obs_source == "flat_next_obs":
                if previous_payload is None:
                    raise KeyError(f"missing previous payload for sequence window {index}")
                flat_next_ids = previous_payload.get("wm_next_obs_input_ids")
                if not isinstance(flat_next_ids, list):
                    raise KeyError(
                        "sequence_next_obs_source='flat_next_obs' requires "
                        f"wm_next_obs_input_ids in {window.records[idx - 1].file}"
                    )
                token_ids = flat_next_ids
            if idx == 0:
                first_flat = self._flat_item_from_payload(
                    {"file": frame.file, "meta": frame.record_meta},
                    frame_path,
                    payload,
                    index,
                )
            token_seq.append([int(x) for x in token_ids])
            meta = dict(payload.get("meta", {}))
            meta.update(
                {
                    "task_name": frame.task_name,
                    "trajectory_key": frame.trajectory_key,
                    "frame_index": frame.frame_index,
                    "history": self.history,
                    "file": frame.file,
                    "sequence_next_obs_source": self.sequence_next_obs_source,
                }
            )
            metas_seq.append(meta)
            previous_payload = payload

        if first_flat is None:
            raise IndexError(f"empty sequence window at index {index}")

        for frame in window.records[:-1]:
            actions.append(self._load_step_action(frame.action_path, self.action_dim))
            rewards.append(self._load_step_reward(frame.reward_path))
            dones.append(0.0)

        item = dict(first_flat)
        item["wm_obs_input_ids"] = list(token_seq[0])
        if len(token_seq) > 1:
            item["wm_next_obs_input_ids"] = list(token_seq[1])
        item.update(
            {
                "wm_obs_input_ids_seq": token_seq,
                "action_seq": torch.stack(actions, dim=0),
                "reward_seq": torch.tensor(rewards, dtype=torch.float32),
                "done_seq": torch.tensor(dones, dtype=torch.float32),
                "meta_seq": metas_seq,
                "history": self.history,
                "sequence_length": self.sequence_length,
            }
        )
        return item

    @staticmethod
    def _pad_action_batch(
        actions: list[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not actions:
            return None, None
        max_steps = max(int(tensor.shape[0]) for tensor in actions)
        max_dim = max(int(tensor.shape[1]) if tensor.ndim == 2 else 0 for tensor in actions)
        if max_steps <= 0 or max_dim <= 0:
            return None, None
        padded = torch.zeros(len(actions), max_steps, max_dim, dtype=torch.float32)
        mask = torch.zeros(len(actions), max_steps, dtype=torch.bool)
        for idx, tensor in enumerate(actions):
            if tensor.ndim != 2 or tensor.numel() == 0:
                continue
            steps = int(tensor.shape[0])
            dim = int(tensor.shape[1])
            padded[idx, :steps, :dim] = tensor
            mask[idx, :steps] = True
        return padded, mask

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if batch and "wm_obs_input_ids_seq" in batch[0]:
            padded_action, action_mask = PretokenizeDataset._pad_action_batch(
                [item["wm_action"] for item in batch]
            )
            if action_mask is not None:
                max_steps = int(action_mask.shape[1])
                for idx, item in enumerate(batch):
                    sample_mask = item.get("wm_action_mask")
                    if not isinstance(sample_mask, torch.Tensor) or sample_mask.numel() == 0:
                        continue
                    n = min(int(sample_mask.shape[0]), max_steps)
                    action_mask[idx, :n] &= sample_mask[:n].to(action_mask.device)
                    if n < max_steps:
                        action_mask[idx, n:] = False
            return {
                "input_ids": [list(item["input_ids"]) for item in batch],
                "labels": [list(item["labels"]) for item in batch],
                "lengths": torch.tensor([int(item["length"]) for item in batch], dtype=torch.long),
                "image": [item["image"] for item in batch],
                "action": padded_action,
                "action_mask": action_mask,
                "state": [item["state"] for item in batch],
                "next_obs": [item["next_obs"] for item in batch],
                "reward": torch.tensor(
                    [float(item["reward"]) for item in batch], dtype=torch.float32
                ),
                "task_name": [str(item["task_name"]) for item in batch],
                "wm_obs_input_ids": [list(item["wm_obs_input_ids"]) for item in batch],
                "wm_next_obs_input_ids": [list(item["wm_next_obs_input_ids"]) for item in batch],
                "meta": [item["meta"] for item in batch],
                "file": [item["file"] for item in batch],
                "id": torch.tensor([int(item["id"]) for item in batch], dtype=torch.long),
                "wm_obs_input_ids_seq": [
                    [list(step_ids) for step_ids in item["wm_obs_input_ids_seq"]] for item in batch
                ],
                "action_seq": torch.stack([item["action_seq"] for item in batch], dim=0),
                "reward_seq": torch.stack([item["reward_seq"] for item in batch], dim=0),
                "done_seq": torch.stack([item["done_seq"] for item in batch], dim=0),
                "meta_seq": [item["meta_seq"] for item in batch],
                "history": int(batch[0].get("history", 1)),
                "sequence_length": int(batch[0].get("sequence_length", 0) or 0),
            }

        padded_action, action_mask = PretokenizeDataset._pad_action_batch(
            [item["wm_action"] for item in batch]
        )
        # AND in the per-sample EOT padding mask so WM's _apply_action_mask
        # zeroes positions past each sample's effective_horizon.
        if action_mask is not None:
            max_steps = int(action_mask.shape[1])
            for idx, item in enumerate(batch):
                sample_mask = item.get("wm_action_mask")
                if not isinstance(sample_mask, torch.Tensor) or sample_mask.numel() == 0:
                    continue
                n = min(int(sample_mask.shape[0]), max_steps)
                action_mask[idx, :n] &= sample_mask[:n].to(action_mask.device)
                if n < max_steps:
                    action_mask[idx, n:] = False
        return {
            "input_ids": [list(item["input_ids"]) for item in batch],
            "labels": [list(item["labels"]) for item in batch],
            "lengths": torch.tensor([int(item["length"]) for item in batch], dtype=torch.long),
            "image": [item["image"] for item in batch],
            "action": padded_action,
            "action_mask": action_mask,
            "state": [item["state"] for item in batch],
            "next_obs": [item["next_obs"] for item in batch],
            "reward": torch.tensor([float(item["reward"]) for item in batch], dtype=torch.float32),
            "task_name": [str(item["task_name"]) for item in batch],
            "wm_obs_input_ids": [list(item["wm_obs_input_ids"]) for item in batch],
            "wm_next_obs_input_ids": [list(item["wm_next_obs_input_ids"]) for item in batch],
            "meta": [item["meta"] for item in batch],
            "file": [item["file"] for item in batch],
            "id": torch.tensor([int(item["id"]) for item in batch], dtype=torch.long),
        }


class PretokenizeActionChunkDataset(PretokenizeDataset):
    """Build multi-step VLA action chunks from atomic his=1/horizon=1 samples.

    The pretokenized atom already contains the current observation prompt and a
    single action-token block.  For VLA training with a chunking action head, we
    can reuse the atom manifest and concatenate action-token blocks from
    contiguous frames in the same trajectory, avoiding duplicate pretokenized
    datasets for every action horizon.
    """

    def __init__(
        self,
        config_path: str | Path,
        action_horizon: int = 10,
        history: int | None = None,
        batch_length: int | None = None,
        replay_context: int | None = None,
        sequence_length: int | None = None,
        stride: int | None = None,
        sequence_next_obs_source: str | None = None,
    ) -> None:
        super().__init__(
            config_path=config_path,
            history=history,
            batch_length=batch_length,
            replay_context=replay_context,
            sequence_length=sequence_length,
            stride=stride,
            sequence_next_obs_source=sequence_next_obs_source,
        )
        if self.sequence_mode:
            raise ValueError("PretokenizeActionChunkDataset only supports flat atomic manifests.")
        self.action_horizon = max(int(action_horizon), 1)
        self._chunk_windows: list[tuple[int, ...]] = []
        self._record_payload_cache: dict[int, dict[str, Any]] = {}
        self._record_paths: list[Path] = [
            self.resolve_project_path(record["file"], base_dir=self.manifest_path.parent)
            for record in self.records
        ]
        self._build_chunk_windows()
        self._record_payload_cache.clear()
        self._data_spec = PretokenizeDataSpec(
            config_path=str(self.config_path),
            manifest_path=str(self.manifest_path),
            num_samples=len(self._chunk_windows),
            max_token_length=self.max_token_length + max(self.action_horizon - 1, 0) * 9,
            history=self.history,
            prompt_text=self.data_spec.prompt_text,
        )

    def __len__(self) -> int:
        return len(self._chunk_windows)

    def _load_payload_by_index(self, index: int) -> dict[str, Any]:
        cached = self._record_payload_cache.get(index)
        if cached is not None:
            return cached
        with self._record_paths[index].open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {self._record_paths[index]}")
        self._record_payload_cache[index] = payload
        return payload

    def _build_chunk_windows(self) -> None:
        frames_by_key: dict[str, dict[int, int]] = {}
        for record_index, _record in enumerate(self.records):
            try:
                payload = self._load_payload_by_index(record_index)
            except Exception:
                continue
            image_path = self._select_current_third_view(payload.get("image", []))
            parsed = self._parse_image_path(image_path)
            if parsed is None:
                continue
            _task_name, trajectory_key, frame_index = parsed
            frame_map = frames_by_key.setdefault(trajectory_key, {})
            frame_map.setdefault(frame_index, record_index)

        for frame_map in frames_by_key.values():
            frame_set = set(frame_map)
            for start in sorted(frame_map):
                wanted = range(start, start + self.action_horizon)
                if all(frame in frame_set for frame in wanted):
                    self._chunk_windows.append(tuple(frame_map[frame] for frame in wanted))

    @staticmethod
    def _action_label_block(labels: list[int]) -> list[int]:
        valid = [idx for idx, value in enumerate(labels) if int(value) >= 0]
        if not valid:
            raise ValueError("Atomic pretokenized sample has no action labels.")
        block = [int(value) for value in labels[valid[0] : valid[-1] + 1]]
        # Single-step atoms end with EOT. Multi-step chunks should carry one
        # EOT after all action blocks, matching native horizon>1 pretokenization.
        if block and block[-1] == 8710:
            block = block[:-1]
        return block

    @staticmethod
    def _prompt_prefix(input_ids: list[int], labels: list[int]) -> tuple[list[int], list[int]]:
        first_valid = next(
            (idx for idx, value in enumerate(labels) if int(value) >= 0), len(labels)
        )
        return [int(x) for x in input_ids[:first_valid]], [-100] * first_valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_indices = self._chunk_windows[index]
        first_index = record_indices[0]
        first_payload = self._load_payload_by_index(first_index)
        first_record = self.records[first_index]
        first_file = self._record_paths[first_index]

        item = self._flat_item_from_payload(first_record, first_file, first_payload, first_index)

        prefix_ids, prefix_labels = self._prompt_prefix(
            list(first_payload["token"]),
            list(first_payload["label"]),
        )
        action_token_blocks: list[int] = []
        actions: list[Any] = []
        rewards: list[float] = []
        chunk_files: list[str] = []
        for record_index in record_indices:
            payload = self._load_payload_by_index(record_index)
            action_token_blocks.extend(self._action_label_block(list(payload["label"])))
            actions.extend(list(payload.get("action", [])))
            reward_value = payload.get("reward")
            if reward_value is None and isinstance(payload.get("meta"), dict):
                reward_value = payload["meta"].get("reward", 0.0)
            rewards.append(float(reward_value if reward_value is not None else 0.0))
            chunk_files.append(str(self._record_paths[record_index]))

        input_ids = prefix_ids + action_token_blocks + [8710]
        labels = prefix_labels + action_token_blocks + [8710]
        wm_action = self._load_action_sequence(actions)

        meta = dict(item.get("meta", {}))
        meta.update(
            {
                "action_horizon": self.action_horizon,
                "source_action_horizon": 1,
                "chunk_files": chunk_files,
                "effective_horizon": int(wm_action.shape[0]),
                "full_horizon": self.action_horizon,
            }
        )

        item.update(
            {
                "input_ids": input_ids,
                "labels": labels,
                "length": len(input_ids),
                "action": actions,
                "reward": rewards[-1] if rewards else float(item.get("reward", 0.0)),
                "wm_action": wm_action,
                "wm_action_mask": torch.ones(int(wm_action.shape[0]), dtype=torch.bool),
                "meta": meta,
                "id": int(index),
            }
        )
        return item


__all__ = [
    "PretokenizeActionChunkDataset",
    "PretokenizeDataSpec",
    "PretokenizeDataset",
]
