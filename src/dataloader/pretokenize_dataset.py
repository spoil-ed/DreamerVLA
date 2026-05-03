from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dataloader.base_dataset import BaseDataset


_FRAME_RE = re.compile(r"image_(\d+)\.png$")


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
            raise ValueError(f"PretokenizeDataset expects META to be a non-empty list in {self.config_path}")
        manifest_value = meta_entries[0].get("path")
        if manifest_value is None:
            raise ValueError(f"PretokenizeDataset META[0] is missing 'path' in {self.config_path}")

        self.manifest_path = self.resolve_project_path(manifest_value, base_dir=self.config_path.parent)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = json.load(handle)
        if not isinstance(self.records, list):
            raise ValueError(f"Pretokenize manifest must be a list: {self.manifest_path}")

        self.max_token_length = max((int(item.get("len", 0)) for item in self.records), default=0)
        seq_cfg = config.get("sequence", {}) or {}
        self.history = max(
            int(
                history
                if history is not None
                else seq_cfg.get("history", config.get("history", 1))
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
            else seq_cfg.get("next_obs_source", config.get("sequence_next_obs_source", "current_obs"))
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
                else seq_cfg.get("sequence_length", config.get("sequence_length", default_sequence_length))
            )
            self.stride = max(
                int(stride if stride is not None else seq_cfg.get("stride", config.get("stride", 1))),
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

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.load(handle, Loader=yaml.FullLoader)

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
        next_obs = dict(payload.get("next_obs", {})) if isinstance(payload, dict) and isinstance(payload.get("next_obs"), dict) else {}
        reward_value = payload.get("reward") if isinstance(payload, dict) else None
        if reward_value is None:
            reward_value = meta.get("reward", 0.0)
        task_name = payload.get("task_name") if isinstance(payload, dict) else None
        if task_name is None:
            task_name = meta.get("task_name", "")
        wm_obs_input_ids = payload.get("wm_obs_input_ids") if isinstance(payload, dict) else None
        if not isinstance(wm_obs_input_ids, list):
            wm_obs_input_ids = list(input_ids)
        wm_next_obs_input_ids = payload.get("wm_next_obs_input_ids") if isinstance(payload, dict) else None
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
            wm_action_mask = torch.ones(int(wm_action.shape[0]) if wm_action.ndim >= 1 else 0, dtype=torch.bool)

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

    @staticmethod
    def _load_action_sequence(action: list[Any]) -> torch.Tensor:
        values: list[np.ndarray] = []
        for entry in action:
            if isinstance(entry, str):
                path = Path(entry).expanduser()
                if path.is_file():
                    values.append(np.asarray(np.load(path), dtype=np.float32))
                continue
            values.append(np.asarray(entry, dtype=np.float32))
        if not values:
            return torch.zeros((0, 0), dtype=torch.float32)
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        return torch.tensor(array, dtype=torch.float32)

    def _index_sequence_records(self) -> None:
        for record_index, record in enumerate(self.records):
            if not isinstance(record, dict) or "file" not in record:
                continue
            file_path = self.resolve_project_path(
                record["file"], base_dir=self.manifest_path.parent
            )
            try:
                with file_path.open("rb") as handle:
                    payload = pickle.load(handle)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            image_path = self._select_current_third_view(payload.get("image", []))
            parsed = self._parse_image_path(image_path)
            if parsed is None:
                continue
            task_name, trajectory_key, frame_index = parsed
            frame = _FrameRecord(
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

    @staticmethod
    def _select_current_third_view(images: Any) -> str:
        paths = [str(path) for path in (images or [])]
        third = [path for path in paths if "/imgs_third_view/" in path]
        if third:
            return third[-1]
        return paths[-1] if paths else ""

    @staticmethod
    def _parse_image_path(path: str) -> tuple[str, str, int] | None:
        parts = Path(path).parts
        if "imgs_third_view" not in parts:
            return None
        view_idx = parts.index("imgs_third_view")
        if view_idx < 2:
            return None
        match = _FRAME_RE.match(parts[-1])
        if match is None:
            return None
        task_name = parts[view_idx - 2]
        trj_name = parts[view_idx - 1]
        trajectory_key = f"{task_name}/{trj_name}"
        return task_name, trajectory_key, int(match.group(1))

    @staticmethod
    def _sibling_step_path(image_path: str, dirname: str, prefix: str, suffix: str) -> str:
        path = Path(image_path)
        match = _FRAME_RE.match(path.name)
        if match is None:
            return ""
        frame_index = int(match.group(1))
        trj_dir = path.parent.parent
        return str(trj_dir / dirname / f"{prefix}_{frame_index}{suffix}")

    @staticmethod
    def _load_step_action(path: str, action_dim: int) -> torch.Tensor:
        if path and Path(path).is_file():
            action = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
            return torch.tensor(action, dtype=torch.float32)
        return torch.zeros(action_dim, dtype=torch.float32)

    @staticmethod
    def _load_step_reward(path: str) -> float:
        if path and Path(path).is_file():
            return float(np.asarray(np.load(path), dtype=np.float32).reshape(-1)[0])
        return 0.0

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
            with frame_path.open("rb") as handle:
                payload = pickle.load(handle)
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
        item.update({
            "wm_obs_input_ids_seq": token_seq,
            "action_seq": torch.stack(actions, dim=0),
            "reward_seq": torch.tensor(rewards, dtype=torch.float32),
            "done_seq": torch.tensor(dones, dtype=torch.float32),
            "meta_seq": metas_seq,
            "history": self.history,
            "sequence_length": self.sequence_length,
        })
        return item

    @staticmethod
    def _pad_action_batch(actions: list[torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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
                "reward": torch.tensor([float(item["reward"]) for item in batch], dtype=torch.float32),
                "task_name": [str(item["task_name"]) for item in batch],
                "wm_obs_input_ids": [list(item["wm_obs_input_ids"]) for item in batch],
                "wm_next_obs_input_ids": [list(item["wm_next_obs_input_ids"]) for item in batch],
                "meta": [item["meta"] for item in batch],
                "file": [item["file"] for item in batch],
                "id": torch.tensor([int(item["id"]) for item in batch], dtype=torch.long),
                "wm_obs_input_ids_seq": [
                    [list(step_ids) for step_ids in item["wm_obs_input_ids_seq"]]
                    for item in batch
                ],
                "action_seq": torch.stack([item["action_seq"] for item in batch], dim=0),
                "reward_seq": torch.stack([item["reward_seq"] for item in batch], dim=0),
                "done_seq": torch.stack([item["done_seq"] for item in batch], dim=0),
                "meta_seq": [item["meta_seq"] for item in batch],
                "history": int(batch[0].get("history", 1)),
                "sequence_length": int(batch[0].get("sequence_length", 0) or 0),
            }

        padded_action, action_mask = PretokenizeDataset._pad_action_batch([item["wm_action"] for item in batch])
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


class PretokenizeFlatDataset(PretokenizeDataset):
    """Original flat pretokenized dataset behavior.

    This class intentionally ignores sequence-window arguments.  Use it in
    configs when a run must stay on the original
    ``wm_obs_input_ids``/``wm_next_obs_input_ids`` single-transition path.
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
        # Accept and ignore sequence-window arguments so configs can inherit
        # from sequence-enabled WM configs while still forcing the flat path.
        super().__init__(config_path=config_path)


__all__ = ["PretokenizeDataSpec", "PretokenizeDataset", "PretokenizeFlatDataset"]
