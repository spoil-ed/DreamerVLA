from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from dreamer_vla.dataset.libero_pixel_sequence_dataset import (
    LIBEROPixelSequenceDataset,
)


class LIBEROPixelRynnHiddenSequenceDataset(LIBEROPixelSequenceDataset):
    """LIBERO pixel windows plus precomputed RynnVLA hidden observations.

    The original pixel HDF5 files remain the image/reconstruction source.  This
    dataset reads a sidecar HDF5 directory with matching filenames and per-demo
    ``data/<demo_key>/obs_embedding`` arrays, then returns both:

      images:        [T, C, H, W], uint8-range float tensor from the source HDF5
      obs_embedding: [T, D], precomputed frozen RynnVLA hidden vector
    """

    def __init__(
        self,
        hdf5_dir: str | Path,
        hidden_dir: str | Path,
        sequence_length: int = 32,
        image_size: int = 256,
        image_keys: Sequence[str] = ("agentview_rgb", "eye_in_hand_rgb"),
        hidden_key: str = "obs_embedding",
        max_files: int | None = None,
        max_demos_per_file: int | None = None,
        max_windows: int | None = None,
        stride: int = 1,
        expected_model_path: str | None = None,
        expected_encoder_state_ckpt: str | None = None,
        expected_time_horizon: int | None = None,
        expected_action_head_type: str | None = None,
        expected_obs_hidden_source: str | None = None,
        expected_prompt_style: str | None = None,
        expected_history: int | None = None,
        expected_include_state: bool | None = None,
        expected_rotate_images_180: bool | None = None,
        require_preprocess_config: bool = True,
        load_actor_sequence: bool = False,
        actor_sequence_length: int | None = None,
        actor_hidden_key: str = "actor_hidden_states",
        actor_input_ids_key: str = "actor_input_ids",
        actor_attention_mask_key: str = "actor_attention_mask",
        actor_seq_lens_key: str = "actor_seq_lens",
    ) -> None:
        super().__init__(
            hdf5_dir=hdf5_dir,
            sequence_length=sequence_length,
            image_size=image_size,
            image_keys=image_keys,
            max_files=max_files,
            max_demos_per_file=max_demos_per_file,
            max_windows=max_windows,
            stride=stride,
        )
        self.hidden_dir = self.resolve_project_path(hidden_dir)
        if not self.hidden_dir.exists():
            raise FileNotFoundError(
                f"Rynn hidden sidecar directory does not exist: {self.hidden_dir}"
            )
        self.hidden_key = str(hidden_key)
        self.load_actor_sequence = bool(load_actor_sequence)
        self.actor_sequence_length = (
            int(actor_sequence_length) if actor_sequence_length is not None else None
        )
        self.actor_hidden_key = str(actor_hidden_key)
        self.actor_input_ids_key = str(actor_input_ids_key)
        self.actor_attention_mask_key = str(actor_attention_mask_key)
        self.actor_seq_lens_key = str(actor_seq_lens_key)
        self._hidden_file_cache: dict[str, h5py.File] = {}
        self._validate_hidden_sidecar(
            expected_model_path=expected_model_path,
            expected_encoder_state_ckpt=expected_encoder_state_ckpt,
            expected_time_horizon=expected_time_horizon,
            expected_action_head_type=expected_action_head_type,
            expected_obs_hidden_source=expected_obs_hidden_source,
            expected_prompt_style=expected_prompt_style,
            expected_history=expected_history,
            expected_include_state=expected_include_state,
            expected_rotate_images_180=expected_rotate_images_180,
            require_preprocess_config=bool(require_preprocess_config),
        )

    @staticmethod
    def _canonical_path(value: str) -> str:
        return str(Path(value).expanduser().resolve())

    @staticmethod
    def _legacy_data_checkpoint_suffix(value: str) -> tuple[str, ...] | None:
        parts = Path(value).expanduser().parts
        for index in range(len(parts) - 1):
            if parts[index] == "data" and parts[index + 1] in {
                "ckpts",
                "checkpoints",
            }:
                return ("data", "checkpoints", *parts[index + 2 :])
        return None

    @classmethod
    def _same_path(cls, left: str | None, right: str | None) -> bool:
        if not left or not right:
            return left == right
        if cls._canonical_path(left) == cls._canonical_path(right):
            return True
        left_suffix = cls._legacy_data_checkpoint_suffix(left)
        right_suffix = cls._legacy_data_checkpoint_suffix(right)
        return bool(left_suffix and right_suffix and left_suffix == right_suffix)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _validate_hidden_sidecar(
        self,
        *,
        expected_model_path: str | None,
        expected_encoder_state_ckpt: str | None,
        expected_time_horizon: int | None,
        expected_action_head_type: str | None,
        expected_obs_hidden_source: str | None = None,
        expected_prompt_style: str | None = None,
        expected_history: int | None = None,
        expected_include_state: bool | None = None,
        expected_rotate_images_180: bool | None = None,
        require_preprocess_config: bool = True,
    ) -> None:
        config_path = self.hidden_dir / "preprocess_config.json"
        if not config_path.is_file():
            if require_preprocess_config:
                raise FileNotFoundError(
                    f"Rynn hidden sidecar is missing preprocess_config.json: {config_path}"
                )
            return
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        errors: list[str] = []
        if expected_model_path and not self._same_path(
            config.get("model_path"), expected_model_path
        ):
            errors.append(
                f"model_path mismatch: sidecar={config.get('model_path')!r}, expected={expected_model_path!r}"
            )
        if expected_encoder_state_ckpt and not self._same_path(
            config.get("encoder_state_ckpt"),
            expected_encoder_state_ckpt,
        ):
            errors.append(
                "encoder_state_ckpt mismatch: "
                f"sidecar={config.get('encoder_state_ckpt')!r}, expected={expected_encoder_state_ckpt!r}"
            )
        if expected_time_horizon is not None:
            got = config.get("time_horizon")
            if got is None or int(got) != int(expected_time_horizon):
                errors.append(
                    f"time_horizon mismatch: sidecar={got!r}, expected={int(expected_time_horizon)}"
                )
        if expected_action_head_type:
            got = str(config.get("action_head_type", "legacy"))
            if got != str(expected_action_head_type):
                errors.append(
                    f"action_head_type mismatch: sidecar={got!r}, expected={expected_action_head_type!r}"
                )
        if expected_obs_hidden_source:
            got = str(config.get("obs_hidden_source", "pooled"))
            if got != str(expected_obs_hidden_source):
                errors.append(
                    f"obs_hidden_source mismatch: sidecar={got!r}, expected={expected_obs_hidden_source!r}"
                )
        if expected_prompt_style:
            got = str(config.get("prompt_style", "legacy"))
            if got != str(expected_prompt_style):
                errors.append(
                    f"prompt_style mismatch: sidecar={got!r}, expected={expected_prompt_style!r}"
                )
        if expected_history is not None:
            got = config.get("history")
            if got is None or int(got) != int(expected_history):
                errors.append(
                    f"history mismatch: sidecar={got!r}, expected={int(expected_history)}"
                )
        if expected_include_state is not None:
            got = self._as_bool(config.get("include_state", False))
            expected = self._as_bool(expected_include_state)
            if got != expected:
                errors.append(
                    f"include_state mismatch: sidecar={got!r}, expected={expected!r}"
                )
        if expected_rotate_images_180 is not None:
            got = self._as_bool(config.get("rotate_images_180", False))
            expected = self._as_bool(expected_rotate_images_180)
            if got != expected:
                errors.append(
                    "rotate_images_180 mismatch: "
                    f"sidecar={got!r}, expected={expected!r}"
                )
        if errors:
            joined = "\n  - ".join(errors)
            raise ValueError(
                f"Rynn hidden sidecar metadata does not match this run: {self.hidden_dir}\n"
                f"  - {joined}"
            )
        if self.load_actor_sequence and not bool(
            config.get("save_actor_sequence", False)
        ):
            raise ValueError(
                f"Rynn hidden sidecar was not generated with --save-actor-sequence: {self.hidden_dir}"
            )

    def _hidden_path_for_source(self, source_path: str | Path) -> Path:
        return self.hidden_dir / Path(source_path).name

    def _hidden_file(self, source_path: str | Path) -> h5py.File:
        hidden_path = self._hidden_path_for_source(source_path)
        key = str(hidden_path)
        handle = self._hidden_file_cache.get(key)
        if handle is None:
            if not hidden_path.is_file():
                raise FileNotFoundError(
                    f"Missing Rynn hidden sidecar for {source_path}: {hidden_path}"
                )
            handle = h5py.File(hidden_path, mode="r", swmr=True, libver="latest")
            self._hidden_file_cache[key] = handle
        return handle

    @staticmethod
    def _pad_or_truncate_array(
        array: np.ndarray,
        target_length: int,
        axis: int,
        pad_value: int | float | bool = 0,
    ) -> np.ndarray:
        current = int(array.shape[axis])
        if current == int(target_length):
            return array
        if current > int(target_length):
            slices = [slice(None)] * array.ndim
            slices[axis] = slice(0, int(target_length))
            return array[tuple(slices)]
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, int(target_length) - current)
        return np.pad(array, pad_width, mode="constant", constant_values=pad_value)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._entries[int(index)]
        item = super().__getitem__(index)
        start = int(entry.start)
        end = start + self.sequence_length
        handle = self._hidden_file(entry.file_path)
        try:
            dset = handle["data"][entry.demo_key][self.hidden_key]
        except KeyError as exc:
            raise KeyError(
                f"{self._hidden_path_for_source(entry.file_path)}:{entry.demo_key} "
                f"missing {self.hidden_key}"
            ) from exc
        if int(dset.shape[0]) < end:
            raise ValueError(
                f"Hidden sidecar length mismatch for {entry.demo_key}: "
                f"need end={end}, have {dset.shape[0]}"
            )
        hidden = np.asarray(dset[start:end])
        item["obs_embedding"] = torch.from_numpy(hidden)
        if self.load_actor_sequence:
            demo = handle["data"][entry.demo_key]
            try:
                actor_hidden = np.asarray(demo[self.actor_hidden_key][start:end])
                actor_input_ids = np.asarray(demo[self.actor_input_ids_key][start:end])
                actor_attention_mask = np.asarray(
                    demo[self.actor_attention_mask_key][start:end]
                )
                actor_seq_lens = np.asarray(demo[self.actor_seq_lens_key][start:end])
            except KeyError as exc:
                raise KeyError(
                    f"{self._hidden_path_for_source(entry.file_path)}:{entry.demo_key} "
                    "missing full actor sequence fields"
                ) from exc
            if int(actor_hidden.shape[0]) != self.sequence_length:
                raise ValueError(
                    f"Actor hidden sidecar length mismatch for {entry.demo_key}: "
                    f"need {self.sequence_length}, got {actor_hidden.shape[0]}"
                )
            if self.actor_sequence_length is not None:
                seq_len = int(self.actor_sequence_length)
                actor_hidden = self._pad_or_truncate_array(
                    actor_hidden, seq_len, axis=1, pad_value=0
                )
                actor_input_ids = self._pad_or_truncate_array(
                    actor_input_ids, seq_len + 1, axis=1, pad_value=0
                )
                actor_attention_mask = self._pad_or_truncate_array(
                    actor_attention_mask,
                    seq_len + 1,
                    axis=1,
                    pad_value=False,
                )
                actor_seq_lens = np.minimum(actor_seq_lens, seq_len).astype(
                    np.int32, copy=False
                )
            item["actor_hidden_states"] = torch.from_numpy(actor_hidden)
            item["actor_input_ids"] = torch.from_numpy(
                actor_input_ids.astype(np.int64, copy=False)
            )
            item["actor_attention_mask"] = torch.from_numpy(
                actor_attention_mask.astype(np.bool_, copy=False)
            )
            item["actor_seq_lens"] = torch.from_numpy(
                actor_seq_lens.astype(np.int64, copy=False)
            )
        item["hidden_path"] = str(self._hidden_path_for_source(entry.file_path))
        return item


__all__ = ["LIBEROPixelRynnHiddenSequenceDataset"]
