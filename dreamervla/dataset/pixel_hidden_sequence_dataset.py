from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from dreamervla.dataset.pixel_sequence_dataset import (
    PixelSequenceDataset,
)
from dreamervla.preprocess.sidecar_schema import (
    DEFAULT_HIDDEN_KEY,
    validate_hidden_token_preprocess_config,
    validate_hidden_token_sidecar_dir,
)


class PixelHiddenSequenceDataset(PixelSequenceDataset):
    """LIBERO pixel windows plus canonical OpenVLA hidden-token observations.

    The original pixel HDF5 files remain the image source. This dataset reads a
    shape-aligned sidecar directory whose only supported payload is:

      images:        [T, C, H, W], uint8-range float tensor from the source HDF5
      obs_embedding: [T, 256, 4096] projected vision hidden tokens
    """

    def __init__(
        self,
        hdf5_dir: str | Path,
        hidden_dir: str | Path,
        sequence_length: int = 32,
        image_size: int = 256,
        image_keys: Sequence[str] = ("agentview_rgb",),
        proprio_keys: Sequence[str] | None = None,
        hidden_key: str = DEFAULT_HIDDEN_KEY,
        lang_emb_dir: str | Path | None = None,
        lang_emb_key: str = "lang_emb",
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
    ) -> None:
        super().__init__(
            hdf5_dir=hdf5_dir,
            sequence_length=sequence_length,
            image_size=image_size,
            image_keys=image_keys,
            proprio_keys=proprio_keys,
            max_files=max_files,
            max_demos_per_file=max_demos_per_file,
            max_windows=max_windows,
            stride=stride,
        )
        self.hidden_dir = self.resolve_project_path(hidden_dir)
        if not self.hidden_dir.exists():
            raise FileNotFoundError(
                f"Hidden sidecar directory does not exist: {self.hidden_dir}"
            )
        if str(hidden_key) != DEFAULT_HIDDEN_KEY:
            raise ValueError(
                f"hidden_key is fixed to {DEFAULT_HIDDEN_KEY!r}, got {hidden_key!r}"
            )
        if not bool(require_preprocess_config):
            raise ValueError("canonical hidden-token sidecars always require metadata")
        self.hidden_key = DEFAULT_HIDDEN_KEY
        self.lang_emb_dir = (
            self.resolve_project_path(lang_emb_dir) if lang_emb_dir is not None else None
        )
        if self.lang_emb_dir is not None and not self.lang_emb_dir.exists():
            raise FileNotFoundError(
                f"Language sidecar directory does not exist: {self.lang_emb_dir}"
            )
        self.lang_emb_key = str(lang_emb_key)
        self._hidden_file_cache: dict[str, h5py.File] = {}
        self._lang_emb_file_cache: dict[str, h5py.File] = {}
        sidecar_config = self._validate_hidden_sidecar(
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
        if sidecar_config.get("hidden_key") != DEFAULT_HIDDEN_KEY:
            raise AssertionError("validated hidden-token sidecar changed hidden_key")

    @staticmethod
    def _canonical_path(value: str) -> str:
        return str(Path(value).expanduser().resolve())

    @staticmethod
    def _same_path(left: str | None, right: str | None) -> bool:
        if not left or not right:
            return left == right
        return PixelHiddenSequenceDataset._canonical_path(
            left
        ) == PixelHiddenSequenceDataset._canonical_path(right)

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
    ) -> dict[str, Any]:
        config_path = self.hidden_dir / "preprocess_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"hidden-token sidecar is missing preprocess_config.json: {config_path}"
            )
        if not bool(require_preprocess_config):
            raise ValueError("canonical hidden-token sidecars always require metadata")
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        validate_hidden_token_preprocess_config(config, context=str(config_path))
        validate_hidden_token_sidecar_dir(
            self.hidden_dir,
            reference_dir=getattr(self, "hdf5_dir", None),
            require_reference_complete=True,
        )
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
            got = str(config["action_head_type"])
            if got != str(expected_action_head_type):
                errors.append(
                    f"action_head_type mismatch: sidecar={got!r}, expected={expected_action_head_type!r}"
                )
        if expected_obs_hidden_source:
            got = str(config["obs_hidden_source"])
            if got != str(expected_obs_hidden_source):
                errors.append(
                    f"obs_hidden_source mismatch: sidecar={got!r}, expected={expected_obs_hidden_source!r}"
                )
        if expected_prompt_style:
            got = str(config.get("prompt_style", ""))
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
                f"Hidden sidecar metadata does not match this run: {self.hidden_dir}\n"
                f"  - {joined}"
            )
        return config

    @staticmethod
    def _flat_hidden_dim_from_shape(shape: tuple[int, ...] | None) -> int | None:
        if shape is None:
            return None
        return int(np.prod(shape, dtype=np.int64))

    def _first_sidecar_hidden_shape(self, hidden_key: str) -> tuple[int, ...] | None:
        for path in sorted(self.hidden_dir.glob("*.hdf5")):
            with h5py.File(path, "r") as handle:
                data = handle.get("data")
                if data is None:
                    continue
                for demo_key in data:
                    demo = data[demo_key]
                    if hidden_key in demo:
                        return tuple(int(dim) for dim in demo[hidden_key].shape[1:])
        return None

    def _first_sidecar_hidden_dim(self, hidden_key: str) -> int | None:
        return self._flat_hidden_dim_from_shape(
            self._first_sidecar_hidden_shape(hidden_key)
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
                    f"Missing hidden sidecar for {source_path}: {hidden_path}"
                )
            handle = h5py.File(hidden_path, mode="r", swmr=True, libver="latest")
            self._hidden_file_cache[key] = handle
        return handle

    def _lang_emb_path_for_source(self, source_path: str | Path) -> Path:
        if self.lang_emb_dir is None:
            raise RuntimeError("lang_emb_dir is not configured")
        return self.lang_emb_dir / Path(source_path).name

    def _lang_emb_file(self, source_path: str | Path) -> h5py.File:
        lang_path = self._lang_emb_path_for_source(source_path)
        key = str(lang_path)
        handle = self._lang_emb_file_cache.get(key)
        if handle is None:
            if not lang_path.is_file():
                raise FileNotFoundError(
                    f"Missing language sidecar for {source_path}: {lang_path}"
                )
            handle = h5py.File(lang_path, mode="r", swmr=True, libver="latest")
            self._lang_emb_file_cache[key] = handle
        return handle

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
        if self.lang_emb_dir is not None:
            lang_handle = self._lang_emb_file(entry.file_path)
            try:
                lang_dset = lang_handle["data"][entry.demo_key][self.lang_emb_key]
            except KeyError as exc:
                raise KeyError(
                    f"{self._lang_emb_path_for_source(entry.file_path)}:{entry.demo_key} "
                    f"missing {self.lang_emb_key}"
                ) from exc
            lang_emb = np.asarray(lang_dset[...], dtype=np.float32)
            if lang_emb.ndim != 1:
                raise ValueError(
                    f"{self.lang_emb_key} must be a per-demo vector, got {lang_emb.shape}"
                )
            item["lang_emb"] = torch.from_numpy(lang_emb)
        item["hidden_path"] = str(self._hidden_path_for_source(entry.file_path))
        return item


__all__ = ["PixelHiddenSequenceDataset"]
