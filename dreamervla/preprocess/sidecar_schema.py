"""Exact schema for the only supported OpenVLA-OFT observation sidecar."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py

DEFAULT_HIDDEN_KEY = "obs_embedding"
REQUIRED_DEMO_DATASETS_KEY = "required_demo_datasets"
SIDECAR_SCHEMA_VERSION = 1

INPUT_TOKEN_SOURCE = "hidden_token"
INPUT_TOKEN_COUNT = 256
INPUT_TOKEN_DIM = 4096
INPUT_TOKEN_HIDDEN_DIM = INPUT_TOKEN_COUNT * INPUT_TOKEN_DIM
INPUT_TOKEN_SHAPE = (INPUT_TOKEN_COUNT, INPUT_TOKEN_DIM)
INPUT_TOKEN_STORAGE_FORMAT = "tokenized"
INPUT_TOKEN_ACTION_HEAD = "oft_discrete_token"
LEGACY_INPUT_TOKEN_SOURCE = "input_token_embedding"
REMOVED_SIDECAR_FIELDS = (
    "save_action_hidden",
    "action_hidden_key",
    "action_hidden_dim",
    "action_hidden_sequence_dim",
    "action_trigger_token_id",
    "save_hidden_token",
    "hidden_token_key",
    "save_actor_sequence",
    "actor_sequence_keys",
    "actor_hidden_dim",
    "actor_sequence_dim",
)
_LEGACY_ACTOR_SEQUENCE_KEYS = {
    "hidden": "actor_hidden_states",
    "input_ids": "actor_input_ids",
    "attention_mask": "actor_attention_mask",
    "seq_lens": "actor_seq_lens",
}
_LEGACY_SAFE_CONFIG_FIELDS: dict[str, Any] = {
    "save_action_hidden": False,
    "action_hidden_key": "action_hidden_states",
    "action_hidden_dim": 0,
    "action_hidden_sequence_dim": 0,
    "action_trigger_token_id": -1,
    "save_actor_sequence": False,
    "actor_sequence_keys": _LEGACY_ACTOR_SEQUENCE_KEYS,
    "actor_hidden_dim": 0,
    "actor_sequence_dim": 0,
}
_LEGACY_SAFE_HDF5_ATTRS: dict[str, Any] = {
    "save_action_hidden": False,
    "action_hidden_dim": 0,
    "action_hidden_sequence_dim": 0,
    "action_trigger_token_id": -1,
    "save_actor_sequence": False,
    "actor_hidden_dim": 0,
    "actor_sequence_dim": 0,
}
ALLOWED_DEMO_DATASETS = {DEFAULT_HIDDEN_KEY, "lang_emb"}
REFERENCE_DEMO_DATASETS = ("actions", "rewards", "dones", "robot_states", "states")
REFERENCE_OBS_DATASETS = (
    "agentview_rgb",
    "eye_in_hand_rgb",
    "ee_pos",
    "ee_ori",
    "ee_states",
    "gripper_states",
    "joint_states",
)


def _input_token_contract_errors(config: Mapping[str, Any]) -> list[str]:
    expected = {
        "obs_hidden_source": INPUT_TOKEN_SOURCE,
        "action_head_type": INPUT_TOKEN_ACTION_HEAD,
        "hidden_key": DEFAULT_HIDDEN_KEY,
        "token_count": INPUT_TOKEN_COUNT,
        "token_dim": INPUT_TOKEN_DIM,
        "hidden_dim": INPUT_TOKEN_HIDDEN_DIM,
        "hidden_storage_format": INPUT_TOKEN_STORAGE_FORMAT,
        "num_images_in_input": 1,
        "patches_per_image": INPUT_TOKEN_COUNT,
        "history": 1,
        "include_state": False,
        "sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        REQUIRED_DEMO_DATASETS_KEY: [DEFAULT_HIDDEN_KEY],
    }
    errors: list[str] = []
    for key, wanted in expected.items():
        got = config.get(key)
        if got != wanted:
            errors.append(f"{key}={got!r}, expected {wanted!r}")
    shape = config.get("obs_embedding_shape")
    if not isinstance(shape, (list, tuple)) or list(shape) != list(INPUT_TOKEN_SHAPE):
        errors.append(f"obs_embedding_shape={shape!r}, expected {list(INPUT_TOKEN_SHAPE)!r}")
    present_removed = [key for key in REMOVED_SIDECAR_FIELDS if key in config]
    if present_removed:
        errors.append(f"removed sidecar fields are present: {present_removed!r}")
    return errors


def validate_input_token_array_shape(
    shape: Sequence[int],
    *,
    context: str,
) -> None:
    """Reject every external observation shape except ``[...,256,4096]``."""

    trailing = tuple(int(dim) for dim in shape[-2:]) if len(shape) >= 2 else tuple(shape)
    if trailing != INPUT_TOKEN_SHAPE:
        raise ValueError(
            f"{context} must use {INPUT_TOKEN_SOURCE} trailing shape "
            f"{INPUT_TOKEN_SHAPE}, got {tuple(int(dim) for dim in shape)}"
        )


def validate_input_token_preprocess_config(
    config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Validate metadata without aliases, inferred defaults, or conversions."""

    errors = _input_token_contract_errors(config)
    if errors:
        raise ValueError(
            f"{context} does not satisfy the only supported observation contract:\n  - "
            + "\n  - ".join(errors)
        )


def _legacy_value_matches(got: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(got, bool) and got is expected
    return got == expected


def _normalize_known_legacy_config(
    config: Mapping[str, Any],
    *,
    context: str,
) -> tuple[dict[str, Any], bool]:
    """Normalize only the projected-token manifest emitted by the old generator.

    That generator persisted dimensional facts in every HDF5 file but omitted
    them from ``preprocess_config.json``. The HDF5 facts are checked separately
    before any normalized metadata is returned.
    """

    if config.get("obs_hidden_source") != LEGACY_INPUT_TOKEN_SOURCE:
        validate_input_token_preprocess_config(config, context=context)
        return dict(config), False

    unsafe_fields: list[str] = []
    for key in REMOVED_SIDECAR_FIELDS:
        if key not in config:
            continue
        expected = _LEGACY_SAFE_CONFIG_FIELDS.get(key, object())
        if not _legacy_value_matches(config[key], expected):
            unsafe_fields.append(f"{key}={config[key]!r}")
    if unsafe_fields:
        raise ValueError(
            f"{context} enables removed action/actor sidecar payloads: " + ", ".join(unsafe_fields)
        )

    normalized = dict(config)
    for key in REMOVED_SIDECAR_FIELDS:
        normalized.pop(key, None)
    normalized.update(
        {
            "obs_hidden_source": INPUT_TOKEN_SOURCE,
            "token_count": INPUT_TOKEN_COUNT,
            "hidden_dim": INPUT_TOKEN_HIDDEN_DIM,
            "patches_per_image": INPUT_TOKEN_COUNT,
            "obs_embedding_shape": list(INPUT_TOKEN_SHAPE),
        }
    )
    validate_input_token_preprocess_config(normalized, context=context)
    return normalized, True


def _validate_known_legacy_hdf5_attrs(attrs: Any, *, context: str) -> None:
    expected_attrs: dict[str, Any] = {
        "obs_hidden_source": LEGACY_INPUT_TOKEN_SOURCE,
        "hidden_key": DEFAULT_HIDDEN_KEY,
        "hidden_dim": INPUT_TOKEN_HIDDEN_DIM,
        "token_count": INPUT_TOKEN_COUNT,
        "token_dim": INPUT_TOKEN_DIM,
        "hidden_storage_format": INPUT_TOKEN_STORAGE_FORMAT,
        "history": 1,
        "include_state": False,
        "action_head_type": INPUT_TOKEN_ACTION_HEAD,
    }
    errors: list[str] = []
    for key, expected in expected_attrs.items():
        got = attrs.get(key)
        if got != expected:
            errors.append(f"{key}={got!r}, expected {expected!r}")
    for key in REMOVED_SIDECAR_FIELDS:
        if key not in attrs:
            continue
        if key not in _LEGACY_SAFE_HDF5_ATTRS:
            errors.append(f"unsafe legacy attribute {key}={attrs[key]!r}")
            continue
        expected = _LEGACY_SAFE_HDF5_ATTRS[key]
        got = attrs[key]
        matches = bool(got) == expected if isinstance(expected, bool) else got == expected
        if not matches:
            errors.append(f"{key}={got!r}, expected inert value {expected!r}")
    if errors:
        raise ValueError(
            f"{context} is not a safe legacy projected-token sidecar:\n  - " + "\n  - ".join(errors)
        )


def validate_input_token_sidecar_dir(
    hidden_dir: str | Path,
    *,
    expected_filenames: Sequence[str] | None = None,
    reference_dir: str | Path | None = None,
    require_reference_complete: bool = False,
    require_sparse_rewards: bool = False,
) -> dict[str, Any]:
    """Validate metadata and the complete reward/sidecar corpus before replay."""

    directory = Path(hidden_dir).expanduser().resolve()
    config_path = directory / "preprocess_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"input-token sidecar is missing preprocess_config.json: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    config, legacy_manifest = _normalize_known_legacy_config(
        config,
        context=str(config_path),
    )

    paths = sorted(directory.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no input-token HDF5 shards under {directory}")

    actual_names = {path.name for path in paths}
    if expected_filenames is not None:
        expected_names = {str(name) for name in expected_filenames}
        if actual_names != expected_names:
            raise ValueError(
                "input-token sidecar file set mismatch: "
                f"missing={sorted(expected_names - actual_names)!r}, "
                f"extra={sorted(actual_names - expected_names)!r}"
            )

    reference = Path(reference_dir).expanduser().resolve() if reference_dir else None
    if reference is not None:
        reference_paths = sorted(reference.glob("*.hdf5"))
        reference_names = {path.name for path in reference_paths}
        if actual_names != reference_names:
            raise ValueError(
                "reward/hidden file set mismatch: "
                f"missing_hidden={sorted(reference_names - actual_names)!r}, "
                f"extra_hidden={sorted(actual_names - reference_names)!r}"
            )

    for path in paths:
        with h5py.File(path, "r") as handle:
            if legacy_manifest:
                _validate_known_legacy_hdf5_attrs(handle.attrs, context=str(path))
            else:
                removed_attrs = [key for key in REMOVED_SIDECAR_FIELDS if key in handle.attrs]
                if removed_attrs:
                    raise ValueError(
                        f"{path} contains removed sidecar attributes: {removed_attrs!r}"
                    )
            data = handle.get("data")
            if not isinstance(data, h5py.Group) or not data.keys():
                raise ValueError(f"{path} is missing a non-empty data group")
            reference_handle = (
                h5py.File(reference / path.name, "r") if reference is not None else None
            )
            try:
                reference_data = (
                    reference_handle.get("data") if reference_handle is not None else None
                )
                if reference_handle is not None and (
                    not isinstance(reference_data, h5py.Group) or not reference_data.keys()
                ):
                    raise ValueError(f"{reference / path.name} is missing a non-empty data group")
                if isinstance(reference_data, h5py.Group):
                    hidden_keys = {str(key) for key in data.keys()}
                    reference_keys = {str(key) for key in reference_data.keys()}
                    if hidden_keys != reference_keys:
                        raise ValueError(
                            f"{path.name} reward/hidden demo set mismatch: "
                            f"missing_hidden={sorted(reference_keys - hidden_keys)!r}, "
                            f"extra_hidden={sorted(hidden_keys - reference_keys)!r}"
                        )

                file_complete = bool(handle.attrs.get("complete", False))
                for demo_key, demo in data.items():
                    if not isinstance(demo, h5py.Group) or DEFAULT_HIDDEN_KEY not in demo:
                        raise ValueError(f"{path}:data/{demo_key} is missing {DEFAULT_HIDDEN_KEY}")
                    unexpected = sorted(set(demo.keys()) - ALLOWED_DEMO_DATASETS)
                    if unexpected:
                        raise ValueError(
                            f"{path}:data/{demo_key} contains unexpected datasets: {unexpected!r}"
                        )
                    if not file_complete and not bool(demo.attrs.get("complete", False)):
                        raise ValueError(f"{path}:data/{demo_key} is not marked complete")
                    dataset = demo[DEFAULT_HIDDEN_KEY]
                    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
                        raise ValueError(
                            f"{path}:data/{demo_key}/{DEFAULT_HIDDEN_KEY} must be "
                            f"[T,{INPUT_TOKEN_COUNT},{INPUT_TOKEN_DIM}], got "
                            f"{getattr(dataset, 'shape', None)!r}"
                        )
                    validate_input_token_array_shape(
                        dataset.shape,
                        context=f"{path}:data/{demo_key}/{DEFAULT_HIDDEN_KEY}",
                    )
                    hidden_length = int(dataset.shape[0])
                    if hidden_length <= 0:
                        raise ValueError(
                            f"{path}:data/{demo_key}/{DEFAULT_HIDDEN_KEY} has zero frames"
                        )
                    if isinstance(reference_data, h5py.Group):
                        reference_demo = reference_data[demo_key]
                        if not isinstance(reference_demo, h5py.Group):
                            raise ValueError(
                                f"{reference / path.name}:data/{demo_key} must be a group"
                            )
                        if require_reference_complete and not (
                            bool(reference_handle.attrs.get("complete", False))
                            or bool(reference_demo.attrs.get("complete", False))
                        ):
                            raise ValueError(
                                f"{reference / path.name}:data/{demo_key} reward demo "
                                "is not marked complete"
                            )
                        actions = reference_demo.get("actions")
                        if (
                            not isinstance(actions, h5py.Dataset)
                            or actions.ndim != 2
                            or int(actions.shape[-1]) != 7
                        ):
                            raise ValueError(
                                f"{reference / path.name}:data/{demo_key} is missing "
                                "frame-aligned actions [T,7]"
                            )
                        reference_length = int(actions.shape[0])
                        if reference_length <= 0:
                            raise ValueError(
                                f"{reference / path.name}:data/{demo_key}/actions has zero frames"
                            )
                        if hidden_length != reference_length:
                            raise ValueError(
                                f"{path.name}:data/{demo_key} reward/hidden length mismatch: "
                                f"reward={reference_length}, hidden={hidden_length}"
                            )
                        required_fields = list(REFERENCE_DEMO_DATASETS)
                        if require_sparse_rewards:
                            required_fields.append("sparse_rewards")
                        elif "sparse_rewards" in reference_demo:
                            required_fields.append("sparse_rewards")
                        for field in required_fields:
                            value = reference_demo.get(field)
                            if not isinstance(value, h5py.Dataset) or value.ndim < 1:
                                raise ValueError(
                                    f"{reference / path.name}:data/{demo_key} is missing "
                                    f"frame-aligned {field}"
                                )
                            if int(value.shape[0]) != reference_length:
                                raise ValueError(
                                    f"{reference / path.name}:data/{demo_key}/{field} "
                                    "length mismatch: "
                                    f"expected={reference_length}, actual={int(value.shape[0])}"
                                )
                        obs = reference_demo.get("obs")
                        if not isinstance(obs, h5py.Group):
                            raise ValueError(
                                f"{reference / path.name}:data/{demo_key} is missing obs"
                            )
                        for field in REFERENCE_OBS_DATASETS:
                            value = obs.get(field)
                            if not isinstance(value, h5py.Dataset) or value.ndim < 1:
                                raise ValueError(
                                    f"{reference / path.name}:data/{demo_key}/obs is "
                                    f"missing frame-aligned {field}"
                                )
                            if int(value.shape[0]) != reference_length:
                                raise ValueError(
                                    f"{reference / path.name}:data/{demo_key}/obs/{field} "
                                    "length mismatch: "
                                    f"expected={reference_length}, actual={int(value.shape[0])}"
                                )
            finally:
                if reference_handle is not None:
                    reference_handle.close()
    return config


def required_demo_datasets() -> list[str]:
    """Return the fixed per-demo dataset list for the canonical sidecar."""

    return [DEFAULT_HIDDEN_KEY]


def required_demo_datasets_from_config(config: Mapping[str, Any]) -> list[str]:
    """Validate the exact metadata contract and return its fixed dataset key."""

    validate_input_token_preprocess_config(
        config,
        context="preprocess_config.json",
    )
    return required_demo_datasets()
