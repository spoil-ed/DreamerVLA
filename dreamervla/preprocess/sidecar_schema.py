"""Shared schema helpers for preprocessing sidecar HDF5 artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_HIDDEN_KEY = "obs_embedding"
ACTION_HIDDEN_KEY = "action_hidden_states"
ACTOR_SEQUENCE_KEYS = {
    "hidden": "actor_hidden_states",
    "input_ids": "actor_input_ids",
    "attention_mask": "actor_attention_mask",
    "seq_lens": "actor_seq_lens",
}
REQUIRED_DEMO_DATASETS_KEY = "required_demo_datasets"
SIDECAR_SCHEMA_VERSION = 1


def _unique_nonempty(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("sidecar dataset keys must be non-empty strings")
        if value not in result:
            result.append(value)
    return result


def actor_sequence_keys_from_config(config: Mapping[str, Any]) -> dict[str, str]:
    """Return actor sequence dataset keys declared by a sidecar config.

    Older configs did not record ``actor_sequence_keys``.  In that case the
    historical dataset names remain the compatibility default.
    """

    configured = config.get("actor_sequence_keys")
    if configured is None:
        return dict(ACTOR_SEQUENCE_KEYS)
    if not isinstance(configured, Mapping):
        raise ValueError("actor_sequence_keys must be an object")
    keys = dict(ACTOR_SEQUENCE_KEYS)
    for name in keys:
        value = configured.get(name)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"actor_sequence_keys.{name} must be a non-empty string")
            keys[name] = value
    return keys


def required_demo_datasets(
    *,
    hidden_key: str = DEFAULT_HIDDEN_KEY,
    save_action_hidden: bool = False,
    save_actor_sequence: bool = False,
    action_hidden_key: str = ACTION_HIDDEN_KEY,
    actor_sequence_keys: Mapping[str, str] | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Build the per-demo datasets expected in one generated sidecar."""

    keys = [hidden_key]
    if save_action_hidden:
        keys.append(action_hidden_key)
    if save_actor_sequence:
        actor_keys = dict(ACTOR_SEQUENCE_KEYS)
        if actor_sequence_keys is not None:
            actor_keys.update(actor_sequence_keys)
        keys.extend(
            [
                actor_keys["hidden"],
                actor_keys["input_ids"],
                actor_keys["attention_mask"],
                actor_keys["seq_lens"],
            ]
        )
    keys.extend(extra)
    return _unique_nonempty(keys)


def required_demo_datasets_from_config(config: Mapping[str, Any]) -> list[str]:
    """Return required per-demo datasets from a sidecar config.

    New configs declare ``required_demo_datasets`` explicitly.  For compatibility
    with older sidecars, fall back to the original flags and key fields.
    """

    configured = config.get(REQUIRED_DEMO_DATASETS_KEY)
    if configured is not None:
        if not isinstance(configured, list):
            raise ValueError(f"{REQUIRED_DEMO_DATASETS_KEY} must be a list")
        return _unique_nonempty(configured)

    hidden_key = config.get("hidden_key", DEFAULT_HIDDEN_KEY)
    if not isinstance(hidden_key, str) or not hidden_key:
        raise ValueError("hidden_key must be a non-empty string")
    action_hidden_key = config.get("action_hidden_key", ACTION_HIDDEN_KEY)
    if not isinstance(action_hidden_key, str) or not action_hidden_key:
        raise ValueError("action_hidden_key must be a non-empty string")
    return required_demo_datasets(
        hidden_key=hidden_key,
        save_action_hidden=bool(config.get("save_action_hidden", False)),
        save_actor_sequence=bool(config.get("save_actor_sequence", False)),
        action_hidden_key=action_hidden_key,
        actor_sequence_keys=actor_sequence_keys_from_config(config),
    )


def annotate_preprocess_config(
    config: dict[str, Any],
    *,
    required: Sequence[str],
    action_hidden_key: str = ACTION_HIDDEN_KEY,
    actor_sequence_keys: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Add schema metadata to a mutable preprocess config and return it."""

    config["sidecar_schema_version"] = SIDECAR_SCHEMA_VERSION
    config["action_hidden_key"] = action_hidden_key
    config["actor_sequence_keys"] = dict(actor_sequence_keys or ACTOR_SEQUENCE_KEYS)
    config[REQUIRED_DEMO_DATASETS_KEY] = _unique_nonempty(list(required))
    return config
