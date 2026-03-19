from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class PathConfig:
    rynnvla_root: str = "/home/yuxinglei/workspace/2026nips/RynnVLA-001"
    dreamerv3_root: str = "/home/yuxinglei/workspace/2026nips/dreamerv3-main"
    output_dir: str = "./outputs"


@dataclass
class ModelConfig:
    image_dim: int = 48
    proprio_dim: int = 8
    text_dim: int = 16
    semantic_dim: int = 256
    bottleneck_dim: int = 32
    rssm_hidden_dim: int = 128
    action_dim: int = 6
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    reward_hidden_dim: int = 128


@dataclass
class DataConfig:
    dataset_type: str = "synthetic"
    dataset_path: str = ""
    synthetic_state_dim: int = 12
    train_num_sequences: int = 128
    val_num_sequences: int = 32
    rollout_num_sequences: int = 16
    sequence_length: int = 15
    train_batch_size: int = 8
    val_batch_size: int = 8
    rollout_batch_size: int = 8
    seed: int = 7


@dataclass
class AlgorithmConfig:
    world_model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    dynamics_kl_scale: float = 1.0
    bottleneck_kl_scale: float = 1e-3
    continue_loss_scale: float = 1.0
    gamma: float = 0.99
    lambda_: float = 0.95
    imagination_horizon: int = 8
    grad_clip_norm: float = 100.0


@dataclass
class TrainerConfig:
    project_name: str = "dreamer-vla"
    experiment_name: str = "minimal-dreamer"
    total_epochs: int = 3
    validate_every: int = 1
    log_every: int = 1
    device: str = "cpu"
    debug: bool = True


@dataclass
class DreamerVLAConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return {}
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            raise ValueError(f"List syntax is not supported by the minimal YAML loader: {path}")
        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"Invalid YAML line in {path}: {raw_line}")
        while indent <= stack[-1][0]:
            stack.pop()
        target = stack[-1][1]
        if not value.strip():
            child: dict[str, Any] = {}
            target[key] = child
            stack.append((indent, child))
        else:
            target[key] = _parse_scalar(value)
    return root


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_dict(instance: Any, data: dict[str, Any]) -> Any:
    for field_info in fields(instance):
        if field_info.name not in data:
            continue
        current_value = getattr(instance, field_info.name)
        incoming_value = data[field_info.name]
        if is_dataclass(current_value) and isinstance(incoming_value, dict):
            _apply_dict(current_value, incoming_value)
        else:
            setattr(instance, field_info.name, incoming_value)
    return instance


def load_config(config_path: str | Path | None = None) -> DreamerVLAConfig:
    config = DreamerVLAConfig()
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    merged: dict[str, Any] = {}

    for default_name in ("base.yaml", "ppo_trainer.yaml"):
        default_path = config_dir / default_name
        if default_path.exists():
            merged = _deep_merge(merged, _load_simple_yaml(default_path))

    if config_path is not None:
        path = Path(config_path)
        if path.suffix == ".json":
            merged = _deep_merge(merged, json.loads(path.read_text(encoding="utf-8")))
        else:
            merged = _deep_merge(merged, _load_simple_yaml(path))

    return _apply_dict(config, merged)
