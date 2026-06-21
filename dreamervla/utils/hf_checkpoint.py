from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

HF_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip a single leading ``module.`` from each key (DDP unwrap)."""
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def resolve_hf_checkpoint_dir(path: str | Path) -> Path:
    """Resolve a Hugging Face checkpoint directory, including one nested level."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if _is_hf_dir(candidate):
        return candidate
    if candidate.is_dir():
        for subdir in sorted(item for item in candidate.iterdir() if item.is_dir()):
            if _is_hf_dir(subdir):
                return subdir.resolve()
    raise FileNotFoundError(
        f"Unable to locate a Hugging Face checkpoint under: {candidate}"
    )


def is_hf_checkpoint(path: str | Path | None) -> bool:
    if path is None:
        return False
    try:
        resolve_hf_checkpoint_dir(path)
    except FileNotFoundError:
        return False
    return True


def load_runner_payload(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Load the legacy DreamerVLA runner payload from a torch checkpoint file."""
    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", False)
    try:
        kwargs.setdefault("mmap", True)
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def load_hf_prefixed_tensors(
    model_dir: str | Path, prefix: str
) -> dict[str, torch.Tensor]:
    """Load tensors under ``prefix`` from HF safetensors or pytorch bin shards."""
    model_dir = resolve_hf_checkpoint_dir(model_dir)
    tensors: dict[str, torch.Tensor] = {}

    safetensor_files = _indexed_weight_files(
        model_dir, "model.safetensors.index.json", prefix
    )
    if safetensor_files is None:
        safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file

        for shard in safetensor_files:
            for key, value in load_file(str(shard)).items():
                if key.startswith(prefix):
                    tensors[key[len(prefix) :]] = value.to(dtype=torch.float32)
        if tensors:
            return tensors

    bin_files = _indexed_weight_files(model_dir, "pytorch_model.bin.index.json", prefix)
    if bin_files is None:
        bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
    for shard in bin_files:
        state = torch.load(shard, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if (
                isinstance(key, str)
                and key.startswith(prefix)
                and isinstance(value, torch.Tensor)
            ):
                tensors[key[len(prefix) :]] = value.to(dtype=torch.float32)
    return tensors


def _is_hf_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and any(
        (path / name).exists() for name in HF_WEIGHT_NAMES
    )


def _indexed_weight_files(
    model_dir: Path, index_name: str, prefix: str
) -> list[Path] | None:
    index_path = model_dir / index_name
    if not index_path.is_file():
        return None
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map", {})
    return sorted(
        {
            (model_dir / shard).resolve()
            for key, shard in weight_map.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
    )
