"""Standalone adapter for OpenPI's official π0.5 LeRobot SFT loader."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

from dreamervla.utils.openpi_imports import ensure_openpi_on_path

OFFICIAL_PI05_LIBERO_REPO = "physical-intelligence/libero"


def resolve_lerobot_source(value: Any) -> str:
    """Resolve one configured LeRobot repo id or complete local dataset root."""

    if value is None:
        raise ValueError("π0.5 SFT requires data.train_data_paths")
    if isinstance(value, (str, Path)):
        source = str(value).strip()
    elif hasattr(value, "get"):
        source = str(value.get("dataset_path") or value.get("data_path") or "").strip()
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("π0.5 SFT data.train_data_paths is empty")
        return resolve_lerobot_source(value[0])
    else:
        source = str(value).strip()
    if not source:
        raise ValueError("π0.5 SFT data.train_data_paths is empty")

    path = Path(source).expanduser()
    if path.exists():
        info = path / "meta" / "info.json"
        if not info.is_file():
            raise ValueError(
                f"local LeRobot dataset is incomplete (missing {info}); "
                "point data.train_data_paths at the dataset root"
            )
        return str(path.resolve())
    if path.is_absolute() or source.startswith((".", "~")):
        raise FileNotFoundError(f"local LeRobot dataset does not exist: {path}")
    return source


def build_pi05_sft_dataloader(
    *,
    model_path: str,
    data_paths: Any,
    config_name: str,
    micro_batch_size: int,
    world_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = True,
) -> Any:
    """Build the same official OpenPI loader used by RLinf, without importing RLinf."""

    if int(micro_batch_size) <= 0 or int(world_size) <= 0:
        raise ValueError("micro_batch_size and world_size must be positive")
    source = resolve_lerobot_source(data_paths)
    source_path = Path(source).expanduser()
    if source_path.exists():
        # LeRobot 0.3 does not pass an explicit ``root`` through OpenPI's loader.
        # Point its cache root at this complete local dataset without copying it.
        expected_suffix = Path(OFFICIAL_PI05_LIBERO_REPO)
        try:
            relative = source_path.resolve().relative_to(
                source_path.resolve().parents[len(expected_suffix.parts) - 1]
            )
        except (IndexError, ValueError):
            relative = None
        if relative != expected_suffix:
            raise ValueError(
                "local π0.5 LIBERO data must end in "
                f"{OFFICIAL_PI05_LIBERO_REPO}; got {source_path.resolve()}"
            )
        os.environ["HF_LEROBOT_HOME"] = str(
            source_path.resolve().parents[len(expected_suffix.parts) - 1]
        )
        source = OFFICIAL_PI05_LIBERO_REPO
    checkpoint = Path(model_path).expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"π0.5 checkpoint is missing model.safetensors: {checkpoint}")

    ensure_openpi_on_path()
    from openpi.training import config as training_config
    from openpi.training import data_loader as openpi_data_loader

    config = training_config.get_config(str(config_name))
    original_repo_id = config.data.repo_id
    data_config = dataclasses.replace(
        config.data,
        repo_id=source,
        assets=dataclasses.replace(
            config.data.assets,
            # Keep normalization tied to the base checkpoint, even when the
            # samples come from a local cache or a different repo id.
            assets_dir=str(checkpoint),
            asset_id=original_repo_id,
        ),
    )
    config = dataclasses.replace(
        config,
        data=data_config,
        batch_size=int(micro_batch_size) * int(world_size),
        num_workers=int(num_workers),
        seed=int(seed),
        pytorch_weight_path=str(checkpoint),
    )
    return openpi_data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=bool(shuffle),
    )


def openpi_torch_loader(data_loader: Any) -> Any:
    """Return OpenPI's inner torch DataLoader for length/resume bookkeeping."""

    wrapper = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(wrapper, "_data_loader", None) or getattr(
        wrapper, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError("official OpenPI loader does not expose its torch DataLoader")
    return torch_loader


def configured_download_endpoint() -> str:
    """Report the explicit endpoint; downloads never invent a proxy route."""

    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")


__all__ = [
    "OFFICIAL_PI05_LIBERO_REPO",
    "build_pi05_sft_dataloader",
    "configured_download_endpoint",
    "openpi_torch_loader",
    "resolve_lerobot_source",
]
