# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenPI model transforms, batch adapters and optional legacy v2 loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreamervla.models.embodiment.pi05.openpi_config import (
    PI05_LIBERO_CONFIG_NAME,
    PI05_LIBERO_REPO_ID,
    get_pi05_libero_config,
)
from dreamervla.utils.integrations.openpi_imports import (
    configure_openpi_jax_runtime,
    ensure_openpi_on_path,
)

OFFICIAL_PI05_LIBERO_REPO = PI05_LIBERO_REPO_ID


OFFICIAL_PI05_LIBERO_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


def configure_openpi_pytorch_runtime() -> None:
    """Apply the shared OpenPI/JAX isolation contract for SFT callers."""

    configure_openpi_jax_runtime()


@dataclass(frozen=True)
class OpenPISFTDataLoaderBundle:
    """Official loader plus its resolved OpenPI data config and source."""

    data_loader: Any
    data_config: Any
    source: str


def resolve_lerobot_source(value: Any) -> str:
    """Resolve one configured LeRobot repo id or complete local dataset root."""

    if value is None:
        raise ValueError("OpenPI SFT requires a local dataset path or LeRobot repo id")
    if isinstance(value, (str, Path)):
        source = str(value).strip()
    elif hasattr(value, "get"):
        source = str(value.get("dataset_path") or value.get("data_path") or "").strip()
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("OpenPI SFT dataset paths are empty")
        return resolve_lerobot_source(value[0])
    else:
        source = str(value).strip()
    if not source:
        raise ValueError("OpenPI SFT dataset paths are empty")

    path = Path(source).expanduser()
    if path.exists():
        info = path / "meta" / "info.json"
        if not info.is_file():
            raise ValueError(
                f"local LeRobot dataset is incomplete (missing {info}); "
                "point the config at the dataset root"
            )
        return str(path.resolve())
    if path.is_absolute() or source.startswith((".", "~")):
        raise FileNotFoundError(f"local LeRobot dataset does not exist: {path}")
    return source


def _repo_id_for_openpi(source: str, expected_repo_id: str) -> str:
    """Expose a local LeRobot 0.3 snapshot through its canonical cache root."""

    source_path = Path(source).expanduser()
    if not source_path.exists():
        return source
    expected_suffix = Path(expected_repo_id)
    resolved = source_path.resolve()
    try:
        cache_root = resolved.parents[len(expected_suffix.parts) - 1]
        relative = resolved.relative_to(cache_root)
    except (IndexError, ValueError):
        relative = None
    if relative != expected_suffix:
        raise ValueError(f"local LeRobot dataset must end in {expected_repo_id}; got {resolved}")
    os.environ["HF_LEROBOT_HOME"] = str(cache_root)
    return expected_repo_id


def build_official_openpi_sft_dataloader(
    *,
    model_path: str,
    assets_path: str,
    data_paths: Any,
    config_name: str,
    micro_batch_size: int,
    action_horizon: int,
    world_size: int,
    rank: int,
    num_workers: int,
    seed: int,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the same official OpenPI loader used by RLinf for LeRobot data."""

    del rank
    configure_openpi_pytorch_runtime()
    if str(config_name) != PI05_LIBERO_CONFIG_NAME:
        raise ValueError(
            f"this migrated route only supports {PI05_LIBERO_CONFIG_NAME}; got {config_name!r}"
        )
    if int(micro_batch_size) <= 0 or int(world_size) <= 0:
        raise ValueError("micro_batch_size and world_size must be positive")
    source = resolve_lerobot_source(data_paths)
    repo_id = _repo_id_for_openpi(source, OFFICIAL_PI05_LIBERO_REPO)

    checkpoint = Path(model_path).expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"π0.5 checkpoint is missing model.safetensors: {checkpoint}")
    assets = Path(assets_path).expanduser().resolve()
    norm_stats = assets / OFFICIAL_PI05_LIBERO_REPO / "norm_stats.json"
    if not norm_stats.is_file():
        raise FileNotFoundError(f"π0.5 LIBERO assets are missing {norm_stats}")

    ensure_openpi_on_path()
    import openpi.training.data_loader as openpi_data_loader

    config = get_pi05_libero_config(
        model_path=str(checkpoint),
        assets_path=str(assets),
        repo_id=repo_id,
        batch_size=int(micro_batch_size) * int(world_size),
        action_horizon=int(action_horizon),
        num_workers=int(num_workers),
        seed=int(seed),
    )
    data_loader = openpi_data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=not eval_dataset,
    )
    return data_loader, data_loader.data_config()


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch DataLoader length used by OpenPI."""

    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()"
        )
    return len(torch_loader)


def openpi_torch_loader(data_loader: Any) -> Any:
    """Return OpenPI's inner torch DataLoader for resume bookkeeping."""

    get_official_openpi_sft_num_batches(data_loader)
    wrapper = data_loader._data_loader
    return getattr(wrapper, "_data_loader", None) or wrapper.torch_loader


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    return getattr(data_loader, "_data_loader", None) is not None


@dataclass
class LeRobotLIBERODataLoaderFactory:
    """Hydra bridge around RLinf's official OpenPI/LeRobot loader."""

    source: str
    model_path: str
    assets_path: str
    repo_id: str = OFFICIAL_PI05_LIBERO_REPO
    revision: str = OFFICIAL_PI05_LIBERO_REVISION
    config_name: str = PI05_LIBERO_CONFIG_NAME
    batch_size: int = 4
    action_horizon: int = 10
    num_workers: int = 2
    seed: int = 0
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.repo_id != OFFICIAL_PI05_LIBERO_REPO:
            raise ValueError(
                f"π0.5 LIBERO SFT requires {OFFICIAL_PI05_LIBERO_REPO}; got {self.repo_id}"
            )
        if self.revision != OFFICIAL_PI05_LIBERO_REVISION:
            raise ValueError("π0.5 LIBERO dataset revision must match the pinned official snapshot")
        self.source = resolve_lerobot_source(self.source)

    def build(
        self,
        *,
        world_size: int,
        rank: int,
        eval_dataset: bool = False,
    ) -> OpenPISFTDataLoaderBundle:
        data_loader, data_config = build_official_openpi_sft_dataloader(
            model_path=self.model_path,
            assets_path=self.assets_path,
            data_paths=self.source,
            config_name=self.config_name,
            micro_batch_size=self.batch_size,
            action_horizon=self.action_horizon,
            world_size=world_size,
            rank=rank,
            num_workers=self.num_workers,
            seed=self.seed,
            eval_dataset=eval_dataset or not self.shuffle,
        )
        return OpenPISFTDataLoaderBundle(data_loader, data_config, self.source)


def configured_download_endpoint() -> str:
    """Report the configured endpoint without inventing a proxy route."""

    return os.environ.get("HF_ENDPOINT", "https://huggingface.co")


def build_local_sft_loader(
    *,
    dataset: Any,
    repo_id: str,
    model_path: str,
    assets_path: str,
    normalization_asset_id: str,
    config_name: str,
    batch_size: int,
    action_horizon: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    world_size: int,
    rank: int,
) -> Any:
    """Apply checkpoint transforms without invoking an external dataset reader.

    The existing OpenPI batch wrapper preserves the Observation/actions and
    inner DataLoader contracts used by SFT, decoder training and resume.
    Source dataset identity and checkpoint normalization asset ID are separate.
    """
    if config_name != PI05_LIBERO_CONFIG_NAME:
        raise ValueError(f"Expected {PI05_LIBERO_CONFIG_NAME}, got {config_name!r}")
    if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size or num_workers < 0:
        raise ValueError("Invalid batch size, world size, rank or worker count")
    checkpoint = Path(model_path).expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"π0.5 checkpoint is missing model.safetensors: {checkpoint}")
    assets = Path(assets_path).expanduser().resolve()
    stats_path = assets / normalization_asset_id / "norm_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"π0.5 normalization assets are missing: {stats_path}")

    ensure_openpi_on_path()
    from openpi.training.data_loader import DataLoaderImpl, TorchDataLoader, transform_dataset
    from torch.utils.data.distributed import DistributedSampler

    config = get_pi05_libero_config(
        model_path=str(checkpoint),
        assets_path=str(assets),
        repo_id=repo_id,
        normalization_asset_id=normalization_asset_id,
        batch_size=batch_size * world_size,
        action_horizon=action_horizon,
        num_workers=num_workers,
        seed=seed,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    transformed = transform_dataset(dataset, data_config)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            transformed,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        if len(sampler) < batch_size:
            raise ValueError("Selected dataset has fewer than one local batch per rank")
    inner = TorchDataLoader(
        transformed,
        local_batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        seed=seed,
        framework="pytorch",
    )
    loader = DataLoaderImpl(data_config, inner)
    return OpenPISFTDataLoaderBundle(loader, data_config, str(dataset.root))


__all__ = [
    "OFFICIAL_PI05_LIBERO_REPO",
    "OFFICIAL_PI05_LIBERO_REVISION",
    "LeRobotLIBERODataLoaderFactory",
    "OpenPISFTDataLoaderBundle",
    "build_local_sft_loader",
    "build_official_openpi_sft_dataloader",
    "configure_openpi_pytorch_runtime",
    "configured_download_endpoint",
    "get_official_openpi_sft_num_batches",
    "is_official_openpi_sft_dataloader",
    "openpi_torch_loader",
    "resolve_lerobot_source",
]
