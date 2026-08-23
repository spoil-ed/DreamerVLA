# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf's OpenPI ``pi05_libero`` training configuration.

This is the π0.5/LIBERO slice of
``RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py``.  DreamerVLA
keeps the source model, normalization assets, and LeRobot samples as separate
paths, so the small builder accepts all three explicitly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from dreamervla.utils.openpi_imports import ensure_openpi_on_path

PI05_LIBERO_CONFIG_NAME = "pi05_libero"
PI05_LIBERO_REPO_ID = "physical-intelligence/libero"


def get_pi05_libero_config(
    *,
    model_path: str,
    assets_path: str,
    repo_id: str = PI05_LIBERO_REPO_ID,
    batch_size: int = 256,
    action_horizon: int = 10,
    num_workers: int | None = None,
    seed: int | None = None,
    learning_rate: float = 5e-5,
    lr_warmup_steps: int = 10_000,
    total_training_steps: int = 30_000,
    data_kwargs: Any | None = None,
) -> Any:
    """Return RLinf's official OpenPI π0.5 LIBERO ``TrainConfig``."""

    ensure_openpi_on_path()
    import openpi.models.pi0_config as pi0_config
    import openpi.training.optimizer as optimizer
    import openpi.training.weight_loaders as weight_loaders
    from openpi.training.config import AssetsConfig, DataConfig, TrainConfig

    from dreamervla.models.embodiment.pi05.libero_dataconfig import (
        LeRobotLiberoDataConfig,
    )

    if data_kwargs:
        raise ValueError("pi05_libero does not accept openpi_data overrides yet")

    config = TrainConfig(
        name=PI05_LIBERO_CONFIG_NAME,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=int(action_horizon),
            discrete_state_input=False,
        ),
        data=LeRobotLiberoDataConfig(
            repo_id=str(repo_id),
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir=str(Path(assets_path).expanduser())),
            extra_delta_transform=False,
        ),
        batch_size=int(batch_size),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=int(lr_warmup_steps),
            peak_lr=float(learning_rate),
            decay_steps=int(total_training_steps),
            decay_lr=float(learning_rate),
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("checkpoints/jax/pi05_base"),
        pytorch_weight_path=str(Path(model_path).expanduser()),
        num_train_steps=int(total_training_steps),
    )
    replacements: dict[str, Any] = {}
    if num_workers is not None:
        replacements["num_workers"] = int(num_workers)
    if seed is not None:
        replacements["seed"] = int(seed)
    return dataclasses.replace(config, **replacements) if replacements else config


__all__ = [
    "PI05_LIBERO_CONFIG_NAME",
    "PI05_LIBERO_REPO_ID",
    "get_pi05_libero_config",
]
