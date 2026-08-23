# Copyright 2025 The DreamerVLA Authors.
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

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    implement_version = cfg.get("implement_version", "dreamervla")
    if implement_version == "dreamervla":
        from dreamervla.models.embodiment.openvla_oft.dreamervla import get_model
    elif implement_version == "official":
        from dreamervla.models.embodiment.openvla_oft.official import get_model
    else:
        raise NotImplementedError(
            f"Unsupported model implementation version: '{implement_version}'. "
            f"Currently supported versions: ['dreamervla', 'official']. "
            f"Please check ...model.version or implement the corresponding model loader."
        )

    return get_model(cfg, torch_dtype)
