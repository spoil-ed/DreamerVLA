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

"""Shared norm-stats helper for embodiment model loaders."""

import json
import os


def merge_norm_stats_into_config(config, model_path) -> None:
    """Merge ``dataset_statistics.json`` (if present) into ``config.norm_stats``.

    Mirrors the original inline blocks exactly: when the sidecar file exists, it
    is loaded and merged on top of the config's existing ``norm_stats`` (empty
    dict if the attribute is missing). When the file is absent the config is left
    untouched.
    """
    dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path) as f:
            new_norm_stats = json.load(f)
            norm_stats = getattr(config, "norm_stats", {})
            norm_stats.update(new_norm_stats)
            config.norm_stats = norm_stats
