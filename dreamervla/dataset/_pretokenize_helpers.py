"""Pure path / IO helpers extracted from ``pretokenize_dataset`` (P3 split).

These are stateless functions (no dataset ``self`` coupling); the dataset classes
expose them as static-method delegators so call sites and subclasses are unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_FRAME_RE = re.compile(r"image_(\d+)\.png$")


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def load_action_sequence(action: list[Any]) -> torch.Tensor:
    values: list[np.ndarray] = []
    for entry in action:
        if isinstance(entry, str):
            path = Path(entry).expanduser()
            if path.is_file():
                values.append(np.asarray(np.load(path), dtype=np.float32))
            continue
        values.append(np.asarray(entry, dtype=np.float32))
    if not values:
        return torch.zeros((0, 0), dtype=torch.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return torch.tensor(array, dtype=torch.float32)


def select_current_third_view(images: Any) -> str:
    paths = [str(path) for path in (images or [])]
    third = [path for path in paths if "/imgs_third_view/" in path]
    if third:
        return third[-1]
    return paths[-1] if paths else ""


def parse_image_path(path: str) -> tuple[str, str, int] | None:
    parts = Path(path).parts
    if "imgs_third_view" not in parts:
        return None
    view_idx = parts.index("imgs_third_view")
    if view_idx < 2:
        return None
    match = _FRAME_RE.match(parts[-1])
    if match is None:
        return None
    task_name = parts[view_idx - 2]
    trj_name = parts[view_idx - 1]
    trajectory_key = f"{task_name}/{trj_name}"
    return task_name, trajectory_key, int(match.group(1))


def sibling_step_path(image_path: str, dirname: str, prefix: str, suffix: str) -> str:
    path = Path(image_path)
    match = _FRAME_RE.match(path.name)
    if match is None:
        return ""
    frame_index = int(match.group(1))
    trj_dir = path.parent.parent
    return str(trj_dir / dirname / f"{prefix}_{frame_index}{suffix}")


def load_step_action(path: str, action_dim: int) -> torch.Tensor:
    if path and Path(path).is_file():
        action = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
        return torch.tensor(action, dtype=torch.float32)
    return torch.zeros(action_dim, dtype=torch.float32)


def load_step_reward(path: str) -> float:
    if path and Path(path).is_file():
        return float(np.asarray(np.load(path), dtype=np.float32).reshape(-1)[0])
    return 0.0
