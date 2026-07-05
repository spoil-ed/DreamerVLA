from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def dict_apply(
    x: dict[str, Any],
    func: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
