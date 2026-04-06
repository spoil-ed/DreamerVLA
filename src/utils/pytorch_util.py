from __future__ import annotations

from typing import Any, Callable

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
