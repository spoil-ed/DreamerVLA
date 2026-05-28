from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


class PolicyChunkActionQueue:
    """Reuse sampled action chunks across environment steps."""

    def __init__(self, collect_chunk_steps: int = 0) -> None:
        self.collect_chunk_steps = int(collect_chunk_steps)
        self._pending: deque[np.ndarray] = deque()

    def clear(self) -> None:
        self._pending.clear()

    def next_action(
        self,
        policy: Callable[[dict[str, Any]], tuple[torch.Tensor, Any, dict[str, Any]]],
        *,
        hidden: torch.Tensor,
        deterministic: bool,
    ) -> np.ndarray:
        if not self._pending:
            action_chunk, _log_prob, _extra = policy({
                "mode": "sample",
                "hidden": hidden,
                "deterministic": bool(deterministic),
                "return_chunk": True,
            })
            chunk_np = (
                action_chunk.reshape(-1, action_chunk.shape[-1])
                .detach()
                .cpu()
                .float()
                .numpy()
            )
            collect_steps = self.collect_chunk_steps
            if collect_steps <= 0:
                collect_steps = int(chunk_np.shape[0])
            collect_steps = max(1, min(int(collect_steps), int(chunk_np.shape[0])))
            self._pending.extend(
                np.asarray(action[:7], dtype=np.float32).copy()
                for action in chunk_np[:collect_steps]
            )
        return self._pending.popleft()
