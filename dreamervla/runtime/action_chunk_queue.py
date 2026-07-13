"""Open-loop action chunk queue shared by cotrain rollout paths."""

from __future__ import annotations

from collections import deque

import numpy as np


class ActionChunkQueue:
    """Queue exactly ``action_steps`` low-level actions from one actor chunk."""

    def __init__(self, *, action_dim: int = 7, action_steps: int | None = None) -> None:
        self.action_dim = int(action_dim)
        self.action_steps = None if action_steps is None else max(1, int(action_steps))
        self._pending: deque[np.ndarray] = deque()

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def refill(self, action_chunk: np.ndarray) -> None:
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.ndim != 2:
            raise ValueError(f"action chunk must be [K,A], got shape {tuple(chunk.shape)}")
        if chunk.shape[1] < self.action_dim:
            raise ValueError(
                f"action chunk dim {chunk.shape[1]} < action_dim={self.action_dim}"
            )
        steps = self.action_steps if self.action_steps is not None else int(chunk.shape[0])
        if chunk.shape[0] < steps:
            raise ValueError(
                f"policy returned {chunk.shape[0]} actions, need action_steps={steps}"
            )
        self._pending.clear()
        for row in chunk[:steps, : self.action_dim]:
            self._pending.append(np.asarray(row, dtype=np.float32).copy())

    def pop(self) -> np.ndarray:
        if not self._pending:
            raise IndexError("empty action chunk queue")
        return self._pending.popleft()

    def clear(self) -> None:
        self._pending.clear()


__all__ = ["ActionChunkQueue"]
