"""Small runtime metric accumulators shared by runners."""

from __future__ import annotations

from collections import deque


class SuccessTracker:
    """Windowed episode success rate with best-so-far and print delta."""

    def __init__(self, window: int) -> None:
        self._buf: deque[float] = deque(maxlen=max(1, int(window)))
        self._best: float = 0.0
        self._last_printed: float | None = None

    def update(self, success: bool) -> None:
        self._buf.append(1.0 if success else 0.0)
        if len(self._buf) == self._buf.maxlen:
            self._best = max(self._best, self.rate())

    def rate(self) -> float:
        return (sum(self._buf) / len(self._buf)) if self._buf else 0.0

    @property
    def best(self) -> float:
        return self._best

    def delta(self) -> float:
        return 0.0 if self._last_printed is None else self.rate() - self._last_printed

    def mark_printed(self) -> None:
        self._last_printed = self.rate()

    def __len__(self) -> int:
        return len(self._buf)


__all__ = ["SuccessTracker"]
