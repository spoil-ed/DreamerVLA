"""Small async-work wrapper used by scheduler collective operations."""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any


class AsyncResult:
    """A minimal wait/done interface mirroring Ray worker group results."""

    def __init__(self, future: Future[Any]) -> None:
        self._future = future

    @classmethod
    def completed(cls, result: Any = None) -> AsyncResult:
        future: Future[Any] = Future()
        future.set_result(result)
        return cls(future)

    def wait(self) -> Any:
        return self._future.result()

    def done(self) -> bool:
        return self._future.done()
