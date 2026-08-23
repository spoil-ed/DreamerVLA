"""Minimal dynamic scheduler for overlapping component work."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScheduledWork:
    """Handle for one scheduled component task."""

    component: str
    future: Future[Any]

    def done(self) -> bool:
        return bool(self.future.done())

    def wait(self) -> Any:
        return self.future.result()


class ComponentScheduler:
    """Small executor-backed scheduler for component-level overlap."""

    def __init__(self, max_workers: int = 4) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._pending: list[ScheduledWork] = []

    def submit(
        self,
        component: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> ScheduledWork:
        work = ScheduledWork(str(component), self.executor.submit(fn, *args, **kwargs))
        self._pending.append(work)
        return work

    def drain_ready(self) -> dict[str, list[Any]]:
        ready: dict[str, list[Any]] = {}
        remaining: list[ScheduledWork] = []
        for work in self._pending:
            if work.done():
                ready.setdefault(work.component, []).append(work.wait())
            else:
                remaining.append(work)
        self._pending = remaining
        return ready

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


__all__ = ["ComponentScheduler", "ScheduledWork"]
