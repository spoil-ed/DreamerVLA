"""Stateful, wall-time-throttled progress reporter.

One uniform progress line for every pipeline loop. No tqdm: prints plain lines
via an injected sink so output is clean in log files, nohup, and Ray worker
logs. clock/sink are injectable for deterministic tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from dreamervla.utils.console import format_progress_line


def _flush_print(line: str) -> None:
    # flush=True so progress surfaces under non-TTY block-buffered stdout
    # (nohup / Ray worker logs), matching console_banner / console_metrics.
    print(line, flush=True)


class ProgressReporter:
    def __init__(
        self,
        total: int | None,
        desc: str,
        *,
        enabled: bool = True,
        min_interval_s: float = 5.0,
        unit: str = "it",
        clock: Callable[[], float] = time.monotonic,
        sink: Callable[[str], None] = _flush_print,
    ) -> None:
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.min_interval_s = float(min_interval_s)
        self.unit = unit
        self._clock = clock
        self._sink = sink
        self._current = 0
        self._start_t = clock()
        self._last_print_t: float | None = None

    def update(self, n: int = 1) -> None:
        self.set(self._current + n)

    def set(self, current: int) -> None:
        self._current = int(current)
        if not self.enabled:
            return
        now = self._clock()
        if self._last_print_t is None or (now - self._last_print_t) >= self.min_interval_s:
            self._emit(now)

    def close(self) -> None:
        if not self.enabled:
            return
        self._emit(self._clock())

    def _emit(self, now: float) -> None:
        elapsed = max(1e-9, now - self._start_t)
        rate = self._current / elapsed
        eta = None
        if self.total and self.total > 0 and rate > 0:
            eta = max(0.0, (self.total - self._current) / rate)
        self._sink(
            format_progress_line(
                self.desc,
                self._current,
                self.total,
                elapsed_s=elapsed,
                eta_s=eta,
                rate=rate,
                unit=self.unit,
            )
        )
        self._last_print_t = now

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
