"""Stateful, wall-time-throttled progress reporter.

One uniform progress line for every pipeline loop. No tqdm: prints plain lines
via an injected sink so output is clean in log files, nohup, and Ray worker
logs. clock/sink are injectable for deterministic tests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

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
        status: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sink: Callable[[str], None] = _flush_print,
    ) -> None:
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.min_interval_s = float(min_interval_s)
        self.unit = unit
        self.status = status
        self._clock = clock
        self._sink = sink
        self._current = 0
        self._start_t = clock()
        self._last_print_t: float | None = None

    def update(self, n: int = 1) -> None:
        self.set(self._current + n)

    def set_status(self, status: str | None) -> None:
        self.status = status

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
                status=self.status,
            )
        )
        self._last_print_t = now

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AggregateProgress:
    """Cross-process collect progress: ONE aggregated bar across independent workers.

    The multi-rank (torchrun) collector has no DDP process group, so each rank persists
    its ``{done, total, finished}`` to a shared ``progress_dir`` (atomic temp+rename) and
    rank 0 renders a single ``global_done / global_total`` bar by summing every rank's file.
    Non-zero ranks only persist (silent), so 6 GPUs show up as one moving total instead of
    a per-episode flood or a rank-0-only view.

    Falls back to a plain per-rank bar (rank 0 only) when ``progress_dir`` is absent or
    ``world_size <= 1`` — e.g. a single-process collect, or a direct torchrun run whose
    ranks do not share an out_dir. Same ``set`` / ``update`` / ``close`` / context-manager
    surface as :class:`ProgressReporter`, so it is a drop-in.

    No live drain after rank 0 finishes its own slice: the launcher prints the exact final
    aggregate once every rank exits, so a brief tail gap before that is acceptable.
    """

    def __init__(
        self,
        total: int | None,
        desc: str,
        *,
        rank: int,
        world_size: int,
        progress_dir: str | Path | None,
        unit: str = "it",
        min_interval_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sink: Callable[[str], None] = _flush_print,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._dir = Path(progress_dir) if progress_dir else None
        self._shared = self._dir is not None and self.world_size > 1
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._my_total = 0 if total is None else int(total)
        self._done = 0
        # Only rank 0 prints; the reporter starts on this rank's own total and is
        # retargeted to the global total once siblings report in.
        self._reporter = ProgressReporter(
            total, desc, enabled=(self.rank == 0), unit=unit,
            min_interval_s=min_interval_s, clock=clock, sink=sink,
        )

    def _file(self, rank: int) -> Path:
        return self._dir / f"rank_{rank}.json"

    def _persist(self, *, finished: bool) -> None:
        if self._dir is None:
            return
        payload = json.dumps({"done": self._done, "total": self._my_total, "finished": finished})
        tmp = self._dir / f".rank_{self.rank}.tmp"
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self._file(self.rank))

    def _global(self) -> tuple[int, int]:
        """Sum done/total across every rank's file; missing/half-written files count as 0."""
        if not self._shared:
            return self._done, self._my_total
        done = total = 0
        for r in range(self.world_size):
            try:
                d = json.loads(self._file(r).read_text(encoding="utf-8"))
                done += int(d["done"])
                total += int(d["total"])
            except (OSError, ValueError, KeyError):
                continue
        return done, total

    def _render(self) -> None:
        done, total = self._global()
        self._reporter.total = total
        self._reporter.set(done)

    def set(self, current: int) -> None:
        self._done = int(current)
        self._persist(finished=False)
        if self.rank == 0:
            self._render()

    def update(self, n: int = 1) -> None:
        self.set(self._done + n)

    def close(self) -> None:
        self._persist(finished=True)
        if self.rank == 0:
            self._render()
            self._reporter.close()

    def __enter__(self) -> AggregateProgress:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
