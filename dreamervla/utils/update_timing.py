"""Low-overhead stage timing for one optimizer update.

CPU stages use a wall clock. CUDA stages use events and synchronize once per
profiled update, so profiling does not serialize every kernel boundary.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch


class GradientUpdateTimer:
    """Attribute data and device time for an optionally profiled update."""

    def __init__(self, device: torch.device, *, enabled: bool) -> None:
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self._use_cuda_events = bool(
            self.enabled
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._wall_timings: dict[str, float] = {}
        self._cuda_events: dict[
            str, tuple[torch.cuda.Event, torch.cuda.Event]
        ] = {}
        self._device_synchronized = False

    @contextmanager
    def wall_stage(self, name: str) -> Iterator[None]:
        """Measure a host-side stage with ``perf_counter`` when enabled."""

        if not self.enabled:
            yield
            return
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self._wall_timings[str(name)] = time.perf_counter() - started_at

    @contextmanager
    def device_stage(self, name: str) -> Iterator[None]:
        """Measure a compute/H2D stage using CUDA events or a CPU wall clock."""

        if not self.enabled:
            yield
            return
        if not self._use_cuda_events:
            with self.wall_stage(name):
                yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._cuda_events[str(name)] = (start, end)
            self._device_synchronized = False

    def synchronize_device(self) -> None:
        """Synchronize once after all timed device stages have been enqueued."""

        if self._use_cuda_events and not self._device_synchronized:
            torch.cuda.synchronize(self.device)
            self._device_synchronized = True

    def finish(self) -> dict[str, float]:
        """Return seconds per stage, synchronizing CUDA at most once."""

        if not self.enabled:
            return {}
        self.synchronize_device()
        timings = dict(self._wall_timings)
        for name, (start, end) in self._cuda_events.items():
            timings[name] = float(start.elapsed_time(end)) / 1000.0
        return timings


__all__ = ["GradientUpdateTimer"]
