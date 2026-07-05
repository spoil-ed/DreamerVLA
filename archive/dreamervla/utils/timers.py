"""Structured timing helpers (RLinf-style NamedTimer).

`Timers` accumulates per-name elapsed seconds and reduces them (mean/sum/min/
max) into the ``time/`` metric namespace, replacing scattered ad-hoc
``f"time/..."`` wall-clock points. `Profiler` is a config-gated, default-off
``torch.profiler`` wrapper so kernel-level hotspots can be sampled on demand.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from statistics import fmean
from typing import Any

_REDUCTIONS: dict[str, Callable[[list[float]], float]] = {
    "mean": fmean,
    "sum": sum,
    "min": min,
    "max": max,
}


class Timers:
    """Accumulate elapsed seconds per name and reduce them for logging."""

    def __init__(
        self,
        *,
        cuda_sync: bool = False,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._cuda_sync = bool(cuda_sync)
        self._time_fn = time_fn
        self._samples: dict[str, list[float]] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        self._maybe_sync()
        start = self._time_fn()
        try:
            yield
        finally:
            self._maybe_sync()
            self._samples.setdefault(name, []).append(self._time_fn() - start)

    def reduce(self, reduction: str = "mean") -> dict[str, float]:
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"unknown reduction {reduction!r}; expected one of {sorted(_REDUCTIONS)}"
            )
        fn = _REDUCTIONS[reduction]
        return {name: float(fn(samples)) for name, samples in self._samples.items()}

    def to_metrics(self, prefix: str = "time", reduction: str = "mean") -> dict[str, float]:
        normalized = prefix.rstrip("/")
        reduced = self.reduce(reduction)
        if not normalized:
            return reduced
        return {f"{normalized}/{name}": value for name, value in reduced.items()}

    def reset(self) -> None:
        self._samples.clear()

    def _maybe_sync(self) -> None:
        if not self._cuda_sync:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass


class Profiler:
    """Config-gated ``torch.profiler`` wrapper; a no-op when disabled.

    Usage mirrors the schedule-based torch profiler::

        with Profiler(enabled=cfg.profile, output_dir=trace_dir) as prof:
            for step in loop:
                ...
                prof.step()
    """

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Any = None,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
    ) -> None:
        self._prof = None
        if not enabled:
            return
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        on_trace_ready = (
            torch.profiler.tensorboard_trace_handler(str(output_dir))
            if output_dir is not None
            else None
        )
        self._prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            ),
            on_trace_ready=on_trace_ready,
        )

    @property
    def enabled(self) -> bool:
        return self._prof is not None

    def step(self) -> None:
        if self._prof is not None:
            self._prof.step()

    def __enter__(self) -> Profiler:
        if self._prof is not None:
            self._prof.__enter__()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._prof is not None:
            return bool(self._prof.__exit__(*exc))
        return False
