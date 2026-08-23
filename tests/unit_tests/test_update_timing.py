from __future__ import annotations

import torch

from dreamervla.utils.update_timing import GradientUpdateTimer


def test_gradient_update_timer_records_cpu_and_device_stages() -> None:
    timer = GradientUpdateTimer(torch.device("cpu"), enabled=True)

    with timer.wall_stage("data_wait"):
        sum(range(10))
    with timer.device_stage("forward"):
        torch.ones(2).square()

    timings = timer.finish()

    assert set(timings) == {"data_wait", "forward"}
    assert all(value >= 0.0 for value in timings.values())


def test_gradient_update_timer_is_zero_overhead_when_disabled() -> None:
    timer = GradientUpdateTimer(torch.device("cpu"), enabled=False)

    with timer.wall_stage("data_wait"):
        pass
    with timer.device_stage("forward"):
        pass

    assert timer.finish() == {}
