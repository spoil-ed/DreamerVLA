"""RLINF-02: structured Timers helper.

RLinf has a NamedTimer with mean/sum/min/max reduction + optional CUDA sync.
The repo previously only had scattered ad-hoc ``f"time/..."`` wall-clock points
and no reusable timer. These tests pin the canonical helper feeding ``time/``.
"""


def test_timers_reduce_supports_mean_sum_min_max():
    from dreamervla.utils.timers import Timers

    # Injected clock: block "step" measures 1.0s then 3.0s.
    ticks = iter([0.0, 1.0, 10.0, 13.0])
    timers = Timers(time_fn=lambda: next(ticks))
    with timers.time("step"):
        pass
    with timers.time("step"):
        pass

    assert timers.reduce("sum") == {"step": 4.0}
    assert timers.reduce("mean") == {"step": 2.0}
    assert timers.reduce("min") == {"step": 1.0}
    assert timers.reduce("max") == {"step": 3.0}


def test_timers_to_metrics_namespaces_under_prefix():
    from dreamervla.utils.timers import Timers

    ticks = iter([0.0, 0.5])
    timers = Timers(time_fn=lambda: next(ticks))
    with timers.time("wm_forward"):
        pass

    assert timers.to_metrics(prefix="time") == {"time/wm_forward": 0.5}


def test_timers_reset_clears_samples():
    from dreamervla.utils.timers import Timers

    ticks = iter([0.0, 1.0])
    timers = Timers(time_fn=lambda: next(ticks))
    with timers.time("x"):
        pass
    timers.reset()

    assert timers.reduce("mean") == {}


def test_timers_cuda_sync_is_safe_without_cuda():
    from dreamervla.utils.timers import Timers

    # cuda_sync=True must be a no-op (not a crash) when CUDA is unavailable.
    timers = Timers(cuda_sync=True)
    with timers.time("x"):
        pass

    assert "x" in timers.reduce("mean")


def test_profiler_disabled_is_a_safe_noop():
    from dreamervla.utils.timers import Profiler

    prof = Profiler(enabled=False)
    assert prof.enabled is False
    with prof as p:
        p.step()  # must not raise when disabled


def test_profiler_enabled_writes_a_trace_on_cpu(tmp_path):
    import torch

    from dreamervla.utils.timers import Profiler

    prof = Profiler(enabled=True, output_dir=tmp_path, wait=0, warmup=0, active=1, repeat=1)
    assert prof.enabled is True
    with prof as p:
        for _ in range(4):
            torch.ones(8, 8) @ torch.ones(8, 8)
            p.step()

    # tensorboard_trace_handler flushes a trace file into output_dir.
    assert any(tmp_path.iterdir())
