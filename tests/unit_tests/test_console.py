import torch

from dreamervla.utils import console
from dreamervla.utils.console import count_trainable
from dreamervla.runners.online_utils import SuccessTracker


def test_fmt_value_thresholds():
    assert console.fmt_value(0.0) == "0"
    assert console.fmt_value(2) == "2"
    assert console.fmt_value(0.12345) == "0.123"
    assert console.fmt_value(0.0009) == "9.00e-04"
    assert console.fmt_value(123456.0) == "1.23e+05"
    assert console.fmt_value("warmup") == "warmup"


def test_phase_banner_start_and_done_are_symmetric_width():
    start = console.phase_banner("[1/3] WM WARMUP", subtitle="256 steps", width=65)
    done = console.phase_banner("[1/3] WM WARMUP", subtitle="wm_loss 0.012", done=True, width=65)
    assert start.startswith("=") and start.endswith("=")
    assert len(start) == 65 and len(done) == 65
    assert "WM WARMUP" in start
    assert "done" in done


def test_metric_box_renders_header_and_rows():
    box = console.metric_box(
        "cotrain · env_step 1600/8000 · 20%",
        ["VLA    succ@50=0.62 (d +0.08 best 0.66)", "train  wm=0.182 actor=0.226"],
        width=65,
    )
    lines = box.splitlines()
    assert lines[0].startswith("╭") and lines[0].endswith("╮")   # top corners
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")  # bottom corners
    assert all(len(ln) == 65 for ln in lines)
    assert any("succ@50" in ln for ln in lines)


def test_count_trainable_counts_only_grad_params():
    m = torch.nn.Linear(4, 3)            # 4*3 + 3 = 15 params
    assert count_trainable(m) == 15
    for p in m.parameters():
        p.requires_grad_(False)
    assert count_trainable(m) == 0


def test_success_tracker_window_best_and_delta():
    t = SuccessTracker(window=4)
    assert t.rate() == 0.0 and len(t) == 0
    for s in (True, False, True, False):   # 2/4 = 0.5 over window
        t.update(s)
    assert t.rate() == 0.5
    assert t.best == 0.5
    # delta is vs last marked print; nothing marked yet -> 0.0
    assert t.delta() == 0.0
    t.mark_printed()
    t.update(True)  # window now (F,T,F,T) -> 0.5 still; then drops oldest True
    t.update(True)  # window (F,T,T,T) -> 0.75
    assert t.rate() == 0.75
    assert round(t.delta(), 3) == 0.25
    assert t.best == 0.75
