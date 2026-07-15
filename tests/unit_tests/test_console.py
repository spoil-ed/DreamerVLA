import torch

from dreamervla.runtime.metrics import SuccessTracker
from dreamervla.utils import console
from dreamervla.utils.console import format_progress_line


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
    assert console.count_trainable(m) == 15
    for p in m.parameters():
        p.requires_grad_(False)
    assert console.count_trainable(m) == 0
    assert console.count_trainable(None) == 0


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


def test_cotrain_box_strings_are_wellformed_for_a_synthetic_step():
    tr = SuccessTracker(window=50)
    for s in (True, False, True):
        tr.update(s)
    rows = [
        f"VLA    succ@50={console.fmt_value(tr.rate())} (Δ {tr.delta():+.3f} · best {tr.best:.3f})   return=0.71",
        "train  wm=0.182  actor=0.226  cls_acc=0.95",
        "data   buf=10000  ep=3  cum_succ=0.667",
    ]
    box = console.metric_box("cotrain · step 1600", rows, width=65)
    assert all(len(ln) == 65 for ln in box.splitlines())
    done = console.phase_banner("[2/2] CLASSIFIER WARMUP", subtitle="acc 0.950", done=True)
    assert "acc 0.950" in done and len(done) == 65


def test_format_progress_line_with_total():
    s = format_progress_line(
        "pretokenize", 12800, 50000, elapsed_s=201.0, eta_s=585.0, rate=63.7
    )
    bar = "█" * 5 + "░" * 15  # 12800/50000 -> round(20*0.256)=5 filled
    assert s == f"pretokenize [{bar}] 12800/50000 (26%) · 03:21<09:45 · 63.7 it/s"


def test_format_progress_line_with_status_suffix():
    s = format_progress_line(
        "train",
        1,
        10,
        elapsed_s=10.0,
        eta_s=90.0,
        rate=0.1,
        unit="step",
        status="env_steps=128/200000 collect=t0:s32",
    )

    assert s.endswith(" · env_steps=128/200000 collect=t0:s32")


def test_format_progress_line_bar_fill_scales_with_progress():
    empty = format_progress_line("c", 0, 100, elapsed_s=1.0, eta_s=1.0, rate=1.0)
    half = format_progress_line("c", 50, 100, elapsed_s=1.0, eta_s=1.0, rate=1.0)
    full = format_progress_line("c", 100, 100, elapsed_s=1.0, eta_s=1.0, rate=1.0)
    assert f"[{'░' * 20}]" in empty
    assert f"[{'█' * 10}{'░' * 10}]" in half
    assert f"[{'█' * 20}]" in full


def test_format_progress_line_open_ended():
    s = format_progress_line(
        "collect", 812, None, elapsed_s=201.0, eta_s=None, rate=4.0, unit="ep"
    )
    assert s == "collect 812 · 03:21 · 4.0 ep/s"


def test_format_progress_line_hour_duration_and_zero_total():
    s = format_progress_line("train", 0, 0, elapsed_s=3725.0, eta_s=0.0, rate=0.0)
    # total<=0 is treated as open-ended (no pct/eta); duration rolls to h:mm:ss
    assert s == "train 0 · 1:02:05 · 0.0 it/s"


def test_format_metric_table_groups_metrics_like_rlinf():
    table = console.format_metric_table(
        step=4,
        total_steps=10,
        elapsed_s=20.0,
        metrics={
            "time/env/step": 1.234,
            "env/success_once": 0.5,
            "rollout/returns_mean": 1.0,
            "eval/success_rate": 0.25,
            "train/actor/loss": 0.1234,
            "train/critic/loss": 123.456,
            "train/replay_buffer/qsize": 7,
            "train/rl_loss": 0.2222,
        },
        start_step=0,
        width=120,
    )

    assert "Metric Table" in table
    assert "Global Step:    5/10" in table
    assert "Progress:" in table and "50.0%" in table
    assert "Elapsed: 00:20" in table and "ETA: 00:20" in table
    assert "Environment" in table and "success_once=0.500" in table
    assert "Rollout" in table and "returns_mean=1.000" in table
    assert "Evaluation" in table and "success_rate=0.250" in table
    assert "Replay Buffer" in table and "qsize=7" in table
    assert "Training/Actor" in table and "actor/loss=0.123" in table
    assert "Training/Critic" in table and "critic/loss=123.5" in table
    assert "Training/Other" in table and "rl_loss=0.222" in table
