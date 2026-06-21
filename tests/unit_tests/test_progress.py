from dreamervla.utils.progress import ProgressReporter


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _reporter(total=100, **kw):
    clk = _Clock()
    out = []
    r = ProgressReporter(
        total, "train", clock=clk, sink=out.append, min_interval_s=5.0, **kw
    )
    return r, clk, out


def test_first_update_prints_then_throttled_by_walltime():
    r, clk, out = _reporter()
    r.update()                 # first tick always prints
    assert len(out) == 1 and "train [" in out[0] and "1/100" in out[0]
    clk.t += 2.0
    r.update()                 # 2s < 5s -> suppressed
    assert len(out) == 1
    clk.t += 4.0
    r.update()                 # 6s since last print -> prints
    assert len(out) == 2 and "3/100" in out[1]


def test_close_always_prints_final_summary():
    r, clk, out = _reporter()
    r.update()                 # prints (first)
    clk.t += 1.0
    r.set(100)                 # throttled
    r.close()                  # always prints final
    assert "100/100" in out[-1]


def test_default_sink_flushes(monkeypatch):
    # Progress lines must flush so they surface under non-TTY block-buffered
    # stdout (nohup / Ray worker logs); a missing flush is why they vanish.
    calls = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: calls.append((a, k)))
    r = ProgressReporter(10, "train", clock=_Clock())  # default sink
    r.update()
    assert calls and calls[-1][1].get("flush") is True


def test_disabled_is_silent():
    out = []
    r = ProgressReporter(10, "x", enabled=False, sink=out.append, clock=_Clock())
    r.update()
    r.set(5)
    r.close()
    assert out == []


def test_open_ended_total_none_has_no_pct():
    clk = _Clock()
    out = []
    r = ProgressReporter(None, "collect", clock=clk, sink=out.append, unit="ep")
    r.update()
    assert out[0].startswith("collect 1 ·") and "%" not in out[0]
