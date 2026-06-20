import types

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner, _group_metric_rows


def _runner(cfg, *, main=True):
    obj = types.SimpleNamespace()
    obj.cfg = cfg
    obj.is_main_process = main
    for name in ("console_banner", "console_record_success", "console_metrics",
                 "_console_state_get", "console_success_rate"):
        setattr(obj, name, types.MethodType(getattr(BaseRunner, name), obj))
    return obj


def test_group_metric_rows_groups_by_namespace_and_skips_meta():
    rows = _group_metric_rows({"train/wm_loss": 0.182, "train/actor_loss": 0.226,
                               "rollout/success_rate": 0.55, "global_step": 5, "phase": "cotrain"})
    joined = "\n".join(rows)
    assert any(r.startswith("train") for r in rows)
    assert "wm_loss=0.182" in joined and "actor_loss=0.226" in joined
    assert "global_step" not in joined and "phase" not in joined


def test_console_banner_guarded(capsys):
    cfg = OmegaConf.create({"console": {"banner_width": 65}})
    _runner(cfg).console_banner("[1/3] WM WARMUP", subtitle="256 steps")
    out = capsys.readouterr().out
    assert "WM WARMUP" in out and len(out.strip()) == 65
    _runner(cfg, main=False).console_banner("X")
    assert capsys.readouterr().out == ""


def test_console_metrics_throttle_and_vla_row(capsys):
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 2, "success_window": 4}})
    r = _runner(cfg)
    for s in (True, False, True, False):
        r.console_record_success(s)
    r.console_metrics("cotrain · step 1", {"train/wm_loss": 0.18})  # counter=1, log_every=2 -> no print
    assert capsys.readouterr().out == ""
    r.console_metrics("cotrain · step 2", {"train/wm_loss": 0.18})  # counter=2 -> print
    out = capsys.readouterr().out
    assert "VLA" in out and "succ@4=" in out and "wm_loss=0.18" in out
    assert all(len(ln) == 65 for ln in out.strip().splitlines())


def test_console_metrics_force_bypasses_throttle(capsys):
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 5, "success_window": 4}})
    r = _runner(cfg)
    r.console_metrics("step 1", {"train/loss": 0.5}, force=True)  # call #1, log_every=5 -> should print
    out = capsys.readouterr().out
    assert out.strip() != "", "force=True must print regardless of throttle"
    r.console_metrics("step 2", {"train/loss": 0.4})  # call #2, not a multiple of 5 -> no print
    assert capsys.readouterr().out == ""


def test_console_success_rate_zero_before_any_record():
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 1, "success_window": 4}})
    r = _runner(cfg)
    assert r.console_success_rate() == 0.0


def test_console_success_rate_after_records():
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 1, "success_window": 4}})
    r = _runner(cfg)
    for s in (True, True, False, True):  # 3 successes / 4 total -> rate = 0.75
        r.console_record_success(s)
    assert abs(r.console_success_rate() - 0.75) < 1e-9


def test_console_record_success_non_main():
    cfg = OmegaConf.create({"console": {"banner_width": 65, "log_every": 1, "success_window": 4}})
    r = _runner(cfg, main=False)
    for s in (True, True, False):
        r.console_record_success(s)
    assert abs(r.console_success_rate() - (2 / 3)) < 1e-9


def test_group_metric_rows_skip_success_drops_rollout_success_rate():
    rows = _group_metric_rows(
        {"rollout/success_rate": 0.5, "train/loss": 0.1},
        skip_success=True,
    )
    joined = "\n".join(rows)
    assert "success_rate" not in joined
    assert "loss=0.1" in joined
