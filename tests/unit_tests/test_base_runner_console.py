import types

from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner, _group_metric_rows


def _runner(cfg, *, main=True):
    obj = types.SimpleNamespace()
    obj.cfg = cfg
    obj.is_main_process = main
    for name in ("console_banner", "console_record_success", "console_metrics",
                 "console_metric_table", "_console_state_get", "console_success_rate"):
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


def test_console_metric_table_prints_rlinf_style_table(capsys):
    cfg = OmegaConf.create({"console": {"metric_table_width": 120}})
    r = _runner(cfg)

    r.console_metric_table(
        step=2,
        total_steps=5,
        elapsed_s=12.0,
        metrics={"env/success_once": 0.5, "train/actor/loss": 0.25},
        start_step=0,
    )

    out = capsys.readouterr().out
    assert "Metric Table" in out
    assert "Global Step:    3/5" in out
    assert "Environment" in out and "success_once=0.500" in out
    assert "Training/Actor" in out and "actor/loss=0.250" in out


def test_console_metric_table_guarded_on_non_main(capsys):
    cfg = OmegaConf.create({"console": {"metric_table_width": 120}})
    _runner(cfg, main=False).console_metric_table(
        step=0,
        total_steps=1,
        elapsed_s=1.0,
        metrics={"train/loss": 0.5},
    )
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


def _progress_runner(cfg, *, main=True):
    obj = types.SimpleNamespace()
    obj.cfg = cfg
    obj.is_main_process = main
    for name in ("console_progress", "_console_state_get"):
        setattr(obj, name, types.MethodType(getattr(BaseRunner, name), obj))
    return obj


def test_console_progress_prints_and_caches_per_desc(capsys):
    cfg = OmegaConf.create({"console": {"progress_every_s": 0.0}})
    r = _progress_runner(cfg)
    r.console_progress(1, 10, "train")
    r.console_progress(2, 10, "train")
    out = capsys.readouterr().out
    assert "train [" in out and "1/10" in out and "2/10" in out
    # one cached reporter per desc
    assert set(r._console_state["progress"].keys()) == {"train"}


def test_console_progress_updates_status_without_new_reporter(capsys):
    cfg = OmegaConf.create({"console": {"progress_every_s": 0.0}})
    r = _progress_runner(cfg)

    r.console_progress(0, 10, "train", unit="step", status="env_steps=1/100 collect=t0:s1")
    r.console_progress(0, 10, "train", unit="step", status="env_steps=2/100 collect=t0:s2")

    out = capsys.readouterr().out
    assert "env_steps=1/100 collect=t0:s1" in out
    assert "env_steps=2/100 collect=t0:s2" in out
    assert set(r._console_state["progress"].keys()) == {"train"}


def test_console_progress_guarded_on_non_main(capsys):
    cfg = OmegaConf.create({"console": {"progress_every_s": 0.0}})
    _progress_runner(cfg, main=False).console_progress(1, 10, "train")
    assert capsys.readouterr().out == ""


def test_console_progress_always_emits_and_releases_terminal_state(capsys):
    cfg = OmegaConf.create({"console": {"progress_every_s": 3600.0}})
    r = _progress_runner(cfg)

    r.console_progress(0, 10, "train")
    r.console_progress(10, 10, "train")

    out = capsys.readouterr().out
    assert "0/10" in out
    assert "10/10" in out
    assert "train" not in r._console_state["progress"]
