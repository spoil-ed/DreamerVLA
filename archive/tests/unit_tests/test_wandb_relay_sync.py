import importlib
from pathlib import Path

import pytest

relay = importlib.import_module("dreamervla.diagnostics.wandb_relay_sync")


def _min_argv(**over):
    argv = ["--wandb-dir", "/shared/run/cotrain/log/wandb/all/wandb"]
    for k, v in over.items():
        argv += [k, v]
    return argv


def test_parse_args_defaults():
    cfg = relay.parse_args(_min_argv())
    assert cfg.wandb_dir == Path("/shared/run/cotrain/log/wandb/all/wandb")
    assert cfg.wandb_project is None
    assert cfg.wandb_entity is None
    assert cfg.interval == 60
    assert cfg.wandb_bin == "wandb"
    assert cfg.dry_run is False
    assert cfg.once is False
    assert cfg.log_file is None
    assert cfg.lock_file == Path("/shared/run/cotrain/log/wandb/all/wandb") / ".wandb_relay.lock"


def test_parse_args_overrides():
    cfg = relay.parse_args([
        "--wandb-dir", "/s/wandb",
        "--wandb-project", "proj", "--wandb-entity", "ent",
        "--interval", "120", "--wandb-bin", "/opt/wandb",
        "--log-file", "/var/log/relay.log", "--lock-file", "/var/lock/relay.lock",
        "--dry-run", "--once",
    ])
    assert cfg.wandb_dir == Path("/s/wandb")
    assert cfg.wandb_project == "proj"
    assert cfg.wandb_entity == "ent"
    assert cfg.interval == 120
    assert cfg.wandb_bin == "/opt/wandb"
    assert cfg.log_file == Path("/var/log/relay.log")
    assert cfg.lock_file == Path("/var/lock/relay.lock")
    assert cfg.dry_run is True
    assert cfg.once is True


def test_sync_cwd_is_parent_of_wandb_dir():
    cfg = relay.parse_args(["--wandb-dir", "/s/run/log/wandb/all/wandb"])
    assert relay.sync_cwd(cfg) == "/s/run/log/wandb/all"


def test_build_wandb_sync_command_with_project_entity():
    cfg = relay.parse_args([
        "--wandb-dir", "/s/wandb",
        "--wandb-project", "dreamervla", "--wandb-entity", "myteam",
    ])
    assert relay.build_wandb_sync_command(cfg) == [
        "wandb", "sync", "--sync-all", "--include-offline",
        "-p", "dreamervla", "-e", "myteam",
    ]


def test_build_wandb_sync_command_without_project_entity():
    cfg = relay.parse_args(["--wandb-dir", "/s/wandb"])
    assert relay.build_wandb_sync_command(cfg) == [
        "wandb", "sync", "--sync-all", "--include-offline",
    ]


def test_lock_is_exclusive(tmp_path):
    lock = tmp_path / "relay.lock"
    fh1 = relay.acquire_lock(lock)
    assert lock.read_text().strip().isdigit()  # PID written
    with pytest.raises(relay.RelayLockError):
        relay.acquire_lock(lock)
    fh1.close()  # releasing the flock lets a later acquire succeed
    fh2 = relay.acquire_lock(lock)
    fh2.close()


def test_run_once_dry_run_executes_nothing(tmp_path, caplog):
    cfg = relay.parse_args([
        "--wandb-dir", str(tmp_path / "wandb"),
        "--wandb-project", "proj", "--wandb-entity", "ent",
        "--dry-run",
    ])
    logger = relay.setup_logging(None)
    with caplog.at_level("INFO"):
        ok = relay.run_once(cfg, logger)
    assert ok is True
    text = caplog.text
    assert "wandb sync --sync-all --include-offline -p proj -e ent" in text
    assert "dry-run: not executing" in text


def test_run_once_reports_wandb_failure(tmp_path):
    cfg = relay.parse_args([
        "--wandb-dir", str(tmp_path / "wandb"),
        "--wandb-bin", "this-binary-does-not-exist-xyz",
    ])
    logger = relay.setup_logging(None)
    ok = relay.run_once(cfg, logger)
    assert ok is False


def test_main_dry_run_returns_zero_and_skips_lock(tmp_path):
    rc = relay.main([
        "--wandb-dir", str(tmp_path / "wandb"),
        "--dry-run",
    ])
    assert rc == 0
    # dry-run never acquires the lock
    assert not (tmp_path / "wandb" / ".wandb_relay.lock").exists()


def test_main_once_failure_returns_one(tmp_path):
    rc = relay.main([
        "--wandb-dir", str(tmp_path / "wandb"),
        "--wandb-bin", "this-binary-does-not-exist-xyz",
        "--lock-file", str(tmp_path / "relay.lock"),
        "--once",
    ])
    assert rc == 1


def test_main_once_lock_conflict_returns_two(tmp_path):
    lock = tmp_path / "relay.lock"
    held = relay.acquire_lock(lock)
    try:
        rc = relay.main([
            "--wandb-dir", str(tmp_path / "wandb"),
            "--lock-file", str(lock),
            "--once",
        ])
        assert rc == 2
    finally:
        held.close()
