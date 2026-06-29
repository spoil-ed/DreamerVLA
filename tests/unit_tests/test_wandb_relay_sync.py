import importlib
from pathlib import Path

import pytest

relay = importlib.import_module("dreamervla.diagnostics.wandb_relay_sync")


def _min_argv(**over):
    argv = [
        "--remote-host", "gpu.example.com",
        "--remote-wandb-dir", "/remote/run/cotrain/log/wandb",
        "--local-mirror-dir", "/tmp/mirror",
    ]
    for k, v in over.items():
        argv += [k, v]
    return argv


def test_parse_args_defaults():
    cfg = relay.parse_args(_min_argv())
    assert cfg.remote_host == "gpu.example.com"
    assert cfg.remote_user is None
    assert cfg.remote_wandb_dir == "/remote/run/cotrain/log/wandb"
    assert cfg.local_mirror_dir == Path("/tmp/mirror")
    assert cfg.wandb_project is None
    assert cfg.wandb_entity is None
    assert cfg.interval == 60
    assert cfg.rsync_bin == "rsync"
    assert cfg.wandb_bin == "wandb"
    assert cfg.ssh_port == 22
    assert cfg.dry_run is False
    assert cfg.once is False
    assert cfg.log_file is None
    assert cfg.excludes == []
    assert cfg.lock_file == Path("/tmp/mirror") / ".wandb_relay.lock"


def test_parse_args_overrides_and_repeated_excludes():
    cfg = relay.parse_args([
        "--remote-host", "h", "--remote-user", "u",
        "--remote-wandb-dir", "/r/wandb", "--local-mirror-dir", "/m",
        "--wandb-project", "proj", "--wandb-entity", "ent",
        "--interval", "120", "--ssh-port", "2222",
        "--rsync-bin", "/usr/bin/rsync", "--wandb-bin", "/opt/wandb",
        "--log-file", "/var/log/relay.log", "--lock-file", "/var/lock/relay.lock",
        "--dry-run", "--once",
        "--exclude", "*.tmp", "--exclude", "media/**",
    ])
    assert cfg.remote_user == "u"
    assert cfg.wandb_project == "proj"
    assert cfg.wandb_entity == "ent"
    assert cfg.interval == 120
    assert cfg.ssh_port == 2222
    assert cfg.rsync_bin == "/usr/bin/rsync"
    assert cfg.wandb_bin == "/opt/wandb"
    assert cfg.log_file == Path("/var/log/relay.log")
    assert cfg.lock_file == Path("/var/lock/relay.lock")
    assert cfg.dry_run is True
    assert cfg.once is True
    assert cfg.excludes == ["*.tmp", "media/**"]


def test_build_remote_spec():
    assert relay.build_remote_spec("host", None) == "host"
    assert relay.build_remote_spec("host", "user") == "user@host"


def test_build_rsync_command():
    cfg = relay.parse_args([
        "--remote-host", "gpu", "--remote-user", "alice",
        "--remote-wandb-dir", "/data/run/cotrain/log/wandb",
        "--local-mirror-dir", "/tmp/mirror",
        "--ssh-port", "2200",
        "--exclude", "*.tmp", "--exclude", "media/**",
    ])
    assert relay.build_rsync_command(cfg) == [
        "rsync", "-a", "-z", "--partial", "--inplace",
        "-e", "ssh -p 2200",
        "--exclude", "*.tmp",
        "--exclude", "media/**",
        "alice@gpu:/data/run/cotrain/log/wandb/",
        "/tmp/mirror/wandb/",
    ]


def test_build_rsync_command_no_user_default_port():
    cfg = relay.parse_args([
        "--remote-host", "gpu",
        "--remote-wandb-dir", "/r/wandb",
        "--local-mirror-dir", "/m",
    ])
    assert relay.build_rsync_command(cfg) == [
        "rsync", "-a", "-z", "--partial", "--inplace",
        "-e", "ssh -p 22",
        "gpu:/r/wandb/",
        "/m/wandb/",
    ]


def test_build_wandb_sync_command_with_project_entity():
    cfg = relay.parse_args([
        "--remote-host", "gpu", "--remote-wandb-dir", "/r/wandb",
        "--local-mirror-dir", "/m",
        "--wandb-project", "dreamervla", "--wandb-entity", "myteam",
    ])
    assert relay.build_wandb_sync_command(cfg) == [
        "wandb", "sync", "--sync-all", "--include-offline",
        "-p", "dreamervla", "-e", "myteam",
    ]


def test_build_wandb_sync_command_without_project_entity():
    cfg = relay.parse_args([
        "--remote-host", "gpu", "--remote-wandb-dir", "/r/wandb",
        "--local-mirror-dir", "/m",
    ])
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
        "--remote-host", "gpu", "--remote-wandb-dir", "/r/wandb",
        "--local-mirror-dir", str(tmp_path / "mirror"),
        "--wandb-project", "proj", "--wandb-entity", "ent",
        "--dry-run",
    ])
    logger = relay.setup_logging(None)
    with caplog.at_level("INFO"):
        ok = relay.run_once(cfg, logger)
    assert ok is True
    assert not (tmp_path / "mirror").exists()
    text = caplog.text
    assert "rsync -a -z --partial --inplace" in text
    assert "wandb sync --sync-all --include-offline" in text


def test_run_once_reports_rsync_failure(tmp_path, monkeypatch):
    cfg = relay.parse_args([
        "--remote-host", "gpu", "--remote-wandb-dir", "/r/wandb",
        "--local-mirror-dir", str(tmp_path / "mirror"),
        "--rsync-bin", "this-binary-does-not-exist-xyz",
    ])
    logger = relay.setup_logging(None)
    calls = []
    real_run = relay.subprocess.run

    def spy(cmd, *a, **k):
        calls.append(cmd[0])
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(relay.subprocess, "run", spy)
    ok = relay.run_once(cfg, logger)
    assert ok is False
    assert calls == ["this-binary-does-not-exist-xyz"]  # wandb never reached


def test_main_dry_run_returns_zero_and_skips_lock(tmp_path):
    rc = relay.main([
        "--remote-host", "gpu", "--remote-wandb-dir", "/r/wandb",
        "--local-mirror-dir", str(tmp_path / "mirror"),
        "--dry-run",
    ])
    assert rc == 0
    assert not (tmp_path / "mirror").exists()


def test_main_once_failure_returns_one(tmp_path):
    rc = relay.main([
        "--remote-host", "gpu", "--remote-wandb-dir", "/r/wandb",
        "--local-mirror-dir", str(tmp_path / "mirror"),
        "--rsync-bin", "this-binary-does-not-exist-xyz",
        "--once",
    ])
    assert rc == 1


def test_main_once_lock_conflict_returns_two(tmp_path):
    lock = tmp_path / "relay.lock"
    held = relay.acquire_lock(lock)
    try:
        rc = relay.main([
            "--remote-host", "gpu", "--remote-wandb-dir", "/r/wandb",
            "--local-mirror-dir", str(tmp_path / "mirror"),
            "--lock-file", str(lock),
            "--once",
        ])
        assert rc == 2
    finally:
        held.close()
