"""The real-data diagnostic owns an explicit, hostname-independent 8-rank launch."""

import sys

import pytest


@pytest.mark.parametrize("world_size", [None, "1"])
def test_probe_defaults_to_eight_ranks_with_static_loopback(monkeypatch, world_size):
    from dreamervla.diagnostics.checks import vjepa2_ac_state_smoke as diagnostic

    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    args = [
        "probe",
        "--output-dir",
        "/unused/output",
        "--decoder-ckpt",
        "/unused/decoder",
        "--wandb-mode",
        "online",
    ]
    monkeypatch.setattr(sys, "argv", args)
    commands = []
    monkeypatch.setattr(diagnostic.subprocess, "run", lambda cmd, **kw: commands.append((cmd, kw)))
    diagnostic.main()
    command, kwargs = commands[0]
    assert "--nproc-per-node=8" in command
    assert "--master-addr=127.0.0.1" in command
    assert "--standalone" not in command
    assert command[-len(args[1:]) :] == args[1:]
    assert kwargs == {"check": True}


def test_help_does_not_spawn_workers(monkeypatch):
    from dreamervla.diagnostics.checks import vjepa2_ac_state_smoke as diagnostic

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(sys, "argv", ["probe", "--help"])
    monkeypatch.setattr(
        diagnostic.subprocess, "run", lambda *a, **k: pytest.fail("spawned on help")
    )
    with pytest.raises(SystemExit) as caught:
        diagnostic.main()
    assert caught.value.code == 0
