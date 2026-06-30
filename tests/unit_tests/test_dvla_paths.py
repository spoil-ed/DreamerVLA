from __future__ import annotations

from pathlib import Path

import pytest


def test_data_root_falls_back_to_dvla_root_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dreamervla.utils.paths import data_root

    monkeypatch.delenv("DVLA_DATA_ROOT", raising=False)
    monkeypatch.setenv("DVLA_ROOT", "/repo/root")

    assert data_root() == Path("/repo/root/data")


def test_data_root_prefers_dvla_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    from dreamervla.utils.paths import data_path, data_root

    monkeypatch.setenv("DVLA_ROOT", "/repo/root")
    monkeypatch.setenv("DVLA_DATA_ROOT", "/asset/root")

    assert data_root() == Path("/asset/root")
    assert data_path("checkpoints", "model") == Path("/asset/root/checkpoints/model")


def test_coldstart_launcher_uses_dvla_root_data_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import _data_root

    monkeypatch.delenv("DVLA_DATA_ROOT", raising=False)
    monkeypatch.setenv("DVLA_ROOT", "/repo/root")

    assert _data_root() == Path("/repo/root/data")


def test_coldstart_plan_uses_dvla_root_data_interpolation(tmp_path: Path) -> None:
    from hydra import compose, initialize_config_dir

    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(mode="noray", run_root=tmp_path, python="python")

    assert "task=openvla_onetraj_coldstart_libero" in plan.cotrain_cmd
    assert not any(override.startswith("+policy.init_lm_head_ckpt=") for override in plan.cotrain_cmd)

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=openvla_onetraj_libero_cotrain_noray",
                "task=openvla_onetraj_coldstart_libero",
            ],
        )
    assert cfg.policy.init_lm_head_ckpt == cfg.task.openvla_oft.ckpt_path


def test_coldstart_dry_run_defaults_use_dvla_root_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import main

    monkeypatch.delenv("DVLA_DATA_ROOT", raising=False)
    monkeypatch.setenv("DVLA_ROOT", "/repo/root")

    assert main(["mode=noray", "dry_run=true", "python=python"]) == 0

    out = capsys.readouterr().out
    assert "run_root: /repo/root/data/outputs/coldstart_warmup_cotrain/" in out
