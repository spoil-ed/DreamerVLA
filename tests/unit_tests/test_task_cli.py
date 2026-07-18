from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", "dreamer-wm", "--task", "libero_goal", "dry_run=true"],
        ["--config", "dreamer-wm", "--task=libero_goal", "dry_run=true"],
    ],
)
def test_experiment_launcher_accepts_task_flag(argv: list[str]) -> None:
    from dreamervla.launchers.train import build_launch

    launch = build_launch(argv)

    assert launch.cfg.task.name == "libero_goal"
    assert "task=libero_goal" in launch.command


@pytest.mark.parametrize(
    ("config_name", "expected_override"),
    [
        ("preprocess/preprocess_suite", "task=libero_goal"),
        ("preprocess/preprocess_all", "tasks=[libero_goal]"),
        ("preprocess/preprocess_libero", "tasks=[libero_goal]"),
        ("preprocess/validate_libero_data", "tasks=[libero_goal]"),
        ("download/config", "env.LIBERO_SUITES=[libero_goal]"),
    ],
)
def test_workflow_task_flag_dispatches_only_selected_suite(
    config_name: str,
    expected_override: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dreamervla.launchers.workflow import main

    result = main(
        [
            "--config-name",
            config_name,
            "--task",
            "libero_goal",
            "dry_run=true",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    if config_name == "download/config":
        assert "SUITE" not in output or "libero_goal" in output
    else:
        assert "task=libero_goal" in output or "TASK" in output
    assert "libero_object" not in output
    assert "libero_spatial" not in output
    assert "libero_10" not in output

    # The parser exposes the exact normalization contract independently of log rendering.
    from dreamervla.launchers.task_cli import normalize_task_flag

    remaining, override = normalize_task_flag(
        ["--task", "libero_goal", "dry_run=true"],
        hydra_key=(
            "env.LIBERO_SUITES"
            if config_name == "download/config"
            else "tasks"
            if config_name != "preprocess/preprocess_suite"
            else "task"
        ),
        as_list=config_name != "preprocess/preprocess_suite",
    )
    assert remaining == ["dry_run=true"]
    assert override == expected_override


def test_reproduction_launcher_accepts_supported_task_flag() -> None:
    from dreamervla.launchers.reproduce import build_workflow

    workflow = build_workflow(
        [
            "--config-name",
            "reproduce/prepare_assets",
            "--task",
            "libero_goal",
            "dry_run=true",
        ]
    )

    assert workflow.cfg.profile.task == "libero_goal"
    assert workflow.dry_run is True


def test_official_openvla_eval_accepts_task_flag() -> None:
    from dreamervla.diagnostics.eval_openvla_oft_libero import _parse_hydra_like_argv

    config_name, overrides = _parse_hydra_like_argv(["--task", "libero_goal", "ckpt=/tmp/model"])

    assert config_name == "eval"
    assert overrides == ["ckpt=/tmp/model", "suite=libero_goal"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--task"],
        ["--task", "libero_goal", "--task", "libero_object"],
        ["--task", "unknown_suite"],
        ["--task", "libero_goal", "task=libero_object"],
    ],
)
def test_task_flag_rejects_missing_duplicate_unknown_or_conflicting_values(
    argv: list[str],
) -> None:
    from dreamervla.launchers.task_cli import normalize_task_flag

    with pytest.raises(SystemExit):
        normalize_task_flag(argv, hydra_key="task")
