from __future__ import annotations

from pathlib import Path

from dreamervla.utils.episode_end import resolve_episode_end


def test_episode_end_marks_success_as_terminal_not_timeout() -> None:
    end = resolve_episode_end(success=True, elapsed_steps=1, max_steps=1)

    assert end.terminated is True
    assert end.truncated is False
    assert end.done is True


def test_episode_end_marks_max_steps_failure_as_timeout_not_terminal() -> None:
    end = resolve_episode_end(success=False, elapsed_steps=2, max_steps=2)

    assert end.terminated is False
    assert end.truncated is True
    assert end.done is True


def test_online_env_wrappers_use_shared_episode_end_logic() -> None:
    repo = Path(__file__).resolve().parents[2]
    libero_env = (repo / "dreamervla/envs/libero/libero_env.py").read_text(
        encoding="utf-8"
    )

    assert "from dreamervla.utils.episode_end import resolve_episode_end" in libero_env
    assert "DreamerVLAOnlineTrainEnv" in libero_env
    assert "episode_end = resolve_episode_end" in libero_env


def test_online_training_script_separates_episode_horizon_from_training_budget() -> (
    None
):
    repo = Path(__file__).resolve().parents[2]
    runner = (repo / "dreamervla/runners/online_cotrain_runner.py").read_text(
        encoding="utf-8"
    )

    assert 'OmegaConf.select(oc, "total_env_steps"' in runner
    assert 'OmegaConf.select(cfg, "env.episode_horizon"' in runner
    assert 'OmegaConf.select(oc, "max_train_updates"' in runner
    assert "total_env_steps" in runner
    assert "episode_horizon" in runner
    assert "max_env_steps" not in runner
