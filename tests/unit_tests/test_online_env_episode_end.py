from __future__ import annotations

from pathlib import Path

from dreamer_vla.utils.episode_end import resolve_episode_end


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
    train_env = (repo / "dreamer_vla/envs/train_env.py").read_text(encoding="utf-8")
    libero_online_env = (repo / "dreamer_vla/envs/libero_online_env.py").read_text(
        encoding="utf-8"
    )

    assert "from dreamer_vla.utils.episode_end import resolve_episode_end" in train_env
    assert "from dreamer_vla.utils.episode_end import resolve_episode_end" in libero_online_env
    assert "episode_end = resolve_episode_end" in train_env
    assert "episode_end = resolve_episode_end" in libero_online_env


def test_online_training_script_separates_episode_horizon_from_training_budget() -> (
    None
):
    repo = Path(__file__).resolve().parents[2]
    script = (
        repo / "scripts/training/train_online_pi0_action_hidden_dreamervla.py"
    ).read_text(encoding="utf-8")

    assert "--episode-horizon" in script
    assert "--total-env-steps" in script
    assert "--max-train-updates" in script
    assert "max_steps=args.episode_horizon" in script
    assert "args.total_env_steps" in script
    assert "args.max_env_steps" not in script
