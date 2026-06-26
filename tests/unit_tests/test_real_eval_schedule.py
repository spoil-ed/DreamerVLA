from __future__ import annotations

from dreamervla.runners.real_eval_schedule import RealEvalState, should_run_real_eval


def test_real_eval_schedule_triggers_on_completed_episode_window() -> None:
    state = RealEvalState(last_eval_episode=0, last_eval_update=0)

    assert (
        should_run_real_eval(
            enabled=True,
            completed_episodes=5,
            learner_updates=0,
            every_episodes=5,
            every_learner_updates=0,
            state=state,
        )
        is True
    )


def test_real_eval_schedule_does_not_trigger_without_new_budget() -> None:
    state = RealEvalState(last_eval_episode=5, last_eval_update=10)

    assert (
        should_run_real_eval(
            enabled=True,
            completed_episodes=7,
            learner_updates=12,
            every_episodes=5,
            every_learner_updates=5,
            state=state,
        )
        is False
    )


def test_real_eval_schedule_triggers_on_update_window() -> None:
    state = RealEvalState(last_eval_episode=0, last_eval_update=10)

    assert (
        should_run_real_eval(
            enabled=True,
            completed_episodes=0,
            learner_updates=15,
            every_episodes=0,
            every_learner_updates=5,
            state=state,
        )
        is True
    )


def test_real_eval_schedule_stays_disabled_even_when_windows_pass() -> None:
    state = RealEvalState(last_eval_episode=0, last_eval_update=0)

    assert (
        should_run_real_eval(
            enabled=False,
            completed_episodes=100,
            learner_updates=100,
            every_episodes=1,
            every_learner_updates=1,
            state=state,
        )
        is False
    )
