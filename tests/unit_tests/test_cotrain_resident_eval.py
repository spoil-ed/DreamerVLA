from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dreamervla.runners import CotrainRunner
from dreamervla.runners.cotrain_runner import _resident_eval_worker_layout
from dreamervla.workers.cotrain.messages import RealTrajectory, RealTrajectoryBatch


class _Ready:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value

    def done(self) -> bool:
        return True

    def ready(self):
        return []


class _Actor:
    def __init__(self) -> None:
        self.sync_versions: list[int] = []

    def sync_model_to_rollout(self, _key: str, version: int):
        self.sync_versions.append(int(version))
        return _Ready([{"sync/policy_version": float(version)}])

    def release_synced_model(self, _key: str, version: int):
        return _Ready([{"sync/policy_buckets_released": float(version >= 0)}])

    def reencode_real_trajectories(self, batch: RealTrajectoryBatch):
        self.reencoded_batch = batch
        return _Ready([batch])

    def finalize_policy_transaction(self, observed_kl: float, max_kl: float):
        accepted = float(observed_kl) <= float(max_kl)
        return _Ready(
            [
                {
                    "actor/kl_transaction_accepted": float(accepted),
                    "actor/kl_transaction_rolled_back": float(not accepted),
                }
            ]
        )


class _Rollout:
    workers = (object(), object())

    def __init__(self) -> None:
        self.pull_calls = 0
        self.generate_calls = 0

    def sync_model_from_actor(
        self,
        _key: str,
        expected_version: int | None = None,
    ):
        del expected_version
        self.pull_calls += 1
        return _Ready([{"sync/rollout_policy_updated": 1.0}])

    def generate(self, *_args):
        self.generate_calls += 1
        return _Ready([{"rollout/generated": 1.0}])


class _EvalEnv:
    def set_global_step(self, _step: int):
        return _Ready([None])

    def begin_evaluation_pass(self):
        return _Ready([{}])

    def configure_progress(self, *_args, **_kwargs):
        return _Ready([None])

    def interact(self, *_args):
        return _Ready(
            [
                {
                    "env/eval_env/episodes_successful": 56.0,
                    "env/eval_env/chunk_steps": 3800.0,
                }
            ]
        )

    def drain_real_trajectories(self, global_step: int):
        return _Ready([_eval_batch(global_step)])


class _Learner:
    def __init__(self) -> None:
        self.evaluated_batch = None

    def evaluate_cotrain_trajectories(self, batch: RealTrajectoryBatch):
        self.evaluated_batch = batch
        return _Ready(
            [
                {
                    "eval/wm_closed_loop_cosine": 0.75,
                    "eval/classifier_real_f1": 0.8,
                    "eval/classifier_real_accuracy": 0.82,
                }
            ]
        )


class _Channel:
    def __init__(self) -> None:
        self.puts = []

    def put(self, value, *, key: str):
        self.puts.append((str(key), value))


def _eval_batch(global_step: int) -> RealTrajectoryBatch:
    trajectories = tuple(
        RealTrajectory(
            env_rank=0,
            slot_id=index,
            task_id=index % 10,
            episode_id=index,
            global_step=int(global_step),
            success=index % 2 == 0,
            transitions=(
                {
                    "obs_embedding": [[1.0]],
                    "action": [0.0],
                },
            ),
        )
        for index in range(100)
    )
    return RealTrajectoryBatch(global_step=int(global_step), trajectories=trajectories)


def _cfg():
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.CotrainRunner",
            "training": {"out_dir": "/tmp/dvla-resident-eval", "seed": 7},
            "logger": {"logger_backends": []},
            "manual_cotrain": {
                "ngpu": 8,
                "global_steps": 1,
                "real_env_enabled": True,
                "learner_updates_enabled": True,
                "staged_policy_update": True,
                "eval_interval_global_steps": 1,
                "env_rollout_timeout_s": 10,
                "num_action_chunks": 8,
                "real_task_ids": list(range(10)),
                "eval_protocol": {
                    "task_ids": list(range(10)),
                    "num_episodes_per_task": 10,
                    "num_envs": 25,
                    "max_steps": 300,
                    "render_backend": "osmesa",
                },
            },
        }
    )


@pytest.mark.parametrize(
    ("eval_slots", "rollout_workers", "expected"),
    [
        (25, 6, (5, 5, 4)),
        (25, 8, (5, 5, 4)),
        (24, 6, (6, 4, 4)),
        (8, 1, (1, 8, 12)),
    ],
)
def test_resident_eval_layout_uses_all_compatible_rollout_ranks(
    eval_slots: int,
    rollout_workers: int,
    expected: tuple[int, int, int],
) -> None:
    assert (
        _resident_eval_worker_layout(
            eval_slots=eval_slots,
            eval_episodes=eval_slots * expected[2],
            rollout_workers=rollout_workers,
        )
        == expected
    )


def test_resident_eval_layout_rejects_uneven_episode_budget() -> None:
    with pytest.raises(ValueError, match="total episodes must be divisible"):
        _resident_eval_worker_layout(
            eval_slots=25,
            eval_episodes=101,
            rollout_workers=6,
        )


def test_resumed_eval_can_establish_paired_baseline_at_nonzero_step() -> None:
    runner = CotrainRunner(_cfg())

    baseline = runner._paired_eval_outcome_metrics(
        _eval_batch(1),
        global_step=1,
        establish_baseline=True,
    )
    post = runner._paired_eval_outcome_metrics(
        _eval_batch(2),
        global_step=2,
    )

    assert baseline["eval/paired/is_baseline"] == 1.0
    assert baseline["eval/paired/baseline_success_rate"] == 0.5
    assert post["eval/paired/post_success_rate"] == 0.5


def test_eval_guard_commits_only_non_regressing_candidate() -> None:
    cfg = _cfg()
    cfg.manual_cotrain.rollback_on_eval_regression = True
    runner = CotrainRunner(cfg)
    runner._eval_guard_best_score = 0.56
    actor = _Actor()

    accepted = {
        "eval/success_rate": 0.58,
        "actor/policy_kl_total": 0.01,
        "actor/success_sft_optimizer_steps": 32.0,
        "actor/success_sft_update_committed": 1.0,
    }
    accepted_metrics = runner._finalize_eval_guard(actor, accepted)

    assert accepted_metrics["train/eval_guard_accepted"] == 1.0
    assert accepted_metrics["eval/accepted_success_rate"] == 0.58
    assert runner._applied_policy_steps == 32

    rejected = {
        "eval/success_rate": 0.55,
        "actor/policy_kl_total": 0.01,
        "actor/success_sft_optimizer_steps": 32.0,
        "actor/success_sft_update_committed": 1.0,
    }
    rejected_metrics = runner._finalize_eval_guard(actor, rejected)

    assert rejected_metrics["train/eval_guard_rolled_back"] == 1.0
    assert rejected_metrics["eval/accepted_success_rate"] == 0.58
    assert rejected["actor/success_sft_update_committed"] == 0.0
    assert rejected["actor/policy_kl_total"] == 0.0
    assert runner._applied_policy_steps == 32
    assert runner._expected_eval_guard_rejection(rejected_metrics) is True
    assert runner._expected_eval_guard_rejection(accepted_metrics) is False


def test_resident_eval_reuses_rollout_group_without_checkpoint_reload() -> None:
    runner = CotrainRunner(_cfg())
    runner.console_progress = lambda *_args, **_kwargs: None
    runner._paired_eval_outcome_metrics(_eval_batch(0), global_step=0)
    actor = _Actor()
    rollout = _Rollout()
    learner = _Learner()
    channel = _Channel()
    groups = {
        "ActorGroup": actor,
        "RolloutGroup": rollout,
        "LearnerGroup": learner,
        "EvaluationEnvGroup": _EvalEnv(),
        "eval_env_channel": channel,
        "eval_env_channel_name": "eval-env",
        "eval_rollout_channel_name": "eval-rollout",
        "eval_actor_channel_name": "eval-actor",
    }

    metrics = runner._evaluate_resident_policy(
        groups,
        global_step=1,
        sync_policy=True,
    )

    assert actor.sync_versions == [3]
    assert rollout.pull_calls == 1
    assert rollout.generate_calls == 1
    assert len(channel.puts) == 2
    assert metrics["eval/episodes"] == 100.0
    assert metrics["eval/successes"] == 56.0
    assert metrics["eval/success_rate"] == 0.56
    assert metrics["eval/paired/baseline_success_rate"] == 0.5
    assert metrics["eval/paired/post_success_rate"] == 0.5
    assert metrics["eval/paired/success_rate_delta"] == 0.0
    assert metrics["eval/paired/success_rate_percentage_points"] == 0.0
    assert metrics["eval/paired/relative_success_rate_improvement_percent"] == 0.0
    assert metrics["eval/wm_closed_loop_cosine"] == 0.75
    assert metrics["eval/classifier_real_f1"] == 0.8
    assert metrics["eval/classifier_real_accuracy"] == 0.82
    assert actor.reencoded_batch.num_trajectories == 100
    assert learner.evaluated_batch.num_trajectories == 100


def test_resident_eval_can_skip_expensive_wm_diagnostics() -> None:
    cfg = _cfg()
    cfg.manual_cotrain.eval_protocol.compute_diagnostics = False
    runner = CotrainRunner(cfg)
    runner.console_progress = lambda *_args, **_kwargs: None
    actor = _Actor()
    learner = _Learner()
    groups = {
        "ActorGroup": actor,
        "RolloutGroup": _Rollout(),
        "LearnerGroup": learner,
        "EvaluationEnvGroup": _EvalEnv(),
        "eval_env_channel": _Channel(),
        "eval_env_channel_name": "eval-env",
        "eval_rollout_channel_name": "eval-rollout",
        "eval_actor_channel_name": "eval-actor",
    }

    metrics = runner._evaluate_resident_policy(
        groups,
        global_step=0,
        sync_policy=False,
    )

    assert metrics["eval/success_rate"] == 0.56
    assert "eval/wm_closed_loop_cosine" not in metrics
    assert not hasattr(actor, "reencoded_batch")
    assert learner.evaluated_batch is None
