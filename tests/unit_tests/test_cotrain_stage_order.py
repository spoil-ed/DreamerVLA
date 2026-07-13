from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from dreamervla.runners.cotrain_runner import CotrainRunner
from dreamervla.workers.cotrain.messages import (
    RealTrajectory,
    RealTrajectoryBatch,
    StopMsg,
)


class _Ready:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value

    def done(self) -> bool:
        return True


def _cfg():
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.CotrainRunner",
            "seed": 7,
            "training": {"out_dir": "/tmp/dvla-stage-order", "seed": 7},
            "logger": {"logger_backends": []},
            "cluster": {"num_nodes": 1, "num_gpus": 1},
            "manual_cotrain": {
                "ngpu": 1,
                "global_steps": 1,
                "staged_policy_update": True,
                "learner_updates_enabled": True,
                "learner_update_step": 1,
                "learner_update_phase": "cotrain",
                "learner_updates_per_global_step": 4,
                "learner_early_stop_patience": 2,
                "real_env_enabled": True,
                "real_rollout_target_trajectories": 1,
                "rollout_epoch": 1,
                "real_rollout_epoch": 1,
                "wm_rollout_epoch": 1,
                "max_steps_per_rollout_epoch": 2,
                "num_action_chunks": 1,
                "envs_per_worker": 1,
                "wm_envs_per_worker": 1,
                "sync_every": 1,
                "checkpoint_every": 0,
                "save_replay_state": False,
                "publish_learner_weights": False,
            },
            "actor": {"train_cfg": {"algorithm_cfg": {"group_size": 1}}},
        }
    )


def _real_batch(global_step: int, *, encoded: bool = False) -> RealTrajectoryBatch:
    transition = {
        "agentview_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "action": np.zeros((7,), dtype=np.float32),
        "reward": 1.0,
        "done": True,
        "policy_decision": True,
        "action_token_ids_chunk": np.zeros((8, 7), dtype=np.int64),
    }
    if encoded:
        transition.update(
            {
                "obs_embedding": np.zeros((2, 3), dtype=np.float32),
                "lang_emb": np.zeros((3,), dtype=np.float32),
                "encoder_version": int(global_step),
            }
        )
    return RealTrajectoryBatch(
        global_step=int(global_step),
        trajectories=(
            RealTrajectory(
                env_rank=0,
                slot_id=0,
                task_id=0,
                episode_id=1,
                global_step=int(global_step),
                success=True,
                transitions=(transition,),
            ),
        ),
    )


class _Actor:
    workers = (object(),)

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def execute_on(self, *ranks: int):
        assert ranks == (0,)
        return self

    def set_global_step(self, step: int):
        return _Ready([None])

    def sync_model_to_rollout(self, key: str, version: int):
        self.events.append(f"policy_push:{version}")
        return _Ready([{"sync/policy_version": float(version)}])

    def encoder_sft(self, batch: RealTrajectoryBatch):
        self.events.append("encoder_sft")
        assert batch.num_trajectories == 1
        return _Ready([{"actor/encoder_sft_optimizer_steps": 1.0}])

    def reencode_real_trajectories(self, batch: RealTrajectoryBatch):
        self.events.append("reencode")
        encoded = _real_batch(batch.global_step, encoded=True)
        return _Ready([encoded])

    def recv_rollout_trajectories(self, *args, **kwargs):
        self.events.append("actor_recv_wm_only")
        return _Ready([{"actor/received_shards": 1.0}])

    def compute_advantages_and_returns(self):
        self.events.append("advantages")
        return _Ready([{"actor/trajectory_count": 1.0}])

    def run_training(self):
        self.events.append("ppo")
        return _Ready([{"actor/ppo_updates": 1.0}])


class _Rollout:
    workers = (object(),)

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls = 0

    def set_global_step(self, step: int):
        return _Ready([None])

    def sync_model_from_actor(self, key: str):
        self.events.append("policy_pull")
        return _Ready([{"sync/rollout_policy_updated": 1.0}])

    def generate(self, *args):
        self.calls += 1
        phase = "real_generate" if self.calls == 1 else "wm_generate"
        self.events.append(phase)
        return _Ready([{"rollout/generated": 1.0}])


class _Env:
    def __init__(self, role: str, events: list[str]) -> None:
        self.role = role
        self.events = events

    def set_global_step(self, step: int):
        return _Ready([None])

    def begin_step_local_real_collection(self, global_step: int):
        assert self.role == "real"
        self.events.append("real_step_reset")
        return _Ready(
            [
                {
                    "env/real_env/discarded_partial_episodes": 0.0,
                    "env/real_env/discarded_partial_transitions": 0.0,
                }
            ]
        )

    def configure_progress(self, *args, **kwargs):
        return _Ready([None])

    def interact(self, *args):
        self.events.append(f"{self.role}_collect")
        if self.role == "real":
            metrics = {
                "env/real_env/episodes_completed": 1.0,
                "env/real_env/episodes_successful": 1.0,
                "env/trajectory_shards": 0.0,
            }
        else:
            metrics = {
                "env/wm_env/trajectory_shards": 1.0,
                "env/trajectory_shards": 1.0,
            }
        return _Ready([metrics])

    def drain_real_trajectories(self, global_step: int):
        self.events.append("drain")
        return _Ready([_real_batch(global_step)])

    def load_component_states(self, states, version: int):
        self.events.append("wm_cls_sync")
        assert sorted(states) == [
            "classifier",
            "classifier_threshold",
            "world_model",
        ]
        return _Ready([{"sync/load_component_states_s": 0.1}])

    def refresh_wm_initial_conditions(self):
        self.events.append("wm_refresh")
        return _Ready([{"env/wm_env/initial_conditions_refreshed": 1.0}])


class _Replay:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def set_policy_version(self, version: int):
        return _Ready([None])

    def replace_real_trajectories(self, batch: RealTrajectoryBatch):
        self.events.append("replay_replace")
        assert batch.trajectories[0].transitions[0]["encoder_version"] == 1
        return _Ready([{"replay_buffer/step_local_trajectories": 1.0}])

    def size(self):
        return _Ready([1])

    def num_transitions(self):
        return _Ready([1])


class _Learner:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def update_current_step(self, phase: str, num_steps: int, patience: int):
        self.events.append("wm_cls_update")
        assert (phase, num_steps, patience) == ("cotrain", 4, 2)
        return _Ready([{"learner/updates": 4.0, "cls/updated": 1.0}])

    def state_dicts(self):
        return _Ready(
            [
                {
                    "world_model": {"wm": 1},
                    "classifier": {"cls": 2},
                    "classifier_threshold": 0.4,
                }
            ]
        )


class _Channel:
    def __init__(self) -> None:
        self.puts: list[tuple[str, object]] = []

    def put(self, value, *, key: str):
        self.puts.append((str(key), value))

    def qsize(self, *, key: str | None = None):
        return 1 if key == "wm_env" else 0


def test_staged_global_step_has_explicit_real_model_imagination_barriers(monkeypatch) -> None:
    events: list[str] = []
    runner = CotrainRunner(_cfg())
    progress_names: list[str] = []
    runner.console_progress = (
        lambda _done, _total, desc, **_kwargs: progress_names.append(str(desc))
    )
    actor = _Actor(events)
    rollout = _Rollout(events)
    real_env = _Env("real", events)
    wm_env = _Env("wm", events)
    replay = _Replay(events)
    env_channel = _Channel()
    monkeypatch.setattr(
        "dreamervla.runners.cotrain_runner._share_ray_value",
        lambda value, *, cluster: value,
    )
    groups = {
        "ActorGroup": actor,
        "RolloutGroup": rollout,
        "LearnerGroup": _Learner(events),
        "RealEnvGroup": real_env,
        "WMEnvGroup": wm_env,
        "ReplayGroup": replay,
        "cluster": object(),
        "env_channel": env_channel,
        "actor_channel": _Channel(),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    metrics = runner._run_global_step(groups, global_step=1)

    ordered = [
        "real_step_reset",
        "real_collect",
        "real_generate",
        "drain",
        "encoder_sft",
        "reencode",
        "replay_replace",
        "wm_cls_update",
        "wm_cls_sync",
        "wm_refresh",
        "wm_collect",
        "wm_generate",
        "actor_recv_wm_only",
        "advantages",
        "ppo",
    ]
    positions = [events.index(event) for event in ordered]
    assert positions == sorted(positions)
    assert events.count("actor_recv_wm_only") == 1
    assert [value for _key, value in env_channel.puts].count(
        StopMsg(reason="global_step_complete")
    ) == 2
    assert metrics["replay_buffer/step_local_trajectories"] == 1.0
    assert metrics["actor/ppo_updates"] == 1.0
    assert list(dict.fromkeys(progress_names)) == [
        "cotrain-real-rollout/00000001",
        "cotrain-vla-real-sft/00000001",
        "cotrain-wmcls-training/00000001",
        "cotrain-imagined-rollout/00000001",
        "cotrain-vla-ppo/00000001",
    ]
