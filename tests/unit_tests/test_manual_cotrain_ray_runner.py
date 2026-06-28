from __future__ import annotations

import torch
from omegaconf import OmegaConf

import dreamervla.runners as runners
from dreamervla.workers.cotrain.messages import StopMsg


class _Ready:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value


def _cfg(
    ngpu: int = 2,
    *,
    out_dir: str = "/tmp/dvla-manual-cotrain-test",
    checkpoint_every: int = 0,
):
    return OmegaConf.create(
        {
            "_target_": "dreamervla.runners.ManualCotrainRayRunner",
            "seed": 7,
            "training": {"out_dir": out_dir, "seed": 7},
            "logger": {"logger_backends": []},
            "cluster": {"num_nodes": 1, "num_gpus": ngpu},
            "manual_cotrain": {
                "ngpu": ngpu,
                "global_steps": 1,
                "learner_update_step": 1,
                "checkpoint_every": checkpoint_every,
                "rollout_epoch": 1,
                "max_steps_per_rollout_epoch": 2,
                "num_action_chunks": 2,
                "envs_per_worker": 1,
                "sync_every": 1,
            },
            "actor": {"train_cfg": {"algorithm_cfg": {"group_size": 2}}},
        }
    )


def test_runner_plans_manual_notes_groups() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=5))
    plan = runner._placement_plan()

    assert [spec.role for spec in plan.env_specs] == [
        "real_env",
        "wm_env",
        "wm_env",
        "wm_env",
        "wm_env",
    ]
    assert len(plan.actor_specs) == 4
    assert len(plan.rollout_specs) == 5
    assert plan.learner_spec.gpu_ids == [0]


def test_runner_loop_order_names_actor_before_learner_update() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=2))
    order = runner._global_step_operation_names()

    assert order[:4] == [
        "set_global_step",
        "actor_to_rollout_sync",
        "env_interact_and_rollout_generate",
        "actor_recv_trajectories",
    ]
    assert "actor_run_training" in order
    assert "learner_update_wm_classifier" in order


def test_runner_builds_group_names_from_target_topology() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=3))
    names = runner._target_group_names()

    assert names == ["LearnerGroup", "ActorGroup", "RolloutGroup", "EnvGroup"]


def _compose_train_config(*overrides: str):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from dreamervla.config_resolvers import register_dreamervla_resolvers

    register_dreamervla_resolvers()
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="train",
            overrides=list(overrides),
        )


def test_manual_cotrain_oft_backbone_experiment_composes() -> None:
    cfg = _compose_train_config(
        "experiment=manual_cotrain_ray_oft_backbone_latent",
        "task=openvla_onetraj_coldstart_libero",
    )

    assert cfg._target_ == "dreamervla.runners.ManualCotrainRayRunner"
    assert cfg.learner.train_cfg.mode == "wm_classifier_only"
    assert cfg.algorithm.group_size == 8
    assert cfg.algorithm.rollout_epoch == 16
    assert cfg.actor.train_cfg.algorithm_cfg.clip_ratio_low == 0.2
    assert cfg.actor.train_cfg.algorithm_cfg.clip_ratio_high == 0.28


def test_manual_cotrain_oft_wm_env_num_envs_tracks_envs_per_worker() -> None:
    cfg = _compose_train_config(
        "experiment=manual_cotrain_ray_oft_backbone_latent",
        "task=openvla_onetraj_coldstart_libero",
    )

    assert cfg.env.wm.cfg.kwargs.num_envs == cfg.manual_cotrain.envs_per_worker
    assert cfg.env.wm.cfg.kwargs.device == "cuda"


def test_manual_cotrain_oft_real_rollout_uses_oft_encoder_and_action_postprocess() -> None:
    cfg = _compose_train_config(
        "experiment=manual_cotrain_ray_oft_backbone_latent",
        "task=openvla_onetraj_coldstart_libero",
    )

    assert cfg.rollout.encoder_cfg.target.endswith("oft_rollout:OFTRolloutBundle")
    assert cfg.rollout.encoder_cfg.kwargs.unnorm_key == cfg.task.openvla_oft.dataset_statistics_key
    assert cfg.rollout.encoder_cfg.kwargs.image_keys == cfg.task.image_keys
    assert cfg.rollout.encoder_cfg.kwargs.history == cfg.task.openvla_oft.input_tokens.expected_history
    assert (
        cfg.rollout.encoder_cfg.kwargs.obs_hidden_source
        == cfg.task.openvla_oft.input_tokens.expected_obs_hidden_source
    )
    assert cfg.env.real.cfg.action_postprocess == "openvla_oft"


def test_manual_cotrain_tiny_wm_env_num_envs_tracks_envs_per_worker_and_disables_loggers() -> None:
    cfg = _compose_train_config("experiment=manual_cotrain_ray_tiny")

    assert cfg.env.wm.cfg.kwargs.num_envs == cfg.manual_cotrain.envs_per_worker
    assert list(cfg.runner.logger.logger_backends) == []


def test_manual_runner_loads_component_init_ckpt(tmp_path) -> None:
    ckpt = tmp_path / "warmup.ckpt"
    torch.save(
        {
            "state_dicts": {
                "world_model": {"wm.weight": torch.ones(1)},
                "classifier": {"cls.weight": torch.full((1,), 2.0)},
                "policy": {"unused": torch.full((1,), 3.0)},
            }
        },
        ckpt,
    )
    cfg = _cfg(ngpu=0)
    cfg.learner = {
        "init_ckpt": {
            "path": str(ckpt),
            "components": ["world_model", "classifier"],
        }
    }

    runner = runners.ManualCotrainRayRunner(cfg)
    loaded = runner._load_init_ckpt("learner.init_ckpt")

    assert sorted(loaded) == ["classifier", "world_model"]
    assert torch.equal(loaded["world_model"]["wm.weight"], torch.ones(1))
    assert torch.equal(
        loaded["classifier"]["cls.weight"],
        torch.full((1,), 2.0),
    )


class _FakeActorGroup:
    def __init__(self) -> None:
        self.workers = ["actor0", "actor1"]
        self.selected: tuple[int, ...] | None = None
        self.sync_calls: list[tuple[tuple[int, ...] | None, str, int]] = []
        self.state_dict_calls: list[tuple[int, ...] | None] = []
        self.loaded_shards: list[object] = []

    def execute_on(self, *ranks: int):
        self.selected = tuple(int(rank) for rank in ranks)
        return self

    def set_global_step(self, global_step: int):
        self.global_step = int(global_step)
        return _Ready([None for _ in self.workers])

    def sync_model_to_rollout(self, key: str, version: int):
        self.sync_calls.append((self.selected, str(key), int(version)))
        self.selected = None
        return _Ready([{"sync/policy_version": float(version)}])

    def load_trajectory_shards(self, shards: list[object]):
        self.loaded_shards = list(shards)
        return _Ready([None for _ in self.workers])

    def compute_advantages_and_returns(self):
        return _Ready([{"actor/trajectory_count": float(len(self.loaded_shards))}])

    def run_training(self):
        return _Ready([{"actor/ppo_updates": 1.0}])

    def state_dict(self):
        self.state_dict_calls.append(self.selected)
        self.selected = None
        return _Ready([{"policy.weight": torch.ones(1)}])


class _FakeRolloutGroup:
    def __init__(self) -> None:
        self.workers = ["rollout0", "rollout1"]
        self.pulled: list[tuple[str, int | None]] = []
        self.generate_call: tuple[str, str, int] | None = None

    def set_global_step(self, global_step: int):
        self.global_step = int(global_step)
        return _Ready([None for _ in self.workers])

    def sync_model_from_actor(self, key: str, local_version: int | None = None):
        self.pulled.append((str(key), None if local_version is None else int(local_version)))
        return _Ready([None for _ in self.workers])

    def generate(
        self,
        env_channel_name: str,
        rollout_channel_name: str,
        num_slots: int = 1,
    ):
        self.generate_call = (
            str(env_channel_name),
            str(rollout_channel_name),
            int(num_slots),
        )
        return _Ready(
            [
                {"rollout/generated": 2.0},
                {"rollout/generated": 3.0},
            ]
        )


class _FakeEnvGroup:
    def __init__(self, metrics):
        self.metrics = metrics
        self.wm_versions: list[int] = []
        self.classifier_versions: list[int] = []
        self.world_model_states: list[dict] = []
        self.classifier_states: list[dict] = []

    def interact(self, env_channel_name: str, rollout_channel_name: str, actor_channel_name: str):
        del env_channel_name, rollout_channel_name, actor_channel_name
        return _Ready(self.metrics)

    def load_world_model_state(self, state_dict, version: int):
        self.world_model_states.append(dict(state_dict))
        self.wm_versions.append(int(version))
        return _Ready([None])

    def load_classifier_state(self, state_dict, version: int):
        self.classifier_states.append(dict(state_dict))
        self.classifier_versions.append(int(version))
        return _Ready([None])


class _FakeLearnerGroup:
    def __init__(self) -> None:
        self.synced: list[tuple[str, int]] = []

    def update(self, phase: str, num_steps: int):
        self.update_call = (str(phase), int(num_steps))
        return _Ready([{"learner/updates": 1.0}])

    def sync_weights(self, what: str, version: int):
        self.synced.append((str(what), int(version)))
        return _Ready([None])

    def state_dicts(self):
        return _Ready(
            [
                {
                    "world_model": {"wm": 1},
                    "classifier": {"cls": 2},
                    "policy": {"unused": 3},
                }
            ]
        )


class _FakeChannel:
    def __init__(self, items: list[object] | None = None) -> None:
        self.items = list(items or [])
        self.puts: list[tuple[str, object]] = []

    def get(self, *, key: str = "default"):
        del key
        return self.items.pop(0)

    def put(self, value, *, key: str = "default"):
        self.puts.append((str(key), value))


class _FakeReplayGroup:
    def __init__(self) -> None:
        self.policy_versions: list[int] = []

    def set_policy_version(self, version: int):
        self.policy_versions.append(int(version))
        return _Ready([None])

    def size(self):
        return _Ready([3])

    def num_transitions(self):
        return _Ready([7])


def test_run_global_step_syncs_actor_policy_and_wm_env_states() -> None:
    runner = runners.ManualCotrainRayRunner(_cfg(ngpu=2))
    actor = _FakeActorGroup()
    rollout = _FakeRolloutGroup()
    learner = _FakeLearnerGroup()
    wm_env = _FakeEnvGroup([{"env/trajectory_shards": 2.0, "env/steps": 4.0}])
    groups = {
        "ActorGroup": actor,
        "RolloutGroup": rollout,
        "LearnerGroup": learner,
        "RealEnvGroup": _FakeEnvGroup(
            {"env/trajectory_shards": 1.0, "env/steps": 2.0}
        ),
        "WMEnvGroup": wm_env,
        "ReplayGroup": _FakeReplayGroup(),
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["real-shard", "wm-shard-0", "wm-shard-1"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    metrics = runner._run_global_step(groups, global_step=1)

    assert actor.sync_calls == [(None, "policy", 1)]
    assert rollout.pulled == [("policy", None)]
    assert rollout.generate_call == ("env", "rollout", 1)
    assert [key for key, _ in groups["env_channel"].puts] == ["0:0", "1:0"]
    assert all(isinstance(value, StopMsg) for _, value in groups["env_channel"].puts)
    assert actor.loaded_shards == ["real-shard", "wm-shard-0", "wm-shard-1"]
    assert learner.synced == [("world_model", 1), ("classifier", 1)]
    assert wm_env.wm_versions == [1]
    assert wm_env.classifier_versions == [1]
    assert wm_env.world_model_states == [{"wm": 1}]
    assert wm_env.classifier_states == [{"cls": 2}]
    assert metrics["env/trajectory_shards"] == 3.0
    assert metrics["env/steps"] == 6.0
    assert metrics["rollout/generated"] == 5.0
    assert metrics["replay_buffer/size"] == 3.0
    assert metrics["replay_buffer/transitions"] == 7.0


def test_run_global_step_writes_manual_checkpoint_when_enabled(tmp_path) -> None:
    runner = runners.ManualCotrainRayRunner(
        _cfg(ngpu=2, out_dir=str(tmp_path), checkpoint_every=1)
    )
    actor = _FakeActorGroup()
    groups = {
        "ActorGroup": actor,
        "RolloutGroup": _FakeRolloutGroup(),
        "LearnerGroup": _FakeLearnerGroup(),
        "RealEnvGroup": _FakeEnvGroup({"env/trajectory_shards": 1.0, "env/steps": 2.0}),
        "WMEnvGroup": None,
        "ReplayGroup": None,
        "env_channel": _FakeChannel(),
        "actor_channel": _FakeChannel(["real-shard"]),
        "env_channel_name": "env",
        "rollout_channel_name": "rollout",
        "actor_channel_name": "actor",
    }

    runner._run_global_step(groups, global_step=1)

    ckpt = (
        tmp_path
        / "checkpoints"
        / "manual_cotrain_step_1"
        / "manual_cotrain.ckpt"
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert actor.state_dict_calls == [None]
    assert payload["global_step"] == 1
    assert sorted(payload["state_dicts"]) == ["classifier", "policy", "world_model"]
    assert torch.equal(
        payload["state_dicts"]["policy"]["policy.weight"],
        torch.ones(1),
    )
