from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.config_resolvers import register_dreamervla_resolvers


def _compose(experiment: str):
    register_dreamervla_resolvers()
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])
    OmegaConf.resolve(cfg)
    return cfg


def _compose_with_task(experiment: str, task: str):
    register_dreamervla_resolvers()
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", f"task={task}"],
        )
    OmegaConf.resolve(cfg)
    return cfg


def test_wm_upper_bound_reads_only_official_data() -> None:
    cfg = _compose("wm_official_upper_bound")

    assert cfg.pre_mainline.stage == "wm_upper_bound"
    assert cfg.offline_warmup.data_dir == cfg.task.hdf5_reward_dir
    assert cfg.offline_warmup.hidden_dir == cfg.task.openvla_oft.hidden_token_dir
    assert cfg.training.classifier_warmup_steps == 0
    assert cfg.online_rollout.total_env_steps == 0
    assert cfg.offline_warmup.require_reference_complete is False
    assert list(cfg.offline_warmup.required_task_ids) == list(range(10))
    assert cfg.dataloader.batch_size == 16
    assert cfg.optim.world_model.lr == 3.0e-5


def test_wm_upper_bound_profile_is_bounded_and_keeps_training_hyperparameters() -> None:
    cfg = _compose("wm_official_upper_bound_profile")

    assert cfg.pre_mainline.stage == "wm_upper_bound"
    assert cfg.offline_warmup.data_dir == cfg.task.hdf5_reward_dir
    assert cfg.offline_warmup.hidden_dir == cfg.task.openvla_oft.hidden_token_dir
    assert cfg.training.wm_warmup_steps == 64
    assert cfg.training.warmup_replay_epochs == 0
    assert cfg.training.wm_profile_steps == 32
    assert cfg.training.wm_prefetch_workers == 1
    assert cfg.training.replay_warmup_log_every == 10
    assert cfg.training.warmup_checkpoint_every == 0
    assert cfg.training.classifier_warmup_steps == 0
    assert cfg.online_rollout.total_env_steps == 0
    assert cfg.dataloader.batch_size == 16
    assert cfg.optim.world_model.lr == 3.0e-5


def test_classifier_upper_bound_reads_only_official_data() -> None:
    cfg = _compose("classifier_official_upper_bound")

    assert cfg.pre_mainline.stage == "classifier_upper_bound"
    assert cfg.data.success_dir_raw == cfg.task.hdf5_reward_dir
    assert cfg.data.success_dir_hidden == cfg.task.openvla_oft.hidden_token_dir
    assert cfg.data.failure_dir_raw is None
    assert cfg.data.failure_dir_hidden is None
    assert cfg.data.train_split == "train"
    assert cfg.data.val_split == "val"
    assert cfg.data.val_fraction == 0.2
    assert cfg.data.require_sidecar_contract is True
    assert cfg.data.require_reference_complete is False
    assert len(cfg.data.required_filenames) == 10
    assert cfg.training.episode_eval_enabled is False
    assert cfg.training.batch_size == 4
    assert cfg.training.lr == 3.0e-5


def test_frozen_models_rl_is_policy_only_and_reads_official_replay() -> None:
    cfg = _compose("dreamervla_frozen_models_rl")

    assert cfg.pre_mainline.stage == "frozen_models_rl"
    assert cfg._target_ == "dreamervla.runners.FrozenModelPolicyRunner"
    assert cfg.official_replay.data_dir == cfg.task.hdf5_reward_dir
    assert cfg.official_replay.hidden_dir == cfg.task.openvla_oft.hidden_token_dir
    assert list(cfg.official_replay.task_ids) == list(range(10))
    assert cfg.official_replay.capacity_mode == "total_sharded"
    assert cfg.official_replay.task_balanced is True
    assert cfg.official_replay.rank == 0
    assert cfg.official_replay.replay_sampling.enabled is False
    assert cfg.official_replay.require_reference_complete is False
    assert cfg.init.world_model_state_ckpt is None
    assert cfg.init.classifier_state_ckpt is None
    assert OmegaConf.select(cfg, "optim.world_model", default=None) is None
    assert OmegaConf.select(cfg, "optim.classifier", default=None) is None
    assert OmegaConf.select(cfg, "optim.critic", default=None) is None
    assert OmegaConf.select(cfg, "env", default=None) is None
    assert OmegaConf.select(cfg, "online_rollout", default=None) is None


def test_frozen_models_rl_ray_is_eight_gpu_policy_only_and_uses_official_replay() -> None:
    cfg = _compose("dreamervla_frozen_models_rl_ray")

    assert cfg.pre_mainline.stage == "frozen_models_rl_ray"
    assert cfg._target_ == "dreamervla.runners.ManualCotrainRayRunner"
    assert cfg.manual_cotrain.ngpu == 8
    assert cfg.cluster.num_nodes == 1
    assert cfg.cluster.num_gpus == 8
    assert cfg.manual_cotrain.real_env_enabled is False
    assert cfg.manual_cotrain.learner_updates_enabled is False
    assert cfg.manual_cotrain.real_env_workers == 0
    assert cfg.manual_cotrain.refresh_wm_initial_conditions_per_lease is True
    assert cfg.env.real is None
    assert cfg.learner is None
    assert list(cfg.replay.seed.task_ids) == list(range(10))
    assert list(cfg.replay.cfg.task_ids) == list(range(10))
    assert list(cfg.env.wm.cfg.bootstrap_task_ids) == list(range(10))
    assert cfg.env.wm.cfg.require_balanced_initial_conditions is True
    assert cfg.replay.seed.data_dir == cfg.task.hdf5_reward_dir
    assert cfg.replay.seed.hidden_dir == cfg.task.openvla_oft.hidden_token_dir
    assert cfg.replay.seed.max_episodes_per_task is None
    assert cfg.replay.seed.require_reference_complete is False
    assert set(cfg.actor.train_cfg.optimizers) == {"policy"}
    assert OmegaConf.to_container(
        cfg.actor.train_cfg.optimizers.policy,
        resolve=True,
    ) == {
        "name": cfg.optim.policy.name,
        "lr": cfg.optim.policy.lr,
        "betas": list(cfg.optim.policy.betas),
        "eps": cfg.optim.policy.eps,
        "weight_decay": cfg.optim.policy.weight_decay,
        "grad_clip_norm": cfg.optim.grad_clip_norm,
        "zero_grad_set_to_none": cfg.optim.zero_grad_set_to_none,
    }
    assert cfg.actor.train_cfg.fsdp.sync_module_states is True
    assert cfg.actor.train_cfg.fsdp.strategy == "fsdp"
    assert cfg.actor.train_cfg.fsdp.cpu_offload is False
    assert (
        cfg.actor.train_cfg.algorithm_cfg.update_micro_batch_starts
        == cfg.algorithm.lumos.update_micro_batch_starts
    )
    assert cfg.actor.train_cfg.algorithm_cfg.clip_log_ratio == cfg.algorithm.clip_log_ratio
    assert cfg.env.wm.cfg.bootstrap_group_size == cfg.manual_cotrain.wm_envs_per_worker
    assert cfg.env.wm.cfg.defer_initial_condition_bootstrap is True
    assert cfg.env.wm.cfg.bootstrap_group_size % cfg.algorithm.group_size == 0
    assert OmegaConf.to_container(
        cfg.env.wm.cfg.kwargs.world_model,
        resolve=True,
    ) == OmegaConf.to_container(cfg.world_model, resolve=True)
    assert OmegaConf.to_container(
        cfg.env.wm.cfg.kwargs.classifier,
        resolve=True,
    ) == OmegaConf.to_container(cfg.classifier, resolve=True)


def test_frozen_models_rl_ray_validates_with_explicit_component_checkpoints(
    tmp_path: Path,
) -> None:
    cfg = _compose("dreamervla_frozen_models_rl_ray")
    OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
    OmegaConf.update(
        cfg,
        "init.classifier_state_ckpt",
        str(tmp_path / "classifier.ckpt"),
    )

    validate_cfg(cfg)


def test_frozen_models_rl_ray_geometry_counts_all_eight_wm_workers(
    tmp_path: Path,
) -> None:
    cfg = _compose("dreamervla_frozen_models_rl_ray")
    OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
    OmegaConf.update(
        cfg,
        "init.classifier_state_ckpt",
        str(tmp_path / "classifier.ckpt"),
    )
    OmegaConf.update(cfg, "manual_cotrain.wm_rollout_target_trajectories", 112)

    with pytest.raises(ValueError, match="each WM worker"):
        validate_cfg(cfg)


def test_frozen_models_rl_ray_resume_requires_policy_checkpoint(tmp_path: Path) -> None:
    cfg = _compose("dreamervla_frozen_models_rl_ray")
    OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
    OmegaConf.update(
        cfg,
        "init.classifier_state_ckpt",
        str(tmp_path / "classifier.ckpt"),
    )
    OmegaConf.update(cfg, "training.resume", True)
    resume_dir = tmp_path / "run"
    resume_dir.mkdir()
    OmegaConf.update(cfg, "training.resume_dir", str(resume_dir))

    with pytest.raises(ValueError, match="resume.*policy checkpoint"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("manual_cotrain.real_env_enabled", True, "real environment"),
        ("manual_cotrain.learner_updates_enabled", True, "learner"),
        ("env.real", {"cfg": {"target": "forbidden.RealEnv"}}, "real environment"),
        ("learner", {"train_cfg": {"optimizers": {}}}, "learner"),
        (
            "cluster.component_placement",
            {"env": "0", "rollout": "0", "actor": "1-7"},
            "eight-GPU topology",
        ),
        (
            "actor.train_cfg.optimizers.world_model",
            {"lr": 1.0e-4},
            "policy optimizer",
        ),
        (
            "env.wm.cfg.bootstrap_group_size",
            8,
            "complete policy groups",
        ),
        (
            "manual_cotrain.refresh_wm_initial_conditions_per_lease",
            False,
            "refresh",
        ),
        (
            "env.wm.cfg.defer_initial_condition_bootstrap",
            False,
            "defer",
        ),
        (
            "actor.train_cfg.fsdp.sync_module_states",
            False,
            "sync_module_states",
        ),
        (
            "actor.train_cfg.fsdp.strategy",
            "none",
            "FSDP strategy",
        ),
        (
            "actor.train_cfg.fsdp.cpu_offload",
            True,
            "CPU offload",
        ),
        (
            "actor.train_cfg.algorithm_cfg.update_micro_batch_starts",
            1,
            "policy-update settings",
        ),
        (
            "env.wm.cfg.bootstrap_task_ids",
            [0],
            "all ten task IDs",
        ),
        (
            "env.wm.cfg.require_balanced_initial_conditions",
            False,
            "balanced initial conditions",
        ),
    ],
)
def test_frozen_models_rl_ray_rejects_trainable_or_real_components(
    tmp_path: Path,
    path: str,
    value: object,
    match: str,
) -> None:
    cfg = _compose("dreamervla_frozen_models_rl_ray")
    OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
    OmegaConf.update(
        cfg,
        "init.classifier_state_ckpt",
        str(tmp_path / "classifier.ckpt"),
    )
    OmegaConf.update(cfg, path, value, force_add=True)

    with pytest.raises(ValueError, match=match):
        validate_cfg(cfg)


def test_upper_bound_component_configs_match_frozen_rl_exactly() -> None:
    wm_cfg = _compose("wm_official_upper_bound")
    classifier_cfg = _compose("classifier_official_upper_bound")
    rl_cfg = _compose("dreamervla_frozen_models_rl")

    assert OmegaConf.to_container(wm_cfg.world_model, resolve=True) == OmegaConf.to_container(
        rl_cfg.world_model,
        resolve=True,
    )
    assert OmegaConf.to_container(
        classifier_cfg.classifier,
        resolve=True,
    ) == OmegaConf.to_container(rl_cfg.classifier, resolve=True)


def test_upper_bound_component_configs_match_frozen_ray_rl_exactly() -> None:
    wm_cfg = _compose("wm_official_upper_bound")
    classifier_cfg = _compose("classifier_official_upper_bound")
    rl_cfg = _compose("dreamervla_frozen_models_rl_ray")

    assert OmegaConf.to_container(wm_cfg.world_model, resolve=True) == OmegaConf.to_container(
        rl_cfg.world_model,
        resolve=True,
    )
    assert OmegaConf.to_container(
        classifier_cfg.classifier,
        resolve=True,
    ) == OmegaConf.to_container(rl_cfg.classifier, resolve=True)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("official_replay.data_dir", "/tmp/collected/reward", "official LIBERO"),
        ("official_replay.hidden_dir", "/tmp/collected/hidden", "official LIBERO"),
        ("optim.world_model", {"name": "adam", "lr": 1e-4}, "optimizer"),
        ("optim.world_model", None, "optimizer"),
        ("optim.classifier", {"name": "adam", "lr": 1e-4}, "optimizer"),
        ("env", {"_target_": "forbidden.RealEnv"}, "real environment"),
        ("env", None, "real environment"),
        ("online_rollout", {"total_env_steps": 1}, "real rollout"),
        ("algorithm.update_type", "LUMOS_DENSE_CHUNK", "requires classifier"),
        ("official_replay.task_balanced", False, "official replay"),
        ("official_replay.capacity_mode", "per_task", "official replay"),
        ("official_replay.replay_sampling.enabled", True, "official replay"),
    ],
)
def test_frozen_models_rl_rejects_non_policy_only_mutations(
    tmp_path: Path,
    path: str,
    value: object,
    match: str,
) -> None:
    cfg = _compose("dreamervla_frozen_models_rl")
    OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
    OmegaConf.update(
        cfg,
        "init.classifier_state_ckpt",
        str(tmp_path / "classifier.ckpt"),
    )
    OmegaConf.update(cfg, path, value, force_add=True)

    with pytest.raises(ValueError, match=match):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    "checkpoint_key",
    ["init.world_model_state_ckpt", "init.classifier_state_ckpt"],
)
def test_frozen_models_rl_requires_explicit_frozen_checkpoints(
    tmp_path: Path,
    checkpoint_key: str,
) -> None:
    cfg = _compose("dreamervla_frozen_models_rl")
    OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
    OmegaConf.update(
        cfg,
        "init.classifier_state_ckpt",
        str(tmp_path / "classifier.ckpt"),
    )
    OmegaConf.update(cfg, checkpoint_key, None)

    with pytest.raises(ValueError, match="explicit.*checkpoint"):
        validate_cfg(cfg)


def test_wm_upper_bound_requires_a_positive_training_budget() -> None:
    cfg = _compose("wm_official_upper_bound")
    OmegaConf.update(cfg, "training.wm_warmup_steps", 0)

    with pytest.raises(ValueError, match="wm_warmup_steps > 0"):
        validate_cfg(cfg)


def test_wm_upper_bound_rejects_debug_budget_rewrite() -> None:
    cfg = _compose("wm_official_upper_bound")
    OmegaConf.update(cfg, "training.debug", True)

    with pytest.raises(ValueError, match="forbids training.debug"):
        validate_cfg(cfg)


def test_wm_upper_bound_requires_all_official_task_ids() -> None:
    cfg = _compose("wm_official_upper_bound")
    OmegaConf.update(cfg, "offline_warmup.required_task_ids", [0])

    with pytest.raises(ValueError, match="all ten official task IDs"):
        validate_cfg(cfg)


def test_classifier_upper_bound_rejects_extra_failure_dataset(tmp_path: Path) -> None:
    cfg = _compose("classifier_official_upper_bound")
    OmegaConf.update(cfg, "data.failure_dir_raw", str(tmp_path / "failures"))
    OmegaConf.update(cfg, "data.failure_dir_hidden", str(tmp_path / "hidden"))

    with pytest.raises(ValueError, match="failure datasets"):
        validate_cfg(cfg)


def test_classifier_upper_bound_requires_complete_sidecar_validation() -> None:
    cfg = _compose("classifier_official_upper_bound")
    OmegaConf.update(cfg, "data.require_sidecar_contract", False)

    with pytest.raises(ValueError, match="complete official sidecar"):
        validate_cfg(cfg)


def test_classifier_upper_bound_requires_all_official_shards() -> None:
    cfg = _compose("classifier_official_upper_bound")
    OmegaConf.update(cfg, "data.required_filenames", ["one_demo.hdf5"])

    with pytest.raises(ValueError, match="all ten official reward shards"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("experiment", "config_path"),
    [
        ("wm_official_upper_bound", "offline_warmup.require_reference_complete"),
        (
            "classifier_official_upper_bound",
            "data.require_reference_complete",
        ),
        (
            "dreamervla_frozen_models_rl",
            "official_replay.require_reference_complete",
        ),
    ],
)
def test_official_routes_reject_rollout_completion_marker_requirement(
    tmp_path: Path,
    experiment: str,
    config_path: str,
) -> None:
    cfg = _compose(experiment)
    if experiment == "dreamervla_frozen_models_rl":
        OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
        OmegaConf.update(
            cfg,
            "init.classifier_state_ckpt",
            str(tmp_path / "classifier.ckpt"),
        )
    OmegaConf.update(cfg, config_path, True)

    with pytest.raises(ValueError, match="do not use rollout complete markers"):
        validate_cfg(cfg)


def test_frozen_models_rl_requires_complete_all_task_replay(tmp_path: Path) -> None:
    cfg = _compose("dreamervla_frozen_models_rl")
    OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
    OmegaConf.update(
        cfg,
        "init.classifier_state_ckpt",
        str(tmp_path / "classifier.ckpt"),
    )
    OmegaConf.update(cfg, "official_replay.task_ids", [0])

    with pytest.raises(ValueError, match="all ten task IDs"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    "experiment",
    [
        "wm_official_upper_bound",
        "classifier_official_upper_bound",
        "dreamervla_frozen_models_rl",
    ],
)
def test_pre_mainline_experiments_use_canonical_goal_sidecars(
    experiment: str,
) -> None:
    cfg = _compose_with_task(experiment, "openvla_onetraj_libero")

    assert Path(str(cfg.task.openvla_oft.hidden_token_dir)).name == (
        "no_noops_t_256_oft_hidden_token_vla_policy_h1"
    )
    assert cfg.task.openvla_oft.expected_obs_hidden_source == "hidden_token"
    assert (
        cfg.task.openvla_oft.hidden_token.expected_obs_hidden_source
        == "hidden_token"
    )
    assert cfg.task.openvla_oft.hidden_token.token_count == 256
    assert cfg.task.openvla_oft.hidden_token.token_dim == 4096


def test_pre_mainline_exposes_only_hidden_token_hydra_names() -> None:
    cfg = _compose("classifier_official_upper_bound")

    legacy_namespace = "input_" + "tokens"
    legacy_dir = "input_" + "token_dir"
    assert legacy_namespace not in cfg.task.openvla_oft
    assert legacy_dir not in cfg.task.openvla_oft
    assert cfg.task.openvla_oft.hidden_token.expected_obs_hidden_source == "hidden_token"
    assert cfg.task.openvla_oft.hidden_token.token_count == 256
    assert cfg.task.openvla_oft.hidden_token.token_dim == 4096
    assert Path(str(cfg.task.openvla_oft.hidden_token_dir)).name == (
        "no_noops_t_256_oft_hidden_token_vla_policy_h1"
    )


def test_classifier_official_recipe_enables_bounded_update_profiling_and_bf16() -> None:
    cfg = _compose("classifier_official_upper_bound")

    assert cfg.training.precision == "bf16"
    assert cfg.training.update_profile_steps == 8


@pytest.mark.parametrize(
    ("experiment", "raw_path", "hidden_path"),
    [
        (
            "wm_official_upper_bound",
            "offline_warmup.data_dir",
            "offline_warmup.hidden_dir",
        ),
        (
            "classifier_official_upper_bound",
            "data.success_dir_raw",
            "data.success_dir_hidden",
        ),
    ],
)
def test_upper_bound_routes_reject_nonofficial_data_roots(
    tmp_path: Path,
    experiment: str,
    raw_path: str,
    hidden_path: str,
) -> None:
    cfg = _compose(experiment)
    wrong_raw = tmp_path / "collected_rollouts" / "reward"
    wrong_hidden = tmp_path / "collected_rollouts" / "hidden"
    wrong_raw.mkdir(parents=True)
    wrong_hidden.mkdir(parents=True)
    OmegaConf.update(cfg, raw_path, str(wrong_raw))
    OmegaConf.update(cfg, hidden_path, str(wrong_hidden))

    with pytest.raises(ValueError, match="official LIBERO"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    "experiment",
    [
        "wm_official_upper_bound",
        "classifier_official_upper_bound",
        "dreamervla_frozen_models_rl",
    ],
)
def test_pre_mainline_rejects_joint_task_path_override(
    tmp_path: Path,
    experiment: str,
) -> None:
    cfg = _compose(experiment)
    fake_reward = tmp_path / "libero_goal_failures"
    fake_hidden = tmp_path / "libero_goal_hidden_token_failures"
    OmegaConf.update(cfg, "task.hdf5_reward_dir", str(fake_reward))
    OmegaConf.update(cfg, "task.openvla_oft.hidden_token_dir", str(fake_hidden))
    if experiment == "wm_official_upper_bound":
        OmegaConf.update(cfg, "offline_warmup.data_dir", str(fake_reward))
        OmegaConf.update(cfg, "offline_warmup.hidden_dir", str(fake_hidden))
    elif experiment == "classifier_official_upper_bound":
        OmegaConf.update(cfg, "data.success_dir_raw", str(fake_reward))
        OmegaConf.update(cfg, "data.success_dir_hidden", str(fake_hidden))
    else:
        OmegaConf.update(cfg, "official_replay.data_dir", str(fake_reward))
        OmegaConf.update(cfg, "official_replay.hidden_dir", str(fake_hidden))
        OmegaConf.update(cfg, "init.world_model_state_ckpt", str(tmp_path / "wm.ckpt"))
        OmegaConf.update(
            cfg,
            "init.classifier_state_ckpt",
            str(tmp_path / "classifier.ckpt"),
        )

    with pytest.raises(ValueError, match="canonical official LIBERO"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    "experiment",
    [
        "wm_official_upper_bound",
        "classifier_official_upper_bound",
        "dreamervla_frozen_models_rl",
    ],
)
def test_pre_mainline_is_goal_only(experiment: str) -> None:
    cfg = _compose_with_task(experiment, "openvla_onetraj_libero_object")
    if experiment == "dreamervla_frozen_models_rl":
        OmegaConf.update(cfg, "init.world_model_state_ckpt", "/tmp/wm.ckpt")
        OmegaConf.update(cfg, "init.classifier_state_ckpt", "/tmp/classifier.ckpt")

    with pytest.raises(ValueError, match="only task.suite=libero_goal"):
        validate_cfg(cfg)
