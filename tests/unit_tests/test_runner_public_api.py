from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf


def _compose_experiment(name: str, extra_overrides: list[str] | None = None):
    overrides = [f"experiment={name}"]
    if extra_overrides is not None:
        overrides.extend(extra_overrides)
    return compose(config_name="train", overrides=overrides)


def test_active_experiments_use_expected_run_roots() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    experiment_dir = config_dir / "experiment"
    experiments = sorted(path.stem for path in experiment_dir.glob("*.yaml"))

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for experiment in experiments:
            cfg = _compose_experiment(experiment)
            OmegaConf.resolve(cfg)
            out_dir = Path(str(cfg.training.out_dir))
            assert str(cfg.run.name) == experiment
            if str(cfg._target_) == "dreamervla.runners.LIBEROVLAEvaluationRunner":
                expected_eval_dir = (
                    Path(str(cfg.run.output_root)) / "eval" / str(cfg.eval.task_suite_name)
                )
                assert out_dir == expected_eval_dir
            else:
                assert out_dir.parent.name == experiment
                assert out_dir.name == str(cfg.run.timestamp)


def test_active_configs_target_route_specific_runner_classes() -> None:
    expected = {
        "eval_libero_vla": "dreamervla.runners.LIBEROVLAEvaluationRunner",
        "wm_full_dataset_train": "dreamervla.runners.WorldModelTrainingRunner",
        "wmpo_token_classifier_openvla_onetraj_libero_goal_h1": (
            "dreamervla.runners.SuccessClassifierTrainingRunner"
        ),
        "openvla_libero": "dreamervla.runners.DreamerRunner",
    }

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for config_name, target in expected.items():
            cfg = _compose_experiment(config_name)
            assert cfg._target_ == target
            assert "workspace" not in cfg
            cls = get_class(target)
            assert cls.__name__ == target.rsplit(".", 1)[-1]


def test_train_config_exposes_tensorboard_and_wandb_logger_routes() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        default_cfg = _compose_experiment("collect_rollouts")
        wandb_cfg = _compose_experiment(
            "collect_rollouts",
            extra_overrides=["logger=wandb"],
        )

    assert default_cfg.runner.logger.project_name == "dreamervla"
    assert default_cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert default_cfg.runner.logger.log_path == default_cfg.training.out_dir
    assert default_cfg.runner.logger.wandb_mode == "online"

    assert wandb_cfg.runner.logger.project_name == "dreamervla"
    assert wandb_cfg.runner.logger.logger_backends == ["wandb"]
    assert wandb_cfg.runner.logger.log_path == wandb_cfg.training.out_dir
    assert wandb_cfg.runner.logger.wandb_mode == "online"


def test_active_experiments_default_to_online_wandb() -> None:
    """Every public stage must publish metrics through the default W&B route."""

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    experiment_dir = config_dir / "experiment"
    experiments = sorted(path.stem for path in experiment_dir.glob("*.yaml"))

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for experiment in experiments:
            cfg = _compose_experiment(experiment)
            assert "wandb" in cfg.runner.logger.logger_backends, experiment
            assert cfg.runner.logger.wandb_mode == "online", experiment


def test_train_config_requires_explicit_experiment() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train")
    assert OmegaConf.select(cfg, "_target_", default=None) is None


def test_all_configs_compose_and_resolve_route_specific_runner_targets() -> None:
    import dreamervla.runners as runners

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    config_names = sorted(
        str(path.relative_to(config_dir).with_suffix(""))
        for path in config_dir.rglob("*.yaml")
        if "experiment" not in path.relative_to(config_dir).parts
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for config_name in config_names:
            cfg = compose(config_name=config_name)
            target = cfg.get("_target_")
            if target is not None:
                cls = get_class(str(target))
                assert cls.__module__.startswith("dreamervla.runners.")
                assert str(target).rsplit(".", 1)[-1] in runners.PUBLIC_RUNNERS
                assert "workspace" not in cfg
        for experiment_name in sorted(
            path.stem for path in (config_dir / "experiment").glob("*.yaml")
        ):
            cfg = _compose_experiment(experiment_name)
            target = cfg.get("_target_")
            assert target is not None
            cls = get_class(str(target))
            assert cls.__module__.startswith("dreamervla.runners.")
            assert str(target).rsplit(".", 1)[-1] in runners.PUBLIC_RUNNERS
            assert "workspace" not in cfg
