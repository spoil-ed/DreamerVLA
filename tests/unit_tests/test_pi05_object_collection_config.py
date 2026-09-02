from __future__ import annotations

from pathlib import Path


def test_pi05_object_sft_collection_covers_all_tasks_and_outcomes() -> None:
    from hydra import compose, initialize_config_dir

    from dreamervla.config import validate_cfg
    from dreamervla.runners import RolloutCollectionRunner

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=collect_rollouts_pi05_object",
                "collect.policy_ckpt_path=/tmp/pi05-object-one-traj-sft",
            ],
        )

    validate_cfg(cfg)
    plan = RolloutCollectionRunner(cfg).build_vla_worker_plan()

    assert cfg.task.suite == "libero_object"
    assert str(cfg.task.collected_reward_dir).endswith("pi05_libero_object/reward")
    assert cfg.collect.task_ids == "all"
    assert cfg.collect.episodes_per_task == 200
    assert cfg.collect.num_inference_workers == 8
    assert cfg.collect.num_dump_workers == 8
    assert cfg.collect.demos_per_shard == 1
    assert cfg.env.num_workers == 32
    assert cfg.rollout.target_episodes == 2000
    assert cfg.launch.ngpu == 8
    assert cfg.launch.required_target_values == ["collect.policy_ckpt_path"]
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.runner.logger.wandb_mode == "online"
    assert plan["collect"]["task_suite_name"] == "libero_object"
    assert plan["collect"]["episodes_per_task"] == 200
    assert plan["inference"]["action_steps"] == 10
    assert plan["inference"]["decoder"]["kwargs"]["action_chunk"] == 50
    assert plan["dump"]["preprocess_config"]["action_steps"] == 10
    assert plan["inference"]["decoder"]["kwargs"]["model_path"].endswith(
        "data/checkpoints/pi05_base"
    )
    assert plan["inference"]["decoder"]["kwargs"]["assets_path"].endswith(
        "data/checkpoints/RLinf-Pi05-LIBERO-SFT"
    )
    assert (
        plan["inference"]["decoder"]["kwargs"]["model_path"]
        != plan["inference"]["decoder"]["kwargs"]["assets_path"]
    )
    assert (
        plan["inference"]["decoder"]["kwargs"]["policy_ckpt_path"]
        == "/tmp/pi05-object-one-traj-sft"
    )
    assert plan["inference"]["emit_hidden_sidecar"] is False
    assert plan["dump"]["write_hidden_sidecar"] is False


def test_pi05_object_sft_checkpoint_can_come_from_environment(monkeypatch) -> None:
    from hydra import compose, initialize_config_dir

    checkpoint = "/tmp/pi05-object-one-traj-sft/checkpoints/latest.ckpt"
    monkeypatch.setenv("PI05_SFT_CKPT", checkpoint)
    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=collect_rollouts_pi05_object"],
        )

    assert cfg.collect.policy_ckpt_path == checkpoint


def test_pi05_object_sinfra_profile_owns_remote_runtime_configuration(
    monkeypatch,
) -> None:
    from dreamervla.launchers.train import build_launch

    for key in (
        "DVLA_DATA_ROOT",
        "RUN_ROOT",
        "LIBERO_CONFIG_PATH",
        "PI05_BASE_CKPT",
        "PI05_LIBERO_CKPT",
        "PI05_ASSETS_CKPT",
        "PI05_SFT_CKPT",
        "OPENPI_ROOT",
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)

    launch = build_launch(
        [
            "--config",
            "collect_rollouts_pi05_object",
            "profile=sinfra_pi05_object",
            "dry_run=true",
        ]
    )

    assert launch.cfg.profile.name == "sinfra_pi05_object"
    assert launch.cfg.run.output_root == "/jfs/oss-import/xinglei/pi05_outputs"
    assert launch.cfg.launch.write_libero_config is False
    assert launch.cfg.task.pi05.base_ckpt_path.endswith("/lerobot/pi05_base")
    assert launch.cfg.task.pi05.assets_path.endswith("/RLinf-Pi05-LIBERO-SFT")
    assert launch.cfg.task.pi05.action_horizon == 50
    assert launch.cfg.task.pi05.replan_steps == 10
    assert launch.cfg.collect.action_steps == 10
    assert str(launch.cfg.collect.policy_ckpt_path).endswith(
        "/pi05_libero_sft_one_episode_per_task/20260823_162557/checkpoints/latest.ckpt"
    )
    assert launch.cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert launch.cfg.runner.logger.wandb_mode == "online"
    assert launch.env["DVLA_DATA_ROOT"] == "/jfs/oss-import/xinglei/DreamerVLA/data"
    assert launch.env["PYTHONUNBUFFERED"] == "1"
    assert launch.env["JAX_PLATFORMS"] == "cpu"
    assert launch.env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert launch.env["MUJOCO_GL"] == "osmesa"
    assert launch.env["PYOPENGL_PLATFORM"] == "osmesa"
    assert launch.command[0] == "/jfs/oss-import/xinglei/DreamerVLA/.venv-pi05/bin/python"
    assert "profile=sinfra_pi05_object" in launch.command
