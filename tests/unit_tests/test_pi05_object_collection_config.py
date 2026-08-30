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
