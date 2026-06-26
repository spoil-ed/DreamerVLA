from pathlib import Path

from hydra import compose, initialize_config_dir


def test_world_model_env_tiny_experiment_composes():
    config_dir = str(Path("configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=online_cotrain_ray_world_model_env_tiny",
                "logger=tensorboard",
            ],
        )

    assert cfg._target_.endswith("OnlineCotrainRayRunner")
    assert cfg.runner_name == "online_cotrain_ray"
    assert cfg.sync.world_model_env is True
    target = str(cfg.env.cfg.get("target", cfg.env.cfg.get("_target_", "")))
    assert "world_model" in target
    assert cfg.inference.cfg.emit_hidden_sidecar is False
