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


def test_inference_worker_can_disable_obs_embedding_sidecar():
    from dreamervla.workers.inference.inference_worker import InferenceWorker

    worker = InferenceWorker(
        {
            "encoder": {
                "target": "dreamervla.workers.inference._test_models:TinyEncoder"
            },
            "world_model": {
                "target": "dreamervla.workers.actor._test_models:TinyLumosWorldModel",
                "kwargs": {"hidden_dim": 4, "action_dim": 7},
            },
            "policy": {
                "target": "dreamervla.workers.actor._test_models:TinyLumosPolicy",
                "kwargs": {"hidden_dim": 4, "action_dim": 7, "chunk_size": 1},
            },
            "device": "cpu",
            "emit_hidden_sidecar": False,
        },
        {},
        num_envs=1,
    )
    worker.init()

    out = worker.forward_batch([{"step": 0, "env_id": 0, "is_first": True}], [0])

    assert "actions" in out
    assert "obs_embedding" not in out
