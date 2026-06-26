from __future__ import annotations


def test_task_conditioning_config_is_component_visible():
    from pathlib import Path

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "configs"
        / "dreamervla"
        / "online_cotrain_pipeline_libero_goal.yaml"
    )

    assert OmegaConf.select(cfg, "world_model.task_conditioning.enabled") is False
    assert OmegaConf.select(cfg, "world_model.task_conditioning.num_tasks") == 10
    assert OmegaConf.select(cfg, "classifier.task_conditioning.enabled") is False
    assert OmegaConf.select(cfg, "classifier.task_conditioning.embedding_dim") == 64


def test_pipeline_config_uses_full_replay_epoch_warmup():
    from pathlib import Path

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "configs"
        / "dreamervla"
        / "online_cotrain_pipeline_libero_goal.yaml"
    )

    assert OmegaConf.select(cfg, "training.warmup_replay_epochs") == 1
    assert OmegaConf.select(cfg, "training.warmup_replay_max_steps") == 0
    assert OmegaConf.select(cfg, "training.warmup_checkpoint_every") == 0
    assert OmegaConf.select(cfg, "training.replay_warmup_log_every") == 1


def test_pipeline_smoke_config_uses_fixed_step_warmup():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=online_cotrain_pipeline_oft_action_hidden_smoke"],
        )

    assert OmegaConf.select(cfg, "training.debug") is True
    assert OmegaConf.select(cfg, "training.warmup_replay_epochs") == 0
    assert OmegaConf.select(cfg, "training.wm_warmup_steps") == 1200


def test_oft_backbone_pipeline_uses_proprio_language_wm_profile():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=online_cotrain_pipeline_oft_backbone_latent"],
        )

    wm = cfg.world_model
    assert cfg.latent_type == "backbone_latent"
    assert cfg.online_rollout.sequence_length == 12
    assert wm.model_dim == 4148
    assert wm.proprio_dim == 8
    assert wm.proprio_emb_dim == 10
    assert wm.num_proprio_repeat == 1
    assert wm.lang_dim == 4096
    assert wm.lang_emb_dim == 32
    assert wm.num_lang_repeat == 1
    assert wm.cosine_loss_scale == 0.0
    assert wm.chunk_rollout_chunks == 1
    assert wm.chunk_rollout_loss_scale == 0.0


def test_validate_cfg_warmup(tmp_path):
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.config import validate_cfg
    base = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
        "training": {"wm_warmup_steps": 10, "classifier_warmup_steps": 10},
    })
    validate_cfg(base)  # existing dir -> ok
    bad = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path / "nope"), "hidden_dir": str(tmp_path)},
        "training": {"wm_warmup_steps": 10, "classifier_warmup_steps": 10},
    })
    with pytest.raises(Exception, match="offline_warmup.data_dir"):
        validate_cfg(bad)
    neg = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
        "training": {"wm_warmup_steps": -1, "classifier_warmup_steps": 10},
    })
    with pytest.raises(Exception, match="wm_warmup_steps"):
        validate_cfg(neg)
    neg_replay_max = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
        "training": {
            "wm_warmup_steps": 10,
            "classifier_warmup_steps": 10,
            "warmup_replay_epochs": 1,
            "warmup_replay_max_steps": -1,
        },
    })
    with pytest.raises(Exception, match="warmup_replay_max_steps"):
        validate_cfg(neg_replay_max)
    neg_ckpt_every = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
        "training": {
            "wm_warmup_steps": 10,
            "classifier_warmup_steps": 10,
            "warmup_checkpoint_every": -1,
        },
    })
    with pytest.raises(Exception, match="warmup_checkpoint_every"):
        validate_cfg(neg_ckpt_every)
    bad_log_every = OmegaConf.create({
        "_target_": "dreamervla.runners.OnlineCotrainPipelineRunner",
        "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
        "training": {
            "wm_warmup_steps": 10,
            "classifier_warmup_steps": 10,
            "replay_warmup_log_every": 0,
        },
    })
    with pytest.raises(Exception, match="replay_warmup_log_every"):
        validate_cfg(bad_log_every)
