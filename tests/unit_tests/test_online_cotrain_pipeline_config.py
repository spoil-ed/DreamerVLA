from __future__ import annotations


def test_oft_backbone_pipeline_uses_traj1_proprio_language_wm_profile():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=openvla_onetraj_libero_cotrain_noray"],
        )

    wm = cfg.world_model
    classifier = cfg.classifier
    assert cfg.latent_type == "backbone_latent"
    assert cfg.training.wm_warmup_steps == 20000
    assert cfg.training.warmup_replay_epochs == 10
    assert cfg.training.warmup_checkpoint_every == 500
    assert cfg.training.warmup_topk_k == 3
    assert cfg.training.wm_profile_steps == -1
    assert cfg.dataloader.batch_size == 16
    assert cfg.optim.world_model.lr == 3.0e-5
    assert cfg.online_rollout.sequence_length == 36
    assert wm.model_dim == 4148
    assert wm.proprio_dim == 8
    assert wm.proprio_emb_dim == 10
    assert wm.num_proprio_repeat == 1
    assert wm.lang_dim == 4096
    assert wm.lang_emb_dim == 32
    assert wm.num_lang_repeat == 1
    assert wm.action_emb_dim == 10
    assert wm.model_dim == (
        wm.token_dim
        + wm.proprio_emb_dim * wm.num_proprio_repeat
        + wm.lang_emb_dim * wm.num_lang_repeat
        + wm.action_emb_dim * wm.num_action_repeat
    )
    assert wm.cosine_loss_scale == 0.0
    assert wm.chunk_rollout_chunks == 4
    assert wm.chunk_rollout_loss_scale == 0.2
    assert wm.proprio_reconstruction_loss_scale == 0.0
    assert classifier.head_type == "spatial_tf"
    assert classifier.num_layers == 12


def test_full_dataset_wm_experiment_owns_complete_training_recipe(tmp_path, monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=wm_full_dataset_train"],
        )

    assert cfg._target_ == "dreamervla.runners.OnlineCotrainPipelineRunner"
    assert cfg.task.name == "OpenVLA_Onetraj_LIBERO"
    assert cfg.training.out_dir == f"{tmp_path}/cotrain"
    assert cfg.training.resume is False
    assert cfg.training.wm_warmup_steps == 20000
    assert cfg.training.classifier_warmup_steps == 0
    assert cfg.training.warmup_replay_epochs == 10
    assert cfg.training.warmup_checkpoint_every == 500
    assert cfg.training.warmup_topk_k == 3
    assert cfg.training.wm_profile_steps == -1
    assert cfg.dataloader.batch_size == 16
    assert cfg.optim.world_model.lr == 3.0e-5
    assert cfg.online_rollout.buffer_size == 160000
    assert cfg.online_rollout.sequence_length == 36
    assert cfg.online_rollout.total_env_steps == 0
    assert cfg.env.task_ids == list(range(10))
    assert cfg.offline_warmup.infer_task_id_from_shard is True
    assert cfg.world_model.chunk_rollout_chunks == 4
    assert cfg.world_model.chunk_rollout_loss_scale == 0.2
    assert cfg.world_model.proprio_reconstruction_loss_scale == 0.0


def test_classifier_and_full_dataset_wm_share_mainline_success_sidecar(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cls_cfg = compose(
            config_name="train",
            overrides=["experiment=wmpo_token_classifier_openvla_onetraj_libero_goal_h1"],
        )
        wm_cfg = compose(
            config_name="train",
            overrides=["experiment=wm_full_dataset_train"],
        )

    assert cls_cfg.task.name == wm_cfg.task.name == "OpenVLA_Onetraj_LIBERO"
    assert cls_cfg.data.success_dir_hidden == wm_cfg.offline_warmup.hidden_dir
    assert cls_cfg.data.success_dir_raw == cls_cfg.task.openvla_oft.hdf5_dir
    assert wm_cfg.offline_warmup.data_dir == cls_cfg.task.openvla_oft.hdf5_reward_dir
    assert wm_cfg.offline_warmup.data_dir == f"{cls_cfg.data.success_dir_raw}_remaining_reward"


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
