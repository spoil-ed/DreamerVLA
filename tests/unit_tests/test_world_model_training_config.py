from __future__ import annotations


def test_hidden_token_pipeline_uses_traj1_proprio_language_wm_profile():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=wm_full_dataset_train"],
        )

    wm = cfg.world_model
    assert "classifier" not in cfg
    assert "latent_type" not in cfg
    assert cfg.task.openvla_oft.hidden_token.expected_obs_hidden_source == "hidden_token"
    assert wm.token_count == 256
    assert wm.token_dim == 4096
    assert wm.token_normalization == "layer_norm"
    assert wm.token_norm_eps == 1.0e-6
    assert wm.mlp_dim == 4096
    assert cfg.training.wm_warmup_steps == 20000
    assert cfg.training.warmup_replay_epochs == 10
    assert cfg.training.warmup_checkpoint_every_epochs == 1
    assert cfg.training.warmup_topk_k == 3
    assert cfg.training.wm_profile_steps == 8
    assert cfg.training.wm_prefetch_workers == 1
    assert cfg.dataloader.batch_size == 16
    assert cfg.optim.param_precision == "fp32"
    assert cfg.optim.precision == "bf16"
    assert cfg.optim.world_model.name == "adamw"
    assert cfg.optim.world_model.lr == 1.0e-4
    assert cfg.optim.world_model.eps == 1.0e-8
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

    assert cfg._target_ == "dreamervla.runners.WorldModelTrainingRunner"
    assert cfg.task.name == "OpenVLA_Onetraj_LIBERO"
    assert Path(cfg.training.out_dir).parent == tmp_path / "wm_full_dataset_train"
    assert cfg.training.resume is False
    assert cfg.training.wm_warmup_steps == 20000
    assert cfg.training.classifier_warmup_steps == 0
    assert cfg.training.warmup_replay_epochs == 10
    assert cfg.training.warmup_checkpoint_every_epochs == 1
    assert cfg.training.warmup_topk_k == 3
    assert cfg.training.wm_profile_steps == 8
    assert cfg.training.wm_prefetch_workers == 1
    assert cfg.training.world_model_ddp.find_unused_parameters is False
    assert cfg.training.world_model_ddp.broadcast_buffers is False
    assert cfg.training.world_model_ddp.static_graph is True
    assert cfg.training.world_model_ddp.gradient_as_bucket_view is True
    assert cfg.dataloader.batch_size == 16
    assert cfg.optim.param_precision == "fp32"
    assert cfg.optim.precision == "bf16"
    assert cfg.optim.world_model.name == "adamw"
    assert cfg.optim.world_model.lr == 1.0e-4
    assert cfg.optim.world_model.eps == 1.0e-8
    assert cfg.online_rollout.buffer_size == 160000
    assert cfg.online_rollout.sequence_length == 36
    assert cfg.online_rollout.total_env_steps == 0
    assert cfg.env.task_ids == list(range(10))
    assert cfg.offline_warmup.infer_task_id_from_shard is True
    assert cfg.world_model.chunk_rollout_chunks == 4
    assert cfg.world_model.chunk_rollout_loss_scale == 0.2
    assert cfg.world_model.proprio_reconstruction_loss_scale == 0.0
    assert cfg.world_model.token_normalization == "layer_norm"
    assert cfg.world_model.token_norm_eps == 1.0e-6


def test_dino_token_architecture_and_data_protocol_recipe(tmp_path, monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=wm_dino_token_official"],
        )

    assert cfg._target_ == "dreamervla.runners.DinoTokenWorldModelTrainingRunner"
    assert cfg.world_model._target_.endswith("DinoTokenWorldModel")
    assert cfg.world_model.token_count == 256
    assert cfg.world_model.token_dim == 4096
    assert cfg.world_model.num_hist == 3
    assert cfg.world_model.num_pred == 1
    assert cfg.world_model.depth == 6
    assert cfg.world_model.heads == 16
    assert cfg.world_model.dim_head == 64
    assert cfg.world_model.mlp_dim == 2048
    assert cfg.world_model.dropout == 0.1
    assert cfg.world_model.emb_dropout == 0.0
    assert cfg.dino_wm.frameskip == 5
    assert cfg.world_model.action_dim == 35
    assert cfg.world_model.action_emb_dim == 10
    assert cfg.world_model.proprio_emb_dim == 10
    assert "token_normalization" not in cfg.world_model
    assert "token_norm_eps" not in cfg.world_model
    assert cfg.dataset.train.frameskip == 5
    assert cfg.dataset.train.train_fraction == 0.9
    assert cfg.dataset.train.split_seed == 42
    assert cfg.dataset.train.slice_seed == 0
    assert cfg.dataset.train.normalize_action is True
    assert cfg.dataset.train.normalize_proprio is True
    assert cfg.dataset.valid.split == "valid"
    assert "lang_dim" not in cfg.world_model
    assert "chunk_rollout_chunks" not in cfg.world_model
    assert "global_batch_size" not in cfg.training
    assert cfg.dataloader.batch_size == 16
    assert cfg.training.num_epochs == 100
    assert cfg.optim.param_precision == "fp32"
    assert cfg.optim.precision == "fp32"
    assert cfg.optim.predictor.name == "adamw"
    assert cfg.optim.predictor.lr == 1.0e-4
    assert cfg.optim.predictor.betas == [0.9, 0.999]
    assert cfg.optim.predictor.eps == 1.0e-8
    assert cfg.optim.predictor.weight_decay == 0.01
    assert cfg.optim.conditioning == cfg.optim.predictor
    assert cfg.dataset.train.raw_dir == cfg.task.hdf5_reward_dir
    assert cfg.dataset.train.hidden_dir == cfg.task.openvla_oft.hidden_token_dir


def test_user_facing_dino_and_dreamer_configs_align_batch_and_lr(tmp_path, monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("RUN_ROOT", str(tmp_path))
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        dino = compose(config_name="train", overrides=["experiment=dino-wm"])
        dreamer = compose(config_name="train", overrides=["experiment=dreamer-wm"])

    assert dino.dataloader.batch_size == dreamer.dataloader.batch_size == 16
    assert dino.optim.predictor.lr == dreamer.optim.world_model.lr == 1.0e-4
    assert dino.optim.conditioning.lr == dreamer.optim.world_model.lr
    assert dreamer.training.warmup_replay_epochs == 100


def test_classifier_and_full_dataset_wm_share_mainline_success_sidecar(tmp_path, monkeypatch):
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
    assert cls_cfg.data.success_dir_raw == cls_cfg.task.collected_reward_dir
    assert wm_cfg.offline_warmup.data_dir == cls_cfg.task.collected_reward_dir
    assert wm_cfg.offline_warmup.data_dir == cls_cfg.data.success_dir_raw


def test_validate_cfg_warmup(tmp_path):
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.config import validate_cfg

    base = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.WorldModelTrainingRunner",
            "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
            "training": {"wm_warmup_steps": 10, "classifier_warmup_steps": 10},
        }
    )
    validate_cfg(base)  # existing dir -> ok
    bad = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.WorldModelTrainingRunner",
            "offline_warmup": {"data_dir": str(tmp_path / "nope"), "hidden_dir": str(tmp_path)},
            "training": {"wm_warmup_steps": 10, "classifier_warmup_steps": 10},
        }
    )
    with pytest.raises(Exception, match="offline_warmup.data_dir"):
        validate_cfg(bad)
    neg = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.WorldModelTrainingRunner",
            "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
            "training": {"wm_warmup_steps": -1, "classifier_warmup_steps": 10},
        }
    )
    with pytest.raises(Exception, match="wm_warmup_steps"):
        validate_cfg(neg)
    neg_replay_max = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.WorldModelTrainingRunner",
            "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
            "training": {
                "wm_warmup_steps": 10,
                "classifier_warmup_steps": 10,
                "warmup_replay_epochs": 1,
                "warmup_replay_max_steps": -1,
            },
        }
    )
    with pytest.raises(Exception, match="warmup_replay_max_steps"):
        validate_cfg(neg_replay_max)
    neg_ckpt_every = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.WorldModelTrainingRunner",
            "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
            "training": {
                "wm_warmup_steps": 10,
                "classifier_warmup_steps": 10,
                "warmup_checkpoint_every_epochs": -1,
            },
        }
    )
    with pytest.raises(Exception, match="warmup_checkpoint_every_epochs"):
        validate_cfg(neg_ckpt_every)

    removed_ckpt_every = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.WorldModelTrainingRunner",
            "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
            "training": {
                "wm_warmup_steps": 10,
                "classifier_warmup_steps": 10,
                "warmup_checkpoint_every": 1,
            },
        }
    )
    with pytest.raises(Exception, match="warmup_checkpoint_every_epochs"):
        validate_cfg(removed_ckpt_every)


def test_classifier_checkpoint_cadence_uses_epochs_and_rejects_removed_field(tmp_path):
    import pytest
    from omegaconf import OmegaConf

    from dreamervla.config import validate_cfg

    valid = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.SuccessClassifierTrainingRunner",
            "training": {"num_epochs": 2, "checkpoint_every_epochs": 1},
        }
    )
    validate_cfg(valid)

    disabled = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.SuccessClassifierTrainingRunner",
            "training": {"num_epochs": 2, "checkpoint_every_epochs": 0},
        }
    )
    with pytest.raises(Exception, match="checkpoint_every_epochs"):
        validate_cfg(disabled)

    removed = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.SuccessClassifierTrainingRunner",
            "training": {"num_epochs": 2, "ckpt_every": 1},
        }
    )
    with pytest.raises(Exception, match="checkpoint_every_epochs"):
        validate_cfg(removed)
    bad_log_every = OmegaConf.create(
        {
            "_target_": "dreamervla.runners.WorldModelTrainingRunner",
            "offline_warmup": {"data_dir": str(tmp_path), "hidden_dir": str(tmp_path)},
            "training": {
                "wm_warmup_steps": 10,
                "classifier_warmup_steps": 10,
                "replay_warmup_log_every": 0,
            },
        }
    )
    with pytest.raises(Exception, match="replay_warmup_log_every"):
        validate_cfg(bad_log_every)
