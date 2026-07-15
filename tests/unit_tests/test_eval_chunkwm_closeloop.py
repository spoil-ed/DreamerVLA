from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from dreamervla.diagnostics.eval_chunkwm_closeloop import (
    load_chunk_wm,
    load_demo,
    rollout,
    truncate_demo_to_wm_context,
)
from dreamervla.diagnostics.eval_dino_token_wm import _runner_config_from_checkpoint
from dreamervla.models.embodiment.world_model.wm_chunk import ChunkAwareWorldModel


def _tiny_wm_cfg() -> dict:
    return {
        "_target_": "dreamervla.models.embodiment.world_model.wm_chunk.ChunkAwareWorldModel",
        "chunk_size": 2,
        "obs_dim": 8,
        "action_dim": 2,
        "token_count": 2,
        "token_dim": 4,
        "time_horizon": 1,
        "latent_stage": "query_after",
        "latent_source": "tiny hidden-token fixture",
        "action_emb_dim": 2,
        "num_action_repeat": 1,
        "model_dim": 6,
        "depth": 1,
        "heads": 2,
        "dim_head": 4,
        "mlp_dim": 16,
        "dropout": 0.0,
        "num_hist": 2,
        "num_pred": 1,
        "max_seq_len": 8,
        "hidden_loss_scale": 1.0,
        "cosine_loss_scale": 0.1,
        "reward_head_type": "binary",
        "reward_loss_scale": 1.0,
        "reward_hidden_dim": 8,
        "return_predictions": False,
    }


def test_eval_chunkwm_loader_accepts_pipeline_split_warmup_ckpt(tmp_path) -> None:
    cfg = _tiny_wm_cfg()
    wm = ChunkAwareWorldModel(**{k: v for k, v in cfg.items() if k != "_target_"})

    run_dir = tmp_path / "cotrain"
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True)
    config_path = run_dir / ".hydra" / "config.yaml"
    config_path.parent.mkdir()
    OmegaConf.save({"world_model": cfg}, config_path)
    ckpt_path = ckpt_dir / "wm_warmup.ckpt"
    torch.save({"global_step": 0, "world_model": wm.state_dict()}, ckpt_path)

    loaded = load_chunk_wm(str(ckpt_path), torch.device("cpu"))

    assert loaded.chunk_size == 2
    assert loaded.obs_dim == 8
    assert loaded.token_count == 2
    assert loaded.token_dim == 4


def test_eval_chunkwm_loader_materializes_worker_target_config(tmp_path) -> None:
    cfg = _tiny_wm_cfg()
    cfg.update(
        {
            "model_dim": 10,
            "proprio_dim": 3,
            "proprio_emb_dim": 2,
            "num_proprio_repeat": 1,
            "lang_dim": 5,
            "lang_emb_dim": 2,
            "num_lang_repeat": 1,
            "chunk_rollout_chunks": 2,
            "chunk_rollout_loss_scale": 0.2,
        }
    )
    wm = ChunkAwareWorldModel(**{k: v for k, v in cfg.items() if k != "_target_"})

    run_dir = tmp_path / "cotrain"
    ckpt_dir = run_dir / "ckpt" / "warmup_progress"
    ckpt_dir.mkdir(parents=True)
    worker_cfg = {
        "target": cfg["_target_"],
        "kwargs": {k: v for k, v in cfg.items() if k != "_target_"},
    }
    (run_dir / ".hydra").mkdir()
    OmegaConf.save(
        {
            "world_model": {
                "chunk_rollout_chunks": 2,
                "chunk_rollout_loss_scale": 0.2,
            },
            "ray_components": {"world_model": worker_cfg},
        },
        run_dir / ".hydra" / "config.yaml",
    )
    ckpt_path = ckpt_dir / "wm_step_00000100.ckpt"
    torch.save(
        {
            "config": {
                "world_model": {
                    "chunk_rollout_chunks": 2,
                    "chunk_rollout_loss_scale": 0.2,
                }
            },
            "world_model": wm.state_dict(),
        },
        ckpt_path,
    )

    loaded = load_chunk_wm(
        str(ckpt_path),
        torch.device("cpu"),
        config_path=str(run_dir / ".hydra" / "config.yaml"),
    )

    assert loaded.model_dim == 10
    assert loaded.proprio_dim == 3
    assert loaded.lang_dim == 5


def test_dino_runner_config_prefers_complete_checkpoint_payload(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    persisted = checkpoint.parents[1] / ".hydra" / "config.yaml"
    persisted.parent.mkdir()
    OmegaConf.save({"dataset": {"valid": {"source": "disk"}}}, persisted)
    payload_cfg = OmegaConf.create({"dataset": {"valid": {"source": "payload"}}})

    loaded = _runner_config_from_checkpoint(
        {"cfg": payload_cfg}, checkpoint, config_path=None
    )

    assert loaded.dataset.valid.source == "payload"


def test_dino_runner_config_discovers_native_hydra_config_from_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "global_step_7" / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    persisted = checkpoint.parents[2] / ".hydra" / "config.yaml"
    persisted.parent.mkdir()
    OmegaConf.save({"dataset": {"valid": {"source": "canonical"}}}, persisted)

    loaded = _runner_config_from_checkpoint({}, checkpoint, config_path=None)

    assert loaded.dataset.valid.source == "canonical"


def test_dino_runner_config_accepts_explicit_yaml_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    config = tmp_path / "different" / "runner.yaml"
    config.parent.mkdir()
    OmegaConf.save({"dataset": {"valid": {"source": "explicit"}}}, config)

    loaded = _runner_config_from_checkpoint({}, checkpoint, config_path=str(config))

    assert loaded.dataset.valid.source == "explicit"


def test_eval_chunkwm_truncates_demo_to_world_model_context() -> None:
    cfg = _tiny_wm_cfg()
    cfg["max_seq_len"] = 5
    wm = ChunkAwareWorldModel(**{k: v for k, v in cfg.items() if k != "_target_"})
    obs = torch.zeros(12, cfg["obs_dim"]).numpy()
    actions = torch.zeros(10, cfg["action_dim"]).numpy()

    obs_out, actions_out = truncate_demo_to_wm_context(wm, obs, actions)

    assert obs_out.shape == (5, cfg["obs_dim"])
    assert actions_out.shape == (5, cfg["action_dim"])


def test_eval_chunkwm_load_demo_preserves_tokenized_sidecar_shape(tmp_path) -> None:
    raw_path = tmp_path / "raw.hdf5"
    hidden_path = tmp_path / "hidden.hdf5"
    with h5py.File(raw_path, "w") as handle:
        handle.create_dataset("data/demo_0/actions", data=np.zeros((3, 2)))
        handle.create_dataset("data/demo_0/obs/ee_pos", data=np.ones((3, 3)))
        handle.create_dataset("data/demo_0/obs/ee_ori", data=np.ones((3, 3)) * 2)
        handle.create_dataset(
            "data/demo_0/obs/gripper_states", data=np.ones((3, 2)) * 3
        )
    with h5py.File(hidden_path, "w") as handle:
        handle.create_dataset(
            "data/demo_0/obs_embedding",
            data=np.zeros((3, 2, 4), dtype=np.float16),
        )
        handle.create_dataset(
            "data/demo_0/lang_emb", data=np.arange(5, dtype=np.float16)
        )

    loaded = load_demo(raw_path, hidden_path, "data/demo_0")

    assert loaded is not None
    observations, actions, proprio, lang_emb = loaded
    assert observations.shape == (3, 2, 4)
    assert actions.shape == (3, 2)
    assert proprio.shape == (3, 8)
    assert np.array_equal(proprio[0], np.array([1, 1, 1, 2, 2, 2, 3, 3]))
    assert lang_emb.shape == (5,)


def test_eval_chunkwm_rollout_uses_proprio_and_language_conditioning() -> None:
    cfg = _tiny_wm_cfg()
    cfg.update(
        {
            "model_dim": 10,
            "proprio_dim": 3,
            "proprio_emb_dim": 2,
            "lang_dim": 5,
            "lang_emb_dim": 2,
        }
    )
    wm = ChunkAwareWorldModel(**{k: v for k, v in cfg.items() if k != "_target_"})
    observations = torch.zeros(6, 2, 4)
    actions = torch.zeros(6, 2)
    proprio = torch.zeros(6, 3)
    lang_emb = torch.zeros(5)

    prediction, target = rollout(
        wm,
        observations,
        actions,
        num_chunks=2,
        mode="close",
        proprio=proprio,
        lang_emb=lang_emb,
    )

    assert prediction.shape == (4, 2, 6)
    assert target.shape == prediction.shape


def test_eval_chunkwm_rollout_scores_layer_normalized_visual_targets() -> None:
    cfg = _tiny_wm_cfg()
    cfg.update({"token_normalization": "layer_norm", "token_norm_eps": 1.0e-6})
    wm = ChunkAwareWorldModel(**{k: v for k, v in cfg.items() if k != "_target_"})
    observations = torch.arange(1, 1 + 6 * 2 * 4, dtype=torch.float32).reshape(
        6, 2, 4
    )

    _prediction, target = rollout(
        wm,
        observations,
        torch.zeros(6, 2),
        num_chunks=2,
        mode="open",
    )

    assert torch.allclose(target.mean(dim=-1), torch.zeros(4, 2), atol=1.0e-6)
    assert torch.allclose(
        target.var(dim=-1, unbiased=False),
        torch.ones(4, 2),
        atol=1.0e-5,
    )
