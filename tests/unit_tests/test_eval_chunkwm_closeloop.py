from __future__ import annotations

import torch
from omegaconf import OmegaConf

from dreamervla.diagnostics.eval_chunkwm_closeloop import (
    load_chunk_wm,
    truncate_demo_to_wm_context,
)
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
    OmegaConf.save({"world_model": cfg}, run_dir / "resolved_config.yaml")
    ckpt_path = ckpt_dir / "wm_warmup.ckpt"
    torch.save({"global_step": 0, "world_model": wm.state_dict()}, ckpt_path)

    loaded = load_chunk_wm(str(ckpt_path), torch.device("cpu"))

    assert loaded.chunk_size == 2
    assert loaded.obs_dim == 8
    assert loaded.token_count == 2
    assert loaded.token_dim == 4


def test_eval_chunkwm_truncates_demo_to_world_model_context() -> None:
    cfg = _tiny_wm_cfg()
    cfg["max_seq_len"] = 5
    wm = ChunkAwareWorldModel(**{k: v for k, v in cfg.items() if k != "_target_"})
    obs = torch.zeros(12, cfg["obs_dim"]).numpy()
    actions = torch.zeros(10, cfg["action_dim"]).numpy()

    obs_out, actions_out = truncate_demo_to_wm_context(wm, obs, actions)

    assert obs_out.shape == (5, cfg["obs_dim"])
    assert actions_out.shape == (5, cfg["action_dim"])
