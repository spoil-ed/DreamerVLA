from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.dataset.pi05_sft import (
    OFFICIAL_PI05_LIBERO_REPO,
    resolve_lerobot_source,
)
from dreamervla.runners.vla_sft_training_runner import _build_lr_scheduler


def test_pi05_sft_experiment_composes_native_ddp_recipe() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=pi05_libero_sft"])
    OmegaConf.resolve(cfg)
    assert cfg._target_ == "dreamervla.runners.VLASFTTrainingRunner"
    assert cfg.data.train_data_paths.endswith(
        "data/datasets/lerobot/physical-intelligence/libero"
    )
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.global_batch_size == 128
    assert cfg.actor.use_action_chunk_loss is False
    assert cfg.actor.policy_cfg.kwargs.add_value_head is False
    assert cfg.actor.policy_cfg.kwargs.model_path == cfg.task.pi05.base_ckpt_path
    assert cfg.task.pi05.base_ckpt_path.endswith("data/checkpoints/pi05_base")
    assert cfg.task.pi05.ckpt_path.endswith(
        "data/checkpoints/RLinf-Pi05-LIBERO-SFT"
    )
    assert cfg.task.pi05.base_ckpt_path != cfg.task.pi05.ckpt_path
    assert cfg.actor.distributed.strategy == "ddp"
    assert cfg.training.distributed_strategy == "ddp"
    assert cfg.actor.optim.lr_scheduler == "constant"
    assert cfg.actor.optim.lr == pytest.approx(2.5e-5)
    assert cfg.actor.optim.adam_beta1 == pytest.approx(0.9)
    assert cfg.actor.optim.adam_beta2 == pytest.approx(0.95)
    assert cfg.actor.optim.adam_eps == pytest.approx(1.0e-8)
    assert cfg.actor.optim.weight_decay == pytest.approx(1.0e-10)
    assert cfg.actor.optim.clip_grad == pytest.approx(1.0)
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.total_training_steps == 30000
    assert cfg.training.max_steps == 30000
    assert cfg.training.checkpoint_every == 2000


def test_resolve_pi05_lerobot_source_accepts_repo_and_complete_local_root(
    tmp_path: Path,
) -> None:
    assert resolve_lerobot_source(OFFICIAL_PI05_LIBERO_REPO) == OFFICIAL_PI05_LIBERO_REPO
    root = tmp_path / "libero"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    assert resolve_lerobot_source(root) == str(root.resolve())


def test_resolve_pi05_lerobot_source_rejects_incomplete_local_root(tmp_path: Path) -> None:
    root = tmp_path / "libero"
    root.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        resolve_lerobot_source(root)


def test_pi05_sft_runtime_does_not_import_sibling_rlinf() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_files = (
        root / "dreamervla/models/embodiment/pi05/policy.py",
        root / "dreamervla/dataset/pi05_sft.py",
        root / "dreamervla/runners/vla_sft_training_runner.py",
    )
    for path in runtime_files:
        source = path.read_text(encoding="utf-8")
        assert "from rlinf" not in source
        assert "import rlinf" not in source


def test_pi05_ddp_freezes_structurally_unused_expert_lm_head() -> None:
    policy_source = (
        Path(__file__).resolve().parents[2]
        / "dreamervla/models/embodiment/pi05/policy.py"
    ).read_text(encoding="utf-8")
    assert "gemma_expert.lm_head" in policy_source
    assert "parameter.requires_grad = False" in policy_source


def test_pi05_sft_scheduler_matches_rlinf_effective_constant_warmup() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=2.5e-5)
    cfg = OmegaConf.create(
        {"lr_scheduler": "constant", "lr_warmup_steps": 10, "total_training_steps": 30}
    )
    scheduler = _build_lr_scheduler(optimizer, cfg)
    assert optimizer.param_groups[0]["lr"] == 0.0
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-5)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-5)
