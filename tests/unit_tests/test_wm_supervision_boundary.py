"""Latent training, independent readout evaluation, and explicit legacy migration."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamervla.config import validate_cfg
from dreamervla.diagnostics.compare_wm_libero_rollout import _world_model_hydra_config
from dreamervla.utils.legacy_wm_readout import discard_legacy_wm_readout


def test_latent_recipe_has_no_decoder_dependency(monkeypatch) -> None:
    from dreamervla.config_resolvers import register_dreamervla_resolvers

    register_dreamervla_resolvers()
    monkeypatch.delenv("WM_PIXEL_DECODER_CKPT", raising=False)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None
    ):
        cfg = compose(config_name="train", overrides=["experiment=wm_pi05_vjepa2_latent_train"])
    model = OmegaConf.to_container(cfg.world_model, resolve=True)
    assert "decoded_visual_loss" not in model
    assert model["transition_init"] == "pretrained"
    assert model["vjepa2_truncate_rollout_gradients"] is False
    assert cfg.launch.ngpu == 8


@pytest.mark.parametrize("path", ["world_model", "ray_components.world_model.kwargs"])
def test_validation_rejects_image_supervision_before_instantiation(path) -> None:
    cfg = OmegaConf.create({})
    OmegaConf.update(cfg, f"{path}.decoded_visual_loss", {"_target_": "does.not.exist"})
    with pytest.raises(ValueError, match="latent-only"):
        validate_cfg(cfg)


def test_historical_evaluation_does_not_resolve_or_build_retired_decoder(monkeypatch) -> None:
    monkeypatch.delenv("WM_PIXEL_DECODER_CKPT", raising=False)
    cfg = OmegaConf.create(
        {
            "world_model": {
                "_target_": "torch.nn.Identity",
                "decoded_visual_loss": {"checkpoint_path": "${oc.env:WM_PIXEL_DECODER_CKPT}"},
            }
        }
    )
    selected = _world_model_hydra_config(cfg)
    assert OmegaConf.to_container(selected, resolve=True) == {"_target_": "torch.nn.Identity"}
    assert "decoded_visual_loss" in cfg.world_model


def test_legacy_migration_discards_only_readout_and_reports_keys(caplog) -> None:
    state = {"weight": torch.ones(2, 2), "decoded_visual_loss.decoder.weight": torch.ones(5)}
    filtered = discard_legacy_wm_readout(state)
    assert list(filtered) == ["weight"]
    assert filtered["weight"] is state["weight"]
    assert len(state) == 2
    assert "1 tensors, 5 values" in caplog.text
    assert "decoded_visual_loss.decoder.weight" in caplog.text
    with pytest.raises(RuntimeError, match="Missing key"):
        torch.nn.Linear(2, 2).load_state_dict(filtered, strict=True)
