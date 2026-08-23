from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _config_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "configs")


def test_collection_and_cotrain_default_to_osmesa() -> None:
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        collect_cfg = compose(
            config_name="train",
            overrides=["experiment=collect_rollouts"],
        )
        cotrain_cfg = compose(
            config_name="train",
            overrides=["experiment=openvla_onetraj_libero_cotrain"],
        )

    assert collect_cfg.collect.backend == "ray"
    assert collect_cfg.env.cfg.render_backend == "osmesa"
    assert cotrain_cfg.render_backend == "osmesa"
    assert cotrain_cfg.env.cfg.render_backend == "osmesa"
    assert OmegaConf.select(cotrain_cfg, "manual_cotrain.real_render_backend") is None


def test_cotrain_base_config_default_documents_osmesa() -> None:
    cfg = OmegaConf.load(
        Path(_config_dir()) / "dreamervla" / "openvla_onetraj_libero_cotrain_base.yaml"
    )

    assert cfg.render_backend == "osmesa"


def test_standalone_eval_config_defaults_to_osmesa() -> None:
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=eval_libero_vla"])

    assert cfg.eval.render_backend == "osmesa"
    assert cfg.eval.render_gpu_pool is None
