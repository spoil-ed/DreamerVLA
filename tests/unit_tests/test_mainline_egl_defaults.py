from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _script_cfg() -> dict:
    from dreamervla.utils.hydra_config import script_config

    return script_config("coldstart_warmup_cotrain")


def _async_script_cfg(*, render_backend: str | None = None) -> dict:
    cfg = _script_cfg()
    cfg["cotrain_engine"] = "async"
    if render_backend is not None:
        cfg["render_backend"] = render_backend
    return cfg


def _config_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "configs")


def test_coldstart_ray_mainline_defaults_render_with_egl(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    cfg = _async_script_cfg()

    plan = build_pipeline_plan(
        mode="ray",
        profile="multi_gpu",
        ngpu=2,
        run_root=tmp_path,
        python="python",
        launcher_cfg=cfg,
    )

    assert cfg["render_backend"] == "egl"
    assert cfg["profiles"]["multi_gpu"].get("ray_online_real_render_backend") is None
    assert "env.cfg.render_backend=egl" in plan.collect_cmd
    assert "render_backend=egl" in plan.cotrain_online_cmd
    assert "env.cfg.render_backend=egl" in plan.cotrain_online_cmd
    assert not any(
        item.startswith("manual_cotrain.real_render_backend=")
        for item in plan.cotrain_online_cmd
    )
    assert plan.eval_cfg.get("render_backend") == "egl"


def test_coldstart_ray_mainline_keeps_explicit_osmesa_fallback(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="ray",
        profile="multi_gpu",
        ngpu=2,
        run_root=tmp_path,
        python="python",
        launcher_cfg=_async_script_cfg(render_backend="osmesa"),
    )

    assert "env.cfg.render_backend=osmesa" in plan.collect_cmd
    assert "render_backend=osmesa" in plan.cotrain_online_cmd
    assert "env.cfg.render_backend=osmesa" in plan.cotrain_online_cmd
    assert not any(
        item.startswith("manual_cotrain.real_render_backend=")
        for item in plan.cotrain_online_cmd
    )


def test_default_egl_rejects_zero_gpu_without_osmesa_override(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(ValueError, match="render_backend=egl requires ngpu>=1"):
        build_pipeline_plan(
            mode="ray",
            profile="smoke",
            ngpu=0,
            run_root=tmp_path,
            python="python",
            launcher_cfg=_async_script_cfg(),
        )


def test_zero_gpu_accepts_explicit_osmesa_fallback(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="ray",
        profile="smoke",
        ngpu=0,
        run_root=tmp_path,
        python="python",
        launcher_cfg=_async_script_cfg(render_backend="osmesa"),
    )

    assert "env.cfg.render_backend=osmesa" in plan.collect_cmd
    assert "render_backend=osmesa" in plan.cotrain_online_cmd


def test_collect_and_manual_cotrain_configs_default_to_egl() -> None:
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        collect_cfg = compose(config_name="train", overrides=["experiment=collect_rollouts_ray"])
        ray_cfg = compose(
            config_name="train",
            overrides=["experiment=openvla_onetraj_libero_cotrain_ray"],
        )

    assert collect_cfg.env.cfg.render_backend == "egl"
    assert ray_cfg.render_backend == "egl"
    assert ray_cfg.env.cfg.render_backend == "egl"
    assert OmegaConf.select(ray_cfg, "manual_cotrain.real_render_backend") is None


def test_ray_base_config_default_documents_egl() -> None:
    cfg = OmegaConf.load(
        Path(_config_dir()) / "dreamervla" / "openvla_onetraj_libero_cotrain_ray_base.yaml"
    )

    assert cfg.render_backend == "egl"
