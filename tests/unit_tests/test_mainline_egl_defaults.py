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


def test_coldstart_ray_mainline_defaults_render_with_osmesa(tmp_path) -> None:
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

    assert cfg["render_backend"] == "osmesa"
    assert cfg["profiles"]["multi_gpu"].get("ray_online_real_render_backend") is None
    assert "env.cfg.render_backend=osmesa" in plan.collect_cmd
    assert "render_backend=osmesa" in plan.cotrain_online_cmd
    assert "env.cfg.render_backend=osmesa" in plan.cotrain_online_cmd
    assert not any(
        item.startswith("manual_cotrain.real_render_backend=")
        for item in plan.cotrain_online_cmd
    )
    assert plan.eval_cfg.get("render_backend") == "osmesa"


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


def test_coldstart_noray_collect_receives_render_backend(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="noray",
        profile="smoke",
        ngpu=1,
        run_root=tmp_path,
        python="python",
        launcher_cfg=_script_cfg(),
    )

    assert "collect.render_backend=osmesa" in plan.collect_cmd


def test_default_osmesa_accepts_zero_gpu(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    plan = build_pipeline_plan(
        mode="ray",
        profile="smoke",
        ngpu=0,
        run_root=tmp_path,
        python="python",
        launcher_cfg=_async_script_cfg(),
    )

    assert "env.cfg.render_backend=osmesa" in plan.collect_cmd
    assert "render_backend=osmesa" in plan.cotrain_online_cmd


def test_explicit_egl_still_rejects_zero_gpu_without_osmesa_override(tmp_path) -> None:
    from dreamervla.launchers.coldstart_warmup_cotrain import build_pipeline_plan

    with pytest.raises(ValueError, match="render_backend=egl requires ngpu>=1"):
        build_pipeline_plan(
            mode="ray",
            profile="smoke",
            ngpu=0,
            run_root=tmp_path,
            python="python",
            launcher_cfg=_async_script_cfg(render_backend="egl"),
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


def test_collect_and_manual_cotrain_configs_default_to_osmesa() -> None:
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        noray_collect_cfg = compose(
            config_name="train",
            overrides=["experiment=collect_rollouts_onetraj"],
        )
        collect_cfg = compose(config_name="train", overrides=["experiment=collect_rollouts_ray"])
        sync_cfg = compose(
            config_name="train",
            overrides=["experiment=openvla_onetraj_libero_cotrain_noray"],
        )
        ray_cfg = compose(
            config_name="train",
            overrides=["experiment=openvla_onetraj_libero_cotrain_ray"],
        )

    assert noray_collect_cfg.collect.render_backend == "osmesa"
    assert collect_cfg.env.cfg.render_backend == "osmesa"
    assert sync_cfg.online_rollout.render_backend == "osmesa"
    assert ray_cfg.render_backend == "osmesa"
    assert ray_cfg.env.cfg.render_backend == "osmesa"
    assert OmegaConf.select(ray_cfg, "manual_cotrain.real_render_backend") is None


def test_ray_base_config_default_documents_osmesa() -> None:
    cfg = OmegaConf.load(
        Path(_config_dir()) / "dreamervla" / "openvla_onetraj_libero_cotrain_ray_base.yaml"
    )

    assert cfg.render_backend == "osmesa"


def test_standalone_eval_config_defaults_to_osmesa() -> None:
    with initialize_config_dir(config_dir=_config_dir(), version_base=None):
        cfg = compose(config_name="train", overrides=["experiment=eval_libero_vla"])

    assert cfg.eval.render_backend == "osmesa"
    assert cfg.eval.render_gpu_pool is None
