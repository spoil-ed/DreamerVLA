"""Hydra-to-worker configuration adapter for rollout collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from dreamervla.runtime.oft_collect import (
    resolve_num_images_in_input,
    select_vla_image_keys,
)


def build_oft_collect_config(cfg: DictConfig) -> dict[str, Any]:
    """Resolve the task and collection groups into the OFT worker contract."""

    oft = cfg.task.openvla_oft
    latent_spec = OmegaConf.select(cfg, "collect.oft_latent_spec", default=None)
    hidden_token = OmegaConf.select(
        cfg,
        "task.openvla_oft.hidden_token",
        default=None,
    )
    if hidden_token is None:
        raise ValueError("task.openvla_oft.hidden_token is required for collection")
    latent = latent_spec if latent_spec is not None else hidden_token
    expected_history = int(latent.expected_history)
    num_images_in_input = resolve_num_images_in_input(cfg.collect)
    image_keys = select_vla_image_keys(
        list(cfg.task.image_keys),
        history=expected_history,
        num_images_in_input=num_images_in_input,
    )
    task_ids = cfg.collect.task_ids
    if OmegaConf.is_config(task_ids):
        task_ids = OmegaConf.to_container(task_ids, resolve=True)
    reward_dir = OmegaConf.select(
        cfg,
        "collect.hdf5_reward_dir",
        default=oft.hdf5_reward_dir,
    )
    hidden_dir = OmegaConf.select(
        cfg,
        "collect.hidden_dir",
        default=oft.hidden_token_dir,
    )
    render_devices = OmegaConf.select(cfg, "collect.render_devices", default=[])
    if OmegaConf.is_config(render_devices):
        render_devices = OmegaConf.to_container(render_devices, resolve=True)
    collect_cfg: dict[str, Any] = {
        "model_path": str(oft.ckpt_path),
        "policy_mode": str(
            OmegaConf.select(cfg, "collect.policy_mode", default="discrete")
        ),
        "unnorm_key": str(oft.dataset_statistics_key),
        "task_suite_name": str(cfg.task.suite),
        "task_ids": task_ids,
        "episodes_per_task": int(cfg.collect.episodes_per_task),
        "num_tasks": OmegaConf.select(cfg, "collect.num_tasks", default=None),
        "episode_horizon": int(cfg.collect.episode_horizon),
        "envs_per_gpu": int(cfg.collect.envs_per_gpu),
        "demos_per_shard": int(
            OmegaConf.select(cfg, "collect.demos_per_shard", default=0)
        ),
        "memory_fraction": float(cfg.collect.memory_fraction),
        "render_backend": str(
            OmegaConf.select(
                cfg,
                "collect.render_backend",
                default=OmegaConf.select(cfg, "render_backend", default="osmesa"),
            )
        ),
        "render_devices": list(render_devices or []),
        "reward_dir": str(reward_dir),
        "hidden_dir": str(hidden_dir),
        "image_keys": image_keys,
        "expected_history": expected_history,
        "num_images_in_input": num_images_in_input,
        "expected_action_head_type": str(latent.expected_action_head_type),
        "expected_include_state": bool(latent.expected_include_state),
        "expected_obs_hidden_source": str(latent.expected_obs_hidden_source),
        "expected_prompt_style": str(latent.expected_prompt_style),
        "expected_rotate_images_180": bool(latent.expected_rotate_images_180),
        "time_horizon": int(
            OmegaConf.select(latent, "time_horizon", default=latent.chunk_size)
        ),
        "token_dim": int(latent.token_dim),
        "action_dim": int(cfg.task.action_dim),
        "chunk_size": int(latent.chunk_size),
        "resolution": int(cfg.task.image_resolution),
        "gpu_id": int(OmegaConf.select(cfg, "collect.gpu_id", default=0)),
        "min_free_gpu_gb": float(
            OmegaConf.select(cfg, "collect.min_free_gpu_gb", default=18.0)
        ),
        "progress_dir": str(
            Path(str(OmegaConf.select(cfg, "training.out_dir", default=".")))
            / ".progress"
        ),
    }
    optional_ints = {
        "token_count": OmegaConf.select(latent, "token_count", default=None),
        "hidden_dim": OmegaConf.select(latent, "wm_obs_dim", default=None),
        "patches_per_image": OmegaConf.select(
            latent,
            "patches_per_image",
            default=None,
        ),
        "num_inference_workers": OmegaConf.select(
            cfg,
            "collect.num_inference_workers",
            default=None,
        ),
    }
    for key, value in optional_ints.items():
        if value is not None:
            collect_cfg[key] = int(value)
    return collect_cfg
