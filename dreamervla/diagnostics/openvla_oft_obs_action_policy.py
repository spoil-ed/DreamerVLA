"""OpenVLA-OFT obs-to-action adapter.

The public surface is intentionally small:

    policy = OpenVLAOFTObsActionPolicy.from_checkpoint(...)
    actions = policy(obs, task_description)

`obs` should be the policy observation dict already prepared by the LIBERO eval
code and must contain the single mainline ``full_image`` view.
"""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ActionBackend = Callable[..., list[Any]]


def default_openvla_oft_root() -> Path:
    from dreamervla.utils.openvla_oft_imports import default_openvla_oft_root as _default_root

    return _default_root()


def set_runtime_env(gpu_id: str | int | None = None) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def ensure_openvla_oft_importable(
    openvla_oft_root: str | Path | None = None, *, chdir: bool = True
) -> Path:
    root = (
        Path(openvla_oft_root).expanduser().resolve()
        if openvla_oft_root
        else default_openvla_oft_root()
    )
    if not root.is_dir():
        raise FileNotFoundError(f"OpenVLA-OFT root not found: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    if chdir:
        os.chdir(root)
    return root


def checkpoint_has_component(checkpoint: str | Path, component: str) -> bool:
    return any(Path(checkpoint).expanduser().resolve().glob(f"{component}--*_checkpoint.pt"))


def resolve_unnorm_key(
    model: Any, task_suite_name: str, requested_unnorm_key: str | None = None
) -> str:
    norm_stats = getattr(model, "norm_stats", None)
    if norm_stats is None:
        raise AttributeError("OpenVLA-OFT model does not expose norm_stats")

    key = requested_unnorm_key or task_suite_name
    if key not in norm_stats and f"{key}_no_noops" in norm_stats:
        key = f"{key}_no_noops"
    if key not in norm_stats:
        available = ", ".join(sorted(norm_stats))
        raise KeyError(f"Action unnorm key {key!r} not found; available keys: {available}")
    return key


def filter_observation_for_config(cfg: Any, obs: dict[str, Any]) -> dict[str, Any]:
    """Keep the single image consumed by the discrete one-trajectory policy."""

    if int(getattr(cfg, "num_images_in_input", 1)) != 1:
        raise ValueError("OpenVLA-OFT mainline requires num_images_in_input=1")
    if bool(getattr(cfg, "use_proprio", False)):
        raise ValueError("OpenVLA-OFT mainline does not include proprio")
    return {"full_image": obs["full_image"]}


@dataclass
class OpenVLAOFTObsActionPolicy:
    cfg: Any
    model: Any
    processor: Any
    action_backend: ActionBackend
    action_head: Any = None
    proprio_projector: Any = None
    noisy_action_projector: Any = None

    @classmethod
    def from_backend(
        cls,
        *,
        cfg: Any,
        model: Any,
        processor: Any,
        action_backend: ActionBackend,
        action_head: Any = None,
        proprio_projector: Any = None,
        noisy_action_projector: Any = None,
    ) -> OpenVLAOFTObsActionPolicy:
        if action_head is not None:
            raise ValueError("L1/action-query checkpoints are closed")
        if proprio_projector is not None:
            raise ValueError("proprio checkpoint components are not part of the mainline")
        filter_observation_for_config(cfg, {"full_image": object()})
        return cls(
            cfg=cfg,
            model=model,
            processor=processor,
            action_backend=action_backend,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        task_suite_name: str,
        openvla_oft_root: str | Path | None = None,
        gpu_id: str | int | None = None,
        policy_mode: str = "discrete",
        num_images_in_input: int | None = None,
        use_proprio: bool | None = None,
        center_crop: bool = True,
        num_open_loop_steps: int = 8,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        unnorm_key: str | None = None,
    ) -> OpenVLAOFTObsActionPolicy:
        set_runtime_env(gpu_id)
        ensure_openvla_oft_importable(openvla_oft_root)

        from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
        from experiments.robot.openvla_utils import get_vla_action

        checkpoint = Path(checkpoint).expanduser().resolve()
        has_action_head = checkpoint_has_component(checkpoint, "action_head")
        has_proprio = checkpoint_has_component(checkpoint, "proprio_projector")

        if str(policy_mode) != "discrete":
            raise ValueError(
                f"OpenVLA-OFT mainline requires policy_mode='discrete', got {policy_mode!r}"
            )
        if has_action_head:
            raise ValueError("L1/action-query checkpoints are closed")
        if has_proprio or bool(use_proprio):
            raise ValueError("proprio checkpoint components are not part of the mainline")
        if num_images_in_input is not None and int(num_images_in_input) != 1:
            raise ValueError("OpenVLA-OFT mainline requires num_images_in_input=1")
        num_images_in_input = 1

        cfg = GenerateConfig(
            pretrained_checkpoint=str(checkpoint),
            use_l1_regression=False,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=int(num_images_in_input),
            use_proprio=False,
            center_crop=bool(center_crop),
            num_open_loop_steps=int(num_open_loop_steps),
            load_in_8bit=bool(load_in_8bit),
            load_in_4bit=bool(load_in_4bit),
            task_suite_name=task_suite_name,
            use_wandb=False,
        )

        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(
            cfg
        )
        cfg.unnorm_key = resolve_unnorm_key(
            model, task_suite_name, unnorm_key or getattr(cfg, "unnorm_key", "")
        )

        return cls(
            cfg=cfg,
            model=model,
            processor=processor,
            action_backend=get_vla_action,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
        )

    def __call__(self, obs: dict[str, Any], task_description: str) -> list[Any]:
        debug = os.environ.get("OPENVLA_OFT_ACTION_DEBUG")
        if debug:
            print(f"[OpenVLA-OFT action] begin task={task_description!r}", flush=True)
        policy_obs = filter_observation_for_config(self.cfg, obs)
        actions = self.action_backend(
            cfg=self.cfg,
            vla=self.model,
            processor=self.processor,
            obs=policy_obs,
            task_label=task_description,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            noisy_action_projector=self.noisy_action_projector,
            use_film=bool(getattr(self.cfg, "use_film", False)),
        )
        if debug:
            print(f"[OpenVLA-OFT action] done chunk_len={len(actions)}", flush=True)
        if os.environ.get("OPENVLA_OFT_EMPTY_CACHE", "1") != "0":
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return actions

    def as_get_action(self) -> Callable[..., list[Any]]:
        def get_action(_cfg, _model, obs, task_label, **_kwargs):
            return self(obs, task_label)

        return get_action
