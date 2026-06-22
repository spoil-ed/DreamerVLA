"""Checkpoint save/resume for the standalone online DreamerVLA loop.

Extracted verbatim from online_dreamervla.py (P3 god-file split, pure relocation). Re-exported by online_dreamervla so main() and frozen_wm_actor_critic keep working. This is the clean seam for the X-01 checkpoint-schema work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from dreamervla.constants import CHECKPOINT_FORMAT_VERSION
from dreamervla.models.critic.twohot_critic import ReturnPercentileTracker
from dreamervla.runners._online_dreamervla_dist import _unwrap
from dreamervla.utils.hf_checkpoint import load_runner_payload


def save_checkpoint(
    out_dir: Path,
    *,
    world_model: torch.nn.Module,
    policy: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
    wm_optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    return_tracker: ReturnPercentileTracker,
    cfg: Any,
    env_step: int,
    update_step: int,
    classifier: torch.nn.Module | None = None,
    classifier_optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step={env_step:07d}-updates={update_step:07d}.ckpt"
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "env_step": int(env_step),
        "update_step": int(update_step),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "state_dicts": {
            "world_model": world_model.state_dict(),
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": target_critic.state_dict(),
            "world_model_optimizer": wm_optimizer.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "return_tracker": return_tracker.state_dict(),
        },
    }
    if classifier is not None:
        payload["state_dicts"]["classifier"] = classifier.state_dict()
    if classifier_optimizer is not None:
        payload["state_dicts"]["classifier_optimizer"] = (
            classifier_optimizer.state_dict()
        )
    torch.save(payload, path)
    latest = ckpt_dir / "latest.ckpt"
    torch.save(payload, latest)
    print(f"[ckpt] saved {path}", flush=True)
    return path


def load_training_checkpoint(
    ckpt_path: str | Path,
    *,
    world_model: torch.nn.Module,
    policy: torch.nn.Module,
    critic: torch.nn.Module,
    target_critic: torch.nn.Module,
    wm_optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    return_tracker: ReturnPercentileTracker,
    classifier: torch.nn.Module | None = None,
    classifier_optimizer: torch.optim.Optimizer | None = None,
    policy_strict: bool = True,
    load_policy_optimizer: bool = True,
) -> tuple[int, int]:
    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume ckpt not found: {path}")
    payload = load_runner_payload(path)
    state_dicts = payload.get("state_dicts", {})
    modules = {
        "world_model": world_model,
        "policy": policy,
        "critic": critic,
        "target_critic": target_critic,
    }
    if classifier is not None:
        modules["classifier"] = classifier
    optimizers = {
        "world_model_optimizer": wm_optimizer,
        "policy_optimizer": policy_optimizer,
        "critic_optimizer": critic_optimizer,
    }
    if classifier_optimizer is not None:
        optimizers["classifier_optimizer"] = classifier_optimizer
    for key, module in modules.items():
        if key in state_dicts:
            use_strict = True if key != "policy" else bool(policy_strict)
            missing, unexpected = _unwrap(module).load_state_dict(
                state_dicts[key], strict=use_strict
            )
            if not use_strict and (missing or unexpected):
                print(
                    f"[resume] {key} loaded non-strict: "
                    f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}",
                    flush=True,
                )
    for key, optimizer in optimizers.items():
        if key in state_dicts:
            if key == "policy_optimizer" and not bool(load_policy_optimizer):
                print(
                    "[resume] skipping policy_optimizer state (fresh moments for new params)",
                    flush=True,
                )
                continue
            optimizer.load_state_dict(state_dicts[key])
    if "return_tracker" in state_dicts:
        return_tracker.load_state_dict(state_dicts["return_tracker"])
    env_step = int(payload.get("env_step", 0))
    update_step = int(payload.get("update_step", 0))
    print(
        f"[resume] loaded {path} env_step={env_step} update_step={update_step}",
        flush=True,
    )
    return env_step, update_step
