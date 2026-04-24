"""Eval-only workspace: load a VLA checkpoint and run LIBERO rollouts.

No training, no optimizer, no dataset. Reuses the rollout logic that already
lives on ``PretokenizeVLAWorkspace.evaluate_libero`` so there is exactly one
code path for LIBERO success-rate measurement.

Typical use:

  bash scripts/eval_libero_vla.sh \\
    eval.ckpt_path=/path/to/pretokenize_vla/checkpoints/epoch=013-train_vla_loss=1.984.ckpt \\
    eval.task_suite_name=libero_goal \\
    eval.num_episodes_per_task=10

LIBERO rollout is strictly single-process; the script enforces a single GPU
and this workspace forces ``distributed_strategy=ddp`` so the encoder is not
sharded (FSDP sharding would block single-rank inference).
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from src.workspace.pretokenize_vla_workspace import PretokenizeVLAWorkspace


class EvalLiberoVLAWorkspace(PretokenizeVLAWorkspace):
    """Load a VLA ckpt → run LIBERO rollout → dump JSON metrics."""

    default_output_dir = "/home/user01/yuxinglei/workspace/DreamerVLA/data/outputs/eval_libero_vla"

    def run(self) -> list[dict[str, Any]]:
        if self.distributed.is_main_process:
            print("EvalLiberoVLA Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

        if self.world_size != 1:
            raise RuntimeError(
                f"EvalLiberoVLAWorkspace must run on a single process (got world_size={self.world_size}). "
                "Rollout evaluation does not support multi-process inference."
            )
        if self.distributed.uses_fsdp:
            raise RuntimeError(
                "EvalLiberoVLAWorkspace requires DDP (not FSDP). "
                "Pass `training.distributed_strategy=ddp`."
            )

        # ── encoder (inference only; no optimiser, no distributed wrapping) ──
        encoder_cfg = self._build_trainable_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.freeze_backbone = True
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        self.encoder.eval()

        # ── optional: load VLA checkpoint (produced by PretokenizeVLAWorkspace) ─
        ckpt_path = OmegaConf.select(cfg, "eval.ckpt_path", default=None)
        if ckpt_path:
            ckpt_path = str(pathlib.Path(str(ckpt_path)).expanduser().resolve())
            if self.distributed.is_main_process:
                print(f"  [Eval] loading VLA checkpoint: {ckpt_path}")
            # Only restore the encoder; skip optimiser / EMA / step counters.
            # (The ckpt was produced by PretokenizeVLAWorkspace which writes
            # vla_optimizer too, but that attribute is None here.)
            import torch as _torch
            payload = _torch.load(ckpt_path, map_location="cpu")
            self.load_payload(
                payload,
                exclude_keys=("vla_optimizer", "vla_ema"),
                include_keys=(),  # don't restore global_step / epoch
            )
        else:
            if self.distributed.is_main_process:
                print("  [Eval] no eval.ckpt_path set → evaluating init VLA weights "
                      f"({OmegaConf.select(cfg, 'init.vla_ckpt_path')})")

        # ── rollout ──────────────────────────────────────────────────────────
        os.makedirs(self.output_dir, exist_ok=True)
        metrics = self.evaluate_libero(epoch=-1)

        # ── dump metrics ─────────────────────────────────────────────────────
        if self.distributed.is_main_process:
            metrics_out = {
                "ckpt_path": ckpt_path,
                "task_suite": str(OmegaConf.select(cfg, "eval.task_suite_name", default="libero_goal")),
                "num_episodes_per_task": int(OmegaConf.select(cfg, "eval.num_episodes_per_task", default=10)),
                "action_steps": int(OmegaConf.select(cfg, "eval.action_steps", default=10)),
                **metrics,
            }
            out_path = os.path.join(self.output_dir, "eval_libero_metrics.json")
            with open(out_path, "w") as f:
                json.dump(metrics_out, f, indent=2)
            print(f"  [Eval] wrote metrics → {out_path}")

        return [metrics]


__all__ = ["EvalLiberoVLAWorkspace"]
