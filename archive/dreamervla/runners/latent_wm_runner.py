from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from dreamervla.runners.dreamerv3_pixel_runner import DreamerV3PixelRunner


class LatentWMTrainingRunner(DreamerV3PixelRunner):
    """Latent WM trainer for full RynnVLA action-token hidden sidecars."""

    runner_name = "wm"
    runner_status = "secondary"
    runner_family = "world_model"

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.log_path = self.out_dir / "latent_wm_logs.json.txt"

    @torch.no_grad()
    def _maybe_save_viz(
        self, model_core: torch.nn.Module, batch: dict[str, Any]
    ) -> None:
        del model_core, batch
        if self.is_main_process and bool(
            OmegaConf.select(self.cfg, "viz.enabled", default=False)
        ):
            if self.global_step == 0:
                self._print(
                    "[wm] viz is disabled for hidden-only predictor training."
                )

    def _progress_postfix(self, row: dict[str, Any], max_steps: int) -> dict[str, Any]:
        postfix: dict[str, Any] = {
            "step": f"{self.global_step}/{max_steps}",
            "wm": float(row["loss"]),
        }
        if "next_latent_mse" in row:
            postfix["next"] = float(row["next_latent_mse"])
        if "rollout_mse" in row:
            postfix["roll"] = float(row["rollout_mse"])
        if "reward_loss" in row:
            postfix["rew"] = float(row["reward_loss"])
        if "reward_binary_acc" in row:
            postfix["acc"] = float(row["reward_binary_acc"])
        return postfix


__all__ = ["LatentWMTrainingRunner"]
