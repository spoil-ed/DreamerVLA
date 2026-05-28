"""Aligned online evaluation environment for DreamerVLA on LIBERO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import imageio
import numpy as np

from dreamer_vla.envs.train_env import (
    DreamerVLAOnlineTrainEnv,
    DreamerVLAOnlineTrainEnvConfig,
)


@dataclass(frozen=True)
class DreamerVLAOnlineEvalEnvConfig(DreamerVLAOnlineTrainEnvConfig):
    task_sampling: Literal["sequential", "random"] = "sequential"
    init_state_sampling: Literal["sequential", "random"] = "sequential"
    deterministic: bool = True
    record_video: bool = False
    video_fps: int = 30


class DreamerVLAOnlineEvalEnv(DreamerVLAOnlineTrainEnv):
    """Evaluation wrapper with deterministic reset order and rollout helpers."""

    cfg: DreamerVLAOnlineEvalEnvConfig

    def __init__(
        self,
        config: DreamerVLAOnlineEvalEnvConfig | dict[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            config = DreamerVLAOnlineEvalEnvConfig()
        elif isinstance(config, dict):
            config = DreamerVLAOnlineEvalEnvConfig(**dict(config))
        super().__init__(config=config, **overrides)

    @classmethod
    def from_config(
        cls,
        config: DreamerVLAOnlineEvalEnvConfig | dict[str, Any],
    ) -> "DreamerVLAOnlineEvalEnv":
        return cls(config)

    def reset_eval(
        self,
        *,
        task_id: int | None = None,
        episode_id: int | None = None,
        seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.reset(seed=seed, task_id=task_id, episode_id=episode_id)

    def rollout(
        self,
        policy_fn: Callable[[dict[str, Any]], np.ndarray | Sequence[float]],
        *,
        task_id: int | None = None,
        episode_id: int | None = None,
        max_steps: int | None = None,
        video_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run one episode with a callable policy.

        The policy receives the aligned observation dict and should return one
        action in the configured ``action_input`` scale.
        """
        obs, info = self.reset_eval(task_id=task_id, episode_id=episode_id)
        frames: list[np.ndarray] = []
        total_reward = 0.0
        success = False
        steps = 0
        limit = int(max_steps if max_steps is not None else self.max_steps)

        while steps < limit:
            if video_path is not None or self.cfg.record_video:
                frames.append(np.asarray(obs["third_image"], dtype=np.uint8))
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = self.step(action)
            total_reward += float(reward)
            steps += 1
            success = bool(info.get("success", terminated))
            if terminated or truncated:
                break

        saved_video = None
        if video_path is not None:
            saved_video = self.save_video(frames, video_path)
        return {
            "success": bool(success),
            "return": float(total_reward),
            "length": int(steps),
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
            "task_id": int(info.get("task_id", self.task_id)),
            "task_description": str(
                info.get("task_description", self.task_description)
            ),
            "video_path": saved_video,
        }

    def save_video(self, frames: Sequence[np.ndarray], path: str | Path) -> str:
        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output_path), fps=int(self.cfg.video_fps))
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        finally:
            writer.close()
        return str(output_path)


EvalEnv = DreamerVLAOnlineEvalEnv


__all__ = [
    "DreamerVLAOnlineEvalEnv",
    "DreamerVLAOnlineEvalEnvConfig",
    "EvalEnv",
]
