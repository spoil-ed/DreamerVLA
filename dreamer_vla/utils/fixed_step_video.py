from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class FixedStepVideoRecorder:
    """Save fixed-step training video segments without extra env rendering."""

    def __init__(
        self,
        *,
        every_steps: int,
        output_dir: str | Path,
        fps: int = 30,
        max_frames: int = 0,
        frame_key: str = "third_image",
    ) -> None:
        self.every_steps = max(int(every_steps), 0)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.fps = max(int(fps), 1)
        self.max_frames = max(int(max_frames), 0)
        self.frame_key = str(frame_key)
        self._frames: list[np.ndarray] = []

    @property
    def enabled(self) -> bool:
        return self.every_steps > 0

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def capture(self, obs: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self.frame_key in obs:
            frame = obs[self.frame_key]
        elif "agentview_rgb" in obs:
            frame = obs["agentview_rgb"]
        else:
            raise KeyError(
                f"observation has no video frame key {self.frame_key!r} or 'agentview_rgb'"
            )
        frame_arr = np.asarray(frame, dtype=np.uint8)
        if frame_arr.ndim != 3 or frame_arr.shape[-1] != 3:
            raise ValueError(
                f"video frame must be HWC uint8 RGB, got shape {tuple(frame_arr.shape)}"
            )
        self._frames.append(np.ascontiguousarray(frame_arr))
        if self.max_frames > 0 and len(self._frames) > self.max_frames:
            del self._frames[: len(self._frames) - self.max_frames]

    def maybe_save(
        self,
        *,
        env_step: int,
        episode_len: int,
        episode_return: float,
        task_id: int,
        success: bool = False,
    ) -> str | None:
        if not self.enabled or env_step <= 0 or int(env_step) % self.every_steps != 0:
            return None
        return self.save(
            env_step=env_step,
            episode_len=episode_len,
            episode_return=episode_return,
            task_id=task_id,
            success=success,
        )

    def save(
        self,
        *,
        env_step: int,
        episode_len: int,
        episode_return: float,
        task_id: int,
        success: bool = False,
    ) -> str | None:
        if not self.enabled or not self._frames:
            return None
        import imageio

        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_return = f"{float(episode_return):.3f}".replace("-", "neg")
        path = self.output_dir / (
            f"env_step={int(env_step):07d}_task={int(task_id)}_"
            f"ep_len={int(episode_len):04d}_return={safe_return}_success={int(bool(success))}.mp4"
        )
        writer = imageio.get_writer(str(path), fps=self.fps)
        try:
            for frame in self._frames:
                writer.append_data(frame)
        finally:
            writer.close()
        self._frames.clear()
        return str(path)


__all__ = ["FixedStepVideoRecorder"]
