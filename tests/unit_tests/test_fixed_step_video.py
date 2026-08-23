from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dreamervla.utils.fixed_step_video import FixedStepVideoRecorder


def test_fixed_step_video_recorder_saves_and_resets_buffer(tmp_path, monkeypatch):
    writes: list[dict[str, object]] = []

    class FakeWriter:
        def __init__(self, path: str, fps: int) -> None:
            self.path = path
            self.fps = fps
            self.frames: list[np.ndarray] = []

        def append_data(self, frame: np.ndarray) -> None:
            self.frames.append(np.asarray(frame))

        def close(self) -> None:
            Path(self.path).write_text("fake video", encoding="utf-8")
            writes.append({"path": self.path, "fps": self.fps, "frames": list(self.frames)})

    def get_writer(path: str, fps: int):
        return FakeWriter(path, fps)

    monkeypatch.setitem(sys.modules, "imageio", SimpleNamespace(get_writer=get_writer))

    recorder = FixedStepVideoRecorder(
        every_steps=3,
        output_dir=tmp_path,
        fps=12,
        max_frames=2,
    )
    for step in range(1, 3):
        recorder.capture({"third_image": np.full((4, 4, 3), step, dtype=np.uint8)})
        assert (
            recorder.maybe_save(env_step=step, episode_len=step, episode_return=0.0, task_id=2)
            is None
        )

    recorder.capture({"third_image": np.full((4, 4, 3), 3, dtype=np.uint8)})
    saved = recorder.maybe_save(
        env_step=3,
        episode_len=3,
        episode_return=1.0,
        task_id=2,
        success=True,
    )

    assert saved is not None
    assert Path(saved).is_file()
    assert "env_step=0000003" in Path(saved).name
    assert "task=2" in Path(saved).name
    assert "success=1" in Path(saved).name
    assert len(writes) == 1
    assert writes[0]["fps"] == 12
    assert len(writes[0]["frames"]) == 2
    assert recorder.num_frames == 0


def test_fixed_step_video_recorder_disabled(tmp_path):
    recorder = FixedStepVideoRecorder(every_steps=0, output_dir=tmp_path)
    recorder.capture({"third_image": np.zeros((4, 4, 3), dtype=np.uint8)})

    assert recorder.num_frames == 0
    assert recorder.maybe_save(env_step=1, episode_len=1, episode_return=0.0, task_id=0) is None
