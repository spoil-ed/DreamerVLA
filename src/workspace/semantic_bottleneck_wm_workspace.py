from __future__ import annotations

from omegaconf import DictConfig

from src.workspace.dreamerv3_pixel_workspace import DreamerV3PixelWorkspace


class SemanticBottleneckWMWorkspace(DreamerV3PixelWorkspace):
    """Training workspace for the Dreamer-VLA semantic bottleneck world model."""

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir)
        self.log_path = self.out_dir / "semantic_bottleneck_wm_logs.json.txt"


__all__ = ["SemanticBottleneckWMWorkspace"]
