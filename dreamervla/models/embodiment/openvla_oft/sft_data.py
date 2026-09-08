"""OpenVLA-OFT transforms over raw LIBERO HDF5 samples."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from dreamervla.utils.integrations.openvla_oft_imports import ensure_openvla_oft_on_path


def _normalize_bounds_q99(
    values: np.ndarray, stats: dict[str, Any], mask_default: bool = True
) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    if not mask_default and "mask" not in stats:
        mask = np.zeros_like(low, dtype=bool)
    normalized = np.clip(2.0 * (values - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
    return np.where(mask, normalized, values).astype(np.float32, copy=False)


def _libero_oft_action_transform(actions: np.ndarray) -> np.ndarray:
    actions = actions.astype(np.float32, copy=True)
    gripper = actions[:, -1:]
    # Match OpenVLA-OFT's LIBERO transform: -1=open, +1=close -> 1=open, 0=close.
    actions[:, -1:] = 1.0 - np.clip(gripper, 0.0, 1.0)
    return actions


class OpenVLAHDF5Transform:
    """Convert raw images/actions into checkpoint-specific token supervision."""

    def __init__(
        self, processor: Any, action_tokenizer: Any, dataset_statistics: dict[str, Any]
    ) -> None:
        ensure_openvla_oft_on_path()
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.vla.constants import IGNORE_INDEX

        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.dataset_statistics = dataset_statistics
        self.prompt_builder_cls = PurePromptBuilder
        self.ignore_index = int(IGNORE_INDEX)

    def __call__(self, sample: dict[str, Any], *, task: str, image_key: str) -> dict[str, Any]:
        image = Image.fromarray(sample["images"][image_key])
        pixel_values = self.processor.image_processor.apply_transform(image)
        actions = _normalize_bounds_q99(
            _libero_oft_action_transform(sample["actions"]), self.dataset_statistics["action"]
        )
        current_action_string = self.action_tokenizer(actions[0])
        future_actions_string = "".join(self.action_tokenizer(actions[1:]))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        prompt_builder = self.prompt_builder_cls("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {task}?")
        prompt_builder.add_turn("gpt", action_chunk_string)
        input_ids = self.processor.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels = list(input_ids)
        labels[: -(action_chunk_len + 1)] = [self.ignore_index] * (
            len(labels) - (action_chunk_len + 1)
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "actions": actions.astype(np.float32, copy=False),
            "dataset_name": "libero_goal_no_noops",
        }
