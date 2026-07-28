"""Frozen OpenVLA-OFT encoder used transiently by raw classifier training."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import torch

from dreamervla.preprocess.preprocess_oft_hidden_token import (
    _load_oft_components,
    _predict_hidden_token_images,
)


class OnlineVLAHiddenEncoder:
    """Encode raw frames without persisting their hidden-token representation."""

    def __init__(
        self,
        *,
        model_path: str,
        unnorm_key: str,
        device: torch.device | str,
        token_dim: int = 4096,
        token_count: int = 256,
        micro_batch_size: int = 8,
        rotate_images_180: bool = True,
        center_crop: bool = True,
        image_keys: Sequence[str] = ("agentview_rgb",),
        history: int = 1,
        num_images_in_input: int = 1,
        policy_mode: str = "discrete",
    ) -> None:
        self.device = torch.device(device)
        self.token_dim = int(token_dim)
        self.token_count = int(token_count)
        self.micro_batch_size = max(1, int(micro_batch_size))
        self.rotate_images_180 = bool(rotate_images_180)
        self.args = SimpleNamespace(
            oft_ckpt=str(model_path),
            unnorm_key=str(unnorm_key),
            token_dim=self.token_dim,
            fake_num_patches=self.token_count,
            fake_oft_components=False,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=bool(center_crop),
            image_keys=list(image_keys),
            history=int(history),
            num_images_in_input=int(num_images_in_input),
            policy_mode=str(policy_mode),
        )
        self.components = _load_oft_components(self.args, self.device)

    def encode(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return GPU tensors ``tokens[B,W,256,4096]`` and ``language[B,4096]``."""
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(f"raw classifier images must be [B,W,H,W,3], got {images.shape}")
        batch, window = int(images.shape[0]), int(images.shape[1])
        if len(prompts) != batch:
            raise ValueError(f"prompt batch mismatch: {len(prompts)} != {batch}")
        raw = images.detach().cpu().numpy()
        if raw.dtype != np.uint8:
            raw = np.clip(raw, 0, 255).astype(np.uint8)
        flat_images: list[np.ndarray] = []
        flat_prompts: list[str] = []
        for batch_idx in range(batch):
            for frame_idx in range(window):
                image = raw[batch_idx, frame_idx]
                if self.rotate_images_180:
                    image = image[::-1, ::-1].copy()
                flat_images.append(image)
                flat_prompts.append(str(prompts[batch_idx]))

        token_chunks: list[torch.Tensor] = []
        language_chunks: list[torch.Tensor] = []
        # Prompts differ across tasks, while the preprocessing helper accepts
        # one prompt per call. Group consecutive equal prompts, then microbatch.
        start = 0
        while start < len(flat_images):
            prompt = flat_prompts[start]
            group_end = start + 1
            while group_end < len(flat_images) and flat_prompts[group_end] == prompt:
                group_end += 1
            for offset in range(start, group_end, self.micro_batch_size):
                end = min(group_end, offset + self.micro_batch_size)
                tokens, language = _predict_hidden_token_images(
                    components=self.components,
                    args=self.args,
                    images_by_frame=[[image] for image in flat_images[offset:end]],
                    prompt=prompt,
                    return_torch=True,
                )
                token_chunks.append(tokens.detach())
                language_chunks.append(language.detach())
            start = group_end

        tokens = torch.cat(token_chunks, dim=0).reshape(
            batch, window, self.token_count, self.token_dim
        )
        language = torch.cat(language_chunks, dim=0).reshape(batch, window, self.token_dim)
        # Every frame in a trajectory uses the same instruction.
        return tokens, language[:, 0]

    def close(self) -> None:
        self.components.clear()
