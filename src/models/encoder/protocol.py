from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass
class EncoderInputBatch:
    prompt_text: list[str]
    conversations: list[list[dict[str, str]]]
    images: list[list[Any]]
    state: torch.Tensor | None = None
    state_mask: torch.Tensor | None = None
    action: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    task_type: list[str] | None = None
    task_id: torch.Tensor | None = None
    meta: list[dict[str, Any]] | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "conversations": self.conversations,
            "images": self.images,
            "state": self.state,
            "state_mask": self.state_mask,
            "action": self.action,
            "action_mask": self.action_mask,
            "task_type": self.task_type,
            "task_id": self.task_id,
            "meta": self.meta,
        }


def build_encoder_input_batch(
    obs: Mapping[str, Any],
    *,
    action: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    meta: list[dict[str, Any]] | None = None,
) -> EncoderInputBatch:
    if "encoder_inputs" in obs and isinstance(obs["encoder_inputs"], Mapping):
        encoded = dict(obs["encoder_inputs"])
        return EncoderInputBatch(
            prompt_text=list(encoded.get("prompt_text", [])),
            conversations=list(encoded.get("conversations", [])),
            images=list(encoded.get("images", [])),
            state=encoded.get("state"),
            state_mask=encoded.get("state_mask"),
            action=encoded.get("action", action),
            action_mask=encoded.get("action_mask", action_mask),
            task_type=list(encoded.get("task_type", [])) if encoded.get("task_type") is not None else None,
            task_id=encoded.get("task_id"),
            meta=encoded.get("meta", meta),
        )

    prompt_text = obs.get("prompt_text")
    if isinstance(prompt_text, str):
        prompt_batch = [prompt_text]
    else:
        prompt_batch = list(prompt_text or [])

    conversations = obs.get("conversations")
    if conversations is None:
        conversations_batch: list[list[dict[str, str]]] = [[] for _ in prompt_batch]
    elif conversations and isinstance(conversations, list) and conversations[0] and isinstance(conversations[0], dict):
        conversations_batch = [list(conversations)]
    else:
        conversations_batch = list(conversations)

    images = obs.get("images")
    if images is None:
        image_batch: list[list[Any]] = [[] for _ in prompt_batch]
    elif images and isinstance(images, list) and not isinstance(images[0], list):
        image_batch = [list(images)]
    else:
        image_batch = list(images)

    task_type = obs.get("task_type")
    if isinstance(task_type, str):
        task_type_batch: list[str] | None = [task_type]
    elif task_type is None:
        task_type_batch = None
    else:
        task_type_batch = list(task_type)

    return EncoderInputBatch(
        prompt_text=prompt_batch,
        conversations=conversations_batch,
        images=image_batch,
        state=obs.get("state"),
        state_mask=obs.get("state_mask"),
        action=action,
        action_mask=action_mask,
        task_type=task_type_batch,
        task_id=obs.get("task_id"),
        meta=meta,
    )


__all__ = ["EncoderInputBatch", "build_encoder_input_batch"]
