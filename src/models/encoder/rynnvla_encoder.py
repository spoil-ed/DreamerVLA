from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
)

from .base_encoder import BaseEncoder
from .protocol import EncoderInputBatch
from .rynnvla_runtime import FlexARItemProcessorActionState


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


@dataclass
class RynnVLAEncoderOutput:
    hidden: torch.Tensor
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class RynnVLAEncoder(BaseEncoder):
    def __init__(
        self,
        model_path: str = "/home/yuxinglei/workspace/2026nips/Dreamer-VLA/data/ckpts/starting_point",
        tokenizer_path: str = "/home/yuxinglei/workspace/2026nips/Dreamer-VLA/data/ckpts/chameleon/base_model",
        text_tokenizer_path: str = "/home/yuxinglei/workspace/2026nips/Dreamer-VLA/data/ckpts/chameleon/tokenizer/text_tokenizer.json",
        chameleon_vqgan_config: str = "/home/yuxinglei/workspace/2026nips/Dreamer-VLA/data/ckpts/chameleon/tokenizer/vqgan.yaml",
        chameleon_vqgan_ckpt: str = "/home/yuxinglei/workspace/2026nips/Dreamer-VLA/data/ckpts/chameleon/tokenizer/vqgan.ckpt",
        resolution: int = 256,
        action_dim: int = 7,
        time_horizon: int = 5,
        pool: str = "mean",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_path = str(_resolve_path(model_path))
        self.tokenizer_path = str(_resolve_path(tokenizer_path))
        self.text_tokenizer_path = str(_resolve_path(text_tokenizer_path))
        self.chameleon_vqgan_config = str(_resolve_path(chameleon_vqgan_config))
        self.chameleon_vqgan_ckpt = str(_resolve_path(chameleon_vqgan_ckpt))
        self.resolution = int(resolution)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.pool = str(pool)

        self._processor: FlexARItemProcessorActionState | None = None
        self.backbone = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
            self.model_path,
            action_dim=self.action_dim,
            time_horizon=self.time_horizon,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        if hasattr(self.backbone.model, "vqmodel"):
            del self.backbone.model.vqmodel
        if freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _build_processor(self, device: torch.device) -> FlexARItemProcessorActionState:
        if self._processor is None or self._processor.device != str(device):
            self._processor = FlexARItemProcessorActionState(
                tokenizer_path=self.tokenizer_path,
                text_tokenizer_path=self.text_tokenizer_path,
                vqgan_cfg_path=self.chameleon_vqgan_config,
                vqgan_ckpt_path=self.chameleon_vqgan_ckpt,
                target_size=self.resolution,
                device=str(device),
            )
        return self._processor

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def _build_observation_conversation(
        self,
        prompt_text: str,
        task_type: str | None,
        num_images: int,
        has_state: bool,
    ) -> list[dict[str, Any]]:
        state_placeholder = "<|state|>" if has_state else ""
        image_placeholders = "<|image|>" * num_images
        if task_type == "action":
            human_value = f"What action should the robot take to {prompt_text}?" + state_placeholder + image_placeholders
        elif task_type == "world":
            human_value = f"Observe the current scene for {prompt_text}." + state_placeholder + image_placeholders
        else:
            human_value = prompt_text + state_placeholder + image_placeholders
        return [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": None},
        ]

    def _to_data_item(self, batch: EncoderInputBatch, idx: int) -> dict[str, Any]:
        images = batch.images[idx]
        state = None
        if batch.state is not None:
            state_array = batch.state[idx].detach().cpu().numpy()
            if batch.state_mask is not None:
                valid = int(batch.state_mask[idx].sum().item())
                state_array = state_array[:valid]
            if state_array.size > 0:
                state = state_array
        prompt_text = batch.prompt_text[idx] if idx < len(batch.prompt_text) else ""
        task_type = None
        if batch.task_type is not None and idx < len(batch.task_type):
            task_type = batch.task_type[idx]
        conversations = self._build_observation_conversation(
            prompt_text=prompt_text,
            task_type=task_type,
            num_images=len(images),
            has_state=state is not None,
        )
        data_item = {
            "conversations": conversations,
            "image": images,
        }
        if state is not None:
            data_item["state"] = state
        return data_item

    def encode_inputs(self, batch: EncoderInputBatch) -> RynnVLAEncoderOutput:
        device = self.device
        processor = self._build_processor(device)
        input_ids_list = []
        labels_list = []
        lengths = []
        for idx in range(len(batch.prompt_text)):
            data_item = self._to_data_item(batch, idx)
            input_ids, labels = processor.process_item(data_item, training_mode=True)
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            lengths.append(len(input_ids))

        with torch.no_grad():
            _, _, _, hidden_states, labels_tensor, _, _ = self.backbone(
                input_ids=input_ids_list,
                labels=labels_list,
                training=True,
                output_hidden_states=True,
            )

        attention_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        for idx, length in enumerate(lengths):
            attention_mask[idx, :length] = True

        if self.pool == "last":
            pooled = torch.stack([hidden_states[idx, length - 1] for idx, length in enumerate(lengths)], dim=0)
        else:
            weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        return RynnVLAEncoderOutput(
            hidden=pooled.float(),
            hidden_states=hidden_states.float(),
            attention_mask=attention_mask,
            labels=labels_tensor,
        )

    def encode(self, obs: dict[str, Any]) -> torch.Tensor:
        meta = obs.get("meta") if isinstance(obs.get("meta"), list) else None
        batch = self.prepare_inputs(obs, action=None, action_mask=None, meta=meta)
        return self.encode_inputs(batch).hidden


__all__ = ["RynnVLAEncoder", "RynnVLAEncoderOutput"]
