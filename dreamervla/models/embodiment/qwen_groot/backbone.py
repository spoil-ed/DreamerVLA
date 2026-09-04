"""Qwen3-VL backbone migrated from SiPAI's Qwen-GR00T implementation."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from dreamervla.models.embodiment.protocol import EncoderInputBatch

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
ACTION_TOKEN_MIN = 151669
ACTION_TOKEN_MAX = 153716


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


class Qwen3VLInterface(nn.Module):
    """Qwen3-VL wrapper implementing the SiPAI multimodal input contract."""

    def __init__(
        self,
        model_path: str,
        *,
        cot_prompt: str | None = None,
        attn_implementation: str = "sdpa",
        torch_dtype: torch.dtype = torch.bfloat16,
        ignore_mismatched_sizes: bool = True,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen-GR00T requires transformers==4.57.6 or another release "
                "that exports Qwen3VLForConditionalGeneration. Use the isolated "
                "Qwen-GR00T environment; the OpenVLA transformers fork is incompatible."
            ) from exc

        implementation = str(attn_implementation)
        if implementation == "flash_attention_2" and not _has_flash_attn():
            warnings.warn(
                "flash_attn is unavailable; Qwen3-VL is falling back to SDPA",
                RuntimeWarning,
                stacklevel=2,
            )
            implementation = "sdpa"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation=implementation,
            dtype=torch_dtype,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        self.cot_prompt = cot_prompt
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        if "-Action" in str(model_path):
            self._ACTION_TOKEN_MIN = ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = ACTION_TOKEN_MAX

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, **kwargs: Any) -> Any:
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            return self.model(**kwargs)

    def generate(self, **kwargs: Any) -> Any:
        with torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available(),
        ):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(
        self,
        images: Sequence[Sequence[Any]],
        instructions: Sequence[str],
        solutions: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch size")
        messages: list[list[dict[str, Any]]] = []
        for index, (sample_images, instruction) in enumerate(
            zip(images, instructions, strict=True)
        ):
            content = [{"type": "image", "image": image} for image in sample_images]
            prompt = (
                self.cot_prompt.replace("{instruction}", str(instruction))
                if self.cot_prompt
                else str(instruction)
            )
            content.append({"type": "text", "text": prompt})
            sample_messages = [{"role": "user", "content": content}]
            if solutions is not None:
                sample_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": solutions[index]}],
                    }
                )
            messages.append(sample_messages)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if solutions is not None:
            labels = inputs["input_ids"].clone()
            for sequence in labels:
                action_tokens = (sequence >= ACTION_TOKEN_MIN) & (sequence <= ACTION_TOKEN_MAX)
                indices = torch.nonzero(action_tokens, as_tuple=False)
                if indices.numel():
                    sequence[: indices[0].item()] = IGNORE_INDEX
                else:
                    sequence[:] = IGNORE_INDEX
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            inputs["labels"] = labels
        return inputs.to(self.device)


def _image_token_mask(model_inputs: Mapping[str, Any]) -> torch.Tensor | None:
    input_ids = model_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return None
    return (input_ids == IMAGE_TOKEN_INDEX) | (input_ids == VIDEO_TOKEN_INDEX)


def _mapping_view(batch: EncoderInputBatch | Mapping[str, Any]) -> dict[str, Any]:
    value = batch.to_mapping() if isinstance(batch, EncoderInputBatch) else dict(batch)
    images = value.get("images")
    prompts = value.get("prompt_text", value.get("instructions"))
    if images is None or prompts is None:
        raise KeyError("Qwen backbone inputs require images and prompt_text")
    image_batch = list(images)
    prompt_batch = [str(prompt) for prompt in prompts]
    if len(image_batch) != len(prompt_batch) or not image_batch:
        raise ValueError("Qwen images and prompt_text must be non-empty aligned batches")
    return {
        "images": image_batch,
        "instructions": prompt_batch,
        "actions": value.get("action", value.get("actions")),
        "action_mask": value.get("action_mask"),
        "state": value.get("state"),
        "embodiment_id": value.get("task_id", value.get("embodiment_id")),
    }


class QwenBackbone(nn.Module):
    """Qwen-family VLM backbone with DreamerVLA's encoder-input boundary."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        input_cameras: Sequence[str] | None = None,
        include_state: bool = False,
        state_keys: Sequence[str] | None = None,
        cot_prompt: str | None = None,
        attn_implementation: str = "sdpa",
        vlm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if vlm is None and not model_path:
            raise ValueError("QwenBackbone requires model_path or an injected vlm")
        self.model_path = model_path
        self.input_cameras = list(input_cameras or [])
        self.include_state = bool(include_state)
        self.state_keys = list(state_keys or [])
        self.vlm = (
            vlm
            if vlm is not None
            else Qwen3VLInterface(
                str(model_path),
                cot_prompt=cot_prompt,
                attn_implementation=attn_implementation,
            )
        )

    @property
    def hidden_size(self) -> int:
        model = getattr(self.vlm, "model", None)
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Qwen VLM does not expose config.hidden_size")
        return int(hidden_size)

    def encode(
        self,
        batch: EncoderInputBatch | Mapping[str, Any],
        *,
        mode: str = "loss",
        **kwargs: Any,
    ) -> dict[str, Any]:
        del mode, kwargs
        view = _mapping_view(batch)
        model_inputs = self.vlm.build_qwenvl_inputs(
            images=view["images"],
            instructions=view["instructions"],
        )
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            outputs = self.vlm(
                **model_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = list(outputs.hidden_states)
        last_hidden = hidden_states[-1]
        state = view["state"] if self.include_state else None
        state_tensor = (
            None
            if state is None
            else torch.as_tensor(state, device=last_hidden.device, dtype=last_hidden.dtype)
        )
        return {
            "last_hidden": last_hidden,
            "hidden_states": hidden_states,
            "encoder_attention_mask": attention_mask,
            "image_mask": _image_token_mask(model_inputs),
            "model_inputs": model_inputs,
            "actions": view["actions"],
            "action_mask": view["action_mask"],
            "state": state_tensor,
            "embodiment_id": view["embodiment_id"],
        }

    def forward(
        self,
        batch: EncoderInputBatch | Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.encode(batch, **kwargs)


__all__ = [
    "ACTION_TOKEN_MAX",
    "ACTION_TOKEN_MIN",
    "IMAGE_TOKEN_INDEX",
    "VIDEO_TOKEN_INDEX",
    "Qwen3VLInterface",
    "QwenBackbone",
]
