from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from transformers.modeling_outputs import CausalLMOutputWithPast

from dreamervla.utils.openvla_oft_imports import ensure_openvla_oft_on_path


def _torch_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def _left_pad_prompt_batch(
    input_ids_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    *,
    pad_token_id: int,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad prompt tensors while keeping the multimodal BOS at index zero."""

    if not input_ids_list or len(input_ids_list) != len(attention_mask_list):
        raise ValueError("prompt ids and masks must be non-empty aligned lists")
    lengths = [int(ids.shape[-1]) for ids in input_ids_list]
    max_length = max(lengths)
    ref_ids = input_ids_list[0]
    ref_mask = attention_mask_list[0]
    output_ids = torch.full(
        (len(input_ids_list), max_length),
        int(pad_token_id),
        dtype=ref_ids.dtype,
        device=ref_ids.device,
    )
    output_mask = torch.zeros(
        (len(input_ids_list), max_length),
        dtype=ref_mask.dtype,
        device=ref_mask.device,
    )
    for index, (ids, mask, length) in enumerate(
        zip(input_ids_list, attention_mask_list, lengths, strict=True)
    ):
        offset = max_length - length
        output_ids[index, offset:] = ids.reshape(-1)
        output_mask[index, offset:] = mask.reshape(-1)
        output_ids[index, offset] = int(pad_token_id)
        output_mask[index, offset] = 0
        output_ids[index, 0] = int(bos_token_id)
        output_mask[index, 0] = 1
    return output_ids, output_mask


def _resolve_loaded_token_dim(vla: Any) -> int:
    for path in (
        ("token_dim",),
        ("hidden_size",),
        ("config", "hidden_size"),
        ("language_model", "config", "hidden_size"),
        ("llm_backbone", "llm", "config", "hidden_size"),
    ):
        value = vla
        for attribute in path:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if value is not None:
            return int(value)
    raise ValueError("could not derive token_dim from loaded OpenVLA-OFT policy")


def _validate_loaded_hidden_token_geometry(
    vla: Any,
    *,
    expected_token_count: int | None = None,
    expected_token_dim: int | None = None,
    expected_num_images: int | None = None,
) -> tuple[int, int]:
    """Derive loaded VLA geometry and validate optional checkpoint metadata."""

    try:
        patches = int(vla.vision_backbone.get_num_patches())
        num_images = int(vla.vision_backbone.get_num_images_in_input())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("loaded OpenVLA-OFT policy must expose vision patch geometry") from exc
    token_dim = _resolve_loaded_token_dim(vla)
    if patches <= 0 or num_images <= 0 or token_dim <= 0:
        raise ValueError(
            "loaded OpenVLA-OFT policy exposes non-positive hidden-token geometry: "
            f"patches={patches}, token_dim={token_dim}, num_images_in_input={num_images}"
        )
    token_count = patches * num_images
    mismatches: list[str] = []
    if expected_token_count is not None and token_count != int(expected_token_count):
        mismatches.append(
            f"token_count expected={int(expected_token_count)} loaded={token_count}"
        )
    if expected_token_dim is not None and token_dim != int(expected_token_dim):
        mismatches.append(f"token_dim expected={int(expected_token_dim)} loaded={token_dim}")
    if expected_num_images is not None and num_images != int(expected_num_images):
        mismatches.append(
            f"num_images_in_input expected={int(expected_num_images)} loaded={num_images}"
        )
    if mismatches:
        raise ValueError(
            "loaded OpenVLA-OFT geometry does not match task/checkpoint metadata: "
            + "; ".join(mismatches)
        )
    return token_count, token_dim


@dataclass(frozen=True)
class NativeOFTForwardOutput:
    """Differentiable output of the native OpenVLA action-token path.

    ``action_logits`` is restricted to the checkpoint's action-token vocabulary;
    its last dimension is ordered by increasing action-bin index.  The corresponding
    original LM token ids are stored in ``action_token_ids``.
    """

    action_logits: torch.Tensor
    action_token_ids: torch.Tensor
    projected_tokens: torch.Tensor
    language_embedding: torch.Tensor
    multimodal_attention_mask: torch.Tensor | None
    full_logits: torch.Tensor


class OpenVLAOFTPolicy(nn.Module):
    """DreamerVLA wrapper around OpenVLA-OFT policy fine-tuning components."""

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "bf16",
        num_images_in_input: int = 1,
        token_count: int | None = None,
        token_dim: int | None = None,
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_dropout: float = 0.0,
        use_l1_regression: bool = False,
        use_diffusion: bool = False,
        use_proprio: bool = False,
        use_film: bool = False,
        freeze_vla_backbone: bool = False,
        low_cpu_mem_usage: bool = True,
        trust_remote_code: bool = True,
        unnorm_key: str = "libero_goal_no_noops",
        action_dim: int = 7,
        time_horizon: int = 8,
        image_keys: list[str] | None = None,
        history: int = 1,
        rotate_images_180: bool = True,
        center_crop: bool = True,
    ) -> None:
        super().__init__()
        if bool(use_l1_regression):
            raise ValueError("L1/action-query checkpoints are closed")
        if bool(use_proprio):
            raise ValueError("OpenVLA-OFT hidden-token mainline does not include proprio")
        if bool(use_film):
            raise ValueError("OpenVLA-OFT hidden-token mainline does not use FiLM")
        if int(num_images_in_input) <= 0:
            raise ValueError("num_images_in_input must be positive")
        if use_diffusion:
            raise NotImplementedError(
                "DreamerVLA OpenVLA-OFT workspace currently does not implement diffusion training."
            )
        from dreamervla.preprocess.preprocess_oft_hidden_token import (
            resolve_oft_policy_mode,
        )

        resolve_oft_policy_mode(model_path, "discrete")
        ensure_openvla_oft_on_path()

        from peft import LoraConfig, get_peft_model
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModelForVision2Seq,
            AutoProcessor,
        )

        AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.use_l1_regression = False
        self.use_diffusion = False
        self.use_proprio = False
        self.use_film = False
        self.num_images_in_input = int(num_images_in_input)
        self.unnorm_key = str(unnorm_key)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.image_keys = list(image_keys or ["agentview_rgb"])
        self.history = int(history)
        self.rotate_images_180 = bool(rotate_images_180)
        self.center_crop = bool(center_crop)

        dtype = _torch_dtype(torch_dtype)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code
        )
        vla = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=bool(low_cpu_mem_usage),
            trust_remote_code=trust_remote_code,
        )
        vla.vision_backbone.set_num_images_in_input(self.num_images_in_input)
        loaded_token_count, loaded_token_dim = _validate_loaded_hidden_token_geometry(
            vla,
            expected_token_count=token_count,
            expected_token_dim=token_dim,
            expected_num_images=self.num_images_in_input,
        )
        if freeze_vla_backbone:
            for parameter in vla.parameters():
                parameter.requires_grad = False
        elif use_lora:
            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=min(int(lora_rank), 16),
                lora_dropout=float(lora_dropout),
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )
            vla = get_peft_model(vla, lora_config)
        self.vla = vla

        self.action_head = None
        self.proprio_projector = None

        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.token_count = loaded_token_count
        self.token_dim = loaded_token_dim
        # Legacy name used by the checkpoint loss path. It denotes the complete
        # visual-token prefix, including every image configured in the backbone.
        self.num_patches = loaded_token_count
        self._prompt_token_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def from_modules(
        cls,
        *,
        vla: nn.Module,
        action_head: nn.Module | None,
        action_tokenizer: Any,
        num_patches: int,
        token_dim: int | None = None,
        use_l1_regression: bool = False,
        use_diffusion: bool = False,
        use_proprio: bool = False,
        use_film: bool = False,
        proprio_projector: nn.Module | None = None,
    ) -> OpenVLAOFTPolicy:
        if bool(use_l1_regression) or action_head is not None:
            raise ValueError("L1/action-query checkpoints are closed")
        if bool(use_proprio) or proprio_projector is not None:
            raise ValueError("OpenVLA-OFT hidden-token mainline does not include proprio")
        if bool(use_diffusion):
            raise ValueError("diffusion checkpoints are outside the discrete mainline")
        if bool(use_film):
            raise ValueError("OpenVLA-OFT hidden-token mainline does not use FiLM")
        loaded_token_count, loaded_token_dim = _validate_loaded_hidden_token_geometry(
            vla,
            expected_token_count=int(num_patches),
            expected_token_dim=token_dim,
        )
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.vla = vla
        self.action_head = None
        self.proprio_projector = None
        self.action_tokenizer = action_tokenizer
        self.token_count = loaded_token_count
        self.token_dim = loaded_token_dim
        self.num_patches = loaded_token_count
        self.num_images_in_input = int(vla.vision_backbone.get_num_images_in_input())
        self.use_l1_regression = False
        self.use_diffusion = False
        self.use_proprio = False
        self.use_film = False
        self.processor = None
        self.unnorm_key = "libero_goal_no_noops"
        self.action_dim = 7
        self.time_horizon = 8
        self.image_keys = ["agentview_rgb"]
        self.history = 1
        self.rotate_images_180 = True
        self.center_crop = True
        self._prompt_token_cache = {}
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encoder_parameter_names(self) -> tuple[str, ...]:
        """Return the input-to-projected-token parameter partition.

        The partition is derived from the loaded checkpoint module names.  It
        intentionally excludes the language model/action decoder, which owns
        the projected-token-to-action PPO update.
        """

        names = []
        for name, parameter in self.named_parameters():
            del parameter
            lowered = str(name).lower()
            if (
                "vision_backbone" in lowered
                or "vision_projector" in lowered
                or ".projector." in lowered
                or lowered.endswith(".projector.weight")
                or lowered.endswith(".projector.bias")
                or lowered.endswith(".vision_scale")
            ):
                names.append(str(name))
        if not names:
            raise RuntimeError(
                "loaded OpenVLA policy exposes no vision backbone/projector parameters"
            )
        return tuple(names)

    def encode_raw(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw images into the native projected visual-token space."""

        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [B,L] shapes")
        model = self.vla
        input_ids = input_ids.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)
        if not bool(torch.all(input_ids[:, -1] == 29871)):
            delimiter = torch.full(
                (input_ids.shape[0], 1),
                29871,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, delimiter], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(delimiter, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
        labels = torch.full_like(input_ids, -100)
        input_ids, attention_mask = model._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = model._prepare_labels_for_action_prediction(labels, input_ids)
        input_embeddings = model.get_input_embeddings()(input_ids)
        action_mask = model._process_action_masks(labels).bool()
        language_embeddings = input_embeddings[~action_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected = model._process_vision_features(
            pixel_values.to(device=self.device, dtype=input_embeddings.dtype),
            language_embeddings,
            use_film=False,
        )
        expected_count = int(model.vision_backbone.get_num_patches()) * int(
            model.vision_backbone.get_num_images_in_input()
        )
        expected_dim = int(input_embeddings.shape[-1])
        if projected.ndim != 3 or tuple(projected.shape[1:]) != (
            expected_count,
            expected_dim,
        ):
            raise ValueError(
                "projected visual tokens must match loaded VLA geometry "
                f"[B,{expected_count},{expected_dim}], got {tuple(projected.shape)}"
            )
        return projected, language_embeddings.mean(dim=1).float()

    def prepare_raw_batch(
        self,
        transitions: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Apply the deployment image/prompt preprocessing to real transitions."""

        if not transitions:
            raise ValueError("prepare_raw_batch requires at least one transition")
        extractor = self.make_extractor()
        extractor.reset()
        prepared: list[dict[str, Any]] = []
        for transition in transitions:
            obs: dict[str, Any] = {}
            for key in self.image_keys:
                if key in transition:
                    obs[key] = transition[key]
                elif key == "agentview_rgb" and "image" in transition:
                    obs[key] = transition["image"]
                else:
                    raise KeyError(f"real transition is missing VLA image key {key!r}")
            prepared.append(
                extractor.prepare(
                    obs,
                    str(transition.get("task_description", "")),
                )
            )
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        input_ids, attention_mask = _left_pad_prompt_batch(
            [item["input_ids"] for item in prepared],
            [item["attention_mask"] for item in prepared],
            pad_token_id=int(pad_id),
            bos_token_id=int(bos_id),
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.cat([item["pixel_values"] for item in prepared], dim=0),
        }

    def forward_action_tokens(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        projected_tokens: torch.Tensor | None = None,
    ) -> NativeOFTForwardOutput:
        """Run the original OpenVLA multimodal LM/action decoder.

        Exactly one visual source is required.  Supplying ``pixel_values`` executes
        the trainable vision backbone/projector; supplying ``projected_tokens`` skips
        only that encoder and feeds WM latents through the identical native decoder.
        No replacement Transformer or learned action-query bank is constructed.
        """

        if (pixel_values is None) == (projected_tokens is None):
            raise ValueError(
                "forward_action_tokens requires exactly one of pixel_values or projected_tokens"
            )
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [B,L] shapes")

        model = self.vla
        input_ids = input_ids.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)

        # The upstream OFT discrete path requires this delimiter before it appends
        # the checkpoint-defined action slots.
        if not bool(torch.all(input_ids[:, -1] == 29871)):
            delimiter = torch.full(
                (input_ids.shape[0], 1),
                29871,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, delimiter], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones_like(delimiter, dtype=attention_mask.dtype),
                ],
                dim=1,
            )

        ignore_index = -100
        labels = torch.full_like(input_ids, ignore_index)
        num_prompt_tokens = int(input_ids.shape[-1] - 1)
        input_ids, attention_mask = model._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = model._prepare_labels_for_action_prediction(labels, input_ids)

        input_embeddings = model.get_input_embeddings()(input_ids)
        action_mask = model._process_action_masks(labels).bool()
        action_counts = action_mask.sum(dim=1)
        if not bool(torch.all(action_counts == action_counts[0])):
            raise ValueError("OpenVLA action slot count must be uniform across a batch")
        action_span = int(action_counts[0].item())
        if action_span <= 0:
            raise ValueError("OpenVLA action prediction produced no action slots")

        language_embeddings = input_embeddings[~action_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        language_embedding = language_embeddings.mean(dim=1).float()

        if projected_tokens is None:
            assert pixel_values is not None
            pixel_values = pixel_values.to(device=self.device, dtype=input_embeddings.dtype)
            projected = model._process_vision_features(
                pixel_values,
                language_embeddings,
                use_film=False,
            )
        else:
            projected = projected_tokens.to(device=self.device, dtype=input_embeddings.dtype)

        expected_count = int(model.vision_backbone.get_num_patches()) * int(
            model.vision_backbone.get_num_images_in_input()
        )
        expected_dim = int(input_embeddings.shape[-1])
        if projected.ndim != 3 or tuple(projected.shape[1:]) != (
            expected_count,
            expected_dim,
        ):
            raise ValueError(
                "projected visual tokens must match loaded VLA geometry "
                f"[B,{expected_count},{expected_dim}], got {tuple(projected.shape)}"
            )

        masked_embeddings = input_embeddings * ~action_mask.unsqueeze(-1)
        multimodal_embeddings, multimodal_attention_mask = model._build_multimodal_attention(
            masked_embeddings,
            projected,
            attention_mask,
        )
        position_ids = None
        if multimodal_attention_mask is not None and bool((multimodal_attention_mask == 0).any()):
            position_ids = (multimodal_attention_mask.long().cumsum(-1) - 1).clamp_(min=0)
        lm_out = model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        action_start = expected_count + num_prompt_tokens
        full_action_logits = lm_out.logits[:, action_start : action_start + action_span, :]
        if int(full_action_logits.shape[1]) != action_span:
            raise ValueError("native OpenVLA LM output does not contain all action positions")
        num_bins = int(len(model.bin_centers))
        action_token_ids = int(model.vocab_size) - torch.arange(
            1,
            num_bins + 1,
            device=full_action_logits.device,
            dtype=torch.long,
        )
        if int(action_token_ids.min().item()) < 0 or int(action_token_ids.max().item()) >= int(
            full_action_logits.shape[-1]
        ):
            raise ValueError(
                "checkpoint action-token ids fall outside the language-model vocabulary"
            )
        action_logits = full_action_logits.index_select(-1, action_token_ids)
        return NativeOFTForwardOutput(
            action_logits=action_logits,
            action_token_ids=action_token_ids,
            projected_tokens=projected,
            language_embedding=language_embedding,
            multimodal_attention_mask=multimodal_attention_mask,
            full_logits=lm_out.logits,
        )

    def prepare_prompt_batch(self, task_descriptions: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize and left-pad deployment prompts for latent-only decoding.

        Raw observations obtain these exact tensors from the full processor in
        :class:`OFTRolloutHiddenExtractor`.  A world-model observation has no image,
        so this method runs the same tokenizer branch directly and applies the same
        BOS-at-zero left-padding convention used by batched raw inference.
        """

        if self.processor is None:
            raise RuntimeError("prepare_prompt_batch requires a checkpoint-loaded processor")
        descriptions = [str(value) for value in task_descriptions]
        if not descriptions:
            raise ValueError("prepare_prompt_batch requires at least one task")
        tokenizer = self.processor.tokenizer
        ids_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        for description in descriptions:
            cached = self._prompt_token_cache.get(description)
            if cached is None:
                prompt = f"In: What action should the robot take to {description.lower()}?\nOut:"
                encoded = tokenizer(prompt, return_tensors="pt")
                cached = (
                    encoded["input_ids"].detach().cpu(),
                    encoded["attention_mask"].detach().cpu(),
                )
                self._prompt_token_cache[description] = cached
            ids_list.append(cached[0].to(self.device))
            masks_list.append(cached[1].to(self.device))

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        input_ids, attention_mask = _left_pad_prompt_batch(
            ids_list,
            masks_list,
            pad_token_id=int(pad_id),
            bos_token_id=int(bos_id),
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def action_token_loss(
        self,
        output: NativeOFTForwardOutput,
        action_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Cross entropy for sampled native action-token ids or bin indices."""

        labels = self._action_class_labels(output, action_tokens)
        if labels.shape != output.action_logits.shape[:-1]:
            raise ValueError("action token labels must match action logits [B,action_slots]")
        num_bins = int(output.action_logits.shape[-1])
        loss = F.cross_entropy(output.action_logits.reshape(-1, num_bins), labels.reshape(-1))
        return loss, {
            "action_token_loss": float(loss.detach().item()),
            "action_token_count": float(labels.numel()),
        }

    def _action_class_labels(
        self,
        output: NativeOFTForwardOutput,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        labels = action_tokens.to(
            device=output.action_logits.device,
            dtype=torch.long,
        )
        num_bins = int(output.action_logits.shape[-1])
        if labels.numel() and (
            int(labels.min().item()) < 0 or int(labels.max().item()) >= num_bins
        ):
            labels = int(self.vla.vocab_size) - labels - 1
        if labels.numel() and (
            int(labels.min().item()) < 0 or int(labels.max().item()) >= num_bins
        ):
            raise ValueError("action token labels are outside the checkpoint action bins")
        return labels

    def forward(self, batch: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        """ActorGroup-compatible native OpenVLA sample/evaluate interface."""

        if not isinstance(batch, dict):
            raise TypeError("OpenVLAOFTPolicy.forward expects a mapping")
        mode = str(batch.get("mode", "sample")).lower()
        if mode not in {"sample", "evaluate", "encoder_sft", "encode_raw"}:
            raise ValueError(
                "OpenVLAOFTPolicy mode must be sample, evaluate, encoder_sft, or encode_raw"
            )
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise KeyError("native OpenVLA actor requires input_ids and attention_mask")
        hidden = batch.get("hidden")
        pixel_values = batch.get("pixel_values")
        if not isinstance(hidden, torch.Tensor):
            hidden = None
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = None
        if mode == "encode_raw":
            if pixel_values is None:
                raise KeyError("encode_raw requires pixel_values")
            projected, language_embedding = self.encode_raw(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            extras = {
                "hidden": projected,
                "lang_emb": language_embedding,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            return projected, projected.new_zeros(()), extras
        output = self.forward_action_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            projected_tokens=hidden,
        )
        if mode == "encoder_sft":
            raw_labels = batch.get("action_token_ids")
            if not isinstance(raw_labels, torch.Tensor):
                raise KeyError("encoder_sft requires action_token_ids")
            labels = raw_labels.reshape(output.action_logits.shape[:-1])
            loss, metrics = self.action_token_loss(output, labels)
            class_labels = self._action_class_labels(output, labels)
            label_logprobs = (
                torch.log_softmax(
                    output.action_logits.float(),
                    dim=-1,
                )
                .gather(-1, class_labels.unsqueeze(-1))
                .squeeze(-1)
            )
            return (
                loss,
                loss.new_zeros(()),
                {
                    "hidden": output.projected_tokens,
                    "lang_emb": output.language_embedding,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "action_logits": output.action_logits,
                    "action_label_logprobs": label_logprobs,
                    **metrics,
                },
            )
        distribution = Categorical(logits=output.action_logits.float())
        if mode == "sample":
            if bool(batch.get("deterministic", False)):
                action_classes = output.action_logits.argmax(dim=-1)
            else:
                action_classes = distribution.sample()
        else:
            raw_ids = batch.get("action_token_ids")
            if not isinstance(raw_ids, torch.Tensor):
                raise KeyError("native OpenVLA evaluation requires sampled action_token_ids")
            action_classes = raw_ids.to(
                device=output.action_logits.device, dtype=torch.long
            ).reshape(output.action_logits.shape[:-1])
            num_bins = int(output.action_logits.shape[-1])
            if action_classes.numel() and (
                int(action_classes.min().item()) < 0 or int(action_classes.max().item()) >= num_bins
            ):
                action_classes = int(self.vla.vocab_size) - action_classes - 1
            if action_classes.numel() and (
                int(action_classes.min().item()) < 0 or int(action_classes.max().item()) >= num_bins
            ):
                raise ValueError("sampled action token ids are outside action bins")

        token_ids = output.action_token_ids.index_select(0, action_classes.reshape(-1)).reshape_as(
            action_classes
        )
        log_prob = distribution.log_prob(action_classes)
        entropy = distribution.entropy()
        batch_size, action_slots = action_classes.shape
        expected_slots = int(self.time_horizon * self.action_dim)
        if action_slots != expected_slots:
            raise ValueError(
                "checkpoint action slot count does not match configured action geometry: "
                f"{action_slots} != {self.time_horizon} * {self.action_dim}"
            )
        token_ids = token_ids.reshape(batch_size, self.time_horizon, self.action_dim)
        log_prob = log_prob.reshape(batch_size, self.time_horizon, self.action_dim)
        entropy = entropy.reshape(batch_size, self.time_horizon, self.action_dim)
        extras = {
            "action_token_ids": token_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "hidden": output.projected_tokens,
            "lang_emb": output.language_embedding,
        }
        if str(batch.get("logprob_type", "")).lower() != "token_level":
            log_prob = log_prob.sum(dim=(-1, -2))
            entropy = entropy.sum(dim=(-1, -2))
        if mode == "evaluate":
            return log_prob, entropy, extras

        centers = torch.as_tensor(
            np.asarray(self.vla.bin_centers),
            dtype=torch.float32,
            device=action_classes.device,
        )
        normalized_actions = centers.index_select(0, action_classes.reshape(-1)).reshape(
            batch_size, self.time_horizon, self.action_dim
        )
        unnormalized = self.vla._unnormalize_actions(
            normalized_actions.detach().cpu().numpy(), self.unnorm_key
        )
        actions = torch.as_tensor(
            unnormalized,
            dtype=torch.float32,
            device=action_classes.device,
        )
        return actions, log_prob, extras

    def make_extractor(self) -> Any:
        """Create the raw-observation preprocessor bound to this policy copy."""

        if self.processor is None:
            raise RuntimeError("make_extractor requires a checkpoint-loaded processor")
        from dreamervla.runners.rollout_hidden_extractor import (
            OFTRolloutHiddenExtractor,
        )

        return OFTRolloutHiddenExtractor(
            self,
            image_keys=self.image_keys,
            history=self.history,
            rotate_images_180=self.rotate_images_180,
            center_crop=self.center_crop,
            unnorm_key=self.unnorm_key,
        )

    def compute_loss(
        self, batch: dict[str, Any], device: torch.device | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = device or self.device
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device).to(torch.bfloat16)

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
                proprio=None,
                proprio_projector=None,
                use_film=self.use_film,
            )

        metrics: dict[str, float] = {}
        ground_truth_token_ids = labels[:, 1:].to(device)
        explicit_mask = batch.get("action_token_mask")
        if explicit_mask is not None:
            current_action_mask = explicit_mask.to(device)
            next_actions_mask = torch.zeros_like(current_action_mask)
        else:
            ensure_openvla_oft_on_path()
            from prismatic.training.train_utils import (
                get_current_action_mask,
                get_next_actions_mask,
            )

            current_action_mask = get_current_action_mask(ground_truth_token_ids)
            next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        ensure_openvla_oft_on_path()
        from prismatic.training.train_utils import (
            compute_actions_l1_loss,
            compute_token_accuracy,
        )

        loss = output.loss
        predicted_token_ids = output.logits[:, self.num_patches : -1].argmax(dim=2)
        metrics.update(
            {
                "loss_value": float(loss.detach().item()),
                "curr_action_accuracy": float(
                    compute_token_accuracy(
                        predicted_token_ids,
                        ground_truth_token_ids,
                        mask=current_action_mask,
                    ).item()
                ),
                "curr_action_l1_loss": float(
                    compute_actions_l1_loss(
                        self.action_tokenizer,
                        predicted_token_ids,
                        ground_truth_token_ids,
                        mask=current_action_mask,
                    ).item()
                ),
                "next_actions_accuracy": float(
                    compute_token_accuracy(
                        predicted_token_ids,
                        ground_truth_token_ids,
                        mask=next_actions_mask,
                    ).item()
                ),
                "next_actions_l1_loss": float(
                    compute_actions_l1_loss(
                        self.action_tokenizer,
                        predicted_token_ids,
                        ground_truth_token_ids,
                        mask=next_actions_mask,
                    ).item()
                ),
            }
        )
        return loss, metrics


__all__ = ["NativeOFTForwardOutput", "OpenVLAOFTPolicy"]
