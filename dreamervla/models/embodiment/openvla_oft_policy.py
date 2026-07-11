from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from dreamervla.preprocess.sidecar_schema import INPUT_TOKEN_COUNT, INPUT_TOKEN_DIM
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


def _validate_loaded_input_token_geometry(vla: Any) -> int:
    """Validate the actual loaded backbone at the policy construction boundary."""

    try:
        patches = int(vla.vision_backbone.get_num_patches())
        num_images = int(vla.vision_backbone.get_num_images_in_input())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "loaded OpenVLA-OFT policy must expose vision patch geometry"
        ) from exc
    token_dim = _resolve_loaded_token_dim(vla)
    if (
        patches != INPUT_TOKEN_COUNT
        or token_dim != INPUT_TOKEN_DIM
        or num_images != 1
    ):
        raise ValueError(
            "loaded OpenVLA-OFT policy violates the input-token contract: "
            f"patches={patches}, token_dim={token_dim}, "
            f"num_images_in_input={num_images}; expected "
            f"{INPUT_TOKEN_COUNT}x{INPUT_TOKEN_DIM} from one image"
        )
    return INPUT_TOKEN_COUNT


class OpenVLAOFTPolicy(nn.Module):
    """DreamerVLA wrapper around OpenVLA-OFT policy fine-tuning components."""

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "bf16",
        num_images_in_input: int = 1,
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
    ) -> None:
        super().__init__()
        if bool(use_l1_regression):
            raise ValueError("L1/action-query checkpoints are closed")
        if bool(use_proprio):
            raise ValueError("OpenVLA-OFT input-token mainline does not include proprio")
        if bool(use_film):
            raise ValueError("OpenVLA-OFT input-token mainline does not use FiLM")
        if int(num_images_in_input) != 1:
            raise ValueError("OpenVLA-OFT input-token mainline requires num_images_in_input=1")
        if use_diffusion:
            raise NotImplementedError(
                "DreamerVLA OpenVLA-OFT workspace currently does not implement diffusion training."
            )
        from dreamervla.preprocess.preprocess_oft_input_tokens import (
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
        AutoImageProcessor.register(
            OpenVLAConfig, PrismaticImageProcessor, exist_ok=True
        )
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(
            OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True
        )

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.use_l1_regression = False
        self.use_diffusion = False
        self.use_proprio = False
        self.use_film = False
        self.num_images_in_input = int(num_images_in_input)

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
        num_patches = _validate_loaded_input_token_geometry(vla)
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
        self.num_patches = num_patches

    @classmethod
    def from_modules(
        cls,
        *,
        vla: nn.Module,
        action_head: nn.Module | None,
        action_tokenizer: Any,
        num_patches: int,
        use_l1_regression: bool = False,
        use_diffusion: bool = False,
        use_proprio: bool = False,
        use_film: bool = False,
        proprio_projector: nn.Module | None = None,
    ) -> OpenVLAOFTPolicy:
        if bool(use_l1_regression) or action_head is not None:
            raise ValueError("L1/action-query checkpoints are closed")
        if bool(use_proprio) or proprio_projector is not None:
            raise ValueError("OpenVLA-OFT input-token mainline does not include proprio")
        if bool(use_diffusion):
            raise ValueError("diffusion checkpoints are outside the discrete mainline")
        if bool(use_film):
            raise ValueError("OpenVLA-OFT input-token mainline does not use FiLM")
        if int(num_patches) != INPUT_TOKEN_COUNT:
            raise ValueError(
                f"OpenVLA-OFT input-token mainline requires num_patches={INPUT_TOKEN_COUNT}, "
                f"got {int(num_patches)}"
            )
        _validate_loaded_input_token_geometry(vla)
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.vla = vla
        self.action_head = None
        self.proprio_projector = None
        self.action_tokenizer = action_tokenizer
        self.num_patches = int(num_patches)
        self.use_l1_regression = False
        self.use_diffusion = False
        self.use_proprio = False
        self.use_film = False
        self.processor = None
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

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


__all__ = ["OpenVLAOFTPolicy"]
