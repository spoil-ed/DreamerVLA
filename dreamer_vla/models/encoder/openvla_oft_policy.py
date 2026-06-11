from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from dreamer_vla.utils.openvla_oft_imports import ensure_openvla_oft_on_path


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


def _strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _load_component_state(
    module: nn.Module,
    component_dir: str | Path,
    component_name: str,
    step: int | None,
) -> None:
    component_dir = Path(component_dir).expanduser().resolve()
    if step is not None:
        path = component_dir / f"{component_name}--{int(step)}_checkpoint.pt"
    else:
        matches = sorted(component_dir.glob(f"{component_name}--*_checkpoint.pt"))
        if not matches:
            return
        path = matches[-1]
    if not path.is_file():
        raise FileNotFoundError(f"OpenVLA-OFT component checkpoint not found: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(_strip_module_prefix(state_dict))


class OpenVLAOFTPolicy(nn.Module):
    """DreamerVLA wrapper around OpenVLA-OFT policy fine-tuning components."""

    def __init__(
        self,
        model_path: str,
        component_ckpt_dir: str | None = None,
        resume_step: int | None = None,
        torch_dtype: str = "bf16",
        num_images_in_input: int = 2,
        use_lora: bool = True,
        lora_rank: int = 32,
        lora_dropout: float = 0.0,
        use_l1_regression: bool = True,
        use_diffusion: bool = False,
        use_proprio: bool = True,
        use_film: bool = False,
        freeze_vla_backbone: bool = False,
        low_cpu_mem_usage: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        if use_diffusion:
            raise NotImplementedError(
                "DreamerVLA OpenVLA-OFT workspace currently does not implement diffusion training."
            )
        ensure_openvla_oft_on_path()

        from peft import LoraConfig, get_peft_model
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM
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
        self.component_ckpt_dir = str(
            Path(component_ckpt_dir or model_path).expanduser().resolve()
        )
        self.resume_step = resume_step
        self.use_l1_regression = bool(use_l1_regression)
        self.use_diffusion = bool(use_diffusion)
        self.use_proprio = bool(use_proprio)
        self.use_film = bool(use_film)
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
        if self.use_l1_regression:
            from prismatic.models.action_heads import L1RegressionActionHead

            self.action_head = L1RegressionActionHead(
                input_dim=int(self.vla.llm_dim),
                hidden_dim=int(self.vla.llm_dim),
                action_dim=ACTION_DIM,
            ).to(dtype=dtype)
            _load_component_state(
                self.action_head,
                self.component_ckpt_dir,
                "action_head",
                self.resume_step,
            )

        self.proprio_projector = None
        if self.use_proprio:
            self.proprio_projector = ProprioProjector(
                llm_dim=int(self.vla.llm_dim), proprio_dim=PROPRIO_DIM
            )
            _load_component_state(
                self.proprio_projector,
                self.component_ckpt_dir,
                "proprio_projector",
                self.resume_step,
            )

        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.num_patches = (
            self.vla.vision_backbone.get_num_patches()
            * self.vla.vision_backbone.get_num_images_in_input()
        )
        if self.use_proprio:
            self.num_patches += 1

    @classmethod
    def from_modules(
        cls,
        *,
        vla: nn.Module,
        action_head: nn.Module | None,
        action_tokenizer: Any,
        num_patches: int,
        use_l1_regression: bool = True,
        use_diffusion: bool = False,
        use_proprio: bool = False,
        use_film: bool = False,
        proprio_projector: nn.Module | None = None,
    ) -> OpenVLAOFTPolicy:
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.vla = vla
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.action_tokenizer = action_tokenizer
        self.num_patches = int(num_patches)
        self.use_l1_regression = bool(use_l1_regression)
        self.use_diffusion = bool(use_diffusion)
        self.use_proprio = bool(use_proprio)
        self.use_film = bool(use_film)
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
        ground_truth_actions = batch["actions"].to(device).to(torch.bfloat16)
        num_actions_chunk = int(ground_truth_actions.shape[1])
        action_dim = int(ground_truth_actions.shape[2])
        pixel_values = batch["pixel_values"].to(device).to(torch.bfloat16)
        proprio = batch.get("proprio")
        if proprio is not None:
            proprio = proprio.to(device)

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
                proprio=proprio if self.use_proprio else None,
                proprio_projector=self.proprio_projector if self.use_proprio else None,
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

        if not self.use_l1_regression:
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

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, self.num_patches : -1]
        action_mask = current_action_mask | next_actions_mask
        batch_size = input_ids.shape[0]
        if self.action_head is None:
            raise RuntimeError(
                "use_l1_regression=True requires an OpenVLA-OFT L1 action_head."
            )
        actions_hidden_states = text_hidden_states[action_mask].reshape(
            batch_size,
            num_actions_chunk * action_dim,
            -1,
        )
        predicted_actions = self.action_head.predict_action(
            actions_hidden_states.to(torch.bfloat16)
        )
        loss = torch.nn.functional.l1_loss(predicted_actions, ground_truth_actions)
        metrics["loss_value"] = float(loss.detach().item())
        metrics["curr_action_l1_loss"] = float(
            torch.nn.functional.l1_loss(
                predicted_actions[:, 0], ground_truth_actions[:, 0]
            )
            .detach()
            .item()
        )
        metrics["next_actions_l1_loss"] = float(
            torch.nn.functional.l1_loss(
                predicted_actions[:, 1:], ground_truth_actions[:, 1:]
            )
            .detach()
            .item()
        )
        return loss, metrics


__all__ = ["OpenVLAOFTPolicy"]
