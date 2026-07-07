from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers.utils import logging as hf_logging

from dreamervla.constants import DEFAULT_ACTION_TOKEN_ID
from dreamervla.models.embodiment.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
)
from dreamervla.utils.paths import checkpoints_path

from .base_encoder import BaseEncoder
from .protocol import EncoderInputBatch
from .rynnvla_runtime import FlexARItemProcessorActionState

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _default_ckpt_path(*parts: str) -> str:
    return str(checkpoints_path(*parts).resolve())


def _image_content_token_spans(
    tokens: list[int], *, start_id: int, end_id: int, new_line_id: int
) -> list[list[int]]:
    """Per-image VQ content token ids; grid-size, newline, and marker tokens
    stripped. Mirrors ``preprocess._image_content_token_spans`` so the online
    backbone-latent extractor matches the offline input-token sidecar layout."""
    spans: list[list[int]] = []
    inside = False
    current: list[int] = []
    skip = 0
    for tok in tokens:
        if tok == start_id:
            inside, current, skip = True, [], 2
            continue
        if not inside:
            continue
        if tok == end_id:
            spans.append(current)
            inside = False
            continue
        if skip > 0:
            skip -= 1
            continue
        if tok == new_line_id:
            continue
        current.append(tok)
    return spans


def _resolve_pretrained_model_dir(path: str | Path) -> Path:
    candidate = _resolve_path(path)
    if candidate.is_file():
        return candidate.parent
    if candidate.is_dir():
        if (candidate / "config.json").is_file():
            return candidate
        for subdir in sorted(item for item in candidate.iterdir() if item.is_dir()):
            if (subdir / "config.json").is_file():
                return subdir.resolve()
    raise FileNotFoundError(
        f"Unable to locate a Hugging Face checkpoint directory under: {candidate}"
    )


@dataclass
class RynnVLAEncoderOutput:
    hidden: torch.Tensor
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    input_ids: list[list[int]]
    lengths: list[int]


class RynnVLAEncoder(BaseEncoder):
    def __init__(
        self,
        model_path: str = _default_ckpt_path("VLA_model_256", "libero_10"),
        tokenizer_path: str = _default_ckpt_path(
            "models--Alpha-VLLM--Lumina-mGPT-7B-768"
        ),
        text_tokenizer_path: str = _default_ckpt_path(
            "chameleon", "tokenizer", "text_tokenizer.json"
        ),
        chameleon_vqgan_config: str = _default_ckpt_path(
            "chameleon", "tokenizer", "vqgan.yaml"
        ),
        chameleon_vqgan_ckpt: str = _default_ckpt_path(
            "chameleon", "tokenizer", "vqgan.ckpt"
        ),
        resolution: int = 256,
        action_dim: int = 7,
        time_horizon: int = 5,
        action_head_type: str = "legacy",
        pool: str = "mean",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model_path = str(_resolve_pretrained_model_dir(model_path))
        self.tokenizer_path = str(_resolve_path(tokenizer_path))
        self.text_tokenizer_path = str(_resolve_path(text_tokenizer_path))
        self.chameleon_vqgan_config = str(_resolve_path(chameleon_vqgan_config))
        self.chameleon_vqgan_ckpt = str(_resolve_path(chameleon_vqgan_ckpt))
        for required_path in (
            self.model_path,
            self.tokenizer_path,
            self.text_tokenizer_path,
            self.chameleon_vqgan_config,
            self.chameleon_vqgan_ckpt,
        ):
            if not Path(required_path).exists():
                raise FileNotFoundError(
                    f"Required encoder asset not found: {required_path}. "
                    "Please check the local checkpoint layout under data/checkpoints/."
                )
        self.resolution = int(resolution)
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.action_head_type = str(action_head_type)
        self.pool = str(pool)

        self._processor: FlexARItemProcessorActionState | None = None
        hf_verbosity = hf_logging.get_verbosity()
        hf_progress_enabled = hf_logging.is_progress_bar_enabled()
        try:
            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
            self.backbone = (
                ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
                    self.model_path,
                    action_dim=self.action_dim,
                    time_horizon=self.time_horizon,
                    action_head_type=self.action_head_type,
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16,
                    ignore_mismatched_sizes=self.action_head_type != "legacy",
                    low_cpu_mem_usage=False,
                )
            )
        finally:
            hf_logging.set_verbosity(hf_verbosity)
            if hf_progress_enabled:
                hf_logging.enable_progress_bar()
        if hasattr(self.backbone.model, "vqmodel"):
            del self.backbone.model.vqmodel
        if freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _build_processor(self, device: torch.device) -> FlexARItemProcessorActionState:
        if self._processor is None or self._processor.device != str(device):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*torch\.load.*weights_only=False.*",
                    category=FutureWarning,
                )
                with contextlib.redirect_stdout(io.StringIO()):
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
        num_actions: int | None = None,
    ) -> list[dict[str, Any]]:
        state_placeholder = "<|state|>" if has_state else ""
        image_placeholders = "<|image|>" * num_images
        if task_type == "action":
            human_value = prompt_text + state_placeholder + image_placeholders
            assistant_value = "<|action|>" * max(int(num_actions or 0), 1)
        elif task_type == "world":
            num_actions = max(int(num_actions or 0), 1)
            if num_images == num_actions * 2:
                human_value = prompt_text + "<|image|><|image|><|action|>" * num_actions
                assistant_value = "<|image|><|image|>"
            else:
                human_value = prompt_text + "<|image|><|action|>" * num_actions
                assistant_value = "<|image|>"
        else:
            human_value = prompt_text + state_placeholder + image_placeholders
            assistant_value = None
        return [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": assistant_value},
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
        conversations: list[dict[str, Any]] = []
        if idx < len(batch.conversations):
            raw_conversations = batch.conversations[idx]
            if raw_conversations:
                conversations = [dict(message) for message in raw_conversations]

        num_actions = None
        if batch.action_mask is not None:
            num_actions = int(batch.action_mask[idx].sum().item())
        elif batch.action is not None:
            num_actions = int(batch.action[idx].shape[0])
        elif batch.meta is not None and idx < len(batch.meta):
            action_indices = batch.meta[idx].get("action_indices")
            if action_indices is not None:
                num_actions = len(action_indices)

        if not conversations:
            conversations = self._build_observation_conversation(
                prompt_text=prompt_text,
                task_type=task_type,
                num_images=len(images),
                has_state=state is not None,
                num_actions=num_actions,
            )
        data_item = {
            "conversations": conversations,
            "image": images,
        }
        if state is not None:
            data_item["state"] = state
        return data_item

    @staticmethod
    def _record_to_data_item(record: dict[str, Any]) -> dict[str, Any]:
        data_item = {
            "conversations": list(record.get("conversations", [])),
            "image": list(record.get("image", [])),
            "action": list(record.get("action", [])),
        }
        state = record.get("state", [])
        if state:
            data_item["state"] = list(state)
        return data_item

    def compute_action_sft_loss(
        self,
        records: list[dict[str, Any]],
        token_loss_coef: float = 1.0,
        action_loss_coef: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if not records:
            raise ValueError(
                "compute_action_sft_loss requires at least one action record."
            )

        device = self.device
        processor = self._build_processor(device)
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        for record in records:
            data_item = self._record_to_data_item(record)
            input_ids, labels = processor.process_item(data_item, training_mode=True)
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        (
            token_loss,
            additional_loss_dict,
            _logits,
            _hidden_states,
            _labels_tensor,
            predicted_actions,
            action_loss,
        ) = self.backbone(
            input_ids=input_ids_list,
            labels=labels_list,
            training=True,
            output_hidden_states=True,
            att_mask=True,
        )

        total_loss = token_loss_coef * token_loss + action_loss_coef * action_loss
        z_loss = token_loss.new_zeros(())
        for key, value in additional_loss_dict.items():
            if not isinstance(value, tuple) or len(value) != 2:
                continue
            extra_loss, weight = value
            total_loss = total_loss + float(weight) * extra_loss
            if key == "z_loss":
                z_loss = extra_loss

        return {
            "loss": total_loss,
            "token_loss": token_loss,
            "action_loss": action_loss,
            "z_loss": z_loss,
            "predicted_action_mean": predicted_actions.float().mean(),
        }

    def compute_action_sft_loss_from_tokenized(
        self,
        input_ids_list: list[list[int]],
        labels_list: list[list[int]],
        token_loss_coef: float = 1.0,
        action_loss_coef: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if not input_ids_list or not labels_list:
            raise ValueError(
                "compute_action_sft_loss_from_tokenized requires non-empty tokenized samples."
            )

        (
            token_loss,
            additional_loss_dict,
            _logits,
            _hidden_states,
            _labels_tensor,
            predicted_actions,
            action_loss,
        ) = self.backbone(
            input_ids=input_ids_list,
            labels=labels_list,
            training=True,
            output_hidden_states=True,
            att_mask=True,
        )

        total_loss = token_loss_coef * token_loss + action_loss_coef * action_loss
        z_loss = token_loss.new_zeros(())
        for key, value in additional_loss_dict.items():
            if not isinstance(value, tuple) or len(value) != 2:
                continue
            extra_loss, weight = value
            total_loss = total_loss + float(weight) * extra_loss
            if key == "z_loss":
                z_loss = extra_loss

        return {
            "loss": total_loss,
            "token_loss": token_loss,
            "action_loss": action_loss,
            "z_loss": z_loss,
            "predicted_action_mean": predicted_actions.float().mean(),
        }

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

        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*nested_from_padded CUDA kernels only support.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            _, _, _, hidden_states, labels_tensor, _, _ = self.backbone(
                input_ids=input_ids_list,
                labels=labels_list,
                training=True,
                output_hidden_states=True,
                att_mask=False,
            )

        attention_mask = torch.zeros(
            hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
        )
        for idx, length in enumerate(lengths):
            attention_mask[idx, :length] = True

        if self.pool == "last":
            pooled = torch.stack(
                [hidden_states[idx, length - 1] for idx, length in enumerate(lengths)],
                dim=0,
            )
        else:
            weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1.0)

        return RynnVLAEncoderOutput(
            hidden=pooled.float(),
            hidden_states=hidden_states.float(),
            attention_mask=attention_mask,
            labels=labels_tensor,
            input_ids=input_ids_list,
            lengths=lengths,
        )

    def extract_action_hidden(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        target_token_id: int = DEFAULT_ACTION_TOKEN_ID,
        eval: bool = True,
    ) -> torch.Tensor:
        action_head = getattr(self.backbone, "action_head", None)
        if action_head is None or not hasattr(action_head, "extract_action_hidden"):
            raise ValueError(
                "RynnVLAEncoder.extract_action_hidden requires an action head "
                "that exposes extract_action_hidden; use action_head_type='legacy'."
            )
        try:
            param = next(action_head.parameters())
            hidden_states = hidden_states.to(device=param.device, dtype=param.dtype)
            input_ids = input_ids.to(device=param.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=param.device)
        except StopIteration:
            pass
        action_hidden, ok = action_head.extract_action_hidden(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_token_id=int(target_token_id),
            eval=bool(eval),
        )
        if not ok:
            raise ValueError("RynnVLA action head did not find a usable action context")
        return action_hidden.float()

    def extract_input_token_embedding(
        self,
        *,
        input_ids_list: list[list[int]],
        processor: Any,
        num_views: int,
    ) -> torch.Tensor:
        """Scheme-1 backbone latent: current-frame VQ image content tokens through
        the backbone input-embedding table (no transformer forward). Mirrors
        ``preprocess._input_token_embedding_obs``. Returns ``[T, N*token_dim]``.

        This is the *pre-Action-Query* visual-language latent (DINO-style), the
        online counterpart of the offline input-token sidecar.
        """
        start_id = int(processor.token2id(processor.image_start_token))
        end_id = int(processor.token2id(processor.image_end_token))
        new_line_id = int(processor.token2id(processor.new_line_token))
        backbone = self.backbone
        embed = (
            backbone.get_input_embeddings()
            if hasattr(backbone, "get_input_embeddings")
            else backbone.model.embed_tokens
        )
        frames: list[torch.Tensor] = []
        expected: int | None = None
        for tokens in input_ids_list:
            spans = _image_content_token_spans(
                [int(t) for t in tokens],
                start_id=start_id,
                end_id=end_id,
                new_line_id=new_line_id,
            )
            if len(spans) < num_views:
                raise RuntimeError(
                    "extract_input_token_embedding expected >= "
                    f"{num_views} image spans per frame, got {len(spans)}"
                )
            # Record layout is [history ... current] x views; current frame is last.
            current = [tok for span in spans[-num_views:] for tok in span]
            if expected is None:
                expected = len(current)
            elif len(current) != expected:
                raise RuntimeError(
                    f"inconsistent image token count across frames: {len(current)} != {expected}"
                )
            ids = torch.as_tensor(current, dtype=torch.long, device=self.device)
            with torch.no_grad():
                emb = embed(ids)
            frames.append(emb.reshape(-1).float())
        return torch.stack(frames, dim=0)

    def encode(self, obs: dict[str, Any]) -> torch.Tensor:
        meta = obs.get("meta") if isinstance(obs.get("meta"), list) else None
        batch = self.prepare_inputs(obs, action=None, action_mask=None, meta=meta)
        return self.encode_inputs(batch).hidden


__all__ = ["RynnVLAEncoder", "RynnVLAEncoderOutput"]
