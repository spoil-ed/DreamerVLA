"""GR00T N1.7 flow-matching action head migrated from SiPAI.

Source: ``SIPAI@9672af6/sipai/models/action_heads/gr00t.py``. Parameter names
and tensor geometry intentionally match the SiPAI implementation so exported
SiPAI action-head weights load without key remapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from dreamervla.models.embodiment.qwen_groot.dit import (
    DiT,
    SelfAttentionTransformer,
)


def _swish(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim,
            dtype=torch.float,
            device=timesteps.device,
        ) * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / half_dim)
        frequencies = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=-1)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(value)))


class _CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific GR00T embodiment weights."""

    def __init__(self, num_categories: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, output_dim))

    def forward(self, value: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        return torch.bmm(value, self.W[category_ids]) + self.b[category_ids].unsqueeze(1)


class _CategorySpecificMLP(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.layer1 = _CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = _CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, value: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(value, category_ids)), category_ids)


class _ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = _SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.dim() != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [B]")
        timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        action_embedding = self.layer1(actions)
        time_embedding = self.pos_encoding(timesteps).to(dtype=action_embedding.dtype)
        hidden = _swish(self.layer2(torch.cat([action_embedding, time_embedding], dim=-1)))
        return self.layer3(hidden)


class _MultiEmbodimentActionEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        num_embodiments: int,
    ) -> None:
        super().__init__()
        self.W1 = _CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = _CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = _CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = _SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.dim() != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [B]")
        timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        action_embedding = self.W1(actions, category_ids)
        time_embedding = self.pos_encoding(timesteps).to(dtype=action_embedding.dtype)
        hidden = _swish(
            self.W2(
                torch.cat([action_embedding, time_embedding], dim=-1),
                category_ids,
            )
        )
        return self.W3(hidden, category_ids)


class _AlternateVLDiT(DiT):
    """Alternate cross-attention over Qwen text and image tokens."""

    def __init__(
        self,
        *args: Any,
        attend_text_every_n_blocks: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = int(attend_text_every_n_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        image_mask: torch.Tensor | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if image_mask is None:
            raise ValueError("alternate GR00T DiT requires image_mask")
        if timestep is None:
            raise ValueError("alternate GR00T DiT requires timesteps")
        if backbone_attention_mask is None:
            backbone_attention_mask = torch.ones_like(image_mask, dtype=torch.bool)
        timestep_embedding = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        image_attention_mask = image_mask.bool() & backbone_attention_mask.bool()
        text_attention_mask = (~image_mask.bool()) & backbone_attention_mask.bool()
        all_hidden_states = [hidden_states]
        if not self.config.interleave_self_attention:
            raise ValueError("alternate GR00T DiT requires interleave_self_attention")

        for index, block in enumerate(self.transformer_blocks):
            if index % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=timestep_embedding,
                )
            else:
                current_mask = (
                    text_attention_mask
                    if index % (2 * self.attend_text_every_n_blocks) == 0
                    else image_attention_mask
                )
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=current_mask,
                    temb=timestep_embedding,
                )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(F.silu(timestep_embedding)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)
        return (output, all_hidden_states) if return_all_hidden_states else output


_DIT_VARIANTS = {
    "DiT-B": {
        "input_embedding_dim": 768,
        "attention_head_dim": 64,
        "num_attention_heads": 12,
    },
    "DiT-L": {
        "input_embedding_dim": 1536,
        "attention_head_dim": 48,
        "num_attention_heads": 32,
    },
}

_DEFAULT_DIFFUSION_MODEL_CFG = {
    "dropout": 0.2,
    "final_dropout": True,
    "interleave_self_attention": True,
    "norm_type": "ada_norm",
    "num_layers": 32,
    "output_dim": 1024,
    "positional_embeddings": None,
}

_DEFAULT_VL_SELF_ATTENTION_CFG = {
    "attention_head_dim": 64,
    "dropout": 0.2,
    "final_dropout": True,
    "num_attention_heads": 32,
    "num_layers": 4,
    "positional_embeddings": None,
}


class GR00TActionHead(nn.Module):
    """SiPAI's GR00T N1.7-aligned flow-matching action head."""

    def __init__(
        self,
        *,
        variant: str = "DiT-L",
        action_horizon: int,
        action_dim: int | None = None,
        state_dim: int = 0,
        max_action_dim: int | None = None,
        max_state_dim: int | None = None,
        hidden_size: int = 1024,
        action_hidden_dim: int | None = None,
        input_embedding_dim: int | None = None,
        backbone_embedding_dim: int | None = None,
        diffusion_model_cfg: Mapping[str, Any] | None = None,
        vl_self_attention_cfg: Mapping[str, Any] | None = None,
        state_history_length: int = 1,
        max_num_embodiments: int = 32,
        add_pos_embed: bool = True,
        max_seq_len: int = 1024,
        use_vlln: bool = True,
        use_alternate_vl_dit: bool = True,
        attend_text_every_n_blocks: int = 2,
        state_dropout_prob: float = 0.2,
        repeated_diffusion_steps: int = 8,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        num_inference_timesteps: int = 4,
        num_target_vision_tokens: int | None = None,
    ) -> None:
        super().__init__()
        variant_cfg = dict(_DIT_VARIANTS.get(variant, {}))
        self.variant = variant
        self.hidden_size = int(action_hidden_dim or hidden_size)
        self.input_embedding_dim = int(
            input_embedding_dim or variant_cfg.get("input_embedding_dim", self.hidden_size)
        )
        diffusion_cfg = {
            **_DEFAULT_DIFFUSION_MODEL_CFG,
            **variant_cfg,
            **dict(diffusion_model_cfg or {}),
        }
        self.backbone_embedding_dim = int(
            backbone_embedding_dim
            or diffusion_cfg.get("cross_attention_dim", action_hidden_dim or hidden_size)
        )
        diffusion_cfg["cross_attention_dim"] = int(
            diffusion_cfg.get("cross_attention_dim", self.backbone_embedding_dim)
        )
        self.use_alternate_vl_dit = bool(use_alternate_vl_dit)
        self.model = (
            _AlternateVLDiT(
                **diffusion_cfg,
                attend_text_every_n_blocks=int(attend_text_every_n_blocks),
            )
            if self.use_alternate_vl_dit
            else DiT(**diffusion_cfg)
        )

        resolved_action_dim = max_action_dim if max_action_dim is not None else action_dim
        if resolved_action_dim is None:
            raise ValueError("GR00TActionHead requires action_dim")
        self.action_horizon = int(action_horizon)
        self.action_dim = int(resolved_action_dim)
        self.state_dim = int(max_state_dim if max_state_dim is not None else state_dim)
        self.state_history_length = int(state_history_length)
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.num_embodiments = int(max_num_embodiments)
        self.state_dropout_prob = float(state_dropout_prob)
        self.repeated_diffusion_steps = int(repeated_diffusion_steps)
        self.noise_s = float(noise_s)
        self.add_pos_embed = bool(add_pos_embed)
        self.num_target_vision_tokens = num_target_vision_tokens

        if self.num_embodiments > 1:
            self.state_encoder = _CategorySpecificMLP(
                self.num_embodiments,
                self.state_dim * self.state_history_length,
                self.hidden_size,
                self.input_embedding_dim,
            )
            self.action_encoder = _MultiEmbodimentActionEncoder(
                self.action_dim,
                self.input_embedding_dim,
                self.num_embodiments,
            )
            self.action_decoder = _CategorySpecificMLP(
                self.num_embodiments,
                int(self.model.config.output_dim),
                self.hidden_size,
                self.action_dim,
            )
        else:
            self.state_encoder = (
                _MLP(
                    self.state_dim * self.state_history_length,
                    self.hidden_size,
                    self.input_embedding_dim,
                )
                if self.state_dim
                else None
            )
            self.action_encoder = _ActionEncoder(
                self.action_dim,
                self.input_embedding_dim,
            )
            self.action_decoder = _MLP(
                int(self.model.config.output_dim),
                self.hidden_size,
                self.action_dim,
            )

        self.vlln = nn.LayerNorm(self.backbone_embedding_dim) if use_vlln else nn.Identity()
        vl_attention_cfg = {
            **_DEFAULT_VL_SELF_ATTENTION_CFG,
            **dict(vl_self_attention_cfg or {}),
        }
        self.vl_self_attention = (
            SelfAttentionTransformer(**vl_attention_cfg)
            if int(vl_attention_cfg.get("num_layers", 0)) > 0
            else nn.Identity()
        )
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.beta_dist = Beta(
            torch.tensor(noise_beta_alpha, dtype=torch.float32, device="cpu"),
            torch.tensor(noise_beta_beta, dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = int(num_timestep_buckets)

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.noise_s

    def _repeated_diffusion_steps(self) -> int:
        return self.repeated_diffusion_steps

    def _embodiment_id(
        self,
        encoded: Mapping[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        value = encoded.get("embodiment_id")
        if value is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        ids = torch.as_tensor(value, dtype=torch.long, device=device)
        if ids.ndim == 0:
            ids = ids.expand(batch_size)
        if tuple(ids.shape) != (batch_size,):
            raise ValueError(f"embodiment_id must be [B], got {tuple(ids.shape)}")
        return ids

    def _state_input(
        self,
        state: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.state_dim == 0:
            return None
        if state is None:
            state = torch.zeros(
                batch_size,
                self.state_history_length,
                self.state_dim,
                device=device,
                dtype=dtype,
            )
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"GR00T state width must be {self.state_dim}, got {state.shape[-1]}")
        if state.shape[1] != self.state_history_length:
            if state.shape[1] == 1:
                state = state.expand(-1, self.state_history_length, -1)
            else:
                raise ValueError("GR00T state history does not match state_history_length")
        return state.reshape(batch_size, 1, self.state_history_length * self.state_dim)

    def _encode_state(
        self,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor | None:
        if state is None or self.state_encoder is None:
            return None
        if self.num_embodiments > 1:
            return self.state_encoder(state, embodiment_id)
        return self.state_encoder(state)

    def _encode_action(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_embodiments > 1:
            return self.action_encoder(actions, timesteps, embodiment_id)
        return self.action_encoder(actions, timesteps)

    def _decode_action(
        self,
        model_output: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_embodiments > 1:
            return self.action_decoder(model_output, embodiment_id)
        return self.action_decoder(model_output)

    def _process_backbone(self, embeddings: torch.Tensor) -> torch.Tensor:
        # SiPAI normally relies on its DeepSpeed mixed-precision wrapper to
        # align the bf16 Qwen output and action-head parameters. DreamerVLA can
        # run the same module without that wrapper, so make the component
        # boundary explicit while preserving every parameter tensor unchanged.
        embeddings = embeddings.to(dtype=self.dtype)
        return self.vl_self_attention(self.vlln(embeddings))

    def _model_forward(
        self,
        state_action_embeddings: torch.Tensor,
        vl_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_alternate_vl_dit:
            if image_mask is None:
                raise ValueError("alternate GR00T DiT requires encoded image_mask")
            output = self.model(
                hidden_states=state_action_embeddings,
                encoder_hidden_states=vl_embeddings,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps,
                image_mask=image_mask,
                backbone_attention_mask=encoder_attention_mask,
            )
        else:
            output = self.model(
                hidden_states=state_action_embeddings,
                encoder_hidden_states=vl_embeddings,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps,
            )
        if not isinstance(output, torch.Tensor):
            raise TypeError("GR00T DiT unexpectedly returned hidden-state history")
        return output

    def _flow_matching_loss(
        self,
        vl_embeddings: torch.Tensor,
        actions: torch.Tensor,
        *,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vl_embeddings = self._process_backbone(vl_embeddings)
        noise = torch.randn_like(actions)
        time = self.sample_time(actions.shape[0], actions.device, actions.dtype)[:, None, None]
        noisy_trajectory = (1 - time) * noise + time * actions
        velocity = actions - noise
        discretized_time = (time[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self._encode_action(
            noisy_trajectory,
            discretized_time,
            embodiment_id,
        )
        state_features = self._encode_state(state, embodiment_id)
        if self.training and self.state_dropout_prob > 0 and state_features is not None:
            drop = torch.rand(state_features.shape[0], device=state_features.device)
            keep = (drop >= self.state_dropout_prob)[:, None, None]
            state_features = state_features * keep.to(dtype=state_features.dtype)
        if self.add_pos_embed:
            position_ids = torch.arange(action_features.shape[1], device=actions.device)
            action_features = action_features + self.position_embedding(position_ids).unsqueeze(0)
        state_action_embeddings = (
            torch.cat((state_features, action_features), dim=1)
            if state_features is not None
            else action_features
        )
        output = self._model_forward(
            state_action_embeddings,
            vl_embeddings,
            discretized_time,
            encoder_attention_mask=encoder_attention_mask,
            image_mask=image_mask,
        )
        prediction = self._decode_action(output, embodiment_id)[:, -actions.shape[1] :]
        loss = F.mse_loss(prediction, velocity, reduction="none")
        if action_mask is None:
            return loss.mean()
        masked_loss = loss * action_mask
        return masked_loss.sum() / (action_mask.sum() + 1e-6)

    def compute_loss(
        self,
        encoded: Mapping[str, Any],
        *,
        batch: Any = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del batch, kwargs
        last_hidden = encoded["last_hidden"]
        actions = encoded.get("actions")
        if not isinstance(last_hidden, torch.Tensor):
            raise TypeError("encoded last_hidden must be a tensor")
        if actions is None:
            raise ValueError("GR00TActionHead requires encoded actions for training")
        compute_dtype = self.dtype
        actions_tensor = torch.as_tensor(
            np.asarray(actions) if not isinstance(actions, torch.Tensor) else actions,
            device=last_hidden.device,
            dtype=compute_dtype,
        )
        actions_target = actions_tensor[:, -self.action_horizon :, : self.action_dim]
        repeats = self._repeated_diffusion_steps()
        actions_target = actions_target.repeat(repeats, 1, 1)
        hidden = last_hidden.to(dtype=compute_dtype).repeat(repeats, 1, 1)
        attention_mask = encoded.get("encoder_attention_mask")
        if attention_mask is not None:
            attention_mask = (
                torch.as_tensor(attention_mask, device=hidden.device).repeat(repeats, 1).bool()
            )
        image_mask = encoded.get("image_mask")
        if image_mask is not None:
            image_mask = torch.as_tensor(image_mask, device=hidden.device).repeat(repeats, 1).bool()
        state = encoded.get("state")
        if state is not None:
            state = torch.as_tensor(state, device=hidden.device, dtype=hidden.dtype)
            state = state.repeat(repeats, 1, 1) if state.dim() == 3 else state.repeat(repeats, 1)
        state = self._state_input(state, hidden.shape[0], hidden.device, hidden.dtype)
        embodiment_id = self._embodiment_id(
            encoded,
            last_hidden.shape[0],
            last_hidden.device,
        ).repeat(repeats)
        action_mask = encoded.get("action_mask")
        if action_mask is not None:
            action_mask = torch.as_tensor(
                action_mask,
                device=hidden.device,
                dtype=hidden.dtype,
            )[:, -self.action_horizon :, : self.action_dim].repeat(repeats, 1, 1)
        return {
            "action_loss": self._flow_matching_loss(
                hidden,
                actions_target,
                state=state,
                embodiment_id=embodiment_id,
                action_mask=action_mask,
                encoder_attention_mask=attention_mask,
                image_mask=image_mask,
            )
        }

    @torch.no_grad()
    def predict_action(
        self,
        encoded: Mapping[str, Any],
        *,
        batch: Any = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        del batch, kwargs
        vl_embeddings = self._process_backbone(encoded["last_hidden"])
        batch_size = vl_embeddings.shape[0]
        device = vl_embeddings.device
        attention_mask = encoded.get("encoder_attention_mask")
        image_mask = encoded.get("image_mask")
        state = encoded.get("state")
        if state is not None:
            state = torch.as_tensor(state, device=device, dtype=vl_embeddings.dtype)
        state = self._state_input(state, batch_size, device, vl_embeddings.dtype)
        embodiment_id = self._embodiment_id(encoded, batch_size, device)
        actions = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embeddings.dtype,
            device=device,
        )
        state_features = self._encode_state(state, embodiment_id)
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            discretized = int(
                (step / float(self.num_inference_timesteps)) * self.num_timestep_buckets
            )
            timesteps = torch.full((batch_size,), discretized, device=device)
            action_features = self._encode_action(actions, timesteps, embodiment_id)
            if self.add_pos_embed:
                positions = torch.arange(action_features.shape[1], device=device)
                action_features = action_features + self.position_embedding(positions).unsqueeze(0)
            state_action_embeddings = (
                torch.cat((state_features, action_features), dim=1)
                if state_features is not None
                else action_features
            )
            output = self._model_forward(
                state_action_embeddings,
                vl_embeddings,
                timesteps,
                encoder_attention_mask=attention_mask,
                image_mask=image_mask,
            )
            velocity = self._decode_action(output, embodiment_id)[:, -self.action_horizon :]
            actions = actions + dt * velocity
        return {"normalized_actions": actions.detach().float().cpu().numpy()}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


__all__ = ["GR00TActionHead"]
