from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.actor.base_actor import BaseActor
from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    L1RegressionActionHead,
    PerTokenRegressionActionHead,
)


class VLAActionHeadActor(BaseActor):
    """Reuses VLA ActionHead modules with a world-model feature adapter."""

    def __init__(
        self,
        hidden_dim: int = 768,
        action_dim: int = 7,
        time_horizon: int = 10,
        vla_hidden_size: int = 4096,
        hidden_size_factor: float = 0.25,
        num_encoder_layers: int = 2,
        adapter_hidden_dim: int = 1024,
        adapter_type: str = "mlp",
        initial_log_std: float = -0.5,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        init_action_head_ckpt: str | None = None,
        action_head_type: str = "legacy",
        **_: Any,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.time_horizon = int(time_horizon)
        self.hidden_size = int(vla_hidden_size)
        self.reduced_hidden_size = int(self.hidden_size * hidden_size_factor)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.adapter_type = str(adapter_type).lower()
        self.action_head_type = str(action_head_type)
        if self.adapter_type not in {"mlp", "identity"}:
            raise ValueError("adapter_type must be one of {'mlp', 'identity'}")
        if self.action_head_type not in {"legacy", "pi0_query"}:
            raise ValueError("action_head_type must be one of {'legacy', 'pi0_query'}")
        self.action_token_count = (
            self.time_horizon if self.action_head_type == "pi0_query" else self.time_horizon * self.action_dim
        )

        if self.adapter_type == "identity":
            if int(hidden_dim) != self.hidden_size:
                raise ValueError(
                    "adapter_type='identity' requires hidden_dim == vla_hidden_size "
                    f"({hidden_dim} != {self.hidden_size})"
                )
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.LayerNorm(int(hidden_dim)),
                nn.Linear(int(hidden_dim), int(adapter_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(adapter_hidden_dim), self.hidden_size),
            )

        self.action_token_embeddings = nn.Embedding(
            1,
            self.action_token_count * self.hidden_size,
        )
        nn.init.normal_(self.action_token_embeddings.weight, std=0.02)

        self.hidden_projection = nn.Linear(self.hidden_size, self.reduced_hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.reduced_hidden_size,
            nhead=4,
            dim_feedforward=self.reduced_hidden_size * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_encoder_layers),
            norm=nn.LayerNorm(self.reduced_hidden_size),
        )
        if self.action_head_type == "pi0_query":
            self.output_projection = PerTokenRegressionActionHead(
                self.reduced_hidden_size,
                self.reduced_hidden_size,
                self.action_dim,
            )
        else:
            self.output_projection = L1RegressionActionHead(
                self.reduced_hidden_size,
                self.reduced_hidden_size,
                self.time_horizon,
                self.action_dim,
            )

        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(initial_log_std)))

        if init_action_head_ckpt:
            self._load_action_head_from_vla_ckpt(str(init_action_head_ckpt))

    def _load_action_head_from_vla_ckpt(self, ckpt_path: str) -> None:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder_sd = payload.get("state_dicts", {}).get("encoder")
        if encoder_sd is None:
            raise RuntimeError(f"VLA action head checkpoint has no state_dicts.encoder: {ckpt_path}")
        prefix = "backbone.action_head."
        action_head_sd = {
            k[len(prefix):]: v
            for k, v in encoder_sd.items()
            if k.startswith(prefix)
        }
        if not action_head_sd:
            raise RuntimeError(
                f"VLA action head checkpoint has no '{prefix}' tensors: {ckpt_path}"
            )
        emb = action_head_sd.get("action_token_embeddings.weight")
        if emb is not None:
            expected = self.action_token_count * self.hidden_size
            got = int(emb.shape[-1])
            if got != expected:
                if self.action_head_type == "pi0_query":
                    ckpt_horizon = got // max(self.hidden_size, 1)
                else:
                    ckpt_horizon = got // max(self.action_dim * self.hidden_size, 1)
                raise ValueError(
                    "VLA action head checkpoint is not compatible with this actor: "
                    f"configured action_head_type={self.action_head_type}, "
                    f"time_horizon={self.time_horizon}, action_dim={self.action_dim}, "
                    f"hidden_size={self.hidden_size}, but checkpoint embedding width={got} "
                    f"(implied time_horizon={ckpt_horizon})."
                )
        missing, unexpected = self.load_state_dict(action_head_sd, strict=False)
        non_adapter_missing = [k for k in missing if not k.startswith("adapter.") and k != "log_std"]
        print(
            f"[VLAActor] loaded {len(action_head_sd)} action_head tensors from VLA ckpt; "
            f"unexpected={len(unexpected)}, missing-action-head-only={len(non_adapter_missing)}"
        )
        if non_adapter_missing:
            print(f"[VLAActor] WARN missing action_head tensors (first 5): {non_adapter_missing[:5]}")
        if unexpected:
            print(f"[VLAActor] WARN unexpected (first 5): {unexpected[:5]}")
        del payload

    def _action_chunk_from_single_context(self, wm_feat: torch.Tensor) -> torch.Tensor:
        """WM feat [B, hidden_dim] -> action chunk [B, T, A]."""
        param_dtype = self.hidden_projection.weight.dtype
        wm_feat = wm_feat.to(dtype=param_dtype)
        bs = wm_feat.shape[0]

        context = self.adapter(wm_feat).unsqueeze(1)
        context_red = self.hidden_projection(context)

        action_tokens = self.action_token_embeddings.weight.view(
            1,
            self.action_token_count,
            self.hidden_size,
        ).expand(bs, -1, -1)
        action_tokens_red = self.hidden_projection(action_tokens)

        combined = torch.cat([context_red, action_tokens_red], dim=1)
        out = self.transformer_encoder(combined)

        action_part = out[:, 1:, :]
        actions = self.output_projection(action_part)
        if self.action_head_type == "pi0_query":
            return actions.reshape(bs, self.time_horizon, self.action_dim)
        return actions

    def _action_chunk_from_sequence(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        target_token_id: int = 10004,
    ) -> torch.Tensor:
        """Native ActionHead path using full VLA token hidden states."""
        param_dtype = self.hidden_projection.weight.dtype
        hidden_states = hidden_states.to(dtype=param_dtype)
        input_ids = input_ids.to(device=hidden_states.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=hidden_states.device)

        batch_size = hidden_states.shape[0]
        action_tokens = self.action_token_embeddings.weight.view(
            1,
            self.action_token_count,
            self.hidden_size,
        ).expand(batch_size, -1, -1).to(dtype=param_dtype)

        extracted_hidden_states: list[torch.Tensor] = []
        extracted_attention_masks: list[torch.Tensor] = []
        for idx in range(batch_size):
            target_positions = (input_ids[idx] == int(target_token_id)).nonzero(as_tuple=True)[0]
            if len(target_positions) == 0:
                end_pos = int(hidden_states.shape[1])
            else:
                end_pos = int(target_positions[0].item())
            end_pos = max(1, min(end_pos, int(hidden_states.shape[1])))
            extracted_hidden_states.append(hidden_states[idx, :end_pos, :])
            if attention_mask is not None:
                extracted_attention_masks.append(attention_mask[idx, :end_pos])

        combined_states: list[torch.Tensor] = []
        combined_masks: list[torch.Tensor] = []
        max_length = 0
        for idx, context in enumerate(extracted_hidden_states):
            combined = torch.cat([context, action_tokens[idx]], dim=0)
            combined_states.append(combined)
            max_length = max(max_length, int(combined.shape[0]))
            if attention_mask is not None:
                mask = torch.cat(
                    [
                        extracted_attention_masks[idx],
                        torch.ones(
                            self.action_token_count,
                            device=hidden_states.device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=0,
                )
                combined_masks.append(mask)

        padded_states: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        for idx, combined in enumerate(combined_states):
            pad = max_length - int(combined.shape[0])
            if pad > 0:
                combined = torch.cat(
                    [
                        combined,
                        torch.zeros(pad, self.hidden_size, device=hidden_states.device, dtype=param_dtype),
                    ],
                    dim=0,
                )
            padded_states.append(combined)
            if attention_mask is not None:
                mask = combined_masks[idx]
                if pad > 0:
                    mask = torch.cat(
                        [
                            mask,
                            torch.zeros(pad, device=hidden_states.device, dtype=attention_mask.dtype),
                        ],
                        dim=0,
                    )
                padded_masks.append(mask)

        processed_hidden = torch.stack(padded_states, dim=0)
        if attention_mask is None:
            processed_mask = torch.ones(
                batch_size,
                processed_hidden.shape[1],
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            processed_mask = torch.stack(padded_masks, dim=0).bool()

        projected = self.hidden_projection(processed_hidden)
        out = self.transformer_encoder(projected, src_key_padding_mask=(~processed_mask))

        action_outputs: list[torch.Tensor] = []
        for idx, context in enumerate(extracted_hidden_states):
            start = int(context.shape[0])
            end = start + self.action_token_count
            action_outputs.append(out[idx, start:end, :])
        action_part = torch.stack(action_outputs, dim=0)
        actions = self.output_projection(action_part)
        if self.action_head_type == "pi0_query":
            return actions.reshape(batch_size, self.time_horizon, self.action_dim)
        return actions

    def _action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        hidden_states = batch.get("hidden_states")
        input_ids = batch.get("input_ids")
        if hidden_states is not None:
            if input_ids is None:
                raise ValueError("VLAActionHeadActor sequence mode requires `input_ids`.")
            return self._action_chunk_from_sequence(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=batch.get("attention_mask"),
                target_token_id=int(batch.get("target_token_id", 10004)),
            )
        return self._action_chunk_from_single_context(batch["hidden"])

    def _action_mean(self, batch: dict[str, Any]) -> torch.Tensor:
        return self._action_chunk(batch)[:, 0, :].float()

    def forward(self, batch: dict[str, Any]) -> Any:
        mode = batch.get("mode")
        action_chunk = self._action_chunk(batch)
        dist, mean, std = self._normal_from_action_chunk(action_chunk)
        chunk_dist, mean_chunk, std_chunk = self._normal_from_full_action_chunk(action_chunk)
        if mode == "sample":
            deterministic = bool(batch.get("deterministic", False))
            if bool(batch.get("return_chunk", False)):
                action = mean_chunk if deterministic else chunk_dist.rsample()
                log_prob = chunk_dist.log_prob(action).sum(dim=(-1, -2))
                return action, log_prob, {
                    "mean": mean,
                    "std": std,
                    "mean_chunk": mean_chunk,
                    "std_chunk": std_chunk,
                    "action_chunk": mean_chunk,
                }
            action = mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, {"mean": mean, "std": std, "action_chunk": mean_chunk}
        if mode == "evaluate":
            action = batch["action"]
            if action.ndim == 3:
                log_prob = chunk_dist.log_prob(action).sum(dim=(-1, -2))
                entropy = chunk_dist.entropy().sum(dim=(-1, -2))
                return log_prob, entropy, {"mean": mean, "std": std, "mean_chunk": mean_chunk, "std_chunk": std_chunk}
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy, {"mean": mean, "std": std, "mean_chunk": mean_chunk, "std_chunk": std_chunk}
        raise ValueError(f"Unknown VLAActionHeadActor forward mode: {mode!r}")


__all__ = ["VLAActionHeadActor"]
