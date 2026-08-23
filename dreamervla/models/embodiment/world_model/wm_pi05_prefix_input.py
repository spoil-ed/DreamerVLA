"""World model over native π0.5 prefix-input embeddings."""

from __future__ import annotations

from typing import Any

import torch

from dreamervla.models.embodiment.pi05.prefix_input import (
    PI05_IMAGE_TOKEN_COUNT,
    PI05_PREFIX_INPUT_TOKEN_COUNT,
    PI05_PREFIX_INPUT_TOKEN_DIM,
    PI05_TEXT_TOKEN_COUNT,
    Pi05PrefixInputLatent,
    build_prefix_input_latent,
    validate_text_mode,
)
from dreamervla.models.embodiment.world_model.wm_chunk import ChunkAwareWorldModel


class Pi05PrefixInputWorldModel(ChunkAwareWorldModel):
    """Predict only native π0.5 visual embeddings with static language slots."""

    def __init__(
        self,
        *args: Any,
        image_token_count: int = PI05_IMAGE_TOKEN_COUNT,
        text_token_count: int = PI05_TEXT_TOKEN_COUNT,
        text_mode: str = "exact",
        wm_use_text: bool = True,
        policy_use_text: bool = True,
        predict_text_tokens: bool = False,
        **kwargs: Any,
    ) -> None:
        token_count = int(kwargs.get("token_count", PI05_PREFIX_INPUT_TOKEN_COUNT))
        token_dim = int(kwargs.get("token_dim", PI05_PREFIX_INPUT_TOKEN_DIM))
        obs_dim = int(kwargs.get("obs_dim", token_count * token_dim))
        if int(image_token_count) != PI05_IMAGE_TOKEN_COUNT:
            raise ValueError("π0.5 prefix-input image_token_count must be 768")
        if int(text_token_count) != PI05_TEXT_TOKEN_COUNT:
            raise ValueError("π0.5 prefix-input text_token_count must be 200")
        if token_count != int(image_token_count) + int(text_token_count):
            raise ValueError(
                "π0.5 prefix-input token_count must equal image_token_count + "
                f"text_token_count ({token_count} != {image_token_count} + {text_token_count})"
            )
        if token_count != PI05_PREFIX_INPUT_TOKEN_COUNT or token_dim != PI05_PREFIX_INPUT_TOKEN_DIM:
            raise ValueError(
                "π0.5 prefix_input_latent geometry must be [968,2048], got "
                f"[{token_count},{token_dim}]"
            )
        if obs_dim != token_count * token_dim:
            raise ValueError("π0.5 prefix-input obs_dim must equal token_count * token_dim")
        if bool(predict_text_tokens):
            raise ValueError("π0.5 prefix-input WM must set predict_text_tokens=false")
        token_normalization = str(kwargs.get("token_normalization", "none")).lower()
        if token_normalization != "none":
            raise ValueError(
                "π0.5 prefix-input WM requires token_normalization=none so predictions remain "
                "native PaliGemma input embeddings"
            )

        self.image_token_count = int(image_token_count)
        self.text_token_count = int(text_token_count)
        self.text_mode = validate_text_mode(text_mode)
        self.wm_use_text = bool(wm_use_text)
        self.policy_use_text = bool(policy_use_text)
        self.predict_text_tokens = False
        super().__init__(*args, **kwargs)

    def _canonical_prefix_mask(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, steps = int(tokens.shape[0]), int(tokens.shape[1])
        if attention_mask is None:
            mask = torch.ones(
                batch_size,
                steps,
                self.token_count,
                device=tokens.device,
                dtype=torch.bool,
            )
        else:
            mask = attention_mask.to(device=tokens.device, dtype=torch.bool)
            if mask.ndim == 2:
                mask = mask[:, None].expand(-1, steps, -1)
            if mask.shape != (batch_size, steps, self.token_count):
                raise ValueError(
                    "prefix_attention_mask must be [B,968] or [B,T,968], got "
                    f"{tuple(attention_mask.shape)}"
                )
        if self.text_mode == "masked":
            mask = mask.clone()
            mask[..., self.image_token_count :] = False
        return mask

    def _external_prefix_tokens(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        tokens = self.obs_to_tokens(obs_embedding)
        if tokens.shape[-2:] != (self.token_count, self.token_dim):
            raise ValueError(
                "π0.5 prefix_input_latent must have trailing shape [968,2048], got "
                f"{tuple(tokens.shape)}"
            )
        return tokens

    def _prepare_wm_observation(
        self,
        obs_embedding: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self._external_prefix_tokens(obs_embedding)
        mask = self._canonical_prefix_mask(raw, attention_mask)
        wm_tokens = raw * mask.unsqueeze(-1).to(dtype=raw.dtype)
        if not self.wm_use_text:
            wm_tokens = wm_tokens.clone()
            wm_tokens[..., self.image_token_count :, :] = 0
        return wm_tokens, raw, mask

    def encode_latent(
        self,
        hidden: torch.Tensor,
        prefix_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Create WM state while retaining exact language for native policy prefill."""

        wm_hidden, raw, mask = self._prepare_wm_observation(hidden, prefix_attention_mask)
        state = super().encode_latent(wm_hidden)
        state["lang"] = raw[:, -1, self.image_token_count :]
        state["prefix_attention_mask"] = mask[:, -1]
        return state

    def observe_sequence(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Expose sequence latents with exact static language and prefix masks."""

        wm_obs, raw, mask = self._prepare_wm_observation(
            batch["obs_embedding"],
            batch.get("prefix_attention_mask"),
        )
        prepared = dict(batch)
        prepared["obs_embedding"] = wm_obs
        prepared["lang_emb"] = raw[..., self.image_token_count :, :]
        prepared["prefix_attention_mask"] = mask
        return super().observe_sequence(prepared)

    def chunk_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Mask text/padding before WM input and exclude text from every dynamic loss."""

        wm_obs, raw, mask = self._prepare_wm_observation(
            batch["obs_embedding"],
            batch.get("prefix_attention_mask"),
        )
        prepared = dict(batch)
        prepared["obs_embedding"] = wm_obs
        prepared["lang_emb"] = raw[..., self.image_token_count :, :]
        prepared["prefix_attention_mask"] = mask
        metrics = super().chunk_loss(prepared)
        metrics["predicted_visual_tokens"] = metrics["loss"].new_tensor(
            float(self.image_token_count)
        )
        metrics["predicted_text_tokens"] = metrics["loss"].new_zeros(())
        return metrics

    def _hidden_loss_terms(
        self,
        hidden_pred: torch.Tensor,
        hidden_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_pred = hidden_pred[..., : self.image_token_count, : self.token_dim]
        visual_target = hidden_target[..., : self.image_token_count, : self.token_dim]
        return super()._hidden_loss_terms(visual_pred, visual_target)

    def predict_next(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Advance visual state and copy static WM-language slots unchanged."""

        source = self._latent_hidden(latent)
        output = super().predict_next(latent, actions)
        static_text = source[..., self.image_token_count :, : self.token_dim]
        if not self.wm_use_text or self.text_mode == "masked":
            static_text = torch.zeros_like(static_text)
        next_text = output["hidden"][..., self.image_token_count :, :].clone()
        next_text[..., : self.token_dim] = static_text
        next_hidden = torch.cat(
            [output["hidden"][..., : self.image_token_count, :], next_text],
            dim=-2,
        )
        next_history = output["history"].clone()
        next_history[:, -1] = next_hidden
        output["hidden"] = next_hidden
        output["history"] = next_history
        return output

    def policy_prefix_input(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        *,
        exact_language_embeddings: torch.Tensor | None = None,
        prefix_attention_mask: torch.Tensor | None = None,
    ) -> Pi05PrefixInputLatent:
        """Recompose imagined visual tokens for native PaliGemma prefill."""

        hidden = self._latent_hidden(latent)
        image_embeddings = hidden[..., : self.image_token_count, : self.token_dim]
        language = exact_language_embeddings
        if language is None and isinstance(latent, dict):
            candidate = latent.get("lang")
            if isinstance(candidate, torch.Tensor):
                language = candidate[:, -1] if candidate.ndim == 4 else candidate
        if language is None:
            language = hidden[..., self.image_token_count :, : self.token_dim]
        if language.shape != (
            image_embeddings.shape[0],
            self.text_token_count,
            self.token_dim,
        ):
            raise ValueError(
                f"exact language embeddings must be [B,200,2048], got {tuple(language.shape)}"
            )

        mask = prefix_attention_mask
        if mask is None and isinstance(latent, dict):
            candidate_mask = latent.get("prefix_attention_mask")
            if isinstance(candidate_mask, torch.Tensor):
                mask = candidate_mask[:, -1] if candidate_mask.ndim == 3 else candidate_mask
        if mask is None:
            if self.policy_use_text:
                raise ValueError("policy_use_text=true requires the exact prefix_attention_mask")
            mask = torch.ones(
                image_embeddings.shape[0],
                self.token_count,
                device=image_embeddings.device,
                dtype=torch.bool,
            )
        mode = "exact" if self.policy_use_text else "masked"
        return build_prefix_input_latent(
            image_embeddings,
            language,
            mask[..., : self.image_token_count],
            mask[..., self.image_token_count :],
            text_mode=mode,
        )

    def actor_input(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Return `[B,968,2048]` ready for the native π0.5 prefix prefill."""

        return self.policy_prefix_input(latent).latent

    def critic_input(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Pool only dynamic visual slots for latent critics."""

        hidden = self._latent_hidden(latent)
        return hidden[..., : self.image_token_count, : self.token_dim].mean(dim=-2)


__all__ = ["Pi05PrefixInputWorldModel"]
