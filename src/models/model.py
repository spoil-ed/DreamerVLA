from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from .vla_encoder.encoder import MultimodalEncoder, MultimodalEncoderOutput
from .vla_encoder.vla_encoder import RynnVLAEncoder


class DreamerVLA(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        image_channels: int = 3,
        vocab_size: int = 32000,
        max_language_length: int = 128,
        proprio_dim: int = 16,
        embed_dim: int = 256,
        fused_dim: int = 256,
        image_depth: int = 4,
        language_depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        proprio_hidden_dim: int = 256,
        dropout: float = 0.0,
        pad_token_id: int = 0,
        encoder_type: str = "multimodal",
        pretrained_policy_ckpt: str | None = None,
        precision: str = "bf16",
        device: str = "auto",
        encoder_configs: dict | None = None,
        condition_frame_num: int = 1,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type

        if self.encoder_type == "rynn_vla":
            if pretrained_policy_ckpt is None:
                raise ValueError("`pretrained_policy_ckpt` is required when `encoder_type='rynn_vla'`.")
            if encoder_configs is None:
                raise ValueError("`encoder_configs` is required when `encoder_type='rynn_vla'`.")

            self.encoder = RynnVLAEncoder(
                pretrained_policy_ckpt=pretrained_policy_ckpt,
                precision=precision,
                device=device,
                configs=encoder_configs,
                condition_frame_num=condition_frame_num,
            )
            encoder_embed_dim = int(self.encoder.model.config.hidden_size)
            self.fusion = nn.Sequential(
                nn.Linear(encoder_embed_dim, fused_dim),
                nn.GELU(),
                nn.LayerNorm(fused_dim),
            )
        else:
            self.encoder = MultimodalEncoder(
                image_size=image_size,
                patch_size=patch_size,
                image_channels=image_channels,
                vocab_size=vocab_size,
                max_language_length=max_language_length,
                proprio_dim=proprio_dim,
                embed_dim=embed_dim,
                image_depth=image_depth,
                language_depth=language_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                proprio_hidden_dim=proprio_hidden_dim,
                dropout=dropout,
                pad_token_id=pad_token_id,
            )
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, fused_dim),
                nn.GELU(),
                nn.LayerNorm(fused_dim),
            )

    def forward(
        self,
        image: Tensor | None = None,
        language: Tensor | None = None,
        proprio: Tensor | None = None,
        language_attention_mask: Tensor | None = None,
        obs: dict | None = None,
        text: str | None = None,
    ) -> dict[str, Tensor | MultimodalEncoderOutput]:
        if self.encoder_type == "rynn_vla":
            if obs is None or text is None:
                raise ValueError("`obs` and `text` are required when `encoder_type='rynn_vla'`.")

            obs_batch = [obs] if isinstance(obs, dict) else list(obs)
            text_batch = [text] if isinstance(text, str) else list(text)
            if len(obs_batch) != len(text_batch):
                raise ValueError("`obs` and `text` must have the same batch size.")

            token_sequences = []
            pooled_embeddings = []
            for sample_obs, sample_text in zip(obs_batch, text_batch):
                sample_tokens = self.encoder.encode(sample_obs, sample_text).squeeze(0)
                token_sequences.append(sample_tokens)
                pooled_embeddings.append(sample_tokens.mean(dim=0))

            multimodal_tokens = pad_sequence(token_sequences, batch_first=True)
            pooled_embedding = torch.stack(pooled_embeddings, dim=0)
            encoder_output = multimodal_tokens
        else:
            if image is None or language is None or proprio is None:
                raise ValueError(
                    "`image`, `language`, and `proprio` are required when `encoder_type='multimodal'`."
                )

            encoder_output = self.encoder(
                image=image,
                language=language,
                proprio=proprio,
                language_attention_mask=language_attention_mask,
            )

            pooled_embedding = torch.cat(
                [
                    encoder_output.image_embedding,
                    encoder_output.language_embedding,
                    encoder_output.proprio_embedding,
                ],
                dim=-1,
            )
            multimodal_tokens = torch.cat(
                [
                    encoder_output.image_tokens,
                    encoder_output.language_tokens,
                    encoder_output.proprio_tokens,
                ],
                dim=1,
            )

        return {
            "encoder_output": encoder_output,
            "pooled_embedding": pooled_embedding,
            "multimodal_tokens": multimodal_tokens,
            "latent": self.fusion(pooled_embedding),
        }
