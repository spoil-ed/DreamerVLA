"""π0.5 prefix-input latent geometry and attention metadata."""

from __future__ import annotations

from dataclasses import dataclass

import torch

PI05_IMAGE_TOKEN_COUNT = 768
PI05_IMAGE_GROUP_COUNT = 3
PI05_IMAGE_PATCH_GRID = (16, 16)
PI05_TEXT_TOKEN_COUNT = 200
PI05_PREFIX_INPUT_TOKEN_COUNT = PI05_IMAGE_TOKEN_COUNT + PI05_TEXT_TOKEN_COUNT
PI05_PREFIX_INPUT_TOKEN_DIM = 2048
PI05_PREFIX_INPUT_SOURCE = "prefix_input_latent"
PI05_TEXT_MODES = frozenset({"exact", "masked"})


@dataclass(frozen=True)
class Pi05PrefixInputLatent:
    """Pre-PaliGemma prefix embeddings and their exact validity metadata."""

    latent: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    text_mode: str

    def __post_init__(self) -> None:
        validate_prefix_input_latent(
            self.latent,
            self.attention_mask,
            self.position_ids,
            text_mode=self.text_mode,
        )


@dataclass(frozen=True)
class Pi05ImagePrefixLatent:
    """Post-PaliGemma image-prefix outputs with exact image-slot validity."""

    latent: torch.Tensor
    attention_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.latent.ndim != 3 or self.latent.shape[1:] != (
            PI05_IMAGE_TOKEN_COUNT,
            PI05_PREFIX_INPUT_TOKEN_DIM,
        ):
            raise ValueError(
                f"π0.5 image prefix latent must be [B,768,2048], got {tuple(self.latent.shape)}"
            )
        expected = (int(self.latent.shape[0]), PI05_IMAGE_TOKEN_COUNT)
        if self.attention_mask.shape != expected:
            raise ValueError(
                f"π0.5 image prefix attention_mask must be {expected}, "
                f"got {tuple(self.attention_mask.shape)}"
            )


def validate_text_mode(text_mode: str) -> str:
    """Return one canonical π0.5 text mode or raise a contract error."""

    mode = str(text_mode).strip().lower()
    if mode not in PI05_TEXT_MODES:
        raise ValueError(f"text_mode must be one of {sorted(PI05_TEXT_MODES)}, got {text_mode!r}")
    return mode


def prefix_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build the position ids used by the native OpenPI prefix prefill."""

    if attention_mask.ndim != 2 or attention_mask.shape[-1] != PI05_PREFIX_INPUT_TOKEN_COUNT:
        raise ValueError(
            f"π0.5 prefix attention_mask must be [B,968], got {tuple(attention_mask.shape)}"
        )
    mask = attention_mask.to(dtype=torch.bool)
    return torch.cumsum(mask.to(dtype=torch.long), dim=1) - 1


def prefix_attention_matrix(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build OpenPI's full-attention prefix matrix from the validity mask."""

    if attention_mask.ndim != 2 or attention_mask.shape[-1] != PI05_PREFIX_INPUT_TOKEN_COUNT:
        raise ValueError(
            f"π0.5 prefix attention_mask must be [B,968], got {tuple(attention_mask.shape)}"
        )
    mask = attention_mask.to(dtype=torch.bool)
    return mask[:, None, :] & mask[:, :, None]


def build_prefix_input_latent(
    image_embeddings: torch.Tensor,
    language_embeddings: torch.Tensor,
    image_attention_mask: torch.Tensor,
    language_attention_mask: torch.Tensor,
    *,
    text_mode: str = "exact",
) -> Pi05PrefixInputLatent:
    """Concatenate native image/language embeddings without running PaliGemma."""

    mode = validate_text_mode(text_mode)
    if image_embeddings.ndim != 3 or image_embeddings.shape[1:] != (
        PI05_IMAGE_TOKEN_COUNT,
        PI05_PREFIX_INPUT_TOKEN_DIM,
    ):
        raise ValueError(
            f"π0.5 image prefix input must be [B,768,2048], got {tuple(image_embeddings.shape)}"
        )
    if language_embeddings.ndim != 3 or language_embeddings.shape[1:] != (
        PI05_TEXT_TOKEN_COUNT,
        PI05_PREFIX_INPUT_TOKEN_DIM,
    ):
        raise ValueError(
            "π0.5 language prefix input must be [B,200,2048], got "
            f"{tuple(language_embeddings.shape)}"
        )
    batch_size = int(image_embeddings.shape[0])
    if language_embeddings.shape[0] != batch_size:
        raise ValueError("image and language prefix inputs must have the same batch size")
    if image_attention_mask.shape != (batch_size, PI05_IMAGE_TOKEN_COUNT):
        raise ValueError(
            f"π0.5 image attention mask must be [B,768], got {tuple(image_attention_mask.shape)}"
        )
    if language_attention_mask.shape != (batch_size, PI05_TEXT_TOKEN_COUNT):
        raise ValueError(
            "π0.5 language attention mask must be [B,200], got "
            f"{tuple(language_attention_mask.shape)}"
        )

    image_mask = image_attention_mask.to(device=image_embeddings.device, dtype=torch.bool)
    language_mask = language_attention_mask.to(
        device=image_embeddings.device,
        dtype=torch.bool,
    )
    language = language_embeddings.to(
        device=image_embeddings.device,
        dtype=image_embeddings.dtype,
    )
    if mode == "masked":
        language = torch.zeros_like(language)
        language_mask = torch.zeros_like(language_mask)
    latent = torch.cat([image_embeddings, language], dim=1)
    attention_mask = torch.cat([image_mask, language_mask], dim=1)
    return Pi05PrefixInputLatent(
        latent=latent,
        attention_mask=attention_mask,
        position_ids=prefix_position_ids(attention_mask),
        text_mode=mode,
    )


def validate_prefix_input_latent(
    latent: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    text_mode: str = "exact",
) -> None:
    """Validate the fixed `[B,968,2048]` native-prefix contract."""

    mode = validate_text_mode(text_mode)
    if latent.ndim != 3 or latent.shape[1:] != (
        PI05_PREFIX_INPUT_TOKEN_COUNT,
        PI05_PREFIX_INPUT_TOKEN_DIM,
    ):
        raise ValueError(
            f"π0.5 prefix_input_latent must be [B,968,2048], got {tuple(latent.shape)}"
        )
    expected_mask_shape = (int(latent.shape[0]), PI05_PREFIX_INPUT_TOKEN_COUNT)
    if attention_mask.shape != expected_mask_shape:
        raise ValueError(
            f"π0.5 prefix attention mask must be {expected_mask_shape}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if position_ids is not None and position_ids.shape != expected_mask_shape:
        raise ValueError(
            f"π0.5 prefix position ids must be {expected_mask_shape}, "
            f"got {tuple(position_ids.shape)}"
        )
    if mode == "masked":
        text = latent[:, PI05_IMAGE_TOKEN_COUNT:]
        text_mask = attention_mask[:, PI05_IMAGE_TOKEN_COUNT:]
        if torch.count_nonzero(text).item() != 0:
            raise ValueError("masked π0.5 text embeddings must be exactly zero")
        if torch.count_nonzero(text_mask).item() != 0:
            raise ValueError("masked π0.5 text attention mask must be false")


__all__ = [
    "PI05_IMAGE_GROUP_COUNT",
    "PI05_IMAGE_PATCH_GRID",
    "PI05_IMAGE_TOKEN_COUNT",
    "PI05_PREFIX_INPUT_SOURCE",
    "PI05_PREFIX_INPUT_TOKEN_COUNT",
    "PI05_PREFIX_INPUT_TOKEN_DIM",
    "PI05_TEXT_MODES",
    "PI05_TEXT_TOKEN_COUNT",
    "Pi05ImagePrefixLatent",
    "Pi05PrefixInputLatent",
    "build_prefix_input_latent",
    "prefix_attention_matrix",
    "prefix_position_ids",
    "validate_prefix_input_latent",
    "validate_text_mode",
]
