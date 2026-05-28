"""
VQ Image Decoder — reconstruct pixel images from Chameleon token IDs.

Two entry points depending on what you have:

    1. BPE token IDs  (output of Chameleon LLM, integers in the full vocab space)
       → bpe_tokens_to_pixels()

    2. VQ token IDs   (raw VQGAN indices 0..8191)
       → vq_tokens_to_pixels()

Pipeline:
    BPE token IDs [N]
        ↓  bpe2img mapping  (vocabulary_mapping)
    VQ token IDs [N]           (0 .. num_embeddings-1,  default 8192)
        ↓  codebook lookup  (VQModel.quantize.get_codebook_entry)
    Quantized latents [B, embed_dim, h_lat, w_lat]
        ↓  VQModel.decode   (post_quant_conv → CNN Decoder)
    Pixel image [B, 3, H, W]   range [-1, 1]
        ↓  (optional) tensor_to_pil
    PIL.Image

Reference:
    ImageTokenizer.pil_from_img_toks()
        dreamer_vla/models/chameleon_model/chameleon_vae_ori/image_tokenizer.py:117
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image

from dreamer_vla.models.chameleon_model.chameleon_vae_ori.vqgan import VQModel


# Default latent spatial size for 512×512 images with the Chameleon VQ config.
# 512 / (2^4) = 32  (four 2× downsampling stages)
_DEFAULT_LATENT_H = 32
_DEFAULT_LATENT_W = 32


def load_vq_model(
    cfg_path: str | Path,
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> VQModel:
    """
    Load the VQGAN model from a yaml config + checkpoint.

    Default paths for the DreamerVLA repo:
        cfg_path  = data/ckpts/chameleon/tokenizer/vqgan.yaml
        ckpt_path = data/ckpts/chameleon/tokenizer/vqgan.ckpt
    """
    cfg_path = Path(cfg_path)
    ckpt_path = Path(ckpt_path)

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    params = config["model"]["params"]
    params.pop("lossconfig", None)  # not needed for inference
    params["ckpt_path"] = str(ckpt_path)

    model = VQModel(**params)
    model.eval()
    model.to(device)
    return model


# ── BPE → VQ index mapping ────────────────────────────────────────────────────


def build_bpe2vq_tensor(vocabulary_mapping) -> torch.Tensor:
    """
    Build a lookup tensor: bpe_token_id → vq_index.

    vocabulary_mapping is the ChameleonImageVocabularyMapping object attached
    to the loaded Chameleon model:
        model.model.vocabulary_mapping   (ChameleonModel)
        model.model.model.vocabulary_mapping  (ChameleonForConditionalGeneration)

    Returns:
        Tensor of shape [max_bpe_id + 1], dtype=torch.long.
        Entries for non-image BPE tokens are -1 (invalid).
    """
    bpe2img: dict[int, int] = vocabulary_mapping.bpe2img
    max_bpe = max(bpe2img.keys())
    table = torch.full((max_bpe + 1,), fill_value=-1, dtype=torch.long)
    for bpe_id, vq_id in bpe2img.items():
        table[bpe_id] = vq_id
    return table


def bpe_to_vq(
    bpe_ids: torch.Tensor,
    bpe2vq_table: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a flat tensor of BPE image token IDs to VQ indices.

    Args:
        bpe_ids:      [N]  — BPE token IDs (all must be valid image tokens)
        bpe2vq_table: [max_bpe+1]  — lookup table from build_bpe2vq_tensor()

    Returns:
        [N]  VQ indices in range [0, num_embeddings)
    """
    bpe_ids = bpe_ids.to(bpe2vq_table.device)
    vq_ids = bpe2vq_table[bpe_ids]
    if (vq_ids < 0).any():
        raise ValueError("bpe_ids contains non-image BPE tokens (mapped to -1).")
    return vq_ids


# ── Core reconstruction ───────────────────────────────────────────────────────


def vq_tokens_to_pixels(
    vq_ids: torch.Tensor,
    vq_model: VQModel,
    h_latent: int = _DEFAULT_LATENT_H,
    w_latent: int = _DEFAULT_LATENT_W,
) -> torch.Tensor:
    """
    Reconstruct pixel images from raw VQ token IDs (0..8191).

    Args:
        vq_ids:    [B, h_latent * w_latent]  or  [B, h_latent, w_latent]
                   Long tensor of VQ codebook indices.
        vq_model:  loaded VQModel (from load_vq_model).
        h_latent:  latent spatial height (default 32 for 512px images).
        w_latent:  latent spatial width  (default 32 for 512px images).

    Returns:
        [B, 3, H, W]  float tensor, range [-1, 1].
    """
    vq_ids = vq_ids.to(next(vq_model.parameters()).device)

    B = vq_ids.shape[0]
    flat_ids = vq_ids.reshape(-1)  # [B * h_lat * w_lat]
    embed_dim = vq_model.quantize.embedding.weight.shape[-1]

    # codebook lookup → [B, embed_dim, h_lat, w_lat]
    quant = vq_model.quantize.get_codebook_entry(
        flat_ids,
        shape=(B, h_latent, w_latent, embed_dim),
    )

    # CNN decoder → pixels
    with torch.no_grad():
        pixels = vq_model.decode(quant)  # [B, 3, H, W],  range [-1, 1]

    return pixels


def bpe_tokens_to_pixels(
    bpe_ids: torch.Tensor,
    vq_model: VQModel,
    bpe2vq_table: torch.Tensor,
    h_latent: int = _DEFAULT_LATENT_H,
    w_latent: int = _DEFAULT_LATENT_W,
) -> torch.Tensor:
    """
    Reconstruct pixel images from BPE image token IDs.

    Args:
        bpe_ids:       [B, h_latent * w_latent]  — BPE token IDs for image tokens.
        vq_model:      loaded VQModel.
        bpe2vq_table:  lookup table from build_bpe2vq_tensor().
        h_latent:      latent spatial height.
        w_latent:      latent spatial width.

    Returns:
        [B, 3, H, W]  float tensor, range [-1, 1].
    """
    B = bpe_ids.shape[0]
    flat_bpe = bpe_ids.reshape(-1)
    flat_vq = bpe_to_vq(flat_bpe, bpe2vq_table)
    vq_ids = flat_vq.reshape(B, h_latent * w_latent)
    return vq_tokens_to_pixels(vq_ids, vq_model, h_latent, w_latent)


# ── Tensor → PIL helper ───────────────────────────────────────────────────────


def tensor_to_pil(chw: torch.Tensor) -> Image.Image:
    """
    Convert a single [3, H, W] float tensor in [-1, 1] to a PIL RGB image.
    """
    arr = chw.detach().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) / 2.0 * 255.0).byte()  # → [0, 255]
    arr = arr.permute(1, 2, 0).numpy()  # [H, W, 3]
    return Image.fromarray(arr, mode="RGB")


def pixels_to_pil(pixels: torch.Tensor) -> list[Image.Image]:
    """
    Convert a batch [B, 3, H, W] float tensor in [-1, 1] to a list of PIL images.
    """
    return [tensor_to_pil(pixels[i]) for i in range(pixels.shape[0])]


# ── Convenience wrapper ───────────────────────────────────────────────────────


class ChameleonImageReconstructor:
    """
    One-stop helper that bundles the VQModel and the BPE↔VQ mapping.

    Usage:
        rec = ChameleonImageReconstructor.from_paths(
            vqgan_cfg  = "data/ckpts/chameleon/tokenizer/vqgan.yaml",
            vqgan_ckpt = "data/ckpts/chameleon/tokenizer/vqgan.ckpt",
            vocabulary_mapping = model.model.vocabulary_mapping,
        )

        # From BPE token IDs (LLM output):
        pil_images = rec.from_bpe(bpe_ids)   # bpe_ids: [B, 1024]

        # From raw VQ indices:
        pil_images = rec.from_vq(vq_ids)     # vq_ids:  [B, 1024]
    """

    def __init__(
        self,
        vq_model: VQModel,
        bpe2vq_table: torch.Tensor | None = None,
        h_latent: int = _DEFAULT_LATENT_H,
        w_latent: int = _DEFAULT_LATENT_W,
    ) -> None:
        self.vq_model = vq_model
        self.bpe2vq_table = bpe2vq_table
        self.h_latent = h_latent
        self.w_latent = w_latent

    @classmethod
    def from_paths(
        cls,
        vqgan_cfg: str | Path,
        vqgan_ckpt: str | Path,
        vocabulary_mapping=None,
        device: str | torch.device = "cpu",
        h_latent: int = _DEFAULT_LATENT_H,
        w_latent: int = _DEFAULT_LATENT_W,
    ) -> "ChameleonImageReconstructor":
        vq_model = load_vq_model(vqgan_cfg, vqgan_ckpt, device=device)
        bpe2vq_table = (
            build_bpe2vq_tensor(vocabulary_mapping).to(device)
            if vocabulary_mapping is not None
            else None
        )
        return cls(vq_model, bpe2vq_table, h_latent, w_latent)

    def from_vq(self, vq_ids: torch.Tensor) -> list[Image.Image]:
        """vq_ids: [B, h_lat*w_lat] or [B, h_lat, w_lat]"""
        pixels = vq_tokens_to_pixels(
            vq_ids, self.vq_model, self.h_latent, self.w_latent
        )
        return pixels_to_pil(pixels)

    def from_bpe(self, bpe_ids: torch.Tensor) -> list[Image.Image]:
        """bpe_ids: [B, h_lat*w_lat]  — must all be image BPE tokens"""
        if self.bpe2vq_table is None:
            raise RuntimeError(
                "vocabulary_mapping was not provided at construction time. "
                "Use from_vq() for raw VQ indices, or pass vocabulary_mapping "
                "to ChameleonImageReconstructor.from_paths()."
            )
        pixels = bpe_tokens_to_pixels(
            bpe_ids, self.vq_model, self.bpe2vq_table, self.h_latent, self.w_latent
        )
        return pixels_to_pil(pixels)


__all__ = [
    "load_vq_model",
    "build_bpe2vq_tensor",
    "bpe_to_vq",
    "vq_tokens_to_pixels",
    "bpe_tokens_to_pixels",
    "tensor_to_pil",
    "pixels_to_pil",
    "ChameleonImageReconstructor",
]
