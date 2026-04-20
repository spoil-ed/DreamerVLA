"""
Causal Transformer dynamics backbone for TransDreamer-style world model.

Two backbone options with the same [B, T, d_model] -> [B, T, d_model] interface:

    CausalTransformerCell  — lightweight, random-init PyTorch encoder
    LLMBackboneCell        — pretrained Chameleon decoder LLM (uses existing ckpt)

Equivalent to TransDreamer's `self.cell = Transformer(cfg)` in
modules_transformer.py:259.

Input:  [B, T, d_model]   — sequence of (z_t ⊕ a_t) token embeddings
Output: [B, T, d_model]   — h_t at position t encodes history z_{0:t}, a_{0:t}
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CausalTransformerCell(nn.Module):
    """
    Lightweight causal Transformer used as the sequence dynamics backbone.

    Architecture:
        - Standard PyTorch TransformerEncoder with pre-LayerNorm (norm_first=True)
        - Upper-triangular causal mask: position t attends only to 0..t
        - batch_first=True so shapes are [B, T, d_model] throughout

    Comparison with TransDreamer (transformer.py):
        TransDreamer uses a custom MultiheadAttention + GRU gating + positional
        embeddings. This implementation uses PyTorch's built-in encoder for
        simplicity and better hardware utilisation via SDPA.

    Args:
        d_model:  token / hidden dimension
        n_heads:  number of attention heads  (d_model must be divisible by n_heads)
        n_layers: number of Transformer layers
        d_ff:     feedforward inner dimension (typically 4 * d_model)
        dropout:  dropout probability applied in attention and FFN
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # [B, T, d_model]
            norm_first=True,   # pre-LayerNorm: more stable, matches TransDreamer's pre_lnorm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """
        Build an upper-triangular boolean causal mask.

        True  = position is masked (not attended to).
        False = position is visible.

        TransDreamer equivalent:
            transformer.py _generate_square_subsequent_mask (line 215)

        Shape: [T, T]
        """
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]  — input token sequence

        Returns:
            [B, T, d_model]  — contextualised output where position t has
                               attended to all positions 0..t (causal).
        """
        T = x.size(1)
        mask = self._causal_mask(T, x.device)
        return self.encoder(x, mask=mask, is_causal=True)


class LLMBackboneCell(nn.Module):
    """
    Drop-in replacement for CausalTransformerCell that uses a pretrained
    Chameleon decoder LLM as the causal Transformer.

    The external interface matches CausalTransformerCell exactly:
        Input:  [B, T, d_model]
        Output: [B, T, d_model]

    Internally it projects d_model <-> llm_hidden around the LLM call, so
    callers (prior_head, transition_head, reward_head) keep operating on the
    compact d_model dimension.

    Args:
        pretrained_model_path: path to a Chameleon ckpt directory (same format
            loaded by TSSMWorldModel._load_transition_backbone_like_rynnvla002).
        d_model: external token dimension expected by the TransDreamer-style
            world model (e.g. 512).
        action_dim: forwarded to the HF from_pretrained call (the backbone was
            built with a small action head that expects this).
        time_horizon: forwarded to from_pretrained; if None, read from the
            ckpt's config.json.
        backbone_dtype: dtype used for the pretrained weights (default bfloat16
            to match the on-disk checkpoint).
        freeze: if True, set backbone to eval() and stop its gradients.
    """

    def __init__(
        self,
        pretrained_model_path: str,
        d_model: int,
        action_dim: int = 7,
        time_horizon: int | None = None,
        backbone_dtype: str = "bfloat16",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.freeze = bool(freeze)
        self.d_model = int(d_model)

        resolved_horizon = _resolve_transition_time_horizon(
            pretrained_model_path=pretrained_model_path,
            fallback=time_horizon,
        )
        torch_dtype = getattr(torch, backbone_dtype)
        self.backbone = _load_transition_backbone_like_rynnvla002(
            pretrained_model_path=pretrained_model_path,
            action_dim=action_dim,
            time_horizon=resolved_horizon,
            torch_dtype=torch_dtype,
        )

        llm_hidden = int(getattr(self.backbone.config, "hidden_size", 4096))
        self.llm_hidden = llm_hidden

        self.proj_in = nn.Linear(self.d_model, llm_hidden)
        self.proj_out = nn.Linear(llm_hidden, self.d_model)

        if self.freeze:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        backbone_dtype = next(self.backbone.parameters()).dtype

        embeds = self.proj_in(x).to(dtype=backbone_dtype)
        attention_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        grad_ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with grad_ctx:
            outputs = self.backbone.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        h = outputs.last_hidden_state.to(dtype=x.dtype)
        return self.proj_out(h)


def _resolve_pretrained_model_dir(pretrained_model_path: str) -> Path:
    candidate = Path(pretrained_model_path).expanduser().resolve()
    if candidate.is_dir():
        if (candidate / "config.json").is_file():
            return candidate
        for subdir in sorted(path for path in candidate.iterdir() if path.is_dir()):
            if (subdir / "config.json").is_file():
                return subdir.resolve()
    return candidate


def _resolve_transition_time_horizon(pretrained_model_path: str, fallback: int | None) -> int:
    if fallback is not None:
        return int(fallback)
    config_path = _resolve_pretrained_model_dir(pretrained_model_path) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        time_horizon = config.get("time_horizon")
        if time_horizon is not None:
            return int(time_horizon)
    return 5


def _load_transition_backbone_like_rynnvla002(
    pretrained_model_path: str,
    action_dim: int,
    time_horizon: int,
    torch_dtype: torch.dtype,
) -> Any:
    from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
        ChameleonXLLMXForConditionalGeneration_ck_action_head,
    )

    model_dir = _resolve_pretrained_model_dir(pretrained_model_path)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Transition backbone config.json not found under {model_dir}")

    config = json.loads(config_path.read_text())
    model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
        str(model_dir),
        action_dim=int(action_dim),
        time_horizon=int(time_horizon),
        max_position_embeddings=int(config.get("max_position_embeddings", 8192)),
        mask_image_logits=bool(config.get("mask_image_logits", False)),
        dropout=float(config.get("dropout", 0.0)),
        z_loss_weight=float(config.get("z_loss_weight", 0.0)),
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
        device_map="cpu",
        ignore_mismatched_sizes=False,
        low_cpu_mem_usage=True,
    )
    if hasattr(model.model, "vqmodel"):
        del model.model.vqmodel
    return model


__all__ = ["CausalTransformerCell", "LLMBackboneCell"]
