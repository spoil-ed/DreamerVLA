from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


def build_frame_causal_attention_mask(
    frames: int,
    tokens_per_frame: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the V-JEPA2-AC block-causal mask (``True`` means visible).

    All tokens within one frame may attend to one another, while frame ``t``
    can only attend to frames ``<= t``.  This is the same visibility rule as
    V-JEPA2-AC's ``build_action_block_causal_attention_mask`` without assuming
    that the representation tokens form a particular spatial grid.
    """

    if int(frames) < 1:
        raise ValueError(f"frames must be positive, got {frames}")
    if int(tokens_per_frame) < 1:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
    frame_ids = torch.arange(int(frames), device=device).repeat_interleave(int(tokens_per_frame))
    return frame_ids[None, :] <= frame_ids[:, None]


def _rotate_queries_or_keys(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Apply the RoPE convention used by the released V-JEPA2-AC weights."""

    dim = int(x.shape[-1])
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / (10000 ** (omega / (dim / 2.0)))
    frequency = torch.einsum("..., f -> ... f", pos.to(dtype=x.dtype), omega)

    # Keep the released checkpoint's pair-frequency convention.  Although a
    # repeat_interleave formulation looks more natural, changing it alters the
    # function represented by the pretrained Q/K weights.
    sin = frequency.sin().squeeze(-1).repeat(1, 1, 1, 2)
    cos = frequency.cos().squeeze(-1).repeat(1, 1, 1, 2)
    rotated = x.unflatten(-1, (-1, 2))
    first, second = rotated.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return (x * cos) + (rotated * sin)


class _VJEPA2ACMLP(nn.Module):
    """MLP with state-dict names matching the official AC predictor."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class _VJEPA2ACAttention(nn.Module):
    """V-JEPA2-AC attention with representation-aware RoPE positions."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        qkv_bias: bool,
        attention_dropout: float,
        projection_dropout: float,
        use_rope: bool,
        pretrained_grid_size: int,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"predictor dim {dim} must be divisible by heads {num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bool(qkv_bias))
        self.attn_drop = nn.Dropout(float(attention_dropout))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = float(projection_dropout)
        self.proj_drop = nn.Dropout(float(projection_dropout))
        self.use_rope = bool(use_rope)
        self.grid_size = int(pretrained_grid_size)

        # The released AC predictor splits each 64-d head into 20 temporal,
        # 20 height, 20 width and 4 unrotated channels.
        self.d_dim = int(2 * ((self.head_dim // 3) // 2))
        self.h_dim = int(2 * ((self.head_dim // 3) // 2))
        self.w_dim = int(2 * ((self.head_dim // 3) // 2))

    def _rope_visual_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        frames: int,
        tokens: int,
        spatial_grid: tuple[int, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rope:
            return q, k

        token_ids = torch.arange(frames * tokens, device=q.device)
        frame_pos = token_ids // tokens
        temporal_q = _rotate_queries_or_keys(q[..., : self.d_dim], frame_pos)
        temporal_k = _rotate_queries_or_keys(k[..., : self.d_dim], frame_pos)
        offset = self.d_dim

        if spatial_grid is None:
            return (
                torch.cat([temporal_q, q[..., offset:]], dim=-1),
                torch.cat([temporal_k, k[..., offset:]], dim=-1),
            )

        height, width = spatial_grid
        spatial_id = token_ids - (tokens * frame_pos)
        height_pos = (spatial_id // width) * (self.grid_size / height)
        width_pos = (spatial_id % width) * (self.grid_size / width)
        height_q = _rotate_queries_or_keys(q[..., offset : offset + self.h_dim], height_pos)
        height_k = _rotate_queries_or_keys(k[..., offset : offset + self.h_dim], height_pos)
        offset += self.h_dim
        width_q = _rotate_queries_or_keys(q[..., offset : offset + self.w_dim], width_pos)
        width_k = _rotate_queries_or_keys(k[..., offset : offset + self.w_dim], width_pos)
        offset += self.w_dim
        return (
            torch.cat([temporal_q, height_q, width_q, q[..., offset:]], dim=-1),
            torch.cat([temporal_k, height_k, width_k, k[..., offset:]], dim=-1),
        )

    def _rope_condition_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rope:
            return q, k
        frame_pos = torch.arange(frames, device=q.device)
        temporal_q = _rotate_queries_or_keys(q[..., : self.d_dim], frame_pos)
        temporal_k = _rotate_queries_or_keys(k[..., : self.d_dim], frame_pos)
        return (
            torch.cat([temporal_q, q[..., self.d_dim :]], dim=-1),
            torch.cat([temporal_k, k[..., self.d_dim :]], dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        frames: int,
        visual_tokens: int,
        condition_tokens: int,
        spatial_grid: tuple[int, int] | None,
    ) -> torch.Tensor:
        batch, sequence, dim = x.shape
        expected = int(frames) * (int(condition_tokens) + int(visual_tokens))
        if sequence != expected:
            raise ValueError(f"AC sequence has {sequence} tokens, expected {expected}")

        split = x.view(batch, frames, condition_tokens + visual_tokens, dim)
        conditions = [
            split[:, :, index : index + 1].flatten(1, 2) for index in range(condition_tokens)
        ]
        visual = split[:, :, condition_tokens:].flatten(1, 2)

        def qkv(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            packed = self.qkv(value).unflatten(-1, (3, self.num_heads, self.head_dim))
            packed = packed.permute(2, 0, 3, 1, 4)
            return packed[0], packed[1], packed[2]

        visual_q, visual_k, visual_v = qkv(visual)
        visual_q, visual_k = self._rope_visual_tokens(
            visual_q,
            visual_k,
            frames=frames,
            tokens=visual_tokens,
            spatial_grid=spatial_grid,
        )

        condition_q: list[torch.Tensor] = []
        condition_k: list[torch.Tensor] = []
        condition_v: list[torch.Tensor] = []
        for value in conditions:
            query, key, val = qkv(value)
            query, key = self._rope_condition_tokens(query, key, frames=frames)
            condition_q.append(query.view(batch, self.num_heads, frames, 1, self.head_dim))
            condition_k.append(key.view(batch, self.num_heads, frames, 1, self.head_dim))
            condition_v.append(val.view(batch, self.num_heads, frames, 1, self.head_dim))

        def interleave(
            visual_value: torch.Tensor, condition_values: list[torch.Tensor]
        ) -> torch.Tensor:
            visual_value = visual_value.view(
                batch, self.num_heads, frames, visual_tokens, self.head_dim
            )
            conditions_value = torch.cat(condition_values, dim=3)
            return torch.cat([conditions_value, visual_value], dim=3).flatten(2, 3)

        query = interleave(visual_q, condition_q)
        key = interleave(visual_k, condition_k)
        value = interleave(visual_v, condition_v)
        # Match the released ACRoPEAttention SDPA path, which feeds the
        # projection-drop probability into attention dropout.
        dropout_p = self.proj_drop_prob if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence, dim)
        return self.proj_drop(self.proj(attended))


class _VJEPA2ACBlock(nn.Module):
    """Pre-norm residual block with official checkpoint key names."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        dropout: float,
        attention_dropout: float,
        use_rope: bool,
        pretrained_grid_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1.0e-6)
        self.attn = _VJEPA2ACAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            use_rope=use_rope,
            pretrained_grid_size=pretrained_grid_size,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1.0e-6)
        self.mlp = _VJEPA2ACMLP(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        frames: int,
        visual_tokens: int,
        condition_tokens: int,
        spatial_grid: tuple[int, int] | None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            attention_mask=attention_mask,
            frames=frames,
            visual_tokens=visual_tokens,
            condition_tokens=condition_tokens,
            spatial_grid=spatial_grid,
        )
        return x + self.mlp(self.norm2(x))


@dataclass(frozen=True)
class VJEPA2ACLoadReport:
    """Explicit accounting for a partial V-JEPA2-AC checkpoint load."""

    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    mismatched_keys: tuple[str, ...]
    unused_checkpoint_keys: tuple[str, ...]
    loaded_parameters: int
    transition_parameters: int

    @property
    def pretrained_parameter_ratio(self) -> float:
        if self.transition_parameters == 0:
            return 0.0
        return self.loaded_parameters / self.transition_parameters


class VJEPA2ACTransition(nn.Module):
    """Adapter-wrapped V-JEPA2-AC transition over DreamerVLA latent tokens.

    The 24 Transformer blocks retain the released 1024-d QKV/MLP/LayerNorm
    shapes.  Representation mismatches are isolated in ``input_adapter``,
    ``output_adapter`` and ``state_input_adapter``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        action_dim: int,
        state_dim: int,
        token_count: int,
        context_dim: int = 0,
        state_output_dim: int = 0,
        predictor_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_rope: bool = True,
        spatial_grid: tuple[int, int] | None = None,
        pretrained_grid_size: int = 16,
        use_activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.token_count = int(token_count)
        self.context_dim = int(context_dim)
        self.state_output_dim = int(state_output_dim)
        self.predictor_dim = int(predictor_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.use_rope = bool(use_rope)
        self.pretrained_grid_size = int(pretrained_grid_size)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.condition_tokens = 2 + int(self.context_dim > 0)
        self.spatial_grid = self._validate_spatial_grid(spatial_grid)

        self.input_adapter = nn.Linear(self.input_dim, self.predictor_dim, bias=True)
        self.action_encoder = nn.Linear(self.action_dim, self.predictor_dim, bias=True)
        # The released AC predictor uses the same 7-d input width for action
        # and state.  Preserve that state_encoder behind a small adapter when
        # DreamerVLA's proprio width differs (currently 8).
        self.state_input_adapter: nn.Module = (
            nn.Identity()
            if self.state_dim == self.action_dim
            else nn.Linear(self.state_dim, self.action_dim, bias=True)
        )
        self.state_encoder = nn.Linear(self.action_dim, self.predictor_dim, bias=True)
        self.context_encoder = (
            nn.Linear(self.context_dim, self.predictor_dim, bias=True)
            if self.context_dim > 0
            else None
        )
        self.predictor_blocks = nn.ModuleList(
            [
                _VJEPA2ACBlock(
                    self.predictor_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    dropout=float(dropout),
                    attention_dropout=float(attention_dropout),
                    use_rope=self.use_rope,
                    pretrained_grid_size=self.pretrained_grid_size,
                )
                for _ in range(self.depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(self.predictor_dim, eps=1.0e-6)
        self.output_adapter = nn.Linear(self.predictor_dim, self.output_dim, bias=True)
        self.state_output_adapter = (
            nn.Linear(self.predictor_dim, self.state_output_dim, bias=True)
            if self.state_output_dim > 0
            else None
        )
        self.pretrained_load_report: VJEPA2ACLoadReport | None = None
        self._init_weights()

    def _validate_spatial_grid(
        self, spatial_grid: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        if spatial_grid is None:
            return None
        if len(spatial_grid) != 2:
            raise ValueError("spatial_grid must be [height, width] or null")
        height, width = (int(value) for value in spatial_grid)
        if height < 1 or width < 1:
            raise ValueError("spatial_grid dimensions must be positive")
        if height * width != self.token_count:
            raise ValueError(
                "spatial_grid may only be set for genuine spatial patch tokens: "
                f"{height}*{width} != token_count={self.token_count}; use null for temporal-only RoPE"
            )
        return height, width

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for layer_id, block in enumerate(self.predictor_blocks, start=1):
            block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    @staticmethod
    def _unwrap_predictor_state(checkpoint_payload: Any) -> Mapping[str, torch.Tensor]:
        if not isinstance(checkpoint_payload, Mapping):
            raise TypeError("V-JEPA2-AC checkpoint must contain a state-dict mapping")
        candidate: Any = checkpoint_payload
        for key in ("predictor", "state_dict"):
            nested = candidate.get(key) if isinstance(candidate, Mapping) else None
            if isinstance(nested, Mapping):
                candidate = nested
                if key == "predictor":
                    break
        if not isinstance(candidate, Mapping):
            raise TypeError("V-JEPA2-AC predictor state dict is not a mapping")
        tensors = {str(key): value for key, value in candidate.items() if torch.is_tensor(value)}
        if not tensors:
            raise ValueError("V-JEPA2-AC checkpoint contains no predictor tensors")
        return tensors

    @staticmethod
    def _canonical_checkpoint_key(key: str) -> str:
        clean = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone.", "predictor.", "vjepa2_transition."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        return clean

    @staticmethod
    def _source_key_for_model_key(model_key: str) -> str:
        if model_key.startswith("input_adapter."):
            return model_key.replace("input_adapter.", "predictor_embed.", 1)
        if model_key.startswith("output_adapter."):
            return model_key.replace("output_adapter.", "predictor_proj.", 1)
        return model_key

    def load_pretrained(self, checkpoint_path: str | Path) -> VJEPA2ACLoadReport:
        """Load every shape-compatible AC predictor tensor and report all gaps."""

        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"V-JEPA2-AC checkpoint not found: {path}. Set world_model.vjepa2_checkpoint_path."
            )
        load_kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
        try:
            payload = torch.load(path, mmap=True, **load_kwargs)
        except (TypeError, RuntimeError):
            payload = torch.load(path, **load_kwargs)
        raw_state = self._unwrap_predictor_state(payload)
        checkpoint_state = {
            self._canonical_checkpoint_key(key): value for key, value in raw_state.items()
        }

        model_state = self.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        mismatched: list[str] = []
        consumed_source_keys: set[str] = set()
        for model_key, model_value in model_state.items():
            source_key = self._source_key_for_model_key(model_key)
            source_value = checkpoint_state.get(source_key)
            if source_value is None:
                missing.append(f"{model_key} <- {source_key}")
                continue
            consumed_source_keys.add(source_key)
            if tuple(source_value.shape) != tuple(model_value.shape):
                mismatched.append(
                    f"{model_key} <- {source_key}: checkpoint{tuple(source_value.shape)} "
                    f"!= model{tuple(model_value.shape)}"
                )
                continue
            compatible[model_key] = source_value

        incompatible = self.load_state_dict(compatible, strict=False)
        unexpected_model_keys = tuple(sorted(incompatible.unexpected_keys))
        if unexpected_model_keys:
            raise RuntimeError(
                "filtered V-JEPA2-AC loading produced unexpected model keys: "
                f"{unexpected_model_keys}"
            )
        loaded = tuple(sorted(compatible))
        expected_skipped = set(model_state).difference(compatible)
        if set(incompatible.missing_keys) != expected_skipped:
            raise RuntimeError(
                "filtered V-JEPA2-AC loading returned an inconsistent missing-key set"
            )
        missing = sorted(set(missing))
        unused = tuple(sorted(set(checkpoint_state).difference(consumed_source_keys)))
        loaded_parameters = sum(int(model_state[key].numel()) for key in loaded)
        transition_parameters = sum(int(value.numel()) for value in model_state.values())
        report = VJEPA2ACLoadReport(
            loaded_keys=loaded,
            missing_keys=tuple(missing),
            mismatched_keys=tuple(sorted(mismatched)),
            unused_checkpoint_keys=unused,
            loaded_parameters=loaded_parameters,
            transition_parameters=transition_parameters,
        )
        self.pretrained_load_report = report
        self._log_load_report(path, report)
        return report

    @staticmethod
    def _log_load_report(path: Path, report: VJEPA2ACLoadReport) -> None:
        logger.info("V-JEPA2-AC pretrained checkpoint: %s", path)
        logger.info("V-JEPA2-AC loaded keys (%d): %s", len(report.loaded_keys), report.loaded_keys)
        logger.info(
            "V-JEPA2-AC missing keys (%d): %s", len(report.missing_keys), report.missing_keys
        )
        logger.info(
            "V-JEPA2-AC mismatched keys (%d): %s",
            len(report.mismatched_keys),
            report.mismatched_keys,
        )
        logger.info(
            "V-JEPA2-AC unused checkpoint keys (%d): %s",
            len(report.unused_checkpoint_keys),
            report.unused_checkpoint_keys,
        )
        logger.info(
            "V-JEPA2-AC loaded parameters: %d / %d (pretrained parameter ratio %.6f)",
            report.loaded_parameters,
            report.transition_parameters,
            report.pretrained_parameter_ratio,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict latent tokens with the action/state-conditioned causal backbone."""

        if tokens.ndim != 4:
            raise ValueError(f"tokens must be [B,T,N,D], got {tuple(tokens.shape)}")
        batch, frames, token_count, width = tokens.shape
        if token_count != self.token_count or width != self.input_dim:
            raise ValueError(
                f"token shape mismatch: got N={token_count}, D={width}; "
                f"expected N={self.token_count}, D={self.input_dim}"
            )
        expected_condition_shape = (batch, frames)
        if actions.shape[:2] != expected_condition_shape or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must be [B,T,{self.action_dim}], got {tuple(actions.shape)}")
        if states.shape[:2] != expected_condition_shape or states.shape[-1] != self.state_dim:
            raise ValueError(f"states must be [B,T,{self.state_dim}], got {tuple(states.shape)}")
        if self.context_dim > 0:
            if context is None or context.shape[:2] != expected_condition_shape:
                raise ValueError(
                    f"context must be [B,T,{self.context_dim}], got "
                    f"{None if context is None else tuple(context.shape)}"
                )
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"context must be [B,T,{self.context_dim}], got {tuple(context.shape)}"
                )
        elif context is not None:
            raise ValueError("context was provided but context_dim=0")

        visual = self.input_adapter(tokens)
        action = self.action_encoder(actions).unsqueeze(2)
        state = self.state_encoder(self.state_input_adapter(states)).unsqueeze(2)
        conditions = [action, state]
        if self.context_encoder is not None:
            if context is None:
                raise RuntimeError("context encoder requires context")
            conditions.append(self.context_encoder(context).unsqueeze(2))
        hidden = torch.cat([*conditions, visual], dim=2).flatten(1, 2)
        attention_mask = build_frame_causal_attention_mask(
            frames,
            self.condition_tokens + self.token_count,
            device=hidden.device,
        )
        for block in self.predictor_blocks:
            kwargs = {
                "attention_mask": attention_mask,
                "frames": frames,
                "visual_tokens": self.token_count,
                "condition_tokens": self.condition_tokens,
                "spatial_grid": self.spatial_grid,
            }
            if self.use_activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, use_reentrant=False, **kwargs)
            else:
                hidden = block(hidden, **kwargs)

        hidden = hidden.view(
            batch,
            frames,
            self.condition_tokens + self.token_count,
            self.predictor_dim,
        )
        state_hidden = hidden[:, :, 1]
        visual = hidden[:, :, self.condition_tokens :]
        output = self.output_adapter(self.predictor_norm(visual))
        if self.state_output_adapter is not None:
            state_output = self.state_output_adapter(self.predictor_norm(state_hidden))
            state_output = state_output[:, :, None, :].expand(-1, -1, self.token_count, -1)
            output = torch.cat([output, state_output], dim=-1)
        return output
