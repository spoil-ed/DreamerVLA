from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dreamervla.models.embodiment.world_model.vjepa2_ac_transition import (
    VJEPA2ACLoadReport,
    VJEPA2ACTransition,
)
from dreamervla.models.embodiment.world_model.wm import WorldModel


class _WMStyleFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _WMStyleAttention(nn.Module):
    """WM-style attention with residual dim independent of QKV inner dim."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        attn_impl: str = "manual",
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if dim_head < 1:
            raise ValueError(f"dim_head must be >= 1, got {dim_head}")
        if attn_impl not in ("manual", "sdpa"):
            raise ValueError(f"attn_impl must be 'manual' or 'sdpa', got {attn_impl!r}")
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.attn_impl = str(attn_impl)
        self.dropout_p = float(dropout)
        self.scale = float(dim_head) ** -0.5
        inner_dim = self.heads * self.dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _dim = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(bsz, seq_len, self.heads, self.dim_head).transpose(1, 2) for t in qkv)
        if self.attn_impl == "sdpa":
            attn_mask = (
                mask.to(device=q.device, dtype=q.dtype)[None, None] if mask is not None else None
            )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=self.scale,
            )
            out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
            return self.to_out(out)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            dots = dots + mask.to(device=dots.device, dtype=dots.dtype)[None, None]
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.to_out(out)


class _WMStyleTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        attn_impl: str = "manual",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _WMStyleAttention(
                            dim,
                            heads=int(heads),
                            dim_head=int(dim_head),
                            dropout=float(dropout),
                            attn_impl=str(attn_impl),
                        ),
                        _WMStyleFeedForward(dim, int(mlp_dim), dropout=float(dropout)),
                    ]
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return self.norm(x)


class ChunkAwareWorldModel(WorldModel):
    """Chunk WM over per-frame latent tokens with WM-style conditioning.

    The transition model keeps each observation token in source token space and
    concatenates an encoded action to every observation token channel, matching
    the default WM ``concat_dim=1`` pattern.  A chunk is rolled out
    autoregressively: every step predicts ``e_{t+1}`` from the latest
    ``num_hist`` latent frames conditioned on the current action, then slides
    the predicted observation tokens into the next history.
    """

    def __init__(
        self,
        *args: Any,
        chunk_size: int = 8,
        chunk_rollout_chunks: int = 1,
        chunk_rollout_loss_scale: float = 0.0,
        grad_checkpoint: bool = False,
        action_emb_dim: int = 10,
        num_action_repeat: int = 1,
        proprio_dim: int = 0,
        proprio_emb_dim: int = 0,
        num_proprio_repeat: int = 1,
        proprio_reconstruction_loss_scale: float = 0.0,
        lang_dim: int = 0,
        lang_emb_dim: int = 0,
        num_lang_repeat: int = 1,
        dim_head: int = 64,
        attn_impl: str = "manual",
        token_normalization: str = "layer_norm",
        token_norm_eps: float = 1.0e-6,
        task_conditioning: dict | None = None,
        transition_type: str = "original",
        transition_init: str = "random",
        vjepa2_checkpoint_path: str | None = None,
        vjepa2_predictor_dim: int = 1024,
        vjepa2_depth: int = 24,
        vjepa2_num_heads: int = 16,
        vjepa2_mlp_ratio: float = 4.0,
        vjepa2_dropout: float = 0.0,
        vjepa2_attention_dropout: float = 0.0,
        vjepa2_use_rope: bool = True,
        vjepa2_spatial_grid: list[int] | tuple[int, int] | None = None,
        vjepa2_spatial_group_count: int = 1,
        vjepa2_pretrained_grid_size: int = 16,
        vjepa2_residual_prediction: bool = False,
        vjepa2_residual_output_init_std: float = 1.0e-3,
        vjepa2_truncate_rollout_gradients: bool = True,
        vjepa2_rollout_bptt_steps: int = 1,
        vjepa2_proprio_representation: str = "legacy",
        vjepa2_proprio_loss_scale: float = 0.0,
        vjepa2_one_step_loss_scale: float = 0.0,
        vjepa2_temporal_difference_loss_scale: float = 0.0,
        vjepa2_layer_scale_init: float | None = None,
        decoded_visual_loss: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        self.transition_type = str(transition_type).strip().lower()
        self.transition_init = str(transition_init).strip().lower()
        self.vjepa2_truncate_rollout_gradients = bool(vjepa2_truncate_rollout_gradients)
        self.vjepa2_rollout_bptt_steps = int(vjepa2_rollout_bptt_steps)
        if self.vjepa2_rollout_bptt_steps < 1:
            raise ValueError("vjepa2_rollout_bptt_steps must be positive")
        self.vjepa2_proprio_representation = str(vjepa2_proprio_representation)
        if self.vjepa2_proprio_representation not in {"legacy", "raw_padded"}:
            raise ValueError("vjepa2_proprio_representation must be legacy or raw_padded")
        self.vjepa2_proprio_loss_scale = float(vjepa2_proprio_loss_scale)
        self.vjepa2_one_step_loss_scale = float(vjepa2_one_step_loss_scale)
        self.vjepa2_temporal_difference_loss_scale = float(vjepa2_temporal_difference_loss_scale)
        if self.vjepa2_one_step_loss_scale < 0 or self.vjepa2_temporal_difference_loss_scale < 0:
            raise ValueError("AC one-step and temporal-difference loss scales must be non-negative")
        if self.vjepa2_proprio_loss_scale < 0:
            raise ValueError("vjepa2_proprio_loss_scale must be non-negative")
        self._raw_proprio_slots = (
            self.transition_type == "vjepa2_ac"
            and self.vjepa2_proprio_representation == "raw_padded"
        )
        self.vjepa2_residual_output_init_std = float(vjepa2_residual_output_init_std)
        if self.transition_type not in {"original", "vjepa2_ac"}:
            raise ValueError("transition_type must be 'original' or 'vjepa2_ac'")
        if self.transition_init not in {"random", "pretrained"}:
            raise ValueError("transition_init must be 'random' or 'pretrained'")
        if self.transition_type == "original" and self.transition_init != "random":
            raise ValueError("transition_init='pretrained' requires transition_type='vjepa2_ac'")
        args_list = list(args)
        requested_model_dim = args_list[5] if len(args_list) > 5 else kwargs.get("model_dim")
        token_dim_hint = int(args_list[3] if len(args_list) > 3 else kwargs.get("token_dim", 4096))
        heads_hint = int(args_list[7] if len(args_list) > 7 else kwargs.get("heads", 8))
        # The parent allocates a transition that this subclass replaces.  Keep
        # its temporary allocation tiny on the V-JEPA route; the original path
        # retains its historical construction exactly.
        safe_parent_model_dim = (
            heads_hint if self.transition_type == "vjepa2_ac" else max(token_dim_hint, heads_hint)
        )
        if safe_parent_model_dim % heads_hint != 0:
            safe_parent_model_dim += heads_hint - (safe_parent_model_dim % heads_hint)
        if len(args_list) > 5:
            args_list[5] = safe_parent_model_dim
        else:
            kwargs = dict(kwargs)
            kwargs["model_dim"] = safe_parent_model_dim

        super().__init__(*args_list, **kwargs)
        self.action_emb_dim = int(action_emb_dim)
        self.num_action_repeat = int(num_action_repeat)
        self.action_condition_dim = self.action_emb_dim * self.num_action_repeat
        if self.action_emb_dim < 1:
            raise ValueError(f"action_emb_dim must be >= 1, got {action_emb_dim}")
        if self.num_action_repeat < 1:
            raise ValueError(f"num_action_repeat must be >= 1, got {num_action_repeat}")
        self.proprio_dim = int(proprio_dim)
        self.proprio_emb_dim = int(proprio_emb_dim)
        self.num_proprio_repeat = int(num_proprio_repeat)
        self.proprio_reconstruction_loss_scale = float(proprio_reconstruction_loss_scale)
        if self.proprio_reconstruction_loss_scale < 0:
            raise ValueError(
                "proprio_reconstruction_loss_scale must be >= 0, got "
                f"{proprio_reconstruction_loss_scale}"
            )
        if self.proprio_emb_dim < 0:
            raise ValueError(f"proprio_emb_dim must be >= 0, got {proprio_emb_dim}")
        if self.num_proprio_repeat < 1:
            raise ValueError(f"num_proprio_repeat must be >= 1, got {num_proprio_repeat}")
        self.proprio_condition_dim = self.proprio_emb_dim * self.num_proprio_repeat
        if self.proprio_condition_dim > 0:
            if self.proprio_dim < 1:
                raise ValueError("proprio_emb_dim>0 requires proprio_dim>=1")
            self.proprio_encoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, self.proprio_emb_dim),
            )
            self.proprio_decoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.proprio_condition_dim),
                nn.Linear(self.proprio_condition_dim, self.proprio_dim),
            )
        else:
            self.proprio_encoder = None
            self.proprio_decoder = None
        if self._raw_proprio_slots and self.proprio_dim > 0:
            if self.proprio_condition_dim < self.proprio_dim:
                raise ValueError("raw_padded proprio slots must fit every raw proprio dimension")
            if self.vjepa2_proprio_loss_scale <= 0:
                raise ValueError(
                    "raw_padded state prediction requires vjepa2_proprio_loss_scale > 0"
                )
            # Raw state is already the external physical contract. Padding and
            # slicing preserve it exactly, unlike a learned LayerNorm codec.
            self.proprio_encoder = nn.Identity()
            self.proprio_decoder = nn.Identity()
        self.obs_token_dim = self.token_dim + self.proprio_condition_dim

        self.lang_dim = int(lang_dim)
        self.lang_emb_dim = int(lang_emb_dim)
        self.num_lang_repeat = int(num_lang_repeat)
        if self.lang_emb_dim < 0:
            raise ValueError(f"lang_emb_dim must be >= 0, got {lang_emb_dim}")
        if self.num_lang_repeat < 1:
            raise ValueError(f"num_lang_repeat must be >= 1, got {num_lang_repeat}")
        self.lang_condition_dim = self.lang_emb_dim * self.num_lang_repeat
        if self.lang_condition_dim > 0:
            if self.lang_dim < 1:
                raise ValueError("lang_emb_dim>0 requires lang_dim>=1")
            self.lang_proj: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.lang_dim),
                nn.Linear(self.lang_dim, self.lang_emb_dim),
            )
        else:
            self.lang_proj = None

        expected_model_dim = (
            self.obs_token_dim + self.lang_condition_dim + self.action_condition_dim
        )
        if requested_model_dim is None:
            requested_model_dim = expected_model_dim
        self.model_dim = int(requested_model_dim)
        if self.model_dim != expected_model_dim:
            raise ValueError(
                "ChunkAwareWorldModel uses WM concat conditioning; "
                "set model_dim == token_dim + proprio_emb_dim * num_proprio_repeat "
                "+ lang_emb_dim * num_lang_repeat + action_emb_dim * "
                "num_action_repeat, "
                f"got model_dim={self.model_dim}, token_dim={self.token_dim}, "
                f"proprio_emb_dim={self.proprio_emb_dim}, "
                f"num_proprio_repeat={self.num_proprio_repeat}, "
                f"lang_emb_dim={self.lang_emb_dim}, "
                f"num_lang_repeat={self.num_lang_repeat}, "
                f"action_emb_dim={self.action_emb_dim}, "
                f"num_action_repeat={self.num_action_repeat}"
            )
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if int(chunk_rollout_chunks) < 1:
            raise ValueError(f"chunk_rollout_chunks must be >= 1, got {chunk_rollout_chunks}")
        if float(chunk_rollout_loss_scale) < 0:
            raise ValueError(
                f"chunk_rollout_loss_scale must be >= 0, got {chunk_rollout_loss_scale}"
            )
        self.chunk_size = int(chunk_size)
        # Close-loop multi-chunk rollout (anti-drift, à la Dreamer rollout loss):
        # train the model to predict chunk c+1..c+N-1 from its OWN predicted
        # history rolled forward from chunk 0.  chunk 0 is teacher-forced (covered
        # by the base chunk_loss).  Set ``chunk_rollout_chunks`` > 1 AND
        # ``chunk_rollout_loss_scale`` > 0 to enable.
        self.chunk_rollout_chunks = int(chunk_rollout_chunks)
        self.chunk_rollout_loss_scale = float(chunk_rollout_loss_scale)
        # Recompute each autoregressive step's activations in backward instead of
        # storing them — cuts the chunk rollout's activation memory from O(N*K) to
        # O(1). Opt-in; numerically identical to the plain path. See predict_next_chunk.
        self.grad_checkpoint = bool(grad_checkpoint)
        self.dim_head = int(dim_head)
        self.attn_impl = str(attn_impl)
        self.token_normalization = str(token_normalization).strip().lower()
        self.token_norm_eps = float(token_norm_eps)
        if self.token_normalization not in {"layer_norm", "none"}:
            raise ValueError("token_normalization must be 'layer_norm' or 'none'")
        if self.token_norm_eps <= 0.0:
            raise ValueError("token_norm_eps must be positive")
        task_cfg = dict(task_conditioning or {})
        self.task_conditioning_enabled = bool(task_cfg.get("enabled", False))
        self.supports_task_conditioning = bool(self.task_conditioning_enabled)
        if self.task_conditioning_enabled:
            num_tasks = int(task_cfg.get("num_tasks", 0) or 0)
            embedding_dim = int(task_cfg.get("embedding_dim", 0) or 0)
            if num_tasks <= 0 or embedding_dim <= 0:
                raise ValueError(
                    "world_model.task_conditioning requires positive num_tasks and embedding_dim"
                )
            if embedding_dim != int(self.token_dim):
                raise ValueError(
                    "ChunkAwareWorldModel task_conditioning.embedding_dim must match "
                    f"token_dim ({embedding_dim} != {int(self.token_dim)})"
                )
            self.task_embedding = nn.Embedding(num_tasks, int(self.token_dim))
        else:
            self.task_embedding = None
        self.slots_per_step = self.token_count
        self.pos_context_len = self.num_hist
        self.obs_norm = (
            nn.LayerNorm(
                self.token_dim,
                eps=self.token_norm_eps,
                elementwise_affine=False,
            )
            if self.token_normalization == "layer_norm"
            else nn.Identity()
        )
        self.obs_proj = nn.Identity()
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.action_emb_dim),
        )
        if self.reward_enabled:
            self.reward_norm = nn.LayerNorm(self.obs_token_dim)
            self.reward_head = nn.Sequential(
                nn.Linear(self.obs_token_dim, self.reward_hidden_dim),
                nn.GELU(),
                nn.Linear(self.reward_hidden_dim, 1),
            )
            final = self.reward_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.constant_(final.bias, self.reward_init_logit)
        if self.success_return_enabled:
            self.success_return_norm = nn.LayerNorm(self.obs_token_dim)
            self.success_return_head = nn.Sequential(
                nn.Linear(self.obs_token_dim, self.success_return_hidden_dim),
                nn.GELU(),
                nn.Linear(self.success_return_hidden_dim, 1),
            )
            final = self.success_return_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.constant_(final.bias, self.success_return_init_logit)
        self.vjepa2_transition: VJEPA2ACTransition | None = None
        self.pretrained_load_report: VJEPA2ACLoadReport | None = None
        self._vjepa2_token_input_dim = self.token_dim
        self._vjepa2_context_dim = self.lang_condition_dim
        self._vjepa2_state_dim = self.proprio_dim if self.proprio_dim > 0 else self.action_dim
        if self.transition_type == "original":
            self.pos_embedding: nn.Parameter | None = nn.Parameter(
                torch.randn(1, self.pos_context_len * self.slots_per_step, self.model_dim) * 0.02
            )
            self.predictor = _WMStyleTransformer(
                dim=self.model_dim,
                depth=int(kwargs.get("depth", 6) if len(args_list) <= 6 else args_list[6]),
                heads=int(kwargs.get("heads", 8) if len(args_list) <= 7 else args_list[7]),
                dim_head=self.dim_head,
                mlp_dim=int(kwargs.get("mlp_dim", 2048) if len(args_list) <= 8 else args_list[8]),
                dropout=float(kwargs.get("dropout", 0.1) if len(args_list) <= 9 else args_list[9]),
                attn_impl=self.attn_impl,
            )
        else:
            self.register_parameter("pos_embedding", None)
            self.predictor = nn.Identity()
            # The AC route consumes raw actions with the pretrained
            # action_encoder, so the baseline's projected-and-concatenated
            # action embedding is intentionally out of graph.
            for parameter in self.action_proj.parameters():
                parameter.requires_grad = False
            spatial_grid = (
                None
                if vjepa2_spatial_grid is None
                else tuple(int(value) for value in vjepa2_spatial_grid)
            )
            self.vjepa2_transition = VJEPA2ACTransition(
                input_dim=self._vjepa2_token_input_dim,
                output_dim=self.token_dim,
                action_dim=self.action_dim,
                state_dim=self._vjepa2_state_dim,
                token_count=self.token_count,
                context_dim=self._vjepa2_context_dim,
                state_output_dim=self.proprio_condition_dim,
                predictor_dim=int(vjepa2_predictor_dim),
                depth=int(vjepa2_depth),
                num_heads=int(vjepa2_num_heads),
                mlp_ratio=float(vjepa2_mlp_ratio),
                dropout=float(vjepa2_dropout),
                attention_dropout=float(vjepa2_attention_dropout),
                use_rope=bool(vjepa2_use_rope),
                spatial_grid=spatial_grid,
                spatial_group_count=int(vjepa2_spatial_group_count),
                pretrained_grid_size=int(vjepa2_pretrained_grid_size),
                use_activation_checkpointing=self.grad_checkpoint,
                residual_prediction=bool(vjepa2_residual_prediction),
                residual_output_init_std=float(vjepa2_residual_output_init_std),
                state_residual_prediction=self._raw_proprio_slots and self.proprio_dim > 0,
                layer_scale_init=vjepa2_layer_scale_init,
            )
            if self.transition_init == "pretrained":
                if not vjepa2_checkpoint_path:
                    raise ValueError("transition_init='pretrained' requires vjepa2_checkpoint_path")
                self.pretrained_load_report = self.vjepa2_transition.load_pretrained(
                    vjepa2_checkpoint_path
                )
        self.out_norm = nn.Identity()
        self.out_proj = nn.Identity()
        self.decoded_visual_loss = decoded_visual_loss
        if decoded_visual_loss is not None:
            if self.transition_type != "vjepa2_ac":
                raise ValueError("decoded_visual_loss is opt-in on the V-JEPA2-AC route only")
            if self.vjepa2_truncate_rollout_gradients:
                raise ValueError("decoded_visual_loss requires full closed-loop gradients")
            if self.task_conditioning_enabled:
                raise ValueError("decoded_visual_loss requires unmodified visual tokens")
        if self.freeze_input_embeddings_requested:
            self.freeze_input_embeddings()
            if self.vjepa2_transition is not None:
                input_modules = (
                    self.vjepa2_transition.input_adapter,
                    self.vjepa2_transition.action_encoder,
                    self.vjepa2_transition.state_input_adapter,
                    self.vjepa2_transition.state_encoder,
                    self.vjepa2_transition.context_encoder,
                )
                for module in input_modules:
                    if module is None:
                        continue
                    module.eval()
                    for parameter in module.parameters():
                        parameter.requires_grad = False

    # ------------------------------------------------------------------ #
    # WM-style action concat transition                                  #
    # ------------------------------------------------------------------ #
    def optimizer_parameter_groups(self) -> list[dict[str, Any]]:
        """Separate transferred AC weights from representation adapters."""

        trainable = [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        if self.transition_type != "vjepa2_ac":
            return [
                {
                    "group_name": "default",
                    "params": [parameter for _, parameter in trainable],
                }
            ]
        if self.transition_init == "pretrained":
            if self.pretrained_load_report is None:
                raise RuntimeError("pretrained transition is missing its load report")
            loaded_names = {
                f"vjepa2_transition.{name}" for name in self.pretrained_load_report.loaded_keys
            }
        else:
            # The random-AC control must use the same alignment and LR protocol
            # as the transferred backbone. Keep the historical group name for
            # optimizer/checkpoint compatibility; here it names the module role.
            prefixes = tuple(
                f"vjepa2_transition.{prefix}"
                for prefix in (
                    "predictor_blocks.",
                    "predictor_norm.",
                    "action_encoder.",
                    "state_encoder.",
                )
            )
            loaded_names = {
                name
                for name, _ in trainable
                if name.startswith(prefixes) and not name.endswith("_layer_scale")
            }
        backbone = [parameter for name, parameter in trainable if name in loaded_names]
        adapters = [parameter for name, parameter in trainable if name not in loaded_names]
        return [
            {"group_name": "adapter", "params": adapters},
            {"group_name": "pretrained_backbone", "params": backbone},
        ]

    def _module_dtype(self) -> torch.dtype:
        return self.action_proj[-1].weight.dtype

    def _module_device(self) -> torch.device:
        return self.action_proj[-1].weight.device

    def _apply_task_conditioning(
        self,
        obs_tokens: torch.Tensor,
        task_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.task_conditioning_enabled:
            return obs_tokens
        if task_ids is None:
            raise ValueError("task_ids are required when world model task conditioning is enabled")
        if self.task_embedding is None:
            raise RuntimeError("task conditioning is enabled without an embedding")
        task_emb = self.task_embedding(task_ids.to(obs_tokens.device).long())
        while task_emb.ndim < obs_tokens.ndim:
            task_emb = task_emb.unsqueeze(1)
        return obs_tokens + task_emb.to(obs_tokens.dtype)

    def _normalize_raw_vision_tokens(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        """Apply the configured per-token norm at the external latent boundary."""

        return self.obs_norm(self.obs_to_tokens(obs_embedding))

    def encode_latent(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Normalize a real latent observation before creating online state."""

        tokens = self._normalize_raw_vision_tokens(hidden)
        batch_size = int(tokens.shape[0])
        if tokens.shape[1] >= self.num_hist:
            history = tokens[:, -self.num_hist :]
        else:
            pad = tokens[:, :1].expand(
                -1,
                self.num_hist - tokens.shape[1],
                -1,
                -1,
            )
            history = torch.cat([pad, tokens], dim=1)
        return {
            "hidden": history[:, -1],
            "history": history,
            "actions": torch.zeros(
                batch_size,
                self.num_hist,
                self.action_dim,
                device=history.device,
                dtype=history.dtype,
            ),
        }

    def observe_next(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        is_first: bool | torch.Tensor = False,
    ) -> dict[str, torch.Tensor]:
        """Normalize each newly observed real token grid exactly once."""

        reset = (
            bool(is_first.detach().flatten()[0].item())
            if isinstance(is_first, torch.Tensor)
            else bool(is_first)
        )
        if reset:
            return self.encode_latent(hidden)
        history = self._latent_history(latent)
        batch_size = int(history.shape[0])
        next_hidden = self._normalize_raw_vision_tokens(hidden)[:, -1]
        action_history = self._latent_actions(latent, batch_size).clone()
        action = actions[:, 0] if actions.ndim == 3 else actions
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"observe_next action must be [B,{self.action_dim}], got {tuple(actions.shape)}"
            )
        action_history[:, -1] = action.to(
            device=history.device,
            dtype=history.dtype,
        )
        return {
            "hidden": next_hidden,
            "history": torch.cat([history[:, 1:], next_hidden[:, None]], dim=1),
            "actions": torch.cat(
                [
                    action_history[:, 1:],
                    action_history.new_zeros(batch_size, 1, self.action_dim),
                ],
                dim=1,
            ),
        }

    def _obs_tokens_from_obs(self, obs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Normalize raw vision tokens or expanded observation tokens."""
        obs_embedding = self._obs_embedding_from_obs(obs)
        if obs_embedding.ndim == 3 and obs_embedding.shape[1:] == (
            self.token_count,
            self.obs_token_dim,
        ):
            tokens = obs_embedding[:, None]
        elif obs_embedding.ndim == 4 and obs_embedding.shape[-2] == self.token_count:
            if obs_embedding.shape[-1] not in {self.token_dim, self.obs_token_dim}:
                raise ValueError(
                    "tokenized observation width mismatch: got "
                    f"{obs_embedding.shape[-1]}, expected {self.token_dim} "
                    f"or {self.obs_token_dim}"
                )
            tokens = obs_embedding
        else:
            tokens = self.obs_to_tokens(obs_embedding)
        if int(tokens.shape[1]) > self.max_seq_len:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_seq_len={self.max_seq_len}"
            )
        return tokens.to(device=self._module_device(), dtype=self._module_dtype())

    def _observation_tokens(
        self,
        vision_tokens: torch.Tensor,
        proprio_raw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fold encoded proprio into every observation token channel."""
        if self.proprio_condition_dim == 0:
            return vision_tokens
        if proprio_raw is None:
            raise ValueError("proprio is required when proprio_emb_dim>0")
        if self.proprio_encoder is None:
            raise RuntimeError("proprio encoder is missing")
        proprio = proprio_raw.to(device=self._module_device(), dtype=self._module_dtype())
        if self._raw_proprio_slots:
            emb = F.pad(proprio, (0, self.proprio_condition_dim - self.proprio_dim))
        else:
            emb = self.proprio_encoder(proprio)
            if self.num_proprio_repeat > 1:
                emb = emb.repeat(1, 1, self.num_proprio_repeat)
        tiled = emb[:, :, None, :].expand(-1, -1, vision_tokens.shape[2], -1)
        return torch.cat([vision_tokens, tiled], dim=-1)

    def _proprio_for_steps(
        self,
        proprio_raw: torch.Tensor | None,
        steps: int,
    ) -> torch.Tensor | None:
        if proprio_raw is None:
            return None
        proprio = proprio_raw.to(device=self._module_device(), dtype=self._module_dtype())
        if proprio.ndim == 2:
            return proprio[:, None].expand(-1, int(steps), -1)
        if proprio.ndim != 3:
            raise ValueError(f"proprio must be [B,P] or [B,T,P], got {tuple(proprio.shape)}")
        if proprio.shape[1] == int(steps):
            return proprio
        if proprio.shape[1] > int(steps):
            return proprio[:, -int(steps) :]
        pad = proprio[:, :1].expand(-1, int(steps) - proprio.shape[1], -1)
        return torch.cat([pad, proprio], dim=1)

    def _raw_proprio_from_obs_tokens(
        self, obs_tokens: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decode predicted proprio tokens back to raw proprio for classifier scoring."""
        if self.proprio_condition_dim == 0:
            raise RuntimeError("raw proprio decoding requires proprio_emb_dim>0")
        if self.proprio_decoder is None:
            raise RuntimeError("proprio decoder is missing")
        if obs_tokens.shape[-1] < self.obs_token_dim:
            raise ValueError(
                f"obs token width {obs_tokens.shape[-1]} is smaller than obs_token_dim={self.obs_token_dim}"
            )
        proprio_tokens = obs_tokens[..., self.token_dim : self.obs_token_dim]
        if token_mask is None:
            proprio_emb = proprio_tokens.mean(dim=-2)
        else:
            weights = token_mask.to(device=obs_tokens.device, dtype=obs_tokens.dtype)
            if weights.shape != obs_tokens.shape[:-1]:
                raise ValueError("proprio pooling mask must match observation token axes")
            proprio_emb = (proprio_tokens * weights[..., None]).sum(dim=-2) / (
                weights.sum(dim=-1, keepdim=True).clamp_min(1)
            )
        if self._raw_proprio_slots:
            return proprio_emb[..., : self.proprio_dim]
        return self.proprio_decoder(proprio_emb)

    def _condition_tokens(
        self,
        obs_tokens: torch.Tensor,
        lang_emb: torch.Tensor | None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions is None:
            actions = lang_emb
            lang_emb = None
        if actions is None:
            raise ValueError("actions are required for WM conditioning")
        actions = self._validate_actions(actions, int(obs_tokens.shape[1]))
        parts = [obs_tokens]
        if self.lang_condition_dim > 0:
            if lang_emb is None:
                raise ValueError("lang_emb is required when lang_emb_dim>0")
            if self.lang_proj is None:
                raise RuntimeError("language projection is missing")
            lang = lang_emb.to(device=self._module_device(), dtype=self._module_dtype())
            le = self.lang_proj(lang)
            if self.num_lang_repeat > 1:
                le = le.repeat(1, self.num_lang_repeat)
            lang_tokens = le[:, None, None, :].expand(
                -1, obs_tokens.shape[1], obs_tokens.shape[2], -1
            )
            parts.append(lang_tokens)
        action_emb = self.action_proj(actions)
        if self.num_action_repeat > 1:
            action_emb = action_emb.repeat(1, 1, self.num_action_repeat)
        action_tokens = action_emb[:, :, None, :].expand(-1, -1, obs_tokens.shape[2], -1)
        parts.append(action_tokens)
        return torch.cat(parts, dim=-1)

    def _vjepa2_condition_tokens(
        self,
        obs_tokens: torch.Tensor,
        lang_emb: torch.Tensor | None,
        actions: torch.Tensor,
        proprio_raw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pack adapter inputs plus raw AC condition values without resizing weights."""

        actions = self._validate_actions(actions, int(obs_tokens.shape[1]))
        parts = [obs_tokens]
        if self.lang_condition_dim > 0:
            if lang_emb is None:
                raise ValueError("lang_emb is required when lang_emb_dim>0")
            if self.lang_proj is None:
                raise RuntimeError("language projection is missing")
            lang = lang_emb.to(device=self._module_device(), dtype=self._module_dtype())
            encoded_lang = self.lang_proj(lang)
            if self.num_lang_repeat > 1:
                encoded_lang = encoded_lang.repeat(1, self.num_lang_repeat)
            parts.append(
                encoded_lang[:, None, None, :].expand(
                    -1, obs_tokens.shape[1], obs_tokens.shape[2], -1
                )
            )

        raw_action_tokens = actions[:, :, None, :].expand(-1, -1, obs_tokens.shape[2], -1)
        parts.append(raw_action_tokens)
        if self.proprio_dim > 0:
            proprio = self._proprio_for_steps(proprio_raw, int(obs_tokens.shape[1]))
            if proprio is None:
                raise ValueError("proprio is required by the V-JEPA2-AC state adapter")
        else:
            proprio = actions.new_zeros(actions.shape[0], actions.shape[1], self._vjepa2_state_dim)
        parts.append(proprio[:, :, None, :].expand(-1, -1, obs_tokens.shape[2], -1))
        return torch.cat(parts, dim=-1)

    def observe_sequence(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Encode replay windows into per-step latent starts for imagination."""
        vision_tokens = self._normalize_raw_vision_tokens(self._obs_embedding_from_obs(batch))
        vision_tokens = self._apply_task_conditioning(vision_tokens, batch.get("task_ids"))
        obs_tokens = self._observation_tokens(vision_tokens, batch.get("proprio"))
        bsz, steps = obs_tokens.shape[:2]
        actions = self._actions_or_zeros(batch.get("actions"), bsz, steps)

        histories: list[torch.Tensor] = []
        action_histories: list[torch.Tensor] = []
        for step in range(steps):
            indices = torch.arange(
                step - self.num_hist + 1,
                step + 1,
                device=obs_tokens.device,
            ).clamp_min(0)
            histories.append(obs_tokens.index_select(1, indices))
            action_histories.append(actions.index_select(1, indices))

        latent = {
            "hidden": obs_tokens,
            "history": torch.stack(histories, dim=1),
            "actions": torch.stack(action_histories, dim=1),
            "proprio": batch.get("proprio"),
            "lang": batch.get("lang_emb"),
            "prefix_attention_mask": batch.get("prefix_attention_mask"),
        }
        return {"latent": latent}

    def encode(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        act: torch.Tensor,
        lang: torch.Tensor | None = None,
        *,
        normalize_observations: bool = True,
    ) -> torch.Tensor:
        if normalize_observations:
            obs_tokens = self._normalize_raw_vision_tokens(self._obs_embedding_from_obs(obs))
        else:
            obs_tokens = self._obs_tokens_from_obs(obs)
        proprio = obs.get("proprio") if isinstance(obs, dict) else None
        if obs_tokens.shape[-1] == self.token_dim and self.proprio_condition_dim > 0:
            obs_tokens = self._observation_tokens(
                obs_tokens,
                self._proprio_for_steps(proprio, int(obs_tokens.shape[1])),
            )
        if self.transition_type == "vjepa2_ac":
            z = self._vjepa2_condition_tokens(obs_tokens, lang, act, proprio)
            return z
        z = self._condition_tokens(obs_tokens, lang, act)
        bsz, steps, slots, dim = z.shape
        flat = z.reshape(bsz, steps * slots, dim)
        if self.pos_embedding is None:
            raise RuntimeError("original transition is missing its positional embedding")
        if flat.shape[1] > self.pos_embedding.shape[1]:
            raise ValueError(
                "ChunkAwareWorldModel WM predictor is configured for "
                f"num_hist={self.pos_context_len} frames; got {steps} frames"
            )
        flat = flat + self.pos_embedding[:, : flat.shape[1]].to(
            device=flat.device, dtype=flat.dtype
        )
        return flat.reshape(bsz, steps, slots, dim)

    def separate_emb(
        self,
        z: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        visual = z[..., : self.token_dim]
        proprio = z[..., self.token_dim : self.obs_token_dim]
        cond_emb = z[..., self.obs_token_dim :].mean(dim=2)
        return {"visual": visual, "proprio": proprio}, cond_emb

    def replace_actions_from_z(
        self,
        z: torch.Tensor,
        act: torch.Tensor,
        lang: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.transition_type == "vjepa2_ac":
            actions = self._validate_actions(act, int(z.shape[1]))
            output = z.clone()
            start = self.obs_token_dim + self._vjepa2_context_dim
            stop = start + self.action_dim
            output[..., start:stop] = actions[:, :, None, :]
            return output
        obs_tokens = z[..., : self.obs_token_dim]
        return self._condition_tokens(obs_tokens, lang, act)

    def predict(
        self,
        z: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run either the baseline transition or adapter-wrapped V-JEPA2-AC."""

        if self.transition_type == "original":
            return super().predict(z)
        if self.vjepa2_transition is None:
            raise RuntimeError("transition_type='vjepa2_ac' without a transition module")
        if z.ndim != 4 or z.shape[2] != self.token_count:
            raise ValueError(
                f"V-JEPA2-AC z must be [B,T,{self.token_count},D], got {tuple(z.shape)}"
            )
        action_start = self.obs_token_dim + self._vjepa2_context_dim
        action_stop = action_start + self.action_dim
        state_stop = action_stop + self._vjepa2_state_dim
        if z.shape[-1] != state_stop:
            raise ValueError(f"V-JEPA2-AC z width is {z.shape[-1]}, expected {state_stop}")
        tokens = z[..., : self._vjepa2_token_input_dim]
        context = (
            z[:, :, 0, self.obs_token_dim : action_start] if self._vjepa2_context_dim > 0 else None
        )
        actions = z[:, :, 0, action_start:action_stop]
        states = z[:, :, 0, action_stop:state_stop]
        prediction = self.vjepa2_transition(
            tokens,
            actions,
            states,
            context,
            token_mask=token_mask,
        )
        return torch.cat([prediction, z[..., self.obs_token_dim :]], dim=-1)

    def _history_token_mask(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        *,
        batch_size: int,
        frames: int,
    ) -> torch.Tensor | None:
        if not isinstance(latent, dict):
            return None
        mask = latent.get("prefix_attention_mask")
        if not isinstance(mask, torch.Tensor):
            return None
        if mask.ndim == 2:
            if mask.shape != (batch_size, self.token_count):
                raise ValueError(
                    "prefix_attention_mask must be "
                    f"[{batch_size},{self.token_count}], got {tuple(mask.shape)}"
                )
            return mask[:, None].expand(-1, frames, -1).to(dtype=torch.bool)
        if mask.ndim != 3 or mask.shape[0] != batch_size or mask.shape[2] != self.token_count:
            raise ValueError(
                f"prefix_attention_mask must be [B,N] or [B,T,N], got {tuple(mask.shape)}"
            )
        if mask.shape[1] < frames:
            pad = mask[:, :1].expand(-1, frames - int(mask.shape[1]), -1)
            mask = torch.cat([pad, mask], dim=1)
        return mask[:, -frames:].to(dtype=torch.bool)

    def actor_input(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Return the visual token segment consumed by VLA action actors."""
        hidden = self._latent_hidden(latent)
        return hidden[..., : self.token_dim]

    def critic_input(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Mean-pool visual tokens for critics while retaining proprio for WM losses."""
        hidden = self._latent_hidden(latent)
        tokens = self._obs_tokens_from_obs(hidden)[..., : self.token_dim]
        pooled = tokens.mean(dim=2)
        if tokens.shape[1] == 1:
            return pooled[:, 0]
        return pooled

    def _latent_lang(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor | None:
        if isinstance(latent, dict) and isinstance(latent.get("lang"), torch.Tensor):
            return latent["lang"]
        return None

    def _latent_proprio(
        self, latent: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor | None:
        if isinstance(latent, dict) and isinstance(latent.get("proprio"), torch.Tensor):
            return latent["proprio"]
        return None

    def _latent_hidden(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(latent, torch.Tensor):
            return self._obs_tokens_from_obs(latent)[:, -1]
        hidden = latent.get("hidden") if isinstance(latent, dict) else None
        if isinstance(hidden, torch.Tensor):
            return self._obs_tokens_from_obs(hidden)[:, -1]
        history = latent.get("history") if isinstance(latent, dict) else None
        if isinstance(history, torch.Tensor):
            if history.ndim == 5:
                history = history[:, -1]
            if (
                history.ndim == 4
                and history.shape[-1] == self.obs_dim
                and history.shape[-2:]
                not in {(self.token_count, self.token_dim), (self.token_count, self.obs_token_dim)}
            ):
                history = history[:, -1]
            return self._obs_tokens_from_obs(history)[:, -1]
        raise KeyError("VLA latent must contain `hidden` or `history`.")

    def _latent_history(self, latent: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(latent, dict) and isinstance(latent.get("history"), torch.Tensor):
            history = latent["history"]
            if history.ndim == 5:
                history = history[:, -1]
            elif (
                history.ndim == 4
                and history.shape[-1] == self.obs_dim
                and history.shape[-2:]
                not in {(self.token_count, self.token_dim), (self.token_count, self.obs_token_dim)}
            ):
                history = history[:, -1]
            tokens = self._obs_tokens_from_obs(history)
        else:
            tokens = self._obs_tokens_from_obs(self._latent_hidden(latent))
        if tokens.shape[1] >= self.num_hist:
            return tokens[:, -self.num_hist :]
        pad = tokens[:, :1].expand(-1, self.num_hist - tokens.shape[1], -1, -1)
        return torch.cat([pad, tokens], dim=1)

    def predict_next(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history = self._latent_history(latent)
        bsz = int(history.shape[0])
        lang = self._latent_lang(latent)
        proprio = self._latent_proprio(latent)
        proprio_history = None
        if self.transition_type == "vjepa2_ac" and proprio is not None:
            history_value = (
                latent.get("proprio_history", proprio) if isinstance(latent, dict) else proprio
            )
            if (
                self._raw_proprio_slots
                and history_value.ndim == 2
                and history.shape[-1] == self.obs_token_dim
            ):
                # Older callers may retain only observation history. Raw slots
                # still contain every historical state, so recover them exactly
                # rather than broadcasting the latest state over real frames.
                history_mask = self._history_token_mask(
                    latent, batch_size=bsz, frames=int(history.shape[1])
                )
                history_value = self._raw_proprio_from_obs_tokens(history, history_mask)
            proprio_history = self._proprio_for_steps(history_value, int(history.shape[1]))
        action = actions[:, 0] if actions.ndim == 3 else actions
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Dreamer action must be [B,{self.action_dim}], got {tuple(actions.shape)}"
            )

        action_history = self._latent_actions(latent, bsz).clone()
        action_history[:, -1] = action.to(device=action_history.device, dtype=action_history.dtype)
        model_history = history
        if (
            proprio is not None
            and model_history.shape[-1] == self.token_dim
            and self.proprio_condition_dim > 0
        ):
            model_history = self._observation_tokens(
                model_history,
                self._proprio_for_steps(proprio, int(model_history.shape[1])),
            )
        encode_input: dict[str, torch.Tensor] | torch.Tensor = model_history
        if self.transition_type == "vjepa2_ac":
            encode_input = {"obs_embedding": model_history}
            if proprio is not None:
                encode_input["proprio"] = proprio_history
        z = self.encode(
            encode_input,
            action_history,
            lang,
            normalize_observations=False,
        )
        token_mask = self._history_token_mask(
            latent,
            batch_size=bsz,
            frames=int(model_history.shape[1]),
        )
        pred_z = self.predict(z, token_mask=token_mask)
        next_hidden = pred_z[:, -1][..., : self.obs_token_dim]
        next_proprio = (
            self._raw_proprio_from_obs_tokens(
                next_hidden,
                token_mask[:, -1]
                if self.transition_type == "vjepa2_ac" and token_mask is not None
                else None,
            )
            if self.proprio_condition_dim > 0
            else None
        )

        if self.num_hist > 1:
            next_history = torch.cat(
                [model_history[:, 1:], next_hidden[:, None]],
                dim=1,
            )
            next_action_history = torch.cat(
                [
                    action_history[:, 1:],
                    action_history.new_zeros(bsz, 1, self.action_dim),
                ],
                dim=1,
            )
        else:
            next_history = next_hidden[:, None]
            next_action_history = action_history.new_zeros(bsz, 1, self.action_dim)
        out = {
            "hidden": next_hidden,
            "history": next_history,
            "actions": next_action_history,
            "lang": lang,
        }
        if token_mask is not None:
            next_mask = token_mask[:, -1]
            if self.num_hist > 1:
                out["prefix_attention_mask"] = torch.cat(
                    [token_mask[:, 1:], next_mask[:, None]],
                    dim=1,
                )
            else:
                out["prefix_attention_mask"] = next_mask[:, None]
        if next_proprio is not None:
            out["proprio"] = next_proprio
            if proprio_history is not None:
                out["proprio_history"] = torch.cat(
                    [proprio_history[:, 1:], next_proprio[:, None]], dim=1
                )
        return out

    def _predict_next_step(
        self,
        cur: dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One autoregressive step, optionally gradient-checkpointed.

        When ``grad_checkpoint`` is on (and we are building a graph), the step's
        activations are recomputed in backward instead of stored. Numerically
        identical to the plain path; ``use_reentrant=False`` preserves RNG so
        dropout matches on recompute.
        """
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return self.predict_next(cur, action)

        # Non-reentrant checkpointing supports nested tensor dictionaries.
        # Preserve every sidecar, including the shifted mask and state history.
        return checkpoint(self.predict_next, cur, action, use_reentrant=False)

    def _truncate_vjepa2_rollout_state(
        self,
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Stop gradients between AC rollout steps without changing their values.

        A chunk objective can invoke the 24-layer AC predictor dozens of times.
        Limit recurrent credit assignment to configured segments to bound memory
        and gradient growth. Per-step supervision remains active, but truncation
        does remove gradients across segment boundaries. The original transition
        never takes this path.
        """

        if not (
            self.transition_type == "vjepa2_ac"
            and self.vjepa2_truncate_rollout_gradients
            and self.training
            and torch.is_grad_enabled()
        ):
            return state
        return {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }

    def predict_next_chunk(
        self,
        latent: dict[str, torch.Tensor] | torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Advance the WM by ``chunk_size`` env-steps autoregressively.

        Args:
            latent: dict with ``history`` [B,H,N,token_dim], ``actions`` [B,H,A],
                and tokenized ``hidden`` [B,N,token_dim].
            action_chunk: [B, K, A] where K == ``self.chunk_size``.

        Returns dict with:
            ``hidden``     [B, N, token_dim]    — last predicted frame h_K.
            ``hidden_seq`` [B, K, N, token_dim] — all K predicted frames h_1..h_K.
            ``history``    [B, H, N, token_dim] — rolled history after K steps.
            ``actions``    [B, H, A]            — rolled action history, last slot zero.
        """
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_chunk must be [B,K,{self.action_dim}], got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[1] != self.chunk_size:
            raise ValueError(
                f"action_chunk time dim {action_chunk.shape[1]} != chunk_size {self.chunk_size}"
            )
        K = self.chunk_size
        bsz = int(action_chunk.shape[0])

        history = self._latent_history(latent)
        action_history = self._latent_actions(latent, bsz).clone()
        lang = self._latent_lang(latent)

        device = self._module_device()
        dtype = self._module_dtype()
        action_chunk_v = action_chunk.to(device=device, dtype=dtype)
        cur: dict[str, torch.Tensor] = {
            "hidden": history[:, -1],
            "history": history,
            "actions": action_history,
            "lang": lang,
        }
        if isinstance(latent, dict) and isinstance(
            latent.get("prefix_attention_mask"), torch.Tensor
        ):
            cur["prefix_attention_mask"] = latent["prefix_attention_mask"]
        if isinstance(latent, dict) and isinstance(latent.get("proprio"), torch.Tensor):
            cur["proprio"] = latent["proprio"].to(device=device, dtype=dtype)
        if isinstance(latent, dict) and isinstance(latent.get("proprio_history"), torch.Tensor):
            cur["proprio_history"] = latent["proprio_history"].to(device=device, dtype=dtype)
        # A previous chunk may have returned a live graph.  Detach it before
        # beginning this chunk, then truncate again between its individual
        # autoregressive steps below.
        cur = self._truncate_vjepa2_rollout_state(cur)
        preds: list[torch.Tensor] = []
        proprio_preds: list[torch.Tensor] = []
        for step in range(K):
            cur = self._predict_next_step(cur, action_chunk_v[:, step])
            preds.append(cur["hidden"])
            if isinstance(cur.get("proprio"), torch.Tensor):
                proprio_preds.append(cur["proprio"])
            if step + 1 < K and (step + 1) % self.vjepa2_rollout_bptt_steps == 0:
                cur = self._truncate_vjepa2_rollout_state(cur)
        hidden_seq = torch.stack(preds, dim=1)

        out = {
            "hidden": cur["hidden"],
            "hidden_seq": hidden_seq,
            "history": cur["history"],
            "actions": cur["actions"],
            "lang": cur.get("lang"),
        }
        if isinstance(cur.get("prefix_attention_mask"), torch.Tensor):
            out["prefix_attention_mask"] = cur["prefix_attention_mask"]
        if proprio_preds:
            out["proprio"] = cur["proprio"]
            out["proprio_seq"] = torch.stack(proprio_preds, dim=1)
        if isinstance(cur.get("proprio_history"), torch.Tensor):
            out["proprio_history"] = cur["proprio_history"]
        return out

    def initial_imagination_state(
        self,
        hidden: torch.Tensor,
        *,
        lang_emb: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        prefix_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build the canonical rolling state used by chunked WM inference."""

        visual = self._normalize_raw_vision_tokens(hidden)
        if visual.shape[1] >= self.num_hist:
            history = visual[:, -self.num_hist :]
        else:
            pad = visual[:, :1].expand(-1, self.num_hist - visual.shape[1], -1, -1)
            history = torch.cat([pad, visual], dim=1)
        if proprio is not None and self.proprio_condition_dim > 0:
            history = self._observation_tokens(
                history,
                self._proprio_for_steps(proprio, self.num_hist),
            )
        state = {
            "hidden": history[:, -1],
            "history": history,
            "actions": torch.zeros(
                history.shape[0],
                self.num_hist,
                self.action_dim,
                device=history.device,
                dtype=history.dtype,
            ),
        }
        if lang_emb is not None:
            state["lang"] = lang_emb.to(device=history.device, dtype=history.dtype)
        if proprio is not None:
            state["proprio"] = proprio.to(device=history.device, dtype=history.dtype)
            if self.transition_type == "vjepa2_ac":
                state["proprio_history"] = self._proprio_for_steps(proprio, self.num_hist)
                state["proprio"] = state["proprio_history"][:, -1]
        if prefix_attention_mask is not None:
            state["prefix_attention_mask"] = prefix_attention_mask.to(device=history.device)
        return state

    # ------------------------------------------------------------------ #
    # Chunk-objective training loss                                      #
    # ------------------------------------------------------------------ #
    def chunk_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Train end-to-end as a K-step chunk predictor.

        Per sampled window of length T >= H + K:
          - first H frames                  → h_history
          - actions[:, H-1 : H-1+K]         → chunk a_0..a_{K-1}
          - obs[:, H : H + K]               → targets h_1..h_K

        Reward / success-return heads (if enabled) are trained on TARGET
        hiddens, matching parent's convention.
        """
        obs = batch["obs_embedding"]
        actions = batch.get("current_actions")
        if not isinstance(actions, torch.Tensor):
            actions = batch["actions"]
        H = self.num_hist
        K = self.chunk_size
        vision_tokens = self._normalize_raw_vision_tokens(obs)
        vision_tokens = self._apply_task_conditioning(vision_tokens, batch.get("task_ids"))
        obs_tokens = self._observation_tokens(vision_tokens, batch.get("proprio"))
        lang_emb = batch.get("lang_emb")
        T = int(obs_tokens.shape[1])
        if T < H + K:
            raise ValueError(f"chunk_loss requires T >= H+K = {H + K}, got T={T}")
        if actions.shape[1] < H - 1 + K:
            raise ValueError(
                "action sequence length "
                f"{actions.shape[1]} too short for chunk loss "
                f"(need >= {H - 1 + K})"
            )

        actions = self._validate_actions(actions, int(actions.shape[1]))
        bsz = int(obs_tokens.shape[0])

        history = obs_tokens[:, :H]
        chunk_actions = actions[:, H - 1 : H - 1 + K]
        hidden_target = obs_tokens[:, H : H + K].detach()
        prefix_attention_mask = batch.get("prefix_attention_mask")
        target_token_mask: torch.Tensor | None = None
        history_token_mask: torch.Tensor | None = None
        if isinstance(prefix_attention_mask, torch.Tensor):
            if prefix_attention_mask.shape != (bsz, T, self.token_count):
                raise ValueError(
                    "prefix_attention_mask must match [B,T,N] = "
                    f"[{bsz},{T},{self.token_count}], got "
                    f"{tuple(prefix_attention_mask.shape)}"
                )
            prefix_attention_mask = prefix_attention_mask.to(dtype=torch.bool)
            history_token_mask = prefix_attention_mask[:, :H]
            target_token_mask = prefix_attention_mask[:, H : H + K]

        action_history = torch.zeros(
            bsz,
            H,
            self.action_dim,
            device=self._module_device(),
            dtype=self._module_dtype(),
        )
        if H > 1:
            action_history[:, : H - 1] = actions[:, : H - 1]
        action_history[:, -1] = chunk_actions[:, 0]

        latent = {
            "hidden": history[:, -1],
            "history": history,
            "actions": action_history,
            "lang": lang_emb,
        }
        if history_token_mask is not None:
            latent["prefix_attention_mask"] = history_token_mask
        if isinstance(batch.get("proprio"), torch.Tensor):
            latent["proprio"] = batch["proprio"][:, H - 1]
            if self.transition_type == "vjepa2_ac":
                latent["proprio_history"] = batch["proprio"][:, :H]
        out = self.predict_next_chunk(latent, chunk_actions)
        hidden_pred = out["hidden_seq"]
        self._last_hidden_target_width = int(hidden_target.shape[-1])

        loss, hidden_mse, hidden_cosine = self._hidden_loss_terms(
            hidden_pred,
            hidden_target,
            token_mask=target_token_mask,
        )
        ac_loss_metrics: dict[str, torch.Tensor] = {}
        if self.transition_type == "vjepa2_ac":
            # Give the prediction from a real context an explicit objective;
            # otherwise it is diluted among K recursively generated contexts.
            if self.vjepa2_one_step_loss_scale > 0:
                one_step_loss, _, _ = self._hidden_loss_terms(
                    hidden_pred[:, :1],
                    hidden_target[:, :1],
                    token_mask=None if target_token_mask is None else target_token_mask[:, :1],
                )
                loss = loss + self.vjepa2_one_step_loss_scale * one_step_loss
                ac_loss_metrics["one_step_prediction_loss"] = one_step_loss.detach()
            if self.vjepa2_temporal_difference_loss_scale > 0:
                predicted_visual = hidden_pred[..., : self.token_dim].float()
                previous = torch.cat(
                    [vision_tokens[:, H - 1 : H].detach().float(), predicted_visual[:, :-1]], dim=1
                )
                true_delta = (
                    (vision_tokens[:, H : H + K] - vision_tokens[:, H - 1 : H + K - 1])
                    .detach()
                    .float()
                )
                delta_mask = (
                    None
                    if target_token_mask is None
                    else (target_token_mask & prefix_attention_mask[:, H - 1 : H + K - 1])
                )
                _, delta_mse, _ = self._hidden_loss_terms(
                    predicted_visual - previous, true_delta, token_mask=delta_mask
                )
                loss = loss + self.vjepa2_temporal_difference_loss_scale * delta_mse
                ac_loss_metrics["temporal_difference_loss"] = delta_mse.detach()
        # Comparable visual diagnostics deliberately exclude proprio/language
        # conditioning and never contribute to the optimized objective.  A
        # "model step" is one replay transition for Chunk-WM (environment
        # t -> t+1); multi-step chunk and closed-loop quality stay separate.
        with torch.no_grad():
            visual_pred = hidden_pred.detach()[..., : self.token_dim].float()
            visual_target = hidden_target.detach()[..., : self.token_dim].float()
            previous_pred = torch.cat(
                [vision_tokens[:, H - 1 : H].float(), visual_pred[:, :-1]], dim=1
            )
            previous_target = vision_tokens[:, H - 1 : H + K - 1].float()
            pred_delta = visual_pred - previous_pred
            target_delta = visual_target - previous_target
            motion_mask = torch.ones_like(visual_pred[..., 0], dtype=torch.bool)
            if target_token_mask is not None:
                motion_mask = target_token_mask & prefix_attention_mask[:, H - 1 : H + K - 1]
            motion_weights = motion_mask[..., None].to(dtype=torch.float32)
            motion_count = (motion_weights.sum() * self.token_dim).clamp_min(1)
            predicted_motion_rms = (
                (pred_delta.square() * motion_weights).sum() / motion_count
            ).sqrt()
            target_motion_rms = (
                (target_delta.square() * motion_weights).sum() / motion_count
            ).sqrt()
            visual_delta_mse = (
                (pred_delta - target_delta).square() * motion_weights
            ).sum() / motion_count
            del (
                previous_pred,
                previous_target,
                pred_delta,
                target_delta,
                motion_weights,
                motion_mask,
            )
            one_step_cosine = F.cosine_similarity(
                visual_pred[:, 0],
                visual_target[:, 0],
                dim=-1,
            )
            chunk_cosine = F.cosine_similarity(
                visual_pred,
                visual_target,
                dim=-1,
            )
            persistence_cosine = F.cosine_similarity(
                vision_tokens[:, H - 1].detach().float(),
                vision_tokens[:, H].detach().float(),
                dim=-1,
            )
            if target_token_mask is None:
                one_step_cosine_similarity = one_step_cosine.mean()
                chunk_cosine_similarity = chunk_cosine.mean()
                persistence_cosine_similarity = persistence_cosine.mean()
            else:
                one_mask = target_token_mask[:, 0].to(dtype=one_step_cosine.dtype)
                chunk_mask = target_token_mask.to(dtype=chunk_cosine.dtype)
                persistence_mask = prefix_attention_mask[:, H].to(dtype=persistence_cosine.dtype)
                one_step_cosine_similarity = (
                    one_step_cosine * one_mask
                ).sum() / one_mask.sum().clamp_min(1)
                chunk_cosine_similarity = (
                    chunk_cosine * chunk_mask
                ).sum() / chunk_mask.sum().clamp_min(1)
                persistence_cosine_similarity = (
                    persistence_cosine * persistence_mask
                ).sum() / persistence_mask.sum().clamp_min(1)
        proprio_out: dict[str, torch.Tensor] = {}
        proprio_loss_scale = (
            self.vjepa2_proprio_loss_scale
            if self._raw_proprio_slots
            else self.proprio_reconstruction_loss_scale
        )
        if self.proprio_condition_dim > 0 and isinstance(out.get("proprio_seq"), torch.Tensor):
            proprio_target = batch.get("proprio")
            if not isinstance(proprio_target, torch.Tensor):
                if proprio_loss_scale > 0:
                    raise KeyError(
                        "proprio_reconstruction_loss_scale > 0 requires batch['proprio']"
                    )
            else:
                target = proprio_target[:, H : H + K].to(
                    device=out["proprio_seq"].device,
                    dtype=out["proprio_seq"].dtype,
                )
                proprio_loss = F.mse_loss(out["proprio_seq"], target)
                if proprio_loss_scale > 0:
                    loss = loss + proprio_loss_scale * proprio_loss
                proprio_out = {
                    "proprio_reconstruction_loss": proprio_loss.detach(),
                    "proprio_pred_norm": out["proprio_seq"].detach().float().norm(dim=-1).mean(),
                    "proprio_target_norm": target.detach().float().norm(dim=-1).mean(),
                }

        # --- Close-loop multi-chunk rollout loss (anti-drift) ---
        # Continue rolling forward N-1 MORE chunks from chunk 0's output, feeding
        # the predicted hidden as next chunk's history (no teacher forcing).
        # Actions are still REAL demo chunk actions throughout — this loss only
        # cures WM-internal drift, not actor sensitivity to drift.
        rollout_out: dict[str, torch.Tensor] = {}
        visual_loss_predictions = [hidden_pred]
        if self.chunk_rollout_chunks > 1 and self.chunk_rollout_loss_scale > 0.0:
            N = self.chunk_rollout_chunks
            if T < H + N * K:
                raise ValueError(
                    f"chunk_rollout_chunks={N} requires T >= H + N*K = {H + N * K}, got T={T}"
                )
            cur_latent = {
                "hidden": out["hidden"],
                "history": out["history"],
                "actions": out["actions"],
                "lang": lang_emb,
            }
            if isinstance(out.get("prefix_attention_mask"), torch.Tensor):
                cur_latent["prefix_attention_mask"] = out["prefix_attention_mask"]
            if isinstance(out.get("proprio"), torch.Tensor):
                cur_latent["proprio"] = out["proprio"]
            if isinstance(out.get("proprio_history"), torch.Tensor):
                cur_latent["proprio_history"] = out["proprio_history"]
            rollout_preds: list[torch.Tensor] = []
            rollout_proprio_preds: list[torch.Tensor] = []
            for c in range(1, N):
                cca = actions[:, H - 1 + c * K : H - 1 + (c + 1) * K]
                out_c = self.predict_next_chunk(cur_latent, cca)
                rollout_preds.append(out_c["hidden_seq"])
                if isinstance(out_c.get("proprio_seq"), torch.Tensor):
                    rollout_proprio_preds.append(out_c["proprio_seq"])
                cur_latent = {
                    "hidden": out_c["hidden"],
                    "history": out_c["history"],
                    "actions": out_c["actions"],
                    "lang": lang_emb,
                }
                if isinstance(out_c.get("prefix_attention_mask"), torch.Tensor):
                    cur_latent["prefix_attention_mask"] = out_c["prefix_attention_mask"]
                if isinstance(out_c.get("proprio"), torch.Tensor):
                    cur_latent["proprio"] = out_c["proprio"]
                if isinstance(out_c.get("proprio_history"), torch.Tensor):
                    cur_latent["proprio_history"] = out_c["proprio_history"]
            rollout_pred = torch.cat(rollout_preds, dim=1)
            visual_loss_predictions.append(rollout_pred)
            rollout_target = obs_tokens[:, H + K : H + N * K].detach()
            rollout_token_mask = (
                None
                if prefix_attention_mask is None
                else prefix_attention_mask[:, H + K : H + N * K]
            )
            rollout_loss_total, rollout_mse, rollout_cosine = self._hidden_loss_terms(
                rollout_pred,
                rollout_target,
                token_mask=rollout_token_mask,
            )
            if self._raw_proprio_slots and rollout_proprio_preds:
                rollout_proprio = torch.cat(rollout_proprio_preds, dim=1)
                raw_target = batch["proprio"][:, H + K : H + N * K].to(rollout_proprio)
                state_loss = F.mse_loss(rollout_proprio.float(), raw_target.float())
                rollout_loss_total = rollout_loss_total + proprio_loss_scale * state_loss
                proprio_out["rollout_proprio_reconstruction_loss"] = state_loss.detach()
            with torch.no_grad():
                rollout_cosine_values = F.cosine_similarity(
                    rollout_pred.detach()[..., : self.token_dim].float(),
                    rollout_target.detach()[..., : self.token_dim].float(),
                    dim=-1,
                )
                if rollout_token_mask is None:
                    rollout_cosine_similarity = rollout_cosine_values.mean()
                else:
                    rollout_mask = rollout_token_mask.to(dtype=rollout_cosine_values.dtype)
                    rollout_cosine_similarity = (
                        rollout_cosine_values * rollout_mask
                    ).sum() / rollout_mask.sum().clamp_min(1)
            loss = loss + self.chunk_rollout_loss_scale * rollout_loss_total
            rollout_out = {
                "rollout_loss": rollout_loss_total.detach(),
                "rollout_mse": rollout_mse.detach(),
                "rollout_cosine_loss": rollout_cosine.detach(),
                "rollout_cosine_similarity": rollout_cosine_similarity,
                "rollout_chunks": loss.new_tensor(float(N)),
            }

        if self.decoded_visual_loss is not None:
            visual_prediction = torch.cat(visual_loss_predictions, dim=1)[..., : self.token_dim]
            stop = H + visual_prediction.shape[1]
            decoded_terms = self.decoded_visual_loss(
                visual_prediction,
                vision_tokens[:, H:stop].detach(),
                vision_tokens[:, H - 1 : H].detach(),
                token_mask=None
                if prefix_attention_mask is None
                else prefix_attention_mask[:, H:stop],
                anchor_mask=None
                if prefix_attention_mask is None
                else prefix_attention_mask[:, H - 1 : H],
            )
            loss = loss + decoded_terms["_loss"]
            ac_loss_metrics.update(
                {key: value.detach() for key, value in decoded_terms.items() if key != "_loss"}
            )

        reward_out: dict[str, torch.Tensor] = {}
        if self.reward_loss_scale > 0.0:
            rewards = batch.get("rewards")
            if rewards is None:
                raise KeyError("reward_loss_scale > 0 requires batch['rewards']")
            rewards_chunk = self._slice_per_frame_signal(rewards, T)
            reward_out = self._reward_loss_terms(hidden_target, rewards_chunk)
            loss = loss + self.reward_loss_scale * reward_out["reward_loss"]

        success_return_out: dict[str, torch.Tensor] = {}
        if self.success_return_loss_scale > 0.0:
            success_to_go = (
                batch.get("success_to_go")
                if batch.get("success_to_go") is not None
                else batch.get("return_to_go", batch.get("return_targets"))
            )
            if success_to_go is None:
                raise KeyError("success_return_loss_scale > 0 requires batch['success_to_go']")
            success_chunk = self._slice_per_frame_signal(success_to_go, T)
            success_return_out = self._success_return_loss_terms(hidden_target, success_chunk)
            loss = loss + self.success_return_loss_scale * success_return_out["success_return_loss"]

        zero = loss.new_zeros(())
        out_dict: dict[str, torch.Tensor] = {
            **ac_loss_metrics,
            "_loss": loss,
            "loss": loss.detach(),
            "next_latent_loss": hidden_mse.detach(),
            "next_latent_mse": hidden_mse.detach(),
            "next_latent_cosine_loss": hidden_cosine.detach(),
            "hidden_loss": hidden_mse.detach(),
            "hidden_mse": hidden_mse.detach(),
            "visual_predicted_motion_rms": predicted_motion_rms,
            "visual_target_motion_rms": target_motion_rms,
            "visual_motion_ratio": predicted_motion_rms / target_motion_rms.clamp_min(1.0e-8),
            "visual_delta_mse": visual_delta_mse,
            "hidden_cosine_loss": hidden_cosine.detach(),
            "one_step_cosine_similarity": one_step_cosine_similarity,
            "persistence_cosine_similarity": persistence_cosine_similarity,
            "chunk_cosine_similarity": chunk_cosine_similarity,
            "hidden_pred_norm": hidden_pred.detach().float().norm(dim=-1).mean(),
            "hidden_target_norm": hidden_target.detach().float().norm(dim=-1).mean(),
            "chunk_size": loss.new_tensor(float(self.chunk_size)),
            "rec_loss": zero.detach(),
            "dyn_loss": zero.detach(),
            "rep_loss": zero.detach(),
            "image_mse": zero.detach(),
            "image_psnr": zero.detach(),
        }
        if rollout_out:
            out_dict.update(rollout_out)
        if proprio_out:
            out_dict.update(proprio_out)
        if reward_out:
            out_dict.update(
                {
                    "reward_loss": reward_out["reward_loss"].detach(),
                    "reward_pred_mean": reward_out["reward_pred_mean"],
                    "reward_target_mean": reward_out["reward_target_mean"],
                }
            )
            if "reward_binary_acc" in reward_out:
                out_dict["reward_binary_acc"] = reward_out["reward_binary_acc"]
            if "reward_mae" in reward_out:
                out_dict["reward_mae"] = reward_out["reward_mae"]
        if success_return_out:
            out_dict.update(
                {
                    "success_return_loss": success_return_out["success_return_loss"].detach(),
                    "success_return_pred_mean": success_return_out["success_return_pred_mean"],
                    "success_return_target_mean": success_return_out["success_return_target_mean"],
                    "success_return_mse": success_return_out["success_return_mse"],
                }
            )
        return out_dict

    def reward_logits(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        self._require_reward_head()
        tokens = self._obs_tokens_from_obs(obs_embedding)
        pooled = self.reward_norm(tokens).mean(dim=2)
        return self.reward_head(pooled).squeeze(-1)

    def success_return_logits(self, obs_embedding: torch.Tensor) -> torch.Tensor:
        self._require_success_return_head()
        tokens = self._obs_tokens_from_obs(obs_embedding)
        pooled = self.success_return_norm(tokens).mean(dim=2)
        return self.success_return_head(pooled).squeeze(-1)

    def _slice_per_frame_signal(self, signal: torch.Tensor, T: int) -> torch.Tensor:
        """Slice a per-frame signal (rewards / success_to_go) to the K target frames h_1..h_K.

        Accepts [B], [B,T], [B,T,1], [B,K], or [B,K+1] layouts. Returns [B,K].
        """
        H = self.num_hist
        K = self.chunk_size
        if signal.ndim == 1:
            signal = signal[:, None]
        if signal.ndim == 3 and signal.shape[-1] == 1:
            signal = signal.squeeze(-1)
        if signal.ndim != 2:
            raise ValueError(
                f"per-frame signal must be [B,T] or [B,T,1], got {tuple(signal.shape)}"
            )
        if signal.shape[1] == T:
            return signal[:, H : H + K]
        if signal.shape[1] == K:
            return signal
        if signal.shape[1] == K + 1:
            return signal[:, 1:]
        raise ValueError(
            f"per-frame signal length {signal.shape[1]} not aligned with T={T}, K={K}, K+1={K + 1}"
        )

    # ------------------------------------------------------------------ #
    # Routing                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _latent_with_batch_sidecars(
        latent: dict[str, torch.Tensor] | torch.Tensor,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if (
            "lang_emb" not in batch
            and "proprio" not in batch
            and "prefix_attention_mask" not in batch
        ):
            return latent
        if isinstance(latent, dict):
            enriched = dict(latent)
        else:
            enriched = {"hidden": latent}
        if "lang_emb" in batch and "lang" not in enriched:
            enriched["lang"] = batch["lang_emb"]
        if "proprio" in batch and "proprio" not in enriched:
            enriched["proprio"] = batch["proprio"]
        if "prefix_attention_mask" in batch and "prefix_attention_mask" not in enriched:
            enriched["prefix_attention_mask"] = batch["prefix_attention_mask"]
        return enriched

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return self.chunk_loss(batch)

    def forward(self, batch: dict[str, Any]) -> Any:  # type: ignore[override]
        if isinstance(batch, dict) and batch.get("mode") == "predict_next_chunk":
            latent = self._latent_with_batch_sidecars(batch["latent"], batch)
            return self.predict_next_chunk(latent, batch["actions"])
        if isinstance(batch, dict) and batch.get("mode") == "predict_next":
            latent = self._latent_with_batch_sidecars(batch["latent"], batch)
            return self.predict_next(latent, batch["actions"])
        if isinstance(batch, dict) and batch.get("mode") == "classifier_input":
            return self.actor_input(batch["latent"])
        if isinstance(batch, dict) and batch.get("mode") == "chunk_loss":
            return self.chunk_loss(batch)
        return super().forward(batch)
