"""DINO-WM predictor reproduced over persisted OpenVLA-OFT tokens.

The predictor, conditioning layout, shifted embedding loss, and rollout are adapted
from the MIT-licensed DINO-WM implementation at
``Related_Work/worldmodel/dino_wm/models``. DreamerVLA replaces only DINO's frozen
visual encoder: ``visual`` is already the external ``[B,T,N,D]`` token sidecar.
See ``licenses/DINO_WM_LICENSE`` for the retained upstream license.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def generate_dino_causal_mask(
    num_patches: int,
    num_frames: int,
) -> torch.Tensor:
    """Return DINO-WM's frame-causal, within-frame-bidirectional mask."""

    patches = int(num_patches)
    frames = int(num_frames)
    if patches < 1 or frames < 1:
        raise ValueError("num_patches and num_frames must be positive")
    frame_mask = torch.tril(torch.ones(frames, frames, dtype=torch.float32))
    mask = frame_mask.repeat_interleave(patches, dim=0).repeat_interleave(
        patches, dim=1
    )
    return mask.unsqueeze(0).unsqueeze(0)


class DinoTokenEmbedding(nn.Module):
    """DINO-WM's kernel-one Conv1d embedding for action or proprio sequences."""

    def __init__(
        self,
        num_frames: int = 1,
        tubelet_size: int = 1,
        in_chans: int = 8,
        emb_dim: int = 384,
        use_3d_pos: bool = False,
    ) -> None:
        super().__init__()
        if int(tubelet_size) != 1:
            raise ValueError("DINO token reproduction requires tubelet_size=1")
        if bool(use_3d_pos):
            raise ValueError("DINO token reproduction requires use_3d_pos=false")
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.in_chans = int(in_chans)
        self.emb_dim = int(emb_dim)
        self.patch_embed = nn.Conv1d(
            self.in_chans,
            self.emb_dim,
            kernel_size=self.tubelet_size,
            stride=self.tubelet_size,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Embed ``[B,T,D]`` values as ``[B,T,E]``."""

        if values.ndim != 3 or int(values.shape[-1]) != self.in_chans:
            raise ValueError(
                f"embedding input must be [B,T,{self.in_chans}], "
                f"got {tuple(values.shape)}"
            )
        values = values.to(dtype=self.patch_embed.weight.dtype)
        return self.patch_embed(values.permute(0, 2, 1)).permute(0, 2, 1)


class _DinoFeedForward(nn.Module):
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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class _DinoAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_patches: int,
        num_frames: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = int(dim_head) * int(heads)
        project_out = not (int(heads) == 1 and int(dim_head) == int(dim))
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.scale = self.dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        # Upstream calls ``.to('cuda')`` in __init__. A non-persistent buffer has
        # identical math and follows normal CPU/DDP/device movement.
        self.register_buffer(
            "bias",
            generate_dino_causal_mask(num_patches, num_frames),
            persistent=False,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = values.shape
        if sequence > int(self.bias.shape[-1]):
            raise ValueError(
                f"predictor sequence {sequence} exceeds mask {self.bias.shape[-1]}"
            )
        values = self.norm(values)
        qkv = self.to_qkv(values).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, sequence, self.heads, self.dim_head).permute(
                0, 2, 1, 3
            )

        query, key, value = (split_heads(tensor) for tensor in qkv)
        dots = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(
            self.bias[:, :, :sequence, :sequence] == 0,
            float("-inf"),
        )
        attention = self.dropout(self.attend(dots))
        output = torch.matmul(attention, value)
        output = output.permute(0, 2, 1, 3).reshape(batch, sequence, -1)
        return self.to_out(output)


class _DinoTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        *,
        num_patches: int,
        num_frames: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _DinoAttention(
                            dim,
                            num_patches=num_patches,
                            num_frames=num_frames,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        _DinoFeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            values = attention(values) + values
            values = feed_forward(values) + values
        return self.norm(values)


class DinoTokenViTPredictor(nn.Module):
    """DINO-WM ViT predictor with its original learned positional embedding."""

    def __init__(
        self,
        *,
        num_patches: int,
        num_frames: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "mean",
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be 'cls' or 'mean'")
        self.num_patches = int(num_patches)
        self.num_frames = int(num_frames)
        self.dim = int(dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_frames * self.num_patches, self.dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _DinoTransformer(
            self.dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            num_patches=self.num_patches,
            num_frames=self.num_frames,
            dropout=dropout,
        )
        self.pool = pool

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predict an embedding at every input patch position."""

        sequence = int(values.shape[1])
        if sequence > int(self.pos_embedding.shape[1]):
            raise ValueError(
                f"input sequence {sequence} exceeds positional capacity "
                f"{self.pos_embedding.shape[1]}"
            )
        values = values + self.pos_embedding[:, :sequence]
        return self.transformer(self.dropout(values))


class DinoTokenWorldModel(nn.Module):
    """DINO-WM's token dynamics with DreamerVLA hidden tokens as visual input."""

    def __init__(
        self,
        *,
        obs_dim: int | None = None,
        token_count: int = 256,
        token_dim: int = 4096,
        action_dim: int = 7,
        proprio_dim: int = 8,
        action_emb_dim: int = 10,
        proprio_emb_dim: int = 10,
        num_action_repeat: int = 1,
        num_proprio_repeat: int = 1,
        num_hist: int = 3,
        num_pred: int = 1,
        concat_dim: int = 1,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        return_predictions: bool = False,
        latent_stage: str | None = None,
        latent_source: str = "OpenVLA-OFT hidden_token [256,4096]",
    ) -> None:
        super().__init__()
        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.obs_dim = int(obs_dim) if obs_dim is not None else self.token_count * self.token_dim
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_emb_dim = int(action_emb_dim)
        self.proprio_emb_dim = int(proprio_emb_dim)
        self.num_action_repeat = int(num_action_repeat)
        self.num_proprio_repeat = int(num_proprio_repeat)
        self.action_condition_dim = self.action_emb_dim * self.num_action_repeat
        self.proprio_condition_dim = self.proprio_emb_dim * self.num_proprio_repeat
        self.model_dim = (
            self.token_dim + self.proprio_condition_dim + self.action_condition_dim
        )
        self.num_hist = int(num_hist)
        self.num_pred = int(num_pred)
        self.concat_dim = int(concat_dim)
        self.return_predictions = bool(return_predictions)
        self.latent_stage = latent_stage
        self.latent_source = str(latent_source)
        if self.obs_dim != self.token_count * self.token_dim:
            raise ValueError(
                "token_count * token_dim must equal obs_dim: "
                f"{self.token_count} * {self.token_dim} != {self.obs_dim}"
            )
        if self.num_hist < 1:
            raise ValueError("num_hist must be positive")
        if self.num_pred != 1:
            raise ValueError("DINO-WM reproduction supports num_pred=1 only")
        if self.concat_dim != 1:
            raise ValueError("DINO token reproduction requires concat_dim=1")
        for name, value in (
            ("token_count", self.token_count),
            ("token_dim", self.token_dim),
            ("action_dim", self.action_dim),
            ("proprio_dim", self.proprio_dim),
            ("action_emb_dim", self.action_emb_dim),
            ("proprio_emb_dim", self.proprio_emb_dim),
            ("num_action_repeat", self.num_action_repeat),
            ("num_proprio_repeat", self.num_proprio_repeat),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")

        # Upstream DINO-WM consumes ``x_norm_patchtokens`` from the frozen
        # DINOv2 encoder. Persisted OpenVLA projector tokens replace that
        # encoder here, so its normalized-output boundary is part of this token
        # adapter rather than an optional training recipe.
        self.token_norm: nn.Module = nn.LayerNorm(
            self.token_dim,
            eps=1.0e-6,
            elementwise_affine=False,
        )
        self.proprio_encoder = DinoTokenEmbedding(
            in_chans=self.proprio_dim,
            emb_dim=self.proprio_emb_dim,
        )
        self.action_encoder = DinoTokenEmbedding(
            in_chans=self.action_dim,
            emb_dim=self.action_emb_dim,
        )
        self.predictor = DinoTokenViTPredictor(
            num_patches=self.token_count,
            num_frames=self.num_hist,
            dim=self.model_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            pool="mean",
        )
        self.emb_criterion = nn.MSELoss()

    def encode_act(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode action vectors with DINO-WM's Conv1d embedding."""

        return self.action_encoder(actions)

    def encode_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """Encode proprio vectors with DINO-WM's Conv1d embedding."""

        return self.proprio_encoder(proprio)

    def encode_obs(self, observations: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Use DreamerVLA tokens directly in place of DINO's visual encoder."""

        visual = observations.get("visual")
        proprio = observations.get("proprio")
        if not isinstance(visual, torch.Tensor):
            raise KeyError("observations must contain Tensor key 'visual'")
        if not isinstance(proprio, torch.Tensor):
            raise KeyError("observations must contain Tensor key 'proprio'")
        if visual.ndim != 4 or tuple(visual.shape[-2:]) != (
            self.token_count,
            self.token_dim,
        ):
            raise ValueError(
                "visual tokens must be "
                f"[B,T,{self.token_count},{self.token_dim}], got {tuple(visual.shape)}"
            )
        visual = self.token_norm(
            visual.to(dtype=self.predictor.pos_embedding.dtype)
        )
        proprio_emb = self.encode_proprio(proprio)
        if proprio_emb.shape[:2] != visual.shape[:2]:
            raise ValueError("visual and proprio batch/time dimensions must match")
        return {"visual": visual, "proprio": proprio_emb}

    def encode(
        self,
        observations: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate visual, repeated proprio, and repeated action dimensions."""

        encoded = self.encode_obs(observations)
        action_emb = self.encode_act(actions)
        visual = encoded["visual"]
        if action_emb.shape[:2] != visual.shape[:2]:
            raise ValueError("visual and action batch/time dimensions must match")
        patches = int(visual.shape[2])
        proprio_tiled = encoded["proprio"].unsqueeze(2).expand(-1, -1, patches, -1)
        proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
        action_tiled = action_emb.unsqueeze(2).expand(-1, -1, patches, -1)
        action_repeated = action_tiled.repeat(1, 1, 1, self.num_action_repeat)
        return torch.cat([visual, proprio_repeated, action_repeated], dim=-1)

    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        """Run DINO-WM's causal predictor in embedding space."""

        if latent.ndim != 4 or int(latent.shape[2]) != self.token_count:
            raise ValueError(
                f"latent must be [B,T,{self.token_count},{self.model_dim}], "
                f"got {tuple(latent.shape)}"
            )
        batch, frames, patches, width = latent.shape
        if int(width) != self.model_dim:
            raise ValueError(f"latent width must be {self.model_dim}, got {width}")
        predicted = self.predictor(latent.reshape(batch, frames * patches, width))
        return predicted.reshape(batch, frames, patches, width)

    def separate_emb(
        self,
        latent: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Separate DINO's concatenated visual, proprio, and action embeddings."""

        visual_end = self.token_dim
        proprio_end = visual_end + self.proprio_condition_dim
        visual = latent[..., :visual_end]
        proprio = latent[..., visual_end:proprio_end]
        actions = latent[..., proprio_end:]
        proprio = proprio[:, :, 0, : self.proprio_emb_dim]
        actions = actions[:, :, 0, : self.action_emb_dim]
        return {"visual": visual, "proprio": proprio}, actions

    def replace_actions_from_z(
        self,
        latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Overwrite predicted action dimensions with the commanded future action."""

        action_emb = self.encode_act(actions)
        patches = int(latent.shape[2])
        action_tiled = action_emb.unsqueeze(2).expand(-1, -1, patches, -1)
        action_repeated = action_tiled.repeat(1, 1, 1, self.num_action_repeat)
        latent = latent.clone()
        latent[..., -self.action_condition_dim :] = action_repeated
        return latent

    def _loss_from_batch(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        visual = batch.get("obs_embedding")
        proprio = batch.get("proprio")
        actions = batch.get("current_actions")
        if actions is None:
            actions = batch.get("actions")
        if actions is None:
            actions = batch.get("action")
        if not isinstance(visual, torch.Tensor):
            raise KeyError("DinoTokenWorldModel requires Tensor 'obs_embedding'")
        if not isinstance(proprio, torch.Tensor):
            raise KeyError("DinoTokenWorldModel requires Tensor 'proprio'")
        if not isinstance(actions, torch.Tensor):
            raise KeyError(
                "DinoTokenWorldModel requires 'current_actions', 'actions', or 'action'"
            )
        expected_frames = self.num_hist + self.num_pred
        if int(visual.shape[1]) != expected_frames:
            raise ValueError(
                "DINO shifted training requires sequence length num_hist + num_pred "
                f"({expected_frames}), got {visual.shape[1]}"
            )

        latent = self.encode({"visual": visual, "proprio": proprio}, actions)
        source = latent[:, : self.num_hist]
        target = latent[:, self.num_pred :].detach()
        predicted = self.predict(source)

        visual_pred = predicted[..., : self.token_dim]
        visual_target = target[..., : self.token_dim]
        proprio_start = self.token_dim
        proprio_end = proprio_start + self.proprio_condition_dim
        proprio_pred = predicted[..., proprio_start:proprio_end]
        proprio_target = target[..., proprio_start:proprio_end]
        z_loss = self.emb_criterion(
            predicted[..., : -self.action_condition_dim],
            target[..., : -self.action_condition_dim],
        )
        visual_loss = self.emb_criterion(visual_pred, visual_target)
        proprio_loss = self.emb_criterion(proprio_pred, proprio_target)
        # Cosine is diagnostic-only. Keep it outside the autograd graph so it
        # cannot contribute to the optimized DINO shifted-MSE objective.
        with torch.no_grad():
            cosine_similarity = F.cosine_similarity(
                visual_pred.detach().float(),
                visual_target.detach().float(),
                dim=-1,
            ).mean()
            cosine_loss = 1.0 - cosine_similarity
        losses: dict[str, torch.Tensor] = {
            "_loss": z_loss,
            "loss": z_loss,
            "z_loss": z_loss,
            "z_visual_loss": visual_loss,
            "z_proprio_loss": proprio_loss,
            "hidden_mse": visual_loss,
            "next_latent_mse": visual_loss,
            "hidden_rec_loss": visual_loss,
            "hidden_cosine_loss": cosine_loss,
            "hidden_cosine_similarity": cosine_similarity,
            "hidden_pred_norm": visual_pred.float().norm(dim=-1).mean(),
            "hidden_target_norm": visual_target.float().norm(dim=-1).mean(),
            "proprio_reconstruction_loss": proprio_loss,
            "proprio_pred_norm": proprio_pred.float().norm(dim=-1).mean(),
            "proprio_target_norm": proprio_target.float().norm(dim=-1).mean(),
            "teacher_forced_steps": torch.tensor(
                self.num_hist,
                device=z_loss.device,
                dtype=z_loss.dtype,
            ),
        }
        if self.return_predictions:
            losses["hidden_pred"] = visual_pred
            losses["hidden_target"] = visual_target
        return losses

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Compute DINO's shifted one-step loss from a DreamerVLA replay batch."""

        return self._loss_from_batch(batch)

    def rollout(
        self,
        initial_observations: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Autoregress exactly as DINO-WM, including its final extra prediction."""

        visual = initial_observations.get("visual")
        if not isinstance(visual, torch.Tensor):
            raise KeyError("initial_observations must contain Tensor key 'visual'")
        num_initial = int(visual.shape[1])
        if num_initial < 1:
            raise ValueError("rollout requires at least one initial observation")
        if int(actions.shape[1]) < num_initial:
            raise ValueError("rollout actions must cover every initial observation")
        initial_actions = actions[:, :num_initial]
        future_actions = actions[:, num_initial:]
        latent = self.encode(initial_observations, initial_actions)
        for step in range(int(future_actions.shape[1])):
            predicted = self.predict(latent[:, -self.num_hist :])
            new_latent = predicted[:, -1:]
            new_latent = self.replace_actions_from_z(
                new_latent,
                future_actions[:, step : step + 1],
            )
            latent = torch.cat([latent, new_latent], dim=1)
        predicted = self.predict(latent[:, -self.num_hist :])
        latent = torch.cat([latent, predicted[:, -1:]], dim=1)
        observations, _ = self.separate_emb(latent)
        return observations, latent


__all__ = [
    "DinoTokenEmbedding",
    "DinoTokenViTPredictor",
    "DinoTokenWorldModel",
    "generate_dino_causal_mask",
]
