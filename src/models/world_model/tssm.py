from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.chameleon_model.modeling_xllmx_chameleon_ck_action_head import (
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
)
from src.models.world_model.image_codec import (
    BspaceConvDecoderHead,
    ConvEncoderStem,
)
from src.models.world_model.token_io import ImageTokenEmbedder


@dataclass
class TSSMState:
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor

    def feature(self) -> torch.Tensor:
        return torch.cat([self.stoch, self.deter], dim=-1)


class TSSMWorldModel(nn.Module):
    """
    A TSSM-style world model adapted to the current DreamerVLA interface.

    The transition backbone is loaded exactly like RynnVLA-002 loads its
    action/world-model checkpoint: instantiate the full HF
    `ChameleonXLLMXForConditionalGeneration_ck_action_head` checkpoint and then
    drop the VQ module. For TSSM we only consume the decoder transformer hidden
    states as a causal dynamics backbone; we intentionally do not add a second
    autoregressive token head on top because the checkpoint already contains the
    causal decoder stack and TSSM's supervision is carried by the latent
    prior/posterior + transition/reward heads defined below.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        latent_dim: int = 256,
        state_token_count: int = 1,
        mapper_hidden_dim: int = 1024,
        dynamics_hidden_dim: int = 4096,
        reward_hidden_dim: int = 512,
        reward_loss_coef: float = 0.0,
        kl_loss_coef: float = 0.1,
        min_std: float = 0.1,
        pretrained_model_path: str = "/home/user01/yuxinglei/workspace/DreamerVLA/data/ckpts/Action_World_model_512/libero_10",
        freeze_transition_backbone: bool = False,
        backbone_dtype: str = "bfloat16",
        transition_time_horizon: int | None = None,
        training: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.state_token_count = int(state_token_count)
        self.mapper_hidden_dim = int(mapper_hidden_dim)
        self.dynamics_hidden_dim = int(dynamics_hidden_dim)
        self.reward_loss_coef = float(reward_loss_coef)
        self.kl_loss_coef = float(kl_loss_coef)
        self.min_std = float(min_std)
        self.freeze_transition_backbone = bool(freeze_transition_backbone)
        self.pretrained_model_path = str(pretrained_model_path)
        self.transition_time_horizon = self._resolve_transition_time_horizon(
            pretrained_model_path=self.pretrained_model_path,
            fallback=transition_time_horizon,
        )

        self.obs_to_stoch = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, 2 * self.latent_dim),
        )
        self.obs_to_deter = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, self.dynamics_hidden_dim),
        )
        self.state_to_tokens = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.dynamics_hidden_dim),
            nn.Linear(self.latent_dim + self.dynamics_hidden_dim, self.state_token_count * self.dynamics_hidden_dim),
        )
        self.action_to_tokens = nn.Sequential(
            nn.LayerNorm(self.action_dim),
            nn.Linear(self.action_dim, self.dynamics_hidden_dim),
        )
        self.token_type_embeddings = nn.Embedding(2, self.dynamics_hidden_dim)

        self.transition_summary = nn.Sequential(
            nn.LayerNorm(self.dynamics_hidden_dim),
            nn.Linear(self.dynamics_hidden_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, self.dynamics_hidden_dim),
        )
        self.prior_head = nn.Sequential(
            nn.LayerNorm(self.dynamics_hidden_dim),
            nn.Linear(self.dynamics_hidden_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, 2 * self.latent_dim),
        )
        self.posterior_head = nn.Sequential(
            nn.LayerNorm(self.dynamics_hidden_dim + self.obs_dim),
            nn.Linear(self.dynamics_hidden_dim + self.obs_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, 2 * self.latent_dim),
        )
        self.transition_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.dynamics_hidden_dim),
            nn.Linear(self.latent_dim + self.dynamics_hidden_dim, self.mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mapper_hidden_dim, self.obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.LayerNorm(2 * (self.latent_dim + self.dynamics_hidden_dim) + self.action_dim),
            nn.Linear(2 * (self.latent_dim + self.dynamics_hidden_dim) + self.action_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, 1),
        )

        torch_dtype = getattr(torch, backbone_dtype)
        self.transition_backbone = self._load_transition_backbone_like_rynnvla002(
            pretrained_model_path=self.pretrained_model_path,
            action_dim=self.action_dim,
            time_horizon=self.transition_time_horizon,
            torch_dtype=torch_dtype,
        )
        if self.freeze_transition_backbone:
            self.transition_backbone.eval()
            for parameter in self.transition_backbone.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _resolve_pretrained_model_dir(pretrained_model_path: str) -> Path:
        candidate = Path(pretrained_model_path).expanduser().resolve()
        if candidate.is_dir():
            if (candidate / "config.json").is_file():
                return candidate
            for subdir in sorted(path for path in candidate.iterdir() if path.is_dir()):
                if (subdir / "config.json").is_file():
                    return subdir.resolve()
        return candidate

    @classmethod
    def _load_transition_backbone_like_rynnvla002(
        cls,
        pretrained_model_path: str,
        action_dim: int,
        time_horizon: int,
        torch_dtype: torch.dtype,
    ) -> ChameleonXLLMXForConditionalGeneration_ck_action_head:
        model_dir = cls._resolve_pretrained_model_dir(pretrained_model_path)
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

    @staticmethod
    def _resolve_transition_time_horizon(pretrained_model_path: str, fallback: int | None) -> int:
        if fallback is not None:
            return int(fallback)
        config_path = TSSMWorldModel._resolve_pretrained_model_dir(pretrained_model_path) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            time_horizon = config.get("time_horizon")
            if time_horizon is not None:
                return int(time_horizon)
        return 5

    def _reduce_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim == 3:
            return hidden.mean(dim=1)
        raise ValueError(f"Unsupported hidden shape: {tuple(hidden.shape)}")

    @staticmethod
    def _apply_action_mask(actions: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None or actions.ndim != 3:
            return actions
        mask = action_mask.to(device=actions.device, dtype=actions.dtype).unsqueeze(-1)
        return actions * mask

    def _stats_to_stoch(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std_param = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std_param) + self.min_std
        if self.training:
            stoch = mean + std * torch.randn_like(mean)
        else:
            stoch = mean
        return mean, std, stoch

    def _build_state(self, stats: torch.Tensor, deter: torch.Tensor) -> TSSMState:
        mean, std, stoch = self._stats_to_stoch(stats)
        return TSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def encode_latent(self, hidden: torch.Tensor) -> TSSMState:
        reduced_hidden = self._reduce_hidden(hidden)
        deter = self.obs_to_deter(reduced_hidden)
        stats = self.obs_to_stoch(reduced_hidden)
        return self._build_state(stats, deter)

    def _build_transition_inputs(self, state: TSSMState, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, _ = action.shape
        state_feature = state.feature()
        state_tokens = self.state_to_tokens(state_feature).view(batch_size, self.state_token_count, self.dynamics_hidden_dim)
        action_tokens = self.action_to_tokens(action)

        state_type_ids = torch.zeros(batch_size, self.state_token_count, dtype=torch.long, device=action.device)
        action_type_ids = torch.ones(batch_size, horizon, dtype=torch.long, device=action.device)
        state_tokens = state_tokens + self.token_type_embeddings(state_type_ids)
        action_tokens = action_tokens + self.token_type_embeddings(action_type_ids)

        input_embeds = torch.cat([state_tokens, action_tokens], dim=1)
        attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.bool, device=input_embeds.device)
        return input_embeds, attention_mask

    def _predict_prior(self, state: TSSMState, action: torch.Tensor) -> TSSMState:
        input_embeds, attention_mask = self._build_transition_inputs(state, action)
        backbone_dtype = next(self.transition_backbone.parameters()).dtype
        outputs = self.transition_backbone.model(
            inputs_embeds=input_embeds.to(dtype=backbone_dtype),
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        deter = self.transition_summary(outputs.last_hidden_state[:, -1])
        stats = self.prior_head(deter)
        return self._build_state(stats, deter)

    def predict_next(self, latent: TSSMState, actions: torch.Tensor) -> TSSMState:
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        return self._predict_prior(latent, actions)

    @staticmethod
    def _gaussian_kl_divergence(post: TSSMState, prior: TSSMState) -> torch.Tensor:
        post_var = post.std.pow(2)
        prior_var = prior.std.pow(2)
        log_ratio = torch.log(prior.std) - torch.log(post.std)
        sq_term = (post_var + (post.mean - prior.mean).pow(2)) / (2.0 * prior_var.clamp_min(1e-6))
        kl = log_ratio + sq_term - 0.5
        return kl.sum(dim=-1).mean()

    def reward(
        self,
        latent: TSSMState,
        actions: torch.Tensor,
        next_latent: TSSMState,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions.ndim == 3:
            actions = actions.mean(dim=1)
        features = torch.cat([latent.feature(), actions, next_latent.feature()], dim=-1)
        return self.reward_head(features).squeeze(-1)

    def pretrain_loss(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
        next_hidden: torch.Tensor,
        reward_target: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        action = self._apply_action_mask(action, action_mask)
        latent = self.encode_latent(hidden)
        prior_next = self.predict_next(latent, action)

        next_hidden_target = self._reduce_hidden(next_hidden)
        posterior_stats = self.posterior_head(torch.cat([prior_next.deter, next_hidden_target], dim=-1))
        posterior_next = self._build_state(posterior_stats, prior_next.deter)

        predicted_next_hidden = self.transition_head(prior_next.feature())
        transition_loss = F.mse_loss(predicted_next_hidden, next_hidden_target)
        kl_loss = self._gaussian_kl_divergence(posterior_next, prior_next)

        predicted_reward = self.reward(latent, action, prior_next)
        if reward_target is None:
            reward_target = torch.zeros_like(predicted_reward)
        reward_target = reward_target.reshape_as(predicted_reward)
        reward_loss = F.mse_loss(predicted_reward, reward_target)

        loss = transition_loss + self.kl_loss_coef * kl_loss
        if self.reward_loss_coef > 0:
            loss = loss + self.reward_loss_coef * reward_loss

        return {
            "loss": loss,
            "transition_loss": transition_loss,
            "kl_loss": kl_loss,
            "reward_loss": reward_loss,
            "predicted_reward_mean": predicted_reward.mean(),
            "latent_norm": latent.feature().norm(dim=-1).mean(),
        }

    def compute_loss_dict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        hidden = batch.get("obs_embedding")
        next_hidden = batch.get("next_obs_embedding")
        action = batch["action"]
        action_mask = batch.get("action_mask")
        reward = batch.get("reward")
        if hidden is None or next_hidden is None:
            raise ValueError("World model expects `obs_embedding` and `next_obs_embedding` in the batch.")
        first_param = next(self.parameters())
        device = first_param.device
        model_dtype = first_param.dtype
        hidden = hidden.to(device=device, dtype=model_dtype)
        next_hidden = next_hidden.to(device=device, dtype=model_dtype)
        action = action.to(device=device, dtype=model_dtype)
        if action_mask is not None:
            action_mask = action_mask.to(device=device)
        if reward is not None:
            reward = reward.to(device=device, dtype=model_dtype)
        return self.pretrain_loss(
            hidden=hidden,
            action=action,
            next_hidden=next_hidden,
            reward_target=reward,
            action_mask=action_mask,
        )

    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.compute_loss_dict(batch)["loss"]

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """FSDP-compatible entry point: delegates to compute_loss_dict.

        Calling compute_loss_dict directly bypasses FSDP's all-gather hook
        (which fires only on __call__ / forward).  Routing through forward
        ensures sharded parameters are fully materialized before use.
        """
        return self.compute_loss_dict(batch)

    @torch.no_grad()
    def predict_next_hidden(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step inference helper (for eval/visualisation).

        Composes encode_latent -> predict_next -> transition_head so callers can
        stay agnostic to which TSSM variant they hold.
        """
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype
        hidden = hidden.to(device=device, dtype=dtype)
        action = action.to(device=device, dtype=dtype)
        if action.ndim == 2:
            action = action.unsqueeze(1)
        latent = self.encode_latent(hidden)
        prior_next = self.predict_next(latent, action)
        return self.transition_head(prior_next.feature())


from .causal_transformer import CausalTransformerCell, LLMBackboneCell

__all__ = ["TSSMState", "TSSMWorldModel", "CausalTransformerCell", "TSSMWorldModelTransDreamer"]


# ---------------------------------------------------------------------------
# TransDreamer-style TSSM
# ---------------------------------------------------------------------------
# Key difference from TSSMWorldModel:
#
#   Old (single-step):
#       prior_next = Transformer([z_t, a_t, ..., a_{t+H-1}])  ← future actions
#
#   New (TransDreamer, sequence):
#       h_t = CausalTransformer([z_0⊕a_0, z_1⊕a_1, ..., z_{t-1}⊕a_{t-1}])
#       prior_z_t ~ N(prior_head(h_t))                         ← history-conditioned
#
# References:
#   TransDreamer (Chen et al. 2022) modules_transformer.py:
#       infer_prior_stoch  → our _infer_prior_seq
#       infer_post_stoch   → our _encode_posterior_seq
#       _generate_square_subsequent_mask → our CausalTransformerCell._causal_mask
# ---------------------------------------------------------------------------


class TSSMWorldModelTransDreamer(nn.Module):
    """
    TransDreamer-style world model for DreamerVLA.

    Data flow (mirrors TransDreamer modules_transformer.py: forward()):

        hidden_seq  [B, T, obs_dim]     ← LLM hidden states (from frozen VLA)
        action_seq  [B, T, action_dim]  ← single action per step

        # Step 1 – posterior: q(z_t | o_t), each frame independently
        #   TransDreamer: infer_post_stoch(obs_emb)
        posterior_seq = encode_posterior_seq(hidden_seq)        # [B, T, latent_dim]

        # Step 2 – prior: p(z_t | z_{0:t-1}, a_{0:t-1}), causal Transformer
        #   TransDreamer: infer_prior_stoch(s_t, temp, actions)
        token_seq  = act_stoch_emb(cat(z_{0:T-2}, a_{0:T-2}))  # [B, T-1, d_model]
        h_seq      = CausalTransformer(token_seq)                # [B, T-1, d_model]
        prior_seq  = prior_head(h_seq)                          # prior for z_{1:T}

        # Step 3 – losses
        #   TransDreamer: world_model_loss() kl_balance
        kl_loss        = KL(posterior_{1:T} || prior_{1:T})  with free_nats + kl_balance
        transition_loss = MSE(transition_head(h_seq), hidden_{1:T})   ← anchor hidden state
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        latent_dim: int = 256,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        mapper_hidden_dim: int = 512,
        reward_hidden_dim: int = 256,
        min_std: float = 0.1,
        free_nats: float = 1.0,
        kl_balance: float = 0.8,       # weight on rep loss (DreamerV3/TransDreamer style)
        kl_loss_coef: float = 1.0,
        transition_loss_coef: float = 1.0,
        reward_loss_coef: float = 0.0,
        use_pretrained_backbone: bool = False,
        pretrained_model_path: str | None = None,
        freeze_transition_backbone: bool = True,
        backbone_dtype: str = "bfloat16",
        transition_time_horizon: int | None = None,
        image_decoder_enabled: bool = False,
        n_image_tokens: int = 256,
        image_decoder_hidden_dim: int = 1024,
        image_decoder_loss_coef: float = 0.0,
        # ── Route-B spatial codec (strided-conv encoder + bspace deconv) ─────
        spatial_codec: bool = False,
        obs_dim: int | None = None,
        in_channels: int = 4096,              # raw per-token hidden size
        spatial_grid: tuple[int, int] = (16, 16),
        stem_init_proj_channels: int = 384,
        stem_stage_channels: tuple[int, ...] = (96, 192),
        stem_kernel: int = 4,
        stem_stride: int = 2,
        stem_padding: int = 1,
        decoder_mid_channels: int = 192,
        decoder_bspace_groups: int = 8,
        decoder_minres: tuple[int, int] = (4, 4),
        decoder_stage_channels: tuple[int, ...] = (96, 48),
        decoder_kernel: int = 4,
        decoder_stride: int = 2,
        decoder_padding: int = 1,
        decoder_stoch_hidden: int = 512,
        image_recon_ce_coef: float = 1.0,    # cross-entropy on token ids
        image_recon_mse_coef: float = 0.1,   # MSE on per-token 4096-d hiddens
        # ── Discrete-token I/O (io_mode="token") ─────────────────────────────
        # Replaces the frozen-Chameleon-hidden I/O with a learnable image-
        # token embedder on the input side and a direct logits head on the
        # output side.  Share the same spatial_codec scaffolding (conv stem
        # + bspace conv decoder) but swap the channel dims:
        #   stem.in_channels     = token_embed_dim          (instead of 4096)
        #   decoder.out_channels = num_image_tokens_vocab   (instead of 4096)
        # No frozen lm_head needed; decoder output IS the image-vocab logits.
        io_mode: str = "hidden",
        token_embed_dim: int = 512,
        num_image_tokens_vocab: int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_codec = bool(spatial_codec)
        self.in_channels = int(in_channels)
        self.spatial_grid = (int(spatial_grid[0]), int(spatial_grid[1]))
        # `obs_dim`: WM's scalar hidden size.  Under spatial_codec it defaults
        # to 1024 (post-stem); without the codec it equals `hidden_dim` to
        # preserve the pre-refactor behaviour.
        if obs_dim is None:
            self.obs_dim = 1024 if self.spatial_codec else int(hidden_dim)
        else:
            self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.min_std = float(min_std)
        self.free_nats = float(free_nats)
        self.kl_balance = float(kl_balance)
        self.kl_loss_coef = float(kl_loss_coef)
        self.transition_loss_coef = float(transition_loss_coef)
        self.reward_loss_coef = float(reward_loss_coef)
        self.image_decoder_enabled = bool(image_decoder_enabled) or self.spatial_codec
        self.n_image_tokens = int(n_image_tokens)
        self.image_decoder_loss_coef = float(image_decoder_loss_coef)
        self.image_recon_ce_coef = float(image_recon_ce_coef)
        self.image_recon_mse_coef = float(image_recon_mse_coef)

        # ── Posterior encoder: q(z_t | o_t), each frame independently ──────
        # TransDreamer: post_stoch_mlp  (modules_transformer.py:290)
        self.obs_to_stoch = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(mapper_hidden_dim, 2 * self.latent_dim),
        )

        # ── Token embedding: cat(z_t, a_t) → d_model ────────────────────────
        # TransDreamer: act_stoch_mlp  (modules_transformer.py:267)
        self.act_stoch_emb = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.action_dim),
            nn.Linear(self.latent_dim + self.action_dim, d_model),
            nn.GELU(),
        )

        # ── Causal Transformer: h_t = Transformer(z_{0:t-1}, a_{0:t-1}) ────
        # TransDreamer: self.cell = Transformer(cfg)  (modules_transformer.py:259)
        # Two backends with identical [B,T,d_model]->[B,T,d_model] interface:
        #   use_pretrained_backbone=False : CausalTransformerCell (random init)
        #   use_pretrained_backbone=True  : LLMBackboneCell wrapping a Chameleon ckpt
        if use_pretrained_backbone:
            if not pretrained_model_path:
                raise ValueError(
                    "use_pretrained_backbone=True requires pretrained_model_path to be set."
                )
            self.causal_transformer = LLMBackboneCell(
                pretrained_model_path=pretrained_model_path,
                d_model=d_model,
                action_dim=action_dim,
                time_horizon=transition_time_horizon,
                backbone_dtype=backbone_dtype,
                freeze=freeze_transition_backbone,
            )
        else:
            self.causal_transformer = CausalTransformerCell(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                dropout=dropout,
            )

        # ── Prior head: p(z_t | h_t) ────────────────────────────────────────
        # TransDreamer: prior_stoch_mlp  (modules_transformer.py:299)
        self.prior_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(mapper_hidden_dim, 2 * self.latent_dim),
        )

        # ── Transition head: reconstruct next hidden from (h_t, z_t) ────────
        # Anchors the Transformer output to real observations (like image recon in Dreamer)
        self.transition_head = nn.Sequential(
            nn.LayerNorm(d_model + self.latent_dim),
            nn.Linear(d_model + self.latent_dim, mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(mapper_hidden_dim, self.obs_dim),
        )

        # ── Reward head (optional) ───────────────────────────────────────────
        self.reward_head = nn.Sequential(
            nn.LayerNorm(d_model + self.latent_dim),
            nn.Linear(d_model + self.latent_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, 1),
        )

        # ── Spatial codec (route B): strided-conv encoder stem + bspace ─────
        # conv decoder, tied to the frozen LLM lm_head at the output edge.
        if self.spatial_codec:
            self.conv_stem = ConvEncoderStem(
                in_channels=self.in_channels,
                spatial=self.spatial_grid,
                obs_dim=self.obs_dim,
                init_proj_channels=stem_init_proj_channels,
                stage_channels=tuple(stem_stage_channels),
                kernel=stem_kernel, stride=stem_stride, padding=stem_padding,
            )
            self.image_decoder: nn.Module | None = BspaceConvDecoderHead(
                deter_dim=self.obs_dim,
                stoch_dim=self.latent_dim,
                minres=tuple(decoder_minres),
                mid_channels=decoder_mid_channels,
                bspace_groups=decoder_bspace_groups,
                stage_channels=tuple(decoder_stage_channels),
                out_channels=self.in_channels,
                out_spatial=self.spatial_grid,
                kernel=decoder_kernel,
                stride=decoder_stride,
                padding=decoder_padding,
                stoch_hidden=decoder_stoch_hidden,
            )
            # Configure number of image tokens to match spatial grid.
            self.n_image_tokens = self.spatial_grid[0] * self.spatial_grid[1]
        else:
            self.conv_stem = None
            # ── Legacy MLP image decoder (route-0 behaviour) ────────────────
            # Maps predicted next pooled hidden [obs_dim] → per-image-token
            # hiddens [n_image_tokens, obs_dim]. Pure latent path: NO
            # current-frame shortcut.  Output will be projected through the
            # frozen LLM lm_head by the viz code to produce image-token logits
            # → VQGAN pixels.
            if self.image_decoder_enabled:
                self.image_decoder = nn.Sequential(
                    nn.LayerNorm(self.obs_dim),
                    nn.Linear(self.obs_dim, image_decoder_hidden_dim),
                    nn.GELU(),
                    nn.Linear(image_decoder_hidden_dim, self.n_image_tokens * self.obs_dim),
                )
            else:
                self.image_decoder = None

        # ── io_mode="token": override stem + decoder for discrete-token I/O ──
        self.io_mode = str(io_mode)
        if self.io_mode not in ("hidden", "token"):
            raise ValueError(f"io_mode must be 'hidden' or 'token', got {io_mode!r}")
        self.token_embed_dim = int(token_embed_dim)
        self.num_image_tokens_vocab = (
            int(num_image_tokens_vocab) if num_image_tokens_vocab is not None else None
        )

        if self.io_mode == "token":
            if not self.spatial_codec:
                raise ValueError("io_mode='token' requires spatial_codec=True")
            if self.num_image_tokens_vocab is None:
                raise ValueError(
                    "io_mode='token' requires num_image_tokens_vocab to be set in config"
                )
            self.token_embedder = ImageTokenEmbedder(
                num_image_tokens_vocab=self.num_image_tokens_vocab,
                d_embed=self.token_embed_dim,
                spatial=self.spatial_grid,
            )
            # Stem input channels = token_embed_dim (not 4096)
            self.conv_stem = ConvEncoderStem(
                in_channels=self.token_embed_dim,
                spatial=self.spatial_grid,
                obs_dim=self.obs_dim,
                init_proj_channels=stem_init_proj_channels,
                stage_channels=tuple(stem_stage_channels),
                kernel=stem_kernel, stride=stem_stride, padding=stem_padding,
            )
            # Decoder output = image-vocab logits (no lm_head needed downstream)
            self.image_decoder = BspaceConvDecoderHead(
                deter_dim=self.obs_dim,
                stoch_dim=self.latent_dim,
                minres=tuple(decoder_minres),
                mid_channels=decoder_mid_channels,
                bspace_groups=decoder_bspace_groups,
                stage_channels=tuple(decoder_stage_channels),
                out_channels=self.num_image_tokens_vocab,
                out_spatial=self.spatial_grid,
                kernel=decoder_kernel,
                stride=decoder_stride,
                padding=decoder_padding,
                stoch_hidden=decoder_stoch_hidden,
            )
            # Token mode's CE-only training makes MSE coef moot.
            self.image_recon_mse_coef = 0.0
        else:
            self.token_embedder = None

        # ── Externally-attached lm_head + image_token_bpe_ids (route B only)
        # These are used to compute CE loss over image-token logits without
        # duplicating the (very large) LLM lm_head weights inside the WM.
        # `attach_lm_head(...)` is called by the workspace after both the
        # encoder and the WM are built.
        self._lm_head_ref: list = []   # non-nn.Module container so FSDP ignores
        self.register_buffer(
            "image_token_bpe_ids",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_bpe_to_img_idx",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

    # ── Wiring helpers ───────────────────────────────────────────────────────

    def attach_lm_head(
        self, lm_head: nn.Module | None, image_token_bpe_ids: torch.Tensor,
        full_vocab_size: int,
    ) -> None:
        """Register the image-vocab mapping, and optionally a reference to the
        (frozen) LLM lm_head for hidden-mode CE over image-token logits.

        - hidden mode: pass the real lm_head — used to project decoded 4096-d
          per-token hiddens to image-vocab logits.
        - token mode:  pass ``lm_head=None`` — decoder already outputs logits.
          Only the ``_bpe_to_img_idx`` buffer is needed (to map raw BPE ids in
          inputs / CE targets to image-vocab indices).

        Stored in a non-``nn.Module`` list so FSDP ignores the reference.
        """
        self._lm_head_ref = [lm_head] if lm_head is not None else []
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        image_token_bpe_ids = image_token_bpe_ids.to(
            dtype=torch.long, device=target_device,
        ).clone()
        self.image_token_bpe_ids = image_token_bpe_ids
        rev = torch.full(
            (int(full_vocab_size),), -1, dtype=torch.long, device=target_device,
        )
        rev[image_token_bpe_ids] = torch.arange(
            image_token_bpe_ids.numel(), dtype=torch.long, device=target_device,
        )
        self._bpe_to_img_idx = rev

    @property
    def has_lm_head(self) -> bool:
        return bool(self._lm_head_ref)

    @property
    def lm_head(self) -> nn.Module | None:
        return self._lm_head_ref[0] if self._lm_head_ref else None

    def _image_logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run per-token hidden [..., 4096] through a restricted lm_head that
        projects only onto the image-token subvocabulary.  Returns
        [..., num_image_vocab].
        """
        lm_head = self.lm_head
        assert lm_head is not None, "lm_head not attached; call attach_lm_head()"
        # Restrict lm_head to image-vocab rows — no full-vocab matmul.
        # lm_head weight shape: [V, C_in].
        w_full = lm_head.weight                              # [V, C]
        w_img = w_full.index_select(0, self.image_token_bpe_ids)   # [num_img, C]
        w_img = w_img.to(dtype=hidden.dtype)
        logits = torch.matmul(hidden, w_img.transpose(-1, -2))
        b_full = getattr(lm_head, "bias", None)
        if b_full is not None:
            b_img = b_full.index_select(0, self.image_token_bpe_ids).to(dtype=hidden.dtype)
            logits = logits + b_img
        return logits

    # ── Distribution helpers ─────────────────────────────────────────────────

    def _stats_to_dist(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split 2*latent output into (mean, std, stoch). Works on any leading shape."""
        mean, std_param = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std_param) + self.min_std
        if self.training:
            stoch = mean + std * torch.randn_like(mean)
        else:
            stoch = mean
        return mean, std, stoch

    @staticmethod
    def _gaussian_kl(
        post_mean: torch.Tensor,
        post_std:  torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Analytic KL(N(post) || N(prior)), summed over latent dim, mean over batch+time.
        Shape of inputs: [B, T, latent_dim]  →  scalar.
        """
        post_var  = post_std.pow(2)
        prior_var = prior_std.pow(2)
        log_ratio = torch.log(prior_std) - torch.log(post_std)
        sq_term   = (post_var + (post_mean - prior_mean).pow(2)) / (2.0 * prior_var.clamp_min(1e-6))
        kl = log_ratio + sq_term - 0.5            # [B, T, latent_dim]
        return kl.sum(dim=-1).mean()               # scalar

    # ── Encoder: posterior, each frame independently ─────────────────────────

    def _encode_posterior_seq(
        self, hidden_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        q(z_t | o_t) for t = 0..T-1, each frame independently.

        TransDreamer: infer_post_stoch(obs_emb)  (modules_transformer.py:442)

        Args:
            hidden_seq: [B, T, obs_dim]
        Returns:
            mean_seq, std_seq, stoch_seq: each [B, T, latent_dim]
        """
        stats = self.obs_to_stoch(hidden_seq)          # [B, T, 2*latent_dim]
        return self._stats_to_dist(stats)

    # ── Prior: causal Transformer over history ───────────────────────────────

    def _infer_prior_seq(
        self,
        stoch_seq: torch.Tensor,    # [B, T-1, latent_dim]  z_{0:T-2}
        action_seq: torch.Tensor,   # [B, T-1, action_dim]  a_{0:T-2}
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        p(z_t | z_{0:t-1}, a_{0:t-1}) for t = 1..T-1.

        TransDreamer: infer_prior_stoch(prev_stoch, temp, actions)  (modules_transformer.py:409)

        Pipeline:
            token_t = act_stoch_emb(cat(z_t, a_t))  →  [B, T-1, d_model]
            h_seq   = CausalTransformer(token_seq)   →  [B, T-1, d_model]
                      h_seq[:, t] encodes z_{0:t}, a_{0:t}
                      → used as prior for z_{t+1}
            prior_t = prior_head(h_seq)              →  [B, T-1, 2*latent_dim]

        Returns:
            prior_mean, prior_std, prior_stoch: [B, T-1, latent_dim]
            h_seq:                              [B, T-1, d_model]
        """
        # token_t = encode(z_t, a_t)
        token_input = torch.cat([stoch_seq, action_seq], dim=-1)   # [B, T-1, latent+action]
        token_seq   = self.act_stoch_emb(token_input)               # [B, T-1, d_model]

        # h_t = Transformer(token_{0:t}) with causal mask
        # TransDreamer: o_t = self.cell(s_t_reshape, None)
        h_seq = self.causal_transformer(token_seq)                  # [B, T-1, d_model]

        # prior distribution for z_{t+1}
        prior_stats = self.prior_head(h_seq)                        # [B, T-1, 2*latent_dim]
        prior_mean, prior_std, prior_stoch = self._stats_to_dist(prior_stats)

        return prior_mean, prior_std, prior_stoch, h_seq

    # ── Loss computation ─────────────────────────────────────────────────────

    def pretrain_loss(
        self,
        hidden_seq: torch.Tensor,
        action_seq: torch.Tensor,          # [B, T, action_dim]
        reward_seq: torch.Tensor | None = None,  # [B, T]
        done_seq:   torch.Tensor | None = None,  # [B, T]
        next_image_hiddens_target: torch.Tensor | None = None,  # [B, T-1, n_img_tok, in_channels]
        next_image_token_ids_target: torch.Tensor | None = None,  # [B, T-1, n_img_tok] bpe ids
    ) -> dict[str, torch.Tensor]:
        """
        TransDreamer-style sequence loss.

        `hidden_seq` accepted shapes:
          - Route-0: [B, T, obs_dim]               — already scalar per frame
          - Route-B: [B, T, n_img_tok, in_channels] — per-image-token, then
            conv_stem compresses to [B, T, obs_dim] internally.

        Mirrors TransDreamer world_model_loss() (modules_transformer.py:88):
            prior  ← infer_prior_stoch(post_stoch[:, :-1], actions[:, 1:])
            post   ← infer_post_stoch(obs_emb)
            KL with kl_balance:
                value_lhs = KL(sg(post) || prior)   ← trains prior  (dyn loss)
                value_rhs = KL(post || sg(prior))   ← trains post   (rep loss)
                kl = (1 - kl_balance) * lhs + kl_balance * rhs
        """
        # Under route-B, hidden_seq is per-image-token and must pass through
        # the learnable conv stem before reaching the RSSM posterior.  The
        # per-token tensor is retained for image-recon losses below.
        per_token_hidden_seq = None
        raw_bpe_ids_seq = None  # token mode only: [B, T, N_img] long
        if self.io_mode == "token":
            # hidden_seq: [B, T, N_img] long BPE ids.  Map to image-vocab
            # indices, embed, then run through the same conv_stem as hidden
            # mode to produce [B, T, obs_dim].
            if hidden_seq.ndim != 3:
                raise ValueError(
                    "io_mode='token' requires hidden_seq shape [B, T, N_img]; "
                    f"got {tuple(hidden_seq.shape)}"
                )
            if self._bpe_to_img_idx.numel() == 0:
                raise RuntimeError(
                    "io_mode='token' requires attach_lm_head(lm_head=None, ...) "
                    "to populate _bpe_to_img_idx before forward"
                )
            raw_bpe_ids_seq = hidden_seq.long()
            img_idx_seq = self._bpe_to_img_idx[raw_bpe_ids_seq]   # [B, T, N_img]
            if (img_idx_seq < 0).any():
                raise ValueError(
                    "io_mode='token': input contains non-image BPE ids; "
                    "workspace must filter image tokens only"
                )
            per_token_hidden_seq = self.token_embedder(img_idx_seq)  # [B, T, N_img, d_embed]
            hidden_seq = self.conv_stem(per_token_hidden_seq)        # [B, T, obs_dim]
        elif self.spatial_codec:
            if hidden_seq.ndim != 4:
                raise ValueError(
                    "spatial_codec=True requires hidden_seq shape [B, T, N_img, C_in]; "
                    f"got {tuple(hidden_seq.shape)}"
                )
            per_token_hidden_seq = hidden_seq
            hidden_seq = self.conv_stem(hidden_seq)            # [B, T, obs_dim]
        elif hidden_seq.ndim != 3:
            raise ValueError(
                "spatial_codec=False requires hidden_seq shape [B, T, obs_dim]; "
                f"got {tuple(hidden_seq.shape)}"
            )

        B, T, _ = hidden_seq.shape

        # ── Step 1: posterior for every frame ────────────────────────────────
        # q(z_t | o_t),  t = 0..T-1
        # TransDreamer: post = infer_post_stoch(obs_emb)
        post_mean, post_std, post_stoch = self._encode_posterior_seq(hidden_seq)
        # post_*: [B, T, latent_dim]

        # ── Step 2: prior from causal Transformer on history ─────────────────
        # TransDreamer (modules_transformer.py:323-325):
        #   s_t = post['stoch'][:, :-1]          # z_0 .. z_{T-2}
        #   prior = infer_prior_stoch(s_t, temp, actions[:, 1:])
        #                                         # actions[t] = action that CAUSED obs_t
        #                                         # so actions[:, 1:] = a causing obs_1..obs_{T-1}
        # We follow the same convention: action_seq[:, 1:] pairs with z_{0:T-2}
        prior_mean, prior_std, prior_stoch, h_seq = self._infer_prior_seq(
            stoch_seq  = post_stoch[:, :-1],    # z_{0:T-2}   [B, T-1, latent_dim]
            action_seq = action_seq[:, 1:],     # a_{1:T-1}   [B, T-1, action_dim]
            # action_seq[t] = action that caused hidden_seq[t] (arrived-at convention)
        )
        # prior_*, h_seq are priors for z_{1:T-1}

        # ── Step 3: KL with kl_balance and free_nats ─────────────────────────
        # TransDreamer (modules_transformer.py:126-133):
        #   value_lhs = KL(post_dist, sg(prior_dist))   = KL(post ‖ sg(prior))  → rep
        #   value_rhs = KL(sg(post_dist), prior_dist)   = KL(sg(post) ‖ prior)  → dyn
        #   kl_loss   = (1 - kl_balance) * lhs + kl_balance * rhs
        #             = (1 - kl_balance) * rep + kl_balance * dyn
        # With kl_balance=0.8: 80% weight on dyn (trains the dynamics/prior Transformer)

        # rep: KL(post ‖ sg(prior)) — gradient to posterior encoder only
        rep_kl = self._gaussian_kl(
            post_mean  = post_mean[:, 1:],
            post_std   = post_std[:, 1:],
            prior_mean = prior_mean.detach(),
            prior_std  = prior_std.detach(),
        )
        # dyn: KL(sg(post) ‖ prior) — gradient to causal Transformer / prior_head only
        dyn_kl = self._gaussian_kl(
            post_mean  = post_mean[:, 1:].detach(),
            post_std   = post_std[:, 1:].detach(),
            prior_mean = prior_mean,
            prior_std  = prior_std,
        )

        # free_nats: do not penalise below this floor
        # TransDreamer: loss_lhs = max(value_lhs.mean(), free_nats)
        rep_kl = torch.clamp(rep_kl, min=self.free_nats)
        dyn_kl = torch.clamp(dyn_kl, min=self.free_nats)

        # Combine: (1-balance)*rep + balance*dyn  (matches TransDreamer exactly)
        kl_loss = (1.0 - self.kl_balance) * rep_kl + self.kl_balance * dyn_kl

        # ── Step 4: transition loss (reconstruction anchor) ───────────────────
        # TransDreamer uses image reconstruction log_prob for this role.
        # We use MSE on LLM hidden states as the observation reconstruction target.
        # post['deter'] = prior['deter'] in TransDreamer (modules_transformer.py:327):
        #   feature = cat(post_stoch, prior_deter) to anchor the Transformer output.
        # Here we concat h_seq (prior deter) with posterior stoch for the same effect.
        post_feature = torch.cat([h_seq, post_stoch[:, 1:]], dim=-1)    # [B, T-1, d_model+latent]
        predicted_next_hidden = self.transition_head(post_feature)        # [B, T-1, obs_dim]
        transition_loss = F.mse_loss(predicted_next_hidden, hidden_seq[:, 1:].detach())

        # ── Step 5: reward loss (optional) ───────────────────────────────────
        predicted_reward = self.reward_head(post_feature).squeeze(-1)  # [B, T-1]
        if reward_seq is not None:
            reward_target = reward_seq[:, 1:].reshape_as(predicted_reward)
        else:
            reward_target = torch.zeros_like(predicted_reward)
        reward_loss = F.mse_loss(predicted_reward, reward_target)

        # ── Step 6: image-decoder loss (pure-latent) ─────────────────────────
        # Decode predicted next latent into per-image-token hiddens (route-B
        # conv deconv) or per-token hiddens of obs_dim (route-0 MLP).  No
        # current-frame tokens are fed in — everything flows through the WM
        # latent, so viz quality directly measures WM capacity.
        image_decoder_loss = predicted_next_hidden.new_zeros(())
        image_recon_ce_loss = predicted_next_hidden.new_zeros(())
        image_recon_mse_loss = predicted_next_hidden.new_zeros(())
        image_recon_accuracy = predicted_next_hidden.new_zeros(())
        pred_entropy = predicted_next_hidden.new_zeros(())
        uniq_per_sample = predicted_next_hidden.new_zeros(())
        gt_uniq_per_sample = predicted_next_hidden.new_zeros(())

        if self.io_mode == "token" and self.image_decoder is not None:
            # Token mode: decoder output IS image-vocab logits; no lm_head.
            logits = self.image_decoder(
                predicted_next_hidden, post_stoch[:, 1:]
            )  # [B, T-1, N_img, num_image_tokens_vocab]

            # CE target: next-frame image-vocab indices.  Derive from the raw
            # input BPE ids if caller did not pass an explicit target.
            if next_image_token_ids_target is None:
                assert raw_bpe_ids_seq is not None
                tgt_bpe = raw_bpe_ids_seq[:, 1:]
            else:
                tgt_bpe = next_image_token_ids_target.to(
                    device=logits.device, dtype=torch.long
                )
            img_idx = self._bpe_to_img_idx[tgt_bpe]             # [B, T-1, N_img]
            if (img_idx < 0).any():
                raise ValueError(
                    "io_mode='token': CE target contains non-image BPE ids"
                )
            image_recon_ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                img_idx.reshape(-1),
            )
            with torch.no_grad():
                pred_idx = logits.argmax(dim=-1)                   # [B, T-1, N_img]
                image_recon_accuracy = (pred_idx == img_idx).float().mean()
                # predicted softmax entropy (avg over positions).  Low ⇒ sharp,
                # ln(V) ⇒ uniform.  Useful for spotting over-confident collapse.
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                pred_entropy = -(probs * log_probs).sum(dim=-1).mean()
                # #unique argmax tokens per sample (max = N_img, min = 1).
                # 1 ⇒ single solid colour; N_img ⇒ fully diverse.
                flat_pred = pred_idx.reshape(-1, pred_idx.shape[-1])    # [B*T-1, N_img]
                uniq_per_sample = torch.tensor(
                    [int(torch.unique(row).numel()) for row in flat_pred],
                    dtype=logits.dtype, device=logits.device,
                ).mean()
                # GT diversity as a baseline — shouldn't be 1 unless data is broken.
                flat_gt = img_idx.reshape(-1, img_idx.shape[-1])
                gt_uniq_per_sample = torch.tensor(
                    [int(torch.unique(row).numel()) for row in flat_gt],
                    dtype=logits.dtype, device=logits.device,
                ).mean()
            image_decoder_loss = self.image_recon_ce_coef * image_recon_ce_loss
        elif self.spatial_codec and self.image_decoder is not None:
            # Route-B: conv deconv decoder → [B, T-1, n_img_tok, in_channels]
            # Inputs: predicted_next_hidden (deter) + post_stoch[:, 1:] (stoch)
            decoded = self.image_decoder(
                predicted_next_hidden, post_stoch[:, 1:]
            )  # [B, T-1, N_img, C_in]

            # MSE on per-image-token 4096-d hiddens (representation anchor)
            if per_token_hidden_seq is not None and self.image_recon_mse_coef > 0:
                tgt = per_token_hidden_seq[:, 1:].to(
                    device=decoded.device, dtype=decoded.dtype
                ).detach()
                image_recon_mse_loss = F.mse_loss(decoded, tgt)

            # CE on predicted token ids via frozen lm_head over image vocab
            if (
                next_image_token_ids_target is not None
                and self.has_lm_head
                and self.image_recon_ce_coef > 0
            ):
                logits = self._image_logits_from_hidden(decoded)   # [..., N_img, num_img_vocab]
                # Map bpe ids → index-into-image-vocab
                tgt_bpe = next_image_token_ids_target.to(
                    device=logits.device, dtype=torch.long
                )
                img_idx = self._bpe_to_img_idx[tgt_bpe]            # [..., N_img]
                if (img_idx < 0).any():
                    # Defensive: any non-image bpe id in the target indicates
                    # a bug in workspace extraction; skip CE loss and leave
                    # the zero-tensor in image_recon_ce_loss.
                    pass
                else:
                    image_recon_ce_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        img_idx.reshape(-1),
                    )
            # Keep legacy aggregate for logging parity
            image_decoder_loss = (
                self.image_recon_ce_coef  * image_recon_ce_loss
                + self.image_recon_mse_coef * image_recon_mse_loss
            )
        elif (
            self.image_decoder is not None
            and next_image_hiddens_target is not None
        ):
            # Route-0 legacy MLP image decoder path.
            Bp, Tm1, _ = predicted_next_hidden.shape
            decoded = self.image_decoder(predicted_next_hidden)            # [B, T-1, n_img_tok*obs_dim]
            decoded = decoded.view(Bp, Tm1, self.n_image_tokens, self.obs_dim)
            target = next_image_hiddens_target.to(
                device=decoded.device, dtype=decoded.dtype
            ).detach()
            image_decoder_loss = F.mse_loss(decoded, target)

        # ── Total loss ───────────────────────────────────────────────────────
        loss = (
            self.transition_loss_coef * transition_loss
            + self.kl_loss_coef       * kl_loss
        )
        if self.reward_loss_coef > 0:
            loss = loss + self.reward_loss_coef * reward_loss
        if self.spatial_codec and self.image_decoder is not None:
            # Route-B: apply CE and MSE with their dedicated coefs directly;
            # `image_decoder_loss_coef` acts as a global multiplier (default
            # 1.0) on the combined image-recon loss.
            scale = self.image_decoder_loss_coef if self.image_decoder_loss_coef > 0 else 1.0
            loss = loss + scale * image_decoder_loss
        elif self.image_decoder is not None and self.image_decoder_loss_coef > 0:
            loss = loss + self.image_decoder_loss_coef * image_decoder_loss

        return {
            "loss":                loss,
            "kl_loss":             kl_loss,
            "dyn_kl":              dyn_kl,
            "rep_kl":              rep_kl,
            "transition_loss":     transition_loss,
            "reward_loss":         reward_loss,
            "image_recon_ce_loss":  image_recon_ce_loss,
            "image_recon_mse_loss": image_recon_mse_loss,
            "image_decoder_loss":  image_decoder_loss,
            "image_recon_accuracy": image_recon_accuracy,
            "pred_entropy":        pred_entropy,
            "pred_unique_tokens":  uniq_per_sample,
            "gt_unique_tokens":    gt_uniq_per_sample,
        }

    def compute_loss_dict(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """FSDP entry point.

        Accepts either native sequence inputs
            obs_embedding_seq: [B, T, obs_dim]
            action_seq:        [B, T, action_dim]
        or the current pretokenize single-transition format
            obs_embedding:      [B, obs_dim]
            next_obs_embedding: [B, obs_dim]
            action:             [B, H, action_dim]  (action chunk of length H)

        When the single-transition format is supplied, a T=2 pseudo-sequence
        is constructed: hidden = [obs, next_obs], action = [0, mean_H(action)].
        This matches the minimal A1 adapter: the causal Transformer still runs,
        but on a 1-step history; upgrading to real multi-step sequences is the
        A3 direction (dataset rework).
        """
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype

        if "obs_embedding_seq" in batch and "action_seq" in batch:
            hidden_seq = batch["obs_embedding_seq"].to(device=device, dtype=dtype)
            action_seq = batch["action_seq"].to(device=device, dtype=dtype)
            reward_seq = batch.get("reward_seq")
            done_seq   = batch.get("done_seq")
            if reward_seq is not None:
                reward_seq = reward_seq.to(device=device, dtype=dtype)
            if done_seq is not None:
                done_seq = done_seq.to(device=device, dtype=dtype)
            next_image_hiddens_target = batch.get("next_image_hiddens_seq")
            if next_image_hiddens_target is not None:
                next_image_hiddens_target = next_image_hiddens_target.to(device=device, dtype=dtype)
            next_img_token_ids = batch.get("next_image_token_ids_seq")
            if next_img_token_ids is not None:
                next_img_token_ids = next_img_token_ids.to(device=device, dtype=torch.long)
            return self.pretrain_loss(
                hidden_seq, action_seq, reward_seq, done_seq,
                next_image_hiddens_target=next_image_hiddens_target,
                next_image_token_ids_target=next_img_token_ids,
            )

        if "obs_embedding" not in batch or "next_obs_embedding" not in batch:
            raise ValueError(
                "TSSMWorldModelTransDreamer expects either "
                "(obs_embedding_seq, action_seq) or "
                "(obs_embedding, next_obs_embedding, action) in the batch."
            )

        if self.io_mode == "token":
            # obs / next_obs are raw image BPE ids [B, N_img], long.
            obs = batch["obs_embedding"].to(device=device, dtype=torch.long)
            next_obs = batch["next_obs_embedding"].to(device=device, dtype=torch.long)
        else:
            obs = batch["obs_embedding"].to(device=device, dtype=dtype)
            next_obs = batch["next_obs_embedding"].to(device=device, dtype=dtype)
        action = batch["action"].to(device=device, dtype=dtype)
        # Under spatial_codec both obs / next_obs are [B, N_img, C_in].
        # Otherwise they are pooled [B, obs_dim].  The stack below handles both.
        # In token mode they are long [B, N_img].

        # action may be [B, H, A] chunk or [B, A] single step
        if action.ndim == 3:
            action_mask = batch.get("action_mask")
            if action_mask is not None:
                mask = action_mask.to(device=device, dtype=action.dtype).unsqueeze(-1)
                action = action * mask
                denom = mask.sum(dim=1).clamp_min(1.0)
                action_step = action.sum(dim=1) / denom
            else:
                action_step = action.mean(dim=1)
        else:
            action_step = action

        hidden_seq = torch.stack([obs, next_obs], dim=1)                # [B, 2, ...]
        action_seq = torch.stack(
            [torch.zeros_like(action_step), action_step], dim=1
        )                                                                # [B, 2, action_dim]

        reward = batch.get("reward")
        reward_seq = None
        if reward is not None:
            reward = reward.to(device=device, dtype=dtype)
            if reward.ndim == 1:
                reward = reward.unsqueeze(-1)
            reward_zero = torch.zeros_like(reward)
            reward_seq = torch.stack([reward_zero, reward], dim=1).squeeze(-1)

        # Optional per-image-token target for the image_decoder head.
        # Route-0: [B, n_img_tok, obs_dim] → [B, T-1=1, n_img_tok, obs_dim].
        # Route-B: under spatial_codec, per-token hiddens are already in
        # `next_obs_embedding`, so we pull them directly into the target slot.
        # Token mode: no hidden-space target (only CE via token ids); skip.
        next_image_hiddens_target = None
        if self.io_mode == "token":
            pass
        elif self.spatial_codec:
            next_image_hiddens_target = next_obs.unsqueeze(1)   # [B, 1, N_img, C_in]
        else:
            raw_img = batch.get("next_obs_image_hiddens")
            if raw_img is not None:
                next_image_hiddens_target = raw_img.to(device=device, dtype=dtype).unsqueeze(1)

        # Next-frame image token ids for CE loss (route B).
        next_image_token_ids_target = None
        raw_ids = batch.get("next_obs_image_token_ids")
        if raw_ids is not None:
            next_image_token_ids_target = raw_ids.to(
                device=device, dtype=torch.long
            ).unsqueeze(1)   # [B, 1, N_img]

        return self.pretrain_loss(
            hidden_seq, action_seq, reward_seq, None,
            next_image_hiddens_target=next_image_hiddens_target,
            next_image_token_ids_target=next_image_token_ids_target,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.compute_loss_dict(batch)

    @torch.no_grad()
    def predict_next_hidden(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step inference helper used for eval/visualisation.

        Mirrors the forward path through one step of the TransDreamer pipeline:
        posterior on the current frame -> causal Transformer with one token ->
        prior -> transition_head producing the predicted next scalar hidden.

        Args:
            hidden:
              - Route-0: [B, obs_dim]                    pooled current hidden
              - Route-B: [B, n_img_tok, in_channels]     per-image-token current hidden
            action: [B, action_dim] or [B, H, action_dim] chunk (averaged)

        Returns:
            predicted next scalar hidden [B, obs_dim] (post-conv-stem under route-B).
            Also stashes post_stoch_step on the returned tensor via attribute.
        """
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype
        action = action.to(device=device, dtype=dtype)
        if action.ndim == 3:
            action = action.mean(dim=1)

        if self.io_mode == "token":
            # hidden is [B, N_img] long BPE ids → map, embed, stem
            bpe = hidden.to(device=device, dtype=torch.long)
            img_idx = self._bpe_to_img_idx[bpe]
            if (img_idx < 0).any():
                raise ValueError("predict_next_hidden (token mode): input has non-image BPE ids")
            per_token = self.token_embedder(img_idx)
            hidden = self.conv_stem(per_token)                          # [B, obs_dim]
        else:
            hidden = hidden.to(device=device, dtype=dtype)
            if self.spatial_codec and hidden.ndim == 3:
                hidden = self.conv_stem(hidden)                         # [B, obs_dim]
            elif self.spatial_codec and hidden.ndim != 2:
                raise ValueError(
                    "predict_next_hidden under spatial_codec expects [B, obs_dim] or "
                    f"[B, N_img, C_in]; got {tuple(hidden.shape)}"
                )

        stats = self.obs_to_stoch(hidden)                             # [B, 2*latent]
        _, _, post_stoch = self._stats_to_dist(stats)                  # [B, latent]

        token_input = torch.cat([post_stoch, action], dim=-1).unsqueeze(1)  # [B, 1, L+A]
        token_seq = self.act_stoch_emb(token_input)                    # [B, 1, d_model]
        h_seq = self.causal_transformer(token_seq)                     # [B, 1, d_model]

        prior_stats = self.prior_head(h_seq)
        _, _, prior_stoch = self._stats_to_dist(prior_stats)           # [B, 1, latent]

        post_feature = torch.cat([h_seq, prior_stoch], dim=-1)         # [B, 1, d_model+latent]
        pred = self.transition_head(post_feature).squeeze(1)           # [B, obs_dim]
        # Remember the step's prior stoch so callers that need it for the
        # conv deconv decoder can fetch it without rerunning the RSSM.
        self._last_predicted_stoch = prior_stoch.squeeze(1).detach()
        return pred

    @torch.no_grad()
    def decode_pooled_to_image_hiddens(
        self, pooled: torch.Tensor, stoch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project a predicted scalar hidden [B, obs_dim] to per-image-token
        hiddens.
          - Route-0 (MLP image_decoder): returns [B, n_image_tokens, obs_dim].
          - Route-B (conv deconv):       returns [B, n_image_tokens, in_channels]
            (4096-d, directly feedable into frozen lm_head).
        """
        if self.image_decoder is None:
            raise RuntimeError("image_decoder is not enabled on this WM.")
        if self.spatial_codec:
            assert isinstance(self.image_decoder, BspaceConvDecoderHead)
            first_param = next(self.image_decoder.parameters())
            x = pooled.to(device=first_param.device, dtype=first_param.dtype)
            # stoch required for route-B; prefer caller-supplied, else use
            # the value stashed by predict_next_hidden, else zero.
            if stoch is None:
                stash = getattr(self, "_last_predicted_stoch", None)
                if stash is None:
                    stoch_tensor = torch.zeros(
                        x.shape[0], self.latent_dim,
                        device=first_param.device, dtype=first_param.dtype,
                    )
                else:
                    stoch_tensor = stash
            else:
                stoch_tensor = stoch
            stoch_tensor = stoch_tensor.to(device=first_param.device, dtype=first_param.dtype)
            return self.image_decoder(x, stoch_tensor)                 # [B, N_img, in_channels]
        first_param = next(self.image_decoder.parameters())
        x = pooled.to(device=first_param.device, dtype=first_param.dtype)
        decoded = self.image_decoder(x)
        return decoded.view(-1, self.n_image_tokens, self.obs_dim)

    @torch.no_grad()
    def predict_next_image_token_ids(
        self,
        cur_bpe_ids: torch.Tensor,   # [B, N_img]  current-frame BPE ids
        action: torch.Tensor,         # [B, action_dim] or [B, H, action_dim]
    ) -> torch.Tensor:
        """Token-mode single-step viz helper.  Runs the full TSSM forward on
        the current frame, decodes to image-vocab logits, argmaxes, and maps
        back to raw BPE ids (what the VQGAN expects).

        Returns: [B, N_img] long BPE ids.
        """
        if self.io_mode != "token":
            raise RuntimeError("predict_next_image_token_ids requires io_mode='token'")
        if self.image_decoder is None or self.token_embedder is None:
            raise RuntimeError("token-mode WM is missing image_decoder / token_embedder")

        pred_obs = self.predict_next_hidden(cur_bpe_ids, action)          # [B, obs_dim]
        stoch = getattr(self, "_last_predicted_stoch", None)
        if stoch is None:
            raise RuntimeError("predict_next_hidden did not stash _last_predicted_stoch")

        dec_param = next(self.image_decoder.parameters())
        logits = self.image_decoder(
            pred_obs.to(device=dec_param.device, dtype=dec_param.dtype),
            stoch.to(device=dec_param.device, dtype=dec_param.dtype),
        )  # [B, N_img, num_image_tokens_vocab]
        img_idx = logits.argmax(dim=-1)                                    # [B, N_img]
        return self.image_token_bpe_ids[img_idx]                           # [B, N_img] BPE ids
