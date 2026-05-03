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


@dataclass
class TransDreamerLatentState:
    """Single-step state for TSSMWorldModelTransDreamer used by Dreamer-style
    actor-critic imagination loops. `feature() = cat(stoch, h)` follows the
    usual Dreamer convention: stochastic latent first, deterministic history
    second."""
    mean: torch.Tensor      # [B, latent_dim]
    std: torch.Tensor       # [B, latent_dim]
    stoch: torch.Tensor     # [B, latent_dim]
    h: torch.Tensor         # [B, d_model] — zero on the first frame

    def feature(self) -> torch.Tensor:
        return torch.cat([self.stoch, self.h], dim=-1)


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
        pretrained_model_path: str = "/home/user01/liops/workspace/DreamerVLA/data/ckpts/Action_World_model_512/libero_10",
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

        # Step 1 – posterior z_0 from q(z_0 | h_0=0, o_0)
        posterior_0 = posterior_head(cat(h_0, hidden_0))

        # Step 2 – for t>=1, first infer h_t from history, then compute both
        # prior and posterior under the same h_t:
        token_seq  = act_stoch_emb(cat(z_{0:T-2}, a_{0:T-2}))  # [B, T-1, d_model]
        h_seq      = CausalTransformer(token_seq)                # [B, T-1, d_model]
        prior_seq  = prior_head(h_seq)                          # p(z_t | h_t)
        posterior_seq = posterior_head(cat(h_seq, hidden_{1:T})) # q(z_t | h_t,o_t)

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
        image_loss_enabled: bool = True,
        freeze_image_decoder: bool = False,
        image_decoder_detach_inputs: bool = False,
        image_decoder_detach_mode: str | None = None,
        image_decoder_stoch_source: str = "post",
        # Ablation: zero out the deter (h) input to the image_decoder, forcing
        # reconstruction to depend solely on the stochastic latent z.  Used to
        # diagnose whether the decoder is bypassing z (h-leakage from action
        # history) or whether the posterior actually fails to encode obs.
        image_decoder_zero_h: bool = False,
        # In-run VAE warmup: for the first `warmup_vae_steps` global steps the
        # WM behaves as a pure encoder→posterior→decoder autoencoder:
        #   * KL loss is forced to 0 (prior path receives no gradient),
        #   * decoder is forced to use only z (h is zeroed), so the decoder
        #     cannot bypass the latent.
        # After this many steps, kl_loss_coef and image_decoder_zero_h revert
        # to their config-specified values and the dynamics start training
        # against an already-meaningful posterior.  0 disables the phase.
        warmup_vae_steps: int = 0,
        # When False, posterior is q(z|o) (TransDreamer paper); when True
        # (default, RSSM-style) posterior is q(z|h, o).
        posterior_uses_h: bool = True,
        dynamic_token_coef: float = 0.0,
        delta_latent_coef: float = 0.0,
        # delta_latent loss type: "mse" (legacy), "normalized_mse" (mse / mean(delta_gt^2)),
        # "cosine" (1 - cos_sim). Use the last two when transition_loss already
        # captures absolute scale and you want a scale-free dynamics signal.
        delta_latent_loss_type: str = "mse",
        action_margin_coef: float = 0.0,
        action_margin: float = 0.1,
        action_margin_every: int = 1,
        # Action ranking: predict next obs hidden under real vs shuffled action,
        # require real to be closer (margin loss). Heavier than action_margin
        # (runs causal Transformer twice per step) but a stronger signal.
        action_ranking_coef: float = 0.0,
        action_ranking_margin: float = 0.1,
        # Continuation predictor (pcont, DreamerV3/TransDreamer).  BCE on a
        # "is this transition non-terminal?" signal derived from sparse reward
        # (reward > pcont_done_threshold counts as terminal). Off by default.
        pcont_loss_coef: float = 0.0,
        pcont_hidden_dim: int = 256,
        pcont_done_threshold: float = 0.5,
        # ── Spatial codec (strided-conv encoder + bspace deconv) ─────────────
        # Always True. Kept as kwarg for config compatibility; setting False
        # raises. Pre-spatial-codec MLP image decoder path was removed.
        spatial_codec: bool = True,
        obs_dim: int | None = None,
        in_channels: int = 4096,              # raw per-token hidden size
        spatial_grid: tuple[int, int] = (16, 16),
        stem_init_proj_channels: int = 384,
        stem_stage_channels: tuple[int, ...] = (96, 192),
        stem_kernel: int = 4,
        stem_stride: int = 2,
        stem_padding: int = 1,
        stem_post_norm: bool = False,         # append LayerNorm after conv_stem proj
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
        # ── Imagination rollout loss ────────────────────────────────────────
        # Unrolls the prior `imagine_loss_steps` steps using its own sampled z
        # (no teacher forcing), then computes image-token CE against the GT
        # next frames. Trains the prior to actually predict multi-step futures
        # rather than only single-step transitions. Disabled when
        # imagine_loss_steps == 0.
        #   imagine_loss_steps:    H — number of rollout steps to imagine
        #   imagine_loss_context:  ctx — number of warm-up post steps to seed
        #                          the rollout (>=1; the warm prefix is fed
        #                          teacher-forced from posterior z)
        #   imagine_loss_scale:    multiplier on the imagined CE before adding
        #                          to total loss
        imagine_loss_steps: int = 0,
        imagine_loss_context: int = 1,
        imagine_loss_scale: float = 1.0,
        # ── Backbone variant (only used when use_pretrained_backbone=False) ──
        # "orig"          → CausalTransformerCell (no pos embed)        — legacy default
        # "v2"            → CausalTransformerCellV2 (abs pos + opt gate) — minimal upgrade
        # "v2_gated"      → CausalTransformerCellV2 with use_gru_gate=True
        # "transdreamer"  → TransDreamerTransformerCell (rel-pos + GRU)  — faithful
        transformer_variant: str = "orig",
        transformer_max_seq_len: int = 64,
        transformer_max_rel_pos: int = 64,
    ) -> None:
        super().__init__()
        if not bool(spatial_codec):
            raise ValueError(
                "spatial_codec=False is no longer supported; the legacy MLP "
                "image-decoder path (Route-0) was removed."
            )
        self.spatial_codec = True
        self.in_channels = int(in_channels)
        self.spatial_grid = (int(spatial_grid[0]), int(spatial_grid[1]))
        # `obs_dim`: WM's scalar hidden size, post conv stem.
        self.obs_dim = 1024 if obs_dim is None else int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.min_std = float(min_std)
        self.free_nats = float(free_nats)
        self.kl_balance = float(kl_balance)
        self.kl_loss_coef = float(kl_loss_coef)
        self.transition_loss_coef = float(transition_loss_coef)
        self.reward_loss_coef = float(reward_loss_coef)
        # spatial_codec is always True, so the conv image_decoder is always built.
        self.image_decoder_enabled = True
        self.n_image_tokens = int(n_image_tokens)
        self.image_decoder_loss_coef = float(image_decoder_loss_coef)
        self.image_loss_enabled = bool(image_loss_enabled)
        self.freeze_image_decoder = bool(freeze_image_decoder)
        self.image_decoder_detach_inputs = bool(image_decoder_detach_inputs)
        if image_decoder_detach_mode is None:
            image_decoder_detach_mode = "all" if self.image_decoder_detach_inputs else "none"
        self.image_decoder_detach_mode = str(image_decoder_detach_mode).lower()
        valid_detach_modes = {"none", "all", "stoch", "h"}
        if self.image_decoder_detach_mode not in valid_detach_modes:
            raise ValueError(
                "image_decoder_detach_mode must be one of "
                f"{sorted(valid_detach_modes)}, got {image_decoder_detach_mode!r}"
            )
        self.image_decoder_stoch_source = str(image_decoder_stoch_source).lower()
        self.image_decoder_zero_h = bool(image_decoder_zero_h)
        self.warmup_vae_steps = int(warmup_vae_steps)
        valid_stoch_sources = {"post", "prior"}
        if self.image_decoder_stoch_source not in valid_stoch_sources:
            raise ValueError(
                "image_decoder_stoch_source must be one of "
                f"{sorted(valid_stoch_sources)}, got {image_decoder_stoch_source!r}"
            )
        self.posterior_uses_h = bool(posterior_uses_h)
        self.image_recon_ce_coef = float(image_recon_ce_coef)
        self.image_recon_mse_coef = float(image_recon_mse_coef)
        self.dynamic_token_coef = float(dynamic_token_coef)
        self.delta_latent_coef = float(delta_latent_coef)
        self.delta_latent_loss_type = str(delta_latent_loss_type).lower()
        if self.delta_latent_loss_type not in {"mse", "normalized_mse", "cosine"}:
            raise ValueError(
                "delta_latent_loss_type must be one of "
                f"{{'mse','normalized_mse','cosine'}}; got {delta_latent_loss_type!r}"
            )
        self.action_margin_coef = float(action_margin_coef)
        self.action_margin = float(action_margin)
        self.action_margin_every = max(int(action_margin_every), 1)
        self.action_ranking_coef = float(action_ranking_coef)
        self.action_ranking_margin = float(action_ranking_margin)
        self.pcont_loss_coef = float(pcont_loss_coef)
        self.pcont_done_threshold = float(pcont_done_threshold)
        # ── Observation-only encoder kept for checkpoint compatibility and
        # legacy diagnostics.  Orthodox Dreamer training below uses
        # posterior_head(h_t, o_t), so prior/posterior share the same h_t.
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
            variant = str(transformer_variant).lower()
            if variant == "orig":
                self.causal_transformer = CausalTransformerCell(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                )
            elif variant in ("v2", "v2_gated"):
                from .causal_transformer_v2 import CausalTransformerCellV2
                self.causal_transformer = CausalTransformerCellV2(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    max_seq_len=int(transformer_max_seq_len),
                    use_gru_gate=(variant == "v2_gated"),
                )
            elif variant == "transdreamer":
                from .transdreamer_transformer import TransDreamerTransformerCell
                self.causal_transformer = TransDreamerTransformerCell(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    max_rel_pos=int(transformer_max_rel_pos),
                )
            else:
                raise ValueError(
                    f"transformer_variant must be one of "
                    f"('orig','v2','v2_gated','transdreamer'); got {transformer_variant!r}"
                )

        # ── Prior head: p(z_t | h_t) ────────────────────────────────────────
        # TransDreamer: prior_stoch_mlp  (modules_transformer.py:299)
        self.prior_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mapper_hidden_dim),
            nn.GELU(),
            nn.Linear(mapper_hidden_dim, 2 * self.latent_dim),
        )

        # ── Posterior head: q(z_t | h_t, o_t) ───────────────────────────────
        # This is the orthodox Dreamer/RSSM alignment: prior and posterior for
        # z_t are compared under the same deterministic history h_t.
        self.posterior_head = nn.Sequential(
            nn.LayerNorm(d_model + self.obs_dim),
            nn.Linear(d_model + self.obs_dim, mapper_hidden_dim),
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

        # ── Continuation (pcont) head: predicts "1 - done" logit ────────────
        # Always built so flipping pcont_loss_coef on at runtime works without
        # rebuilding the model.
        self.pcont_head = nn.Sequential(
            nn.LayerNorm(d_model + self.latent_dim),
            nn.Linear(d_model + self.latent_dim, pcont_hidden_dim),
            nn.GELU(),
            nn.Linear(pcont_hidden_dim, 1),
        )

        # ── Spatial codec: strided-conv encoder stem + bspace conv decoder ──
        self.conv_stem = ConvEncoderStem(
            in_channels=self.in_channels,
            spatial=self.spatial_grid,
            obs_dim=self.obs_dim,
            init_proj_channels=stem_init_proj_channels,
            stage_channels=tuple(stem_stage_channels),
            kernel=stem_kernel, stride=stem_stride, padding=stem_padding,
            post_norm=stem_post_norm,
        )
        self.image_decoder: nn.Module | None = BspaceConvDecoderHead(
            deter_dim=self.d_model,
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

        # ── io_mode="token": override stem + decoder for discrete-token I/O ──
        # TODO(unused): io_mode="hidden" is the legacy path that decodes to
        # 4096-d per-token hiddens and routes through the frozen LLM lm_head
        # for image-vocab CE.  Currently no active config uses it (ABCD all
        # use io_mode="token").  If re-enabling, also restore the Route-B
        # branch in pretrain_loss (search for `image_decoder_stoch_source`
        # below) and the `attach_lm_head` lm_head plumbing in the workspace.
        self.io_mode = str(io_mode)
        if self.io_mode not in ("hidden", "token"):
            raise ValueError(f"io_mode must be 'hidden' or 'token', got {io_mode!r}")
        self.token_embed_dim = int(token_embed_dim)
        self.num_image_tokens_vocab = (
            int(num_image_tokens_vocab) if num_image_tokens_vocab is not None else None
        )

        # Imagination-rollout knobs (see __init__ docstring above).
        self.imagine_loss_steps = max(int(imagine_loss_steps), 0)
        self.imagine_loss_context = max(int(imagine_loss_context), 1)
        self.imagine_loss_scale = float(imagine_loss_scale)

        if self.io_mode == "token":
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
                post_norm=stem_post_norm,
            )
            # Decoder output = image-vocab logits (no lm_head needed downstream)
            self.image_decoder = BspaceConvDecoderHead(
                deter_dim=self.d_model,
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

        if self.freeze_image_decoder and self.image_decoder is not None:
            for parameter in self.image_decoder.parameters():
                parameter.requires_grad = False

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

    def _prepare_decoder_inputs(
        self, dec_h: torch.Tensor, dec_stoch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Control where image reconstruction gradients are allowed to flow.

        The current token-WM baseline uses mode="stoch": image CE trains the
        action-conditioned deterministic history h_t, but not the posterior
        stochastic state z_t from the target observation.
        """
        mode = self.image_decoder_detach_mode
        if mode == "all":
            return dec_h.detach(), dec_stoch.detach()
        if mode == "stoch":
            return dec_h, dec_stoch.detach()
        if mode == "h":
            return dec_h.detach(), dec_stoch
        return dec_h, dec_stoch

    def _prepare_decoder_feature(self, feature: torch.Tensor) -> torch.Tensor:
        mode = self.image_decoder_detach_mode
        return feature.detach() if mode == "all" else feature

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
        Analytic KL(N(post) || N(prior)), per latent dim.
        Shape of inputs: [B, T, latent_dim]  →  output [B, T, latent_dim].
        Caller is responsible for free_nats clamp (per-dim) and reduction.
        """
        post_var  = post_std.pow(2)
        prior_var = prior_std.pow(2)
        log_ratio = torch.log(prior_std) - torch.log(post_std)
        sq_term   = (post_var + (post_mean - prior_mean).pow(2)) / (2.0 * prior_var.clamp_min(1e-6))
        kl = log_ratio + sq_term - 0.5            # [B, T, latent_dim]
        return kl                                  # [B, T, latent_dim] per-dim

    @torch.no_grad()
    def _compute_collapse_metrics(
        self, post_mean: torch.Tensor, prior_mean: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Per-step collapse-tracking metrics.  Computed on detached inputs.

        Returns batch-axis diversity (eff_rank, pairwise_cos) for both
        posterior and prior latent params, plus categorical entropy / max-prob
        when the WM is discrete-stoch.
        """
        out: dict[str, torch.Tensor] = {}

        for name, x in (("post", post_mean), ("prior", prior_mean)):
            flat = x.reshape(-1, x.shape[-1]).float()
            N = flat.shape[0]
            if N < 2:
                out[f"{name}_eff_rank"]               = flat.new_zeros(())
                out[f"{name}_pairwise_cos"]           = flat.new_zeros(())
                out[f"{name}_logits_std_across_batch"] = flat.new_zeros(())
                out[f"{name}_logits_mean_abs"]         = flat.new_zeros(())
                continue
            # Pairwise cosine on raw vectors
            norms = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            unit  = flat / norms
            cos_m = unit @ unit.T
            triu  = torch.triu_indices(N, N, offset=1, device=flat.device)
            out[f"{name}_pairwise_cos"] = cos_m[triu[0], triu[1]].mean()
            # Per-feature std across the (batch, time) population: how much do
            # logits actually vary across samples?  ≈0 ⇒ posterior_head output
            # is the same for every input.
            out[f"{name}_logits_std_across_batch"] = flat.std(dim=0).mean()
            # Magnitude of logits: ≈0 ⇒ uniform softmax (no class is preferred).
            out[f"{name}_logits_mean_abs"] = flat.abs().mean()
            # Effective rank from centered SVD spectrum
            centered = flat - flat.mean(dim=0, keepdim=True)
            try:
                s2 = torch.linalg.svdvals(centered).pow(2)
                p  = (s2 / s2.sum().clamp_min(1e-12)).clamp_min(1e-12)
                out[f"{name}_eff_rank"] = torch.exp(-(p * p.log()).sum())
            except Exception:
                out[f"{name}_eff_rank"] = flat.new_zeros(())

        # Discrete-only: per-dim categorical entropy + max-prob (peakedness).
        if hasattr(self, "stoch_categories"):
            S, K = self.stoch_dims, self.stoch_categories
            for name, logits_flat in (("post", post_mean), ("prior", prior_mean)):
                logits = logits_flat.reshape(*logits_flat.shape[:-1], S, K)
                log_p  = F.log_softmax(logits, dim=-1)
                p      = log_p.exp()
                ent    = -(p * log_p).sum(dim=-1)              # [..., S]
                out[f"{name}_entropy"]      = ent.mean()        # mean per-dim entropy (nats)
                out[f"{name}_max_prob"]     = p.max(dim=-1).values.mean()
                # std doesn't apply to categorical; expose 1 - max_prob as
                # rough "spread" surrogate so downstream logging keys exist.
                out[f"{name}_std_mean"]     = 1.0 - out[f"{name}_max_prob"]
        return out

    # ── Posterior / prior sequence inference ────────────────────────────────

    def _posterior_from_obs_h(
        self,
        hidden: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posterior q(z_t | ·).

        Two modes selected by `self.posterior_uses_h`:
          * True  (default, RSSM/Dreamer): q(z_t | h_t, o_t)  via posterior_head
          * False (TransDreamer paper):    q(z_t | o_t)       via obs_to_stoch
        """
        if self.posterior_uses_h:
            stats = self.posterior_head(torch.cat([h, hidden], dim=-1))
        else:
            stats = self.obs_to_stoch(hidden)
        return self._stats_to_dist(stats)

    def _encode_posterior_seq(
        self,
        hidden_seq: torch.Tensor,
        h_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Posterior encoder.

        If h_seq is supplied AND `self.posterior_uses_h`, runs the orthodox
        Dreamer q(z_t | h_t, o_t) via posterior_head. Otherwise (h_seq=None
        or posterior_uses_h=False) runs the TransDreamer-paper-style
        observation-only posterior via obs_to_stoch.

        Args:
            hidden_seq: [B, T, obs_dim]
            h_seq:      [B, T, d_model] or None
        Returns:
            mean_seq, std_seq, stoch_seq: each [B, T, latent_dim]
        """
        if h_seq is None or not self.posterior_uses_h:
            stats = self.obs_to_stoch(hidden_seq)      # [B, T, 2*latent_dim]
        else:
            stats = self.posterior_head(
                torch.cat([h_seq, hidden_seq], dim=-1)
            )                                          # [B, T, 2*latent_dim]
        return self._stats_to_dist(stats)

    # ── Prior: causal Transformer over history ───────────────────────────────

    def _infer_prior_seq(
        self,
        stoch_seq: torch.Tensor,    # [B, K, latent_dim]  z_{0:K-1}
        action_seq: torch.Tensor,   # [B, K, action_dim]  actions causing z_{1:K}
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        p(z_t | z_{<t}, a_{1:t}) for t = 1..K.

        TransDreamer: infer_prior_stoch(prev_stoch, temp, actions)  (modules_transformer.py:409)

        Pipeline:
            token_i = act_stoch_emb(cat(z_i, a_{i+1})) → [B, K, d_model]
            h_seq   = CausalTransformer(token_seq)     → [B, K, d_model]
                      h_seq[:, i] encodes history up to z_i/action_{i+1}
                      → used as prior for z_{i+1}
            prior_t = prior_head(h_seq)                → [B, K, 2*latent_dim]

        Returns:
            prior_mean, prior_std, prior_stoch: [B, K, latent_dim]
            h_seq:                              [B, K, d_model]
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

    def _infer_dreamer_seq(
        self,
        hidden_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        """Infer posterior/prior in the orthodox Dreamer order.

        For t=0, h_0 is zero and we infer q(z_0 | h_0, o_0). For t>=1:

            h_t       = f(z_{<t}, a_{1:t})
            prior_t   = p(z_t | h_t)
            posterior = q(z_t | h_t, o_t)

        The same h_t is therefore used for both KL sides and downstream
        feature heads.

        Two execution paths:
          * `posterior_uses_h=False` (TransDreamer-paper style):
              post is independent of h, so all T posteriors are computed in
              one batched forward, and a single Transformer call produces all
              T-1 priors. Matches modules_transformer.py:302-326.
          * `posterior_uses_h=True` (orthodox Dreamer/RSSM):
              post depends on h, forcing the iterative path: T-1 sequential
              Transformer calls of growing prefix length.
        """
        B, T, _ = hidden_seq.shape
        if T < 2:
            raise ValueError("Dreamer sequence loss requires T >= 2")

        # ── Fast path: post = q(z|o), parallelisable (TransDreamer style) ─
        if not self.posterior_uses_h:
            # All T posteriors in a single MLP call.
            post_mean, post_std, post_stoch = self._encode_posterior_seq(
                hidden_seq, h_seq=None
            )                                                         # [B, T, latent]
            # All T-1 priors in a single Transformer call.
            prior_mean, prior_std, prior_stoch, h_seq = self._infer_prior_seq(
                stoch_seq=post_stoch[:, :-1],                          # [B, T-1, latent]
                action_seq=action_seq[:, 1:],                          # [B, T-1, action]
            )
            return (
                post_mean, post_std, post_stoch,
                prior_mean, prior_std, prior_stoch,
                h_seq,
            )

        # ── Slow path: post depends on h, must iterate ───────────────────
        device, dtype = hidden_seq.device, hidden_seq.dtype
        h0 = torch.zeros(B, self.d_model, device=device, dtype=dtype)

        post_mean_list: list[torch.Tensor] = []
        post_std_list: list[torch.Tensor] = []
        post_stoch_list: list[torch.Tensor] = []
        prior_mean_list: list[torch.Tensor] = []
        prior_std_list: list[torch.Tensor] = []
        prior_stoch_list: list[torch.Tensor] = []
        h_list: list[torch.Tensor] = []

        mean_t, std_t, stoch_t = self._posterior_from_obs_h(hidden_seq[:, 0], h0)
        post_mean_list.append(mean_t)
        post_std_list.append(std_t)
        post_stoch_list.append(stoch_t)

        for t in range(1, T):
            prefix_stoch = torch.stack(post_stoch_list, dim=1)       # [B, t, latent]
            prefix_action = action_seq[:, 1 : t + 1]                 # [B, t, action]
            prior_mean_seq, prior_std_seq, prior_stoch_seq, h_seq = self._infer_prior_seq(
                stoch_seq=prefix_stoch,
                action_seq=prefix_action,
            )

            h_t = h_seq[:, -1]
            prior_mean_list.append(prior_mean_seq[:, -1])
            prior_std_list.append(prior_std_seq[:, -1])
            prior_stoch_list.append(prior_stoch_seq[:, -1])
            h_list.append(h_t)

            mean_t, std_t, stoch_t = self._posterior_from_obs_h(hidden_seq[:, t], h_t)
            post_mean_list.append(mean_t)
            post_std_list.append(std_t)
            post_stoch_list.append(stoch_t)

        post_mean = torch.stack(post_mean_list, dim=1)               # [B, T, latent]
        post_std = torch.stack(post_std_list, dim=1)
        post_stoch = torch.stack(post_stoch_list, dim=1)
        prior_mean = torch.stack(prior_mean_list, dim=1)             # [B, T-1, latent]
        prior_std = torch.stack(prior_std_list, dim=1)
        prior_stoch = torch.stack(prior_stoch_list, dim=1)
        h_seq = torch.stack(h_list, dim=1)                           # [B, T-1, d_model]
        return (
            post_mean, post_std, post_stoch,
            prior_mean, prior_std, prior_stoch,
            h_seq,
        )

    # ── Imagination rollout (no teacher forcing) ─────────────────────────────

    def _imagine_rollout_loss(
        self,
        post_stoch_warm: torch.Tensor,        # [B, ctx, latent]
        action_seq_full: torch.Tensor,         # [B, T, action_dim]
        raw_bpe_ids_seq: torch.Tensor,         # [B, T, N_img] long
        ctx: int,
        H: int,
    ) -> dict[str, torch.Tensor]:
        """Sequential imagination rollout + image-token CE loss.

        Starts from `post_stoch_warm` (length ctx, taken from the posterior),
        unrolls the prior H steps using its own *sampled* z each step (no
        teacher forcing), then decodes (h_imagine, z_prior) at every imagined
        step and scores image-token CE against the GT next frames at positions
        ctx..ctx+H-1.

        Notes:
        * The first imagined step is degenerate: it uses (z_{ctx-1}^post,
          a_ctx) and is therefore identical to the teacher-forced prior at
          that position.  Real exposure-bias signal kicks in at imagined step
          >= 1, where the prefix contains z_{ctx}^prior sampled from its own
          prior.
        * Each step re-runs the causal Transformer on the full prefix
          (length ctx+s) since the backbone has no KV cache.  Cost is
          O(H * (ctx+H)^2) attention; keep H modest.
        """
        if self.io_mode != "token":
            raise RuntimeError("imagine rollout loss currently requires io_mode='token'")
        if self.image_decoder is None:
            raise RuntimeError("imagine rollout loss requires image_decoder")

        device = post_stoch_warm.device
        B = post_stoch_warm.shape[0]

        # Accumulating list of z_t for the prior input prefix.
        z_history: list[torch.Tensor] = [post_stoch_warm[:, t] for t in range(ctx)]
        h_steps: list[torch.Tensor] = []
        z_steps: list[torch.Tensor] = []

        for step in range(H):
            z_prefix = torch.stack(z_history, dim=1)                    # [B, ctx+step, latent]
            action_prefix = action_seq_full[:, 1 : ctx + step + 1]      # [B, ctx+step, A]
            _pm, _ps, prior_stoch_step, h_seq_step = self._infer_prior_seq(
                z_prefix, action_prefix,
            )
            h_t = h_seq_step[:, -1]                                      # [B, d_model]
            z_t = prior_stoch_step[:, -1]                                # [B, latent]
            h_steps.append(h_t)
            z_steps.append(z_t)
            z_history.append(z_t)

        h_imagine_seq = torch.stack(h_steps, dim=1)                      # [B, H, d_model]
        z_imagine_seq = torch.stack(z_steps, dim=1)                      # [B, H, latent]

        dec_h, dec_stoch = self._prepare_decoder_inputs(h_imagine_seq, z_imagine_seq)
        logits = self.image_decoder(dec_h, dec_stoch)                    # [B, H, N_img, V]

        tgt_bpe = raw_bpe_ids_seq[:, ctx : ctx + H]                      # [B, H, N_img]
        prev_bpe = raw_bpe_ids_seq[:, ctx - 1 : ctx + H - 1]
        img_idx = self._bpe_to_img_idx[tgt_bpe]
        prev_idx = self._bpe_to_img_idx[prev_bpe]
        if (img_idx < 0).any():
            raise ValueError("imagine_rollout_loss: target contains non-image BPE ids")

        ce_per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            img_idx.reshape(-1),
            reduction="none",
        ).view_as(img_idx)
        dynamic_mask = img_idx != prev_idx
        static_mask = ~dynamic_mask
        ce_mean = ce_per_token.mean()
        static_ce = (
            ce_per_token[static_mask].mean() if static_mask.any() else ce_mean.new_zeros(())
        )
        dynamic_ce = (
            ce_per_token[dynamic_mask].mean() if dynamic_mask.any() else ce_mean.new_zeros(())
        )

        if self.dynamic_token_coef > 0:
            weighted = self.image_recon_ce_coef * (
                static_ce + self.dynamic_token_coef * dynamic_ce
            )
        else:
            weighted = self.image_recon_ce_coef * ce_mean

        with torch.no_grad():
            pred_idx = logits.argmax(dim=-1)
            recon_acc = (pred_idx == img_idx).float().mean()
            sta_acc = (
                (pred_idx[static_mask] == img_idx[static_mask]).float().mean()
                if static_mask.any()
                else recon_acc.new_zeros(())
            )
            dyn_acc = (
                (pred_idx[dynamic_mask] == img_idx[dynamic_mask]).float().mean()
                if dynamic_mask.any()
                else recon_acc.new_zeros(())
            )
            dyn_frac = dynamic_mask.float().mean()

        return {
            "imagine_ce_loss": ce_mean,
            "imagine_static_ce_loss": static_ce,
            "imagine_dynamic_ce_loss": dynamic_ce,
            "imagine_loss_weighted": weighted,
            "imagine_recon_accuracy": recon_acc,
            "imagine_static_accuracy": sta_acc,
            "imagine_dynamic_accuracy": dyn_acc,
            "imagine_dynamic_fraction": dyn_frac,
        }

    # ── Loss computation ─────────────────────────────────────────────────────

    def pretrain_loss(
        self,
        hidden_seq: torch.Tensor,
        action_seq: torch.Tensor,          # [B, T, action_dim]
        reward_seq: torch.Tensor | None = None,  # [B, T]
        done_seq:   torch.Tensor | None = None,  # [B, T]
        next_image_hiddens_target: torch.Tensor | None = None,  # [B, T-1, n_img_tok, in_channels]
        next_image_token_ids_target: torch.Tensor | None = None,  # [B, T-1, n_img_tok] bpe ids
        global_step: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        TransDreamer-style sequence loss.

        `hidden_seq` accepted shapes:
          - Route-0: [B, T, obs_dim]               — already scalar per frame
          - Route-B: [B, T, n_img_tok, in_channels] — per-image-token, then
            conv_stem compresses to [B, T, obs_dim] internally.

        Dreamer/RSSM order:
            h_t     ← dynamics(z_{<t}, a_{1:t})
            prior   ← p(z_t | h_t)
            post    ← q(z_t | h_t, obs_t)
            KL compares prior/posterior under the same h_t.
        """
        # In-run VAE warmup: when global_step < warmup_vae_steps, force pure
        # autoencoder behaviour (KL=0, decoder sees zero-h).  Falls through to
        # config-specified values once warmup ends.
        in_vae_warmup = (
            self.warmup_vae_steps > 0
            and global_step is not None
            and global_step < self.warmup_vae_steps
        )
        eff_kl_loss_coef = 0.0 if in_vae_warmup else self.kl_loss_coef
        eff_zero_h = self.image_decoder_zero_h or in_vae_warmup
        # During warmup, also gate reward + pcont losses to 0.  Their targets
        # in libero (reward≈0, done=False) are near-constant, so reward_head
        # and pcont_head gradients on z (via the [h,z] feature) reduce to
        # "make z not affect the constant prediction" → push z to a constant
        # value, collapsing the posterior.  recon CE under gumbel-ST has
        # sparse per-step gradient on z (only 1/K categories per dim get
        # signal), so the dense collapse pull from reward/pcont can dominate.
        eff_reward_loss_coef = 0.0 if in_vae_warmup else self.reward_loss_coef
        eff_pcont_loss_coef = 0.0 if in_vae_warmup else self.pcont_loss_coef

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
        else:
            # io_mode="hidden": spatial_codec is always True, so hidden_seq must
            # be the per-image-token tensor [B, T, N_img, C_in].
            if hidden_seq.ndim != 4:
                raise ValueError(
                    "hidden_seq must have shape [B, T, N_img, C_in]; "
                    f"got {tuple(hidden_seq.shape)}"
                )
            per_token_hidden_seq = hidden_seq
            hidden_seq = self.conv_stem(hidden_seq)            # [B, T, obs_dim]

        B, T, _ = hidden_seq.shape

        # ── Step 1/2: Dreamer inference order ───────────────────────────────
        # action_seq[t] is the action that caused hidden_seq[t] (arrived-at
        # convention), so z_{t-1} pairs with action_seq[t] to produce h_t.
        if in_vae_warmup:
            # Skip the causal-Transformer (`_infer_prior_seq`) and prior_head
            # entirely: during warmup, h is unused (decoder zeros it, reward/
            # pcont/transition heads are gated off, KL coef is 0), so spending
            # compute to produce h is pure waste.  Replace prior + h with
            # zero placeholders matching the post-warmup shapes; downstream
            # KL/transition/reward code receives values it then ignores or
            # multiplies by an effective 0 coefficient.
            post_mean, post_std, post_stoch = self._encode_posterior_seq(
                hidden_seq, h_seq=None
            )                                                       # [B, T, latent]
            prior_mean = torch.zeros_like(post_mean[:, 1:])
            prior_std = torch.ones_like(post_std[:, 1:])
            prior_stoch = torch.zeros_like(post_stoch[:, 1:])
            h_seq = post_stoch.new_zeros(B, T - 1, self.d_model)
        else:
            (
                post_mean, post_std, post_stoch,
                prior_mean, prior_std, prior_stoch,
                h_seq,
            ) = self._infer_dreamer_seq(hidden_seq, action_seq)
        # post_*:  [B, T,   latent_dim]
        # prior_*: [B, T-1, latent_dim] for z_{1:T-1}
        # h_seq:   [B, T-1, d_model]    same h_t for prior/posterior z_t

        post_mean_loss = post_mean[:, 1:]
        post_std_loss = post_std[:, 1:]
        post_stoch_loss = post_stoch[:, 1:]
        stoch_anchor_loss = post_stoch[:, :-1]
        prior_mean_loss = prior_mean
        prior_std_loss = prior_std
        prior_stoch_loss = prior_stoch
        h_loss = h_seq
        target_hidden = hidden_seq[:, 1:]
        cur_hidden = hidden_seq[:, :-1]
        action_loss = action_seq[:, 1:]

        # ── Step 3: KL with kl_balance and free_nats ─────────────────────────
        # TransDreamer (modules_transformer.py:126-133):
        #   value_lhs = KL(post_dist, sg(prior_dist))   = KL(post ‖ sg(prior))  → rep
        #   value_rhs = KL(sg(post_dist), prior_dist)   = KL(sg(post) ‖ prior)  → dyn
        #   kl_loss   = (1 - kl_balance) * lhs + kl_balance * rhs
        #             = (1 - kl_balance) * rep + kl_balance * dyn
        # With kl_balance=0.8: 80% weight on dyn (trains the dynamics/prior Transformer)

        # rep: KL(post ‖ sg(prior)) — gradient to posterior encoder only.
        # _gaussian_kl now returns per-latent-dim KL with shape [B, T, S].
        rep_kl_perdim = self._gaussian_kl(
            post_mean  = post_mean_loss,
            post_std   = post_std_loss,
            prior_mean = prior_mean_loss.detach(),
            prior_std  = prior_std_loss.detach(),
        )                                                  # [B, T, S]
        # dyn: KL(sg(post) ‖ prior) — gradient to causal Transformer / prior_head only
        dyn_kl_perdim = self._gaussian_kl(
            post_mean  = post_mean_loss.detach(),
            post_std   = post_std_loss.detach(),
            prior_mean = prior_mean_loss,
            prior_std  = prior_std_loss,
        )                                                  # [B, T, S]

        # Capture raw KL(post ‖ prior) before free_nats clamping (scalar diag).
        kl_post_prior_raw = rep_kl_perdim.detach().sum(dim=-1).mean()

        # free_nats applied PER-DIM (DreamerV3 free_bits semantics): each of
        # the S latent dims is independently clamped to >= free_nats.  This
        # blocks the degenerate "concentrate all KL in one dim, collapse the
        # rest" solution that a per-total clamp permits.
        rep_kl_clamped = torch.clamp(rep_kl_perdim, min=self.free_nats)  # [B, T, S]
        dyn_kl_clamped = torch.clamp(dyn_kl_perdim, min=self.free_nats)

        # Reduce: sum over latent dim, mean over batch+time.
        rep_kl = rep_kl_clamped.sum(dim=-1).mean()        # scalar
        dyn_kl = dyn_kl_clamped.sum(dim=-1).mean()        # scalar

        # Combine: (1-balance)*rep + balance*dyn  (matches TransDreamer exactly)
        kl_loss = (1.0 - self.kl_balance) * rep_kl + self.kl_balance * dyn_kl
        # Per-direction loss values after clamp+balance (no kl_loss_coef yet).
        rep_loss_value = ((1.0 - self.kl_balance) * rep_kl).detach()
        dyn_loss_value = (self.kl_balance * dyn_kl).detach()

        # ── Diagnostic metrics for collapse tracking (no backward) ───────────
        diversity_logs = self._compute_collapse_metrics(
            post_mean_loss.detach(), prior_mean_loss.detach(),
        )

        # ── Step 4: transition loss (reconstruction anchor) ───────────────────
        # TransDreamer uses image reconstruction log_prob for this role.
        # We use MSE on LLM hidden states as the observation reconstruction target.
        # post['deter'] = prior['deter'] in TransDreamer (modules_transformer.py:327):
        #   feature = cat(post_stoch, prior_deter) to anchor the Transformer output.
        # Here we concat h_seq (prior deter) with posterior stoch for the same effect.
        post_feature = torch.cat([h_loss, post_stoch_loss], dim=-1)      # [B, K, d_model+latent]
        if in_vae_warmup or (self.transition_loss_coef <= 0 and self.delta_latent_coef <= 0):
            # transition_head's output is unused when both transition_loss
            # and delta_latent are gated off — skip its forward.  Provide a
            # zero placeholder that downstream `.new_zeros(())` template
            # users (image_decoder_loss, pcont_loss init) can still use.
            predicted_next_hidden = post_stoch_loss
            transition_loss = post_stoch_loss.new_zeros(())
        else:
            predicted_next_hidden = self.transition_head(post_feature)   # [B, K, obs_dim]
            transition_loss = F.mse_loss(predicted_next_hidden, target_hidden.detach())
        delta_latent_loss = predicted_next_hidden.new_zeros(())
        if self.delta_latent_coef > 0:
            cur_feat = cur_hidden
            delta_pred = predicted_next_hidden - cur_feat
            delta_gt = target_hidden.detach() - cur_feat.detach()
            if self.delta_latent_loss_type == "cosine":
                delta_latent_loss = (
                    1.0 - F.cosine_similarity(
                        delta_pred.flatten(start_dim=1),
                        delta_gt.flatten(start_dim=1),
                        dim=-1,
                    ).mean()
                )
            elif self.delta_latent_loss_type == "normalized_mse":
                delta_mse = F.mse_loss(delta_pred, delta_gt)
                delta_norm = delta_gt.pow(2).mean().detach() + 1e-6
                delta_latent_loss = delta_mse / delta_norm
            else:  # "mse"
                delta_latent_loss = F.mse_loss(delta_pred, delta_gt)

        # Action ranking: real action's predicted next-hidden must be closer
        # to the GT next-hidden than a shuffled action's prediction.
        action_ranking_loss = predicted_next_hidden.new_zeros(())
        if self.action_ranking_coef > 0 and hidden_seq.shape[0] > 1:
            shuffled_action = action_loss.roll(shifts=1, dims=0)
            # Re-run causal Transformer + prior_head with the shuffled action.
            _, _, pz_s, h_shuf = self._infer_prior_seq(
                stoch_anchor_loss.detach(), shuffled_action,
            )
            pred_real_next = self.transition_head(
                torch.cat([h_loss, prior_stoch_loss], dim=-1)
            )
            pred_shuf_next = self.transition_head(
                torch.cat([h_shuf, pz_s], dim=-1)
            )
            target_for_rank = target_hidden.detach()
            dist_real = (
                (pred_real_next - target_for_rank).pow(2)
                .flatten(start_dim=2).mean(dim=-1)
            )
            dist_shuf = (
                (pred_shuf_next - target_for_rank).pow(2)
                .flatten(start_dim=2).mean(dim=-1)
            )
            action_ranking_loss = F.relu(
                self.action_ranking_margin + dist_real - dist_shuf
            ).mean()

        action_margin_loss = predicted_next_hidden.new_zeros(())
        action_margin_active = predicted_next_hidden.new_zeros(())
        run_action_margin = (
            self.action_margin_coef > 0
            and hidden_seq.shape[0] > 1
            and (global_step is None or global_step >= 0)
            and (global_step is None or (global_step % self.action_margin_every) == 0)
        )
        if run_action_margin:
            action_margin_active = predicted_next_hidden.new_ones(())
            shuffled_action = action_loss.roll(shifts=1, dims=0)
            # Cheap action contrast: compare only the action-conditioned input
            # embedding instead of running a second Transformer prior. The heavy
            # version called `_infer_prior_seq()` again and was too slow under FSDP.
            stoch_anchor = stoch_anchor_loss.detach()
            real_token = self.act_stoch_emb(
                torch.cat([stoch_anchor, action_loss], dim=-1)
            )
            shuffle_token = self.act_stoch_emb(
                torch.cat([stoch_anchor, shuffled_action], dim=-1)
            )
            margin_dist = (real_token - shuffle_token).flatten(start_dim=-1).norm(dim=-1)
            action_margin_loss = F.relu(self.action_margin - margin_dist).mean()

        # ── Step 5: reward loss (optional) ───────────────────────────────────
        # Skip reward_head forward entirely when its loss is gated off (warmup
        # or coef=0): saves compute and avoids gradient on `post_feature`.
        if eff_reward_loss_coef > 0:
            predicted_reward = self.reward_head(post_feature).squeeze(-1)  # [B, K]
            if reward_seq is not None:
                reward_target = reward_seq[:, 1:].reshape_as(predicted_reward)
            else:
                reward_target = torch.zeros_like(predicted_reward)
            reward_loss = F.mse_loss(predicted_reward, reward_target)
        else:
            reward_loss = predicted_next_hidden.new_zeros(())

        # ── Step 5b: continuation predictor (pcont) ─────────────────────────
        # Predicts "1 - done" from post_feature; trains h to encode end-of-
        # episode signal. Done target derived from sparse terminal reward
        # (LIBERO-style: reward > threshold marks success/done).
        pcont_loss = predicted_next_hidden.new_zeros(())
        pcont_acc = predicted_next_hidden.new_zeros(())
        if eff_pcont_loss_coef > 0:
            pcont_logits = self.pcont_head(post_feature).squeeze(-1)        # [B, K]
            if reward_seq is not None:
                done_target = (reward_seq[:, 1:].reshape_as(pcont_logits) > self.pcont_done_threshold).float()
            else:
                done_target = torch.zeros_like(pcont_logits)
            pcont_target = 1.0 - done_target                                  # 1=continue, 0=done
            pcont_loss = F.binary_cross_entropy_with_logits(pcont_logits, pcont_target)
            with torch.no_grad():
                pred_continue = (pcont_logits > 0).float()
                pcont_acc = (pred_continue == pcont_target).float().mean()

        # ── Step 6: image-decoder loss from RSSM feature ─────────────────────
        # Decode directly from the predicted deterministic history h_t and
        # stochastic state z_t: p(image_t | h_t, z_t).  Keep transition_head
        # separate as the compact hidden-state reconstruction anchor.
        image_decoder_loss = predicted_next_hidden.new_zeros(())
        image_recon_ce_loss = predicted_next_hidden.new_zeros(())
        image_recon_mse_loss = predicted_next_hidden.new_zeros(())
        image_recon_accuracy = predicted_next_hidden.new_zeros(())
        image_static_ce_loss = predicted_next_hidden.new_zeros(())
        image_dynamic_ce_loss = predicted_next_hidden.new_zeros(())
        image_static_accuracy = predicted_next_hidden.new_zeros(())
        image_dynamic_accuracy = predicted_next_hidden.new_zeros(())
        image_dynamic_fraction = predicted_next_hidden.new_zeros(())
        pred_entropy = predicted_next_hidden.new_zeros(())
        uniq_per_sample = predicted_next_hidden.new_zeros(())
        gt_uniq_per_sample = predicted_next_hidden.new_zeros(())

        if self.image_loss_enabled and self.io_mode == "token" and self.image_decoder is not None:
            # Token mode: decoder output IS image-vocab logits; no lm_head.
            dec_h = h_loss
            if self.image_decoder_stoch_source == "prior":
                dec_stoch = prior_stoch_loss
            else:
                dec_stoch = post_stoch_loss
            if eff_zero_h:
                dec_h = torch.zeros_like(dec_h)
            dec_h, dec_stoch = self._prepare_decoder_inputs(dec_h, dec_stoch)
            logits = self.image_decoder(
                dec_h, dec_stoch
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
            ce_per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                img_idx.reshape(-1),
                reduction="none",
            ).view_as(img_idx)
            assert raw_bpe_ids_seq is not None
            prev_img_idx = self._bpe_to_img_idx[raw_bpe_ids_seq[:, :-1]]
            dynamic_mask = img_idx != prev_img_idx
            static_mask = ~dynamic_mask
            image_recon_ce_loss = ce_per_token.mean()
            if static_mask.any():
                image_static_ce_loss = ce_per_token[static_mask].mean()
            if dynamic_mask.any():
                image_dynamic_ce_loss = ce_per_token[dynamic_mask].mean()
            if self.dynamic_token_coef > 0:
                image_decoder_loss = (
                    self.image_recon_ce_coef
                    * (image_static_ce_loss + self.dynamic_token_coef * image_dynamic_ce_loss)
                )
            else:
                image_decoder_loss = self.image_recon_ce_coef * image_recon_ce_loss
            with torch.no_grad():
                pred_idx = logits.argmax(dim=-1)                   # [B, T-1, N_img]
                image_recon_accuracy = (pred_idx == img_idx).float().mean()
                image_dynamic_fraction = dynamic_mask.float().mean()
                if static_mask.any():
                    image_static_accuracy = (pred_idx[static_mask] == img_idx[static_mask]).float().mean()
                if dynamic_mask.any():
                    image_dynamic_accuracy = (pred_idx[dynamic_mask] == img_idx[dynamic_mask]).float().mean()
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
        elif self.image_loss_enabled and self.image_decoder is not None:
            # Route-B: conv deconv decoder → [B, T-1, n_img_tok, in_channels]
            # Inputs: h_seq (deter/history) + configured stochastic source.
            dec_h = h_loss
            if self.image_decoder_stoch_source == "prior":
                dec_stoch = prior_stoch_loss
            else:
                dec_stoch = post_stoch_loss
            if eff_zero_h:
                dec_h = torch.zeros_like(dec_h)
            dec_h, dec_stoch = self._prepare_decoder_inputs(dec_h, dec_stoch)
            decoded = self.image_decoder(
                dec_h, dec_stoch
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
            self.image_loss_enabled
            and
            self.image_decoder is not None
            and next_image_hiddens_target is not None
        ):
            # Route-0 legacy MLP image decoder path.
            Bp, Tm1, _ = post_feature.shape
            decoder_input = self._prepare_decoder_feature(post_feature)
            decoded = self.image_decoder(decoder_input)                    # [B, T-1, n_img_tok*obs_dim]
            decoded = decoded.view(Bp, Tm1, self.n_image_tokens, self.obs_dim)
            target = next_image_hiddens_target
            target = target.to(
                device=decoded.device, dtype=decoded.dtype
            ).detach()
            image_decoder_loss = F.mse_loss(decoded, target)

        # ── Step 7: imagination rollout loss (optional) ──────────────────────
        # Unrolls the prior using its own sampled z (no teacher forcing) and
        # scores image-token CE against GT future frames.  Only active when
        # imagine_loss_steps > 0 and the token decoder is wired up.
        imagine_ce_loss = predicted_next_hidden.new_zeros(())
        imagine_static_ce_loss = predicted_next_hidden.new_zeros(())
        imagine_dynamic_ce_loss = predicted_next_hidden.new_zeros(())
        imagine_loss_weighted = predicted_next_hidden.new_zeros(())
        imagine_recon_accuracy = predicted_next_hidden.new_zeros(())
        imagine_static_accuracy = predicted_next_hidden.new_zeros(())
        imagine_dynamic_accuracy = predicted_next_hidden.new_zeros(())
        imagine_dynamic_fraction = predicted_next_hidden.new_zeros(())
        imagine_actual_steps = 0
        imagine_actual_context = 0

        if (
            self.imagine_loss_steps > 0
            and self.imagine_loss_scale > 0
            and self.io_mode == "token"
            and self.image_decoder is not None
            and raw_bpe_ids_seq is not None
        ):
            ctx_cap = max(min(self.imagine_loss_context, T - 1), 1)
            H_cap = min(self.imagine_loss_steps, T - ctx_cap)
            if H_cap > 0:
                imagine_dict = self._imagine_rollout_loss(
                    post_stoch_warm=post_stoch[:, :ctx_cap],
                    action_seq_full=action_seq,
                    raw_bpe_ids_seq=raw_bpe_ids_seq,
                    ctx=ctx_cap,
                    H=H_cap,
                )
                imagine_ce_loss = imagine_dict["imagine_ce_loss"]
                imagine_static_ce_loss = imagine_dict["imagine_static_ce_loss"]
                imagine_dynamic_ce_loss = imagine_dict["imagine_dynamic_ce_loss"]
                imagine_loss_weighted = imagine_dict["imagine_loss_weighted"]
                imagine_recon_accuracy = imagine_dict["imagine_recon_accuracy"]
                imagine_static_accuracy = imagine_dict["imagine_static_accuracy"]
                imagine_dynamic_accuracy = imagine_dict["imagine_dynamic_accuracy"]
                imagine_dynamic_fraction = imagine_dict["imagine_dynamic_fraction"]
                imagine_actual_steps = H_cap
                imagine_actual_context = ctx_cap

        # ── Total loss ───────────────────────────────────────────────────────
        loss = (
            self.transition_loss_coef * transition_loss
            + eff_kl_loss_coef        * kl_loss
        )
        if eff_reward_loss_coef > 0:
            loss = loss + eff_reward_loss_coef * reward_loss
        if self.delta_latent_coef > 0:
            loss = loss + self.delta_latent_coef * delta_latent_loss
        if self.action_margin_coef > 0:
            loss = loss + self.action_margin_coef * action_margin_loss
        if self.action_ranking_coef > 0:
            loss = loss + self.action_ranking_coef * action_ranking_loss
        if eff_pcont_loss_coef > 0:
            loss = loss + eff_pcont_loss_coef * pcont_loss
        if self.image_decoder is not None and self.image_decoder_loss_coef > 0:
            loss = loss + self.image_decoder_loss_coef * image_decoder_loss
        if imagine_actual_steps > 0:
            loss = loss + self.imagine_loss_scale * imagine_loss_weighted

        return {
            "loss":                loss,
            "kl_loss":             kl_loss,
            "vae_warmup":          loss.new_tensor(1.0 if in_vae_warmup else 0.0),
            "eff_kl_loss_coef":    loss.new_tensor(eff_kl_loss_coef),
            "dyn_kl":              dyn_kl,
            "rep_kl":              rep_kl,
            "kl_post_prior":       kl_post_prior_raw,
            "dyn_loss":            dyn_loss_value,
            "rep_loss":            rep_loss_value,
            **diversity_logs,
            "transition_loss":     transition_loss,
            "reward_loss":         reward_loss,
            "delta_latent_loss":    delta_latent_loss,
            "action_margin_loss":   action_margin_loss,
            "action_margin_active": action_margin_active,
            "action_ranking_loss":  action_ranking_loss,
            "pcont_loss":           pcont_loss,
            "pcont_acc":            pcont_acc,
            "image_recon_ce_loss":  image_recon_ce_loss,
            "image_static_ce_loss": image_static_ce_loss,
            "image_dynamic_ce_loss": image_dynamic_ce_loss,
            "image_recon_mse_loss": image_recon_mse_loss,
            "image_decoder_loss":  image_decoder_loss,
            "image_recon_accuracy": image_recon_accuracy,
            "image_static_accuracy": image_static_accuracy,
            "image_dynamic_accuracy": image_dynamic_accuracy,
            "image_dynamic_fraction": image_dynamic_fraction,
            "pred_entropy":        pred_entropy,
            "pred_unique_tokens":  uniq_per_sample,
            "gt_unique_tokens":    gt_uniq_per_sample,
            "imagine_ce_loss":         imagine_ce_loss,
            "imagine_static_ce_loss":  imagine_static_ce_loss,
            "imagine_dynamic_ce_loss": imagine_dynamic_ce_loss,
            "imagine_loss":            imagine_loss_weighted,
            "imagine_recon_accuracy":  imagine_recon_accuracy,
            "imagine_static_accuracy": imagine_static_accuracy,
            "imagine_dynamic_accuracy": imagine_dynamic_accuracy,
            "imagine_dynamic_fraction": imagine_dynamic_fraction,
            "sequence_loss_steps":     loss.new_tensor(float(imagine_actual_steps)),
            "sequence_context_steps":  loss.new_tensor(float(imagine_actual_context)),
            "sequence_loss_scale":     loss.new_tensor(float(self.imagine_loss_scale)),
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
        raw_global_step = batch.get("global_step")
        if isinstance(raw_global_step, torch.Tensor):
            global_step = int(raw_global_step.detach().cpu().item())
        elif raw_global_step is None:
            global_step = None
        else:
            global_step = int(raw_global_step)

        if "obs_embedding_seq" in batch and "action_seq" in batch:
            hidden_seq_dtype = torch.long if self.io_mode == "token" else dtype
            hidden_seq = batch["obs_embedding_seq"].to(device=device, dtype=hidden_seq_dtype)
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
                global_step=global_step,
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
        # Token mode: no hidden-space target (only CE via token ids); skip.
        # Hidden mode: per-token hiddens are already in `next_obs_embedding`,
        # pull them directly into the target slot.
        next_image_hiddens_target = None
        if self.io_mode != "token":
            next_image_hiddens_target = next_obs.unsqueeze(1)   # [B, 1, N_img, C_in]

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
            global_step=global_step,
        )

    def forward(self, batch: dict[str, Any]) -> Any:
        """FSDP-compatible dispatcher.

        FSDP only triggers its parameter all-gather on ``__call__`` / ``forward``.
        The cotrain imagination loop needs ``encode_latent`` / ``predict_next`` /
        ``reward`` to run on full (gathered) params, so we route them here via
        a ``mode`` key:

            world_model({'mode': 'encode_latent', 'hidden': h}) -> TransDreamerLatentState
            world_model({'mode': 'predict_next', 'latent': s, 'actions': a}) -> TransDreamerLatentState
            world_model({'mode': 'reward', 'latent': s, 'actions': a, 'next_latent': s2}) -> Tensor

        Without a ``mode`` key (or with ``mode='pretrain'``), the call falls
        back to ``compute_loss_dict`` (existing Phase-1 SFT path).
        """
        mode = batch.get("mode") if isinstance(batch, dict) else None
        if mode in (None, "pretrain"):
            return self.compute_loss_dict(batch)
        if mode == "encode_latent":
            return self.encode_latent(batch["hidden"])
        if mode == "predict_next":
            return self.predict_next(batch["latent"], batch["actions"])
        if mode == "reward":
            return self.reward(
                batch["latent"], batch["actions"], batch["next_latent"],
                attention_mask=batch.get("attention_mask"),
            )
        raise ValueError(f"Unknown TSSMWorldModelTransDreamer forward mode: {mode!r}")

    # ── Single-step RSSM adapters for Dreamer-style imagination ──────────────
    # These are intentionally minimal — they mirror predict_next_hidden() but
    # return a TransDreamerLatentState (mean/std/stoch/h) so the cotrain
    # imagination loop can call encode_latent → predict_next → reward exactly
    # like it does for TSSMWorldModel.  Single-step rollout carries history in
    # latent.h and injects it into the next causal-Transformer token.

    def _hidden_to_pooled(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map raw input → pooled [B, obs_dim] using the same I/O front-end as
        compute_loss_dict.  Handles token-mode BPE ids, spatial-codec per-token
        hiddens, and already-pooled hiddens.
        """
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype
        if self.io_mode == "token":
            h = hidden.to(device=device)
            if h.ndim == 2 and h.shape[-1] == self.obs_dim and torch.is_floating_point(h):
                return h.to(dtype=dtype)
            if hidden.ndim != 2:
                raise ValueError(
                    f"token-mode encode_latent expects [B, N_img] long, "
                    f"got {tuple(hidden.shape)}"
                )
            bpe = h.to(dtype=torch.long)
            if self._bpe_to_img_idx.numel() == 0:
                raise RuntimeError(
                    "encode_latent (token mode) requires attach_lm_head() "
                    "to populate _bpe_to_img_idx before use"
                )
            img_idx = self._bpe_to_img_idx[bpe]
            if (img_idx < 0).any():
                raise ValueError("encode_latent (token mode): non-image BPE ids in input")
            per_token = self.token_embedder(img_idx)              # [B, N_img, d_embed]
            return self.conv_stem(per_token).to(dtype=dtype)       # [B, obs_dim]
        h = hidden.to(device=device, dtype=dtype)
        if h.ndim == 3:
            # [B, N_img, C_in] → conv stem → [B, obs_dim]
            return self.conv_stem(h)
        if h.ndim == 2 and h.shape[-1] == self.obs_dim:
            return h
        raise ValueError(
            f"encode_latent: unsupported input shape {tuple(h.shape)} "
            f"(expected [B, {self.obs_dim}] or feature-wrap [B, {self.latent_dim + self.d_model}])"
        )

    def encode_latent(self, hidden: torch.Tensor) -> "TransDreamerLatentState":
        """Wrap an observation (or a previous .feature()) into a state.

        Two input modes:
          1. Fresh obs: [B, N_img] long ids (token), [B, N_img, C] (spatial),
             or [B, obs_dim] (pooled).  Runs the posterior encoder; h is
             initialised to zeros.
          2. Feature wrap: [B, latent_dim + d_model] — this is the imagination
             loop re-wrapping a previously rolled-out feature.  We split it
             back into (stoch, h) without re-encoding through the posterior.
        """
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype
        if hidden.ndim == 2 and hidden.shape[-1] == self.latent_dim + self.d_model:
            f = hidden.to(device=device, dtype=dtype)
            stoch = f[:, : self.latent_dim]
            h_part = f[:, self.latent_dim :]
            return TransDreamerLatentState(
                mean=stoch,
                std=torch.full_like(stoch, fill_value=self.min_std),
                stoch=stoch,
                h=h_part,
            )
        pooled = self._hidden_to_pooled(hidden).to(device=device, dtype=dtype)
        h_init = torch.zeros(pooled.shape[0], self.d_model, device=device, dtype=dtype)
        mean, std, stoch = self._posterior_from_obs_h(pooled, h_init)
        return TransDreamerLatentState(mean=mean, std=std, stoch=stoch, h=h_init)

    def predict_next(
        self,
        latent: "TransDreamerLatentState",
        actions: torch.Tensor,
    ) -> "TransDreamerLatentState":
        """Single-step prior advance:
            token = (stoch_t, action_t)  →  causal Transformer (length 1)
                  →  h_{t+1}  →  prior_head  →  stoch_{t+1}.
        """
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype
        actions = actions.to(device=device, dtype=dtype)
        if actions.ndim == 3:
            actions = actions.mean(dim=1)
        stoch_in = latent.stoch.to(device=device, dtype=dtype)
        prev_h = latent.h.to(device=device, dtype=dtype)
        token_in = torch.cat([stoch_in, actions], dim=-1).unsqueeze(1)  # [B, 1, L+A]
        token_seq = self.act_stoch_emb(token_in)                         # [B, 1, d_model]
        token_seq = token_seq + prev_h.unsqueeze(1)
        h_seq = self.causal_transformer(token_seq)                       # [B, 1, d_model]
        prior_stats = self.prior_head(h_seq)                             # [B, 1, 2*latent]
        mean, std, stoch_next = self._stats_to_dist(prior_stats)
        return TransDreamerLatentState(
            mean=mean.squeeze(1),
            std=std.squeeze(1),
            stoch=stoch_next.squeeze(1),
            h=h_seq.squeeze(1),
        )

    def reward(
        self,
        latent: "TransDreamerLatentState",
        actions: torch.Tensor,
        next_latent: "TransDreamerLatentState",
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reward head trained in pretrain_loss reads cat(h_{t+1}, stoch_{t+1})."""
        feat = torch.cat([next_latent.h, next_latent.stoch], dim=-1)
        return self.reward_head(feat).squeeze(-1)

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
            Also stashes the predicted h/z pair for image decoding helpers.
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
            if hidden.ndim == 3:
                hidden = self.conv_stem(hidden)                         # [B, obs_dim]
            elif hidden.ndim != 2:
                raise ValueError(
                    "predict_next_hidden expects [B, obs_dim] or "
                    f"[B, N_img, C_in]; got {tuple(hidden.shape)}"
                )

        latent = self.encode_latent(hidden)
        prior_next = self.predict_next(latent, action)

        post_feature = torch.cat(
            [prior_next.h, prior_next.stoch], dim=-1
        )                                                            # [B, d_model+latent]
        pred = self.transition_head(post_feature)                     # [B, obs_dim]
        # Remember the step's prior stoch so callers that need it for the
        # image decoder can fetch it without rerunning the RSSM.
        self._last_predicted_h = prior_next.h.detach()
        self._last_predicted_stoch = prior_next.stoch.detach()
        return pred

    @torch.no_grad()
    def decode_pooled_to_image_hiddens(
        self, pooled: torch.Tensor, stoch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project a predicted RSSM feature to per-image-token hiddens via the
        bspace conv decoder.  Returns [B, n_image_tokens, in_channels].

        The decoder consumes h_t and z_t.  For compatibility with older callers
        that pass the transition_head output, this uses the h_t/z_t stashed by
        predict_next_hidden() when `pooled` is not already shaped like [B, d_model].
        """
        if self.image_decoder is None:
            raise RuntimeError("image_decoder is not enabled on this WM.")
        assert isinstance(self.image_decoder, BspaceConvDecoderHead)
        first_param = next(self.image_decoder.parameters())
        x = pooled.to(device=first_param.device, dtype=first_param.dtype)
        if x.shape[-1] != self.d_model:
            stash_h = getattr(self, "_last_predicted_h", None)
            if stash_h is None:
                raise ValueError(
                    "decode expects h_t with shape [B, d_model]; "
                    "call predict_next_hidden() first or pass h_t directly."
                )
            x = stash_h.to(device=first_param.device, dtype=first_param.dtype)
        # stoch required by the decoder; prefer caller-supplied, else use the
        # value stashed by predict_next_hidden, else zero.
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

        _ = self.predict_next_hidden(cur_bpe_ids, action)                 # stashes h_t/z_t
        h = getattr(self, "_last_predicted_h", None)
        stoch = getattr(self, "_last_predicted_stoch", None)
        if h is None or stoch is None:
            raise RuntimeError("predict_next_hidden did not stash _last_predicted_h/_last_predicted_stoch")

        dec_param = next(self.image_decoder.parameters())
        logits = self.image_decoder(
            h.to(device=dec_param.device, dtype=dec_param.dtype),
            stoch.to(device=dec_param.device, dtype=dec_param.dtype),
        )  # [B, N_img, num_image_tokens_vocab]
        img_idx = logits.argmax(dim=-1)                                    # [B, N_img]
        return self.image_token_bpe_ids[img_idx]                           # [B, N_img] BPE ids
