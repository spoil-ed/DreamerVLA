"""DreamerVLA closed-loop training workspace.

Each training step runs two interleaved phases on the same data batch:

  Phase-1  World-model (WM) pretraining
           WM learns (obs, action) → next_obs transitions and reward prediction.
           Encoder is frozen; policy/critic are not involved.

  Phase-2  Actor-Critic update  (Dreamer-style)
           H-step imagination in WM latent space:
             policy samples actions → WM gives rewards → critic bootstraps λ-returns
           Actor  maximises γᵗ·G_t^λ  (gradient through reward head → actions)
           Critic minimises MSE(V(s_t), stop_grad(G_t^λ))

Both phases can be toggled independently via `training.run_wm_phase` and
`training.run_actor_critic_phase`.
"""
from __future__ import annotations

import copy
import os
import pathlib
import pickle
from typing import Any

import hydra
import torch
import tqdm
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from src.algorithms.dreamer_vla import (
    imagine_actor_critic_step,
    world_model_pretrain_step,
)
from src.dataloader import BaseDataset
from src.trainer import NopretokenizeSFTDistributedHelper
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.ema import EMAHelper
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.utils.torch_utils import freeze_module
from src.workspace.base_workspace import BaseWorkspace


class DreamerVLAWorkspace(BaseWorkspace):
    """Closed-loop DreamerVLA: WM pretraining + Dreamer actor-critic."""

    include_keys = ("global_step", "epoch")
    # encoder is frozen — no need to checkpoint it.
    exclude_keys = ("encoder",)

    default_vla_init_dir = "/home/user01/liops/workspace/DreamerVLA/data/ckpts/VLA_model_256/libero_10"
    default_output_dir = "/home/user01/liops/workspace/DreamerVLA/data/outputs/dreamer_vla"

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        if output_dir is None:
            output_dir = str(OmegaConf.select(config, "training.out_dir", default=self.default_output_dir))
        super().__init__(config, output_dir=output_dir)

        self.distributed = NopretokenizeSFTDistributedHelper.initialize(
            strategy=str(OmegaConf.select(config, "training.distributed_strategy", default="ddp")),
            fsdp_mixed_precision=str(OmegaConf.select(config, "training.fsdp_mixed_precision", default="bf16")),
            enable_activation_checkpointing=bool(
                OmegaConf.select(config, "training.enable_activation_checkpointing", default=True)
            ),
        )
        self.rank = self.distributed.rank
        self.local_rank = self.distributed.local_rank
        self.world_size = self.distributed.world_size
        self.device = self.distributed.resolve_device(str(self.config.trainer.device))
        if self.distributed.is_main_process:
            self.print_config()
        set_seed(int(self.config.seed) + self.rank)

        # ── model placeholders ──────────────────────────────────────────────
        self.encoder = None        # RynnVLAEncoder   — frozen feature extractor
        self.policy = None         # VLAPolicy         — Dreamer actor (latent space)
        self.critic = None         # Critic            — value function (latent space)
        self.world_model = None    # TSSMWorldModel    — dynamics + reward

        # ── optimizer placeholders ──────────────────────────────────────────
        self.policy_optimizer = None
        self.critic_optimizer = None
        self.world_model_optimizer = None
        self.world_model_ema: EMAHelper | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_vla_init_path(self) -> str:
        configured = OmegaConf.select(self.cfg, "init.vla_ckpt_path")
        candidate = (
            pathlib.Path(str(configured)).expanduser().resolve()
            if configured is not None
            else pathlib.Path(self.default_vla_init_dir)
        )
        if candidate.is_dir():
            if (candidate / "config.json").is_file():
                return str(candidate)
            for subdir in sorted(p for p in candidate.iterdir() if p.is_dir()):
                if (subdir / "config.json").is_file():
                    return str(subdir.resolve())
        return str(candidate.resolve())

    def _build_frozen_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = copy.deepcopy(cfg.encoder)
        init_model_path = OmegaConf.select(cfg, "init.vla_ckpt_path")
        if init_model_path is not None and OmegaConf.select(encoder_cfg, "model_path") is None:
            encoder_cfg.model_path = str(init_model_path)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            encoder_cfg.freeze_backbone = True
        return encoder_cfg

    # ──────────────────────────────────────────────────────────────────────
    # Embedding helpers
    # ──────────────────────────────────────────────────────────────────────

    def _encode_hidden_from_tokenized(self, input_ids_list: list[list[int]]) -> torch.Tensor:
        """Run the frozen encoder on a list of token sequences → pooled float32 tensor."""
        if self.encoder is None:
            raise ValueError("Encoder must be initialised before calling _encode_hidden_from_tokenized.")
        if not input_ids_list:
            hidden_dim = int(OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1))
            return torch.zeros((0, hidden_dim), device=self.device, dtype=torch.float32)

        labels_list = [[-100] * len(seq) for seq in input_ids_list]
        lengths = [len(seq) for seq in input_ids_list]
        with torch.no_grad():
            _, _, _, hidden_states, _, _, _ = self.encoder.backbone(
                input_ids=input_ids_list,
                labels=labels_list,
                training=True,
                output_hidden_states=True,
                att_mask=False,
            )
        attention_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        for idx, length in enumerate(lengths):
            if length > 0:
                attention_mask[idx, :length] = True
        weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled.float().detach()

    def _make_obs_dict(self, input_ids_list: list[list[int]]) -> dict[str, torch.Tensor]:
        """Encode tokenized obs → {obs_embedding: tensor} for policy/WM passthrough."""
        return {"obs_embedding": self._encode_hidden_from_tokenized(input_ids_list)}

    # ──────────────────────────────────────────────────────────────────────
    # Batch assembly
    # ──────────────────────────────────────────────────────────────────────

    def _build_wm_pretrain_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        """
        Assemble {obs, next_obs, action, reward?} for world_model_pretrain_step.
        obs / next_obs are {obs_embedding: tensor}.
        Returns None when required keys are absent.
        """
        obs_ids = batch.get("wm_obs_input_ids")
        next_obs_ids = batch.get("wm_next_obs_input_ids")
        if not isinstance(obs_ids, list) or not isinstance(next_obs_ids, list):
            return None

        action = batch.get("conditioning_action") or batch.get("action")
        if not isinstance(action, torch.Tensor):
            return None

        wm_batch: dict[str, Any] = {
            "obs": self._make_obs_dict(obs_ids),
            "next_obs": self._make_obs_dict(next_obs_ids),
            "action": action.to(self.device),
        }
        reward = batch.get("reward")
        if isinstance(reward, torch.Tensor):
            wm_batch["reward"] = reward.to(self.device)
        return wm_batch

    def _build_actor_critic_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        """
        Assemble {obs} for imagine_actor_critic_step.
        obs is {obs_embedding: tensor} — the starting state for imagination.
        Falls back from wm_obs_input_ids → input_ids.
        Returns None when neither key is present.
        """
        obs_ids = batch.get("wm_obs_input_ids") or batch.get("input_ids")
        if not isinstance(obs_ids, list):
            return None
        return {"obs": self._make_obs_dict(obs_ids)}

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: str | pathlib.Path | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
    ) -> str:
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        path = pathlib.Path(path)

        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        if not self.distributed.requires_collective_checkpointing and not self.distributed.is_main_process:
            return str(path.absolute())

        payload = {"cfg": self.cfg, "state_dicts": {}, "pickles": {}}
        for key, value in self.__dict__.items():
            if key in exclude_keys:
                continue
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                state_dict = self._state_dict_for_checkpoint(key, value)
                if self.distributed.is_main_process and state_dict is not None:
                    payload["state_dicts"][key] = _copy_to_cpu(state_dict)
            elif key in include_keys and self.distributed.is_main_process:
                payload["pickles"][key] = pickle.dumps(value)

        if self.distributed.is_main_process:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
        return str(path.absolute())

    def load_payload(
        self,
        payload: dict[str, Any],
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = tuple(payload["pickles"].keys())

        for key, value in payload["state_dicts"].items():
            if key in exclude_keys or key not in self.__dict__ or self.__dict__[key] is None:
                continue
            self._load_state_dict_from_checkpoint(key, self.__dict__[key], value, **kwargs)

        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = pickle.loads(payload["pickles"][key])

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        if key == "policy" and self.policy is not None:
            with self.distributed.model_state_dict_context(self.policy):
                return self.policy.state_dict()
        if key == "policy_optimizer" and self.policy_optimizer is not None and self.policy is not None:
            return self.distributed.optimizer_state_dict(self.policy, self.policy_optimizer)
        if key == "critic" and self.critic is not None:
            with self.distributed.model_state_dict_context(self.critic):
                return self.critic.state_dict()
        if key == "critic_optimizer" and self.critic_optimizer is not None and self.critic is not None:
            return self.distributed.optimizer_state_dict(self.critic, self.critic_optimizer)
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(self.world_model):
                return self.world_model.state_dict()
        if key == "world_model_optimizer" and self.world_model_optimizer is not None and self.world_model is not None:
            return self.distributed.optimizer_state_dict(self.world_model, self.world_model_optimizer)
        return value.state_dict()

    def _load_state_dict_from_checkpoint(
        self, key: str, value: Any, state_dict: dict[str, Any], **kwargs: Any
    ) -> None:
        if key == "policy" and self.policy is not None:
            with self.distributed.model_state_dict_context(self.policy):
                value.load_state_dict(state_dict, **kwargs)
            return
        if key == "policy_optimizer" and self.policy_optimizer is not None and self.policy is not None:
            self.distributed.load_optimizer_state_dict(self.policy, self.policy_optimizer, state_dict)
            return
        if key == "critic" and self.critic is not None:
            with self.distributed.model_state_dict_context(self.critic):
                value.load_state_dict(state_dict, **kwargs)
            return
        if key == "critic_optimizer" and self.critic_optimizer is not None and self.critic is not None:
            self.distributed.load_optimizer_state_dict(self.critic, self.critic_optimizer, state_dict)
            return
        if key == "world_model" and self.world_model is not None:
            with self.distributed.model_state_dict_context(self.world_model):
                value.load_state_dict(state_dict, **kwargs)
            return
        if key == "world_model_optimizer" and self.world_model_optimizer is not None and self.world_model is not None:
            self.distributed.load_optimizer_state_dict(self.world_model, self.world_model_optimizer, state_dict)
            return
        value.load_state_dict(state_dict, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_val_loss(self, val_dataloader: DataLoader, split_name: str) -> dict[str, float]:
        if self.world_model is None:
            return {}
        self.world_model.eval()
        val_losses: list[float] = []
        val_transition_losses: list[float] = []
        for batch in val_dataloader:
            wm_batch = self._build_wm_pretrain_batch(batch)
            if wm_batch is None:
                continue
            wm_batch = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in wm_batch.items()
            }
            loss_dict = self.world_model.compute_loss_dict(wm_batch)
            val_losses.append(float(loss_dict["loss"].item()))
            val_transition_losses.append(float(loss_dict["transition_loss"].item()))
        self.world_model.train()
        if not val_losses:
            return {}
        count = max(self.distributed.reduce_sum(len(val_losses)), 1.0)
        metrics = {
            f"val_{split_name}_wm_loss": self.distributed.reduce_sum(sum(val_losses)) / count,
            f"val_{split_name}_wm_transition_loss": self.distributed.reduce_sum(sum(val_transition_losses)) / count,
        }
        if self.distributed.is_main_process:
            print(f"  [Val {split_name}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics

    def run(self) -> list[dict[str, float | str | int]]:  # noqa: C901
        history: list[dict[str, float | str | int]] = []
        if self.distributed.is_main_process:
            print("DreamerVLA Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

        # ── dataset & dataloader ────────────────────────────────────────
        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset)

        dataloader_kwargs = dict(cfg.dataloader)
        sampler = self.distributed.maybe_make_sampler(
            dataset,
            shuffle=bool(dataloader_kwargs.get("shuffle", True)),
            drop_last=bool(dataloader_kwargs.get("drop_last", False)),
        )
        if sampler is not None:
            dataloader_kwargs["shuffle"] = False
            dataloader_kwargs["sampler"] = sampler
        collate_fn = getattr(dataset, "collate_fn", None)
        if callable(collate_fn):
            dataloader_kwargs["collate_fn"] = collate_fn
        train_dataloader = DataLoader(dataset, **dataloader_kwargs)

        # configure validation dataset
        val_dataloaders: dict[str, DataLoader] = {}
        for split_name in ("val_ind", "val_ood"):
            val_ds_cfg = OmegaConf.select(cfg, f"dataset_{split_name}", default=None)
            if val_ds_cfg is None:
                continue
            val_ds = hydra.utils.instantiate(val_ds_cfg)
            val_dl_kwargs = dict(cfg.dataloader)
            val_dl_kwargs["shuffle"] = False
            val_dl_kwargs["drop_last"] = False
            val_sampler = self.distributed.maybe_make_sampler(val_ds, shuffle=False, drop_last=False)
            if val_sampler is not None:
                val_dl_kwargs["sampler"] = val_sampler
            val_collate = getattr(val_ds, "collate_fn", None)
            if callable(val_collate):
                val_dl_kwargs["collate_fn"] = val_collate
            val_dataloaders[split_name] = DataLoader(val_ds, **val_dl_kwargs)

        # ── encoder (frozen) ────────────────────────────────────────────
        encoder_cfg = self._build_frozen_encoder_cfg(cfg)
        self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
        freeze_module(self.encoder)

        # ── world model (trainable in WM phase, frozen in actor-critic phase) ─
        world_model_cfg = OmegaConf.select(cfg, "world_model")
        if world_model_cfg is None:
            raise ValueError("`world_model` config section is required.")

        wm_hidden_dim = (
            self.infer_hidden_dim_from_dataset(dataset)
            or self.infer_hidden_dim_from_encoder(self.encoder)
        )
        if wm_hidden_dim is not None:
            self.world_model = hydra.utils.instantiate(world_model_cfg, hidden_dim=wm_hidden_dim).to(self.device)
        else:
            self.world_model = hydra.utils.instantiate(world_model_cfg).to(self.device)

        # Uniform dtype before FSDP wrapping (transition backbone loads in bf16)
        fsdp_precision = str(OmegaConf.select(cfg, "training.fsdp_mixed_precision", default="bf16"))
        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.world_model = self.world_model.to(dtype=_dtype_map.get(fsdp_precision, torch.bfloat16))
        self.world_model = self.distributed.wrap_trainable_module(self.world_model)

        wm_optim_cfg = OmegaConf.select(cfg, "optim.world_model")
        if wm_optim_cfg is None:
            raise ValueError("`optim.world_model` must be configured.")
        self.world_model_optimizer = build_optimizer(self.world_model, wm_optim_cfg)

        # ── policy (Dreamer actor, acts in WM latent space) ────────────
        policy_cfg = OmegaConf.select(cfg, "policy")
        if policy_cfg is None:
            raise ValueError("`policy` config section is required.")

        self.policy = hydra.utils.instantiate(policy_cfg).to(self.device)
        self.policy = self.distributed.wrap_trainable_module(self.policy)

        policy_optim_cfg = OmegaConf.select(cfg, "optim.policy")
        if policy_optim_cfg is None:
            raise ValueError("`optim.policy` must be configured.")
        self.policy_optimizer = build_optimizer(self.policy, policy_optim_cfg)

        # ── critic (value function in WM latent space) ──────────────────
        critic_cfg = OmegaConf.select(cfg, "critic")
        if critic_cfg is None:
            raise ValueError("`critic` config section is required.")

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic = self.distributed.wrap_trainable_module(self.critic)

        critic_optim_cfg = OmegaConf.select(cfg, "optim.critic")
        if critic_optim_cfg is None:
            raise ValueError("`optim.critic` must be configured.")
        self.critic_optimizer = build_optimizer(self.critic, critic_optim_cfg)

        # configure ema
        if bool(OmegaConf.select(cfg, "training.use_ema", default=False)) and self.world_model_ema is None:
            self.world_model_ema = EMAHelper(
                self.world_model,
                decay=float(OmegaConf.select(cfg, "ema.decay", default=0.9999)),
                update_after_step=int(OmegaConf.select(cfg, "ema.update_after_step", default=0)),
            )

        # resume training
        self.resume(cfg)

        # configure lr scheduler
        lr_scheduler_name = str(OmegaConf.select(cfg, "training.lr_scheduler", default="constant"))
        lr_warmup_steps = int(OmegaConf.select(cfg, "training.lr_warmup_steps", default=0))
        total_training_steps = (
            len(train_dataloader) * int(cfg.training.num_epochs)
        ) // int(cfg.training.gradient_accumulate_every)
        wm_lr_scheduler = get_scheduler(
            lr_scheduler_name,
            optimizer=self.world_model_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )
        policy_lr_scheduler = get_scheduler(
            lr_scheduler_name,
            optimizer=self.policy_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )
        critic_lr_scheduler = get_scheduler(
            lr_scheduler_name,
            optimizer=self.critic_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )

        # ── training hyper-params ────────────────────────────────────────
        run_wm_phase = bool(OmegaConf.select(cfg, "training.run_wm_phase", default=True))
        run_ac_phase = bool(OmegaConf.select(cfg, "training.run_actor_critic_phase", default=True))
        algorithm_cfg = OmegaConf.select(cfg, "algorithm")
        if algorithm_cfg is None:
            raise ValueError("`algorithm` config section is required.")
        optim_cfg = OmegaConf.select(cfg, "optim")

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        if cfg.training.debug:
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 2
            cfg.training.checkpoint_every = 1

        if self.distributed.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
        self.distributed.barrier()

        train_log_path = os.path.join(self.output_dir, "dreamer_vla_logs.json.txt")
        train_logger_cm = self.distributed.logger_context(train_log_path)

        try:
            with train_logger_cm as train_json_logger:
                reached_max_steps = False

                for _local_epoch_idx in range(cfg.training.num_epochs):
                    if sampler is not None:
                        sampler.set_epoch(self.epoch)

                    step_log: dict[str, float | str | int] = {}
                    epoch_wm_losses: list[float] = []
                    epoch_actor_losses: list[float] = []
                    epoch_critic_losses: list[float] = []
                    epoch_returns: list[float] = []
                    epoch_rewards: list[float] = []

                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Epoch {self.epoch}",
                        disable=not self.distributed.is_main_process,
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            local_metrics: dict[str, float] = {}
                            step_had_update = False

                            # ── Phase 1: world-model pretraining ──────────
                            if run_wm_phase:
                                wm_batch = self._build_wm_pretrain_batch(batch)
                                if wm_batch is not None:
                                    self.world_model.train()
                                    self.policy.eval()
                                    self.critic.eval()
                                    wm_metrics = world_model_pretrain_step(
                                        policy=self.policy,
                                        world_model=self.world_model,
                                        optimizer=self.world_model_optimizer,
                                        batch=wm_batch,
                                        device=self.device,
                                        optim_cfg=optim_cfg,
                                    )
                                    wm_lr_scheduler.step()

                                    # update ema
                                    if self.world_model_ema is not None:
                                        self.world_model_ema.step(self.world_model)

                                    epoch_wm_losses.append(wm_metrics["loss"])
                                    local_metrics["train_wm_loss"] = wm_metrics["loss"]
                                    local_metrics["train_wm_transition_loss"] = wm_metrics["transition_loss"]
                                    local_metrics["train_wm_reward_loss"] = wm_metrics["reward_loss"]
                                    local_metrics["train_wm_grad_norm"] = wm_metrics["grad_norm"]
                                    local_metrics["wm_lr"] = float(wm_lr_scheduler.get_last_lr()[0])
                                    step_had_update = True

                            # ── Phase 2: Dreamer actor-critic update ───────
                            if run_ac_phase:
                                ac_batch = self._build_actor_critic_batch(batch)
                                if ac_batch is not None:
                                    self.world_model.eval()
                                    ac_metrics = imagine_actor_critic_step(
                                        policy=self.policy,
                                        world_model=self.world_model,
                                        critic=self.critic,
                                        actor_optimizer=self.policy_optimizer,
                                        critic_optimizer=self.critic_optimizer,
                                        obs=ac_batch["obs"],
                                        device=self.device,
                                        algorithm_cfg=algorithm_cfg,
                                        optim_cfg=optim_cfg,
                                    )
                                    epoch_actor_losses.append(ac_metrics["actor_loss"])
                                    epoch_critic_losses.append(ac_metrics["critic_loss"])
                                    epoch_returns.append(ac_metrics["returns_mean"])
                                    epoch_rewards.append(ac_metrics["reward_mean"])
                                    local_metrics["train_actor_loss"] = ac_metrics["actor_loss"]
                                    local_metrics["train_critic_loss"] = ac_metrics["critic_loss"]
                                    local_metrics["train_returns_mean"] = ac_metrics["returns_mean"]
                                    local_metrics["train_returns_std"] = ac_metrics["returns_std"]
                                    local_metrics["train_reward_mean"] = ac_metrics["reward_mean"]
                                    local_metrics["train_value_mean"] = ac_metrics["value_mean"]
                                    local_metrics["train_actor_grad_norm"] = ac_metrics["actor_grad_norm"]
                                    local_metrics["train_critic_grad_norm"] = ac_metrics["critic_grad_norm"]
                                    policy_lr_scheduler.step()
                                    critic_lr_scheduler.step()
                                    local_metrics["policy_lr"] = float(policy_lr_scheduler.get_last_lr()[0])
                                    local_metrics["critic_lr"] = float(critic_lr_scheduler.get_last_lr()[0])
                                    step_had_update = True

                            if not step_had_update:
                                continue

                            # ── reduce & log ─────────────────────────────
                            reduced = self.distributed.reduce_mean_dict(local_metrics)
                            step_log = {**reduced, "global_step": self.global_step, "epoch": self.epoch}

                            pbar_postfix: dict[str, float] = {}
                            if "train_wm_loss" in reduced:
                                pbar_postfix["wm"] = reduced["train_wm_loss"]
                            if "train_actor_loss" in reduced:
                                pbar_postfix["actor"] = reduced["train_actor_loss"]
                            if "train_critic_loss" in reduced:
                                pbar_postfix["critic"] = reduced["train_critic_loss"]
                            if "train_returns_mean" in reduced:
                                pbar_postfix["G"] = reduced["train_returns_mean"]
                            if pbar_postfix:
                                tepoch.set_postfix(refresh=False, **pbar_postfix)

                            is_last_batch = batch_idx == len(train_dataloader) - 1
                            if not is_last_batch:
                                train_json_logger.log(step_log)
                                self.global_step += 1

                            if (
                                cfg.training.max_train_steps is not None
                                and batch_idx >= cfg.training.max_train_steps - 1
                            ):
                                reached_max_steps = True
                                break

                    # ── epoch summary ──────────────────────────────────────
                    if not epoch_wm_losses and not epoch_actor_losses:
                        self.global_step += 1
                        self.epoch += 1
                        continue

                    if epoch_wm_losses:
                        wm_n = max(self.distributed.reduce_sum(len(epoch_wm_losses)), 1.0)
                        step_log["epoch_wm_loss"] = self.distributed.reduce_sum(sum(epoch_wm_losses)) / wm_n
                    if epoch_actor_losses:
                        ac_n = max(self.distributed.reduce_sum(len(epoch_actor_losses)), 1.0)
                        step_log["epoch_actor_loss"] = self.distributed.reduce_sum(sum(epoch_actor_losses)) / ac_n
                        step_log["epoch_critic_loss"] = self.distributed.reduce_sum(sum(epoch_critic_losses)) / ac_n
                        step_log["epoch_returns_mean"] = self.distributed.reduce_sum(sum(epoch_returns)) / ac_n
                        step_log["epoch_reward_mean"] = self.distributed.reduce_sum(sum(epoch_rewards)) / ac_n

                    step_log.setdefault("epoch_wm_loss", float("inf"))
                    step_log.setdefault("epoch_actor_loss", float("inf"))
                    step_log.setdefault("epoch_critic_loss", float("inf"))

                    # run validation
                    eval_every = int(OmegaConf.select(cfg, "eval.eval_every", default=1))
                    if val_dataloaders and (self.epoch % eval_every) == 0:
                        for split_name, val_dl in val_dataloaders.items():
                            step_log.update(self.evaluate_val_loss(val_dl, split_name))

                    train_json_logger.log(step_log)
                    history.append(dict(step_log))

                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                        topk_ckpt_path = None
                        if self.distributed.is_main_process:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        topk_ckpt_path = self.distributed.broadcast_object(topk_ckpt_path)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                    self.global_step += 1
                    self.epoch += 1
                    if reached_max_steps:
                        break

        finally:
            self.distributed.barrier()
            self.distributed.cleanup()

        return history


__all__ = ["DreamerVLAWorkspace"]


# ── internal helpers ──────────────────────────────────────────────────────────

def _copy_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _copy_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(v) for v in value]
    return copy.deepcopy(value)
