from __future__ import annotations

import torch
import copy
from src.utils.json_logger import JsonLogger
import hydra
import os
import tqdm
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import numpy as np
from src.dataloader import BaseDataset
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.utils.torch_utils import resolve_device
from src.utils.checkpoint_util import TopKCheckpointManager
from src.workspace.base_workspace import BaseWorkspace


class WorldModelWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")
    exclude_keys = ("encoder",)

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir=output_dir)

        self.print_config()

        self.device = resolve_device(str(self.config.trainer.device))
        set_seed(int(self.config.seed))
        self.world_model = None
        self.world_model_optimizer = None

    def _build_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = copy.deepcopy(cfg.encoder)
        init_model_path = OmegaConf.select(cfg, "init.vla_ckpt_path")
        if init_model_path is not None and OmegaConf.select(encoder_cfg, "model_path") is None:
            encoder_cfg.model_path = str(init_model_path)
        return encoder_cfg

    def _infer_hidden_dim(self) -> int | None:
        if not hasattr(self, "encoder"):
            return None
        backbone = getattr(self.encoder, "backbone", None)
        config = getattr(backbone, "config", None)
        for attr_name in ("hidden_size", "d_model"):
            value = getattr(config, attr_name, None)
            if value is not None:
                return int(value)
        return None

    def _extract_hidden(self, obs: dict[str, object]) -> torch.Tensor:
        proprio = obs.get("proprio")
        if isinstance(proprio, torch.Tensor):
            return proprio.to(self.device)

        image = obs.get("image")
        if isinstance(image, torch.Tensor):
            return image.flatten(start_dim=1).to(self.device)

        raise ValueError("World-model-only debug mode expects `obs.proprio` or `obs.image`.")

    def _attach_encoder_outputs(self, batch: dict[str, object]) -> dict[str, object]:
        if not hasattr(self, "encoder"):
            return batch
        obs = batch.get("obs")
        next_obs = batch.get("next_obs")
        if isinstance(obs, dict):
            batch["obs_embedding"] = self.encoder.encode(obs)
        if isinstance(next_obs, dict):
            batch["next_obs_embedding"] = self.encoder.encode(next_obs)
        return batch

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        print("Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset), "Dataset must be an instance of BaseDataset"
        dataloader_kwargs = dict(cfg.dataloader)
        collate_fn = getattr(dataset, "collate_fn", None)
        if callable(collate_fn):
            dataloader_kwargs["collate_fn"] = collate_fn
        train_dataloader = DataLoader(dataset, **dataloader_kwargs)

        # configure encoder
        if cfg.encoder is not None:
            encoder_cfg = self._build_encoder_cfg(cfg)
            self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)

        world_model_hidden_dim = self._infer_hidden_dim()
        if world_model_hidden_dim is None:
            self.world_model = hydra.utils.instantiate(cfg.world_model).to(self.device)
        else:
            self.world_model = hydra.utils.instantiate(
                cfg.world_model,
                hidden_dim=world_model_hidden_dim,
            ).to(self.device)
        self.world_model_optimizer = build_optimizer(
            self.world_model,
            self.config.optim.world_model,
        )

        # configure validation dataset

        # configure lr scheduler

        # configure ema

        # configure env

        # configure logging

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 2
            cfg.training.checkpoint_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        os.makedirs(self.output_dir, exist_ok=True)
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                train_transition_losses = list()
                train_kl_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = self._attach_encoder_outputs(batch)

                        # compute loss
                        loss_dict = self.world_model.compute_loss_dict(batch)
                        raw_loss = loss_dict["loss"]
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        grad_clip_norm = cfg.optim.get("grad_clip_norm")
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), float(grad_clip_norm))

                        # step optimizer
                        self.world_model_optimizer.step()
                        self.world_model_optimizer.zero_grad(
                            set_to_none=bool(cfg.optim.get("zero_grad_set_to_none", True))
                        )
                        
                        # update ema

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        transition_loss_cpu = float(loss_dict["transition_loss"].item())
                        kl_loss_cpu = float(loss_dict["kl_loss"].item())
                        tepoch.set_postfix(
                            loss=raw_loss_cpu,
                            transition=transition_loss_cpu,
                            kl=kl_loss_cpu,
                            refresh=False,
                        )
                        train_losses.append(raw_loss_cpu)
                        train_transition_losses.append(transition_loss_cpu)
                        train_kl_losses.append(kl_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'train_transition_loss': transition_loss_cpu,
                            'train_kl_loss': kl_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                if train_transition_losses:
                    step_log['train_transition_loss'] = float(np.mean(train_transition_losses))
                if train_kl_losses:
                    step_log['train_kl_loss'] = float(np.mean(train_kl_losses))

                # ========= eval for this epoch ==========
                policy = self.world_model
                if cfg.training.use_ema and hasattr(self, "ema_model"):
                    policy = self.ema_model
                policy.eval()

                # run rollout

                # run validation

                # run diffusion sampling on a training batch
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        return history
