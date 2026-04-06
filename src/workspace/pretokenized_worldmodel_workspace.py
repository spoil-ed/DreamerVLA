from __future__ import annotations

import copy
import os

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.dataloader import BaseDataset
from src.utils.checkpoint_util import TopKCheckpointManager
from src.utils.json_logger import JsonLogger
from src.utils.optim import build_optimizer
from src.utils.seed import set_seed
from src.utils.torch_utils import resolve_device
from src.workspace.base_workspace import BaseWorkspace


class PretokenizedWorldModelWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir=output_dir)

        self.print_config()

        self.device = resolve_device(str(self.config.trainer.device))
        set_seed(int(self.config.seed))
        self.world_model = None
        self.world_model_optimizer = None

    def _infer_hidden_dim(self, dataset: BaseDataset) -> int | None:
        data_spec = getattr(dataset, "data_spec", None)
        value = getattr(data_spec, "hidden_dim", None)
        if value is not None:
            return int(value)
        return None

    def _attach_encoder_outputs(self, batch: dict[str, object]) -> dict[str, object]:
        for key in ("obs_embedding", "next_obs_embedding", "action", "action_mask", "reward"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def run(self) -> list[dict[str, float | str | int]]:
        history: list[dict[str, float | str | int]] = []
        print("Workspace begin.")
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset), "Dataset must be an instance of BaseDataset"
        dataloader_kwargs = dict(cfg.dataloader)
        collate_fn = getattr(dataset, "collate_fn", None)
        if callable(collate_fn):
            dataloader_kwargs["collate_fn"] = collate_fn
        train_dataloader = DataLoader(dataset, **dataloader_kwargs)

        world_model_hidden_dim = self._infer_hidden_dim(dataset)
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

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        if cfg.training.debug:
            cfg.training.num_epochs = 3
            cfg.training.max_train_steps = 2
            cfg.training.checkpoint_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        os.makedirs(self.output_dir, exist_ok=True)
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                train_losses = list()
                train_transition_losses = list()
                train_kl_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = self._attach_encoder_outputs(batch)

                        loss_dict = self.world_model.compute_loss_dict(batch)
                        raw_loss = loss_dict["loss"]
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        grad_clip_norm = cfg.optim.get("grad_clip_norm")
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), float(grad_clip_norm))

                        self.world_model_optimizer.step()
                        self.world_model_optimizer.zero_grad(
                            set_to_none=bool(cfg.optim.get("zero_grad_set_to_none", True))
                        )

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
                            "train_loss": raw_loss_cpu,
                            "train_transition_loss": transition_loss_cpu,
                            "train_kl_loss": kl_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss
                if train_transition_losses:
                    step_log["train_transition_loss"] = float(np.mean(train_transition_losses))
                if train_kl_losses:
                    step_log["train_kl_loss"] = float(np.mean(train_kl_losses))

                policy = self.world_model
                if cfg.training.use_ema and hasattr(self, "ema_model"):
                    policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = dict()
                    for key, value in step_log.items():
                        metric_dict[key.replace("/", "_")] = value
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                policy.train()

                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        return history


__all__ = ["PretokenizedWorldModelWorkspace"]
