from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import pathlib
import pickle
from pprint import pprint
from typing import Any, Mapping

import hydra
import torch
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


class BaseWorkspace(ABC):
    workspace_name = "base"
    workspace_status = "abstract"
    workspace_family = "workspace"
    include_keys = tuple()
    exclude_keys = tuple()
    checkpoint_restore_output_dir = False

    def __init__(self, config: DictConfig, output_dir: str | None = None) -> None:
        # Runtime config
        self.config = config
        self.cfg = config
        self._output_dir = output_dir

        # Resolved config
        OmegaConf.resolve(self.config)

        # Loop state
        self.global_step = 0
        self.epoch = 0

    @property
    def output_dir(self) -> str:
        # Output dir
        if self._output_dir is not None:
            return str(pathlib.Path(self._output_dir).expanduser().resolve())

        configured_output_dir = OmegaConf.select(self.config, "training.out_dir")
        if configured_output_dir is not None:
            output_dir = pathlib.Path(str(configured_output_dir)).expanduser()
            if not output_dir.is_absolute():
                output_dir = PROJECT_ROOT / output_dir
            return str(output_dir.resolve())
        try:
            return HydraConfig.get().runtime.output_dir
        except Exception:
            return str(pathlib.Path(".").resolve())

    def get_checkpoint_dir(self) -> pathlib.Path:
        # Checkpoint dir
        return pathlib.Path(self.output_dir).joinpath("ckpt")

    def get_log_dir(self) -> pathlib.Path:
        # Log dir
        return pathlib.Path(self.output_dir).joinpath("log")

    def get_log_path(self, name: str = "logs.json.txt") -> pathlib.Path:
        # Log file
        return self.get_log_dir().joinpath(name)

    def print_config(self) -> None:
        # Config dump
        pprint(OmegaConf.to_container(self.config, resolve=True))

    def build_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        # Encoder config
        encoder_cfg = copy.deepcopy(cfg.encoder)
        init_model_path = OmegaConf.select(cfg, "init.vla_ckpt_path")
        if init_model_path is not None and OmegaConf.select(encoder_cfg, "model_path") is None:
            encoder_cfg.model_path = str(init_model_path)
        return encoder_cfg

    def _resolve_vla_init_path(self) -> str:
        configured = OmegaConf.select(self.cfg, "init.vla_ckpt_path")
        default_dir = getattr(self, "default_vla_init_dir", None)
        candidate = (
            pathlib.Path(str(configured)).expanduser().resolve()
            if configured is not None
            else pathlib.Path(str(default_dir)).expanduser().resolve()
        )
        if candidate.is_dir():
            if (candidate / "config.json").is_file():
                return str(candidate)
            for subdir in sorted(path for path in candidate.iterdir() if path.is_dir()):
                if (subdir / "config.json").is_file():
                    return str(subdir.resolve())
        return str(candidate.resolve())

    def _build_frozen_encoder_cfg(self, cfg: DictConfig) -> DictConfig:
        encoder_cfg = self.build_encoder_cfg(cfg)
        with open_dict(encoder_cfg):
            encoder_cfg.model_path = self._resolve_vla_init_path()
            encoder_cfg.freeze_backbone = True
        return encoder_cfg

    @staticmethod
    def _dataloader_kwargs(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
        if isinstance(config, DictConfig):
            return dict(OmegaConf.to_container(config, resolve=True))
        return dict(config)

    @staticmethod
    def _sanitize_worker_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        if int(kwargs.get("num_workers", 0) or 0) <= 0:
            kwargs.pop("prefetch_factor", None)
            kwargs["persistent_workers"] = False
        return kwargs

    def make_distributed_dataloader(
        self,
        dataset: Any,
        dataloader_cfg: Mapping[str, Any] | DictConfig,
        *,
        shuffle: bool | None = None,
        drop_last: bool | None = None,
        sanitize_worker_kwargs: bool = False,
    ) -> DataLoader:
        dataloader_kwargs = self._dataloader_kwargs(dataloader_cfg)
        if sanitize_worker_kwargs:
            dataloader_kwargs = self._sanitize_worker_kwargs(dataloader_kwargs)

        effective_shuffle = bool(dataloader_kwargs.get("shuffle", True)) if shuffle is None else bool(shuffle)
        effective_drop_last = (
            bool(dataloader_kwargs.get("drop_last", False))
            if drop_last is None
            else bool(drop_last)
        )
        dataloader_kwargs["shuffle"] = effective_shuffle
        dataloader_kwargs["drop_last"] = effective_drop_last

        distributed = getattr(self, "distributed", None)
        if distributed is not None and hasattr(distributed, "maybe_make_sampler"):
            sampler = distributed.maybe_make_sampler(
                dataset,
                shuffle=effective_shuffle,
                drop_last=effective_drop_last,
            )
            if sampler is not None:
                dataloader_kwargs["shuffle"] = False
                dataloader_kwargs["sampler"] = sampler

        collate_fn = getattr(dataset, "collate_fn", None)
        if callable(collate_fn):
            dataloader_kwargs["collate_fn"] = collate_fn
        return DataLoader(dataset, **dataloader_kwargs)

    def make_val_dataloaders(
        self,
        cfg: DictConfig,
        *,
        split_names: tuple[str, ...] = ("val_ind", "val_ood"),
        sanitize_worker_kwargs: bool = False,
    ) -> dict[str, DataLoader]:
        val_dataloaders: dict[str, DataLoader] = {}
        for split_name in split_names:
            val_ds_cfg = OmegaConf.select(cfg, f"dataset_{split_name}", default=None)
            if val_ds_cfg is None:
                continue
            val_ds = hydra.utils.instantiate(val_ds_cfg)
            val_dataloaders[split_name] = self.make_distributed_dataloader(
                val_ds,
                cfg.dataloader,
                shuffle=False,
                drop_last=False,
                sanitize_worker_kwargs=sanitize_worker_kwargs,
            )
        return val_dataloaders

    @staticmethod
    def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
        set_epoch = getattr(getattr(dataloader, "sampler", None), "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def freeze_module_in_place(self, module: Any) -> Any:
        # Freeze module
        if module is None:
            return None
        if hasattr(module, "eval"):
            module.eval()
        if hasattr(module, "parameters"):
            for parameter in module.parameters():
                parameter.requires_grad = False
        return module

    def infer_hidden_dim_from_encoder(self, encoder: Any | None) -> int | None:
        # Encoder hidden dim
        if encoder is None:
            return None
        backbone = getattr(encoder, "backbone", None)
        config = getattr(backbone, "config", None)
        for attr_name in ("hidden_size", "d_model"):
            value = getattr(config, attr_name, None)
            if value is not None:
                return int(value)
        return None

    def infer_hidden_dim_from_dataset(self, dataset: Any) -> int | None:
        # Dataset hidden dim
        data_spec = getattr(dataset, "data_spec", None)
        value = getattr(data_spec, "hidden_dim", None)
        if value is not None:
            return int(value)
        return None

    def extract_hidden_from_obs(
        self,
        obs: Mapping[str, object],
        device: torch.device,
        fallback_hidden_dim: int | None = None,
    ) -> torch.Tensor:
        # Fallback hidden extraction
        state = obs.get("state")
        if isinstance(state, torch.Tensor) and state.numel() > 0:
            return state.to(device)

        proprio = obs.get("proprio")
        if isinstance(proprio, torch.Tensor) and proprio.numel() > 0:
            return proprio.to(device)

        image = obs.get("image")
        if isinstance(image, torch.Tensor) and image.numel() > 0:
            return image.flatten(start_dim=1).to(device)

        batch_size = 1
        task_id = obs.get("task_id")
        if isinstance(task_id, torch.Tensor) and task_id.ndim >= 1:
            batch_size = int(task_id.shape[0])

        hidden_dim = fallback_hidden_dim
        if hidden_dim is None:
            hidden_dim = int(OmegaConf.select(self.cfg, "world_model.hidden_dim", default=1))
        return torch.zeros(batch_size, int(hidden_dim), device=device, dtype=torch.float32)

    def attach_encoder_outputs(
        self,
        batch: dict[str, object],
        *,
        encoder: Any | None,
        device: torch.device,
        fallback_hidden_dim: int | None = None,
        detach: bool = True,
    ) -> dict[str, object]:
        # Encoder bridge
        obs = batch.get("obs")
        next_obs = batch.get("next_obs")

        if encoder is None:
            if isinstance(obs, Mapping):
                batch["obs_embedding"] = self.extract_hidden_from_obs(
                    obs,
                    device=device,
                    fallback_hidden_dim=fallback_hidden_dim,
                )
            if isinstance(next_obs, Mapping):
                try:
                    batch["next_obs_embedding"] = self.extract_hidden_from_obs(
                        next_obs,
                        device=device,
                        fallback_hidden_dim=fallback_hidden_dim,
                    )
                except ValueError:
                    if "obs_embedding" in batch and isinstance(batch["obs_embedding"], torch.Tensor):
                        batch["next_obs_embedding"] = batch["obs_embedding"].detach().clone()
            return batch

        with torch.no_grad():
            if isinstance(obs, Mapping):
                obs_embedding = encoder.encode(obs)
                batch["obs_embedding"] = obs_embedding.detach() if detach else obs_embedding
            if isinstance(next_obs, Mapping):
                next_obs_embedding = encoder.encode(next_obs)
                batch["next_obs_embedding"] = next_obs_embedding.detach() if detach else next_obs_embedding
        return batch

    @staticmethod
    def slice_batch_mapping(mapping: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
        # Indexed slice for one mapping level plus one nested mapping level.
        sliced: dict[str, Any] = {}
        index_list = indices.tolist()
        for key, value in mapping.items():
            if isinstance(value, torch.Tensor):
                sliced[key] = value.index_select(0, indices)
                continue
            if isinstance(value, list):
                sliced[key] = [value[int(idx)] for idx in index_list]
                continue
            if isinstance(value, tuple):
                sliced[key] = tuple(value[int(idx)] for idx in index_list)
                continue
            if isinstance(value, Mapping):
                nested: dict[str, Any] = {}
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, torch.Tensor):
                        nested[nested_key] = nested_value.index_select(0, indices)
                    elif isinstance(nested_value, list):
                        nested[nested_key] = [nested_value[int(idx)] for idx in index_list]
                    elif isinstance(nested_value, tuple):
                        nested[nested_key] = tuple(nested_value[int(idx)] for idx in index_list)
                    else:
                        nested[nested_key] = nested_value
                sliced[key] = nested
                continue
            sliced[key] = value
        return sliced

    def make_history_entry(
        self,
        stage: str,
        step: int,
        metrics: Mapping[str, float],
    ) -> dict[str, float | str | int]:
        # History entry
        entry = {
            "stage": stage,
            "step": step,
            "global_step": self.global_step,
            "epoch": self.epoch,
            **metrics,
        }
        self.global_step += 1
        return entry

    def finish_epoch(self) -> None:
        # Epoch update
        self.epoch += 1

    @property
    def is_main_process(self) -> bool:
        # Distributed-aware main-process guard; returns True for non-distributed workspaces.
        distributed = getattr(self, "distributed", None)
        if distributed is not None:
            return bool(distributed.is_main_process)
        return True

    def resume(self, cfg: DictConfig | None = None) -> None:
        """Load latest checkpoint when training.resume=True (mirrors diffusion_policy convention)."""
        if cfg is None:
            cfg = self.cfg
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                if self.is_main_process:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

    def print_history(self, history: list[dict[str, float | str | int]]) -> None:
        # Metric print
        for entry in history:
            stage = str(entry["stage"])
            step = int(entry["step"])
            global_step = int(entry["global_step"])
            epoch = int(entry["epoch"])
            metrics = " ".join(
                f"{key}={value:.4f}"
                for key, value in entry.items()
                if key not in {"stage", "step", "global_step", "epoch"}
            )
            print(
                f"{stage}_step={step} global_step={global_step} epoch={epoch} {metrics}"
            )

    def save_checkpoint(
        self,
        path: str | pathlib.Path | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
    ) -> str:
        # Checkpoint path
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        path = pathlib.Path(path)

        # Payload config
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        distributed = getattr(self, "distributed", None)
        is_main_process = True if distributed is None else bool(distributed.is_main_process)
        requires_collective = (
            False
            if distributed is None
            else bool(getattr(distributed, "requires_collective_checkpointing", False))
        )
        if distributed is not None and not requires_collective and not is_main_process:
            return str(path.absolute())

        payload = {
            "cfg": self.cfg,
            "state_dicts": {},
            "pickles": {},
        }

        # State dicts
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    state_dict = self._state_dict_for_checkpoint(key, value)
                    if is_main_process and state_dict is not None:
                        payload["state_dicts"][key] = _copy_to_cpu(state_dict)
            elif key in include_keys and is_main_process:
                payload["pickles"][key] = pickle.dumps(value)

        if is_main_process:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
        return str(path.absolute())

    def get_checkpoint_path(self, tag: str = "latest") -> pathlib.Path:
        # Checkpoint file
        return self.get_checkpoint_dir().joinpath(f"{tag}.ckpt")

    def load_payload(
        self,
        payload: dict[str, Any],
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        # Key filters
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        pickles = payload.get("pickles", {})
        state_dicts = payload.get("state_dicts", {})
        if include_keys is None:
            include_keys = tuple(pickles.keys())
            if not bool(getattr(self, "checkpoint_restore_output_dir", False)):
                include_keys = tuple(key for key in include_keys if key != "_output_dir")

        # State restore
        for key, value in state_dicts.items():
            if key not in exclude_keys and key in self.__dict__ and self.__dict__[key] is not None:
                self._load_state_dict_from_checkpoint(key, self.__dict__[key], value, **kwargs)

        # Pickle restore
        for key in include_keys:
            if key in pickles:
                self.__dict__[key] = pickle.loads(pickles[key])

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        return value.state_dict()

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        value.load_state_dict(state_dict, **kwargs)

    def load_checkpoint(
        self,
        path: str | pathlib.Path | None = None,
        tag: str = "latest",
        exclude_keys: tuple[str, ...] | None = None,
        include_keys: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Checkpoint path
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        path = pathlib.Path(path)

        payload = torch.load(path, **kwargs)
        self.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
        )
        return payload

    def setup(self) -> None:
        """Optional lifecycle hook before execution.

        Existing workspaces build most state inside ``run``.  The hook gives
        new workspaces a common interface without forcing a large rewrite of
        the current training loops.
        """
        return None

    def execute(self) -> object:
        """Run the workspace through the public lifecycle interface."""
        return self.run()

    def teardown(self) -> None:
        """Optional lifecycle hook after execution."""
        return None

    @abstractmethod
    def run(self) -> object:
        # Workspace API
        raise NotImplementedError


def _copy_to_cpu(value: Any) -> Any:
    # CPU copy
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    return copy.deepcopy(value)
