from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import pathlib
import pickle
from pprint import pprint
from typing import Any, Mapping

import torch
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


class BaseWorkspace(ABC):
    include_keys = tuple()
    exclude_keys = tuple()

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
        path.parent.mkdir(parents=True, exist_ok=True)

        # Payload config
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        payload = {
            "cfg": self.cfg,
            "state_dicts": {},
            "pickles": {},
        }

        # State dicts
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
            elif key in include_keys:
                payload["pickles"][key] = pickle.dumps(value)

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
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = tuple(payload["pickles"].keys())

        # State restore
        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys and key in self.__dict__ and self.__dict__[key] is not None:
                self.__dict__[key].load_state_dict(value, **kwargs)

        # Pickle restore
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = pickle.loads(payload["pickles"][key])

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
