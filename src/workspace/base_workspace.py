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
            if key not in exclude_keys and key in self.__dict__:
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
