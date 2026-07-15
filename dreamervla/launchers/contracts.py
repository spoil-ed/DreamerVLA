"""Hydra-selected specialization contracts for the unified train launcher."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf


class LaunchContract(Protocol):
    """Specialize launcher inputs without owning Hydra composition or execution."""

    def normalize_argv(self, argv: Sequence[str]) -> list[str]: ...

    def derive_overrides(
        self,
        cfg: DictConfig,
        overrides: Sequence[str],
    ) -> list[str]: ...

    def validate(self, cfg: DictConfig) -> None: ...

    def update_env(self, cfg: DictConfig, env: dict[str, str]) -> None: ...

    def summary_lines(
        self,
        cfg: DictConfig,
        env: Mapping[str, str],
    ) -> list[str]: ...


class DefaultLaunchContract:
    """No-op specialization used by ordinary Hydra experiments."""

    def normalize_argv(self, argv: Sequence[str]) -> list[str]:
        return list(argv)

    def derive_overrides(
        self,
        cfg: DictConfig,
        overrides: Sequence[str],
    ) -> list[str]:
        del cfg, overrides
        return []

    def validate(self, cfg: DictConfig) -> None:
        del cfg

    def update_env(self, cfg: DictConfig, env: dict[str, str]) -> None:
        del cfg, env

    def summary_lines(
        self,
        cfg: DictConfig,
        env: Mapping[str, str],
    ) -> list[str]:
        del cfg, env
        return []


class CotrainLaunchContract(DefaultLaunchContract):
    """Failure-imagined-RL checkpoint, classifier, and GPU launch contract."""

    def normalize_argv(self, argv: Sequence[str]) -> list[str]:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--wm_ckpt")
        parser.add_argument("--cls_ckpt")
        args, remaining = parser.parse_known_args(list(argv))
        values = list(remaining)

        if (args.wm_ckpt is None) != (args.cls_ckpt is None):
            raise ValueError("--wm_ckpt and --cls_ckpt must be supplied together")
        for option, raw, hydra_key in (
            ("--wm_ckpt", args.wm_ckpt, "init.world_model_state_ckpt"),
            ("--cls_ckpt", args.cls_ckpt, "init.classifier_state_ckpt"),
        ):
            if raw is None:
                continue
            if _has_override(values, hydra_key):
                raise ValueError(f"{option} cannot be combined with the Hydra {hydra_key} override")
            path = _existing_path(raw, label=option)
            values.append(f"{hydra_key}={_hydra_string(path)}")

        self._component_environment_overrides(values)
        self._runtime_environment_overrides(values)
        if _truthy_environment("COTRAIN_DRY_RUN") and not _has_override(
            values,
            "dry_run",
        ):
            values.append("dry_run=true")
        return values

    def derive_overrides(
        self,
        cfg: DictConfig,
        overrides: Sequence[str],
    ) -> list[str]:
        values = list(overrides)
        checkpoint = OmegaConf.select(cfg, "init.classifier_state_ckpt", default=None)
        if checkpoint in {None, ""}:
            return []
        output_dim = _classifier_checkpoint_output_dim(str(checkpoint))
        if output_dim is None:
            return []

        output_key = "ray_components.classifier.kwargs.output_dim"
        loss_key = "learner.train_cfg.classifier_loss_type"
        configured_output = int(OmegaConf.select(cfg, output_key))
        configured_loss = str(OmegaConf.select(cfg, loss_key)).lower()
        expected_loss = "bce" if output_dim == 1 else "ce"
        if _has_override(values, output_key) and configured_output != output_dim:
            raise ValueError(
                f"classifier checkpoint has output_dim={output_dim}, but {output_key}="
                f"{configured_output} was explicitly requested"
            )
        if _has_override(values, loss_key) and configured_loss != expected_loss:
            raise ValueError(
                f"classifier checkpoint output_dim={output_dim} requires {loss_key}="
                f"{expected_loss}, got {configured_loss}"
            )

        derived: list[str] = []
        if not _has_override(values, output_key) and configured_output != output_dim:
            derived.append(f"{output_key}={output_dim}")
        if not _has_override(values, loss_key) and configured_loss != expected_loss:
            derived.append(f"{loss_key}={expected_loss}")
        return derived

    def validate(self, cfg: DictConfig) -> None:
        training_mode = str(
            OmegaConf.select(
                cfg,
                "manual_cotrain.training_mode",
                default="staged_full_cotrain",
            )
        )
        if training_mode != "failure_imagined_rl" or bool(
            OmegaConf.select(cfg, "training.resume", default=False)
        ):
            return
        world_model = OmegaConf.select(
            cfg,
            "init.world_model_state_ckpt",
            default=None,
        )
        classifier = OmegaConf.select(
            cfg,
            "init.classifier_state_ckpt",
            default=None,
        )
        if world_model in {None, ""} or classifier in {None, ""}:
            raise ValueError(
                "failure_imagined_rl freezes WM/CLS and requires both --wm_ckpt and "
                "--cls_ckpt (or --resume); random WM/CLS initialization is invalid"
            )

    def update_env(self, cfg: DictConfig, env: dict[str, str]) -> None:
        count = int(OmegaConf.select(cfg, "manual_cotrain.ngpu", default=1))
        raw = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        visible = [item.strip() for item in raw.split(",") if item.strip()]
        if not visible:
            visible = [str(index) for index in range(count)]
        if len(visible) != count or len(set(visible)) != count:
            raise ValueError(
                f"cotrain requires {count} distinct visible GPUs; CUDA_VISIBLE_DEVICES={raw!r}"
            )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
        env.setdefault(
            "LIBERO_CONFIG_PATH",
            str((Path(env["DVLA_DATA_ROOT"]) / ".libero").resolve()),
        )

    def summary_lines(
        self,
        cfg: DictConfig,
        env: Mapping[str, str],
    ) -> list[str]:
        debug = bool(OmegaConf.select(cfg, "training.debug", default=False))
        global_steps = 10 if debug else OmegaConf.select(cfg, "manual_cotrain.global_steps")
        configured_eval_every = int(
            OmegaConf.select(
                cfg,
                "manual_cotrain.eval_interval_global_steps",
                default=0,
            )
        )
        eval_every = 1 if debug else configured_eval_every
        save_every = 1 if debug else OmegaConf.select(cfg, "manual_cotrain.checkpoint_every")
        return [
            "[cotrain] "
            f"debug={str(debug).lower()} "
            f"global_steps={global_steps} "
            f"eval_every={eval_every} "
            f"save_every={save_every} "
            f"ngpu={OmegaConf.select(cfg, 'manual_cotrain.ngpu')} "
            f"gpus={env['CUDA_VISIBLE_DEVICES']}",
            "[cotrain] checkpoints "
            f"vla={OmegaConf.select(cfg, 'init.vla_ckpt_path')} "
            f"world_model={OmegaConf.select(cfg, 'init.world_model_state_ckpt')} "
            f"classifier={OmegaConf.select(cfg, 'init.classifier_state_ckpt')}",
        ]

    @staticmethod
    def _component_environment_overrides(values: list[str]) -> None:
        wm_key = "init.world_model_state_ckpt"
        classifier_key = "init.classifier_state_ckpt"
        wm_env = os.environ.get("WORLD_MODEL_CKPT", "").strip()
        classifier_env = os.environ.get("CLASSIFIER_CKPT", "").strip()
        wm_supplied = _has_override(values, wm_key) or bool(wm_env)
        classifier_supplied = _has_override(values, classifier_key) or bool(classifier_env)
        if wm_supplied != classifier_supplied:
            raise ValueError(
                "set both WORLD_MODEL_CKPT and CLASSIFIER_CKPT for a warm start, "
                "or train the missing component with its independent runner first"
            )
        for env_name, raw, hydra_key in (
            ("WORLD_MODEL_CKPT", wm_env, wm_key),
            ("CLASSIFIER_CKPT", classifier_env, classifier_key),
        ):
            if not raw or _has_override(values, hydra_key):
                continue
            path = _existing_path(raw, label=env_name)
            values.append(f"{hydra_key}={_hydra_string(path)}")

    @staticmethod
    def _runtime_environment_overrides(values: list[str]) -> None:
        key = "manual_cotrain.global_steps"
        raw = os.environ.get("WMCLS_COTRAIN_GLOBAL_STEPS", "").strip()
        if not raw or _has_override(values, key):
            return
        try:
            global_steps = int(raw)
        except ValueError as exc:
            raise ValueError("WMCLS_COTRAIN_GLOBAL_STEPS must be a positive integer") from exc
        if global_steps <= 0:
            raise ValueError("WMCLS_COTRAIN_GLOBAL_STEPS must be a positive integer")
        values.append(f"{key}={global_steps}")


def _hydra_string(value: str | Path) -> str:
    return json.dumps(str(value))


def _has_override(values: Sequence[str], key: str) -> bool:
    return any(item.split("=", 1)[0].lstrip("+~") == key for item in values)


def _existing_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not (path.is_file() or path.is_dir()):
        raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
    return path


def _truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _nested_output_dim(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    direct = value.get("output_dim")
    if direct is not None:
        try:
            output_dim = int(direct)
        except (TypeError, ValueError):
            output_dim = 0
        if output_dim in {1, 2}:
            return output_dim
    for key in ("init_args", "classifier", "config"):
        inferred = _nested_output_dim(value.get(key))
        if inferred is not None:
            return inferred
    return None


def _classifier_checkpoint_output_dim(path: str | Path) -> int | None:
    """Read the binary-head contract without constructing the classifier."""

    import torch

    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
        if not config_path.is_file():
            return None
        with config_path.open(encoding="utf-8") as handle:
            return _nested_output_dim(json.load(handle))
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        return None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return None
    state_dicts = payload.get("state_dicts")
    state = payload.get("classifier")
    if state is None and isinstance(state_dicts, Mapping):
        state = state_dicts.get("classifier", state_dicts.get("model"))
    if state is None:
        state = payload.get("model")
    if not isinstance(state, Mapping):
        return _nested_output_dim(payload)
    for name, tensor in state.items():
        if (
            isinstance(name, str)
            and (name == "head.weight" or name.endswith(".head.weight"))
            and isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and int(tensor.shape[0]) in {1, 2}
        ):
            return int(tensor.shape[0])
    return _nested_output_dim(payload)


__all__ = [
    "CotrainLaunchContract",
    "DefaultLaunchContract",
    "LaunchContract",
]
