"""Ray LearnerWorker for optional online cotrain backend."""

from __future__ import annotations

import importlib
from typing import Any

import ray
import torch
from torch import nn

from dreamervla.hybrid_engines.weight_syncer.objectstore import ObjectStoreWeightSyncer
from dreamervla.scheduler.worker import Worker


class LearnerWorker(Worker):
    """Single-GPU learner actor.

    The first implementation includes a lightweight ``synthetic_ppo`` update
    path used by Ray e2e tests. Real DreamerVLA WM/CLS/RL phases use the same
    public ``update`` and ``sync_weights`` boundaries.
    """

    def __init__(
        self,
        model_cfg: dict[str, Any],
        init_ckpt: dict[str, Any],
        train_cfg: dict[str, Any],
        replay: Any,
    ) -> None:
        super().__init__()
        self.model_cfg = dict(model_cfg)
        self.init_ckpt = dict(init_ckpt)
        self.train_cfg = dict(train_cfg)
        self.replay = replay
        self.torch_device = torch.device(str(self.train_cfg.get("device", self.device)))
        self.policy: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.syncer: ObjectStoreWeightSyncer | None = None

    def init(self) -> None:
        self.policy = _build_from_cfg(self.model_cfg["policy"]).to(self.torch_device)
        if "policy" in self.init_ckpt:
            self.policy.load_state_dict(_to_device_state(self.init_ckpt["policy"], self.torch_device))
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(self.train_cfg.get("lr", 1e-3)),
        )
        syncer_cfg = dict(self.train_cfg.get("syncer", {}))
        self.syncer = ObjectStoreWeightSyncer(**syncer_cfg)

    def update(self, phase: str, num_steps: int) -> dict[str, float]:
        mode = str(self.train_cfg.get("mode", "synthetic_ppo"))
        if mode != "synthetic_ppo":
            raise NotImplementedError(
                "LearnerWorker currently implements synthetic_ppo; real DreamerVLA "
                "phase wrappers will plug in at this boundary."
            )
        if str(phase) != "rl":
            return {f"train/{phase}_loss": 0.0}
        return self._synthetic_ppo_update(int(num_steps))

    def sync_weights(self, what: str, version: int) -> None:
        if str(what) != "policy":
            raise ValueError("synthetic LearnerWorker only syncs policy weights")
        self._syncer().push("policy", self._policy().state_dict(), int(version))

    def state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        return {"policy": _cpu_state_dict(self._policy())}

    def _synthetic_ppo_update(self, num_steps: int) -> dict[str, float]:
        policy = self._policy()
        optimizer = self._optimizer()
        batch_size = int(self.train_cfg.get("batch_size", 2))
        last_loss = 0.0
        for _ in range(max(1, int(num_steps))):
            batch = ray.get(self.replay.sample.remote(batch_size))
            obs_embedding = batch["obs_embedding"].to(self.torch_device).float()
            target = batch["current_actions"].to(self.torch_device).float().mean(dim=1)
            hidden = obs_embedding.mean(dim=1)
            pred = _predict(policy, hidden)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        return {"train/rl_loss": last_loss}

    def _policy(self) -> nn.Module:
        if self.policy is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.policy

    def _optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.optimizer

    def _syncer(self) -> ObjectStoreWeightSyncer:
        if self.syncer is None:
            raise RuntimeError("LearnerWorker.init() has not been called")
        return self.syncer


def _build_from_cfg(cfg: dict[str, Any]) -> Any:
    target = cfg.get("target") or cfg.get("_target_") or cfg.get("class_path")
    if not target:
        raise ValueError("component config must include target/_target_/class_path")
    kwargs = dict(cfg.get("kwargs", {}))
    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


def _predict(policy: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(policy, "predict"):
        return policy.predict(hidden)
    return policy(hidden)


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _to_device_state(state_dict: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: (value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)).to(device)
        for key, value in state_dict.items()
    }
