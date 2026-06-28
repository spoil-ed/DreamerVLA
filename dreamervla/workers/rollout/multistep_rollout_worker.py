"""Multi-step rollout inference worker for target manual cotrain."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from dreamervla.hybrid_engines.weight_syncer import PatchWeightSyncer
from dreamervla.scheduler.channel import Channel
from dreamervla.scheduler.worker import Worker
from dreamervla.workers.cotrain.messages import ObservationMsg, RolloutResultMsg, StopMsg

_DEFAULT_PATCH_STORE = "DreamerVLAActorRolloutPatchStore"
_EXTRA_FORWARD_KEYS = (
    "action_token_ids",
    "input_ids",
    "attention_mask",
    "hidden_states",
)


class MultiStepRolloutWorker(Worker):
    """Non-FSDP no-grad policy copy used by the target RolloutGroup."""

    def __init__(
        self,
        policy_cfg: Any,
        encoder_cfg: Any | None = None,
        init_ckpt: Any | None = None,
        train_cfg: Any | None = None,
    ) -> None:
        super().__init__()
        self.policy_cfg = _as_plain_dict(policy_cfg)
        self.encoder_cfg = _as_plain_dict(encoder_cfg) if encoder_cfg else None
        self.init_ckpt = _as_plain_dict(init_ckpt)
        self.train_cfg = _as_plain_dict(train_cfg)
        configured_device = str(self.train_cfg.get("device", self.device))
        if configured_device == "auto":
            configured_device = self.device
        self.torch_device = torch.device(configured_device)
        self.encoder: Any | None = None
        self.policy: nn.Module | None = None
        self.syncer: PatchWeightSyncer | None = None
        self.extractors: dict[str, Any] = {}
        self.global_step = 0
        self.versions: dict[str, int] = {
            "policy": int(self.train_cfg.get("policy_version", 0))
        }

    def init(self) -> None:
        """Build the local inference copy and load optional initial policy weights."""

        if self.encoder_cfg is not None:
            self.encoder = _build_from_cfg(self.encoder_cfg)
            if hasattr(self.encoder, "to"):
                self.encoder.to(self.torch_device)

        policy = _build_from_cfg(self.policy_cfg)
        if not isinstance(policy, nn.Module):
            raise TypeError("MultiStepRolloutWorker policy must be a torch.nn.Module")
        policy.to(self.torch_device)
        if "policy" in self.init_ckpt:
            policy.load_state_dict(
                _to_device_state(self.init_ckpt["policy"], self.torch_device)
            )
        policy.eval()
        self.policy = policy

    @torch.no_grad()
    def generate_once(self, obs_msg: ObservationMsg) -> RolloutResultMsg:
        """Sample one action chunk and return ActorGroup training inputs."""

        if not isinstance(obs_msg, ObservationMsg):
            raise TypeError("generate_once expects an ObservationMsg")
        hidden, encoder_extra = self._hidden_and_encoder_extra(obs_msg)
        hidden_t = _to_device_float_tensor(hidden, self.torch_device).reshape(1, -1)

        policy = self._policy()
        action, log_prob, extra = policy(
            {
                "mode": "sample",
                "hidden": hidden_t,
                "return_chunk": True,
                "deterministic": bool(self.train_cfg.get("deterministic", False)),
            }
        )
        action_cpu = _to_cpu_tensor(action)
        forward_inputs = {
            "hidden": hidden_t.detach().cpu(),
            "action": action_cpu,
        }
        lang_emb = obs_msg.obs.get("lang_emb", encoder_extra.get("lang_emb"))
        if lang_emb is not None:
            forward_inputs["lang_emb"] = _to_cpu_tensor(lang_emb)
        if isinstance(extra, dict):
            for key in _EXTRA_FORWARD_KEYS:
                if key in extra and extra[key] is not None:
                    forward_inputs[key] = _to_cpu_tensor(extra[key])
        versions = {"policy": int(self.versions.get("policy", 0))}
        if bool(obs_msg.obs.get("_final_bootstrap", False)):
            versions["final_bootstrap"] = 1

        return RolloutResultMsg(
            env_rank=obs_msg.env_rank,
            slot_id=obs_msg.slot_id,
            task_id=obs_msg.task_id,
            episode_id=obs_msg.episode_id,
            step=obs_msg.step,
            actions=_squeeze_batch(action_cpu),
            prev_logprobs=_to_cpu_tensor(log_prob).reshape(-1),
            prev_values=None,
            forward_inputs=forward_inputs,
            versions=versions,
        )

    def generate(
        self,
        input_channel_name: str,
        output_channel_name: str,
    ) -> dict[str, float]:
        """Drain observations from a named channel until a stop message arrives."""

        input_channel = Channel.connect(input_channel_name)
        output_channel = Channel.connect(output_channel_name)
        generated = 0
        while True:
            msg = input_channel.get()
            if isinstance(msg, StopMsg):
                break
            if not isinstance(msg, ObservationMsg):
                raise TypeError(
                    "MultiStepRolloutWorker.generate expected ObservationMsg or StopMsg, "
                    f"got {type(msg).__name__}"
                )
            result = self.generate_once(msg)
            output_channel.put(result, key=result.key)
            generated += 1
        return {"rollout/generated": float(generated)}

    def sync_model_from_actor(
        self,
        key: str = "policy",
        local_version: int | None = None,
    ) -> int | None:
        """Apply the latest ActorGroup policy patch to the local rollout copy."""

        resolved_local_version = (
            int(self.versions.get(str(key), 0))
            if local_version is None
            else int(local_version)
        )
        version = self._syncer().pull(str(key), self._policy(), resolved_local_version)
        if version is not None:
            self.versions[str(key)] = int(version)
        return version

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU policy state dict."""

        return {
            name: value.detach().cpu().clone()
            for name, value in self._policy().state_dict().items()
        }

    def set_global_step(self, global_step: int) -> None:
        """Set runner-visible progress metadata."""

        self.global_step = int(global_step)

    def _hidden_and_encoder_extra(
        self,
        obs_msg: ObservationMsg,
    ) -> tuple[Any, dict[str, Any]]:
        try:
            return _hidden_from_obs(obs_msg.obs), {}
        except ValueError:
            if self.encoder is None:
                raise
        return self._encode_observation(obs_msg)

    def _encode_observation(self, obs_msg: ObservationMsg) -> tuple[Any, dict[str, Any]]:
        encoder = self._encoder()
        obs = dict(obs_msg.obs)
        task_description = str(obs.get("task_description", obs.get("language", "")))

        extractor = self._extractor_for(obs_msg)
        if extractor is not None:
            if bool(obs.get("is_first", False)):
                reset = getattr(extractor, "reset", None)
                if reset is not None:
                    reset()
            if hasattr(encoder, "predict_batch") and hasattr(extractor, "prepare"):
                decoded = encoder.predict_batch([extractor.prepare(obs, task_description)])[0]
                return _hidden_and_extra_from_encoded(decoded)
            step = getattr(extractor, "step", None)
            if step is not None:
                return _hidden_and_extra_from_encoded(step(obs, task_description))
            if callable(extractor):
                return _hidden_and_extra_from_encoded(extractor(obs, task_description))

        encode = getattr(encoder, "encode_observation", None)
        if encode is None:
            encode = getattr(encoder, "encode_obs", None)
        if encode is not None:
            return _hidden_and_extra_from_encoded(encode(obs, task_description))
        if callable(encoder):
            return _hidden_and_extra_from_encoded(encoder(obs, task_description))
        raise TypeError(
            "rollout encoder must expose make_extractor(), encode_observation(), "
            "encode_obs(), or be callable"
        )

    def _extractor_for(self, obs_msg: ObservationMsg) -> Any | None:
        encoder = self._encoder()
        make_extractor = getattr(encoder, "make_extractor", None)
        if make_extractor is None:
            return None
        key = obs_msg.key
        if key not in self.extractors:
            self.extractors[key] = make_extractor()
        return self.extractors[key]

    def _policy(self) -> nn.Module:
        if self.policy is None:
            raise RuntimeError("MultiStepRolloutWorker.init() has not been called")
        return self.policy

    def _encoder(self) -> Any:
        if self.encoder is None:
            raise RuntimeError("MultiStepRolloutWorker has no encoder configured")
        return self.encoder

    def _syncer(self) -> PatchWeightSyncer:
        if self.syncer is None:
            syncer_cfg = _as_plain_dict(self.train_cfg.get("syncer", {}))
            store_name = str(syncer_cfg.get("store_name", _DEFAULT_PATCH_STORE))
            self.syncer = PatchWeightSyncer(store_name=store_name)
        return self.syncer


def _obs_embedding_from_obs(obs: dict[str, Any]) -> Any:
    if "obs_embedding" in obs:
        return obs["obs_embedding"]
    if "latent" in obs:
        return obs["latent"]
    raise ValueError("ObservationMsg.obs must include obs_embedding or latent")


_hidden_from_obs = _obs_embedding_from_obs


def _hidden_and_extra_from_encoded(value: Any) -> tuple[Any, dict[str, Any]]:
    extra: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in ("obs_embedding", "hidden", "latent"):
            if key in value:
                if value.get("lang_emb") is not None:
                    extra["lang_emb"] = value["lang_emb"]
                return value[key], extra
        raise ValueError(
            "encoded observation mapping must include obs_embedding, hidden, or latent"
        )
    if hasattr(value, "lang_emb") and getattr(value, "lang_emb") is not None:
        extra["lang_emb"] = getattr(value, "lang_emb")
    try:
        hidden = value[1]
    except (TypeError, IndexError) as exc:
        raise ValueError(
            "encoded observation must be a mapping or tuple-compatible "
            "(action_chunk, hidden) output"
        ) from exc
    return hidden, extra


def _build_from_cfg(cfg: dict[str, Any]) -> Any:
    target = cfg.get("target") or cfg.get("_target_") or cfg.get("class_path")
    if not target:
        raise ValueError("component config must include target/_target_/class_path")

    kwargs = {
        key: value
        for key, value in cfg.items()
        if key not in {"target", "_target_", "class_path", "kwargs", "init_args"}
    }
    kwargs.update(_as_plain_dict(cfg.get("init_args", {})))
    kwargs.update(_as_plain_dict(cfg.get("kwargs", {})))

    if ":" in str(target):
        module_name, class_name = str(target).split(":", 1)
    else:
        module_name, class_name = str(target).rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True) or {})
    return dict(value)


def _to_device_state(value: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        str(name): torch.as_tensor(tensor).to(device)
        for name, tensor in dict(value).items()
    }


def _to_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _to_device_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, copy=True)).to(
            device=device,
            dtype=torch.float32,
        )
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _squeeze_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 0 and int(value.shape[0]) == 1:
        return value.squeeze(0)
    return value


__all__ = ["MultiStepRolloutWorker"]
