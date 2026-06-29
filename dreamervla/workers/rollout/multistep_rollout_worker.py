"""Multi-step rollout inference worker for target manual cotrain."""

from __future__ import annotations

import importlib
import time
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
        return self.generate_batch([obs_msg])[0]

    @torch.no_grad()
    def generate_batch(self, obs_msgs: list[ObservationMsg]) -> list[RolloutResultMsg]:
        """Sample action chunks for a batch of observations in one policy call."""

        if not obs_msgs:
            return []
        for obs_msg in obs_msgs:
            if not isinstance(obs_msg, ObservationMsg):
                raise TypeError("generate_batch expects ObservationMsg items")
        hidden_rows: list[torch.Tensor] = []
        encoder_extras: list[dict[str, Any]] = []
        for obs_msg in obs_msgs:
            hidden, encoder_extra = self._hidden_and_encoder_extra(obs_msg)
            hidden_rows.append(_to_device_float_tensor(hidden, self.torch_device).reshape(1, -1))
            encoder_extras.append(dict(encoder_extra))
        hidden_t = torch.cat(hidden_rows, dim=0)

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
        log_prob_cpu = _to_cpu_tensor(log_prob).reshape(len(obs_msgs), -1)
        extra = extra if isinstance(extra, dict) else {}

        results: list[RolloutResultMsg] = []
        for index, obs_msg in enumerate(obs_msgs):
            action_i = action_cpu[index : index + 1]
            forward_inputs = {
                "hidden": hidden_t[index : index + 1].detach().cpu(),
                "action": action_i,
            }
            lang_emb = obs_msg.obs.get(
                "lang_emb",
                encoder_extras[index].get("lang_emb"),
            )
            if lang_emb is not None:
                forward_inputs["lang_emb"] = _to_cpu_tensor(lang_emb)
            for key in _EXTRA_FORWARD_KEYS:
                if key in extra and extra[key] is not None:
                    value = extra[key]
                    if isinstance(value, torch.Tensor) and value.shape[:1] == (len(obs_msgs),):
                        value = value[index : index + 1]
                    forward_inputs[key] = _to_cpu_tensor(value)

            policy_version = int(self.versions.get("policy", 0))
            versions = {
                "policy": policy_version,
                "actor_policy_version": policy_version,
                "rollout_policy_version": policy_version,
                "global_step": int(obs_msg.versions.get("global_step", self.global_step)),
            }
            for name in (
                "world_model_version",
                "wm_version",
                "classifier_version",
                "reward_or_classifier_version",
            ):
                if name in obs_msg.versions:
                    versions[name] = int(obs_msg.versions[name])
            if bool(obs_msg.obs.get("_final_bootstrap", False)):
                versions["final_bootstrap"] = 1

            results.append(
                RolloutResultMsg(
                    env_rank=obs_msg.env_rank,
                    slot_id=obs_msg.slot_id,
                    task_id=obs_msg.task_id,
                    episode_id=obs_msg.episode_id,
                    step=obs_msg.step,
                    actions=_squeeze_batch(action_i),
                    prev_logprobs=log_prob_cpu[index],
                    prev_values=None,
                    forward_inputs=forward_inputs,
                    versions=versions,
                )
            )
        return results

    def generate(
        self,
        input_channel_name: str,
        output_channel_name: str,
        num_slots: int | None = None,
        input_key: str | None = None,
    ) -> dict[str, float]:
        """Drain observations from a named channel until a stop message arrives."""

        input_channel = Channel.connect(input_channel_name)
        output_channel = Channel.connect(output_channel_name)
        if input_key is not None:
            return self._generate_from_key(input_channel, output_channel, str(input_key))
        if num_slots is not None:
            return self._generate_from_rank_slot_keys(
                input_channel,
                output_channel,
                int(num_slots),
            )

        return self._generate_from_key(input_channel, output_channel, "default")

    def _generate_from_key(
        self,
        input_channel: Channel,
        output_channel: Channel,
        key: str,
    ) -> dict[str, float]:
        generated = 0
        channel_get_s = 0.0
        policy_forward_s = 0.0
        channel_put_s = 0.0
        loop_start = time.perf_counter()
        while True:
            get_start = time.perf_counter()
            msg = input_channel.get(key=str(key))
            channel_get_s += time.perf_counter() - get_start
            if isinstance(msg, StopMsg):
                break
            if not isinstance(msg, ObservationMsg):
                raise TypeError(
                    "MultiStepRolloutWorker.generate expected ObservationMsg or StopMsg, "
                    f"got {type(msg).__name__}"
                )
            _sync_if_cuda(self.torch_device)
            forward_start = time.perf_counter()
            result = self._generate_once_with_context(msg, key=str(key))
            _sync_if_cuda(self.torch_device)
            policy_forward_s += time.perf_counter() - forward_start
            put_start = time.perf_counter()
            output_channel.put(result, key=result.key)
            channel_put_s += time.perf_counter() - put_start
            generated += 1
        return _rollout_loop_metrics(
            generated=generated,
            loop_s=time.perf_counter() - loop_start,
            channel_get_s=channel_get_s,
            policy_forward_s=policy_forward_s,
            channel_put_s=channel_put_s,
        )

    def _generate_from_rank_slot_keys(
        self,
        input_channel: Channel,
        output_channel: Channel,
        num_slots: int,
    ) -> dict[str, float]:
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        active_slots = set(range(int(num_slots)))
        generated = 0
        slot_cursor = 0
        channel_get_s = 0.0
        policy_forward_s = 0.0
        channel_put_s = 0.0
        loop_start = time.perf_counter()
        while active_slots:
            messages: list[tuple[str, ObservationMsg]] = []
            for _ in range(len(active_slots)):
                slot_id = slot_cursor % int(num_slots)
                slot_cursor += 1
                if slot_id not in active_slots:
                    continue
                key = f"{int(self.rank)}:{int(slot_id)}"
                get_start = time.perf_counter()
                msg = input_channel.get(key=key)
                channel_get_s += time.perf_counter() - get_start
                if isinstance(msg, StopMsg):
                    active_slots.remove(slot_id)
                    continue
                if not isinstance(msg, ObservationMsg):
                    raise TypeError(
                        "MultiStepRolloutWorker.generate expected ObservationMsg or StopMsg, "
                        f"got {type(msg).__name__}"
                    )
                messages.append((key, msg))
            if not messages:
                continue
            _sync_if_cuda(self.torch_device)
            forward_start = time.perf_counter()
            results = self._generate_batch_with_context(
                [msg for _, msg in messages],
                keys=[key for key, _ in messages],
            )
            _sync_if_cuda(self.torch_device)
            policy_forward_s += time.perf_counter() - forward_start
            for result in results:
                put_start = time.perf_counter()
                output_channel.put(result, key=result.key)
                channel_put_s += time.perf_counter() - put_start
                generated += 1
        return _rollout_loop_metrics(
            generated=generated,
            loop_s=time.perf_counter() - loop_start,
            channel_get_s=channel_get_s,
            policy_forward_s=policy_forward_s,
            channel_put_s=channel_put_s,
        )

    def _generate_once_with_context(
        self,
        msg: ObservationMsg,
        *,
        key: str,
    ) -> RolloutResultMsg:
        try:
            return self.generate_once(msg)
        except Exception as exc:
            raise RuntimeError(
                "RolloutWorker.generate failed "
                f"rank={int(self.rank)} key={str(key)} "
                f"env_rank={int(msg.env_rank)} slot_id={int(msg.slot_id)} "
                f"episode_id={int(msg.episode_id)} step={int(msg.step)}: {exc}"
            ) from exc

    def _generate_batch_with_context(
        self,
        msgs: list[ObservationMsg],
        *,
        keys: list[str],
    ) -> list[RolloutResultMsg]:
        try:
            return self.generate_batch(msgs)
        except Exception as exc:
            details = ",".join(
                f"{key}/env={int(msg.env_rank)}/slot={int(msg.slot_id)}/ep={int(msg.episode_id)}/step={int(msg.step)}"
                for key, msg in zip(keys, msgs, strict=True)
            )
            raise RuntimeError(
                f"RolloutWorker.generate_batch failed rank={int(self.rank)} keys={details}: {exc}"
            ) from exc

    def sync_model_from_actor(
        self,
        key: str = "policy",
        local_version: int | None = None,
    ) -> dict[str, float]:
        """Apply the latest ActorGroup policy patch to the local rollout copy."""

        resolved_local_version = (
            int(self.versions.get(str(key), 0))
            if local_version is None
            else int(local_version)
        )
        pull_start = time.perf_counter()
        syncer = self._syncer()
        version = syncer.pull(str(key), self._policy(), resolved_local_version)
        pull_s = float(time.perf_counter() - pull_start)
        if version is not None:
            self.versions[str(key)] = int(version)
        metrics = {
            f"sync/rollout_{key}_pull_s": pull_s,
            f"sync/rollout_{key}_version": float(
                resolved_local_version if version is None else int(version)
            ),
            f"sync/rollout_{key}_updated": float(version is not None),
        }
        metrics.update(dict(getattr(syncer, "last_pull_metrics", {}) or {}))
        return metrics

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


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rollout_loop_metrics(
    *,
    generated: int,
    loop_s: float,
    channel_get_s: float,
    policy_forward_s: float,
    channel_put_s: float,
) -> dict[str, float]:
    generated_f = float(generated)
    loop_s = float(loop_s)
    return {
        "rollout/generated": generated_f,
        "rollout/loop_s": loop_s,
        "rollout/channel_get_s": float(channel_get_s),
        "rollout/policy_forward_s": float(policy_forward_s),
        "rollout/channel_put_s": float(channel_put_s),
        "rollout/generated_per_s": generated_f / max(loop_s, 1e-9),
    }


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
        return torch.from_numpy(np.array(value, copy=True)).detach().cpu()
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
