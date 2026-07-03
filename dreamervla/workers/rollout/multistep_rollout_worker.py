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
from dreamervla.workers.cotrain.handshake_trace import trace as _hs_trace
from dreamervla.workers.cotrain.messages import (
    ObservationBatchMsg,
    ObservationMsg,
    RolloutResultBatchMsg,
    RolloutResultMsg,
    StopMsg,
    pack_rollout_result_batch,
    rollout_result_batch_to_messages,
)

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
    def generate_batch(
        self,
        obs_msgs: list[ObservationMsg],
        *,
        batched_obs: dict[str, Any] | None = None,
    ) -> list[RolloutResultMsg]:
        """Sample action chunks for a batch of observations in one policy call."""

        return rollout_result_batch_to_messages(
            self.generate_result_batch(obs_msgs, batched_obs=batched_obs)
        )

    @torch.no_grad()
    def generate_result_batch(
        self,
        obs_msgs: list[ObservationMsg],
        *,
        batched_obs: dict[str, Any] | None = None,
    ) -> RolloutResultBatchMsg:
        """Sample action chunks and return one rank-scoped batched result."""

        if not obs_msgs:
            return RolloutResultBatchMsg(env_rank=int(self.rank), results=[])
        for obs_msg in obs_msgs:
            if not isinstance(obs_msg, ObservationMsg):
                raise TypeError("generate_batch expects ObservationMsg items")
        env_rank = int(obs_msgs[0].env_rank)
        for obs_msg in obs_msgs:
            if int(obs_msg.env_rank) != env_rank:
                raise ValueError("generate_result_batch requires one env_rank")
        encoder_extras: list[dict[str, Any]] = []
        batched_hidden = _batched_hidden_from_obs(batched_obs)
        if batched_hidden is not None:
            hidden_t = _to_device_float_tensor(
                batched_hidden,
                self.torch_device,
            ).reshape(len(obs_msgs), -1)
            encoder_extras = [{} for _ in obs_msgs]
        else:
            hidden_values: list[Any] = []
            for obs_msg in obs_msgs:
                hidden, encoder_extra = self._hidden_and_encoder_extra(obs_msg)
                hidden_values.append(hidden)
                encoder_extras.append(dict(encoder_extra))
            hidden_t = _to_device_float_batch(hidden_values, self.torch_device)

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

        forward_inputs: dict[str, Any] = {"action": action_cpu}
        if batched_hidden is None:
            # Encoder-derived hidden only exists here; obs-provided hidden is
            # already held by the env worker, so echoing it back would just
            # duplicate the largest tensor on the rollout->env channel.
            forward_inputs["hidden"] = hidden_t.detach().cpu()
        lang_emb = _batched_forward_input(
            obs_msgs,
            batched_obs=batched_obs,
            encoder_extras=encoder_extras,
            key="lang_emb",
        )
        if lang_emb is not None:
            forward_inputs["lang_emb"] = lang_emb
        for key in _EXTRA_FORWARD_KEYS:
            if key in extra and extra[key] is not None:
                forward_inputs[key] = _batch_extra_forward_value(
                    extra[key],
                    batch_size=len(obs_msgs),
                )

        policy_version = int(self.versions.get("policy", 0))
        version_dicts: list[dict[str, int]] = []
        for index, obs_msg in enumerate(obs_msgs):
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
            if index > 0 and set(versions) != set(version_dicts[0]):
                raise ValueError("batched rollout results must share version keys")
            version_dicts.append(versions)
        version_keys = sorted(version_dicts[0])
        return RolloutResultBatchMsg(
            env_rank=env_rank,
            results=[],
            slot_ids=[int(obs_msg.slot_id) for obs_msg in obs_msgs],
            task_ids=[int(obs_msg.task_id) for obs_msg in obs_msgs],
            episode_ids=[int(obs_msg.episode_id) for obs_msg in obs_msgs],
            steps=[int(obs_msg.step) for obs_msg in obs_msgs],
            actions=action_cpu,
            prev_logprobs=log_prob_cpu,
            prev_values=None,
            forward_inputs=forward_inputs,
            versions={
                key: torch.as_tensor(
                    [versions[key] for versions in version_dicts],
                    dtype=torch.long,
                )
                for key in version_keys
            },
        )

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
            return self._generate_from_rank_batch_key(
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
            _hs_trace(
                f"[rollout rank={int(self.rank)}] recv action request WAIT key={key}"
            )
            get_start = time.perf_counter()
            msg = input_channel.get(key=str(key))
            channel_get_s += time.perf_counter() - get_start
            if isinstance(msg, StopMsg):
                _hs_trace(
                    f"[rollout rank={int(self.rank)}] recv StopMsg key={key}"
                )
                break
            if not isinstance(msg, ObservationMsg):
                raise TypeError(
                    "MultiStepRolloutWorker.generate expected ObservationMsg or StopMsg, "
                    f"got {type(msg).__name__}"
                )
            _hs_trace(
                f"[rollout rank={int(self.rank)}] recv action request "
                f"batch_size=1 key={key}"
            )
            _sync_if_cuda(self.torch_device)
            _hs_trace(
                f"[rollout rank={int(self.rank)}] policy forward start batch_size=1"
            )
            forward_start = time.perf_counter()
            result = self._generate_once_with_context(msg, key=str(key))
            _sync_if_cuda(self.torch_device)
            policy_forward_s += time.perf_counter() - forward_start
            _hs_trace(
                f"[rollout rank={int(self.rank)}] policy forward done batch_size=1"
            )
            put_start = time.perf_counter()
            output_channel.put(result, key=result.key)
            channel_put_s += time.perf_counter() - put_start
            _hs_trace(
                f"[rollout rank={int(self.rank)}] send action response batch_size=1"
            )
            generated += 1
        _hs_trace(
            f"[rollout rank={int(self.rank)}] generate exit generated={int(generated)}"
        )
        return _rollout_loop_metrics(
            generated=generated,
            loop_s=time.perf_counter() - loop_start,
            channel_get_s=channel_get_s,
            policy_forward_s=policy_forward_s,
            channel_put_s=channel_put_s,
        )

    def _generate_from_rank_batch_key(
        self,
        input_channel: Channel,
        output_channel: Channel,
        num_slots: int,
    ) -> dict[str, float]:
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        key = str(int(self.rank))
        generated = 0
        channel_get_s = 0.0
        policy_forward_s = 0.0
        channel_put_s = 0.0
        loop_start = time.perf_counter()
        while True:
            _hs_trace(
                f"[rollout rank={int(self.rank)}] recv action request WAIT key={key}"
            )
            get_start = time.perf_counter()
            msg = input_channel.get(key=key)
            channel_get_s += time.perf_counter() - get_start
            if isinstance(msg, StopMsg):
                _hs_trace(f"[rollout rank={int(self.rank)}] recv StopMsg key={key}")
                break
            if not isinstance(msg, ObservationBatchMsg):
                raise TypeError(
                    "MultiStepRolloutWorker.generate expected ObservationBatchMsg "
                    f"or StopMsg, got {type(msg).__name__}"
                )
            if int(msg.env_rank) != int(self.rank):
                raise ValueError(
                    "observation batch env_rank must match rollout rank: "
                    f"got {int(msg.env_rank)}, expected {int(self.rank)}"
                )
            observations = list(msg.observations)
            if not observations:
                raise ValueError("ObservationBatchMsg.observations must not be empty")
            if len(observations) > int(num_slots):
                raise ValueError(
                    "observation batch is larger than configured num_slots: "
                    f"{len(observations)} > {int(num_slots)}"
                )
            for obs_msg in observations:
                if not isinstance(obs_msg, ObservationMsg):
                    raise TypeError(
                        "ObservationBatchMsg must contain ObservationMsg items, "
                        f"got {type(obs_msg).__name__}"
                    )
                if int(obs_msg.env_rank) != int(self.rank):
                    raise ValueError(
                        "observation env_rank must match rollout rank: "
                        f"got {int(obs_msg.env_rank)}, expected {int(self.rank)}"
                    )
            keys = [obs_msg.key for obs_msg in observations]
            keys_csv = ",".join(keys)
            _hs_trace(
                f"[rollout rank={int(self.rank)}] recv action request OK key={key}"
            )
            _hs_trace(
                f"[rollout rank={int(self.rank)}] recv action request "
                f"batch_size={len(observations)} key={key} keys={keys_csv}"
            )
            _sync_if_cuda(self.torch_device)
            _hs_trace(
                f"[rollout rank={int(self.rank)}] policy forward start "
                f"batch_size={len(observations)}"
            )
            forward_start = time.perf_counter()
            result_batch = self._generate_result_batch_with_context(
                observations,
                keys=keys,
                batched_obs=msg.batched_obs,
            )
            _sync_if_cuda(self.torch_device)
            policy_forward_s += time.perf_counter() - forward_start
            _hs_trace(
                f"[rollout rank={int(self.rank)}] policy forward done "
                f"batch_size={len(observations)}"
            )
            put_start = time.perf_counter()
            output_channel.put(result_batch, key=result_batch.key)
            channel_put_s += time.perf_counter() - put_start
            result_count = len(result_batch.slot_ids or result_batch.results)
            generated += result_count
            result_keys_csv = ",".join(keys)
            _hs_trace(
                f"[rollout rank={int(self.rank)}] send action response "
                f"batch_size={result_count} key={result_batch.key} "
                f"keys={result_keys_csv}"
            )
        _hs_trace(
            f"[rollout rank={int(self.rank)}] generate exit generated={int(generated)}"
        )
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
        batched_obs: dict[str, Any] | None = None,
    ) -> list[RolloutResultMsg]:
        try:
            if batched_obs is not None:
                return self.generate_batch(msgs, batched_obs=batched_obs)
            return self.generate_batch(msgs)
        except Exception as exc:
            details = ",".join(
                f"{key}/env={int(msg.env_rank)}/slot={int(msg.slot_id)}/ep={int(msg.episode_id)}/step={int(msg.step)}"
                for key, msg in zip(keys, msgs, strict=True)
            )
            raise RuntimeError(
                f"RolloutWorker.generate_batch failed rank={int(self.rank)} keys={details}: {exc}"
            ) from exc

    def _generate_result_batch_with_context(
        self,
        msgs: list[ObservationMsg],
        *,
        keys: list[str],
        batched_obs: dict[str, Any] | None = None,
    ) -> RolloutResultBatchMsg:
        try:
            return self.generate_result_batch(msgs, batched_obs=batched_obs)
        except Exception as exc:
            details = ",".join(
                f"{key}/env={int(msg.env_rank)}/slot={int(msg.slot_id)}/ep={int(msg.episode_id)}/step={int(msg.step)}"
                for key, msg in zip(keys, msgs, strict=True)
            )
            raise RuntimeError(
                f"RolloutWorker.generate_result_batch failed rank={int(self.rank)} keys={details}: {exc}"
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


def _batched_hidden_from_obs(batched_obs: dict[str, Any] | None) -> Any | None:
    if not isinstance(batched_obs, dict):
        return None
    for key in ("obs_embedding", "hidden", "latent"):
        if key in batched_obs:
            return batched_obs[key]
    return None


def _batched_obs_row(
    batched_obs: dict[str, Any] | None,
    key: str,
    *,
    index: int,
    batch_size: int,
) -> Any | None:
    if not isinstance(batched_obs, dict) or key not in batched_obs:
        return None
    value = batched_obs[key]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ValueError(f"batched_obs[{key!r}] must include a batch dimension")
        if int(value.shape[0]) != int(batch_size):
            raise ValueError(
                f"batched_obs[{key!r}] batch size mismatch: "
                f"got {int(value.shape[0])}, expected {int(batch_size)}"
            )
        return value[int(index)]
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"batched_obs[{key!r}] must include a batch dimension")
    if int(array.shape[0]) != int(batch_size):
        raise ValueError(
            f"batched_obs[{key!r}] batch size mismatch: "
            f"got {int(array.shape[0])}, expected {int(batch_size)}"
        )
    return array[int(index)]


def _batched_forward_input(
    obs_msgs: list[ObservationMsg],
    *,
    batched_obs: dict[str, Any] | None,
    encoder_extras: list[dict[str, Any]],
    key: str,
) -> torch.Tensor | None:
    batch_size = len(obs_msgs)
    if isinstance(batched_obs, dict) and key in batched_obs:
        value = _to_cpu_tensor(batched_obs[key])
        if value.ndim == 0:
            raise ValueError(f"batched_obs[{key!r}] must include a batch dimension")
        if int(value.shape[0]) != int(batch_size):
            raise ValueError(
                f"batched_obs[{key!r}] batch size mismatch: "
                f"got {int(value.shape[0])}, expected {int(batch_size)}"
            )
        return value

    values: list[Any] = []
    for index, obs_msg in enumerate(obs_msgs):
        value = obs_msg.obs.get(key, encoder_extras[index].get(key))
        if value is None:
            return None
        values.append(value)
    rows = [_to_cpu_tensor(value) for value in values]
    shape = tuple(rows[0].shape)
    if any(tuple(row.shape) != shape for row in rows):
        raise ValueError(f"forward input {key!r} values must share shape")
    return torch.cat([row.reshape(1, *shape) for row in rows], dim=0)


def _batch_extra_forward_value(value: Any, *, batch_size: int) -> torch.Tensor:
    tensor = _to_cpu_tensor(value)
    if tensor.ndim > 0 and int(tensor.shape[0]) == int(batch_size):
        return tensor
    shape = tuple(tensor.shape)
    return torch.cat(
        [tensor.reshape(1, *shape) for _ in range(int(batch_size))],
        dim=0,
    )


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
    if hasattr(value, "lang_emb") and value.lang_emb is not None:
        extra["lang_emb"] = value.lang_emb
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
        tensor = value.detach()
        if tensor.is_floating_point():
            return tensor.to(device=device)
        return tensor.to(device=device, dtype=torch.float32)
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float32)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if not array.flags.writeable:
            array = array.copy()
        return torch.from_numpy(array).to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _to_device_float_batch(values: list[Any], device: torch.device) -> torch.Tensor:
    if not values:
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    if all(isinstance(value, np.ndarray) for value in values):
        arrays = [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in values
        ]
        return torch.from_numpy(np.stack(arrays, axis=0)).to(
            device=device,
            dtype=torch.float32,
        )
    if all(isinstance(value, torch.Tensor) for value in values):
        return torch.stack(
            [value.detach().reshape(-1) for value in values],
            dim=0,
        ).to(device=device)
    return torch.stack(
        [_to_device_float_tensor(value, device).reshape(-1) for value in values],
        dim=0,
    )


def _squeeze_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 0 and int(value.shape[0]) == 1:
        return value.squeeze(0)
    return value


__all__ = ["MultiStepRolloutWorker"]
