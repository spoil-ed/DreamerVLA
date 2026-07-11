from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from dreamervla.runners.base_runner import BaseRunner


@dataclass(frozen=True)
class _RunnerSpec:
    module: str
    implementation: str
    runner_name: str | None = None
    runner_status: str | None = None
    runner_family: str | None = None


_RUNNER_SPECS: dict[str, _RunnerSpec] = {
    "JointDreamerVLARunner": _RunnerSpec(
        "dreamervla.runners.dreamervla_runner",
        "DreamerVLARunner",
        runner_name="joint_dreamervla",
        runner_status="follow_up",
        runner_family="actor",
    ),
    "EmbodiedEvalRunner": _RunnerSpec(
        "dreamervla.runners.embodied_eval_runner",
        "EmbodiedEvalRunner",
        runner_name="embodied_eval",
        runner_status="current",
        runner_family="eval",
    ),
    "LatentClassifierRunner": _RunnerSpec(
        "dreamervla.runners.latent_classifier_runner",
        "LatentClassifierRunner",
        runner_name="latent_classifier",
        runner_status="current",
        runner_family="reward",
    ),
    "OnlineCotrainRunner": _RunnerSpec(
        "dreamervla.runners.online_cotrain_runner",
        "OnlineCotrainRunner",
        runner_name="online_cotrain",
        runner_status="current",
        runner_family="actor",
    ),
    "CollectRolloutsRunner": _RunnerSpec(
        "dreamervla.runners.collect_rollouts_runner",
        "CollectRolloutsRunner",
        runner_name="collect_rollouts",
        runner_status="current",
        runner_family="rollout",
    ),
    "OnlineCotrainPipelineRunner": _RunnerSpec(
        "dreamervla.runners.online_cotrain_pipeline_runner",
        "OnlineCotrainPipelineRunner",
        runner_name="online_cotrain_pipeline",
        runner_status="current",
        runner_family="actor",
    ),
    "OnlineCotrainRayRunner": _RunnerSpec(
        "dreamervla.runners.online_cotrain_ray_runner",
        "OnlineCotrainRayRunner",
        runner_name="online_cotrain_ray",
        runner_status="optional",
        runner_family="actor",
    ),
    "ManualCotrainRayRunner": _RunnerSpec(
        "dreamervla.runners.manual_cotrain_ray_runner",
        "ManualCotrainRayRunner",
        runner_name="manual_cotrain_ray",
        runner_status="current",
        runner_family="actor",
    ),
    "ColdStartRayCollectRunner": _RunnerSpec(
        "dreamervla.runners.cold_start_ray_collect_runner",
        "ColdStartRayCollectRunner",
        runner_name="collect_rollouts_ray",
        runner_status="optional",
        runner_family="rollout",
    ),
    "FrozenModelPolicyRunner": _RunnerSpec(
        "dreamervla.runners.frozen_model_policy_runner",
        "FrozenModelPolicyRunner",
        runner_name="frozen_model_policy",
        runner_status="pre_mainline",
        runner_family="actor",
    ),
}


PUBLIC_RUNNERS = list(_RUNNER_SPECS)


__all__ = [
    "BaseRunner",
    "PUBLIC_RUNNERS",
    *PUBLIC_RUNNERS,
]


_CLASS_CACHE: dict[str, type[Any]] = {}
_IMPLEMENTATION_CACHE: dict[str, type[Any]] = {}


def __getattr__(name: str) -> object:
    if name in _RUNNER_SPECS:
        return _load_public_runner(name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted([*globals(), *PUBLIC_RUNNERS])


def _load_public_runner(name: str) -> type[Any]:
    cached = _CLASS_CACHE.get(name)
    if cached is not None:
        return cached

    spec = _RUNNER_SPECS[name]
    attrs = _public_runner_attrs(spec)

    attrs.update(
        {
            "__doc__": (
                f"Lazy public runner proxy for {spec.module}.{spec.implementation}."
            ),
            "__init__": _make_proxy_init(name),
            "__getattr__": _proxy_getattr,
            "setup": _proxy_setup,
            "execute": _proxy_execute,
            "teardown": _proxy_teardown,
            "run": _proxy_run,
        }
    )
    public_cls = type(name, (BaseRunner,), attrs)
    _CLASS_CACHE[name] = public_cls
    globals()[name] = public_cls
    return public_cls


def _make_proxy_init(name: str):
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        config = args[0] if args else kwargs.get("config")
        output_dir = kwargs.get("output_dir")
        if config is not None:
            BaseRunner.__init__(self, config, output_dir=output_dir)
        implementation = _load_implementation(name)
        self._runner = implementation(*args, **kwargs)

    return __init__


def _load_implementation(name: str) -> type[Any]:
    cached = _IMPLEMENTATION_CACHE.get(name)
    if cached is not None:
        return cached

    spec = _RUNNER_SPECS[name]
    module = importlib.import_module(spec.module)
    implementation = getattr(module, spec.implementation)
    public_impl = type(name, (implementation,), _public_runner_attrs(spec))
    _IMPLEMENTATION_CACHE[name] = public_impl
    return public_impl


def _public_runner_attrs(spec: _RunnerSpec) -> dict[str, Any]:
    attrs: dict[str, Any] = {"__module__": __name__}
    if spec.runner_name is not None:
        attrs["runner_name"] = spec.runner_name
    if spec.runner_status is not None:
        attrs["runner_status"] = spec.runner_status
    if spec.runner_family is not None:
        attrs["runner_family"] = spec.runner_family
    return attrs


def _proxy_getattr(self: Any, name: str) -> Any:
    if name == "_runner":
        raise AttributeError(name)
    return getattr(self._runner, name)


def _proxy_setup(self: Any) -> None:
    return self._runner.setup()


def _proxy_execute(self: Any) -> object:
    return self._runner.execute()


def _proxy_teardown(self: Any) -> None:
    return self._runner.teardown()


def _proxy_run(self: Any) -> object:
    return self._runner.run()
