"""Compatibility names for the canonical LIBERO online train environment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from dreamervla.envs.train_env import (
    ACTION_HIGH,
    ACTION_LOW,
    DreamerVLAOnlineTrainEnv,
    DreamerVLAOnlineTrainEnvConfig,
    normalize_libero_action,
    unnormalize_libero_action,
)


@dataclass(frozen=True)
class LIBEROOnlineEnvConfig(DreamerVLAOnlineTrainEnvConfig):
    """Backwards-compatible config name for the canonical online env."""

    task_suite_name: str = "libero_goal"
    history_length: int = 2
    action_input: Literal["raw", "normalized"] = "raw"


class LIBEROOnlineEnv(DreamerVLAOnlineTrainEnv):
    """Backwards-compatible env name backed by `DreamerVLAOnlineTrainEnv`."""

    cfg: LIBEROOnlineEnvConfig

    def __init__(
        self,
        task_suite_name: str = "libero_goal",
        task_id: int = 0,
        task_ids: Sequence[int] | None = None,
        resolution: int = 256,
        image_size: int = 64,
        history_length: int = 2,
        warmup_steps: int = 10,
        seed: int = 0,
        max_steps: int | None = None,
        action_input: Literal["raw", "normalized"] = "raw",
        clip_actions: bool = True,
        sparse_success_reward: bool = True,
        task_sampling: Literal["sequential", "random"] = "sequential",
        init_state_sampling: Literal["sequential", "random"] = "sequential",
        pixel_rotate_180: bool = False,
        vla_rotate_180: bool = True,
        **overrides: Any,
    ) -> None:
        reward_mode = "sparse_success" if bool(sparse_success_reward) else "raw"
        config = LIBEROOnlineEnvConfig(
            task_suite_name=str(task_suite_name),
            task_id=int(task_id),
            task_ids=None if task_ids is None else tuple(int(x) for x in task_ids),
            resolution=int(resolution),
            image_size=int(image_size),
            history_length=int(history_length),
            warmup_steps=int(warmup_steps),
            seed=int(seed),
            max_steps=None if max_steps is None else int(max_steps),
            action_input=action_input,
            clip_actions=bool(clip_actions),
            reward_mode=reward_mode,
            task_sampling=task_sampling,
            init_state_sampling=init_state_sampling,
            pixel_rotate_180=bool(pixel_rotate_180),
            vla_rotate_180=bool(vla_rotate_180),
        )
        super().__init__(config=config, **overrides)

    @classmethod
    def from_config(
        cls, config: LIBEROOnlineEnvConfig | dict[str, Any]
    ) -> LIBEROOnlineEnv:
        if isinstance(config, LIBEROOnlineEnvConfig):
            payload = config.__dict__.copy()
        else:
            payload = dict(config)
        if "reward_mode" in payload and "sparse_success_reward" not in payload:
            payload["sparse_success_reward"] = (
                payload.pop("reward_mode") == "sparse_success"
            )
        return cls(**payload)


def build_libero_online_envs(
    *,
    task_suite_name: str = "libero_goal",
    task_ids: Iterable[int] | None = None,
    num_envs: int | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> list[LIBEROOnlineEnv]:
    """Build one env per task id for simple synchronous online collection."""

    ids = [int(x) for x in (task_ids if task_ids is not None else [0])]
    if num_envs is not None:
        ids = (
            [ids[0] for _ in range(int(num_envs))]
            if len(ids) == 1
            else ids[: int(num_envs)]
        )
    return [
        LIBEROOnlineEnv(
            task_suite_name=task_suite_name,
            task_id=task_id,
            seed=int(seed) + idx,
            **kwargs,
        )
        for idx, task_id in enumerate(ids)
    ]


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "LIBEROOnlineEnv",
    "LIBEROOnlineEnvConfig",
    "build_libero_online_envs",
    "normalize_libero_action",
    "unnormalize_libero_action",
]
