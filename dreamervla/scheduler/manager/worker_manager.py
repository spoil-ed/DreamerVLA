"""Ray-free manager primitives used by scheduler tests and optional backend."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import ray


@dataclass(frozen=True)
class WorkerRoute:
    """Route metadata for a named worker actor."""

    name: str
    rank: int
    node_rank: int = 0
    actor: Any | None = None


class WorkerManager:
    """In-process route table with Ray-aware send/recv helpers."""

    def __init__(self) -> None:
        self._routes: dict[str, WorkerRoute] = {}

    def register(
        self,
        name: str,
        *,
        rank: int,
        node_rank: int = 0,
        actor: Any | None = None,
    ) -> WorkerRoute:
        route = WorkerRoute(
            name=str(name),
            rank=int(rank),
            node_rank=int(node_rank),
            actor=actor,
        )
        self._routes[route.name] = route
        return route

    def resolve(self, name: str) -> WorkerRoute:
        try:
            return self._routes[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown worker route {name!r}") from exc

    def send(self, name: str, method: str, *args: Any, **kwargs: Any) -> Any:
        route = self.resolve(name)
        if route.actor is None:
            raise RuntimeError(f"worker route {name!r} has no actor bound")
        target = getattr(route.actor, str(method))
        remote = getattr(target, "remote", None)
        if remote is not None:
            return remote(*args, **kwargs)
        return target(*args, **kwargs)

    @staticmethod
    def recv(result: Any) -> Any:
        if isinstance(result, ray.ObjectRef):
            return ray.get(result)
        return result


class DeviceLockManager:
    """Simple named lock table for device/port critical sections."""

    def __init__(self) -> None:
        self._locks: dict[str, Lock] = {}

    def acquire(self, name: str, *, blocking: bool = True) -> bool:
        lock = self._locks.setdefault(str(name), Lock())
        return bool(lock.acquire(blocking=bool(blocking)))

    def release(self, name: str) -> None:
        self._locks[str(name)].release()
