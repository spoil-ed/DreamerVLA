"""Named Ray FIFO channel used by worker actors."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

import ray


@ray.remote
class _ChannelActor:
    """Single-actor FIFO queue."""

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = max(0, int(maxsize))
        self.queues: dict[str, asyncio.Queue[Any]] = {
            "default": asyncio.Queue(maxsize=self.maxsize)
        }

    async def put(self, item: Any, key: str = "default") -> None:
        await self._queue(key).put(item)

    async def get(self, key: str = "default", timeout_s: float | None = None) -> Any:
        queue = self._queue(key)
        if timeout_s is None:
            return await queue.get()
        try:
            return await asyncio.wait_for(queue.get(), timeout=float(timeout_s))
        except TimeoutError as exc:
            raise TimeoutError(
                f"timed out waiting for channel key {key!r} "
                f"after {float(timeout_s):.3f}s"
            ) from exc

    async def get_batch(self, n: int, key: str = "default") -> list[Any]:
        return [await self._queue(key).get() for _ in range(int(n))]

    async def get_weighted_batch(self, weights: dict[str, int]) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = OrderedDict()
        for key, count in weights.items():
            out[str(key)] = [await self._queue(str(key)).get() for _ in range(int(count))]
        return out

    def qsize(self, key: str | None = None) -> int:
        if key is not None:
            return int(self._queue(str(key)).qsize())
        return int(sum(queue.qsize() for queue in self.queues.values()))

    def empty(self, key: str | None = None) -> bool:
        if key is not None:
            return bool(self._queue(str(key)).empty())
        return all(queue.empty() for queue in self.queues.values())

    def _queue(self, key: str) -> asyncio.Queue[Any]:
        normalized = str(key or "default")
        queue = self.queues.get(normalized)
        if queue is None:
            queue = asyncio.Queue(maxsize=self.maxsize)
            self.queues[normalized] = queue
        return queue


class Channel:
    """Synchronous wrapper around a named Ray FIFO actor."""

    def __init__(self, actor: Any) -> None:
        self._actor = actor

    @classmethod
    def create(cls, name: str, maxsize: int = 0) -> Channel:
        actor = (
            _ChannelActor.options(
                name=name,
                namespace="DreamerVLA",
                lifetime="detached",
                max_concurrency=100,
            )
            .remote(int(maxsize))
        )
        ray.get(actor.qsize.remote())
        return cls(actor)

    @classmethod
    def connect(cls, name: str) -> Channel:
        return cls(ray.get_actor(name, namespace="DreamerVLA"))

    def put(self, item: Any, *, key: str = "default") -> None:
        ray.get(self._actor.put.remote(item, str(key)))

    def get(self, *, key: str = "default", timeout_s: float | None = None) -> Any:
        return ray.get(self._actor.get.remote(str(key), timeout_s))

    def get_batch(self, n: int, *, key: str = "default") -> list[Any]:
        return list(ray.get(self._actor.get_batch.remote(int(n), str(key))))

    def get_weighted_batch(self, weights: dict[str, int]) -> dict[str, list[Any]]:
        normalized = {str(key): int(value) for key, value in weights.items()}
        return dict(ray.get(self._actor.get_weighted_batch.remote(normalized)))

    def qsize(self, *, key: str | None = None) -> int:
        return int(ray.get(self._actor.qsize.remote(key)))

    def empty(self, *, key: str | None = None) -> bool:
        return bool(ray.get(self._actor.empty.remote(key)))

    def put_no_wait(self, item: Any, *, key: str = "default") -> AsyncWork:
        return AsyncWork(self._actor.put.remote(item, str(key)))

    def get_no_wait(
        self,
        *,
        key: str = "default",
        timeout_s: float | None = None,
    ) -> AsyncWork:
        return AsyncWork(self._actor.get.remote(str(key), timeout_s))

    def get_batch_no_wait(self, n: int, *, key: str = "default") -> AsyncWork:
        return AsyncWork(self._actor.get_batch.remote(int(n), str(key)))

    def get_weighted_batch_no_wait(self, weights: dict[str, int]) -> AsyncWork:
        normalized = {str(key): int(value) for key, value in weights.items()}
        return AsyncWork(self._actor.get_weighted_batch.remote(normalized))


class AsyncWork:
    """Minimal Ray ObjectRef wait/done handle for Channel async calls."""

    def __init__(self, ref: Any) -> None:
        self.ref = ref

    def wait(self) -> Any:
        return ray.get(self.ref)

    def done(self) -> bool:
        ready, _ = ray.wait([self.ref], num_returns=1, timeout=0)
        return bool(ready)
