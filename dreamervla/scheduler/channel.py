"""Named Ray FIFO channel used by worker actors."""

from __future__ import annotations

import asyncio
from typing import Any

import ray


@ray.remote
class _ChannelActor:
    """Single-actor FIFO queue."""

    def __init__(self, maxsize: int = 0) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max(0, int(maxsize)))

    async def put(self, item: Any) -> None:
        await self.queue.put(item)

    async def get(self) -> Any:
        return await self.queue.get()

    async def get_batch(self, n: int) -> list[Any]:
        return [await self.queue.get() for _ in range(int(n))]

    def qsize(self) -> int:
        return int(self.queue.qsize())

    def empty(self) -> bool:
        return bool(self.queue.empty())


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

    def put(self, item: Any) -> None:
        ray.get(self._actor.put.remote(item))

    def get(self) -> Any:
        return ray.get(self._actor.get.remote())

    def get_batch(self, n: int) -> list[Any]:
        return list(ray.get(self._actor.get_batch.remote(int(n))))

    def qsize(self) -> int:
        return int(ray.get(self._actor.qsize.remote()))

    def empty(self) -> bool:
        return bool(ray.get(self._actor.empty.remote()))
