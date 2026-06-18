"""Importable Ray test workers for DreamerVLA scheduler e2e tests."""

from __future__ import annotations

from dreamervla.scheduler.worker import Worker


class EchoWorker(Worker):
    """Tiny importable worker used by scheduler Ray smoke tests."""

    def __init__(self, base: int) -> None:
        super().__init__()
        self.base = int(base)
        self.initialized = False

    def init(self) -> None:
        self.initialized = True

    def rank_info(self) -> dict[str, int | str | bool]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device": self.device,
            "initialized": self.initialized,
        }

    def add(self, value: int) -> int:
        return self.base + int(value) + self.rank


class ChannelWorker(Worker):
    """Worker that sends and receives through a named scheduler Channel."""

    def __init__(self, channel_name: str) -> None:
        super().__init__()
        from dreamervla.scheduler.channel import Channel

        self.channel = Channel.connect(channel_name)

    def put_many(self, items: list[object]) -> int:
        for item in items:
            self.channel.put(item)
        return len(items)

    def get_many(self, n: int) -> list[object]:
        return self.channel.get_batch(int(n))
