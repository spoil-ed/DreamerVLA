from __future__ import annotations

import uuid

import ray
import torch

from dreamervla.scheduler.cluster import Cluster


def test_channel_create_connect_and_batch_fifo() -> None:
    try:
        from dreamervla.scheduler.channel import Channel
    except ModuleNotFoundError as exc:
        raise AssertionError("Channel module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        name = f"test-channel-{uuid.uuid4().hex}"
        writer = Channel.create(name)
        reader = Channel.connect(name)

        writer.put({"idx": 0})
        writer.put({"idx": 1})
        writer.put({"idx": 2})

        assert reader.qsize() == 3
        assert reader.empty() is False
        assert reader.get_batch(3) == [{"idx": 0}, {"idx": 1}, {"idx": 2}]
        assert reader.empty() is True
    finally:
        cluster.shutdown()


def test_channel_round_trips_tensors() -> None:
    try:
        from dreamervla.scheduler.channel import Channel
    except ModuleNotFoundError as exc:
        raise AssertionError("Channel module should exist") from exc

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        name = f"test-channel-{uuid.uuid4().hex}"
        channel = Channel.create(name)
        expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        channel.put(expected)
        actual = channel.get()

        assert torch.equal(actual, expected)
    finally:
        cluster.shutdown()


def test_channel_routes_by_key_and_weighted_batch() -> None:
    from dreamervla.scheduler.channel import Channel

    if ray.is_initialized():
        ray.shutdown()
    cluster = Cluster()

    try:
        name = f"test-channel-keyed-{uuid.uuid4().hex}"
        channel = Channel.create(name)

        channel.put("a0", key="actor")
        channel.put("r0", key="replay")
        channel.put("a1", key="actor")
        channel.put("r1", key="replay")

        assert channel.qsize() == 4
        assert channel.qsize(key="actor") == 2
        assert channel.get(key="actor") == "a0"
        assert channel.get_weighted_batch({"replay": 2, "actor": 1}) == {
            "replay": ["r0", "r1"],
            "actor": ["a1"],
        }
        assert channel.empty()
    finally:
        cluster.shutdown()
