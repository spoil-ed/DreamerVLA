"""Real 2-rank gloo verification of collective send/recv + multi-channel tags.

Single-node verifiable: spins a 2-process gloo group on loopback (no GPU/NCCL).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dreamervla.scheduler.cluster import Cluster

_CH_A = 7
_CH_B = 3
_MSG_A = [1.0, 2.0, 3.0]
_MSG_B = [9.0, 8.0, 7.0]


def _send_recv_worker(rank: int, world_size: int, port: int) -> None:
    from dreamervla.scheduler.collective.torch_group import TorchCollectiveGroup

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        group = TorchCollectiveGroup()
        if rank == 0:
            # Send channel A first, then channel B.
            group.send(torch.tensor(_MSG_A), dst=1, channel=_CH_A)
            group.send(torch.tensor(_MSG_B), dst=1, channel=_CH_B)
            group.flush_sends()
        else:
            # Receive out of order by tag: B before A. Tag routing must hold.
            buf_b = torch.zeros(3)
            group.recv(buf_b, src=0, channel=_CH_B)
            buf_a = torch.zeros(3)
            group.recv(buf_a, src=0, channel=_CH_A)
            assert torch.allclose(buf_a, torch.tensor(_MSG_A)), buf_a
            assert torch.allclose(buf_b, torch.tensor(_MSG_B)), buf_b
    finally:
        dist.destroy_process_group()


def test_send_recv_round_trip_over_gloo_with_channels() -> None:
    port = Cluster.find_free_port()
    mp.spawn(_send_recv_worker, args=(2, port), nprocs=2, join=True)
