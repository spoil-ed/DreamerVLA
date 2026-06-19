"""Ray-free unit tests for collective point-to-point boundary behavior."""

from __future__ import annotations

import pytest
import torch

from dreamervla.scheduler.collective.torch_group import TorchCollectiveGroup


def test_send_requires_initialized_process_group() -> None:
    group = TorchCollectiveGroup()
    assert not group.is_initialized
    with pytest.raises(RuntimeError, match="send"):
        group.send(torch.zeros(3), dst=1)


def test_recv_requires_initialized_process_group() -> None:
    group = TorchCollectiveGroup()
    with pytest.raises(RuntimeError, match="recv"):
        group.recv(torch.zeros(3), src=0)
