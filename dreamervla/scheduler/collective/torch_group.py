"""Thin torch.distributed collective group wrapper."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class TorchCollectiveGroup:
    """NCCL/Gloo-capable collective helpers.

    Calls are safe in single-process tests: if ``torch.distributed`` is not
    initialized, methods return cloned local values.
    """

    def __init__(self, process_group: Any | None = None) -> None:
        self.process_group = process_group
        self._pending_sends: list[tuple[Any, torch.Tensor]] = []

    @property
    def is_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def broadcast_tensor(self, tensor: torch.Tensor, *, src: int = 0) -> torch.Tensor:
        out = tensor.detach().clone()
        if self.is_initialized:
            dist.broadcast(out, src=int(src), group=self.process_group)
        return out

    def send(self, tensor: torch.Tensor, *, dst: int, channel: int = 0) -> None:
        """Post a point-to-point send to ``dst``.

        ``channel`` maps to a message tag. Sends are posted with ``isend`` and
        completed by :meth:`flush_sends`, which lets callers post several
        channel-tagged messages before the peer receives them in tag order.
        Unlike broadcast there is no single-process fallback: point-to-point
        communication requires a peer rank, so an uninitialized group is a hard
        error.
        """

        if not self.is_initialized:
            raise RuntimeError(
                "TorchCollectiveGroup.send requires an initialized process group "
                "(point-to-point needs a peer rank)"
            )
        payload = tensor if tensor.is_contiguous() else tensor.contiguous()
        work = dist.isend(
            payload,
            dst=int(dst),
            group=self.process_group,
            tag=int(channel),
        )
        self._pending_sends.append((work, payload))

    def flush_sends(self) -> None:
        """Wait for all previously posted point-to-point sends."""

        pending = self._pending_sends
        self._pending_sends = []
        for work, _payload in pending:
            work.wait()

    def recv(self, tensor: torch.Tensor, *, src: int, channel: int = 0) -> torch.Tensor:
        """Point-to-point receive from ``src`` into ``tensor`` (filled in place).

        ``channel`` is matched as a message tag, so receives can be issued in a
        different order than sends as long as the tags line up.
        """

        if not self.is_initialized:
            raise RuntimeError(
                "TorchCollectiveGroup.recv requires an initialized process group "
                "(point-to-point needs a peer rank)"
            )
        dist.recv(tensor, src=int(src), group=self.process_group, tag=int(channel))
        return tensor

    def broadcast_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        src: int = 0,
    ) -> dict[str, Any]:
        """Broadcast tensor values in a state dict.

        Non-tensor values are copied through unchanged. Tensor shapes must
        already match on all ranks, matching normal model state synchronization.
        """

        out: dict[str, Any] = {}
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                out[name] = self.broadcast_tensor(value, src=src)
            else:
                out[name] = value
        return out
