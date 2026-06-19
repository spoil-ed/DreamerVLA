"""Local Ray node discovery helpers."""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

import ray


@dataclass(frozen=True)
class NodeInfo:
    """Small node metadata record used for placement validation."""

    node_id: str
    address: str
    resources: dict[str, float] = field(default_factory=dict)
    alive: bool = True


def probe_local_node() -> NodeInfo:
    """Return local node metadata without requiring a running Ray cluster."""

    return NodeInfo(
        node_id=socket.gethostname(),
        address=_local_ip(),
        resources={},
        alive=True,
    )


def discover_ray_nodes() -> list[NodeInfo]:
    """Return alive Ray nodes when Ray is initialized, else the local node."""

    if not ray.is_initialized():
        return [probe_local_node()]
    nodes: list[NodeInfo] = []
    for item in ray.nodes():
        resources = {
            str(key): float(value)
            for key, value in dict(item.get("Resources", {})).items()
            if isinstance(value, int | float)
        }
        nodes.append(
            NodeInfo(
                node_id=str(item.get("NodeID", "")),
                address=str(item.get("NodeManagerAddress", "")),
                resources=resources,
                alive=bool(item.get("Alive", False)),
            )
        )
    return nodes or [probe_local_node()]


def _local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


__all__ = ["NodeInfo", "discover_ray_nodes", "probe_local_node"]
