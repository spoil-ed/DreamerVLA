"""Single-process Ray cluster bootstrap for the optional online backend."""

from __future__ import annotations

import os
import socket
from typing import Any

import ray
from packaging.version import Version

from dreamervla.scheduler.node import NodeInfo, discover_ray_nodes


class Cluster:
    """Idempotent local Ray bootstrap used by the Ray online cotrain backend."""

    min_ray_version = Version("2.47.0")

    def __init__(self, cfg: Any | None = None) -> None:
        version = Version(ray.__version__)
        if version < self.min_ray_version:
            raise RuntimeError(
                f"DreamerVLA Ray backend requires ray>={self.min_ray_version}, got {version}"
            )

        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        if not ray.is_initialized():
            init_kwargs = self._init_kwargs(cfg)
            ray.init(**init_kwargs)

    @classmethod
    def has_initialized(cls) -> bool:
        """Return whether Ray is initialized in this Python process."""

        return bool(ray.is_initialized())

    @classmethod
    def find_free_port(cls) -> int:
        """Find an unused loopback TCP port."""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @property
    def num_gpus(self) -> int:
        """Number of GPUs Ray reports for the local cluster."""

        return int(ray.cluster_resources().get("GPU", 0))

    @property
    def num_nodes(self) -> int:
        """Number of live Ray nodes visible to the current process."""

        return sum(1 for node in self.nodes() if node.alive)

    def nodes(self) -> list[NodeInfo]:
        """Return Ray node metadata."""

        return discover_ray_nodes()

    def require_single_node(self) -> None:
        """Fail early if the connected Ray cluster spans more than one node."""

        num_nodes = self.num_nodes
        if num_nodes != 1:
            raise RuntimeError(
                "DreamerVLA Ray backend is single-node only; "
                f"connected Ray cluster has {num_nodes} live node(s)"
            )

    def shutdown(self) -> None:
        """Shut down Ray for this Python process."""

        if ray.is_initialized():
            ray.shutdown()

    def _init_kwargs(self, cfg: Any | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "namespace": "DreamerVLA",
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "_node_ip_address": "127.0.0.1",
        }
        if cfg is None:
            return kwargs

        for key in (
            "address",
            "namespace",
            "num_cpus",
            "num_gpus",
            "object_store_memory",
            "local_mode",
            "log_to_driver",
            "include_dashboard",
            "_temp_dir",
            "_node_ip_address",
        ):
            value = self._cfg_get(cfg, key)
            if value is not None:
                kwargs[key] = value
        return kwargs

    @staticmethod
    def _cfg_get(cfg: Any, key: str) -> Any | None:
        if isinstance(cfg, dict):
            return cfg.get(key)
        return getattr(cfg, key, None)
