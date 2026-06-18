"""Ray WorkerGroup launcher and broadcast facade."""

from __future__ import annotations

from typing import Any

import ray

from dreamervla.scheduler.placement import Placement, PlacementStrategy


class WorkerGroup:
    """Group of homogeneous worker actors.

    Launch and RPC behavior is filled in by the S1 WorkerGroup TDD step. The
    constructor is intentionally useful on its own so ``Worker.create_group``
    can produce an inspectable unlaunched group.
    """

    def __init__(self, worker_cls: type, *args: Any, **kwargs: Any) -> None:
        self.worker_cls = worker_cls
        self.args = args
        self.kwargs = kwargs
        self.workers: list[Any] = []
        self.placements: list[Placement] = []
        self._next_ranks: tuple[int, ...] | None = None

    def launch(
        self,
        cluster: Any,
        placement: PlacementStrategy,
        name: str | None = None,
    ) -> WorkerGroup:
        """Launch one Ray actor per placement rank and initialize them."""

        self.placements = placement.get_placement(cluster)
        self.workers = []
        for item in self.placements:
            remote_cls = ray.remote(self.worker_cls)
            options: dict[str, Any] = {
                "num_gpus": len(item.visible_accelerators),
                "runtime_env": {"env_vars": self._env_vars(item)},
            }
            if name is not None:
                options["name"] = f"{name}_{item.rank}"
            actor = remote_cls.options(**options).remote(*self.args, **self.kwargs)
            self.workers.append(actor)

        if self.workers:
            ray.get([worker.init.remote() for worker in self.workers])
        return self

    def execute_on(self, *ranks: int) -> WorkerGroup:
        """Restrict the next method call to the selected ranks."""

        available = set(range(len(self.workers)))
        requested = tuple(int(rank) for rank in ranks)
        missing = [rank for rank in requested if rank not in available]
        if missing:
            raise ValueError(f"unknown worker rank(s): {missing}")
        self._next_ranks = requested
        return self

    def __getattr__(self, method: str) -> WorkerGroupFunc:
        if method.startswith("_"):
            raise AttributeError(method)
        return WorkerGroupFunc(self, method)

    def _consume_selected_workers(self) -> list[Any]:
        if self._next_ranks is None:
            return list(self.workers)
        ranks = self._next_ranks
        self._next_ranks = None
        return [self.workers[rank] for rank in ranks]

    @staticmethod
    def _env_vars(placement: Placement) -> dict[str, str]:
        return {
            "RANK": str(placement.rank),
            "LOCAL_RANK": str(placement.local_rank),
            "WORLD_SIZE": str(placement.local_world_size),
            "CUDA_VISIBLE_DEVICES": ",".join(placement.visible_accelerators),
        }


class WorkerGroupFunc:
    """Deferred broadcast of one method over a WorkerGroup."""

    def __init__(self, group: WorkerGroup, method: str) -> None:
        self.group = group
        self.method = method

    def __call__(self, *args: Any, **kwargs: Any) -> WorkerGroupFuncResult:
        refs = [
            getattr(worker, self.method).remote(*args, **kwargs)
            for worker in self.group._consume_selected_workers()
        ]
        return WorkerGroupFuncResult(refs)


class WorkerGroupFuncResult:
    """ObjectRefs returned by a WorkerGroup method broadcast."""

    def __init__(self, refs: list[Any]) -> None:
        self.refs = refs

    def wait(self) -> list[Any]:
        return list(ray.get(self.refs))

    def done(self) -> bool:
        if not self.refs:
            return True
        ready, _ = ray.wait(self.refs, num_returns=len(self.refs), timeout=0)
        return len(ready) == len(self.refs)
