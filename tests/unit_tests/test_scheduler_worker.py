from __future__ import annotations


def test_worker_reads_ray_rank_environment(monkeypatch) -> None:
    try:
        from dreamervla.scheduler.worker import Worker
    except ModuleNotFoundError as exc:
        raise AssertionError("Worker module should exist") from exc

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    worker = Worker()

    assert worker.rank == 2
    assert worker.local_rank == 1
    assert worker.world_size == 4
    assert worker.local_world_size == 4
    assert worker.visible_accelerators == ["3"]
    assert worker.device == "cuda:0"


def test_worker_defaults_to_single_cpu_rank(monkeypatch) -> None:
    try:
        from dreamervla.scheduler.worker import Worker
    except ModuleNotFoundError as exc:
        raise AssertionError("Worker module should exist") from exc

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    worker = Worker()

    assert worker.rank == 0
    assert worker.local_rank == 0
    assert worker.world_size == 1
    assert worker.local_world_size == 1
    assert worker.visible_accelerators == []
    assert worker.device == "cpu"
    assert worker.init() is None


def test_worker_create_group_returns_unlaunched_group() -> None:
    try:
        from dreamervla.scheduler.worker import Worker
    except ModuleNotFoundError as exc:
        raise AssertionError("Worker module should exist") from exc

    group = Worker.create_group("arg", named=True)

    assert group.worker_cls is Worker
    assert group.args == ("arg",)
    assert group.kwargs == {"named": True}


def test_worker_group_maps_local_gpu_index_through_parent_visible_devices(monkeypatch) -> None:
    from dreamervla.scheduler.placement import Placement
    from dreamervla.scheduler.worker_group import WorkerGroup

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")

    gpu_env = WorkerGroup._env_vars(
        Placement(
            rank=0,
            local_rank=0,
            local_world_size=1,
            visible_accelerators=["0"],
            device="cuda:0",
        )
    )
    cpu_env = WorkerGroup._env_vars(
        Placement(
            rank=0,
            local_rank=0,
            local_world_size=1,
            visible_accelerators=[],
            device="cpu",
        )
    )

    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == ""


def test_worker_group_env_vars_include_single_node_rendezvous() -> None:
    from dreamervla.scheduler.placement import Placement
    from dreamervla.scheduler.worker_group import WorkerGroup

    env = WorkerGroup._env_vars(
        Placement(
            rank=1,
            local_rank=1,
            local_world_size=2,
            visible_accelerators=["1"],
            device="cuda:1",
        ),
        master_addr="127.0.0.1",
        master_port=29519,
    )

    assert env["RANK"] == "1"
    assert env["LOCAL_RANK"] == "1"
    assert env["WORLD_SIZE"] == "2"
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"] == "29519"


def test_worker_group_env_vars_suppress_noisy_library_startup_logs() -> None:
    from dreamervla.scheduler.placement import Placement
    from dreamervla.scheduler.worker_group import WorkerGroup

    env = WorkerGroup._env_vars(
        Placement(
            rank=0,
            local_rank=0,
            local_world_size=1,
            visible_accelerators=["0"],
            device="cuda:0",
        )
    )

    assert env["TF_CPP_MIN_LOG_LEVEL"] == "3"
    assert env["ABSL_MIN_LOG_LEVEL"] == "3"
    assert env["GLOG_minloglevel"] == "2"
    assert env["GYM_DISABLE_WARNINGS"] == "1"
    assert env["USE_TF"] == "0"
    assert env["TF_ENABLE_ONEDNN_OPTS"] == "0"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["TRANSFORMERS_VERBOSITY"] == "error"
    assert "ignore::FutureWarning:libero.libero.benchmark" in env["PYTHONWARNINGS"]
    assert (
        "ignore:enable_nested_tensor is True.*:UserWarning:torch.nn.modules.transformer"
        in env["PYTHONWARNINGS"]
    )


def test_worker_group_launch_assigns_shared_single_node_rendezvous(monkeypatch) -> None:
    import dreamervla.scheduler.worker_group as worker_group_module
    from dreamervla.scheduler.placement import Placement
    from dreamervla.scheduler.worker_group import WorkerGroup

    captured: list[dict[str, object]] = []

    class _PlacementStrategy:
        def get_placement(self, cluster):
            del cluster
            return [
                Placement(
                    rank=0,
                    local_rank=0,
                    local_world_size=2,
                    visible_accelerators=["0"],
                    device="cuda:0",
                ),
                Placement(
                    rank=1,
                    local_rank=1,
                    local_world_size=2,
                    visible_accelerators=["1"],
                    device="cuda:1",
                ),
            ]

    class _Cluster:
        @staticmethod
        def find_free_port() -> int:
            return 29601

    class _RemoteMethod:
        @staticmethod
        def remote() -> str:
            return "initialized"

    class _Actor:
        init = _RemoteMethod()

    class _RemoteClass:
        @staticmethod
        def options(**options):
            captured.append(options)
            return _RemoteClass

        @staticmethod
        def remote(*args, **kwargs):
            del args, kwargs
            return _Actor()

    monkeypatch.setattr(worker_group_module.ray, "remote", lambda _cls: _RemoteClass)
    monkeypatch.setattr(worker_group_module.ray, "get", lambda refs: refs)

    WorkerGroup(object).launch(_Cluster(), _PlacementStrategy())

    envs = [item["runtime_env"]["env_vars"] for item in captured]
    assert [env["RANK"] for env in envs] == ["0", "1"]
    assert {env["MASTER_ADDR"] for env in envs} == {"127.0.0.1"}
    assert {env["MASTER_PORT"] for env in envs} == {"29601"}
    assert [item["enable_task_events"] for item in captured] == [False, False]


def test_worker_group_send_recv_routes_one_rank() -> None:
    from dreamervla.scheduler.worker_group import WorkerGroup

    class _RemoteMethod:
        def __init__(self, value: int) -> None:
            self.value = value

        def remote(self, delta: int) -> int:
            return self.value + int(delta)

    class _Actor:
        def __init__(self, value: int) -> None:
            self.add = _RemoteMethod(value)

    group = WorkerGroup(object)
    group.workers = [_Actor(10), _Actor(20)]

    result = group.send(1, "add", 3)

    assert group.recv(result) == 23
