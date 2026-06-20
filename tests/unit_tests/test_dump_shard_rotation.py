from unittest import mock

from dreamervla.workers.rollout import dump_worker as dw


class _FakeWriter:
    created: list[str] = []
    instances: list = []
    def __init__(self, reward_dir, hidden_dir, shard_name):
        self.shard_name = str(shard_name)
        self.demos: list[int] = []
        _FakeWriter.created.append(self.shard_name)
        _FakeWriter.instances.append(self)
    def write_demo(self, index, steps, preprocess_config=None, data_attrs=None, **kw):
        self.demos.append(int(index))
    def close(self):
        pass


def _episode():
    return [{"task_id": 0, "episode_id": 0, "task_description": "t", "success": True}]


def test_no_rotation_when_disabled():
    _FakeWriter.created = []
    _FakeWriter.instances = []
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        w = dw.RolloutDumpWorker("r", "h", demos_per_shard=0)
        w.init()
        for _ in range(5):
            w.add_episode(_episode())
        assert _FakeWriter.created == ["ray_shard_000.hdf5"]   # single shard
        assert w.size() == 5
        assert _FakeWriter.instances[0].demos == [0, 1, 2, 3, 4]


def test_rotates_every_n_demos():
    _FakeWriter.created = []
    _FakeWriter.instances = []
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        w = dw.RolloutDumpWorker("r", "h", demos_per_shard=2)
        w.init()
        for _ in range(5):
            w.add_episode(_episode())
        # 5 demos / 2 per shard -> shards 000(2), 001(2), 002(1)
        assert _FakeWriter.created == [
            "ray_shard_000.hdf5", "ray_shard_001.hdf5", "ray_shard_002.hdf5"
        ]
        assert w.size() == 5
        assert [w.demos for w in _FakeWriter.instances] == [[0, 1], [0, 1], [0]]
