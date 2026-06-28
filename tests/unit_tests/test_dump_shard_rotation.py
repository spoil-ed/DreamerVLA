from unittest import mock

from dreamervla.workers.rollout import dump_worker as dw


class _FakeWriter:
    created: list[str] = []
    instances: list = []
    def __init__(self, reward_dir, hidden_dir, shard_name):
        self.shard_name = str(shard_name)
        self.demos: list[int] = []
        self.kwargs: list[dict] = []
        self.closed = False
        _FakeWriter.created.append(self.shard_name)
        _FakeWriter.instances.append(self)
    def write_demo(self, index, steps, preprocess_config=None, data_attrs=None, **kw):
        self.demos.append(int(index))
        self.kwargs.append(dict(kw))
    def close(self):
        self.closed = True


class _FileWriter:
    def __init__(self, reward_dir, hidden_dir, shard_name):
        self.shard_name = str(shard_name)
        self.reward_path = reward_dir / self.shard_name
        self.hidden_path = hidden_dir / self.shard_name
        self.reward_path.parent.mkdir(parents=True, exist_ok=True)
        self.hidden_path.parent.mkdir(parents=True, exist_ok=True)
        self.reward_path.write_bytes(b"reward")
        self.hidden_path.write_bytes(b"hidden")
        self.closed = False

    def write_demo(self, index, steps, preprocess_config=None, data_attrs=None, **kw):
        del index, steps, preprocess_config, data_attrs, kw

    def close(self):
        self.closed = True

def _episode():
    return [
        {
            "task_id": 0,
            "episode_id": 0,
            "init_state_index": 0,
            "task_description": "t",
            "success": True,
        }
    ]


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


def test_closes_full_shard_immediately():
    _FakeWriter.created = []
    _FakeWriter.instances = []
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        w = dw.RolloutDumpWorker("r", "h", demos_per_shard=1)
        w.init()
        w.add_episode(_episode())
        assert _FakeWriter.created == ["ray_shard_000.hdf5"]
        assert _FakeWriter.instances[0].closed is True

        w.add_episode(_episode())
        assert _FakeWriter.created == ["ray_shard_000.hdf5", "ray_shard_001.hdf5"]
        assert [writer.demos for writer in _FakeWriter.instances] == [[0], [0]]
        assert _FakeWriter.instances[1].closed is True


def test_one_episode_shard_name_includes_global_step_and_success(tmp_path):
    episode = _episode()
    episode[-1]["success"] = True
    episode[-1]["episode_metadata"] = {"global_step": 10, "env_step": 55}

    with mock.patch.object(dw, "RolloutDumpWriter", _FileWriter):
        w = dw.RolloutDumpWorker(
            str(tmp_path / "reward"),
            str(tmp_path / "hidden"),
            "ray_shard_000.hdf5",
            demos_per_shard=1,
        )
        w.init()
        w.add_episode(episode)

    assert (tmp_path / "reward" / "ray_shard_gs000010_t00_ep000000_success.hdf5").is_file()
    assert (tmp_path / "hidden" / "ray_shard_gs000010_t00_ep000000_success.hdf5").is_file()
    assert not (tmp_path / "reward" / "ray_shard_000.hdf5").exists()
    assert not (tmp_path / "hidden" / "ray_shard_000.hdf5").exists()


def test_one_episode_shard_records_manifest_and_prunes_recent_global_steps(tmp_path):
    with mock.patch.object(dw, "RolloutDumpWriter", _FileWriter):
        w = dw.RolloutDumpWorker(
            str(tmp_path / "reward"),
            str(tmp_path / "hidden"),
            "ray_shard_000.hdf5",
            demos_per_shard=1,
            manifest_root=str(tmp_path),
            keep_last_global_steps=2,
        )
        w.init()
        for episode_id, global_step in ((1, 10), (2, 11), (3, 12)):
            episode = _episode()
            episode[0]["task_id"] = 7
            episode[0]["episode_id"] = episode_id
            episode[0]["init_state_index"] = episode_id
            episode[-1]["success"] = episode_id != 2
            episode[-1]["episode_metadata"] = {
                "global_step": global_step,
                "env_step": 100 + global_step,
                "update_step": global_step,
            }
            w.add_episode(episode)

    entries = dw.read_online_rollout_manifest(tmp_path)
    assert [int(item["global_step"]) for item in entries] == [11, 12]
    assert (
        tmp_path
        / "episodes"
        / "task_07"
        / "global_step000012_success_True"
        / "ep_000003.h5"
    ).is_file()
    assert not (
        tmp_path
        / "episodes"
        / "task_07"
        / "global_step000010_success_True"
        / "ep_000001.h5"
    ).exists()


def test_start_shard_index_appends_on_resume():
    _FakeWriter.created = []
    _FakeWriter.instances = []
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        # A relaunch that already has shards 000..002 starts rotation at 003.
        w = dw.RolloutDumpWorker("r", "h", demos_per_shard=2, start_shard_index=3)
        w.init()
        for _ in range(3):
            w.add_episode(_episode())
        assert _FakeWriter.created == ["ray_shard_003.hdf5", "ray_shard_004.hdf5"]


def test_start_shard_index_names_single_shard_on_resume():
    _FakeWriter.created = []
    _FakeWriter.instances = []
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        # No rotation (demos_per_shard=0) still appends at the resume-aware name.
        w = dw.RolloutDumpWorker("r", "h", "ray_shard_002.hdf5", demos_per_shard=0,
                                 start_shard_index=2)
        w.init()
        w.add_episode(_episode())
        assert _FakeWriter.created == ["ray_shard_002.hdf5"]


def test_episode_metadata_passes_to_writer_from_terminal_step():
    _FakeWriter.created = []
    _FakeWriter.instances = []
    episode = _episode()
    episode[-1]["episode_metadata"] = {"chunk_size": 8, "policy_name": "oft"}
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        w = dw.RolloutDumpWorker("r", "h", demos_per_shard=0)
        w.init()
        w.add_episode(episode)
        assert _FakeWriter.instances[0].kwargs[0]["episode_metadata"] == {
            "chunk_size": 8,
            "policy_name": "oft",
        }


def test_init_state_index_passes_to_writer_from_episode():
    _FakeWriter.created = []
    _FakeWriter.instances = []
    episode = _episode()
    episode[0]["init_state_index"] = 17
    with mock.patch.object(dw, "RolloutDumpWriter", _FakeWriter):
        w = dw.RolloutDumpWorker("r", "h", demos_per_shard=0)
        w.init()
        w.add_episode(episode)
        assert _FakeWriter.instances[0].kwargs[0]["init_state_index"] == 17
