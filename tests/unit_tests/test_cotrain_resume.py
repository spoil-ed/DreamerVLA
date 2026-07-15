"""Pins the base save/load contract that online_cotrain resume (R1) relies on:
optimizer state + scalar attributes round-trip, and exclude_keys keeps frozen
modules out of the payload.
"""

import json
import pathlib
import random
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.cotrain_runner import CotrainRunner


class _Mini(BaseRunner):
    include_keys = ("global_step", "epoch", "classifier_threshold")
    exclude_keys = ("frozen",)

    def __init__(self, cfg, tmp):
        self.cfg = cfg
        self._out = tmp
        self.global_step = 0
        self.epoch = 0
        self.classifier_threshold = 0.5
        self.policy = torch.nn.Linear(3, 2)
        self.frozen = torch.nn.Linear(3, 2)  # must NOT be checkpointed
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def get_checkpoint_path(self, tag="latest", *, prefer_existing=False):
        return pathlib.Path(self._out) / f"{tag}.ckpt"

    def run(self):  # abstract
        return None


class _FakeDistributed:
    def __init__(self, *, rank, is_main_process, gathered):
        self.rank = rank
        self.is_main_process = is_main_process
        self.requires_collective_checkpointing = False
        self.gathered = gathered
        self.calls = []

    def all_gather_objects(self, value):
        self.calls.append(value)
        return self.gathered


def _assert_nested_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_state_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_state_equal(actual_item, expected_item)
        return
    assert actual == expected


def test_base_checkpoint_roundtrips_optimizer_scalars_and_rng(tmp_path):
    cfg = OmegaConf.create({})
    a = _Mini(cfg, tmp_path)
    with torch.no_grad():
        a.policy.weight.copy_(torch.tensor([[0.25, -0.5, 0.75], [1.0, -1.25, 1.5]]))
        a.policy.bias.copy_(torch.tensor([-0.125, 0.375]))

    # Take an optimizer step so Adam's step and moment tensors are checkpointed.
    loss = a.policy(torch.ones(1, 3)).sum()
    loss.backward()
    a.policy_optimizer.step()
    expected_policy = {
        name: parameter.detach().clone() for name, parameter in a.policy.named_parameters()
    }
    expected_optimizer = a.policy_optimizer.state_dict()
    assert expected_optimizer["state"]
    for parameter_state in expected_optimizer["state"].values():
        assert {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state)
    a.global_step = 7
    a.epoch = 4
    a.classifier_threshold = 0.73
    path = a.save_checkpoint()

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(())

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "policy" in payload["state_dicts"]
    assert "policy_optimizer" in payload["state_dicts"]
    assert "frozen" not in payload["state_dicts"]  # exclude_keys honored
    assert len(payload["rng_by_rank"]) == 1

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    b = _Mini(cfg, tmp_path)
    with torch.no_grad():
        b.policy.weight.fill_(-9.0)
        b.policy.bias.fill_(9.0)
    assert any(
        not torch.equal(parameter, expected_policy[name])
        for name, parameter in b.policy.named_parameters()
    )

    b.load_checkpoint(path=path)
    assert b.global_step == 7
    assert b.epoch == 4
    assert abs(b.classifier_threshold - 0.73) < 1e-9
    for name, parameter in b.policy.named_parameters():
        torch.testing.assert_close(parameter, expected_policy[name], rtol=0, atol=0)
    _assert_nested_state_equal(b.policy_optimizer.state_dict(), expected_optimizer)
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(()), expected_torch, rtol=0, atol=0)


def test_base_save_gathers_rng_before_non_main_early_return(tmp_path):
    runner = _Mini(OmegaConf.create({}), tmp_path)
    distributed = _FakeDistributed(rank=1, is_main_process=False, gathered=[])
    runner.distributed = distributed

    path = runner.save_checkpoint()

    assert len(distributed.calls) == 1
    assert set(distributed.calls[0]) == {"python", "numpy", "torch", "cuda"}
    assert not pathlib.Path(path).exists()


def test_base_save_persists_all_gathered_rank_rng_states(tmp_path):
    from dreamervla.utils.seed import capture_rng_state

    states = [capture_rng_state(), capture_rng_state()]
    runner = _Mini(OmegaConf.create({}), tmp_path)
    distributed = _FakeDistributed(rank=0, is_main_process=True, gathered=states)
    runner.distributed = distributed

    path = runner.save_checkpoint()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert len(payload["rng_by_rank"]) == 2
    assert payload["rng_by_rank"][0]["python"] == states[0]["python"]
    np.testing.assert_array_equal(payload["rng_by_rank"][1]["numpy"][1], states[1]["numpy"][1])
    torch.testing.assert_close(
        payload["rng_by_rank"][1]["torch"], states[1]["torch"], rtol=0, atol=0
    )


def test_base_load_selects_distributed_rank_rng(tmp_path):
    from dreamervla.utils.seed import capture_rng_state

    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    rank_zero = capture_rng_state()
    random.seed(20)
    np.random.seed(20)
    torch.manual_seed(20)
    rank_one = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(()))

    runner = _Mini(OmegaConf.create({}), tmp_path)
    runner.distributed = _FakeDistributed(
        rank=1, is_main_process=False, gathered=[rank_zero, rank_one]
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    runner.load_payload(
        {
            "format_version": 2,
            "state_dicts": {},
            "pickles": {},
            "rng_by_rank": [rank_zero, rank_one],
        },
        restore_rng=True,
    )

    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(()), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("rng_by_rank", [None, []])
def test_base_direct_model_only_load_ignores_v2_rng_contract(tmp_path, rng_by_rank):
    source = _Mini(OmegaConf.create({}), tmp_path)
    with torch.no_grad():
        source.policy.weight.fill_(4.25)
        source.policy.bias.fill_(-2.5)
    runner = _Mini(OmegaConf.create({}), tmp_path)
    runner.distributed = _FakeDistributed(rank=3, is_main_process=False, gathered=[])
    payload = {
        "format_version": 2,
        "state_dicts": {"policy": source.policy.state_dict()},
        "pickles": {},
    }
    if rng_by_rank is not None:
        payload["rng_by_rank"] = rng_by_rank

    random.seed(314)
    np.random.seed(314)
    torch.manual_seed(314)
    expected = (random.random(), np.random.random(), torch.rand(()))
    random.seed(314)
    np.random.seed(314)
    torch.manual_seed(314)

    runner.load_payload(payload, include_keys=())

    for name, parameter in runner.policy.named_parameters():
        torch.testing.assert_close(parameter, source.policy.get_parameter(name), rtol=0, atol=0)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(()), expected[2], rtol=0, atol=0)


def test_base_load_checkpoint_rejects_v2_without_current_rank_rng(tmp_path):
    runner = _Mini(OmegaConf.create({}), tmp_path)
    missing_path = tmp_path / "missing_rng.ckpt"
    torch.save({"format_version": 2, "state_dicts": {}, "pickles": {}}, missing_path)

    with pytest.raises(RuntimeError, match="rng_by_rank"):
        runner.load_checkpoint(path=missing_path)

    missing_rank_path = tmp_path / "missing_rank_rng.ckpt"
    torch.save(
        {"format_version": 2, "state_dicts": {}, "pickles": {}, "rng_by_rank": []},
        missing_rank_path,
    )
    with pytest.raises(RuntimeError, match="rank 0"):
        runner.load_checkpoint(path=missing_rank_path)


def test_base_load_payload_legacy_rng_and_missing_rng_warning_once(tmp_path, monkeypatch):
    import dreamervla.runners.base_runner as base_runner_module
    from dreamervla.utils.seed import capture_rng_state

    runner = _Mini(OmegaConf.create({}), tmp_path)
    legacy_rng = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(()))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    runner.load_payload(
        {"format_version": 1, "state_dicts": {}, "pickles": {}, "rng": legacy_rng},
        restore_rng=True,
    )
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(()), expected[2], rtol=0, atol=0)

    monkeypatch.setattr(base_runner_module, "_LEGACY_RNG_WARNING_EMITTED", False)
    legacy_without_rng = {"format_version": 1, "state_dicts": {}, "pickles": {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.load_payload(legacy_without_rng, restore_rng=True)
        runner.load_payload({"state_dicts": {}, "pickles": {}}, restore_rng=True)
    matching = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert len(matching) == 1
    assert "RNG" in str(matching[0].message)


def test_base_legacy_missing_rng_warning_is_thread_safe(tmp_path, monkeypatch):
    import dreamervla.runners.base_runner as base_runner_module

    runner = _Mini(OmegaConf.create({}), tmp_path)
    monkeypatch.setattr(base_runner_module, "_LEGACY_RNG_WARNING_EMITTED", False)
    warning_calls = []
    start = threading.Barrier(8)

    def slow_warning(*args, **kwargs):
        time.sleep(0.02)
        warning_calls.append((args, kwargs))

    monkeypatch.setattr(base_runner_module.warnings, "warn", slow_warning)

    def load_legacy_payload(_index):
        start.wait()
        runner.load_payload(
            {"format_version": 1, "state_dicts": {}, "pickles": {}},
            restore_rng=True,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(load_legacy_payload, range(8)))

    assert len(warning_calls) == 1


class _Ready:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value


class _ActorGroup:
    def state_dict(self):
        return _Ready([{"weight": torch.ones(1)}])

    def optimizer_state_dict(self):
        return _Ready([{"state": {0: {"step": torch.tensor(1.0)}}}])


class _LearnerGroup:
    def state_dicts(self, _include_optimizers):
        return _Ready(
            [
                {
                    "world_model": {"weight": torch.ones(1)},
                    "classifier": {"weight": torch.ones(1)},
                    "world_model_optimizer": {"state": {1: {}}},
                    "classifier_optimizer": {"state": {2: {}}},
                    "classifier_threshold": 0.65,
                }
            ]
        )


def test_cotrain_checkpoint_uses_one_canonical_step_dir_and_latest(tmp_path):
    runner = object.__new__(CotrainRunner)
    runner.cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path)},
            "manual_cotrain": {"checkpoint_every": 1, "save_replay_state": False},
            "actor": {"train_cfg": {"optimizers": {"encoder": None}}},
        }
    )
    runner.config = runner.cfg
    runner._output_dir = str(tmp_path)
    runner._policy_initial_hash = ""
    runner._policy_final_hash = ""
    runner._applied_policy_steps = 3

    path = runner._maybe_save_manual_checkpoint(
        {"ActorGroup": _ActorGroup(), "LearnerGroup": _LearnerGroup()},
        global_step=3,
        metrics={"sync/policy_version": 3.0},
    )

    assert path == tmp_path / "checkpoints" / "global_step_3" / "manual_cotrain.ckpt"
    assert (tmp_path / "checkpoints" / "latest.ckpt").is_file()
    assert not (tmp_path / "checkpoints" / "manual_cotrain_step_3").exists()
    manifest = json.loads(
        (path.parent / "manual_cotrain_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run"]["hydra_config"] == "../../.hydra/config.yaml"
    assert "resolved_config" not in manifest["run"]


def test_cotrain_common_resume_path_loads_manual_payload(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "global_step_7" / "manual_cotrain.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"global_step": 7, "state_dicts": {}}, checkpoint)
    runner = object.__new__(CotrainRunner)
    runner.cfg = OmegaConf.create(
        {
            "training": {"resume": True, "resume_path": str(checkpoint)},
            "manual_cotrain": {"resume_ckpt": None},
        }
    )

    payload = runner._manual_resume_payload()

    assert payload is not None
    assert payload["global_step"] == 7
