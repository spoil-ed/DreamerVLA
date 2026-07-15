"""Pins the base save/load contract that online_cotrain resume (R1) relies on:
optimizer state + scalar attributes round-trip, and exclude_keys keeps frozen
modules out of the payload.
"""

import json
import pathlib

import torch
from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.runners.cotrain_runner import CotrainRunner


class _Mini(BaseRunner):
    include_keys = ("global_step", "classifier_threshold")
    exclude_keys = ("frozen",)

    def __init__(self, cfg, tmp):
        self.cfg = cfg
        self._out = tmp
        self.global_step = 0
        self.classifier_threshold = 0.5
        self.policy = torch.nn.Linear(3, 2)
        self.frozen = torch.nn.Linear(3, 2)  # must NOT be checkpointed
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def get_checkpoint_path(self, tag="latest", *, prefer_existing=False):
        return pathlib.Path(self._out) / f"{tag}.ckpt"

    def run(self):  # abstract
        return None


def test_base_checkpoint_roundtrips_optimizer_and_scalars(tmp_path):
    cfg = OmegaConf.create({})
    a = _Mini(cfg, tmp_path)
    # take an optimizer step so momentum buffers are non-empty
    loss = a.policy(torch.ones(1, 3)).sum()
    loss.backward()
    a.policy_optimizer.step()
    a.global_step = 7
    a.classifier_threshold = 0.73
    path = a.save_checkpoint()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "policy" in payload["state_dicts"]
    assert "policy_optimizer" in payload["state_dicts"]
    assert "frozen" not in payload["state_dicts"]  # exclude_keys honored

    b = _Mini(cfg, tmp_path)
    b.load_checkpoint(path=path)
    assert b.global_step == 7
    assert abs(b.classifier_threshold - 0.73) < 1e-9
    assert b.policy_optimizer.state_dict()["state"]  # momentum restored


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
