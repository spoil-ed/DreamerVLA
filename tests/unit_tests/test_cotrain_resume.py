"""Pins the base save/load contract that online_cotrain resume (R1) relies on:
optimizer state + scalar attributes round-trip, and exclude_keys keeps frozen
modules out of the payload.
"""

import pathlib

import torch
from omegaconf import OmegaConf

from dreamervla.runners.base_runner import BaseRunner


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


def test_online_cotrain_checkpoint_keys_extend_parent():
    # R1 wiring: cotrain rounds-trips its scalar threshold + step and never
    # checkpoints the frozen reference policy / encoder.
    from dreamervla.runners.dreamervla_runner import DreamerVLARunner
    from dreamervla.runners.online_cotrain_runner import OnlineCotrainRunner

    assert set(DreamerVLARunner.include_keys) <= set(OnlineCotrainRunner.include_keys)
    assert "classifier_threshold" in OnlineCotrainRunner.include_keys
    assert "global_step" in OnlineCotrainRunner.include_keys
    assert "ref_policy" in OnlineCotrainRunner.exclude_keys
    assert "encoder" in OnlineCotrainRunner.exclude_keys
