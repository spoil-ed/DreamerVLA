from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

import dreamervla.runners.base_runner as base_runner
from dreamervla.runners.base_runner import BaseRunner


class _FakeDistributed:
    requires_collective_checkpointing = False
    is_main_process = True


class _HookRunner(BaseRunner):
    default_vla_init_dir = ""
    include_keys = ("marker",)

    def __init__(self, config: Any, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir=output_dir)
        self.module = nn.Linear(1, 1)
        self.marker = "saved"

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        return {"custom": torch.tensor([7.0])}

    def run(self) -> object:
        return None


def _make_runner(tmp_path: Path) -> _HookRunner:
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "world_model": {"hidden_dim": 4},
        }
    )
    runner = _HookRunner(cfg)
    runner.distributed = _FakeDistributed()
    return runner


def test_save_checkpoint_writes_atomically_no_leftover_tmp(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    path = tmp_path / "checkpoints" / "latest.ckpt"

    runner.save_checkpoint(path=path)

    assert path.is_file()
    # No partial / leftover temp file from the temp-then-rename write.
    leftovers = list(path.parent.glob("*.tmp"))
    assert leftovers == []

    payload = torch.load(path, weights_only=False)
    assert torch.equal(payload["state_dicts"]["module"]["custom"], torch.tensor([7.0]))
    assert pickle.loads(payload["pickles"]["marker"]) == "saved"


def test_latest_plus_topk_serializes_payload_once(tmp_path: Path, monkeypatch) -> None:
    runner = _make_runner(tmp_path)
    latest = tmp_path / "checkpoints" / "latest.ckpt"
    topk = tmp_path / "checkpoints" / "topk-0001.ckpt"

    real_save = base_runner.torch.save
    calls: list[Any] = []

    def _counting_save(obj: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(dst)
        return real_save(obj, dst, *args, **kwargs)

    monkeypatch.setattr(base_runner.torch, "save", _counting_save)

    runner.save_checkpoint(path=latest, extra_paths=(topk,))

    # Payload serialized exactly once, then materialized at the top-k path.
    assert len(calls) == 1
    assert latest.is_file()
    assert topk.is_file()


def test_latest_and_topk_round_trip_to_identical_state(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    latest = tmp_path / "checkpoints" / "latest.ckpt"
    topk = tmp_path / "checkpoints" / "topk-0001.ckpt"

    runner.save_checkpoint(path=latest, extra_paths=(topk,))

    latest_payload = torch.load(latest, weights_only=False)
    topk_payload = torch.load(topk, weights_only=False)

    for payload in (latest_payload, topk_payload):
        assert torch.equal(payload["state_dicts"]["module"]["custom"], torch.tensor([7.0]))
        assert pickle.loads(payload["pickles"]["marker"]) == "saved"
