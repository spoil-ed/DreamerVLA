from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset, Sampler

from dreamervla.runners.base_runner import BaseRunner
from dreamervla.utils.hf_checkpoint import (
    is_hf_checkpoint,
    load_hf_prefixed_tensors,
    resolve_hf_checkpoint_dir,
)


class _ConcreteRunner(BaseRunner):
    default_vla_init_dir = ""

    def run(self) -> object:
        return None


def test_base_runner_resolves_vla_init_path_and_frozen_encoder_cfg(
    tmp_path: Path,
) -> None:
    ckpt_root = tmp_path / "vla_ckpt"
    nested = ckpt_root / "checkpoint-1"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "init": {"vla_ckpt_path": str(ckpt_root)},
            "encoder": {
                "model_path": None,
                "freeze_backbone": False,
                "time_horizon": 5,
            },
            "world_model": {"hidden_dim": 4},
        }
    )
    workspace = _ConcreteRunner(cfg)

    assert workspace._resolve_vla_init_path() == str(nested.resolve())

    encoder_cfg = workspace._build_frozen_encoder_cfg(cfg)
    assert encoder_cfg.model_path == str(nested.resolve())
    assert encoder_cfg.freeze_backbone is True
    assert encoder_cfg.time_horizon == 5


def test_hf_checkpoint_helpers_resolve_nested_and_load_prefixed_tensors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    hf_dir = root / "checkpoint-1"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    torch.save(
        {
            "action_head.output_projection.weight": torch.ones(2, 3),
            "other.weight": torch.zeros(1),
        },
        hf_dir / "pytorch_model.bin",
    )

    assert is_hf_checkpoint(root)
    assert resolve_hf_checkpoint_dir(root) == hf_dir.resolve()
    tensors = load_hf_prefixed_tensors(root, "action_head.")

    assert set(tensors) == {"output_projection.weight"}
    assert torch.equal(tensors["output_projection.weight"], torch.ones(2, 3))


def test_base_runner_prefers_compat_latest_hf_when_canonical_missing(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "world_model": {"hidden_dim": 4},
        }
    )
    workspace = _ConcreteRunner(cfg)
    compat_hf = tmp_path / "out" / "ckpt" / "latest_hf"
    compat_hf.mkdir(parents=True)

    assert workspace.get_hf_checkpoint_path(prefer_existing=True) == compat_hf


class _ToyDataset(Dataset[int]):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        return int(index)

    @staticmethod
    def collate_fn(batch: list[int]) -> dict[str, list[int]]:
        return {"items": batch}


class _FakeDistributed:
    requires_collective_checkpointing = False
    is_main_process = True

    def __init__(self) -> None:
        self.sampler_calls: list[tuple[bool, bool]] = []

    def maybe_make_sampler(
        self, dataset: Dataset[Any], *, shuffle: bool, drop_last: bool
    ) -> None:
        self.sampler_calls.append((shuffle, drop_last))
        return None


def test_base_runner_builds_distributed_dataloader_with_dataset_collate(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "world_model": {"hidden_dim": 4},
        }
    )
    workspace = _ConcreteRunner(cfg)
    workspace.distributed = _FakeDistributed()

    loader = workspace.make_distributed_dataloader(
        _ToyDataset(),
        {
            "batch_size": 2,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 0,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },
        sanitize_worker_kwargs=True,
    )

    assert workspace.distributed.sampler_calls == [(False, False)]
    assert loader.persistent_workers is False
    assert next(iter(loader)) == {"items": [0, 1]}


class _EpochSampler(Sampler[int]):
    def __init__(self) -> None:
        self.epoch: int | None = None

    def __iter__(self):
        return iter([2, 1, 0])

    def __len__(self) -> int:
        return 3

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class _SamplerDistributed(_FakeDistributed):
    def __init__(self, sampler: _EpochSampler) -> None:
        super().__init__()
        self.sampler = sampler

    def maybe_make_sampler(
        self, dataset: Dataset[Any], *, shuffle: bool, drop_last: bool
    ) -> _EpochSampler:
        super().maybe_make_sampler(dataset, shuffle=shuffle, drop_last=drop_last)
        return self.sampler


def test_base_runner_wires_and_advances_distributed_sampler(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "world_model": {"hidden_dim": 4},
        }
    )
    workspace = _ConcreteRunner(cfg)
    sampler = _EpochSampler()
    workspace.distributed = _SamplerDistributed(sampler)

    loader = workspace.make_distributed_dataloader(
        _ToyDataset(),
        {"batch_size": 2, "shuffle": True, "drop_last": False, "num_workers": 0},
    )

    assert workspace.distributed.sampler_calls == [(True, False)]
    assert loader.sampler is sampler
    assert next(iter(loader)) == {"items": [2, 1]}

    workspace.set_dataloader_epoch(loader, 9)
    assert sampler.epoch == 9


class _HookRunner(_ConcreteRunner):
    include_keys = ("marker",)

    def __init__(self, config: Any, output_dir: str | None = None) -> None:
        super().__init__(config, output_dir=output_dir)
        self.module = nn.Linear(1, 1)
        self.marker = "saved"
        self.saved_keys: list[str] = []
        self.loaded_keys: list[str] = []

    def _state_dict_for_checkpoint(self, key: str, value: Any) -> dict[str, Any] | None:
        self.saved_keys.append(key)
        return {"custom": torch.tensor([7.0])}

    def _load_state_dict_from_checkpoint(
        self,
        key: str,
        value: Any,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.loaded_keys.append(key)
        self.marker = f"loaded_{float(state_dict['custom'][0])}"


def test_base_runner_checkpoint_uses_state_dict_hooks(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "world_model": {"hidden_dim": 4},
        }
    )
    workspace = _HookRunner(cfg)
    workspace.distributed = _FakeDistributed()

    path = tmp_path / "checkpoint.ckpt"
    workspace.save_checkpoint(path=path)
    payload = torch.load(path, weights_only=False)

    assert "module" in workspace.saved_keys
    assert torch.equal(payload["state_dicts"]["module"]["custom"], torch.tensor([7.0]))
    assert pickle.loads(payload["pickles"]["marker"]) == "saved"

    workspace.load_payload(
        {"state_dicts": {"module": {"custom": torch.tensor([3.0])}}, "pickles": {}}
    )
    assert workspace.loaded_keys == ["module"]
    assert workspace.marker == "loaded_3.0"


def test_vla_family_runners_inherit_shared_base_helpers() -> None:
    from dreamervla.runners.libero_vla_evaluation_runner import LIBEROVLAEvaluationRunner
    from dreamervla.runtime.libero_vla_evaluation_base import LIBEROVLAEvaluationBase
    from dreamervla.runtime.world_model_training_base import WorldModelTrainingBase

    for cls in (
        LIBEROVLAEvaluationBase,
        WorldModelTrainingBase,
        LIBEROVLAEvaluationRunner,
    ):
        assert "_resolve_vla_init_path" not in cls.__dict__
        assert "_build_frozen_encoder_cfg" not in cls.__dict__
        assert "make_distributed_dataloader" not in cls.__dict__
        assert "make_val_dataloaders" not in cls.__dict__
        assert "set_dataloader_epoch" not in cls.__dict__

    for cls in (
        LIBEROVLAEvaluationBase,
        WorldModelTrainingBase,
        LIBEROVLAEvaluationRunner,
    ):
        assert cls.save_checkpoint is BaseRunner.save_checkpoint
        assert cls.load_payload is BaseRunner.load_payload


def test_vla_hf_sidecar_strips_encoder_backbone_prefix() -> None:
    from dreamervla.runtime.libero_vla_evaluation_base import LIBEROVLAEvaluationBase

    state = LIBEROVLAEvaluationBase._extract_backbone_state_for_hf(
        {
            "state_dicts": {
                "encoder": {
                    "backbone.model.weight": torch.ones(1),
                    "backbone.module.action_head.bias": torch.zeros(1),
                    "other.weight": torch.full((1,), 2.0),
                }
            }
        }
    )

    assert state is not None
    assert set(state) == {"model.weight", "action_head.bias"}
