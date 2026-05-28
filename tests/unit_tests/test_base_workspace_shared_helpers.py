from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset, Sampler

from src.workspace.base_workspace import BaseWorkspace


class _ConcreteWorkspace(BaseWorkspace):
    default_vla_init_dir = ""

    def run(self) -> object:
        return None


def test_base_workspace_resolves_vla_init_path_and_frozen_encoder_cfg(
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
    workspace = _ConcreteWorkspace(cfg)

    assert workspace._resolve_vla_init_path() == str(nested.resolve())

    encoder_cfg = workspace._build_frozen_encoder_cfg(cfg)
    assert encoder_cfg.model_path == str(nested.resolve())
    assert encoder_cfg.freeze_backbone is True
    assert encoder_cfg.time_horizon == 5


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


def test_base_workspace_builds_distributed_dataloader_with_dataset_collate(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "world_model": {"hidden_dim": 4},
        }
    )
    workspace = _ConcreteWorkspace(cfg)
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


def test_base_workspace_wires_and_advances_distributed_sampler(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "world_model": {"hidden_dim": 4},
        }
    )
    workspace = _ConcreteWorkspace(cfg)
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


class _HookWorkspace(_ConcreteWorkspace):
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


def test_base_workspace_checkpoint_uses_state_dict_hooks(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "training": {"out_dir": str(tmp_path / "out")},
            "world_model": {"hidden_dim": 4},
        }
    )
    workspace = _HookWorkspace(cfg)
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


def test_vla_family_workspaces_inherit_shared_base_helpers() -> None:
    from src.workspace.chameleon_latent_action_wm_workspace import (
        ChameleonLatentActionWMWorkspace,
    )
    from src.workspace.dreamer_vla_workspace import DreamerVLAWorkspace
    from src.workspace.eval_libero_vla_workspace import EvalLiberoVLAWorkspace
    from src.workspace.pretokenize_vla_workspace import PretokenizeVLAWorkspace

    for cls in (
        PretokenizeVLAWorkspace,
        DreamerVLAWorkspace,
        ChameleonLatentActionWMWorkspace,
        EvalLiberoVLAWorkspace,
    ):
        assert "_resolve_vla_init_path" not in cls.__dict__
        assert "_build_frozen_encoder_cfg" not in cls.__dict__
        assert "make_distributed_dataloader" not in cls.__dict__
        assert "make_val_dataloaders" not in cls.__dict__
        assert "set_dataloader_epoch" not in cls.__dict__

    for cls in (
        PretokenizeVLAWorkspace,
        DreamerVLAWorkspace,
        ChameleonLatentActionWMWorkspace,
        EvalLiberoVLAWorkspace,
    ):
        assert cls.save_checkpoint is BaseWorkspace.save_checkpoint
        assert cls.load_payload is BaseWorkspace.load_payload
