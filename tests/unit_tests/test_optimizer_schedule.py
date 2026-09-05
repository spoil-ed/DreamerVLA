from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from dreamervla.algorithms.dreamervla import world_model_pretrain_step
from dreamervla.utils.optim import apply_optimizer_lr_schedule, build_optimizer


class _GroupedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Linear(2, 2, bias=False)
        self.backbone = torch.nn.Linear(2, 2, bias=False)

    def optimizer_parameter_groups(self):
        return [
            {"group_name": "adapter", "params": self.adapter.parameters()},
            {
                "group_name": "pretrained_backbone",
                "params": self.backbone.parameters(),
            },
        ]

    def forward(self, batch):
        value = self.backbone(self.adapter(batch["obs_embedding"]))
        loss = value.square().mean()
        return {"loss": loss, "_loss": loss}


def _optim_cfg():
    return OmegaConf.create(
        {
            "name": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 0.0,
            "parameter_group_lrs": {
                "adapter": 1.0e-4,
                "pretrained_backbone": 1.0e-5,
            },
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 10,
            "adapter_alignment_steps": 20,
            "pretrained_backbone_warmup_steps": 10,
            "min_lr_ratio": 0.1,
        }
    )


def test_grouped_optimizer_uses_independent_peak_learning_rates() -> None:
    model = _GroupedModel()
    optimizer = build_optimizer(model, _optim_cfg())

    assert [group["group_name"] for group in optimizer.param_groups] == [
        "adapter",
        "pretrained_backbone",
    ]
    assert [group["initial_lr"] for group in optimizer.param_groups] == [1.0e-4, 1.0e-5]


def test_adapter_alignment_and_group_cosine_schedule() -> None:
    optimizer = build_optimizer(_GroupedModel(), _optim_cfg())

    start = apply_optimizer_lr_schedule(optimizer, _optim_cfg(), step=0, total_steps=100)
    assert start["adapter_learning_rate"] == 1.0e-5
    assert start["pretrained_backbone_learning_rate"] == 0.0
    assert start["adapter_alignment_active"] == 1.0
    assert optimizer.param_groups[1]["update_enabled"] is False

    backbone_start = apply_optimizer_lr_schedule(
        optimizer,
        _optim_cfg(),
        step=20,
        total_steps=100,
    )
    assert backbone_start["pretrained_backbone_learning_rate"] == pytest.approx(1.0e-6)
    assert backbone_start["adapter_alignment_active"] == 0.0
    assert optimizer.param_groups[1]["update_enabled"] is True

    end = apply_optimizer_lr_schedule(optimizer, _optim_cfg(), step=99, total_steps=100)
    assert end["adapter_learning_rate"] == pytest.approx(1.0e-5)
    assert end["pretrained_backbone_learning_rate"] == pytest.approx(1.0e-6)


def test_disabled_backbone_group_does_not_update_or_build_adam_moments() -> None:
    torch.manual_seed(5)
    model = _GroupedModel()
    optimizer = build_optimizer(model, _optim_cfg())
    apply_optimizer_lr_schedule(optimizer, _optim_cfg(), step=0, total_steps=100)
    adapter_before = model.adapter.weight.detach().clone()
    backbone_before = model.backbone.weight.detach().clone()

    world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=model,
        optimizer=optimizer,
        batch={"obs_embedding": torch.ones(4, 2)},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create(
            {
                "precision": "fp32",
                "grad_clip_norm": 1.0,
                "zero_grad_set_to_none": True,
            }
        ),
    )

    assert not torch.equal(model.adapter.weight, adapter_before)
    assert torch.equal(model.backbone.weight, backbone_before)
    assert model.adapter.weight in optimizer.state
    assert model.backbone.weight not in optimizer.state


def test_grouped_optimizer_schedule_reloads_at_exact_step() -> None:
    model = _GroupedModel()
    optimizer = build_optimizer(model, _optim_cfg())
    apply_optimizer_lr_schedule(optimizer, _optim_cfg(), step=25, total_steps=100)
    world_model_pretrain_step(
        policy=torch.nn.Identity(),
        world_model=model,
        optimizer=optimizer,
        batch={"obs_embedding": torch.ones(4, 2)},
        device=torch.device("cpu"),
        optim_cfg=OmegaConf.create({"precision": "fp32", "grad_clip_norm": 1.0}),
    )
    saved_model = model.state_dict()
    saved_optimizer = optimizer.state_dict()

    reloaded_model = _GroupedModel()
    reloaded_model.load_state_dict(saved_model, strict=True)
    reloaded_optimizer = build_optimizer(reloaded_model, _optim_cfg())
    reloaded_optimizer.load_state_dict(saved_optimizer)
    expected = apply_optimizer_lr_schedule(
        optimizer,
        _optim_cfg(),
        step=26,
        total_steps=100,
    )
    actual = apply_optimizer_lr_schedule(
        reloaded_optimizer,
        _optim_cfg(),
        step=26,
        total_steps=100,
    )

    assert actual == expected
    assert reloaded_optimizer.state_dict()["state"]
