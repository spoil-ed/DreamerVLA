"""Opt-in real-latent full-rollout update, DDP and frozen-readout resume checks.

May run under ``torchrun --nproc_per_node=2 -m pytest`` for the production DDP
static-graph contract. WM_ENCODED_SMOKE_CACHE selects a previously verified cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.nn.parallel import DistributedDataParallel

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VJEPA2_DECODED_SMOKE") != "1",
    reason="Requires real AC/decoder checkpoints, encoded cache and CUDA",
)


def test_full_rollout_frozen_decoder_update_and_reload(tmp_path: Path) -> None:
    from dreamervla.algorithms.dreamervla import world_model_pretrain_step
    from dreamervla.config_resolvers import register_dreamervla_resolvers
    from dreamervla.diagnostics.compare_wm_libero_rollout import _load_trajectory
    from dreamervla.utils.optim import apply_optimizer_lr_schedule, build_optimizer

    torch.set_num_threads(4)
    torch.manual_seed(7)
    register_dreamervla_resolvers()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if distributed:
        torch.distributed.init_process_group("nccl", device_id=device)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=wm_pi05_vjepa2_decoded_train",
                "task=pi05_libero_object",
                "profile=sinfra_pi05_object",
                "optim.world_model.lr_warmup_steps=1",
                "optim.world_model.adapter_alignment_steps=1",
                "optim.world_model.pretrained_backbone_warmup_steps=1",
            ],
        )
    wm = instantiate(cfg.world_model).to(device)
    report = wm.pretrained_load_report
    assert report.loaded_parameters == 302327808
    assert report.pretrained_parameter_ratio > 0.98
    readout_before = {
        key: value.cpu().clone() for key, value in wm.decoded_visual_loss.state_dict().items()
    }
    wrapped = (
        DistributedDataParallel(wm, device_ids=[rank], static_graph=True, broadcast_buffers=False)
        if distributed
        else wm
    )
    optimizer = build_optimizer(wrapped, cfg.optim.world_model)
    frozen_ids = {id(p) for p in wm.decoded_visual_loss.parameters()}
    assert not frozen_ids.intersection(
        id(p) for group in optimizer.param_groups for p in group["params"]
    )
    cache = torch.load(os.environ["WM_ENCODED_SMOKE_CACHE"], map_location="cpu", weights_only=True)
    index = rank % len(cache["episodes"])
    traj = _load_trajectory(
        Path(cfg.offline_warmup.data_dir), cache["episodes"][index], cache["length"]
    )
    encoded = cache["encoded"][index]
    batch = dict(
        obs_embedding=encoded["latent"].unsqueeze(0).to(device),
        prefix_attention_mask=encoded["attention_mask"].unsqueeze(0).to(device),
        actions=torch.as_tensor(traj.actions, device=device).unsqueeze(0),
        proprio=torch.as_tensor(traj.proprio, device=device).unsqueeze(0),
    )
    for step in range(3):
        apply_optimizer_lr_schedule(optimizer, cfg.optim.world_model, step=step, total_steps=3)
        metrics = world_model_pretrain_step(
            None, wrapped, optimizer, batch, device, cfg.optim, metrics_mode="loss_tensor"
        )
        assert all(
            torch.isfinite(metrics[key]) for key in ("loss", "grad_norm", "decoded_visual_loss")
        )
        assert not wm.decoded_visual_loss.decoder.training
        assert all(p.grad is None for p in wm.decoded_visual_loss.parameters())
    for key, before in readout_before.items():
        torch.testing.assert_close(
            wm.decoded_visual_loss.state_dict()[key].cpu(), before, rtol=0, atol=0
        )
    if distributed:
        value = wm.vjepa2_transition.output_adapter.weight.detach().clone()
        torch.distributed.broadcast(value, src=0)
        torch.testing.assert_close(
            wm.vjepa2_transition.output_adapter.weight, value, rtol=0, atol=0
        )
    path = tmp_path / "world_model.ckpt"
    torch.save({"model": wm.state_dict(), "optimizer": optimizer.state_dict()}, path)
    before = wm.vjepa2_transition.output_adapter.weight.detach().clone()
    with torch.no_grad():
        wm.vjepa2_transition.output_adapter.weight.add_(1)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    wm.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.testing.assert_close(wm.vjepa2_transition.output_adapter.weight, before, rtol=0, atol=0)
    print(
        {
            "rank": rank,
            "loss": metrics["loss"].item(),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "strict_reload": True,
        }
    )
    if distributed:
        torch.distributed.destroy_process_group()
