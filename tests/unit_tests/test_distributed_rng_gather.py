from __future__ import annotations

from dreamervla.runtime.distributed import NopretokenizeSFTDistributedHelper


def _helper(world_size: int) -> NopretokenizeSFTDistributedHelper:
    return NopretokenizeSFTDistributedHelper(
        rank=0,
        local_rank=0,
        world_size=world_size,
        strategy="ddp",
        fsdp_mixed_precision="bf16",
        enable_activation_checkpointing=False,
    )


def test_all_gather_objects_single_process_returns_local_value(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("single-process gather must not call torch.distributed")

    monkeypatch.setattr("dreamervla.runtime.distributed.dist.all_gather_object", unexpected)

    value = {"rank": 0}
    assert _helper(world_size=1).all_gather_objects(value) == [value]


def test_all_gather_objects_uses_distributed_collective(monkeypatch):
    def fake_all_gather(output, value):
        output[:] = [value, {"rank": 1}]

    monkeypatch.setattr("dreamervla.runtime.distributed.dist.all_gather_object", fake_all_gather)

    assert _helper(world_size=2).all_gather_objects({"rank": 0}) == [
        {"rank": 0},
        {"rank": 1},
    ]
