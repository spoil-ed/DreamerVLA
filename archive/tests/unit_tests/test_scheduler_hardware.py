from __future__ import annotations


def test_discover_local_accelerators_reads_torch_cuda(monkeypatch) -> None:
    import torch

    from dreamervla.scheduler.hardware import discover_local_accelerators

    class _Props:
        total_memory = 80

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _Props())

    devices = discover_local_accelerators()

    assert [device.index for device in devices] == [0, 1]
    assert [device.name for device in devices] == ["GPU-0", "GPU-1"]
    assert [device.total_memory_bytes for device in devices] == [80, 80]


def test_discover_local_accelerators_is_empty_without_cuda(monkeypatch) -> None:
    import torch

    from dreamervla.scheduler.hardware import discover_local_accelerators

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert discover_local_accelerators() == []
