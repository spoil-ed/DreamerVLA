from __future__ import annotations

import subprocess


def test_parse_nvidia_smi_csv_summarizes_utilization_and_memory() -> None:
    from dreamervla.utils.resource_metrics import parse_nvidia_smi_csv

    metrics = parse_nvidia_smi_csv(
        """
        0, 25, 1024, 81920
        1, 75, 4096, 81920
        """
    )

    assert metrics == {
        "gpu/utilization_pct_mean": 50.0,
        "gpu/utilization_pct_max": 75.0,
        "gpu/memory_used_mb_mean": 2560.0,
        "gpu/memory_used_mb_max": 4096.0,
        "gpu/memory_total_mb_max": 81920.0,
        "gpu/count": 2.0,
    }


def test_collect_nvidia_smi_metrics_degrades_to_empty_on_failure(monkeypatch) -> None:
    from dreamervla.utils.resource_metrics import collect_nvidia_smi_metrics

    def _fail(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _fail)

    assert collect_nvidia_smi_metrics() == {}
