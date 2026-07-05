from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("num_gpus", [2, 3, 4])
def test_noray_torchrun_multigpu_smoke(num_gpus: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        pytest.skip(f"requires at least {num_gpus} visible CUDA GPUs")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    parent = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parent_devices = [part.strip() for part in parent.split(",") if part.strip()]
    if len(parent_devices) >= num_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(parent_devices[:num_gpus])
    else:
        env.setdefault(
            "CUDA_VISIBLE_DEVICES",
            ",".join(str(idx) for idx in range(num_gpus)),
        )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "-m",
            "dreamervla.diagnostics.smoke_torchrun_multigpu",
        ],
        check=False,
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    assert len(records) == num_gpus
    assert {record["rank"] for record in records} == set(range(num_gpus))
    assert {record["local_rank"] for record in records} == set(range(num_gpus))
    expected_sum = float(num_gpus * (num_gpus + 1) // 2)
    assert {record["all_reduce_sum"] for record in records} == {expected_sum}
    assert {record["ray_imported"] for record in records} == {False}
