"""Mixture weighting, distributed epoch partitioning and source contracts."""

from __future__ import annotations

import pytest
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, TensorDataset

from dreamervla.dataset.base.multi_dataloader import DistributedMixtureSampler, MultiDataset


def test_concatenation_preserves_samples_and_can_be_batched() -> None:
    sources = [TensorDataset(torch.tensor([3, 4])), TensorDataset(torch.tensor([10]))]
    dataset = MultiDataset(sources)
    assert [dataset[index][0].item() for index in range(len(dataset))] == [3, 4, 10]
    assert dataset[-1][0].item() == 10
    assert next(iter(DataLoader(dataset, batch_size=3)))[0].tolist() == [3, 4, 10]
    with pytest.raises(IndexError):
        dataset[3]


def test_source_weights_are_independent_of_source_size() -> None:
    dataset = MultiDataset([TensorDataset(torch.arange(1)), TensorDataset(torch.arange(100))])
    sampler = DistributedMixtureSampler(dataset, weights=[1, 1], num_samples=4000, seed=51)
    first_source = sum(index == 0 for index in sampler)
    assert 1800 < first_source < 2200
    only_second = DistributedMixtureSampler(dataset, weights=[0, 1], num_samples=100)
    assert all(index > 0 for index in only_second)


def test_eight_rank_epoch_partition_and_resume_reproduce_global_draws() -> None:
    dataset = MultiDataset([TensorDataset(torch.arange(13)), TensorDataset(torch.arange(29))])
    global_sampler = DistributedMixtureSampler(dataset, weights=[1, 3], num_samples=256, seed=9)
    ranks = [
        DistributedMixtureSampler(
            dataset, weights=[1, 3], num_samples=256, seed=9, num_replicas=8, rank=rank
        )
        for rank in range(8)
    ]
    for sampler in [global_sampler, *ranks]:
        sampler.set_epoch(3)
    expected = list(global_sampler)
    assert [
        value for step in zip(*(list(rank) for rank in ranks), strict=True) for value in step
    ] == expected
    assert all(len(rank) == 32 for rank in ranks)
    global_sampler.set_epoch(4)
    assert list(global_sampler) != expected
    global_sampler.set_epoch(3)
    assert list(global_sampler) == expected


@pytest.mark.parametrize("weights", ([0, 0], [-1, 2], [float("nan"), 1], [1]))
def test_invalid_source_weights_are_rejected(weights: list[float]) -> None:
    dataset = MultiDataset([TensorDataset(torch.arange(2)), TensorDataset(torch.arange(4))])
    with pytest.raises(ValueError, match="weights"):
        DistributedMixtureSampler(dataset, weights=weights, num_samples=16)


def test_hydra_can_instantiate_nested_datasets(tmp_path) -> None:
    import h5py
    import numpy as np

    with h5py.File(tmp_path / "task_demo.hdf5", "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo["actions"] = np.zeros((2, 7), dtype=np.float32)
        demo.create_group("obs")["agentview_rgb"] = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    source = {
        "_target_": "dreamervla.dataset.base.hdf5_dataloader.HDF5ActionChunkDataset",
        "hdf5_dir": str(tmp_path),
        "action_horizon": 2,
    }
    dataset = instantiate(
        {
            "_target_": "dreamervla.dataset.base.multi_dataloader.MultiDataset",
            "datasets": [source, source],
        }
    )
    assert len(dataset) == 4
    assert dataset[2]["actions"].shape == (2, 7)
    assert dataset.get_normalizer() == {"sources": [None, None]}


def test_distributed_epoch_length_must_not_silently_drop_samples() -> None:
    dataset = MultiDataset([TensorDataset(torch.arange(13))])
    with pytest.raises(ValueError, match="divisible"):
        DistributedMixtureSampler(dataset, weights=[1], num_samples=13, num_replicas=8)
