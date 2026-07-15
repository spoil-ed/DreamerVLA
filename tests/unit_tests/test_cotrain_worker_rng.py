"""RNG checkpoint contracts for the manual-cotrain trainable workers."""

import random

import numpy as np
import pytest
import torch

from dreamervla.utils.seed import capture_rng_state
from dreamervla.workers.actor.embodied_fsdp_actor import EmbodiedFSDPActor
from dreamervla.workers.actor.learner_worker import LearnerWorker


@pytest.mark.parametrize("worker_cls", [EmbodiedFSDPActor, LearnerWorker])
def test_cotrain_worker_rng_roundtrips_its_rank(worker_cls):
    worker = object.__new__(worker_cls)
    worker.rank = 1
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    rank_zero = capture_rng_state()
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    rank_one = worker.rng_state_dict()
    expected = (random.random(), np.random.random(), torch.rand(()))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    worker.load_rng_state_dict([rank_zero, rank_one])

    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(()), expected[2], rtol=0, atol=0)
    assert set(rank_one) == {"python", "numpy", "torch", "cuda"}


@pytest.mark.parametrize("worker_cls", [EmbodiedFSDPActor, LearnerWorker])
def test_cotrain_worker_rng_rejects_missing_rank(worker_cls):
    worker = object.__new__(worker_cls)
    worker.rank = 2

    with pytest.raises(RuntimeError, match="rank 2"):
        worker.load_rng_state_dict([capture_rng_state()])
