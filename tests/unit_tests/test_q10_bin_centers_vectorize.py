"""PERF-Q10 safety gate: the vectorized `bin_centers` fancy-index decode must be
numerically identical to the Python list-comprehension reference it replaces.

The decode lives inside VLA embodiment methods whose modules import the vendored
`prismatic` tree (not importable in this worktree env). The mapping itself is a
pure NumPy operation that is identical across all variants, so per the task's
fallback rule the equivalence is asserted at the lowest reachable level here:
build `bin_centers` exactly as the models do, then compare the two decode forms
head-to-head (same values atol=0, same dtype, same shape, same ordering).
"""

import numpy as np
import pytest


def _bin_centers(n_action_bins: int) -> np.ndarray:
    """Reproduce the models' `bin_centers` construction exactly."""
    bins = np.linspace(-1, 1, n_action_bins)
    return (bins[:-1] + bins[1:]) / 2.0


def _reference_listcomp(bin_centers: np.ndarray, discretized_actions: np.ndarray) -> np.ndarray:
    """The original list-comprehension decode (the form being replaced)."""
    return np.asarray([bin_centers[da] for da in discretized_actions])


def _vectorized_fancy(bin_centers: np.ndarray, discretized_actions: np.ndarray) -> np.ndarray:
    """The vectorized fancy-index decode (the replacement, matches the official variant)."""
    return bin_centers[discretized_actions]


# Real config uses n_action_bins=256 -> bin_centers has 255 entries (indices 0..254).
N_ACTION_BINS = 256


def _make_discretized(rng: np.random.Generator, b: int, dim: int, max_bin: int) -> np.ndarray:
    """A [B, dim] int index array guaranteed to include the edge bins 0 and max."""
    arr = rng.integers(low=0, high=max_bin + 1, size=(b, dim)).astype(np.int64)
    # Force both edge bins to appear so they are exercised.
    arr[0, 0] = 0
    arr[-1, -1] = max_bin
    return arr


@pytest.mark.parametrize("n_action_bins", [N_ACTION_BINS, 5, 2])
@pytest.mark.parametrize("shape", [(1, 7), (4, 7), (3, 56), (8, 1)])
def test_fancy_index_equals_listcomp(n_action_bins, shape):
    bin_centers = _bin_centers(n_action_bins)
    max_bin = bin_centers.shape[0] - 1
    rng = np.random.default_rng(0)
    discretized = _make_discretized(rng, shape[0], shape[1], max_bin)

    ref = _reference_listcomp(bin_centers, discretized)
    vec = _vectorized_fancy(bin_centers, discretized)

    assert vec.shape == ref.shape, (vec.shape, ref.shape)
    assert vec.dtype == ref.dtype, (vec.dtype, ref.dtype)
    # atol=0: must be exactly equal, not merely close.
    assert np.array_equal(vec, ref)


def test_edge_bins_explicit():
    """Lower bin (0) and upper bin (max) decode identically under both forms."""
    bin_centers = _bin_centers(N_ACTION_BINS)
    max_bin = bin_centers.shape[0] - 1
    discretized = np.array([[0, max_bin, 0], [max_bin, 0, max_bin]], dtype=np.int64)

    ref = _reference_listcomp(bin_centers, discretized)
    vec = _vectorized_fancy(bin_centers, discretized)

    assert np.array_equal(vec, ref)
    assert vec.dtype == ref.dtype == np.float64
    assert vec.shape == ref.shape == (2, 3)
    # The decoded edge values themselves come straight from bin_centers.
    assert vec[0, 0] == bin_centers[0]
    assert vec[0, 1] == bin_centers[max_bin]
