from dreamervla.algorithms.staleness import is_stale, version_age


def test_version_age_clamps_at_zero():
    assert version_age(3, 7) == 4
    assert version_age(9, 7) == 0  # record newer than current -> 0, not negative


def test_is_stale_threshold_boundary():
    # age == threshold is kept; age > threshold is stale
    assert is_stale(record_version=0, current_version=2, threshold=2) is False
    assert is_stale(record_version=0, current_version=3, threshold=2) is True


def test_negative_threshold_disables_gating():
    assert is_stale(record_version=0, current_version=10_000, threshold=-1) is False
