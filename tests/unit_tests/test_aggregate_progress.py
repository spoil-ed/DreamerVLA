"""AggregateProgress: one global bar across independent (torchrun) collect ranks.

Each rank persists {done,total,finished} to a shared dir; rank 0 renders the SUM so a
multi-rank collect shows up as one moving total instead of a rank-0-only view.
"""

from __future__ import annotations

from dreamervla.utils.progress import AggregateProgress


def test_global_sums_done_and_total_across_ranks(tmp_path):
    pd = tmp_path / ".progress"
    # rank 1 is silent but persists its progress to the shared dir.
    r1 = AggregateProgress(10, "collect", rank=1, world_size=2, progress_dir=pd)
    r1.set(3)
    r0 = AggregateProgress(10, "collect", rank=0, world_size=2, progress_dir=pd)
    r0.set(2)
    # global = (3+2) done / (10+10) total
    assert r0._global() == (5, 20)
    r1.close()
    r0.close()


def test_falls_back_to_own_counts_without_shared_dir():
    p = AggregateProgress(10, "collect", rank=0, world_size=4, progress_dir=None)
    p.set(4)
    assert p._global() == (4, 10)  # no shared dir -> only this rank's counts


def test_single_world_size_is_not_shared(tmp_path):
    p = AggregateProgress(10, "collect", rank=0, world_size=1, progress_dir=tmp_path / "p")
    p.set(7)
    assert p._global() == (7, 10)


def test_rank0_renders_global_total_in_the_line(tmp_path):
    pd = tmp_path / ".progress"
    AggregateProgress(8, "collect", rank=1, world_size=2, progress_dir=pd).set(3)
    lines: list[str] = []
    r0 = AggregateProgress(
        8,
        "collect",
        rank=0,
        world_size=2,
        progress_dir=pd,
        sink=lines.append,
        min_interval_s=0.0,
    )
    r0.set(2)
    assert lines, "rank 0 must render"
    # bar retargets to the global total (8+8=16) and global done (3+2=5)
    assert "16" in lines[-1] and "5" in lines[-1]


def test_nonzero_rank_is_silent(tmp_path):
    pd = tmp_path / ".progress"
    lines: list[str] = []
    r1 = AggregateProgress(
        5,
        "collect",
        rank=1,
        world_size=2,
        progress_dir=pd,
        sink=lines.append,
        min_interval_s=0.0,
    )
    r1.set(2)
    r1.close()
    assert lines == []  # only rank 0 prints; others just persist


def test_missing_sibling_file_counts_as_zero_not_crash(tmp_path):
    pd = tmp_path / ".progress"
    # rank 0 reports before rank 2/3 have written anything.
    r0 = AggregateProgress(10, "collect", rank=0, world_size=3, progress_dir=pd)
    r0.set(4)
    assert r0._global() == (4, 10)  # absent ranks contribute 0, no exception
