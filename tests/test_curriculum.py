"""Backward curriculum from the agent's own successes, and learning-progress archive."""

import numpy as np
from missingno_core import ProgressSignals
from missingno_envs import BackwardCurriculum, StateArchive, cell_key


def snaps(n):
    import zlib

    return [zlib.compress(f"s{i}".encode()) for i in range(n)]


def test_demo_starts_one_snapshot_before_the_success():
    cur = BackwardCurriculum()
    cur.add_demo(snaps(5), target_map=12)
    demo, state = cur.sample(np.random.default_rng(0))
    assert demo.start == 4 and state == b"s4"
    assert demo.target_map == 12


def test_start_moves_back_when_mastered_and_demo_completes():
    cur = BackwardCurriculum(success_needed=2)
    cur.add_demo(snaps(3), target_map=12)
    demo = cur.demos[0]
    for expected_start in (1, 0):
        cur.record(demo, True)
        cur.record(demo, True)
        assert demo.start == expected_start
    cur.record(demo, True)
    cur.record(demo, True)
    assert len(cur) == 0 and cur.completed == 1  # the whole path is learned


def test_failures_do_not_move_back_and_hopeless_demos_are_dropped():
    cur = BackwardCurriculum(max_attempts=5)
    cur.add_demo(snaps(3), target_map=12)
    demo = cur.demos[0]
    for _ in range(4):
        cur.record(demo, False)
    assert demo.start == 2 and len(cur) == 1
    cur.record(demo, False)
    assert len(cur) == 0 and cur.abandoned == 1


def test_budget_grows_as_the_start_moves_back():
    cur = BackwardCurriculum(interval=64, success_needed=1)
    cur.add_demo(snaps(10), target_map=1)
    demo = cur.demos[0]
    near = demo.budget()
    cur.record(demo, True)
    assert demo.budget() > near


def test_only_the_newest_demos_are_kept():
    cur = BackwardCurriculum(max_demos=2)
    for target in (1, 2, 3):
        cur.add_demo(snaps(2), target_map=target)
    assert [d.target_map for d in cur.demos] == [2, 3]


def test_archive_prefers_cells_where_results_are_changing():
    archive = StateArchive()
    S = lambda x: ProgressSignals(map_id=0, x=x, y=0)  # noqa: E731
    archive.observe(S(0), lambda: b"stable")
    archive.observe(S(8), lambda: b"learning")
    stable, learning = cell_key(S(0)), cell_key(S(8))
    for r in (1.0, 1.0, 1.0, 1.0):
        archive.record_outcome(stable, r)  # always the same result: nothing to learn
    for r in (0.0, 0.0, 5.0, 5.0):
        archive.record_outcome(learning, r)  # results improving: learning happens here
    assert archive.cells[learning].progress > archive.cells[stable].progress == 0
    rng = np.random.default_rng(0)
    picks = [archive.sample(rng)[1] for _ in range(300)]
    assert picks.count(b"learning") > picks.count(b"stable")
