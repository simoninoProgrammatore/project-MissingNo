"""State archive (Go-Explore style): cells, saving, sampling, and the environment."""

import numpy as np
from missingno_core import ProgressSignals
from missingno_envs import StateArchive, cell_key


def S(**kw):
    base = {"map_id": 40, "x": 5, "y": 5}
    base.update(kw)
    return ProgressSignals(**base)


def test_cells_are_coarse_in_space():
    assert cell_key(S(x=4, y=4)) == cell_key(S(x=7, y=7))
    assert cell_key(S(x=4, y=4)) != cell_key(S(x=8, y=4))


def test_progress_separates_cells_in_the_same_place():
    before = S()
    with_starter = S(party_levels=(5,), pokedex_owned=1, pokedex_seen=1)
    after_battle = S(party_levels=(5,), pokedex_owned=1, pokedex_seen=2)
    keys = {cell_key(before), cell_key(with_starter), cell_key(after_battle)}
    assert len(keys) == 3


def test_new_cells_are_saved_once():
    archive = StateArchive()
    calls = []

    def save():
        calls.append(1)
        return b"state"

    assert archive.observe(S(), save)
    assert not archive.observe(S(), save)
    assert len(archive) == 1 and len(calls) == 1  # saving is skipped for known cells


def test_archive_is_bounded():
    archive = StateArchive(max_cells=2)
    for x in (0, 8, 16, 24):
        archive.observe(S(x=x), lambda: b"s")
    assert len(archive) == 2


def test_sampling_returns_the_saved_state_and_prefers_the_frontier():
    archive = StateArchive()
    archive.observe(S(x=0), lambda: b"old")
    archive.observe(S(x=8), lambda: b"new")
    old = cell_key(S(x=0))
    archive.cells[old].chosen = 1000
    archive.cells[old].visits = 100000  # a place the agent knows too well
    rng = np.random.default_rng(0)
    picks = [archive.sample(rng)[1] for _ in range(200)]
    assert set(picks) <= {b"old", b"new"}
    assert picks.count(b"new") > picks.count(b"old")
