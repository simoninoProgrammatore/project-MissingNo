"""Shared exploration hub: archive, curriculum and lineage for all parallel games."""

import zlib

import numpy as np
from missingno_envs import BackwardCurriculum
from missingno_envs.exploration import ExplorationHub, compress


def snaps(n):
    return [zlib.compress(f"s{i}".encode()) for i in range(n)]


def test_cells_get_a_lineage_node_and_starts_point_to_it():
    hub = ExplorationHub(archive_prob=1.0)
    hub.absorb({"cells": [((0, 1, 1), compress(b"state"), 0, compress(b"\x01\x02\x03"))]})
    spec = hub.next_start(np.random.default_rng(0))
    assert spec["kind"] == "archive" and zlib.decompress(spec["state"]) == b"state"
    assert hub.chain(spec["node"]) == [b"\x01\x02\x03"]


def test_lineage_chains_back_to_the_start_state():
    hub = ExplorationHub(archive_prob=1.0)
    a = hub.add_node(0, compress(b"\x01\x01"))
    b = hub.add_node(a, compress(b"\x02"))
    assert hub.chain(b) == [b"\x01\x01", b"\x02"]
    assert hub.chain(0) == []


def test_a_demo_is_kept_only_if_its_map_is_new_for_every_game():
    hub = ExplorationHub(curriculum_prob=1.0)
    demo = (13, 0, compress(bytes(200)), snaps(3), [0, 64, 128], 200)
    hub.absorb({"demos": [demo], "maps": {0, 13}})
    hub.absorb({"demos": [demo], "maps": {0, 13}})  # a second game reaches Route 2 too
    assert len(hub.curriculum) == 1


def test_a_demo_snapshot_gets_the_lineage_of_its_prefix():
    hub = ExplorationHub(curriculum_prob=1.0)
    actions = bytes(range(200))
    hub.absorb({"demos": [(13, 0, compress(actions), snaps(3), [0, 64, 128], 200)]})
    spec = hub.next_start(np.random.default_rng(0))
    assert spec["kind"] == "demo" and spec["target_map"] == 13
    assert hub.chain(spec["node"]) == [actions[:128]]  # the state of the last snapshot
    assert spec["budget"] == 2 * (200 - 128) + 256


def test_outcomes_move_the_demo_back_faster_when_flawless():
    cur = BackwardCurriculum(success_needed=4)
    cur.add_demo(snaps(100), target_map=1)
    demo = cur.demos[0]
    for _ in range(4):
        cur.record(demo, True)
    assert demo.start == 99 - 10  # a tenth of the path at once
    for success in (True, False, True, True, True):
        cur.record(demo, success)
    assert demo.start == 89 - 1  # with failures: one snapshot


def test_newer_demos_are_chosen_more_often():
    cur = BackwardCurriculum()
    for target in (1, 2, 3, 4):
        cur.add_demo(snaps(2), target_map=target)
    rng = np.random.default_rng(0)
    picks = [cur.sample_demo(rng).target_map for _ in range(2000)]
    assert picks.count(4) > picks.count(1) * 2


def test_the_hub_survives_a_new_session(tmp_path):
    hub = ExplorationHub(archive_prob=0.5, curriculum_prob=0.3)
    hub.absorb({"cells": [((0, 1, 1), compress(b"s"), 0, compress(b"\x01"))]})
    hub.absorb({"demos": [(13, 0, compress(bytes(10)), snaps(2), [0, 5], 10)]})
    hub.save(tmp_path / "exploration.pkl")
    new = ExplorationHub(archive_prob=0.4, curriculum_prob=0.3)
    new.load(tmp_path / "exploration.pkl")
    assert len(new.archive) == 1 and len(new.curriculum) == 1 and 13 in new.maps_ever
    assert new.archive_prob == 0.4  # the new session's settings win
    assert new.add_node(0, b"") == hub.next_node  # node ids keep counting


def test_visits_and_outcomes_update_the_archive():
    hub = ExplorationHub(archive_prob=1.0)
    key = (0, 1, 1)
    hub.absorb({"cells": [(key, compress(b"s"), 0, compress(b""))]})
    hub.absorb({"visits": {key: 5}, "outcomes": [("archive", key, 2.0)]})
    entry = hub.archive.cells[key]
    assert entry.visits == 6 and entry.outcomes == 1


def test_a_state_counts_as_chosen_only_when_an_episode_started_there():
    hub = ExplorationHub(archive_prob=1.0)
    key = (0, 1, 1)
    hub.absorb({"cells": [(key, compress(b"s"), 0, compress(b""))]})
    hub.starts(10, np.random.default_rng(0))  # handed out, maybe never used
    assert hub.archive.cells[key].chosen == 0
    hub.absorb({"outcomes": [("archive", key, 1.0)]})
    assert hub.archive.cells[key].chosen == 1
