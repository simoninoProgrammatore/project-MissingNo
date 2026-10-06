from missingno_games import RedAdapter
from missingno_games import red as R


class FakeMemory(dict):
    def __missing__(self, address):
        return 0


def test_reads_position_badges_and_levels():
    mem = FakeMemory()
    mem[R.MAP_ID] = 12
    mem[R.PLAYER_X] = 5
    mem[R.PLAYER_Y] = 7
    mem[R.BADGES] = 0b00000101  # two badges
    mem[R.PARTY_COUNT] = 2
    mem[R.PARTY_LEVELS[0]] = 14
    mem[R.PARTY_LEVELS[1]] = 9
    mem[R.PARTY_LEVELS[2]] = 99  # beyond party count: must be ignored
    mem[R.IS_IN_BATTLE] = 1

    s = RedAdapter().read(mem)
    assert s.cell == (12, 5, 7)
    assert s.badges == 2
    assert s.party_levels == (14, 9)
    assert s.total_level == 23
    assert s.in_battle


def test_lost_battle_flag_is_not_in_battle():
    mem = FakeMemory()
    mem[R.IS_IN_BATTLE] = 0xFF
    assert not RedAdapter().read(mem).in_battle


def test_party_count_is_clamped():
    mem = FakeMemory()
    mem[R.PARTY_COUNT] = 0xFF  # garbage during the intro
    assert len(RedAdapter().read(mem).party_levels) == 6


def test_phase1_milestones():
    from missingno_core import ProgressSignals

    reached = lambda **kw: [m.id for m in R.MILESTONES if m.reached(ProgressSignals(**kw))]  # noqa: E731
    assert reached(map_id=R.REDS_HOUSE_2F, x=0, y=0) == []
    assert reached(map_id=R.REDS_HOUSE_1F, x=0, y=0) == ["M1"]
    assert reached(map_id=R.OAKS_LAB, x=0, y=0, party_levels=(5,)) == ["M3", "M4"]
    assert reached(map_id=R.VIRIDIAN_CITY, x=0, y=0, party_levels=(6,)) == ["M4", "M6"]
