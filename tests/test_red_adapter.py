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
    assert reached(
        map_id=R.PEWTER_GYM,
        x=0,
        y=0,
        party_levels=(12,),
        badges=1,
        items=frozenset({R.OAKS_PARCEL}),
    ) == ["M4", "M7", "M11", "M12"]
    assert R.MILESTONES[11].name == "Win the Boulder Badge"


def test_milestones_go_to_the_hall_of_fame_and_the_phase_goal_is_the_first_badge():
    from missingno_core import ProgressSignals
    from missingno_games import RedAdapter, goal_index

    ids = [m.id for m in R.MILESTONES]
    assert ids == [f"M{i}" for i in range(1, 41)]
    assert R.MILESTONES[goal_index(RedAdapter())].name == "Win the Boulder Badge"
    assert R.MILESTONES[goal_index(RedAdapter(), "M40")].reached(
        ProgressSignals(map_id=R.HALL_OF_FAME, x=0, y=0)
    )
    reached = lambda **kw: [m.id for m in R.MILESTONES if m.reached(ProgressSignals(**kw))]  # noqa: E731
    hms = frozenset({R.HM_CUT, R.HM_SURF, R.HM_STRENGTH, R.S_S_TICKET})
    assert {"M18", "M20", "M31", "M32"} <= set(reached(map_id=99, x=0, y=0, items=hms))
    assert {"M12", "M16", "M21", "M25", "M30", "M34", "M36", "M37"} <= set(
        reached(map_id=99, x=0, y=0, badges=8)
    )


def test_reads_bag_pokedex_and_map_size():
    mem = FakeMemory()
    mem[R.MAP_HEIGHT], mem[R.MAP_WIDTH] = 9, 10
    mem[R.NUM_BAG_ITEMS] = 2
    mem[R.BAG_ITEMS], mem[R.BAG_ITEMS + 1] = 0x14, 3  # Potion x3
    mem[R.BAG_ITEMS + 2], mem[R.BAG_ITEMS + 3] = 0x46, 1  # Oak's Parcel
    mem[R.POKEDEX_OWNED] = 0b00000001
    mem[R.POKEDEX_SEEN] = 0b00000111
    mem[R.POKEDEX_SEEN + 18] = 0b00000001

    s = RedAdapter().read(mem)
    assert s.map_area == 4 * 9 * 10
    assert s.items == {0x14, 0x46}
    assert s.key_items == {0x46}
    assert s.pokedex_owned == 1
    assert s.pokedex_seen == 4


def test_bag_count_is_clamped():
    mem = FakeMemory()
    mem[R.NUM_BAG_ITEMS] = 0xFF  # garbage during the intro
    assert len(RedAdapter().read(mem).items) <= 1  # 20 reads of zeros -> {0}


def test_hms_are_key_items():
    assert all(hm in R.KEY_ITEMS for hm in range(0xC4, 0xC9))
    assert 0x14 not in R.KEY_ITEMS  # Potion


def test_reads_party_experience():
    mem = FakeMemory()
    mem[R.PARTY_COUNT] = 2
    mem[R.PARTY_EXP], mem[R.PARTY_EXP + 1], mem[R.PARTY_EXP + 2] = 0x00, 0x01, 0x2C  # 300
    second = R.PARTY_EXP + R.PARTY_MON_SIZE
    mem[second], mem[second + 1], mem[second + 2] = 0x01, 0x00, 0x00  # 65536
    assert RedAdapter().read(mem).party_exp == (300, 65536)


def test_yellow_reads_the_same_signals_one_byte_lower():
    from missingno_games import YellowAdapter

    mem = FakeMemory()
    for red_address, value in [(R.MAP_ID, 40), (R.PLAYER_X, 3), (R.PLAYER_Y, 4), (R.BADGES, 1)]:
        mem[red_address - 1] = value
    mem[R.PARTY_COUNT - 1] = 1
    mem[R.PARTY_LEVELS[0] - 1] = 5
    s = YellowAdapter().read(mem)
    assert (s.cell, s.badges, s.party_levels) == ((40, 3, 4), 1, (5,))
    assert YellowAdapter.cgb is False and YellowAdapter.milestones == R.MILESTONES


def test_blue_is_red():
    from missingno_games import BlueAdapter

    mem = FakeMemory()
    mem[R.MAP_ID] = 12
    assert BlueAdapter().read(mem).map_id == RedAdapter().read(mem).map_id == 12
