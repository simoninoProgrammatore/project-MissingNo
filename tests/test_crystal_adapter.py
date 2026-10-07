"""Crystal adapter: banked memory, (group, number) maps, pockets, two badge sets."""

from missingno_games import ADAPTERS, CrystalAdapter
from missingno_games import crystal as C


class BankedMemory(dict):
    """Fake CGB memory: `m[1, address]` reads WRAM bank 1, `m[address]` the mapped bank."""

    def __init__(self) -> None:
        super().__init__()
        self.bank1: dict[int, int] = {}

    def __getitem__(self, key):
        if isinstance(key, tuple):
            bank, address = key
            assert bank == 1
            return self.bank1.get(address, 0)
        return self.get(key, 0)


def test_reads_from_wram_bank_1_whatever_bank_is_mapped():
    mem = BankedMemory()
    mem.bank1[C.MAP_GROUP], mem.bank1[C.MAP_NUMBER] = 24, 4  # New Bark Town
    mem.bank1[C.PLAYER_X], mem.bank1[C.PLAYER_Y] = 6, 3
    mem[C.MAP_GROUP] = 99  # another bank is mapped right now: must be ignored
    s = CrystalAdapter().read(mem)
    assert s.map_id == C.NEW_BARK_TOWN
    assert s.cell == (C.NEW_BARK_TOWN, 6, 3)


def test_party_badges_and_battle():
    mem = BankedMemory()
    b = mem.bank1
    b[C.PARTY_COUNT] = 2
    b[C.PARTY_MON_1 + C.LEVEL_OFFSET] = 7
    b[C.PARTY_MON_1 + C.PARTY_MON_SIZE + C.LEVEL_OFFSET] = 4
    e = C.PARTY_MON_1 + C.EXP_OFFSET
    b[e], b[e + 1], b[e + 2] = 0, 1, 0x2C  # 300
    b[C.JOHTO_BADGES] = 0b1
    b[C.KANTO_BADGES] = 0b11
    b[C.BATTLE_MODE] = 2
    s = CrystalAdapter().read(mem)
    assert s.party_levels == (7, 4)
    assert s.party_exp == (300, 0)
    assert s.badges == 3
    assert s.in_battle


def test_pockets_key_items_and_hms():
    mem = BankedMemory()
    b = mem.bank1
    b[C.NUM_ITEMS] = 1
    b[C.ITEMS] = 0x12  # a potion-like item, quantity follows
    b[C.NUM_BALLS] = 1
    b[C.BALLS] = 0x05
    b[C.NUM_KEY_ITEMS] = 1
    b[C.KEY_ITEMS] = 0x3B
    b[C.TMS_HMS + 0] = 1  # TM01
    b[C.TMS_HMS + C.NUM_TMS] = 1  # HM01
    s = CrystalAdapter().read(mem)
    assert {0x12, 0x05, 0x3B} <= s.items
    assert 0x3B in s.key_items
    hm01, tm01 = C.TM_HM_ID + C.NUM_TMS, C.TM_HM_ID
    assert hm01 in s.key_items and hm01 in s.items
    assert tm01 in s.items and tm01 not in s.key_items


def test_pokedex_and_map_size():
    mem = BankedMemory()
    b = mem.bank1
    b[C.POKEDEX_CAUGHT] = 0b1
    b[C.POKEDEX_SEEN] = 0b111
    b[C.POKEDEX_SEEN + 31] = 0b1  # the last byte counts too (251 species)
    b[C.MAP_HEIGHT], b[C.MAP_WIDTH] = 9, 10
    s = CrystalAdapter().read(mem)
    assert (s.pokedex_owned, s.pokedex_seen) == (1, 4)
    assert s.map_area == 4 * 9 * 10


def test_counts_are_clamped():
    mem = BankedMemory()
    for address in (C.PARTY_COUNT, C.NUM_ITEMS, C.NUM_KEY_ITEMS, C.NUM_BALLS):
        mem.bank1[address] = 0xFF  # garbage during the intro
    s = CrystalAdapter().read(mem)
    assert len(s.party_levels) == 6


def test_milestones_end_with_the_first_badge_and_crystal_is_held_out():
    ms = CrystalAdapter.milestones
    assert [m.id for m in ms] == [f"M{i}" for i in range(1, 13)]
    mem = BankedMemory()
    mem.bank1[C.JOHTO_BADGES] = 1
    assert ms[-1].reached(CrystalAdapter().read(mem))
    assert ADAPTERS["crystal"].held_out
    assert not getattr(ADAPTERS["red"], "held_out", False)
