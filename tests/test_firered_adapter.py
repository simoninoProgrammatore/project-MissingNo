"""FireRed adapter: pointers to the moving save blocks, encrypted experience, pockets."""

import struct

from missingno_games import ADAPTERS, FireRedAdapter
from missingno_games import firered as F

SB1, SB2 = 0x0202552C + 0x10, 0x02024588 + 0x20  # the game moves the blocks around


class GbaMemory(dict):
    def __missing__(self, address):
        return 0

    def put(self, address, data: bytes):
        for i, b in enumerate(data):
            self[address + i] = b


def memory_with_save_blocks():
    m = GbaMemory()
    m.put(F.SAVE_BLOCK_1_PTR, struct.pack("<I", SB1))
    m.put(F.SAVE_BLOCK_2_PTR, struct.pack("<I", SB2))
    return m


def add_mon(m, slot, level, exp, personality, ot_id):
    mon = F.PLAYER_PARTY + slot * F.MON_SIZE
    m.put(mon + F.MON_PERSONALITY, struct.pack("<I", personality))
    m.put(mon + F.MON_OT_ID, struct.pack("<I", ot_id))
    growth = mon + F.MON_SECURE + 12 * F.GROWTH_INDEX[personality % 24]
    m.put(growth + 4, struct.pack("<I", exp ^ personality ^ ot_id))  # encrypted
    m[mon + F.MON_LEVEL] = level


def test_title_screen_has_no_save_data():
    s = FireRedAdapter().read(GbaMemory())
    assert s.map_id == 0 and s.party_levels == ()


def test_position_map_badges_and_battle():
    m = memory_with_save_blocks()
    m.put(SB1 + F.SB1_POS, struct.pack("<hh", 7, 12))
    m[SB1 + F.SB1_MAP_GROUP], m[SB1 + F.SB1_MAP_NUM] = 3, 19  # Route 1
    m[SB1 + F.BADGE_FLAGS_BYTE] = 0b101
    m[F.MAIN + 0x439] = 0b10  # inBattle
    s = FireRedAdapter().read(m)
    assert s.cell == (F.ROUTE_1, 7, 12)
    assert s.badges == 2 and s.in_battle


def test_encrypted_experience_is_decrypted_for_every_substructure_order():
    m = memory_with_save_blocks()
    personalities = [0x12345678 + k for k in range(6)]
    m[F.PLAYER_PARTY_COUNT] = 6
    for slot, p in enumerate(personalities):
        add_mon(m, slot, level=5 + slot, exp=135 + 100 * slot, personality=p, ot_id=0xBEEF1234)
    s = FireRedAdapter().read(m)
    assert s.party_levels == (5, 6, 7, 8, 9, 10)
    assert s.party_exp == tuple(135 + 100 * k for k in range(6))
    for p in range(24):  # every order of the four substructures
        add_mon(m, 0, level=5, exp=4242, personality=p, ot_id=77)
        assert F.experience(m, F.PLAYER_PARTY) == 4242


def test_pockets_key_items_hms_and_parcel_milestone():
    m = memory_with_save_blocks()
    items, keys = F.POCKETS["items"][0], F.POCKETS["key_items"][0]
    m.put(SB1 + items, struct.pack("<HH", 13, 3))  # a potion-like item
    m.put(SB1 + keys, struct.pack("<HH", F.ITEM_OAKS_PARCEL, 1))
    m.put(SB1 + F.POCKETS["tm_hm"][0], struct.pack("<HHHH", 289, 1, F.ITEM_HM01, 1))
    s = FireRedAdapter().read(m)
    assert {13, F.ITEM_OAKS_PARCEL, 289, F.ITEM_HM01} <= s.items
    assert s.key_items == {F.ITEM_OAKS_PARCEL, F.ITEM_HM01}
    milestones = {ms.id: ms for ms in FireRedAdapter.milestones}
    assert milestones["M7"].reached(s)


def test_pokedex_and_map_size_from_the_rom():
    m = memory_with_save_blocks()
    m[SB2 + F.SB2_DEX_OWNED] = 0b1
    m[SB2 + F.SB2_DEX_SEEN] = 0b11
    m[SB2 + F.SB2_DEX_SEEN + 51] = 0b1
    layout = 0x08345678
    m.put(F.MAP_HEADER, struct.pack("<I", layout))
    m.put(layout, struct.pack("<ii", 24, 20))
    s = FireRedAdapter().read(m)
    assert (s.pokedex_owned, s.pokedex_seen) == (1, 3)
    assert s.map_area == 24 * 20


def test_registered_with_red_milestones():
    assert ADAPTERS["firered"].platform == "gba"
    assert [m.name for m in FireRedAdapter.milestones] == [
        m.name for m in ADAPTERS["red"].milestones[:12]
    ]
