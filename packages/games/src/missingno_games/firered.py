"""Adapter for Pokémon FireRed (Game Boy Advance, Generation 3).

FireRed retells Red on a new console: the same region, the same story, a new
engine and new graphics. That makes it the bridge between Generation 1 and
Generation 3, and the milestones mirror Red's exactly.

All addresses and structures are verified against pret/pokefirered (symbols for
version 1.0 and 1.1 are identical for the addresses used here; English version).

Generation 3 differences, handled here and invisible to the rest of the code:
- The player's data (SaveBlock1/2) is moved around in memory as an anti-cheat
  measure: we follow the pointers gSaveBlock1Ptr and gSaveBlock2Ptr.
- Maps are (group, number) pairs: map_id = group * 256 + number.
- One map tile is one player step (Generation 1 and 2 count 2x2 blocks).
- The party's experience is encrypted: each Pokémon's data is XORed with a key
  and its four parts are shuffled according to its personality value.
- Badges live in the game's flag array, like every other "system" flag. We read
  only the eight badge flags: a badge is manual knowledge, not a story event.
"""

from missingno_core import ProgressSignals

from missingno_games.base import Memory
from missingno_games.milestones import Milestone

# --- addresses (pokefirered.sym) -----------------------------------------------------
SAVE_BLOCK_1_PTR = 0x03005008
SAVE_BLOCK_2_PTR = 0x0300500C
PLAYER_PARTY = 0x02024284  # struct Pokemon[6], 100 bytes each
PLAYER_PARTY_COUNT = 0x02024029
MAIN = 0x030030F0  # struct Main; inBattle is bit 1 of the byte at +0x439
MAP_HEADER = 0x02036DFC  # struct MapHeader; +0 = pointer to the MapLayout (width, height)

# --- SaveBlock1 offsets (include/global.h) ---------------------------------------------
SB1_POS = 0x0000  # s16 x, s16 y
SB1_MAP_GROUP = 0x0004  # struct WarpData location: s8 mapGroup, s8 mapNum
SB1_MAP_NUM = 0x0005
SB1_FLAGS = 0x0EE0
BADGE_FLAGS_BYTE = SB1_FLAGS + 0x820 // 8  # FLAG_BADGE01_GET = 0x820 .. FLAG_BADGE08_GET = 0x827
POCKETS = {  # offset, number of slots (struct ItemSlot: u16 itemId, u16 quantity)
    "items": (0x0310, 42),
    "key_items": (0x03B8, 30),
    "poke_balls": (0x0430, 13),
    "tm_hm": (0x0464, 58),
    "berries": (0x054C, 43),
}
# --- SaveBlock2 offsets ---------------------------------------------------------------
SB2_DEX_OWNED = 0x0018 + 0x10  # struct Pokedex at 0x18: owned[52] at +0x10, seen[52] at +0x44
SB2_DEX_SEEN = 0x0018 + 0x44
DEX_BYTES = 52

# --- struct Pokemon -------------------------------------------------------------------
MON_SIZE = 100
MON_PERSONALITY = 0x00
MON_OT_ID = 0x04
MON_SECURE = 0x20  # 4 encrypted 12-byte substructures
MON_LEVEL = 0x54
# Position of the "growth" substructure (species, item, experience) for personality % 24
# (pokemon.c, GetSubstruct: the first column of SUBSTRUCT_CASE).
GROWTH_INDEX = (0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 2, 3, 1, 1, 2, 3, 2, 3, 1, 1, 2, 3, 2, 3)

# HMs are key items, like in Generation 1 (they can never be bought).
ITEM_HM01, ITEM_HM08 = 339, 346
ITEM_OAKS_PARCEL = 349


def map_id(group: int, number: int) -> int:
    return group * 256 + number


# Maps (pret/pokefirered data/maps/map_groups.json, verified).
VIRIDIAN_FOREST = map_id(1, 0)
PALLET_TOWN = map_id(3, 0)
VIRIDIAN_CITY = map_id(3, 1)
PEWTER_CITY = map_id(3, 2)
ROUTE_1 = map_id(3, 19)
ROUTE_2 = map_id(3, 20)
PLAYERS_HOUSE_1F = map_id(4, 0)
PLAYERS_HOUSE_2F = map_id(4, 1)
OAKS_LAB = map_id(4, 3)
PEWTER_GYM = map_id(6, 2)

# The same milestones as Red, on FireRed's maps. Used ONLY to measure progress.
MILESTONES: tuple[Milestone, ...] = (
    Milestone("M1", "Leave the bedroom", lambda s: s.map_id == PLAYERS_HOUSE_1F),
    Milestone("M2", "Leave the house", lambda s: s.map_id == PALLET_TOWN),
    Milestone("M3", "Enter Oak's lab", lambda s: s.map_id == OAKS_LAB),
    Milestone("M4", "Obtain the first Pokémon", lambda s: len(s.party_levels) > 0),
    Milestone("M5", "Reach Route 1", lambda s: s.map_id == ROUTE_1),
    Milestone("M6", "Reach Viridian City", lambda s: s.map_id == VIRIDIAN_CITY),
    Milestone("M7", "Get Oak's Parcel", lambda s: ITEM_OAKS_PARCEL in s.items),
    Milestone("M8", "Reach Route 2", lambda s: s.map_id == ROUTE_2),
    Milestone("M9", "Enter Viridian Forest", lambda s: s.map_id == VIRIDIAN_FOREST),
    Milestone("M10", "Reach Pewter City", lambda s: s.map_id == PEWTER_CITY),
    Milestone("M11", "Enter Pewter Gym", lambda s: s.map_id == PEWTER_GYM),
    Milestone("M12", "Win the Boulder Badge", lambda s: s.badges >= 1),
)


class FireRedAdapter:
    name = "firered"
    rom_sha1 = "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc"  # English 1.0
    rom_sha1_alternatives = ("dd5945db9b930750cb39d00c84da8571feebf417",)  # English 1.1
    milestones = MILESTONES
    platform = "gba"  # run with mGBA (see missingno_envs/libretro.py)

    def read(self, memory: Memory) -> ProgressSignals:
        sb1 = _u32(memory, SAVE_BLOCK_1_PTR)
        sb2 = _u32(memory, SAVE_BLOCK_2_PTR)
        if not (_is_ram(sb1) and _is_ram(sb2)):
            return ProgressSignals(map_id=0, x=0, y=0)  # title screen: no save data yet

        party_count = min(memory[PLAYER_PARTY_COUNT], 6)
        mons = [PLAYER_PARTY + i * MON_SIZE for i in range(party_count)]
        pockets = {
            name: _pocket(memory, sb1 + offset, slots) for name, (offset, slots) in POCKETS.items()
        }
        hms = frozenset(i for i in pockets["tm_hm"] if ITEM_HM01 <= i <= ITEM_HM08)
        key_items = pockets["key_items"] | hms
        items = frozenset().union(*pockets.values())
        return ProgressSignals(
            map_id=map_id(memory[sb1 + SB1_MAP_GROUP], memory[sb1 + SB1_MAP_NUM]),
            x=_u16(memory, sb1 + SB1_POS),
            y=_u16(memory, sb1 + SB1_POS + 2),
            badges=memory[sb1 + BADGE_FLAGS_BYTE].bit_count(),
            party_levels=tuple(memory[a + MON_LEVEL] for a in mons),
            in_battle=bool((memory[MAIN + 0x439] >> 1) & 1),
            map_area=_map_area(memory),
            items=items,
            key_items=key_items,
            pokedex_owned=_count_bits(memory, sb2 + SB2_DEX_OWNED, DEX_BYTES),
            pokedex_seen=_count_bits(memory, sb2 + SB2_DEX_SEEN, DEX_BYTES),
            party_exp=tuple(experience(memory, a) for a in mons),
        )


def experience(memory: Memory, mon: int) -> int:
    """Experience of the Pokémon at `mon`: decrypt its growth substructure."""
    personality = _u32(memory, mon + MON_PERSONALITY)
    key = personality ^ _u32(memory, mon + MON_OT_ID)
    growth = mon + MON_SECURE + 12 * GROWTH_INDEX[personality % 24]
    return _u32(memory, growth + 4) ^ key  # u16 species, u16 item, u32 experience


def _map_area(memory: Memory) -> int:
    """Width x height of the current map, in player steps (read from the ROM)."""
    layout = _u32(memory, MAP_HEADER)
    if not 0x08000000 <= layout < 0x0A000000:
        return 0
    width, height = _u32(memory, layout), _u32(memory, layout + 4)
    return width * height if width < 1024 and height < 1024 else 0


def _pocket(memory: Memory, start: int, slots: int) -> frozenset[int]:
    ids = (_u16(memory, start + 4 * i) for i in range(slots))
    return frozenset(i for i in ids if i)


def _is_ram(address: int) -> bool:
    return 0x02000000 <= address < 0x02040000


def _u16(memory: Memory, a: int) -> int:
    return memory[a] | memory[a + 1] << 8


def _u32(memory: Memory, a: int) -> int:
    return memory[a] | memory[a + 1] << 8 | memory[a + 2] << 16 | memory[a + 3] << 24


def _count_bits(memory: Memory, start: int, n_bytes: int) -> int:
    return sum(memory[start + i].bit_count() for i in range(n_bytes))
