"""Adapter for Pokémon Crystal: the held-out test game.

Crystal (Generation 2) is NEVER used for training in this project: it is the game
on which we measure how well an agent trained on other generations generalizes
(docs/research.md). The adapter exists to compute the same generic progress
signals and milestones, so the test is scored exactly like training.

All addresses are verified against pret/pokecrystal's symbol files
(pokecrystal.sym and pokecrystal11.sym: the work RAM is the same in 1.0 and 1.1).

Differences from Generation 1, handled here and invisible to the rest of the code:
- Crystal is a Game Boy Color game with banked work RAM: the addresses in
  0xD000-0xDFFF are read from bank 1, where the game keeps the player's data.
- A map is identified by a (group, number) pair: map_id = group * 256 + number.
- The bag has pockets. The Key Items pocket is the game's own definition of a key
  item; TMs and HMs are counted as items (HMs as key items, like in Gen 1).
- Two sets of badges (Johto and Kanto) and a 251-species Pokédex.
"""

from missingno_core import ProgressSignals

from missingno_games.base import Memory
from missingno_games.milestones import Milestone

MAP_GROUP = 0xDCB5
MAP_NUMBER = 0xDCB6
PLAYER_Y = 0xDCB7
PLAYER_X = 0xDCB8
MAP_HEIGHT = 0xD19E  # in blocks; one block = 2x2 player steps
MAP_WIDTH = 0xD19F
JOHTO_BADGES = 0xD857  # bitfields, one bit per badge
KANTO_BADGES = 0xD858
TMS_HMS = 0xD859  # quantity of each TM (50) then HM (7)
NUM_TMS, NUM_HMS = 50, 7
NUM_ITEMS = 0xD892
ITEMS = 0xD893  # (item id, quantity) pairs
ITEMS_CAPACITY = 20
NUM_KEY_ITEMS = 0xD8BC
KEY_ITEMS = 0xD8BD  # item ids, one byte each
KEY_ITEMS_CAPACITY = 25
NUM_BALLS = 0xD8D7
BALLS = 0xD8D8  # (item id, quantity) pairs
BALLS_CAPACITY = 12
BATTLE_MODE = 0xD22D  # 0 = no battle, 1 = wild, 2 = trainer
PARTY_COUNT = 0xDCD7
PARTY_MON_1 = 0xDCDF
PARTY_MON_SIZE = 0x30
LEVEL_OFFSET = 0x1F  # wPartyMon1Level - wPartyMon1
EXP_OFFSET = 0x08  # wPartyMon1Exp - wPartyMon1, 3 bytes big-endian
POKEDEX_CAUGHT = 0xDE99  # 32-byte bit arrays, one bit per species
POKEDEX_SEEN = 0xDEB9
POKEDEX_BYTES = 32

# TMs and HMs live in their own pocket as quantities, not item ids: give them
# ids that cannot clash with real items (0x100 + index).
TM_HM_ID = 0x100


def map_id(group: int, number: int) -> int:
    return group * 256 + number


# Maps (pret/pokecrystal constants/map_constants.asm, verified).
ROUTE_29 = map_id(24, 3)
NEW_BARK_TOWN = map_id(24, 4)
ELMS_LAB = map_id(24, 5)
PLAYERS_HOUSE_1F = map_id(24, 6)
PLAYERS_HOUSE_2F = map_id(24, 7)
ROUTE_30 = map_id(26, 1)
ROUTE_31 = map_id(26, 2)
CHERRYGROVE_CITY = map_id(26, 3)
MR_POKEMONS_HOUSE = map_id(26, 10)
VIOLET_CITY = map_id(10, 5)
VIOLET_GYM = map_id(10, 7)

# Milestones: from the bedroom to the first badge, in the order the game's
# geography imposes. Used ONLY to measure progress, never as rewards.
MILESTONES: tuple[Milestone, ...] = (
    Milestone("M1", "Leave the bedroom", lambda s: s.map_id == PLAYERS_HOUSE_1F),
    Milestone("M2", "Leave the house", lambda s: s.map_id == NEW_BARK_TOWN),
    Milestone("M3", "Enter Elm's lab", lambda s: s.map_id == ELMS_LAB),
    Milestone("M4", "Obtain the first Pokémon", lambda s: len(s.party_levels) > 0),
    Milestone("M5", "Reach Route 29", lambda s: s.map_id == ROUTE_29),
    Milestone("M6", "Reach Cherrygrove City", lambda s: s.map_id == CHERRYGROVE_CITY),
    Milestone("M7", "Reach Route 30", lambda s: s.map_id == ROUTE_30),
    Milestone("M8", "Enter Mr. Pokémon's house", lambda s: s.map_id == MR_POKEMONS_HOUSE),
    Milestone("M9", "Reach Route 31", lambda s: s.map_id == ROUTE_31),
    Milestone("M10", "Reach Violet City", lambda s: s.map_id == VIOLET_CITY),
    Milestone("M11", "Enter Violet Gym", lambda s: s.map_id == VIOLET_GYM),
    Milestone("M12", "Win the Zephyr Badge", lambda s: s.badges >= 1),
)


class _WramBank1:
    """Reads 0xD000-0xDFFF from work RAM bank 1, whatever bank is mapped right now."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def __getitem__(self, address: int) -> int:
        if 0xD000 <= address < 0xE000:
            return self.memory[1, address]
        return self.memory[address]


class CrystalAdapter:
    name = "crystal"
    rom_sha1 = "f4cd194bdee0d04ca4eac29e09b8e4e9d818c133"  # 1.0 (US/EU)
    rom_sha1_alternatives = ("f2f52230b536214ef7c9924f483392993e226cfb",)  # 1.1: same RAM
    milestones = MILESTONES
    cgb = True  # a Game Boy Color-only game
    # The test game: training on it needs an explicit --allow-held-out (training/ppo.py).
    held_out = True

    def read(self, memory: Memory) -> ProgressSignals:
        m = _WramBank1(memory)
        party_count = min(m[PARTY_COUNT], 6)
        mons = [PARTY_MON_1 + i * PARTY_MON_SIZE for i in range(party_count)]
        items = _pairs(m, ITEMS, NUM_ITEMS, ITEMS_CAPACITY) | _pairs(
            m, BALLS, NUM_BALLS, BALLS_CAPACITY
        )
        n_key = min(m[NUM_KEY_ITEMS], KEY_ITEMS_CAPACITY)
        key_items = frozenset(m[KEY_ITEMS + i] for i in range(n_key))
        tms = frozenset(TM_HM_ID + i for i in range(NUM_TMS + NUM_HMS) if m[TMS_HMS + i])
        hms = frozenset(i for i in tms if i >= TM_HM_ID + NUM_TMS)
        key_items |= hms
        return ProgressSignals(
            map_id=map_id(m[MAP_GROUP], m[MAP_NUMBER]),
            x=m[PLAYER_X],
            y=m[PLAYER_Y],
            badges=m[JOHTO_BADGES].bit_count() + m[KANTO_BADGES].bit_count(),
            party_levels=tuple(m[a + LEVEL_OFFSET] for a in mons),
            in_battle=m[BATTLE_MODE] in (1, 2),
            map_area=4 * m[MAP_HEIGHT] * m[MAP_WIDTH],
            items=items | key_items | tms,
            key_items=key_items,
            pokedex_owned=_count_bits(m, POKEDEX_CAUGHT, POKEDEX_BYTES),
            pokedex_seen=_count_bits(m, POKEDEX_SEEN, POKEDEX_BYTES),
            party_exp=tuple(
                (m[a + EXP_OFFSET] << 16) | (m[a + EXP_OFFSET + 1] << 8) | m[a + EXP_OFFSET + 2]
                for a in mons
            ),
        )


def _pairs(m, start: int, count_address: int, capacity: int) -> frozenset[int]:
    n = min(m[count_address], capacity)
    return frozenset(m[start + 2 * i] for i in range(n))


def _count_bits(m, start: int, n_bytes: int) -> int:
    return sum(m[start + i].bit_count() for i in range(n_bytes))
