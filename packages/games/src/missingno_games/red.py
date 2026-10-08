"""Adapters for the Generation 1 games: Red, Blue (same memory) and Yellow.

Yellow uses the same maps, items and milestones as Red, but every address in its
work RAM is shifted one byte lower (verified on pret/pokeyellow's symbol file).

Memory addresses come from the community disassembly of the game (pret/pokered)
and are the same ones used by earlier Pokémon Red RL projects.

Only generic progress is read: where the player is, map size, badges, party
levels, whether a battle is in progress, items in the bag, Pokédex counts.
No story events, no game-specific flags: those would be "walkthrough"
knowledge, which this project does not use.

All addresses are verified against pret/pokered's symbol file (pokered.sym).
"""

from missingno_core import ProgressSignals

from missingno_games.base import Memory
from missingno_games.milestones import Milestone

MAP_ID = 0xD35E
PLAYER_Y = 0xD361
PLAYER_X = 0xD362
BADGES = 0xD356  # bitfield, one bit per badge
PARTY_COUNT = 0xD163
PARTY_LEVELS = (0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268)
PARTY_MON_SIZE = 0x2C
PARTY_EXP = 0xD179  # 3 bytes, big-endian, first party Pokémon; +0x2C for the next
IS_IN_BATTLE = 0xD057  # 0 = no battle, 1 = wild, 2 = trainer, 0xFF = just lost
MAP_HEIGHT = 0xD368  # in blocks; one block = 2x2 player steps
MAP_WIDTH = 0xD369
NUM_BAG_ITEMS = 0xD31D
BAG_ITEMS = 0xD31E  # (item id, quantity) pairs, up to 20
BAG_CAPACITY = 20
POKEDEX_OWNED = 0xD2F7  # 19-byte bit arrays, one bit per species
POKEDEX_SEEN = 0xD30A
POKEDEX_BYTES = 19

# Items the game itself marks as key items: derived from the game's own
# KeyItemFlags table (pret/pokered data/items/key_items.asm), plus the HMs
# ($C4-$C8), which can never be bought. "Key item" is a concept every
# Generation 1-3 game has, so rewarding it is not walkthrough knowledge.
KEY_ITEMS: frozenset[int] = frozenset(
    (
        0x05,
        0x06,
        0x07,
        0x08,
        0x09,
        0x15,
        0x16,
        0x17,
        0x18,
        0x19,
        0x1A,
        0x1B,
        0x1C,
        0x1F,
        0x29,
        0x2A,
        0x2B,
        0x2C,
        0x2D,
        0x30,
        0x3F,
        0x40,
        0x45,
        0x46,
        0x47,
        0x48,
        0x49,
        0x4A,
        0x4C,
        0x4D,
        0x4E,
    )
) | frozenset(range(0xC4, 0xC9))

# Map IDs (pret/pokered constants/map_constants.asm, verified).
PALLET_TOWN = 0
VIRIDIAN_CITY = 1
PEWTER_CITY = 2
ROUTE_1 = 12
ROUTE_2 = 13
REDS_HOUSE_1F = 37
REDS_HOUSE_2F = 38
OAKS_LAB = 40
VIRIDIAN_MART = 42
VIRIDIAN_FOREST = 51
PEWTER_GYM = 54
CERULEAN_CITY = 3
LAVENDER_TOWN = 4
CELADON_CITY = 6
FUCHSIA_CITY = 7
CINNABAR_ISLAND = 8
INDIGO_PLATEAU = 9
ROUTE_3 = 14
MT_MOON_1F = 59
ROCK_TUNNEL_1F = 82
BILLS_HOUSE = 88
SS_ANNE_1F = 95
VICTORY_ROAD_1F = 108
HALL_OF_FAME = 118
SILPH_CO_1F = 181
ROCKET_HIDEOUT_B1F = 199

# Items (pret/pokered constants/item_constants.asm, verified).
OAKS_PARCEL = 0x46
S_S_TICKET = 0x3F
SILPH_SCOPE = 0x48
POKE_FLUTE = 0x49
HM_CUT, HM_SURF, HM_STRENGTH = 0xC4, 0xC6, 0xC7

# Milestones: from the bedroom to the Hall of Fame, in story order. Used ONLY to
# measure progress, never as rewards. M12 (the first badge) is the goal of the
# current phase (`goal` below); --goal M40 aims at the whole game. Some stretches
# can be done in a different order (e.g. the 6th and 7th badges): milestones that
# count badges say "the N-th badge" for that reason.
MILESTONES: tuple[Milestone, ...] = (
    Milestone("M1", "Leave the bedroom", lambda s: s.map_id == REDS_HOUSE_1F),
    Milestone("M2", "Leave the house", lambda s: s.map_id == PALLET_TOWN),
    Milestone("M3", "Enter Oak's lab", lambda s: s.map_id == OAKS_LAB),
    Milestone("M4", "Obtain the first Pokémon", lambda s: len(s.party_levels) > 0),
    Milestone("M5", "Reach Route 1", lambda s: s.map_id == ROUTE_1),
    Milestone("M6", "Reach Viridian City", lambda s: s.map_id == VIRIDIAN_CITY),
    Milestone("M7", "Get Oak's Parcel", lambda s: OAKS_PARCEL in s.items),
    # The old man blocks the way north until the parcel has been delivered to Oak.
    Milestone("M8", "Reach Route 2", lambda s: s.map_id == ROUTE_2),
    Milestone("M9", "Enter Viridian Forest", lambda s: s.map_id == VIRIDIAN_FOREST),
    Milestone("M10", "Reach Pewter City", lambda s: s.map_id == PEWTER_CITY),
    Milestone("M11", "Enter Pewter Gym", lambda s: s.map_id == PEWTER_GYM),
    Milestone("M12", "Win the Boulder Badge", lambda s: s.badges >= 1),
    Milestone("M13", "Reach Route 3", lambda s: s.map_id == ROUTE_3),
    Milestone("M14", "Enter Mt. Moon", lambda s: s.map_id == MT_MOON_1F),
    Milestone("M15", "Reach Cerulean City", lambda s: s.map_id == CERULEAN_CITY),
    Milestone("M16", "Win the 2nd badge (Cascade)", lambda s: s.badges >= 2),
    Milestone("M17", "Reach Bill's house", lambda s: s.map_id == BILLS_HOUSE),
    Milestone("M18", "Get the S.S. Ticket", lambda s: S_S_TICKET in s.items),
    Milestone("M19", "Board the S.S. Anne", lambda s: s.map_id == SS_ANNE_1F),
    Milestone("M20", "Get HM01 (Cut)", lambda s: HM_CUT in s.items),
    Milestone("M21", "Win the 3rd badge (Thunder)", lambda s: s.badges >= 3),
    Milestone("M22", "Enter Rock Tunnel", lambda s: s.map_id == ROCK_TUNNEL_1F),
    Milestone("M23", "Reach Lavender Town", lambda s: s.map_id == LAVENDER_TOWN),
    Milestone("M24", "Reach Celadon City", lambda s: s.map_id == CELADON_CITY),
    Milestone("M25", "Win the 4th badge (Rainbow)", lambda s: s.badges >= 4),
    Milestone("M26", "Enter the Rocket Hideout", lambda s: s.map_id == ROCKET_HIDEOUT_B1F),
    Milestone("M27", "Get the Silph Scope", lambda s: SILPH_SCOPE in s.items),
    Milestone("M28", "Get the Poké Flute", lambda s: POKE_FLUTE in s.items),
    Milestone("M29", "Reach Fuchsia City", lambda s: s.map_id == FUCHSIA_CITY),
    Milestone("M30", "Win the 5th badge", lambda s: s.badges >= 5),
    Milestone("M31", "Get HM03 (Surf)", lambda s: HM_SURF in s.items),
    Milestone("M32", "Get HM04 (Strength)", lambda s: HM_STRENGTH in s.items),
    Milestone("M33", "Enter Silph Co.", lambda s: s.map_id == SILPH_CO_1F),
    Milestone("M34", "Win the 6th badge", lambda s: s.badges >= 6),
    Milestone("M35", "Reach Cinnabar Island", lambda s: s.map_id == CINNABAR_ISLAND),
    Milestone("M36", "Win the 7th badge", lambda s: s.badges >= 7),
    Milestone("M37", "Win the 8th badge (Earth)", lambda s: s.badges >= 8),
    Milestone("M38", "Enter Victory Road", lambda s: s.map_id == VICTORY_ROAD_1F),
    Milestone("M39", "Reach Indigo Plateau", lambda s: s.map_id == INDIGO_PLATEAU),
    Milestone("M40", "Enter the Hall of Fame (beat the game)", lambda s: s.map_id == HALL_OF_FAME),
)


class _Shifted:
    """Memory view with every address moved by `offset` (Yellow: -1)."""

    def __init__(self, memory: Memory, offset: int) -> None:
        self.memory, self.offset = memory, offset

    def __getitem__(self, address: int) -> int:
        return self.memory[address + self.offset]


class RedAdapter:
    name = "red"
    rom_sha1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"
    milestones = MILESTONES
    goal = "M12"  # the goal of the current phase: the first badge (--goal to change it)
    offset = 0  # address shift relative to Red
    cgb: bool | None = None  # None = let the emulator decide; False = force the classic Game Boy

    def read(self, memory: Memory) -> ProgressSignals:
        if self.offset:
            memory = _Shifted(memory, self.offset)
        party_count = min(memory[PARTY_COUNT], 6)
        levels = tuple(memory[address] for address in PARTY_LEVELS[:party_count])
        n_items = min(memory[NUM_BAG_ITEMS], BAG_CAPACITY)
        items = frozenset(memory[BAG_ITEMS + 2 * i] for i in range(n_items))
        return ProgressSignals(
            map_id=memory[MAP_ID],
            x=memory[PLAYER_X],
            y=memory[PLAYER_Y],
            badges=memory[BADGES].bit_count(),
            party_levels=levels,
            in_battle=memory[IS_IN_BATTLE] in (1, 2),
            map_area=4 * memory[MAP_HEIGHT] * memory[MAP_WIDTH],
            items=items,
            key_items=items & KEY_ITEMS,
            pokedex_owned=_count_bits(memory, POKEDEX_OWNED, POKEDEX_BYTES),
            pokedex_seen=_count_bits(memory, POKEDEX_SEEN, POKEDEX_BYTES),
            party_exp=tuple(_read_exp(memory, i) for i in range(party_count)),
        )


class BlueAdapter(RedAdapter):
    """Blue: the same memory, maps and milestones as Red; a few different Pokémon."""

    name = "blue"
    rom_sha1 = "d7037c83e1ae5b39bde3c30787637ba1d4c48ce2"


class YellowAdapter(RedAdapter):
    """Yellow: same maps, items and milestones as Red, memory shifted by one byte.

    Yellow is a Game Boy Color game that also runs on the classic Game Boy. We
    force the classic mode, so that it looks like Red (same gray shades): when
    testing a model trained on Red, the differences are the game's, not the
    colours. Start states must be created in the same mode (make_start_state.py
    --game yellow does it).
    """

    name = "yellow"
    rom_sha1 = "cc7d03262ebfaf2f06772c1a480c7d9d5f4a38e1"
    offset = -1
    cgb = False


def _read_exp(memory: Memory, slot: int) -> int:
    a = PARTY_EXP + slot * PARTY_MON_SIZE
    return (memory[a] << 16) | (memory[a + 1] << 8) | memory[a + 2]


def _count_bits(memory: Memory, start: int, n_bytes: int) -> int:
    return sum(memory[start + i].bit_count() for i in range(n_bytes))
