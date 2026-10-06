"""Adapter for Pokémon Red (international version).

Memory addresses come from the community disassembly of the game (pret/pokered)
and are the same ones used by earlier Pokémon Red RL projects.

Only generic progress is read: where the player is, badges, party levels,
whether a battle is in progress. No story events, no game-specific flags:
those would be "walkthrough" knowledge, which this project does not use.
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
IS_IN_BATTLE = 0xD057  # 0 = no battle, 1 = wild, 2 = trainer, 0xFF = just lost

# Map IDs (pret/pokered constants). 37 and 38 verified in our first run.
PALLET_TOWN = 0
VIRIDIAN_CITY = 1
ROUTE_1 = 12
REDS_HOUSE_1F = 37
REDS_HOUSE_2F = 38
OAKS_LAB = 40

# Phase 1 milestones: from the bedroom to Viridian City (docs/research.md, 6.1).
MILESTONES: tuple[Milestone, ...] = (
    Milestone("M1", "Leave the bedroom", lambda s: s.map_id == REDS_HOUSE_1F),
    Milestone("M2", "Leave the house", lambda s: s.map_id == PALLET_TOWN),
    Milestone("M3", "Enter Oak's lab", lambda s: s.map_id == OAKS_LAB),
    Milestone("M4", "Obtain the first Pokémon", lambda s: len(s.party_levels) > 0),
    Milestone("M5", "Reach Route 1", lambda s: s.map_id == ROUTE_1),
    Milestone("M6", "Reach Viridian City", lambda s: s.map_id == VIRIDIAN_CITY),
)


class RedAdapter:
    name = "red"
    rom_sha1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"
    milestones = MILESTONES

    def read(self, memory: Memory) -> ProgressSignals:
        party_count = min(memory[PARTY_COUNT], 6)
        levels = tuple(memory[address] for address in PARTY_LEVELS[:party_count])
        return ProgressSignals(
            map_id=memory[MAP_ID],
            x=memory[PLAYER_X],
            y=memory[PLAYER_Y],
            badges=memory[BADGES].bit_count(),
            party_levels=levels,
            in_battle=memory[IS_IN_BATTLE] in (1, 2),
        )
