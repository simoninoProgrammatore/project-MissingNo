"""Game adapters. Add one module per game (red.py, crystal.py, emerald.py, ...)."""

from missingno_games.base import GameAdapter, Memory
from missingno_games.crystal import CrystalAdapter
from missingno_games.firered import FireRedAdapter
from missingno_games.milestones import Milestone, goal_index
from missingno_games.paths import default_rom, default_state
from missingno_games.red import BlueAdapter, RedAdapter, YellowAdapter

ADAPTERS: dict[str, type[GameAdapter]] = {
    "red": RedAdapter,
    "blue": BlueAdapter,
    "yellow": YellowAdapter,
    # Game Boy Advance: needs the mGBA libretro core (scripts/get_mgba_core.py).
    "firered": FireRedAdapter,
    # Held out: never trained on, only used to test generalization (docs/research.md).
    "crystal": CrystalAdapter,
}

__all__ = [
    "ADAPTERS",
    "BlueAdapter",
    "CrystalAdapter",
    "FireRedAdapter",
    "GameAdapter",
    "Memory",
    "Milestone",
    "RedAdapter",
    "YellowAdapter",
    "default_rom",
    "goal_index",
    "default_state",
]
