"""Game adapters. Add one module per game (red.py, crystal.py, emerald.py, ...)."""

from missingno_games.base import GameAdapter, Memory
from missingno_games.crystal import CrystalAdapter
from missingno_games.milestones import Milestone
from missingno_games.paths import default_rom, default_state
from missingno_games.red import BlueAdapter, RedAdapter, YellowAdapter

ADAPTERS: dict[str, type[GameAdapter]] = {
    "red": RedAdapter,
    "blue": BlueAdapter,
    "yellow": YellowAdapter,
    # Held out: never trained on, only used to test generalization (docs/research.md).
    "crystal": CrystalAdapter,
}

__all__ = [
    "ADAPTERS",
    "BlueAdapter",
    "CrystalAdapter",
    "GameAdapter",
    "Memory",
    "Milestone",
    "RedAdapter",
    "YellowAdapter",
    "default_rom",
    "default_state",
]
