"""Game adapters. Add one module per game (red.py, crystal.py, emerald.py, ...)."""

from missingno_games.base import GameAdapter, Memory
from missingno_games.milestones import Milestone
from missingno_games.red import BlueAdapter, RedAdapter, YellowAdapter

ADAPTERS: dict[str, type[GameAdapter]] = {
    "red": RedAdapter,
    "blue": BlueAdapter,
    "yellow": YellowAdapter,
}

__all__ = [
    "ADAPTERS",
    "BlueAdapter",
    "GameAdapter",
    "Memory",
    "Milestone",
    "RedAdapter",
    "YellowAdapter",
]
