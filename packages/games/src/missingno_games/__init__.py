"""Game adapters. Add one module per game (red.py, crystal.py, emerald.py, ...)."""

from missingno_games.base import GameAdapter, Memory
from missingno_games.milestones import Milestone
from missingno_games.red import RedAdapter

ADAPTERS: dict[str, type[GameAdapter]] = {
    "red": RedAdapter,
}

__all__ = ["ADAPTERS", "GameAdapter", "Memory", "Milestone", "RedAdapter"]
