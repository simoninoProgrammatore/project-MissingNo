"""Where each game's files are expected, so one name (e.g. "yellow") is enough.

ROMs and save states are never committed (see roms/README.md and states/README.md).
"""

from pathlib import Path

GBA_GAMES = {"firered", "leafgreen", "ruby", "sapphire", "emerald"}


def default_rom(game: str, folder: str = "roms") -> str:
    """roms/pokemon_<game>.gb, .gbc or .gba: whichever file you have."""
    for ext in (".gb", ".gbc", ".gba"):
        path = Path(folder) / f"pokemon_{game}{ext}"
        if path.exists():
            return str(path)
    return str(Path(folder) / f"pokemon_{game}{'.gba' if game in GBA_GAMES else '.gb'}")


def default_state(game: str, folder: str = "states") -> str:
    """states/<game>_start.state: the bedroom, made with scripts/make_start_state.py."""
    return str(Path(folder) / f"{game}_start.state")
