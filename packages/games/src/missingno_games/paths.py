"""Where each game's files are expected, so one name (e.g. "yellow") is enough.

ROMs and save states are never committed (see roms/README.md and states/README.md).
"""

from pathlib import Path


def default_rom(game: str, folder: str = "roms") -> str:
    """roms/pokemon_<game>.gb, or .gbc if that is the file you have."""
    for ext in (".gb", ".gbc"):
        path = Path(folder) / f"pokemon_{game}{ext}"
        if path.exists():
            return str(path)
    return str(Path(folder) / f"pokemon_{game}.gb")


def default_state(game: str, folder: str = "states") -> str:
    """states/<game>_start.state: the bedroom, made with scripts/make_start_state.py."""
    return str(Path(folder) / f"{game}_start.state")
