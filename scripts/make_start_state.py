"""Create the starting save state: play the intro yourself, then close the window.

The intro (choosing names, the opening speech) is setup, not gameplay: we play it
once by hand so every training episode starts in the bedroom. This is not a
demonstration and is never used as training data.

The state is created with the same emulator mode the environment uses for that
game (e.g. Yellow in classic Game Boy mode), otherwise it could not be loaded.

Controls (PyBoy defaults): arrows = D-pad, A = a, S = b, Enter = Start,
Backspace = Select.

Usage:
    uv run python scripts/make_start_state.py                          # Red
    uv run python scripts/make_start_state.py --game yellow            # Yellow
    uv run python scripts/make_start_state.py --game yellow --rom roms/my_yellow.gbc
"""

import argparse
from pathlib import Path

from missingno_games import ADAPTERS
from pyboy import PyBoy


def default_rom(game: str) -> str:
    """roms/pokemon_<game>.gb, or .gbc if that is the file you have."""
    for ext in (".gb", ".gbc"):
        path = Path(f"roms/pokemon_{game}{ext}")
        if path.exists():
            return str(path)
    return f"roms/pokemon_{game}.gb"


def main(game: str, rom: str, out: str) -> None:
    cgb = getattr(ADAPTERS[game], "cgb", None)
    pyboy = PyBoy(rom, window="SDL2", sound_emulated=False, cgb=cgb)
    pyboy.set_emulation_speed(1)
    print("Play until you are in control in the bedroom, then CLOSE THE WINDOW to save.")
    while pyboy.tick():
        pass
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pyboy.save_state(f)
    pyboy.stop(save=False)
    print(f"Saved start state to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--game", default="red", choices=ADAPTERS)
    parser.add_argument("--rom", help="default: roms/pokemon_<game>.gb (or .gbc)")
    parser.add_argument("--out", help="default: states/<game>_start.state")
    args = parser.parse_args()
    main(
        args.game, args.rom or default_rom(args.game), args.out or f"states/{args.game}_start.state"
    )
