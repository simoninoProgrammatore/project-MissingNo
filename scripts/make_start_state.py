"""Create the starting save state: play the intro yourself, then close the window.

The intro (choosing names, the opening speech) is setup, not gameplay: we play it
once by hand so every training episode starts in the bedroom. This is not a
demonstration and is never used as training data.

The state is created with the same emulator mode the environment uses for that
game (e.g. Yellow in classic Game Boy mode), otherwise it could not be loaded.

Controls: arrows = D-pad, A = a, S = b, Enter = Start, Backspace = Select
(Game Boy Advance also: Q = L, W = R).

Usage:
    uv run python scripts/make_start_state.py                          # Red
    uv run python scripts/make_start_state.py --game yellow            # Yellow
    uv run python scripts/make_start_state.py --game firered           # FireRed (needs the mGBA core)
    uv run python scripts/make_start_state.py --game yellow --rom roms/my_yellow.gbc
"""

import argparse
from pathlib import Path

from missingno_envs.emulator import make_emulator
from missingno_games import ADAPTERS, default_rom, default_state


def main(game: str, rom: str, out: str) -> None:
    # The same emulator and mode the environment uses for this game.
    emulator = make_emulator(rom, ADAPTERS[game](), window=True, interactive=True)
    emulator.set_speed(1)
    print("Play until you are in control in the bedroom, then CLOSE THE WINDOW to save.")
    while emulator.tick():
        pass
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(emulator.save_state())
    emulator.stop()
    print(f"Saved start state to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--game", default="red", choices=ADAPTERS)
    parser.add_argument("--rom", help="default: roms/pokemon_<game>.gb (or .gbc)")
    parser.add_argument("--out", help="default: states/<game>_start.state")
    args = parser.parse_args()
    main(args.game, args.rom or default_rom(args.game), args.out or default_state(args.game))
