# Setup

Technical instructions for running the code. The project is in an early phase, so the structure will change.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package and project manager; it installs Python 3.12 if missing)
- Git
- A **legally obtained** Pokémon Red ROM, dumped from your own cartridge

## Installation

```bash
uv sync                          # Python dependencies (PyBoy, Gymnasium, ...)
uv run pre-commit install        # automatic lint/format checks on every commit
uv run pytest                    # everything should pass
```

## First run

```bash
# 1. Put your ROM in roms/ (see roms/README.md for the expected name and checksum)

# 2. Create the starting state: play the intro yourself, then close the window
uv run python scripts/make_start_state.py

# 3. Watch a random agent play, with rewards printed in the terminal
uv run python scripts/watch.py --steps 2000

# 4. Measure how fast the environment runs on your machine
uv run python scripts/benchmark_env.py --steps 5000
```

PyBoy controls in the window: arrows = D-pad, `A` = A, `S` = B, `Enter` = Start, `Backspace` = Select.

Reference speed: about **550 agent steps/s per CPU core** (24 frames per step), measured with PyBoy's bundled test ROM. Pokémon Red may differ: run the benchmark to get your number.

## How the environment works

- **What the agent sees:** only the screen, grayscale, downscaled to 80×72, last 3 frames stacked.
- **What the agent does:** one of 7 buttons (down, left, right, up, A, B, Start), once every 24 frames.
- **Rewards:** generic and the same for every game: first visit to a tile, first visit to a map, new badges, new party levels above the best seen.
- **Game adapters** (`packages/games/`) read the game memory to compute those rewards. The agent never sees this information.

## Structure

```
project-MissingNo/
├── packages/
│   ├── core/            # shared types, including ProgressSignals
│   ├── games/           # one adapter per game (red.py for now)
│   └── envs/            # the Gymnasium environment (PokemonEnv)
├── scripts/             # make_start_state, watch, benchmark_env
├── tests/
├── roms/                # your ROMs (ignored by Git)
├── states/              # save states (ignored by Git)
├── archive/             # earlier experiments, not maintained
├── configs/
├── traces/
└── docs/
```

## Adding a game

1. Write an adapter in `packages/games/src/missingno_games/<game>.py` that reads the same `ProgressSignals` from that game's memory.
2. Register it in `ADAPTERS` in `packages/games/src/missingno_games/__init__.py`.
3. Add the expected ROM name and SHA-1 to `roms/README.md`.

Nothing else should change: if it does, that is a sign the environment is not game-agnostic enough.

## ROMs and save states

They are never committed (see `.gitignore`). ROMs cannot be shared, and save states contain game data.
