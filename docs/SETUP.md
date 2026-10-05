# Setup

Technical instructions for running the code. The project is in an early study phase, so the structure will change.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package and project manager; it installs Python 3.12 if missing)
- [Node.js](https://nodejs.org/) (to run a local Pokémon Showdown server)
- Git

## Installation

```bash
# Python dependencies
uv sync

# Automatic lint/format checks on every commit
uv run pre-commit install
```

## Local Pokémon Showdown server

Run this in a separate terminal and leave it open:

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

## First commands

```bash
uv run pytest                                            # tests
uv run ruff check . && uv run ruff format .              # lint and formatting
uv run python scripts/play_baseline.py --n-battles 100   # rule-based player vs random player
```

Reference result (gen1randombattle, 30 battles): **rule-based 30/30 against random**. This only confirms the pipeline works: beating a random player is easy.

## Current structure

```
project-MissingNo/
├── packages/
│   ├── core/            # shared types: game state, goals, actions, skills
│   ├── envs/            # game environments: Showdown (now), Game Boy emulator (later)
│   └── skills/
│       └── battle/      # battle logic: rule-based baseline, then trained models
├── scripts/             # runnable experiments
├── tests/
├── benchmark/
│   └── checkpoints/     # the "hard spots" collection
├── configs/             # experiment configurations
├── traces/              # decision logs (not versioned)
└── docs/                # plan, setup, assets
```

One rule worth keeping whatever the final design: **only game-specific adapters should know about a particular game's memory layout**. Everything else should depend on shared, game-independent types. That separation is what could later allow moving to a different game.

## Adding a package

```bash
uv init --lib packages/skills/navigation --name missingno-navigation
```

Then add it to `members` and `[tool.uv.sources]` in the root `pyproject.toml`.

## ROMs

ROMs are never committed (they are excluded in `.gitignore`). Use a copy legally obtained from your own cartridge.
