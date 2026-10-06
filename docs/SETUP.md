# Setup

Technical instructions for running the code. The project is in an early phase, so the structure will change.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package and project manager; it installs Python if missing). Python 3.11 or newer.
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

## Training (Phase 1)

```bash
# 1. Smoke test: ~10 minutes, just to check that everything runs and checkpoints are saved
uv run python training/ppo.py --total-steps 1_000_000 --run-name smoke

# 2. Watch the curves (open the link it prints, usually http://localhost:6006)
uv run tensorboard --logdir runs

# 3. Watch the trained agent play
uv run python scripts/watch.py --checkpoint runs/smoke/checkpoints/latest.pt

# 4. A real run: overnight, one seed per night
uv run python training/ppo.py --total-steps 100_000_000 --seed 1 --run-name ppo_s1
```

Every hyperparameter is a command-line option (`--num-envs`, `--learning-rate`, `--ent-coef`, ...): see `training/ppo.py` or run it with `--help`. Each run saves its configuration, TensorBoard logs and checkpoints in `runs/<run-name>/` (ignored by Git).

What to look at in TensorBoard:

- `milestones/M*_rate`: fraction of recent episodes reaching each milestone. **This is the main result.**
- `episode/tiles_visited`, `episode/maps_visited`: is the agent exploring more over time?
- `episode/return`: total reward per episode.
- `losses/entropy`: how random the policy still is. If it collapses early, the agent stops exploring.
- `charts/SPS`: steps per second.

PyTorch from PyPI runs on the CPU on Windows, which is fine for this small network. A GPU is optional.

### Long runs

- **Resume** a run from its last checkpoint: `--resume runs/<run-name>/checkpoints/latest.pt` (keep the same `--run-name`).
- **Stop cleanly** after a number of hours, saving everything: `--time-limit-hours 11`.
- `Ctrl+C` also saves the last complete update.

### Kaggle

`notebooks/kaggle_train.ipynb` runs training on Kaggle (GPU for learning, CPU cores for the emulators). Upload it to Kaggle, then follow the instructions in its first cell: enable GPU and Internet, add a **private** dataset with `pokemon_red.gb` and `red_start.state`, and use *Save Version → Save & Run All* for long runs. To continue a run in a new session, add the previous version's output as input and set `RESUME = True`.

On a laptop the emulators may not scale well across cores (run `scripts/diagnose_speed.py` to check): in that case use your PC for development and short tests, and Kaggle for long runs.

## How the environment works

- **What the agent sees:** only the screen, grayscale, downscaled to 80×72, last 3 frames stacked.
- **What the agent does:** one of 7 buttons (down, left, right, up, A, B, Start), once every 24 frames.
- **Rewards:** generic and the same for every game: first visit to a tile, first visit to a map, new badges, new party levels above the best seen.
- **Game adapters** (`packages/games/`) read the game memory to compute those rewards. The agent never sees this information.
- **Milestones** (e.g. "reach Viridian City") are defined per game and used only to measure progress, never as rewards.

## Structure

```
project-MissingNo/
├── packages/
│   ├── core/            # shared types, including ProgressSignals
│   ├── games/           # one adapter per game (red.py for now)
│   ├── envs/            # the Gymnasium environment (PokemonEnv)
│   └── agents/          # neural networks
├── training/            # training scripts (ppo.py)
├── notebooks/           # Kaggle notebook
├── scripts/             # make_start_state, watch, benchmark_env
├── runs/                # training logs and checkpoints (ignored by Git)
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
