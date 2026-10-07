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

# 2b. Or watch one of the games live while the agent learns (stop with Ctrl+C, not by closing the window)
uv run python training/ppo.py --total-steps 1_000_000 --run-name smoke --show

# 3. Watch the trained agent play
uv run python scripts/watch.py --checkpoint runs/smoke/checkpoints/latest.pt

# ...or save an episode as a GIF, without a window
uv run python scripts/watch.py --checkpoint runs/smoke/checkpoints/latest.pt --no-window --steps 3000 --gif best.gif

# 4. A real run: overnight, one seed per night
uv run python training/ppo.py --total-steps 100_000_000 --seed 1 --run-name ppo_s1
```

**Reward versions** (design in `docs/rewards.md`): `--reward-version v2` is the default; `--reward-version v1` reproduces the original Phase 1 baseline; `--reward-version v2.1` adds exploration that wears out with use; `--reward-version v2.2` also rewards experience instead of levels, so that every battle won pays (with v2 and v2.1 the agent learned to always flee).

To try a new reward **starting from an already trained model**, resume it into a new run, so the original stays untouched:

```bash
uv run python training/ppo.py ... --reward-version v2.1 --run-name v21_from_v2 --resume runs/v2_s1/checkpoints/latest.pt
``` When resuming a run, use the same version it was trained with (the script warns you if not).

**State archive** (`--archive-prob 0.5`): a simple form of Go-Explore. Whenever the agent reaches a new region of a map, or makes progress, the emulator state is saved; then half of the episodes start from one of those states instead of the bedroom, favouring the less explored ones. The agent practices where it gets stuck instead of replaying the beginning. All states come from the agent's own play. Episodes started from the archive are logged under `archive/*` and excluded from the milestone curves, which always measure progress from the bedroom. The archive lives in memory and is rebuilt after a resume.

**Archive by learning progress** (`--archive-progress-weight`, default 1): archived states also remember how the episodes started from them went. States where results are changing, i.e. where the agent is currently learning, are chosen more often; too easy or too hard ones less (inspired by Prioritized Level Replay).

**Backward curriculum** (`--curriculum-prob 0.3`): when an episode reaches a map that no episode ever reached before, its path (one saved state every 64 steps) becomes a *demo*. Some episodes then start just before that success; when the agent reaches the goal in at least 4 of the last 8 tries, the start moves one step back, until the whole path is learned. The demos are the agent's own successes: no human data. Logged under `curriculum/*`, excluded from the milestone curves.

**Self-imitation learning** (`--sil-coef 1.0`): the 5% of actions of each batch that turned out much better than expected are kept in a buffer (~170 MB of RAM for 10,000), and replayed at every learning phase, but only while they are still better than what the agent now expects. A rare success is practiced instead of forgotten. Logged under `sil/*`.

**Final goal and replay** (`--stop-at-goal`): training stops as soon as an episode **from the start state** reaches the last milestone (for Red: the Boulder Badge). Its replay, the start state plus every button pressed, is saved in `runs/<run-name>/replays/`, and the model in `checkpoints/winner.pt`. The emulator is deterministic, so the replay reproduces the exact game:

```bash
uv run python scripts/replay.py runs/<run>/replays/goal_env0_ep12.npz                  # window, normal speed
uv run python scripts/replay.py runs/<run>/replays/goal_env0_ep12.npz --speed 4        # window, 4x
uv run python scripts/replay.py runs/<run>/replays/goal_env0_ep12.npz --video badge.mp4 --video-speed 4
```

Replays contain the start state, i.e. game data: keep them private, like ROMs.

**Stagnation** (`--stagnation-steps`): end an episode after this many steps without progress; `0` = never, `-1` = the reward version's default.

**Highlights** (`--record-every 8`): GIFs of the episodes that went furthest, saved while training runs in `runs/<run-name>/gifs/` (open them with a browser). Every record-breaking episode (more milestones, then more maps, then more tiles) is saved immediately as `record_*.gif` and kept; every `--gif-every-steps` (default 10,000) the best episode of that window is saved as `best_*.gif`, keeping the last `--gif-keep` (default 20). Only episodes from the start state are recorded. Memory: about 60 MB per game with `--record-every 8` and 20,000-step episodes; GIFs are encoded in a separate process.

All of them together, starting from an already trained model:

```bash
uv run python training/ppo.py --total-steps 20_000_000 --seed 1 --num-envs 4 --reward-version v2.1 --episode-steps 20000 --stagnation-steps 0 --archive-prob 0.3 --curriculum-prob 0.3 --sil-coef 1.0 --record-every 8 --run-name explore_all --resume runs/<run>/checkpoints/latest.pt
```

Every hyperparameter is a command-line option (`--num-envs`, `--learning-rate`, `--ent-coef`, ...): see `training/ppo.py` or run it with `--help`. Each run saves its configuration, TensorBoard logs and checkpoints in `runs/<run-name>/` (ignored by Git).

What to look at in TensorBoard:

- `milestones/M*_rate`: fraction of recent episodes reaching each milestone. **This is the main result.**
- `episode/tiles_visited`, `episode/maps_visited`: is the agent exploring more over time?
- `episode/return`: total reward per episode.
- `reward/*`: the return of each reward component. If one dominates, watch the agent before trusting the curves: that is how reward hacking is caught.
- `episode/stagnated`: fraction of episodes ended early for lack of progress (v2).
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
