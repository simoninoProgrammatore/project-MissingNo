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

# 3. Watch the trained agent play. The terminal shows a live panel: reward of each
#    component (total, times it paid, share), last rewards, milestones, team, Pokédex.
#    --speed 1 is real time; --no-live prints plain lines instead.
uv run python scripts/watch.py --checkpoint runs/smoke/checkpoints/latest.pt --steps 40000 --speed 2

# ...or save an episode as a GIF, without a window
uv run python scripts/watch.py --checkpoint runs/smoke/checkpoints/latest.pt --no-window --steps 3000 --gif best.gif

# 4. A real run: overnight, one seed per night
uv run python training/ppo.py --total-steps 100_000_000 --seed 1 --run-name ppo_s1
```

**Reward versions** (design in `docs/rewards.md`): `--reward-version v2.2` is the default; `--reward-version v1` reproduces the original Phase 1 baseline; `v2` weights maps by size and adds items and the Pokédex; `v2.1` adds exploration that wears out with use; `v2.2` rewards experience instead of levels, so that every battle won pays (with v2 and v2.1 the agent learned to always flee), and ends episodes stuck in a loop (see *Stagnation* below).

To try a new reward **starting from an already trained model**, resume it into a new run, so the original stays untouched:

```bash
uv run python training/ppo.py ... --reward-version v2.1 --run-name v21_from_v2 --resume runs/v2_s1/checkpoints/latest.pt
``` When resuming a run, use the same version it was trained with (the script warns you if not).

**State archive** (`--archive-prob 0.5`): a simple form of Go-Explore. Whenever the agent reaches a new region of a map, or makes progress, the emulator state is saved; then half of the episodes start from one of those states instead of the bedroom, favouring the less explored ones. The agent practices where it gets stuck instead of replaying the beginning. All states come from the agent's own play. Episodes started from the archive are logged under `archive/*` and excluded from the milestone curves, which always measure progress from the bedroom. The archive is **shared** by all the parallel games of a title and saved next to the checkpoints (`checkpoints/exploration_<game>.pkl`), so it continues after a resume (`--archive-max-cells`, default 5000; `--no-shared-exploration` gives each parallel game its own, lost at every resume, as in the first runs).

**Archive by learning progress** (`--archive-progress-weight`, default 1): archived states also remember how the episodes started from them went. States where results are changing, i.e. where the agent is currently learning, are chosen more often; too easy or too hard ones less (inspired by Prioritized Level Replay).

**Backward curriculum** (`--curriculum-prob 0.3`): when an episode reaches a map that no episode ever reached before, its path (one saved state every 64 steps) becomes a *demo*. Some episodes then start just before that success; when the agent reaches the goal in at least 4 of the last 8 tries, the start moves one step back, or a tenth of the path at once if it never failed, until the whole path is learned. Newer demos (the frontier) are chosen more often. In long episodes the saved states are thinned out (one every 128 steps, then 256, ...), so a demo always covers the whole path. Shared and saved like the archive. The demos are the agent's own successes: no human data. Logged under `curriculum/*`, excluded from the milestone curves.

**Self-imitation learning** (`--sil-coef 1.0`): the 5% of actions of each batch that turned out much better than expected are kept in a buffer (~170 MB of RAM for 10,000), and replayed at every learning phase, but only while they are still better than what the agent now expects. A rare success is practiced instead of forgotten. Logged under `sil/*`.

**Final goal and replay** (`--stop-at-goal`): training stops as soon as an episode reaches the goal milestone (`--goal`, default the game's goal for the current phase: for Red `M12`, the Boulder Badge; `--goal M40` is the whole game, up to the Hall of Fame). Its replay, the start state plus every button pressed **from the bedroom**, is saved in `runs/<run-name>/replays/`, and the model in `checkpoints/winner.pt`. This holds even when the episode started from an archived state: every saved state remembers the buttons that led to it (its *lineage*, see `packages/envs/src/missingno_envs/exploration.py`), and the replay plays them first, so it is always one continuous game played by the agent. With several games (see below), training stops when every game has been won at least once. The emulator is deterministic, so the replay reproduces the exact game (the ROM is chosen from the game saved in the replay):

```bash
uv run python scripts/replay.py runs/<run>/replays/goal_red_env0_ep12.npz                  # window, normal speed
uv run python scripts/replay.py runs/<run>/replays/goal_red_env0_ep12.npz --speed 4        # window, 4x
uv run python scripts/replay.py runs/<run>/replays/goal_red_env0_ep12.npz --video badge.mp4 --video-speed 4
```

Replays contain the start state, i.e. game data: keep them private, like ROMs.

**Stagnation** (`--stagnation-steps`): end an episode after this many steps without progress; `0` = never, `-1` = the reward version's default (v2.2: 5,000).

**Battle loops** (`--battle-stagnation-steps`): end an episode after this many steps of one battle without progress; `0` = never, `-1` = the reward version's default (v2.2: 1,000; a normal battle takes a few dozen steps). This catches the *flee loop*: an agent that learned to flee wild battles keeps choosing RUN in a trainer battle, where fleeing is impossible, forever. Ending the episode there makes the loop worth nothing, while winning the battle pays experience. Logged as `episode/battle_loop`. `watch.py` and `evaluate.py` never end episodes early, so loops stay visible.

**The "v3" model** (for the whole game, GPU needed): `--downscale 1` (the full 160×144 screen, so the agent can read the 8×8 letters of dialogues and menus), `--network impala` (deeper eyes with residual blocks, better at generalizing), `--memory gru`, `--gamma 0.999` (a longer horizon) and `--norm-rewards` (rewards scaled by the spread of the discounted return, to keep the critic stable with the longer horizon). About 6.6M parameters: on a CPU it is far too slow, on a T4 GPU the learning phase takes a larger share of the time than with the small model. A run started with these options must be resumed with the same ones (the script checks).

```bash
uv run python training/ppo.py --downscale 1 --network impala --memory gru --gamma 0.999 --norm-rewards --episode-steps 100000 --archive-prob 0.3 --curriculum-prob 0.3 --sil-coef 1.0 --record-every 8 --stop-at-goal --run-name red_v3_s1
```

**Short-term memory** (`--memory gru`): without memory the agent decides from the last 3 frames, about one second of game, so it cannot know what is not on the screen: for example that it is carrying Oak's Parcel, which it must bring back south. With `--memory gru` the network carries a small memory (a GRU, `--memory-size 256`) from step to step, wiped at every new episode, and learns by itself what is worth remembering, over stretches of `--num-steps` steps (256 ≈ 100 s of game). Notes:

- It is a different network: a model trained with memory can only be resumed with `--memory gru`, and one trained without it only without it. `watch.py` and `evaluate.py` pick the right network from the checkpoint.
- `--num-envs` must be a multiple of `--num-minibatches` (default 4): each minibatch replays whole games, in order.
- Episodes started from the archive or a curriculum demo start with an empty memory.
- Self-imitation stores the memory of the moment with each replayed action (an approximation, since the network keeps changing).
- Cost: about 25% fewer steps per second on a CPU; much less with a GPU, where learning is a small share of the time.

The comparison that matters: the same run, same seed, with and without memory.

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
- `episode/stagnated`: fraction of episodes ended early for lack of progress (v2); `episode/battle_loop`: those ended stuck in a battle (v2.2).
- `losses/entropy`: how random the policy still is. If it collapses early, the agent stops exploring.
- `charts/SPS`: steps per second.

PyTorch from PyPI runs on the CPU on Windows, which is fine for this small network. A GPU is optional.

### Long runs

- **Resume** a run from its last checkpoint: `--resume runs/<run-name>/checkpoints/latest.pt` (keep the same `--run-name`).
- **Stop cleanly** after a number of hours, saving everything: `--time-limit-hours 11`.
- `Ctrl+C` also saves the last complete update.

### Kaggle

`notebooks/kaggle_train.ipynb` runs training on Kaggle (GPU for learning, CPU cores for the emulators). Upload it to Kaggle, then follow the instructions in its first cell: enable GPU and Internet, add a **private** dataset with the ROMs and start states (`pokemon_red.gb` and `red_start.state`, plus those of the other games for a multi-game run), set `GAMES`, and use *Save Version → Save & Run All* for long runs. To continue a run in a new session, add the previous version's output as input and set `RESUME = True`.

On a laptop the emulators may not scale well across cores (run `scripts/diagnose_speed.py` to check): in that case use your PC for development and short tests, and Kaggle for long runs.

## Other games and transfer

Supported games: `red`, `blue` (same memory as Red), `yellow` (same maps and milestones, memory shifted by one byte; run in classic Game Boy mode so that it looks like Red) and `crystal` (the **held-out** test game, see `docs/research.md`: training on it is refused unless `--allow-held-out` is given). Put the ROMs in `roms/` as `pokemon_<game>.gb` (or `.gbc`), then create each start state in the game's own bedroom:

```bash
uv run python scripts/make_start_state.py --game yellow
uv run python scripts/make_start_state.py --game crystal   # set the clock and names, stop in the bedroom
```

**One model, several games** (`--games red,blue,yellow`): the parallel games are shared out in turn among the titles (6 games, 3 titles = 2 each, so use a multiple of the number of titles). ROMs and start states are taken from the default paths. Curves are kept per title (`milestones/red/M1_rate`, `episode/blue/return`, ...), and so are GIFs (`gifs/<game>/`). With `--stop-at-goal`, the model at the first win of each game is saved as `checkpoints/winner_<game>.pt`, and training stops when all are won (`checkpoints/winner.pt`):

```bash
uv run python training/ppo.py --games red,blue,yellow --num-envs 6 --total-steps 400_000_000 --episode-steps 30000 --archive-prob 0.3 --curriculum-prob 0.3 --sil-coef 1.0 --record-every 8 --stop-at-goal --run-name gen1_s1
```

Evaluate any model on any game, without training on it (zero-shot), over several episodes:

```bash
uv run python scripts/evaluate.py runs/<run>/checkpoints/latest.pt --game red --episodes 10      # reference
uv run python scripts/evaluate.py runs/<run>/checkpoints/latest.pt --game yellow --episodes 10   # transfer
uv run python scripts/evaluate.py runs/<run>/checkpoints/winner.pt --game crystal --episodes 10   # held-out test
```

It prints, for every milestone, the fraction of episodes that reached it and the median step. `watch.py --game yellow --checkpoint ...` shows the same model playing in a window.

### Game Boy Advance (FireRed)

Game Boy Advance games run on **mGBA**, loaded as a libretro core (a single file) through a small Python frontend (`packages/envs/src/missingno_envs/libretro.py`). Nothing to compile:

```bash
uv run python scripts/get_mgba_core.py               # downloads cores/mgba_libretro.<dll|so|dylib>
uv run pytest tests/test_gba.py                      # checks it works (no game needed)
uv run python scripts/make_start_state.py --game firered   # play the intro, stop in the bedroom, close the window
uv run python scripts/evaluate.py runs/<run>/checkpoints/latest.pt --game firered --episodes 10
```

- ROM: `roms/pokemon_firered.gba`, **English** version 1.0 or 1.1 (other languages have different memory addresses).
- The GBA screen (240×160) is resized to the Game Boy's (160×144), so the same network plays both consoles. The buttons are the same 7 (L and R are not used).
- Save states belong to the core version: make the start state and train with the same core. For Kaggle, run `get_mgba_core.py --also-linux` and add `cores/mgba_libretro.so` to your private dataset.
- FireRed's milestones mirror Red's (bedroom to the Boulder Badge), on FireRed's maps.

## How the environment works

- **What the agent sees:** only the screen, grayscale, downscaled to 80×72, last 3 frames stacked.
- **What the agent does:** one of 7 buttons (down, left, right, up, A, B, Start), once every 24 frames.
- **Rewards:** generic and the same for every game: new tiles and maps (worth less the more often they were seen), passages between maps, badges, experience, new items, the Pokédex (`docs/rewards.md`).
- **Game adapters** (`packages/games/`) read the game memory to compute those rewards. The agent never sees this information.
- **Milestones** (e.g. "reach Viridian City") are defined per game and used only to measure progress, never as rewards.

## Structure

```
project-MissingNo/
├── packages/
│   ├── core/            # shared types, including ProgressSignals
│   ├── games/           # one adapter per game (red.py: Red, Blue, Yellow; crystal.py; firered.py)
│   ├── envs/            # the Gymnasium environment (PokemonEnv)
│   └── agents/          # neural networks
├── training/            # training scripts (ppo.py)
├── notebooks/           # Kaggle notebook
├── scripts/             # make_start_state, watch, evaluate, replay, get_mgba_core, benchmark_env
├── cores/               # emulator cores for the Game Boy Advance (ignored by Git)
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
