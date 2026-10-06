"""Measure how many agent steps per second one environment runs (headless).

This is the first real number for the compute estimates: multiply by the number
of CPU cores to get a rough upper bound for a machine.

Usage:
    uv run python scripts/benchmark_env.py --steps 5000
"""

import argparse
import time
from pathlib import Path

from missingno_envs import EnvConfig, PokemonEnv
from missingno_games import ADAPTERS


def main(game: str, rom: str, state: str | None, steps: int) -> None:
    env = PokemonEnv(
        EnvConfig(rom_path=rom, start_state_path=state, max_steps=steps), ADAPTERS[game]()
    )
    env.reset(seed=0)
    start = time.perf_counter()
    for _ in range(steps):
        env.step(env.action_space.sample())
    elapsed = time.perf_counter() - start
    env.close()
    sps = steps / elapsed
    frames = sps * env.config.frames_per_action
    print(
        f"{steps} steps in {elapsed:.1f}s -> {sps:,.0f} steps/s per core ({frames:,.0f} frames/s)"
    )
    print(f"1 billion steps on one core: {1e9 / sps / 3600:,.0f} core-hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--game", default="red", choices=ADAPTERS)
    parser.add_argument("--rom", default="roms/pokemon_red.gb")
    parser.add_argument("--state", default="states/red_start.state")
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()
    state = args.state if Path(args.state).exists() else None
    if state is None:
        print(f"No start state at {args.state}: booting from power-on.")
    main(args.game, args.rom, state, args.steps)
