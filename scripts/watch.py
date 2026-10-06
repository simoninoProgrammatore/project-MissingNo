"""Watch an agent play in a window. For now: a random agent.

It is also the first sanity check of the environment: you should see the player
move, and the reward should grow when it reaches new tiles or maps.

Usage:
    uv run python scripts/watch.py --steps 2000
    uv run python scripts/watch.py --speed 0      # as fast as possible
"""

import argparse
from pathlib import Path

from missingno_envs import ACTIONS, EnvConfig, PokemonEnv
from missingno_games import ADAPTERS


def main(game: str, rom: str, state: str | None, steps: int, speed: int) -> None:
    config = EnvConfig(rom_path=rom, start_state_path=state, max_steps=steps)
    env = PokemonEnv(config, ADAPTERS[game](), render_mode="human", emulation_speed=speed)
    _, info = env.reset(seed=0)
    total = 0.0
    for step in range(1, steps + 1):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if info["reward_parts"]:
            print(f"step {step:5d}  {ACTIONS[action]:>5}  +{reward:.2f}  {info['reward_parts']}")
        if step % 500 == 0:
            print(
                f"--- step {step}: return {total:.2f}, tiles {info['tiles_visited']}, "
                f"maps {info['maps_visited']}, map {info['map_id']}, pos {info['position']}"
            )
        if terminated or truncated:
            break
    print(
        f"Episode return: {total:.2f}  tiles: {info['tiles_visited']}  maps: {info['maps_visited']}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--game", default="red", choices=ADAPTERS)
    parser.add_argument("--rom", default="roms/pokemon_red.gb")
    parser.add_argument("--state", default="states/red_start.state")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--speed", type=int, default=4, help="1 = real time, 0 = unlimited")
    args = parser.parse_args()
    state = args.state if Path(args.state).exists() else None
    if state is None:
        print(f"No start state at {args.state}: booting from power-on.")
    main(args.game, args.rom, state, args.steps, args.speed)
