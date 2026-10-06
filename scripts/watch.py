"""Watch an agent play in a window: a random agent, or a trained one.

It is also the first sanity check of the environment: you should see the player
move, and the reward should grow when it reaches new tiles or maps.

Usage:
    uv run python scripts/watch.py --steps 2000                                  # random agent
    uv run python scripts/watch.py --checkpoint runs/ppo_s1/checkpoints/latest.pt
    uv run python scripts/watch.py --checkpoint ... --greedy --speed 0           # best action, max speed
"""

import argparse
from pathlib import Path

from missingno_envs import ACTIONS, EnvConfig, PokemonEnv
from missingno_games import ADAPTERS


def load_policy(checkpoint: str, greedy: bool):
    import torch
    from missingno_agents import CnnActorCritic

    data = torch.load(checkpoint, map_location="cpu")
    agent = CnnActorCritic(tuple(data["obs_shape"]), data["n_actions"])
    agent.load_state_dict(data["model"])
    agent.eval()

    def policy(obs):
        with torch.no_grad():
            action, *_ = agent.act(torch.as_tensor(obs).unsqueeze(0), greedy=greedy)
        return int(action.item())

    return policy


def main(args) -> None:
    config = EnvConfig(rom_path=args.rom, start_state_path=args.state, max_steps=args.steps)
    env = PokemonEnv(
        config,
        ADAPTERS[args.game](),
        render_mode=None if args.no_window else "human",
        emulation_speed=args.speed,
    )
    policy = load_policy(args.checkpoint, args.greedy) if args.checkpoint else None
    milestones = env.adapter.milestones

    obs, info = env.reset(seed=0)
    total = 0.0
    reached = set()
    for step in range(1, args.steps + 1):
        action = policy(obs) if policy else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if args.verbose and info["reward_parts"]:
            print(f"step {step:5d}  {ACTIONS[action]:>5}  +{reward:.2f}  {info['reward_parts']}")
        for m, at in zip(milestones, info["milestone_step"], strict=True):
            if at >= 0 and m.id not in reached:
                reached.add(m.id)
                print(f"*** step {step:5d}: {m.id} {m.name}")
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
    print(f"Milestones reached: {len(reached)}/{len(milestones)}")
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
    parser.add_argument("--checkpoint", help="trained model (.pt); omit for a random agent")
    parser.add_argument("--greedy", action="store_true", help="always pick the most likely button")
    parser.add_argument("--verbose", action="store_true", help="print every reward")
    parser.add_argument("--no-window", action="store_true", help="run without a window")
    args = parser.parse_args()
    if not Path(args.state).exists():
        print(f"No start state at {args.state}: booting from power-on.")
        args.state = None
    main(args)
