"""Evaluate a trained model on any game: how far does it get, milestone by milestone?

Plays N episodes without a window and prints, for each milestone, the fraction of
episodes that reached it and the median step at which it was reached. This is
how we test transfer: a model trained on Red, evaluated on Blue or Yellow
without any training on them (zero-shot), and how we score the held-out game
(Crystal) for a model trained on Generation 1.

Usage:
    uv run python scripts/evaluate.py runs/badge_v22_s1/checkpoints/winner.pt --game yellow
    uv run python scripts/evaluate.py <checkpoint> --game red --episodes 20 --steps 30000
    uv run python scripts/evaluate.py <checkpoint> --game yellow --greedy
    uv run python scripts/evaluate.py runs/gen1_s1/checkpoints/winner.pt --game crystal  # held-out test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from missingno_agents import Policy
from missingno_envs import EnvConfig, PokemonEnv, RewardConfig
from missingno_games import ADAPTERS, default_rom, default_state


def main(args) -> None:
    torch.set_num_threads(args.torch_threads)
    policy = Policy(args.checkpoint, args.greedy)  # with or without memory, as it was trained
    config = policy.config
    trained_on = config.get("games") or config.get("game", "?")
    reward_version = config.get("reward_version", "v2")

    rom = args.rom or default_rom(args.game)
    state = args.state or default_state(args.game)
    if not Path(state).exists():
        raise SystemExit(
            f"No start state at {state}: create it with "
            f"uv run python scripts/make_start_state.py --game {args.game}"
        )
    env = PokemonEnv(
        EnvConfig(
            rom_path=rom,
            start_state_path=state,
            max_steps=args.steps,
            # No stagnation stop: an evaluation gives the agent its whole time budget.
            rewards=RewardConfig.preset(reward_version).with_weights(
                stagnation_steps=args.stagnation_steps,
                battle_stagnation_steps=args.battle_stagnation_steps,
            ),
        ),
        ADAPTERS[args.game](),
    )
    milestones = env.adapter.milestones
    print(
        f"Model trained on {trained_on}, evaluated on {args.game}: "
        f"{args.episodes} episodes of {args.steps:,} steps ({'greedy' if args.greedy else 'sampled'})"
    )

    reached = np.full((args.episodes, len(milestones)), -1)
    maps = []
    for ep in range(args.episodes):
        start = time.time()
        obs, _ = env.reset(seed=args.seed + ep)
        policy.reset()  # a new episode: wipe the memory, if the model has one
        for _ in range(args.steps):
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if terminated or truncated:
                break
        reached[ep] = info["milestone_step"]
        maps.append(info["maps_visited"])
        done = [m.id for m, s in zip(milestones, reached[ep], strict=True) if s >= 0]
        last = done[-1] if done else "-"  # milestones are in game order
        print(
            f"  episode {ep + 1:2d}: furthest {last:>4}, {info['maps_visited']} maps, "
            f"{info['tiles_visited']} tiles  ({time.time() - start:.0f}s)"
        )
    env.close()

    print(f"\n{'milestone':<32} {'reached':>8} {'median step':>12}")
    for i, m in enumerate(milestones):
        hits = reached[:, i][reached[:, i] >= 0]
        median = f"{int(np.median(hits)):,}" if len(hits) else "-"
        print(f"{m.id + ' ' + m.name:<32} {len(hits) / args.episodes:>7.0%} {median:>12}")
    print(f"\nMaps visited per episode: mean {np.mean(maps):.1f}, max {max(maps)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--game", default="red", choices=ADAPTERS)
    parser.add_argument("--rom", help="default: roms/pokemon_<game>.gb (or .gbc)")
    parser.add_argument("--state", help="default: states/<game>_start.state")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20_000, help="max steps per episode")
    parser.add_argument("--greedy", action="store_true", help="always pick the most likely button")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stagnation-steps",
        type=int,
        default=0,
        help="end an episode after N steps without progress; 0 = never",
    )
    parser.add_argument(
        "--battle-stagnation-steps",
        type=int,
        default=0,
        help="end an episode after N steps of a battle without progress; 0 = never",
    )
    parser.add_argument("--torch-threads", type=int, default=2)
    main(parser.parse_args())
